/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "command_pools.hpp"

#include <cassert>
#include <utility>

#include "check_error.hpp"
#include "debug_util.hpp"
#include "queue.hpp"  // usage example

namespace nvdx12 {

ManagedCommandPools::ManagedCommandPools(ManagedCommandPools&& other) noexcept
{
  std::swap(m_device, other.m_device);
  std::swap(m_flags, other.m_flags);
  std::swap(m_waitCount, other.m_waitCount);
  std::swap(m_managedPools, other.m_managedPools);
  std::swap(m_maxPoolCount, other.m_maxPoolCount);
  std::swap(m_acquisitionCounter, other.m_acquisitionCounter);
  std::swap(m_queueFamilyIndex, other.m_queueFamilyIndex);
  std::swap(m_mode, other.m_mode);
}

ManagedCommandPools& ManagedCommandPools::operator=(ManagedCommandPools&& other) noexcept
{
  if(this != &other)
  {
    assert(m_device == nullptr && "Missing deinit()");

    std::swap(m_device, other.m_device);
    std::swap(m_flags, other.m_flags);
    std::swap(m_waitCount, other.m_waitCount);
    std::swap(m_managedPools, other.m_managedPools);
    std::swap(m_maxPoolCount, other.m_maxPoolCount);
    std::swap(m_acquisitionCounter, other.m_acquisitionCounter);
    std::swap(m_queueFamilyIndex, other.m_queueFamilyIndex);
    std::swap(m_mode, other.m_mode);
  }

  return *this;
}

ManagedCommandPools::~ManagedCommandPools()
{
  assert(m_device == nullptr && "Missing deinit()");
}

HRESULT ManagedCommandPools::init(ID3D12Device* device, uint32_t queueFamilyIndex, Mode mode, D3D12_COMMAND_LIST_FLAGS flags, uint32_t maxPoolCount)
{
  assert(m_device == nullptr && "init() called twice; missing deinit()?");
  assert(device && maxPoolCount > 0);

  // CreateCommandList1 creates closed lists, which every acquisition resets
  // onto its allocator.
  NVDX12_FAIL_RETURN(device->QueryInterface(IID_PPV_ARGS(&m_device)));
  m_queueFamilyIndex   = queueFamilyIndex;
  m_flags              = flags;
  m_maxPoolCount       = maxPoolCount;
  m_mode               = mode;
  m_waitCount          = 0;
  m_acquisitionCounter = 0;

  if(m_mode == Mode::EXPLICIT_INDEX)
  {
    m_managedPools.resize(maxPoolCount);
    for(uint32_t i = 0; i < maxPoolCount; i++)
    {
      const HRESULT hr = createCommandPool(m_managedPools[i]);
      if(FAILED(hr))
      {
        deinit();
        return hr;
      }
    }
  }
  return S_OK;
}

void ManagedCommandPools::deinit()
{
  if(!m_device)
    return;

  // lists before their allocators
  for(size_t i = 0; i < m_managedPools.size(); i++)
  {
    auto& managedPool = m_managedPools[i];
    managedPool.cmd.Reset();
    managedPool.commandPool.Reset();
  }
  m_managedPools = {};
  m_device.Reset();
}

HRESULT ManagedCommandPools::createCommandPool(ManagedCommandPool& managedPool)
{
  NVDX12_FAIL_RETURN(m_device->CreateCommandAllocator(static_cast<D3D12_COMMAND_LIST_TYPE>(m_queueFamilyIndex),
                                                      IID_PPV_ARGS(&managedPool.commandPool)));
  setDebugName(managedPool.commandPool.Get(), "nvdx12::ManagedCommandPools allocator");
  return S_OK;
}

HRESULT ManagedCommandPools::acquireCommandBuffer(const SemaphoreState& submitSemaphoreState, ID3D12GraphicsCommandList7*& cmd, uint64_t waitTimeOut)
{
  assert(m_mode == Mode::SEMAPHORE_STATE);
  assert(submitSemaphoreState.isValid());

  uint64_t            lowestAcquisitionIndex = ~0ULL;
  ManagedCommandPool* oldestManagedPool      = nullptr;

  for(size_t i = 0; i < m_managedPools.size(); i++)
  {
    ManagedCommandPool& managedPool = m_managedPools[i];
    // find if we can retire an old cycle
    if(managedPool.inUse && managedPool.semaphoreState.isValid() && managedPool.semaphoreState.testSignaled())
    {
      NVDX12_FAIL_RETURN(reset(managedPool));
    }

    if(managedPool.commandPool && !managedPool.inUse)
    {
      return getManagedCommandBuffer(managedPool, submitSemaphoreState, cmd);
    }
    else if(managedPool.semaphoreState.isValid() && managedPool.acquisitionIndex < lowestAcquisitionIndex)
    {
      lowestAcquisitionIndex = managedPool.acquisitionIndex;
      oldestManagedPool      = &managedPool;
    }
  }

  // we reached the maximum
  if(size_t(m_maxPoolCount) == m_managedPools.size())
  {
    m_waitCount++;

    assert(oldestManagedPool);

    ManagedCommandPool& managedPool = *oldestManagedPool;

    // must wait for semaphore state; WAIT_TIMEOUT is a success code, so test for S_OK:
    // the allocator must not be reset while its list may still execute
    const HRESULT waitResult = managedPool.semaphoreState.wait(waitTimeOut);
    if(waitResult != S_OK)
    {
      return FAILED(waitResult) ? NVDX12_FAIL_REPORT(waitResult) : waitResult;
    }

    // reset
    NVDX12_FAIL_RETURN(reset(managedPool));

    // return new command list
    return getManagedCommandBuffer(managedPool, submitSemaphoreState, cmd);
  }

  // need a new allocator
  ManagedCommandPool managedPool{};
  NVDX12_FAIL_RETURN(createCommandPool(managedPool));

  // add it and get a command list from it
  return getManagedCommandBuffer(m_managedPools.emplace_back(std::move(managedPool)), submitSemaphoreState, cmd);
}

HRESULT ManagedCommandPools::acquireCommandBuffer(uint32_t explicitIndex, ID3D12GraphicsCommandList7*& cmd)
{
  assert(explicitIndex < m_maxPoolCount);

  NVDX12_FAIL_RETURN(releaseIndexed(explicitIndex));

  return getManagedCommandBuffer(m_managedPools[explicitIndex], {}, cmd);
}

HRESULT ManagedCommandPools::getManagedCommandBuffer(ManagedCommandPool&          managedPool,
                                                     const SemaphoreState&        submitSemaphoreState,
                                                     ID3D12GraphicsCommandList7*& cmd)
{
  assert(!managedPool.inUse);

  if(!managedPool.cmd)
  {
    // created closed; the Reset below opens it on the allocator
    NVDX12_FAIL_RETURN(m_device->CreateCommandList1(0, static_cast<D3D12_COMMAND_LIST_TYPE>(m_queueFamilyIndex),
                                                    m_flags, IID_PPV_ARGS(&managedPool.cmd)));
    setDebugName(managedPool.cmd.Get(), "nvdx12::ManagedCommandPools");
  }

  NVDX12_FAIL_RETURN(managedPool.cmd->Reset(managedPool.commandPool.Get(), nullptr));

  managedPool.inUse            = true;
  managedPool.acquisitionIndex = m_acquisitionCounter++;
  managedPool.semaphoreState   = submitSemaphoreState;
  cmd                          = managedPool.cmd.Get();

  return S_OK;
}

HRESULT ManagedCommandPools::reset(ManagedCommandPool& managedPool)
{
  assert(managedPool.inUse);

  // The list is kept for the next acquisition (D3D12 cannot free it back to
  // the allocator); it is closed, so the allocator can be reset.
  NVDX12_FAIL_RETURN(managedPool.commandPool->Reset());

  managedPool.semaphoreState = {};
  managedPool.inUse          = false;

  return S_OK;
}

HRESULT ManagedCommandPools::releaseCommandBuffer(ID3D12GraphicsCommandList7* cmd)
{
  for(size_t i = 0; i < m_managedPools.size(); i++)
  {
    auto& managedPool = m_managedPools[i];
    if(managedPool.inUse && managedPool.cmd.Get() == cmd)
    {
      // The list may still be recording, which would make the allocator
      // reset fail: release the list object, the next acquisition creates a
      // new one.
      managedPool.cmd.Reset();
      return reset(managedPool);
    }
  }

  return E_INVALIDARG;
}

HRESULT ManagedCommandPools::releaseCompleted()
{
  assert(m_mode == Mode::SEMAPHORE_STATE);

  for(size_t i = 0; i < m_managedPools.size(); i++)
  {
    auto& managedPool = m_managedPools[i];
    // find if we can retire an old cycle
    if(managedPool.inUse && managedPool.semaphoreState.isValid() && managedPool.semaphoreState.testSignaled())
    {
      NVDX12_FAIL_RETURN(reset(managedPool));
    }
  }

  return S_OK;
}

HRESULT ManagedCommandPools::releaseIndexed(uint32_t explicitIndex)
{
  assert(explicitIndex < m_maxPoolCount);
  assert(m_mode == Mode::EXPLICIT_INDEX);

  if(m_managedPools[explicitIndex].inUse)
  {
    NVDX12_FAIL_RETURN(reset(m_managedPools[explicitIndex]));
  }

  return S_OK;
}

}  // namespace nvdx12


//--------------------------------------------------------------------------------------------------
// Usage example
//--------------------------------------------------------------------------------------------------
[[maybe_unused]] static void usage_ManagedCommandPools()
{
  ID3D12Device*         device{};  // EX: from nvdx12::Context::getDevice()
  nvdx12::QueueTimeline queue;     // EX: initialized on a direct queue

  {
    // nvdx12::ManagedCommandPools::Mode::SEMAPHORE_STATE
    // example

    // This class is useful to provide us with a "fresh" command list.
    // In this mode we use the fence value of the submission to track completion
    // of a command list and reset/recycle its command allocator.

    nvdx12::ManagedCommandPools managedCmdPools;
    managedCmdPools.init(device, D3D12_COMMAND_LIST_TYPE_DIRECT, nvdx12::ManagedCommandPools::Mode::SEMAPHORE_STATE);

    // frame loop
    /* while(!glfwWindowShouldClose()) */
    {
      // the fence value the submit below signals
      nvdx12::SemaphoreState semaphoreState = queue.createNextFixedSemaphoreState();

      ID3D12GraphicsCommandList7* cmd{};
      HRESULT                     result = managedCmdPools.acquireCommandBuffer(semaphoreState, cmd);
      (void)result;

      // do stuff with the command list as usual (it is recording already)

      cmd->Close();

      // submit to queue, it signals `semaphoreState`
      queue.submit(cmd, semaphoreState);
    }

    queue.waitIdle();
    managedCmdPools.deinit();
  }

  {
    // nvdx12::ManagedCommandPools::Mode::EXPLICIT_INDEX
    // example

    const uint32_t ringSize  = 3;
    uint32_t       ringIndex = 0;
    uint64_t       ringFenceValues[ringSize]{};

    // This class is useful to provide us with a "fresh" command list.
    // In this mode we use explicit indices and have to externally ensure completion by some means.
    // It allows using a classic ring buffer approach.

    nvdx12::ManagedCommandPools managedCmdPools;
    managedCmdPools.init(device, D3D12_COMMAND_LIST_TYPE_DIRECT, nvdx12::ManagedCommandPools::Mode::EXPLICIT_INDEX,
                         D3D12_COMMAND_LIST_FLAG_NONE, ringSize);

    // frame loop
    /* while(!glfwWindowShouldClose()) */
    {
      // MUST manually ensure that current `ringIndex` has completed.
      queue.waitForValue(ringFenceValues[ringIndex]);

      ID3D12GraphicsCommandList7* cmd{};
      HRESULT                     result = managedCmdPools.acquireCommandBuffer(ringIndex, cmd);
      (void)result;

      // do stuff with the command list as usual

      cmd->Close();
      ringFenceValues[ringIndex] = queue.submit(cmd);

      // increment ringIndex for next cycle
      ringIndex = (ringIndex + 1) % ringSize;
    }

    queue.waitIdle();
    managedCmdPools.deinit();
  }
}
