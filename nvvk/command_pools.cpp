/*
* Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*
* SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
* SPDX-License-Identifier: Apache-2.0
*/

#include <volk.h>

#include "command_pools.hpp"
#include "check_error.hpp"

namespace nvvk {

ManagedCommandPools::ManagedCommandPools(ManagedCommandPools&& other) noexcept
{
  std::swap(m_device, other.m_device);
  std::swap(m_flags, other.m_flags);
  std::swap(m_managedPools, other.m_managedPools);
  std::swap(m_maxPoolCount, other.m_maxPoolCount);
  std::swap(m_acquisitionCounter, other.m_acquisitionCounter);
  std::swap(m_queueFamilyIndex, other.m_queueFamilyIndex);
}

ManagedCommandPools& ManagedCommandPools::operator=(ManagedCommandPools&& other) noexcept
{
  if(this != &other)
  {
    assert(m_device == nullptr && "Missing deinit()");

    std::swap(m_device, other.m_device);
    std::swap(m_flags, other.m_flags);
    std::swap(m_managedPools, other.m_managedPools);
    std::swap(m_maxPoolCount, other.m_maxPoolCount);
    std::swap(m_acquisitionCounter, other.m_acquisitionCounter);
    std::swap(m_queueFamilyIndex, other.m_queueFamilyIndex);
  }

  return *this;
}

ManagedCommandPools::~ManagedCommandPools()
{
  assert(m_device == nullptr && "Missing deinit()");
}

VkResult ManagedCommandPools::init(VkDevice device, uint32_t queueFamilyIndex, Mode mode, VkCommandPoolCreateFlags flags, uint32_t maxPoolCount)
{
  assert((flags & VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT) == 0 && "manual resetting of command buffers is not supported");

  m_device           = device;
  m_queueFamilyIndex = queueFamilyIndex;
  m_flags            = flags;
  m_maxPoolCount     = maxPoolCount;
  m_mode             = mode;

  if(m_mode == Mode::EXPLICIT_INDEX)
  {
    m_managedPools.resize(maxPoolCount);
    for(uint32_t i = 0; i < maxPoolCount; i++)
    {
      VkCommandPoolCreateInfo createInfo = {
          .sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
          .flags            = m_flags,
          .queueFamilyIndex = m_queueFamilyIndex,
      };

      NVVK_FAIL_RETURN(vkCreateCommandPool(m_device, &createInfo, nullptr, &m_managedPools[i].commandPool));
    }
  }
  return VK_SUCCESS;
}

void ManagedCommandPools::deinit()
{
  if(!m_device)
    return;

  for(auto& managedPool : m_managedPools)
  {
    if(managedPool.cmd)
    {
      vkFreeCommandBuffers(m_device, managedPool.commandPool, 1, &managedPool.cmd);
    }
    if(managedPool.commandPool)
    {
      vkDestroyCommandPool(m_device, managedPool.commandPool, nullptr);
    }
  }
  m_managedPools = {};
  m_device       = nullptr;
}

VkResult ManagedCommandPools::acquireCommandBuffer(const nvvk::SemaphoreState& submitSemaphoreState,
                                                   VkCommandBuffer&            cmd,
                                                   VkCommandBufferLevel        level,
                                                   uint64_t                    waitTimeOut)
{
  assert(m_mode == Mode::SEMAPHORE_STATE);
  assert(submitSemaphoreState.isValid());

  uint64_t            lowestAcquisitionIndex = ~0ULL;
  ManagedCommandPool* oldestManagedPool      = nullptr;

  for(ManagedCommandPool& managedPool : m_managedPools)
  {
    // find if we can retire an old cycle
    if(managedPool.semaphoreState.isValid() && managedPool.semaphoreState.testSignaled(m_device))
    {
      NVVK_FAIL_RETURN(reset(managedPool, 0));
    }

    if(managedPool.commandPool && !managedPool.cmd)
    {
      return getManagedCommandBuffer(managedPool, level, submitSemaphoreState, cmd);
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
    assert(oldestManagedPool);

    ManagedCommandPool& managedPool = *oldestManagedPool;

    // must wait for semaphore state
    NVVK_FAIL_RETURN(managedPool.semaphoreState.wait(m_device, waitTimeOut));

    // reset
    NVVK_FAIL_RETURN(reset(managedPool, 0));

    // return new command buffer
    return getManagedCommandBuffer(managedPool, level, submitSemaphoreState, cmd);
  }

  // need a new pool
  ManagedCommandPool managedPool{};

  VkCommandPoolCreateInfo createInfo = {
      .sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .flags            = m_flags,
      .queueFamilyIndex = m_queueFamilyIndex,
  };

  NVVK_CHECK(vkCreateCommandPool(m_device, &createInfo, nullptr, &managedPool.commandPool));

  // add it and get a command buffer from it
  return getManagedCommandBuffer(m_managedPools.emplace_back(managedPool), level, submitSemaphoreState, cmd);
}

VkResult ManagedCommandPools::acquireCommandBuffer(uint32_t explicitIndex, VkCommandBuffer& cmd, VkCommandBufferLevel level /*= VK_COMMAND_BUFFER_LEVEL_PRIMARY*/)
{
  assert(explicitIndex < m_maxPoolCount);

  NVVK_FAIL_RETURN(releaseIndexed(explicitIndex, 0));

  return getManagedCommandBuffer(m_managedPools[explicitIndex], level, {}, cmd);
}

VkResult ManagedCommandPools::getManagedCommandBuffer(ManagedCommandPool&         managedPool,
                                                      VkCommandBufferLevel        level,
                                                      const nvvk::SemaphoreState& submitSemaphoreState,
                                                      VkCommandBuffer&            cmd)
{
  VkCommandBufferAllocateInfo info = {
      .sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .commandPool        = managedPool.commandPool,
      .level              = level,
      .commandBufferCount = 1,
  };

  NVVK_FAIL_RETURN(vkAllocateCommandBuffers(m_device, &info, &managedPool.cmd));

  managedPool.acquisitionIndex = m_acquisitionCounter++;
  managedPool.semaphoreState   = submitSemaphoreState;
  cmd                          = managedPool.cmd;

  return VK_SUCCESS;
}

VkResult ManagedCommandPools::reset(ManagedCommandPool& managedPool, VkCommandPoolResetFlags resetFlags)
{
  assert(managedPool.cmd);

  vkFreeCommandBuffers(m_device, managedPool.commandPool, 1, &managedPool.cmd);
  NVVK_FAIL_RETURN(vkResetCommandPool(m_device, managedPool.commandPool, resetFlags));

  managedPool.semaphoreState = {};
  managedPool.cmd            = {};

  return VK_SUCCESS;
}

VkResult ManagedCommandPools::releaseCommandBuffer(VkCommandBuffer cmd, VkCommandPoolResetFlags resetFlags)
{
  for(auto& managedPool : m_managedPools)
  {
    if(managedPool.cmd == cmd)
    {
      return reset(managedPool, resetFlags);
    }
  }

  return VK_ERROR_UNKNOWN;
}

VkResult ManagedCommandPools::releaseCompleted(VkCommandPoolResetFlags resetFlags)
{
  assert(m_mode == Mode::SEMAPHORE_STATE);

  for(auto& managedPool : m_managedPools)
  {
    // find if we can retire an old cycle
    if(managedPool.semaphoreState.isValid() && managedPool.semaphoreState.testSignaled(m_device))
    {
      NVVK_FAIL_RETURN(reset(managedPool, resetFlags));
    }
  }

  return VK_SUCCESS;
}

VkResult ManagedCommandPools::releaseIndexed(uint32_t explicitIndex, VkCommandPoolResetFlags resetFlags /*= 0*/)
{
  assert(explicitIndex < m_maxPoolCount);
  assert(m_mode == Mode::EXPLICIT_INDEX);

  if(m_managedPools[explicitIndex].cmd)
  {
    NVVK_FAIL_RETURN(reset(m_managedPools[explicitIndex], resetFlags));
  }

  return VK_SUCCESS;
}

}  // namespace nvvk


//--------------------------------------------------------------------------------------------------
// Usage example
//--------------------------------------------------------------------------------------------------
static void usage_ManagedCommandPools()
{
  VkDevice device{};
  VkQueue  queue{};
  uint32_t queueFamilyIndex{};

  {
    // nvvk::ManagedCommandPools::Mode::SEMAPHORE_STATE
    // example

    VkSemaphore timelineSemaphore{};
    uint64_t    timelineValue = 1;

    // This class is useful to provide us with a "fresh" command buffer.
    // In this mode we use the timeline semaphore state to track completion
    // of a command buffer and reset/recycle its corresponding command pool.

    nvvk::ManagedCommandPools managedCmdPools;
    managedCmdPools.init(device, queueFamilyIndex, nvvk::ManagedCommandPools::Mode::SEMAPHORE_STATE);

    // frame loop
    /* while(!glfwWindowShouldClose()) */
    {
      nvvk::SemaphoreState semaphoreState = nvvk::SemaphoreState::makeFixed(timelineSemaphore, timelineValue);

      VkCommandBuffer cmd;
      VkResult        result = managedCmdPools.acquireCommandBuffer(semaphoreState, cmd);

      // do stuff with the command buffer as usual


      VkCommandBufferSubmitInfo cmdSubmitInfo = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO};
      cmdSubmitInfo.commandBuffer             = cmd;

      VkSemaphoreSubmitInfo semSubmitInfo = nvvk::makeSemaphoreSubmitInfo(semaphoreState, VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT);

      // prepare actual submit
      VkSubmitInfo2 submitInfo2            = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
      submitInfo2.commandBufferInfoCount   = 1;
      submitInfo2.pCommandBufferInfos      = &cmdSubmitInfo;
      submitInfo2.signalSemaphoreInfoCount = 1;
      submitInfo2.pSignalSemaphoreInfos    = &semSubmitInfo;

      // submit to queue
      vkQueueSubmit2(queue, 1, &submitInfo2, VK_NULL_HANDLE);

      // increment timeline value for next frame
      timelineValue++;
    }

    vkDeviceWaitIdle(device);
    managedCmdPools.deinit();
  }

  {
    // nvvk::ManagedCommandPools::Mode::EXPLICIT_INDEX
    // example

    const uint32_t ringSize  = 3;
    uint32_t       ringIndex = 0;

    // This class is useful to provide us with a "fresh" command buffer.
    // In this mode we use explict indices and have to externally ensure by some means.
    // It allows using a classic ring buffer approach.

    nvvk::ManagedCommandPools managedCmdPools;
    managedCmdPools.init(device, queueFamilyIndex, nvvk::ManagedCommandPools::Mode::SEMAPHORE_STATE,
                         VK_COMMAND_POOL_CREATE_TRANSIENT_BIT, ringSize);

    // frame loop
    /* while(!glfwWindowShouldClose()) */
    {
      // MUST manually ensure that current `ringIndex` has completed.
      // Typically via a semaphore/fence wait on host (not shown).

      VkCommandBuffer cmd;
      VkResult        result = managedCmdPools.acquireCommandBuffer(ringIndex, cmd);

      // do stuff with the command buffer as usual


      VkCommandBufferSubmitInfo cmdSubmitInfo = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO};
      cmdSubmitInfo.commandBuffer             = cmd;

      // prepare actual submit
      VkSubmitInfo2 submitInfo2          = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
      submitInfo2.commandBufferInfoCount = 1;
      submitInfo2.pCommandBufferInfos    = &cmdSubmitInfo;

      // submit to queue
      vkQueueSubmit2(queue, 1, &submitInfo2, VK_NULL_HANDLE);

      // increment ringIndex for next cycle
      ringIndex = (ringIndex + 1) % ringSize;
    }

    vkDeviceWaitIdle(device);
    managedCmdPools.deinit();
  }
}
