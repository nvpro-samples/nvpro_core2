/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "queue.hpp"

#include <string>

#include <nvutils/logger.hpp>

#include "check_error.hpp"
#include "debug_util.hpp"

namespace nvdx12 {

void SubmitInfo::append(const CmdPreSubmitInfo& preSubmit)
{
  commandLists.push_back(preSubmit.cmd);
  waitSemaphoreStates.insert(waitSemaphoreStates.end(), preSubmit.waitSemaphores.cbegin(), preSubmit.waitSemaphores.cend());
  signalSemaphoreStates.insert(signalSemaphoreStates.end(), preSubmit.signalSemaphores.cbegin(),
                               preSubmit.signalSemaphores.cend());
  if(preSubmit.semaphoreState.isValid())
  {
    signalSemaphoreStates.push_back({preSubmit.semaphoreState});
  }
}

void SubmitInfo::append(ID3D12CommandList* cmd, SemaphoreState signalSemaphoreState)
{
  commandLists.push_back(cmd);
  if(signalSemaphoreState.isValid())
  {
    signalSemaphoreStates.push_back({std::move(signalSemaphoreState)});
  }
}

void SubmitInfo::append(ID3D12CommandList* cmd)
{
  commandLists.push_back(cmd);
}

//////////////////////////////////////////////////////////////////////////

HRESULT QueueTimeline::init(ID3D12Device* device, QueueInfo queueInfo, const char* debugName)
{
  if(!device || !queueInfo.queue)
  {
    return E_INVALIDARG;
  }
  std::lock_guard lock(m_mutex);
  assert(!m_fence && "init() called twice; missing deinit()?");
  // Created first, so that a failure leaves the timeline uninitialized.
  ComPtr<ID3D12Fence> fence;
  NVDX12_FAIL_RETURN(createTimelineSemaphore(device, 0, fence));
  setDebugName(fence.Get(), std::string(debugName) + " fence");
  m_queueInfo = std::move(queueInfo);
  m_fence     = std::move(fence);
  m_value     = 0;
  NVDX12_FAIL_REPORT(m_queueInfo.queue->GetTimestampFrequency(&m_timestampFrequency));
  return S_OK;
}

void QueueTimeline::deinit()
{
  std::lock_guard lock(m_mutex);
  m_fence.Reset();
  m_queueInfo          = {};
  m_value              = 0;
  m_timestampFrequency = 0;
  m_signalListeners.clear();
  m_submitListeners.clear();
}

HRESULT QueueTimeline::signalLocked(uint64_t& value)
{
  // A rejected signal hands out no value: the next signal reuses it.
  value               = 0;
  const uint64_t next = m_value + 1;
  NVDX12_FAIL_RETURN(m_queueInfo.queue->Signal(m_fence.Get(), next));
  m_value = next;
  value   = next;
  for(size_t i = 0; i < m_signalListeners.size(); i++)
  {
    m_signalListeners[i].second(value);
  }
  return S_OK;
}

QueueTimeline::ListenerHandle QueueTimeline::addSignalListener(SignalListener listener)
{
  std::lock_guard lock(m_mutex);
  m_signalListeners.emplace_back(m_nextListenerHandle, std::move(listener));
  return m_nextListenerHandle++;
}

QueueTimeline::ListenerHandle QueueTimeline::addSubmitListener(SubmitListener listener)
{
  std::lock_guard lock(m_mutex);
  m_submitListeners.emplace_back(m_nextListenerHandle, std::move(listener));
  return m_nextListenerHandle++;
}

void QueueTimeline::removeSignalListener(ListenerHandle handle)
{
  std::lock_guard lock(m_mutex);
  std::erase_if(m_signalListeners, [handle](const auto& entry) { return entry.first == handle; });
}

void QueueTimeline::removeSubmitListener(ListenerHandle handle)
{
  std::lock_guard lock(m_mutex);
  std::erase_if(m_submitListeners, [handle](const auto& entry) { return entry.first == handle; });
}

uint64_t QueueTimeline::signal()
{
  std::lock_guard lock(m_mutex);
  uint64_t        value = 0;
  signalLocked(value);
  return value;
}

HRESULT QueueTimeline::waitForValue(uint64_t value) const
{
  // Value 0 is the fence's initial value. Otherwise an infinite
  // SemaphoreState::wait: concurrent waiters share nothing and no lock is
  // held while blocked on the GPU; a removed device reports UINT64_MAX, so
  // every value reads as completed.
  if(value == 0)
  {
    return S_OK;
  }
  return SemaphoreState::makeFixed(m_fence, value).wait(UINT64_MAX);
}

HRESULT QueueTimeline::submitLocked(std::span<ID3D12CommandList* const> lists, uint64_t& value)
{
  if(!lists.empty())
  {
    m_queueInfo.queue->ExecuteCommandLists(UINT(lists.size()), lists.data());
  }
  const HRESULT hr = signalLocked(value);  // reported there
  if(FAILED(hr))
  {
    return hr;
  }
  for(size_t i = 0; i < m_submitListeners.size(); i++)
  {
    m_submitListeners[i].second(lists, value);
  }
  return S_OK;
}

uint64_t QueueTimeline::submit(std::span<ID3D12CommandList* const> lists)
{
  std::lock_guard lock(m_mutex);
  uint64_t        value = 0;
  submitLocked(lists, value);
  return value;
}

uint64_t QueueTimeline::submit(std::span<ID3D12CommandList* const> lists, SemaphoreState& submitState)
{
  std::lock_guard lock(m_mutex);
  uint64_t        value = 0;
  if(FAILED(submitLocked(lists, value)))
  {
    return 0;
  }
  assert((!submitState.isValid() || submitState.getSemaphore() == m_fence.Get()) && "submitState belongs to another queue");
#ifndef NVDX12_DISABLE_DYNAMIC_SEMAPHORE_STATE
  if(submitState.isDynamic())
  {
    submitState.setDynamicValue(value);
    return value;
  }
#endif
  if(submitState.isFixed())
  {
    assert(submitState.getTimelineValue() == value && "unexpected timeline value");
  }
  else
  {
    submitState.initFixed(m_fence, value);
  }
  return value;
}

// Every fence of `info` is set, and every state but the ones of `queueFence`
// (which take the submission's value) has its timeline value.
static bool hasSubmitValues(SubmitInfo& info, ID3D12Fence* queueFence)
{
  for(size_t i = 0; i < info.waitSemaphores.size(); i++)
  {
    if(!info.waitSemaphores[i].semaphore)
    {
      return false;
    }
  }
  for(size_t i = 0; i < info.signalSemaphores.size(); i++)
  {
    if(!info.signalSemaphores[i].semaphore)
    {
      return false;
    }
  }
  for(size_t i = 0; i < info.waitSemaphoreStates.size(); i++)
  {
    if(!info.waitSemaphoreStates[i].semaphoreState.canWait())
    {
      return false;
    }
  }
  for(size_t i = 0; i < info.signalSemaphoreStates.size(); i++)
  {
    SemaphoreState& state = info.signalSemaphoreStates[i].semaphoreState;
    if(state.getSemaphore() != queueFence && !state.canWait())
    {
      return false;
    }
  }
  return true;
}

HRESULT QueueTimeline::submit(SubmitInfo& info, SemaphoreState& submitState)
{
  ID3D12CommandQueue* queue  = m_queueInfo.queue.Get();
  HRESULT             result = S_OK;
  const auto          keep   = [&result](HRESULT hr) {
    if(SUCCEEDED(result))
    {
      result = hr;
    }
  };

  submitState = {};
  if(!hasSubmitValues(info, m_fence.Get()))
  {
    assert(!"a semaphore without fence, or a semaphore state without timeline value (a pending dynamic state)");
    LOGE("QueueTimeline::submit: a semaphore has no fence or a semaphore state no timeline value; nothing submitted\n");
    return E_INVALIDARG;
  }

  std::lock_guard lock(m_mutex);

  // Waits, before the command lists. A failing wait submits nothing more.
  for(size_t i = 0; i < info.waitSemaphores.size(); i++)
  {
    const SemaphoreInfo& it = info.waitSemaphores[i];
    NVDX12_FAIL_RETURN(queue->Wait(it.semaphore.Get(), it.value));
  }
  for(size_t i = 0; i < info.waitSemaphoreStates.size(); i++)
  {
    const SemaphoreInfo wait = makeSemaphoreSubmitInfo(info.waitSemaphoreStates[i]);
    NVDX12_FAIL_RETURN(queue->Wait(wait.semaphore.Get(), wait.value));
  }

  // The states of this queue's fence complete with the submission's signal.
  [[maybe_unused]] const uint64_t submitValue = m_value + 1;
#ifndef NDEBUG
  for(size_t i = 0; i < info.signalSemaphores.size(); i++)
  {
    assert(info.signalSemaphores[i].semaphore.Get() != m_fence.Get()
           && "regular signalSemaphores must not use queue's fence, use signalSemaphoreStates instead");
  }
#endif
  [[maybe_unused]] bool wasAdded = false;
  for(size_t i = 0; i < info.signalSemaphoreStates.size(); i++)
  {
    SemaphoreSubmitState& it = info.signalSemaphoreStates[i];
    if(it.semaphoreState.getSemaphore() != m_fence.Get())
    {
      continue;
    }
    assert(!wasAdded && "must not use signalSemaphoreStates using the queue's fence more than once");
#ifndef NVDX12_DISABLE_DYNAMIC_SEMAPHORE_STATE
    if(!it.semaphoreState.isDynamic())
#endif
    {
      assert(it.semaphoreState.getTimelineValue() == submitValue && "unexpected timeline value");
    }
    wasAdded = true;
  }

  // Command lists, then the signal of this queue's fence.
  uint64_t value = 0;
  keep(submitLocked(info.commandLists, value));
  assert(value == 0 || value == submitValue);
#ifndef NVDX12_DISABLE_DYNAMIC_SEMAPHORE_STATE
  // A dynamic state of this queue's fence takes the value once it is signaled;
  // after a rejected signal it stays pending.
  if(value != 0)
  {
    for(size_t i = 0; i < info.signalSemaphoreStates.size(); i++)
    {
      SemaphoreState& state = info.signalSemaphoreStates[i].semaphoreState;
      if(state.getSemaphore() == m_fence.Get() && state.isDynamic())
      {
        state.setDynamicValue(value);
      }
    }
  }
#endif

  // Other signals, after the command lists.
  for(size_t i = 0; i < info.signalSemaphores.size(); i++)
  {
    const SemaphoreInfo& it = info.signalSemaphores[i];
    keep(NVDX12_FAIL_REPORT(queue->Signal(it.semaphore.Get(), it.value)));
  }
  for(size_t i = 0; i < info.signalSemaphoreStates.size(); i++)
  {
    const SemaphoreSubmitState& it = info.signalSemaphoreStates[i];
    if(it.semaphoreState.getSemaphore() != m_fence.Get())
    {
      const SemaphoreInfo signal = makeSemaphoreSubmitInfo(it);
      keep(NVDX12_FAIL_REPORT(queue->Signal(signal.semaphore.Get(), signal.value)));
    }
  }

  if(value != 0)
  {
    submitState = SemaphoreState::makeFixed(m_fence, value);
  }
  return result;
}

}  // namespace nvdx12

//--------------------------------------------------------------------------------------------------
// Usage example
//--------------------------------------------------------------------------------------------------
[[maybe_unused]] static void usage_QueueTimeline()
{
  ID3D12Device*     device = nullptr;  // EX: from nvdx12::Context::getDevice()
  nvdx12::QueueInfo directQueue;       // EX: nvdx12::Context::getQueueInfo(0)

  nvdx12::QueueTimeline queue;
  NVDX12_CHECK(queue.init(device, directQueue, "main queue"));

  // Something that recycles memory by fence value, told about every signal
  // (nvdx12::UploadRing registers itself in its init); the handle removes it.
  const nvdx12::QueueTimeline::ListenerHandle listener =
      queue.addSignalListener([](uint64_t fenceValue) { /* EX: free what completes with fenceValue */ });

  // per-frame loop
  /* while(!glfwWindowShouldClose()) */
  {
    ID3D12GraphicsCommandList* cmd{};  // EX: record and close a command list

    // Work that must only be freed once this frame's submission completed
    // can keep the state of the upcoming submission.
    nvdx12::SemaphoreState frameState = queue.createNextFixedSemaphoreState();

    // Executes the list and signals the fence; `frameValue` identifies the submission.
    const uint64_t frameValue = queue.submit(cmd);
    assert(frameState.getTimelineValue() == frameValue);

    // Later: test or block on the value.
    if(queue.isComplete(frameValue) || frameState.testSignaled())
    {
      // the GPU finished this frame
    }
    queue.waitForValue(frameValue);
  }

#ifndef NVDX12_DISABLE_DYNAMIC_SEMAPHORE_STATE
  // Several threads submit to the queue: the order of their submissions, and
  // so the fence value of each, is only known at submit time.
  {
    ID3D12GraphicsCommandList* cmd{};  // EX: a command list recorded by this thread

    // Copies of a dynamic state are stored before the submission; they
    // cannot be waited on until the value is set.
    nvdx12::SemaphoreState pending = queue.createDynamicSemaphoreState();
    nvdx12::SemaphoreState copy    = pending;  // EX: kept by a garbage collector
    assert(!copy.canWait());

    // The submission fills in the value for every copy.
    queue.submit(cmd, pending);
    assert(copy.canWait() && copy.getTimelineValue() == pending.getTimelineValue());
    copy.wait(UINT64_MAX);
  }
#endif

  // nvvk-style submission: the lists of another timeline wait on this one.
  {
    nvdx12::QueueTimeline      otherQueue;  // EX: initialized on a compute queue
    ID3D12GraphicsCommandList* cmdA{};      // EX: closed command lists
    ID3D12GraphicsCommandList* cmdB{};

    // The state of the next submission to `queue`, known before it happens.
    nvdx12::SemaphoreState nextStateA = queue.createNextFixedSemaphoreState();

    nvdx12::SubmitInfo submitInfoB;
    submitInfoB.append(cmdB);
    submitInfoB.waitSemaphoreStates.push_back({nextStateA});
    nvdx12::SemaphoreState submittedStateB;
    NVDX12_CHECK(otherQueue.submit(submitInfoB, submittedStateB));

    nvdx12::SubmitInfo submitInfoA;
    submitInfoA.append(cmdA, nextStateA);  // completes nextStateA
    nvdx12::SemaphoreState submittedStateA;
    NVDX12_CHECK(queue.submit(submitInfoA, submittedStateA));
    assert(submittedStateA.getTimelineValue() == nextStateA.getTimelineValue());

    submittedStateB.wait(UINT64_MAX);
  }

  // One-shot work: execute and block.
  ID3D12GraphicsCommandList* setupCmd{};
  queue.submitAndWait(setupCmd);

  // Drain the queue before destroying resources.
  queue.waitIdle();
  queue.removeSignalListener(listener);
  queue.deinit();
}
