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

#include "semaphore.hpp"

#include "check_error.hpp"

namespace nvdx12 {

HRESULT createTimelineSemaphore(ID3D12Device* device, uint64_t initialValue, ComPtr<ID3D12Fence>& semaphore)
{
  return NVDX12_FAIL_REPORT(device->CreateFence(initialValue, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&semaphore)));
}

#ifndef NVDX12_DISABLE_DYNAMIC_SEMAPHORE_STATE
void SemaphoreState::fixate()
{
  if(m_fixedValue == 0 && m_dynamicValue)
  {
    m_fixedValue = m_dynamicValue->load();
    if(m_fixedValue)
    {
      // The value only ever changes once, so the shared copy is no longer needed.
      m_dynamicValue = nullptr;
    }
  }
}
#endif

HRESULT SemaphoreState::wait(uint64_t timeoutNanoseconds) const
{
  const uint64_t timelineValue = getTimelineValue();
  if(!timelineValue)
  {
    return E_FAIL;
  }

  // A removed device reports UINT64_MAX: completed, so that wait loops end.
  const uint64_t completedValue = m_semaphore->GetCompletedValue();
  if(completedValue >= timelineValue)
  {
    return S_OK;
  }

  // A zero timeout polls, as vkWaitSemaphores does: no event needed.
  if(timeoutNanoseconds == 0)
  {
    return WAIT_TIMEOUT;
  }

  // Nanoseconds to milliseconds, rounding up; very large values mean INFINITE.
  const DWORD milliseconds = static_cast<DWORD>(                       //
      (timeoutNanoseconds > (MAXDWORD * 1'000'000ULL - 999'999ULL)) ?  //
          INFINITE :                                                   //
          (timeoutNanoseconds + 999'999u) / 1'000'000u);

  // An infinite wait needs no event: SetEventOnCompletion(..., nullptr) blocks.
  if(milliseconds == INFINITE)
  {
    return NVDX12_FAIL_REPORT(m_semaphore->SetEventOnCompletion(timelineValue, nullptr));
  }

  HANDLE fenceEvent = CreateEvent(nullptr, FALSE, FALSE, nullptr);
  if(!fenceEvent)
  {
    const HRESULT hr = HRESULT_FROM_WIN32(GetLastError());
    return NVDX12_FAIL_REPORT(FAILED(hr) ? hr : E_FAIL);
  }

  const HRESULT setResult = NVDX12_FAIL_REPORT(m_semaphore->SetEventOnCompletion(timelineValue, fenceEvent));
  if(FAILED(setResult))
  {
    CloseHandle(fenceEvent);
    return setResult;
  }

  const DWORD result    = WaitForSingleObjectEx(fenceEvent, milliseconds, FALSE);
  const DWORD lastError = (result == WAIT_FAILED) ? GetLastError() : ERROR_SUCCESS;  // before CloseHandle overwrites it
  CloseHandle(fenceEvent);
  if(result == WAIT_FAILED)
  {
    return NVDX12_FAIL_REPORT(HRESULT_FROM_WIN32(lastError));
  }
  return static_cast<HRESULT>(result);  // WAIT_OBJECT_0 == S_OK, or WAIT_TIMEOUT
}

#ifndef NVDX12_DISABLE_DYNAMIC_SEMAPHORE_STATE
HRESULT SemaphoreState::wait(uint64_t timeoutNanoseconds)
{
  fixate();
  return static_cast<const SemaphoreState*>(this)->wait(timeoutNanoseconds);
}
#endif

uint64_t SemaphoreState::getSemaphoreCounterValue() const
{
  assert(m_semaphore);

  // UINT64_MAX once the device is removed: every value reads as signaled.
  return m_semaphore->GetCompletedValue();
}

bool SemaphoreState::testSignaled() const
{
  const uint64_t timelineValue = getTimelineValue();
  if(!timelineValue)
  {
    return false;
  }

  // UINT64_MAX once the device is removed: signaled, so that polling loops end.
  return m_semaphore->GetCompletedValue() >= timelineValue;
}

#ifndef NVDX12_DISABLE_DYNAMIC_SEMAPHORE_STATE
bool SemaphoreState::testSignaled()
{
  fixate();
  return static_cast<const SemaphoreState*>(this)->testSignaled();
}
#endif

}  // namespace nvdx12

//--------------------------------------------------------------------------------------------------
// Usage example
//--------------------------------------------------------------------------------------------------
[[maybe_unused]] static void usage_SemaphoreState()
{
  ID3D12Device*               device = nullptr;  // EX: from nvdx12::Context::getDevice()
  ID3D12CommandQueue*         queue{};           // EX: the queue that signals the fence
  ID3D12GraphicsCommandList*  cmd{};             // EX: a closed command list
  uint64_t                    nextValue = 1;     // EX: the queue's next fence value
  nvdx12::ComPtr<ID3D12Fence> fence;             // EX: or the fence of a queue (nvdx12::QueueTimeline::getFence())
  NVDX12_CHECK(nvdx12::createTimelineSemaphore(device, 0, fence));

  // Fixed: the value is known when the state is created.
  {
    nvdx12::SemaphoreState submitState = nvdx12::SemaphoreState::makeFixed(fence, nextValue);
    ID3D12CommandList*     lists[]     = {cmd};
    queue->ExecuteCommandLists(1, lists);
    queue->Signal(fence.Get(), nextValue++);

    // Copies of the state outlive the submission and test its completion.
    nvdx12::SemaphoreState copy = submitState;
    if(copy.testSignaled())
    {
      // resources used by `cmd` can be released
    }
    copy.wait(UINT64_MAX);  // or block; S_OK too once the device is removed (Context::checkDeviceRemoved tells)
  }

#ifndef NVDX12_DISABLE_DYNAMIC_SEMAPHORE_STATE
  // Dynamic: the state is handed out before the submission order is known.
  {
    nvdx12::SemaphoreState pending = nvdx12::SemaphoreState::makeDynamic(fence);
    nvdx12::SemaphoreState copy    = pending;  // stored by whoever needs the completion

    assert(!copy.canWait());  // no value yet

    // The code that performs the submission sets the value once.
    ID3D12CommandList* lists[] = {cmd};
    queue->ExecuteCommandLists(1, lists);
    queue->Signal(fence.Get(), nextValue);
    pending.setDynamicValue(nextValue++);

    assert(copy.canWait());  // every copy sees it
    copy.wait(UINT64_MAX);
  }
#endif

  // Many states of the same fence: one counter query for the whole array.
  nvdx12::SemaphoreStateSignalCache cache;
  nvdx12::SemaphoreState            states[2];
  for(size_t i = 0; i < std::size(states); i++)
  {
    nvdx12::SemaphoreState& state = states[i];
    if(state.isValid() && !cache.testSignaled(state))
    {
      break;  // states are pushed in order; the rest is pending too
    }
  }
}
