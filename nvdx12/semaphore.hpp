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

#pragma once

#include <atomic>
#include <cassert>
#include <cstdint>
#include <memory>

#include "resources.hpp"

namespace nvdx12 {

// Creates a fence with `initialValue`, the counterpart of nvvk::createTimelineSemaphore.
HRESULT createTimelineSemaphore(ID3D12Device* device, uint64_t initialValue, ComPtr<ID3D12Fence>& semaphore);

/*-------------------------------------------------------------------------------------------------
# class nvdx12::SemaphoreState

> A fence (D3D12's timeline semaphore) paired with a timeline value, the
> D3D12 counterpart of nvvk::SemaphoreState.

The state is one of two kinds:
  - fixed: the timeline value is known and cannot change
  - dynamic: the value is provided later, exactly once, through `setDynamicValue`

The dynamic kind describes a submission whose fence value is not known
yet; every copy of the state sees the value once the submitting code sets
it. In both cases copies of the state test (`testSignaled`) or block on
(`wait`) the completion of the submission. Define
NVDX12_DISABLE_DYNAMIC_SEMAPHORE_STATE to compile the fixed kind alone.

Once the device is removed, a fence reports UINT64_MAX as its completed
value, and every test and wait here (and in QueueTimeline) reports
completion: polling loops end instead of spinning. Callers tell a loss from
a completion with nvdx12::Context::checkDeviceRemoved.

Differences with nvvk that follow from D3D12:
- The semaphore is an ID3D12Fence, reference counted like Buffer::buffer.
- `wait`, `testSignaled`, `getSemaphoreCounterValue` and
  SemaphoreStateSignalCache::testSignaled take no device: a fence is
  queried on its own. `wait` takes nanoseconds as vkWaitSemaphores does and
  returns S_OK, WAIT_TIMEOUT or an error HRESULT.
- SemaphoreSubmitState has no stage mask or device index: a D3D12 queue
  waits and signals between whole command lists, on one node.
  makeSemaphoreSubmitInfo returns a SemaphoreInfo, what
  ID3D12CommandQueue::Wait / Signal take.

See usage_SemaphoreState() in semaphore.cpp.
-------------------------------------------------------------------------------------------------*/

class SemaphoreState
{
public:
  static inline SemaphoreState makeFixed(const ComPtr<ID3D12Fence>& semaphore, uint64_t timelineValue)
  {
    SemaphoreState semState;
    semState.initFixed(semaphore, timelineValue);
    return semState;
  }

  static inline SemaphoreState makeFixed(const SemaphoreInfo& semaphoreInfo)
  {
    SemaphoreState semState;
    semState.initFixed(semaphoreInfo.semaphore, semaphoreInfo.value);
    return semState;
  }

#ifndef NVDX12_DISABLE_DYNAMIC_SEMAPHORE_STATE
  static inline SemaphoreState makeDynamic(const ComPtr<ID3D12Fence>& semaphore)
  {
    SemaphoreState semState;
    semState.initDynamic(semaphore);
    return semState;
  }
#endif

  SemaphoreState() {}

  inline void initFixed(const ComPtr<ID3D12Fence>& semaphore, uint64_t timelineValue)
  {
    assert(!m_semaphore);
    assert(timelineValue && semaphore);
    m_semaphore  = semaphore;
    m_fixedValue = timelineValue;
#ifndef NVDX12_DISABLE_DYNAMIC_SEMAPHORE_STATE
    m_dynamicValue = nullptr;
#endif
  }

#ifndef NVDX12_DISABLE_DYNAMIC_SEMAPHORE_STATE
  inline void initDynamic(const ComPtr<ID3D12Fence>& semaphore)
  {
    assert(!m_semaphore);
    assert(semaphore);
    m_semaphore    = semaphore;
    m_fixedValue   = 0;
    m_dynamicValue = std::make_shared<std::atomic_uint64_t>(0);
  }
#endif

  inline bool isValid() const
  {
    return m_semaphore
           && (m_fixedValue != 0
#ifndef NVDX12_DISABLE_DYNAMIC_SEMAPHORE_STATE
               || m_dynamicValue
#endif
           );
  }
  inline bool isFixed() const { return m_semaphore && (m_fixedValue != 0); }
#ifndef NVDX12_DISABLE_DYNAMIC_SEMAPHORE_STATE
  inline bool isDynamic() const { return m_semaphore && (m_dynamicValue); }
#endif

  inline ID3D12Fence* getSemaphore() const { return m_semaphore.Get(); }
  // 0 while a dynamic state has no value yet.
  inline uint64_t getTimelineValue() const
  {
    if(m_fixedValue)
    {
      return m_fixedValue;
    }
#ifndef NVDX12_DISABLE_DYNAMIC_SEMAPHORE_STATE
    else if(m_dynamicValue)
    {
      return m_dynamicValue->load();
    }
#endif
    else
    {
      return 0;
    }
  }

  inline uint64_t getTimelineValueFixated()
  {
#ifndef NVDX12_DISABLE_DYNAMIC_SEMAPHORE_STATE
    fixate();
#endif
    return getTimelineValue();
  }

#ifndef NVDX12_DISABLE_DYNAMIC_SEMAPHORE_STATE
  // Legal once, and only for a dynamic state: stores the value every copy
  // shares, then fixates this copy.
  void setDynamicValue(uint64_t value)
  {
    assert(isDynamic() && m_dynamicValue->load() == 0);
    m_dynamicValue->store(value);
    fixate();
  }
#endif

  // A dynamic state without a value yet (the submission is pending) cannot
  // be waited on: `wait` fails and `testSignaled` is false. The non-const
  // versions fixate the value first, which speeds up later waits and tests.
  // Returns S_OK on success (and once the device is removed), WAIT_TIMEOUT
  // on timeout, an error HRESULT otherwise.
  HRESULT wait(uint64_t timeoutNanoseconds) const;
#ifndef NVDX12_DISABLE_DYNAMIC_SEMAPHORE_STATE
  HRESULT wait(uint64_t timeoutNanoseconds);
#endif

  // Completed value of the fence; UINT64_MAX once the device is removed.
  uint64_t getSemaphoreCounterValue() const;

  // True once the value completed, and once the device is removed.
  bool testSignaled() const;
#ifndef NVDX12_DISABLE_DYNAMIC_SEMAPHORE_STATE
  bool testSignaled();
#endif

  inline bool canWait() const
  {
    return m_semaphore
           && (m_fixedValue != 0
#ifndef NVDX12_DISABLE_DYNAMIC_SEMAPHORE_STATE
               || (m_dynamicValue && m_dynamicValue->load() != 0)
#endif
           );
  }
#ifndef NVDX12_DISABLE_DYNAMIC_SEMAPHORE_STATE
  inline bool canWait()
  {
    fixate();
    return static_cast<const SemaphoreState*>(this)->canWait();
  }
#endif

private:
#ifndef NVDX12_DISABLE_DYNAMIC_SEMAPHORE_STATE
  // Converts a dynamic state whose value is known into a fixed one.
  void fixate();
#endif

  ComPtr<ID3D12Fence> m_semaphore{};

  // The fixed value, or the local copy of a dynamic value once known. A
  // dynamic value changes from 0 to its real value exactly once.
  uint64_t m_fixedValue{};

#ifndef NVDX12_DISABLE_DYNAMIC_SEMAPHORE_STATE
  // Value of a pending submission, shared by every copy of the state; 0
  // until the submission sets it. Null for a fixed state.
  std::shared_ptr<std::atomic_uint64_t> m_dynamicValue;
#endif
};

// Caches the counter of one fence so an array of states of the same submit
// is tested with a single query.
struct SemaphoreStateSignalCache
{
  ComPtr<ID3D12Fence> semaphore{};
  // Last observed counter of `semaphore`; the timeline is monotonic, so
  // anything at or below this value is signaled. A fence already cached is
  // deliberately not re-queried: callers scan from oldest to newest, and
  // refreshing would cost a query per pending entry.
  uint64_t timelineValue = 0;

  inline bool testSignaled(const SemaphoreState& semaphoreState)
  {
    // a value of 0 means the state's submit is still pending
    const uint64_t value = semaphoreState.getTimelineValue();
    if(value == 0)
    {
      return false;
    }

    if(semaphore.Get() != semaphoreState.getSemaphore())
    {
      semaphore     = semaphoreState.getSemaphore();
      timelineValue = semaphoreState.getSemaphoreCounterValue();
    }

    return timelineValue >= value;
  }
#ifndef NVDX12_DISABLE_DYNAMIC_SEMAPHORE_STATE
  // The non-const state fixates its value first.
  inline bool testSignaled(SemaphoreState& semaphoreState)
  {
    const uint64_t value = semaphoreState.getTimelineValueFixated();
    if(value == 0)
    {
      return false;
    }

    if(semaphore.Get() != semaphoreState.getSemaphore())
    {
      semaphore     = semaphoreState.getSemaphore();
      timelineValue = semaphoreState.getSemaphoreCounterValue();
    }

    return timelineValue >= value;
  }
#endif
};

// A SemaphoreState to wait on or signal with a submission (QueueTimeline::submit).
struct SemaphoreSubmitState
{
  SemaphoreState semaphoreState;
};

// The fence and value a queue waits on or signals; the state needs a value
// (canWait), which QueueTimeline::submit checks before it lowers the states.
inline SemaphoreInfo makeSemaphoreSubmitInfo(const SemaphoreState& semaphoreState)
{
  assert(semaphoreState.isValid());
  SemaphoreInfo semaphoreSubmitInfo{semaphoreState.getSemaphore(), semaphoreState.getTimelineValue()};
  // assert proper timeline value has been set
  assert(semaphoreSubmitInfo.value && "semaphore state has invalid timelineValue");
  return semaphoreSubmitInfo;
}

inline SemaphoreInfo makeSemaphoreSubmitInfo(const SemaphoreSubmitState& state)
{
  return makeSemaphoreSubmitInfo(state.semaphoreState);
}

}  // namespace nvdx12
