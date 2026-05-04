/*
* Copyright (c) 2025-2026, NVIDIA CORPORATION.  All rights reserved.
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
* SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
* SPDX-License-Identifier: Apache-2.0
*/

#pragma once

#include <memory>
#include <atomic>
#include <cassert>

#include "resources.hpp"

namespace nvvk {

// simple wrapper to create a timeline semaphore
VkResult createTimelineSemaphore(VkDevice device, uint64_t initialValue, VkSemaphore& semaphore);

// The SemaphoreState class wraps a timeline semaphore
// with a timeline value.
//
// It can be only one of two states:
//   - fixed: the timeline value is fixed and cannot be changed
//   - dynamic: the timeline value is provided at a later time, exactly once
//
// The latter usecase is intended in conjunction with the `nvvk::QueueTimeline` class.
// Any semaphore state that is signalled within `nvvk::QueueTimeline::submit(...)` that
// was created from that `nvvk::QueueTimeline` will have its timeline value updated at that time.
//
// In both cases a copy of the struct can be made to later check the completion status of
// the timeline semaphore.

class SemaphoreState
{
public:
  static inline SemaphoreState makeFixed(VkSemaphore semaphore, uint64_t timelineValue)
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

#ifndef NVVK_DISABLE_DYNAMIC_SEMAPHORE_STATE
  static inline SemaphoreState makeDynamic(VkSemaphore semaphore)
  {
    SemaphoreState semState;
    semState.initDynamic(semaphore);
    return semState;
  }
#endif

  SemaphoreState() {}

  inline void initFixed(VkSemaphore semaphore, uint64_t timelineValue)
  {
    assert(m_semaphore == VK_NULL_HANDLE);
    assert(timelineValue && semaphore);
    m_semaphore  = semaphore;
    m_fixedValue = timelineValue;
#ifndef NVVK_DISABLE_DYNAMIC_SEMAPHORE_STATE
    m_dynamicValue = nullptr;
#endif
  }

#ifndef NVVK_DISABLE_DYNAMIC_SEMAPHORE_STATE
  inline void initDynamic(VkSemaphore semaphore)
  {
    assert(m_semaphore == VK_NULL_HANDLE);
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
#ifndef NVVK_DISABLE_DYNAMIC_SEMAPHORE_STATE
               || m_dynamicValue
#endif
           );
  }
  inline bool isFixed() const { return m_semaphore && (m_fixedValue != 0); }
#ifndef NVVK_DISABLE_DYNAMIC_SEMAPHORE_STATE
  inline bool isDynamic() const { return m_semaphore && (m_dynamicValue); }
#endif

  inline VkSemaphore getSemaphore() const { return m_semaphore; }
  inline uint64_t    getTimelineValue() const
  {
    if(m_fixedValue)
    {
      return m_fixedValue;
    }
#ifndef NVVK_DISABLE_DYNAMIC_SEMAPHORE_STATE
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

#ifndef NVVK_DISABLE_DYNAMIC_SEMAPHORE_STATE
  // this function can be called only once and is only legal for
  // dynamic semaphore state
  void setDynamicValue(uint64_t value)
  {
    // must be dynamic, and must not have been set already
    assert(isDynamic() && m_dynamicValue->load() == 0);
    // updated the shared_ptr value so every copy of this
    // semaphore state has access to it.
    m_dynamicValue->store(value);

    // fixate afterwards to update local cache
    // and decouple it
    fixate();
  }
#endif
  // for dynamic values waiting and testing will always return false
  // if the m_dynamicValue->load() returns zero, meaning
  // the SemaphoreState wasn't part of a `QueueTimeline::submit` where it was
  // signaled yet.
  //
  // non-const versions implicitly try to fixate the value in the dynamic
  // case to speed things up a bit for future waits or tests.

  VkResult wait(VkDevice device, uint64_t timeout) const;
  VkResult wait(VkDevice device, uint64_t timeout);

  bool testSignaled(VkDevice device) const;
  bool testSignaled(VkDevice device);

  inline bool canWait() const
  {
    return m_semaphore
           && (m_fixedValue != 0
#ifndef NVVK_DISABLE_DYNAMIC_SEMAPHORE_STATE
               || (m_dynamicValue && m_dynamicValue->load() != 0)
#endif
           );
  }
#ifndef NVVK_DISABLE_DYNAMIC_SEMAPHORE_STATE
  inline bool canWait()
  {
    fixate();
    return static_cast<const SemaphoreState*>(this)->canWait();
  }
#endif

private:
#ifndef NVVK_DISABLE_DYNAMIC_SEMAPHORE_STATE
  // attempts to convert dynamic to fixed value if possible,
  // can speed up future waits.
  void fixate();
#endif

  VkSemaphore m_semaphore{};

  // stores either the fixed value, or is updated to have
  // a local cache of the dynamic value.
  // By design a dynamic value can only once be changed from 0 to its real value.
  uint64_t m_fixedValue{};

#ifndef NVVK_DISABLE_DYNAMIC_SEMAPHORE_STATE
  // Holds the timeline value of a semaphore.
  //
  // We are using a shared_ptr as this struct can be used to encode
  // future submits, and therefore the actual value of the timeline semaphore's
  // submission isn't known yet.
  //
  // The shared_ptr will point towards the location that will contain the final value.
  // By default the value will be 0 (not-submitted).

  // doesn't exist for "fixed" value semaphore state
  std::shared_ptr<std::atomic_uint64_t> m_dynamicValue;
#endif
};

// Simple utility class to avoid querying semaphore state.
// Typically we enqueue many semaphore states of the same submit in
// arrays and test accordingly.
struct SemaphoreStateSignalCache
{
  VkSemaphore semaphore{};
  uint64_t    timelineValue = ~0;
  bool        signaled      = false;

  inline bool testSignaled(VkDevice device, const SemaphoreState& semaphoreState)
  {
    if(semaphore == semaphoreState.getSemaphore() && timelineValue == semaphoreState.getTimelineValue())
    {
      return signaled;
    }

    semaphore     = semaphoreState.getSemaphore();
    timelineValue = semaphoreState.getTimelineValue();
    signaled      = semaphoreState.testSignaled(device);

    return signaled;
  }
#ifndef NVVK_DISABLE_DYNAMIC_SEMAPHORE_STATE
  // semaphore state has an optimization for non-const testing
  inline bool testSignaled(VkDevice device, SemaphoreState& semaphoreState)
  {
    if(semaphore == semaphoreState.getSemaphore() && timelineValue == semaphoreState.getTimelineValue())
    {
      return signaled;
    }

    semaphore     = semaphoreState.getSemaphore();
    timelineValue = semaphoreState.getTimelineValue();
    signaled      = semaphoreState.testSignaled(device);

    return signaled;
  }
#endif
};

struct SemaphoreSubmitState
{
  SemaphoreState           semaphoreState;
  VkPipelineStageFlagBits2 stageMask   = 0;
  uint32_t                 deviceIndex = 0;
};

inline VkSemaphoreSubmitInfo makeSemaphoreSubmitInfo(const SemaphoreState&    semaphoreState,
                                                     VkPipelineStageFlagBits2 stageMask,
                                                     uint32_t                 deviceIndex = 0)
{
  assert(semaphoreState.isValid());

  VkSemaphoreSubmitInfo semaphoreSubmitInfo = {
      .sType       = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
      .semaphore   = semaphoreState.getSemaphore(),
      .value       = semaphoreState.getTimelineValue(),
      .stageMask   = stageMask,
      .deviceIndex = deviceIndex,
  };

  // assert proper timeline value has been set
  assert(semaphoreSubmitInfo.value && "semaphore state has invalid timelineValue");

  return semaphoreSubmitInfo;
};

inline VkSemaphoreSubmitInfo makeSemaphoreSubmitInfo(const SemaphoreSubmitState& state)
{
  assert(state.semaphoreState.isValid());

  VkSemaphoreSubmitInfo semaphoreSubmitInfo = {
      .sType       = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
      .semaphore   = state.semaphoreState.getSemaphore(),
      .value       = state.semaphoreState.getTimelineValue(),
      .stageMask   = state.stageMask,
      .deviceIndex = state.deviceIndex,
  };

  // assert proper timeline value has been set
  assert(semaphoreSubmitInfo.value && "semaphore state has invalid timelineValue");

  return semaphoreSubmitInfo;
};


}  // namespace nvvk
