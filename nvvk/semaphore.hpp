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

  static inline SemaphoreState makeDynamic(VkSemaphore semaphore)
  {
    SemaphoreState semState;
    semState.initDynamic(semaphore);
    return semState;
  }

  SemaphoreState() {}

  inline void initFixed(VkSemaphore semaphore, uint64_t timelineValue)
  {
    assert(m_semaphore == VK_NULL_HANDLE);
    assert(timelineValue && semaphore);
    m_semaphore    = semaphore;
    m_fixedValue   = timelineValue;
    m_dynamicValue = nullptr;
  }

  inline void initDynamic(VkSemaphore semaphore)
  {
    assert(m_semaphore == VK_NULL_HANDLE);
    assert(semaphore);
    m_semaphore    = semaphore;
    m_fixedValue   = 0;
    m_dynamicValue = std::make_shared<std::atomic_uint64_t>(0);
  }

  inline bool isValid() const { return m_semaphore && (m_fixedValue != 0 || m_dynamicValue); }
  inline bool isFixed() const { return m_semaphore && (m_fixedValue != 0); }
  inline bool isDynamic() const { return m_semaphore && (m_dynamicValue); }

  inline VkSemaphore getSemaphore() const { return m_semaphore; }
  inline uint64_t    getTimelineValue() const
  {
    if(m_fixedValue)
    {
      return m_fixedValue;
    }
    else if(m_dynamicValue)
    {
      return m_dynamicValue->load();
    }
    else
    {
      return 0;
    }
  }

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
    return m_semaphore && (m_fixedValue != 0 || (m_dynamicValue && m_dynamicValue->load() != 0));
  }
  inline bool canWait()
  {
    fixate();
    return static_cast<const SemaphoreState*>(this)->canWait();
  }


private:
  // attempts to convert dynamic to fixed value if possible,
  // can speed up future waits.
  void fixate();

  VkSemaphore m_semaphore{};

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

  // stores either the fixed value, or is updated to have
  // a local cache of the dynamic value.
  // By design a dynamic value can only once be changed from 0 to its real value.
  uint64_t m_fixedValue{};
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
