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

#include "semaphore.hpp"
#include <volk.h>

namespace nvvk {

void SemaphoreState::fixate()
{
  if(m_fixedValue == 0 && m_dynamicValue)
  {
    m_fixedValue = m_dynamicValue->load();
    if(m_fixedValue)
    {
      // a dynamicValue can only transition away from zero
      // once, after that it can be treated like a
      // fixedValue, therefore it's safe to release the
      // shared_ptr here.
      m_dynamicValue = nullptr;
    }
  }
}

VkResult SemaphoreState::wait(VkDevice device, uint64_t timeout) const
{
  uint64_t timelineValue = getTimelineValue();
  if(!timelineValue)
    return VK_ERROR_INITIALIZATION_FAILED;

  VkSemaphoreWaitInfo waitInfo{
      .sType          = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
      .pNext          = nullptr,
      .flags          = 0,
      .semaphoreCount = 1,
      .pSemaphores    = &m_semaphore,
      .pValues        = &timelineValue,
  };
  return vkWaitSemaphores(device, &waitInfo, timeout);
}

VkResult SemaphoreState::wait(VkDevice device, uint64_t timeout)
{
  fixate();
  return static_cast<const SemaphoreState*>(this)->wait(device, timeout);
}

bool SemaphoreState::testSignaled(VkDevice device) const
{
  uint64_t timelineValue = getTimelineValue();
  if(!timelineValue)
    return false;

  uint64_t currentValue;
  VkResult r = vkGetSemaphoreCounterValue(device, m_semaphore, &currentValue);
  if(r == VK_SUCCESS)
    return currentValue >= (timelineValue);
  return false;
}

bool SemaphoreState::testSignaled(VkDevice device)
{
  fixate();
  return static_cast<const SemaphoreState*>(this)->testSignaled(device);
}

VkResult createTimelineSemaphore(VkDevice device, uint64_t initialValue, VkSemaphore& semaphore)
{
  VkSemaphoreTypeCreateInfo timelineSemaphoreCreateInfo{.sType         = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,
                                                        .pNext         = nullptr,
                                                        .semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE,
                                                        .initialValue  = initialValue};
  VkSemaphoreCreateInfo     semaphoreCreateInfo{
          .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, .pNext = &timelineSemaphoreCreateInfo, .flags = 0};

  return vkCreateSemaphore(device, &semaphoreCreateInfo, nullptr, &semaphore);
}

}  // namespace nvvk
