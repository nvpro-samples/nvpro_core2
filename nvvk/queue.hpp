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

#include <vector>
#include <mutex>
#include <span>

#include "semaphore.hpp"
#include "resources.hpp"

namespace nvvk {

// When passing command buffers around, it can be useful to keep information
// about their submission to allow more complex scheduling and express
// dependencies.
struct CmdPreSubmitInfo
{
  // The command buffer that a user would typically enqueue operations to
  VkCommandBuffer cmd{};

  // information about its future submission
  uint32_t deviceMask       = 0;
  uint32_t queueFamilyIndex = 0;

  // contains the SemaphoreState that is triggered when this command buffer is submitted.
  // By making a copy of it, one can later test if the command buffer has been completed.
  SemaphoreState semaphoreState;

  // add SemaphoreStates here, that this command buffer must wait on when being submitted,
  // or are required to be signaled as well.
  std::vector<SemaphoreSubmitState> waitSemaphores;
  std::vector<SemaphoreSubmitState> signalSemapores;
};

// Wraps VkSubmitInfo2
class SubmitInfo
{
public:
  const void*                            pNext       = nullptr;
  VkSubmitFlags                          submitFlags = 0;
  std::vector<VkCommandBufferSubmitInfo> commandBuffers;

  std::vector<VkSemaphoreSubmitInfo> waitSemaphores;
  std::vector<VkSemaphoreSubmitInfo> signalSemaphores;

  // any submits using SemaphoreStates should be put in these vectors
  // and not lowered into the above
  std::vector<SemaphoreSubmitState> waitSemaphoreStates;
  std::vector<SemaphoreSubmitState> signalSemaphoreStates;

  void clear()
  {
    commandBuffers.clear();
    waitSemaphores.clear();
    signalSemaphores.clear();
    waitSemaphoreStates.clear();
    signalSemaphoreStates.clear();
    submitFlags = 0;
    pNext       = nullptr;
  }

  void append(const CmdPreSubmitInfo& preSubmit,
              VkPipelineStageFlags2   signalStageMask   = VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT_KHR,
              uint32_t                signalDeviceIndex = 0);

  void append(VkCommandBuffer       cmd,
              SemaphoreState        signalSemaphoreState,
              uint32_t              cmdDeviceMask     = 0,
              VkPipelineStageFlags2 signalStageMask   = VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT_KHR,
              uint32_t              signalDeviceIndex = 0);

  void append(VkCommandBuffer cmd, uint32_t cmdDeviceMask = 0);
};

// The QueueTimeline class manages a dedicated timeline semaphore
// for a given queue. It provides utilities that aid scheduling work
// between queues and tracking completion of submits to it.
// It is thread-safe.
class QueueTimeline
{
public:
  QueueTimeline()                                = default;
  QueueTimeline(const QueueTimeline&)            = delete;
  QueueTimeline& operator=(const QueueTimeline&) = delete;
  QueueTimeline(QueueTimeline&& other) noexcept;
  QueueTimeline& operator=(QueueTimeline&& other) noexcept;
  ~QueueTimeline();

  void init(VkDevice device, QueueInfo queueInfo);
  void deinit();

  QueueInfo getQueueInfo() const { return m_queueInfo; }

  // Create fresh semaphore state for any submission you might want to make at a later time.
  // Can create as many as needed, but must be part of `submitInfo.signalSemaphoreStates`
  SemaphoreState createDynamicSemaphoreState() const;

  // returns semaphore state of submit
  VkResult submit(SubmitInfo& info, SemaphoreState& submitState, VkFence fence = nullptr);

protected:
  VkDevice    m_device{};
  QueueInfo   m_queueInfo{};
  VkSemaphore m_timelineSemaphore{};
  // value of the next submit
  uint64_t   m_timelineValue = 1;
  std::mutex m_mutex{};
};
}  // namespace nvvk