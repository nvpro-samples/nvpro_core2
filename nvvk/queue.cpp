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

#include "queue.hpp"
#include <volk.h>

namespace nvvk {

void SubmitInfo::append(const CmdPreSubmitInfo& preSubmit, VkPipelineStageFlags2 signalStageMask, uint32_t signalDeviceIndex)
{
  VkCommandBufferSubmitInfo cmdSubmitInfo = {
      .sType         = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
      .commandBuffer = preSubmit.cmd,
      .deviceMask    = preSubmit.deviceMask,
  };

  commandBuffers.emplace_back(cmdSubmitInfo);
  waitSemaphoreStates.insert(waitSemaphoreStates.end(), preSubmit.waitSemaphores.cbegin(), preSubmit.waitSemaphores.cend());
  signalSemaphoreStates.insert(signalSemaphoreStates.end(), preSubmit.signalSemapores.cbegin(),
                               preSubmit.signalSemapores.cend());

  SemaphoreSubmitState selfSubmitState{
      .semaphoreState = preSubmit.semaphoreState,
      .stageMask      = signalStageMask,
      .deviceIndex    = signalDeviceIndex,
  };

  signalSemaphoreStates.emplace_back(selfSubmitState);
}

void SubmitInfo::append(VkCommandBuffer       cmd,
                        SemaphoreState        signalSemaphoreState,
                        uint32_t              cmdDeviceMask,
                        VkPipelineStageFlags2 signalStageMask,
                        uint32_t              signalDeviceIndex)
{
  VkCommandBufferSubmitInfo cmdSubmitInfo = {
      .sType         = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
      .commandBuffer = cmd,
      .deviceMask    = cmdDeviceMask,
  };

  commandBuffers.emplace_back(cmdSubmitInfo);

  SemaphoreSubmitState selfSubmitState{
      .semaphoreState = signalSemaphoreState,
      .stageMask      = signalStageMask,
      .deviceIndex    = signalDeviceIndex,
  };

  signalSemaphoreStates.emplace_back(selfSubmitState);
}

void SubmitInfo::append(VkCommandBuffer cmd, uint32_t cmdDeviceMask)
{
  VkCommandBufferSubmitInfo cmdSubmitInfo = {
      .sType         = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
      .commandBuffer = cmd,
      .deviceMask    = cmdDeviceMask,
  };

  commandBuffers.push_back(cmdSubmitInfo);
}

//////////////////////////////////////////////////////////////////////////

QueueTimeline::QueueTimeline(QueueTimeline&& other) noexcept
{
  {
    std::lock_guard lock(other.m_mutex);
    std::swap(m_device, other.m_device);
    std::swap(m_queueInfo, other.m_queueInfo);
    std::swap(m_timelineSemaphore, other.m_timelineSemaphore);
    std::swap(m_timelineValue, other.m_timelineValue);
  }
}

nvvk::QueueTimeline& QueueTimeline::operator=(QueueTimeline&& other) noexcept
{
  if(this != &other)
  {
    assert(m_device == nullptr && "Missing deinit()");

    std::lock_guard lock(other.m_mutex);
    std::swap(m_device, other.m_device);
    std::swap(m_queueInfo, other.m_queueInfo);
    std::swap(m_timelineSemaphore, other.m_timelineSemaphore);
    std::swap(m_timelineValue, other.m_timelineValue);
  }
  return *this;
}

QueueTimeline::~QueueTimeline()
{
  assert(m_device == nullptr && "Missing deinit()");
}

void QueueTimeline::init(VkDevice device, QueueInfo queueInfo)
{
  assert(m_device == nullptr);

  m_device    = device;
  m_queueInfo = queueInfo;

  VkSemaphoreTypeCreateInfo timelineSemaphoreCreateInfo{.sType         = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,
                                                        .pNext         = nullptr,
                                                        .semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE,
                                                        .initialValue  = 0};
  VkSemaphoreCreateInfo     semaphoreCreateInfo{
          .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, .pNext = &timelineSemaphoreCreateInfo, .flags = 0};

  vkCreateSemaphore(device, &semaphoreCreateInfo, nullptr, &m_timelineSemaphore);

  m_device        = device;
  m_timelineValue = 1;
}

void QueueTimeline::deinit()
{
  if(!m_device)
    return;
  vkDestroySemaphore(m_device, m_timelineSemaphore, nullptr);
  m_timelineSemaphore = nullptr;
  m_device            = nullptr;
}

nvvk::SemaphoreState QueueTimeline::createDynamicSemaphoreState() const
{
  return SemaphoreState::makeDynamic(m_timelineSemaphore);
}

VkResult QueueTimeline::submit(SubmitInfo& submitInfo, SemaphoreState& submitState, VkFence fence)
{
  std::vector<VkSemaphoreSubmitInfo> waitSemaphores   = submitInfo.waitSemaphores;
  std::vector<VkSemaphoreSubmitInfo> signalSemaphores = submitInfo.signalSemaphores;

  VkSubmitInfo2 submitInfo2 = {
      .sType                  = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
      .flags                  = submitInfo.submitFlags,
      .commandBufferInfoCount = static_cast<uint32_t>(submitInfo.commandBuffers.size()),
      .pCommandBufferInfos    = submitInfo.commandBuffers.data(),
  };

  for(const SemaphoreSubmitState& it : submitInfo.waitSemaphoreStates)
  {
    assert(it.semaphoreState.isFixed() && "waitSemaphoreStates must have been finalized");
    waitSemaphores.push_back(makeSemaphoreSubmitInfo(it));
  }

#ifndef NDEBUG
  for(const VkSemaphoreSubmitInfo& it : signalSemaphores)
  {
    assert(it.semaphore != m_timelineSemaphore
           && "regular signalSemaphores must not use queue's timelineSemaphore, use signalSemaphoreStates instead");
  }
#endif
  for(SemaphoreSubmitState& it : submitInfo.signalSemaphoreStates)
  {
    if(it.semaphoreState.getSemaphore() != m_timelineSemaphore)
    {
      signalSemaphores.push_back(makeSemaphoreSubmitInfo(it));
    }
  }


  VkResult submitResult        = VK_SUCCESS;
  uint64_t submitTimeLineValue = 0;
  {
    std::lock_guard guard(m_mutex);

    submitTimeLineValue = m_timelineValue++;

    for(SemaphoreSubmitState& it : submitInfo.signalSemaphoreStates)
    {
      // the semaphore state is from our queue, patch in the actual
      // submit value
      if(it.semaphoreState.getSemaphore() == m_timelineSemaphore)
      {
        assert(!submitState.isValid() && "must not use submitInfo.signalSemaphoreState using queue timelineSemaphore more than once");

        it.semaphoreState.setDynamicValue(submitTimeLineValue);

        signalSemaphores.push_back(makeSemaphoreSubmitInfo(it));
      }
    }

    submitInfo2.signalSemaphoreInfoCount = static_cast<uint32_t>(signalSemaphores.size());
    submitInfo2.pSignalSemaphoreInfos    = signalSemaphores.data();
    submitInfo2.waitSemaphoreInfoCount   = static_cast<uint32_t>(waitSemaphores.size());
    submitInfo2.pWaitSemaphoreInfos      = waitSemaphores.data();

    submitResult = vkQueueSubmit2(m_queueInfo.queue, 1, &submitInfo2, fence);
  }

  submitState.initFixed(m_timelineSemaphore, submitTimeLineValue);

  return submitResult;
}

}  // namespace nvvk

#include <queue>

//--------------------------------------------------------------------------------------------------
// Usage example
//--------------------------------------------------------------------------------------------------
[[maybe_unused]] static void usage_QueueTimeline()
{
  // EX fill these somehow
  VkDevice device{};

  // each provides queue and family index and VkQueue handle
  nvvk::QueueInfo queueInfoA;
  nvvk::QueueInfo queueInfoB;

  // let's manage submits to these two VkQueues by
  // the utility class
  nvvk::QueueTimeline queueTimelineA;
  nvvk::QueueTimeline queueTimelineB;

  queueTimelineA.init(device, queueInfoA);
  queueTimelineB.init(device, queueInfoB);


  // basic operations
  {
    nvvk::SemaphoreState semaphoreStateTest;
    nvvk::SubmitInfo     submitInfo;

    // per-frame loop
    /* while(!glfwWindowShouldClose()) */
    {
      // The main purpose of the `nvvk::QueueTimeline` object is to wrap its timeline semaphore
      // that is incremented with each submit.
      //
      // The `nvvk::SemaphoreState` object can be used to get information about a future submit,
      // in such a way that it can be stored and queried safely at a later time.
      //
      // This can be useful when there is more users of a `nvvk::QueueTimeline` and it isn't as easy
      // to figure out the order of submits done to it (and therefore the corresponding timeline values).
      // The `nvvk::SemaphoreState` acts as a `future` to a pending signal operation.

      // In this basic sample we get a command buffer every frame
      VkCommandBuffer cmd{};

      // Also get a semaphore state, so that we detect the completion of the command buffer.
      // Note at this point in time we haven't submitted the command buffer yet, but this object is safe to use
      // and copy even if someone else were to submit to `queueTimelineA` before above `cmd` is.
      //
      // Until it was signaled, this semaphoreState is `dynamic` with an unknown timeline value.

      nvvk::SemaphoreState semaphoreState = queueTimelineA.createDynamicSemaphoreState();

      // let's fake actually caring for the completion of cmd
      if(!semaphoreStateTest.isValid())
      {
        // copy this frame's semaphore state, even if it hasn't been signaled/submitted yet
        semaphoreStateTest = semaphoreState;
      }
      else
      {
        // in a past frame we made the above copy
        // now we can check if the semaphore was completed
        if(semaphoreStateTest.testSignaled(device))
        {
          // do something, knowing that `cmd` from the time we made the `semaphoreStateTest` copy assignment was completed
        }
      }

      // the submits to the `QueueTimeline` object are done through a wrapper struct
      // similar to `VkSubmitInfo2`

      submitInfo.clear();
      submitInfo.append(cmd);

      // don't forget to insert the dynamic semaphore state from this frame to be signaled with the submit.
      submitInfo.signalSemaphoreStates.push_back({std::move(semaphoreState), VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT});

      // the submit process returns a fixed semaphore state, as now the timeline value is known.
      nvvk::SemaphoreState semaphoreSubmittedState;
      VkResult             result = queueTimelineA.submit(submitInfo, semaphoreSubmittedState);
    }
  }


  // more complex scenario implementing a garbage collector for a secondary queue
  {

    struct GarbageEntry
    {
      nvvk::SemaphoreState semaphoreState;
      void*                stagingData = nullptr;
    };

    std::queue<GarbageEntry> garbageEntries;

    // per-frame loop
    while(true)
    {
      // acquires command buffer for queueA
      VkCommandBuffer cmdA{};

      // We can pass this struct around to have something more portable for other code.
      nvvk::CmdPreSubmitInfo preSubmitA;
      preSubmitA.cmd            = cmdA;
      preSubmitA.semaphoreState = queueTimelineA.createDynamicSemaphoreState();

      // myprocessFunc(nvvk::CmdPreSubmitInfo& preSubmitA)
      {
        // some inner render loop function where we don't want to expose all of application
        // instead it is handed over `preSubmitA` as &
        //
        // it wants to do a submit to B that cmdA should wait for

        VkCommandBuffer      cmdB{};
        nvvk::SemaphoreState semaphoreStateB = queueTimelineB.createDynamicSemaphoreState();

        {
          // let's say we have some staging data that we need for this operation on queueB

          // first delete old completed garbage
          while(!garbageEntries.empty())
          {
            // we copied this semaphoreState in a previous frame, see a bit later
            if(garbageEntries.front().semaphoreState.testSignaled(device))
            {
              // safe to delete old stagingData
              garbageEntries.front().stagingData = nullptr;
              garbageEntries.pop();
            }
            else
            {
              // we pushed in order, so if first isn't completed then others cannot either
              break;
            }
          }

          // then get new data
          void* stagingDataB = (void*)1;

          // push new garbage with a copy of the current semaphoreStateB
          GarbageEntry entry;
          entry.semaphoreState = semaphoreStateB;
          entry.stagingData    = stagingDataB;

          garbageEntries.push(entry);
        }

        nvvk::SubmitInfo submitB;
        submitB.append(cmdB, semaphoreStateB);

        // trigger submit of B
        queueTimelineB.submit(submitB, semaphoreStateB);

        // let preSubmitA know it has to wait
        preSubmitA.waitSemaphores.push_back({.semaphoreState = std::move(semaphoreStateB)});
      }

      // application has control again and deals with the final submit

      nvvk::SubmitInfo submitA;
      submitA.append(preSubmitA);
      // might add other signals.waits for swapchain etc.

      // in case we wa
      nvvk::SemaphoreState semaphoreStateA;
      queueTimelineA.submit(submitA, semaphoreStateA);
    }
  }
}
