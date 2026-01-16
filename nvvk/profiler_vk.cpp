/*
* Copyright (c) 2014-2025, NVIDIA CORPORATION.  All rights reserved.
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
* SPDX-FileCopyrightText: Copyright (c) 2014-2024, NVIDIA CORPORATION.
* SPDX-License-Identifier: Apache-2.0
*/

#include <cassert>

#include "check_error.hpp"
#include "profiler_vk.hpp"
#include "debug_util.hpp"

namespace nvvk {

ProfilerGpuTimer::~ProfilerGpuTimer()
{
  assert(m_device == nullptr && "Missing deinit()");
}

void ProfilerGpuTimer::init(nvutils::ProfilerTimeline* profilerTimeline, VkDevice device, VkPhysicalDevice physicalDevice, int queueFamilyIndex, bool useLabels)
{
  assert(m_device == nullptr);

  m_profilerTimeline = profilerTimeline;
  m_device           = device;

  VkPhysicalDeviceProperties properties;
  vkGetPhysicalDeviceProperties(physicalDevice, &properties);

  m_frequency = properties.limits.timestampPeriod;

  std::vector<VkQueueFamilyProperties> queueProperties;
  uint32_t                             queueFamilyCount = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, nullptr);
  queueProperties.resize(queueFamilyCount);
  vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, queueProperties.data());

  uint32_t validBits = queueProperties[queueFamilyIndex].timestampValidBits;

  m_queueFamilyMask            = validBits == 64 ? uint64_t(-1) : ((uint64_t(1) << validBits) - uint64_t(1));
  m_timeProvider.apiName       = "VK";
  m_timeProvider.frameFunction = [&](nvutils::ProfilerTimeline::FrameSectionID sec, double& gpuTime) {
    uint32_t idx = nvutils::ProfilerTimeline::GpuTimeProvider::getTimerBaseIdx(sec);
    return provideTime(m_frame, idx, gpuTime);
  };
  m_timeProvider.asyncFunction = [&](nvutils::ProfilerTimeline::AsyncSectionID sec, double& gpuTime) {
    std::lock_guard lock(m_asyncMutex);
    uint32_t        idx = nvutils::ProfilerTimeline::GpuTimeProvider::getTimerBaseIdx(sec);
    return provideTime(m_async, idx, gpuTime);
  };

  resizePool(m_frame, PoolContainer::POOL_QUERY_COUNT);
  resizePool(m_async, PoolContainer::POOL_QUERY_COUNT);

  m_useLabels = useLabels && (vkCmdBeginDebugUtilsLabelEXT != nullptr) && (vkCmdEndDebugUtilsLabelEXT != nullptr);
}

void ProfilerGpuTimer::deinit()
{
  if(!m_device)
    return;

  for(auto& it : m_frame.queryPools)
  {
    vkDestroyQueryPool(m_device, it, nullptr);
  }
  m_frame = {};

  for(auto& it : m_async.queryPools)
  {
    vkDestroyQueryPool(m_device, it, nullptr);
  }
  m_async = {};

  m_device           = nullptr;
  m_profilerTimeline = nullptr;
}

nvutils::ProfilerTimeline::FrameSectionID ProfilerGpuTimer::cmdFrameBeginSection(VkCommandBuffer cmd, const std::string& name)
{
  nvutils::ProfilerTimeline::FrameSectionID sec = m_profilerTimeline->frameBeginSection(name, &m_timeProvider);
  uint32_t                                  idx = nvutils::ProfilerTimeline::GpuTimeProvider::getTimerBaseIdx(sec);

  uint32_t    idxInPool;
  VkQueryPool queryPool = getPool(m_frame, idx, idxInPool);

  if(m_useLabels)
  {
    VkDebugUtilsLabelEXT label = {VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT};
    label.pLabelName           = name.c_str();
    label.color[1]             = 1.0f;
    vkCmdBeginDebugUtilsLabelEXT(cmd, &label);
  }

  vkResetQueryPool(m_device, queryPool, idxInPool, 2);

  m_profilerTimeline->frameResetCpuBegin(sec);

  // log timestamp
  vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, queryPool, idxInPool);

  return sec;
}

void ProfilerGpuTimer::cmdFrameEndSection(VkCommandBuffer cmd, nvutils::ProfilerTimeline::FrameSectionID sec)
{
  uint32_t    idx = nvutils::ProfilerTimeline::GpuTimeProvider::getTimerBaseIdx(sec) + 1;
  uint32_t    idxInPool;
  VkQueryPool queryPool = getPool(m_frame, idx, idxInPool);

  vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queryPool, idxInPool);
  if(m_useLabels)
  {
    vkCmdEndDebugUtilsLabelEXT(cmd);
  }
  m_profilerTimeline->frameEndSection(sec);
}

nvutils::ProfilerTimeline::AsyncSectionID ProfilerGpuTimer::cmdAsyncBeginSection(VkCommandBuffer cmd, const std::string& name)
{
  nvutils::ProfilerTimeline::AsyncSectionID sec = m_profilerTimeline->asyncBeginSection(name, &m_timeProvider);
  uint32_t                                  idx = nvutils::ProfilerTimeline::GpuTimeProvider::getTimerBaseIdx(sec);

  uint32_t    idxInPool;
  VkQueryPool queryPool;
  {
    std::lock_guard lock(m_asyncMutex);
    queryPool = getPool(m_async, idx, idxInPool);
  }

  if(m_useLabels)
  {
    VkDebugUtilsLabelEXT label = {VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT};
    label.pLabelName           = name.c_str();
    label.color[1]             = 1.0f;
    vkCmdBeginDebugUtilsLabelEXT(cmd, &label);
  }

  vkResetQueryPool(m_device, queryPool, idxInPool, 2);

  m_profilerTimeline->asyncResetCpuBegin(sec);

  // log timestamp
  vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, queryPool, idxInPool);

  return sec;
}

void ProfilerGpuTimer::cmdAsyncEndSection(VkCommandBuffer cmd, nvutils::ProfilerTimeline::AsyncSectionID sec)
{
  uint32_t idx = nvutils::ProfilerTimeline::GpuTimeProvider::getTimerBaseIdx(sec) + 1;

  uint32_t    idxInPool;
  VkQueryPool queryPool = getPool(m_async, idx, idxInPool);

  vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queryPool, idxInPool);
  if(m_useLabels)
  {
    vkCmdEndDebugUtilsLabelEXT(cmd);
  }
  m_profilerTimeline->asyncEndSection(sec);
}


bool ProfilerGpuTimer::provideTime(const PoolContainer& container, uint32_t idxBegin, double& gpuTime) const
{
  uint32_t    idxInPool;
  VkQueryPool queryPool = getPool(container, idxBegin, idxInPool);

  uint64_t times[2];
  VkResult result = vkGetQueryPoolResults(m_device, queryPool, idxInPool, 2, sizeof(uint64_t) * 2, times,
                                          sizeof(uint64_t), VK_QUERY_RESULT_64_BIT);

  if(result == VK_SUCCESS)
  {
    uint64_t mask = m_queueFamilyMask;
    gpuTime       = (double((times[1] & mask) - (times[0] & mask)) * double(m_frequency)) / double(1000);
    return true;
  }
  else
  {
    return false;
  }
}

VkQueryPool ProfilerGpuTimer::getPool(PoolContainer& container, uint32_t idx, uint32_t& idxInPool)
{
  idxInPool = idx % PoolContainer::POOL_QUERY_COUNT;

  // early out
  if(idx <= container.queryPoolSize)
  {
    return container.queryPools[idx / PoolContainer::POOL_QUERY_COUNT];
  }

  resizePool(container, idx);
  return container.queryPools[idx / PoolContainer::POOL_QUERY_COUNT];
}

VkQueryPool ProfilerGpuTimer::getPool(const PoolContainer& container, uint32_t idx, uint32_t& idxInPool) const
{
  idxInPool = idx % PoolContainer::POOL_QUERY_COUNT;

  return container.queryPools[idx / PoolContainer::POOL_QUERY_COUNT];
}

void ProfilerGpuTimer::resizePool(PoolContainer& container, uint32_t requiredSize)
{

  uint32_t oldCount = container.queryPoolSize / PoolContainer::POOL_QUERY_COUNT;
  uint32_t newCount = (requiredSize + PoolContainer::POOL_QUERY_COUNT - 1) / PoolContainer::POOL_QUERY_COUNT;

  container.queryPools.resize(newCount);

  for(uint32_t i = oldCount; i < newCount; i++)
  {
    VkQueryPoolCreateInfo createInfo = {VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
    createInfo.queryType             = VK_QUERY_TYPE_TIMESTAMP;
    createInfo.queryCount            = PoolContainer::POOL_QUERY_COUNT;

    NVVK_CHECK(vkCreateQueryPool(m_device, &createInfo, nullptr, &container.queryPools[i]));
    NVVK_DBG_NAME(container.queryPools[i]);
  }

  container.queryPoolSize = newCount * PoolContainer::POOL_QUERY_COUNT;
}
}  // namespace nvvk

//--------------------------------------------------------------------------------------------------
// Usage example
//--------------------------------------------------------------------------------------------------
[[maybe_unused]] static void usage_ProfilerGpuTimer()
{
  VkDevice         device{};
  VkPhysicalDevice physicalDevice{};


  nvutils::ProfilerManager profilerManager;

  // in a typical single-threaded main loop we have one timeline on which we submit commands.
  nvutils::ProfilerTimeline* profilerTimeline = profilerManager.createTimeline({"primary"});

  // the above timeline will represent a VkQueue
  uint32_t queueFamilyIndex = 0;

  nvvk::ProfilerGpuTimer gpuTimer;
  gpuTimer.init(profilerTimeline, device, physicalDevice, queueFamilyIndex, true);

  // or re-occurring per-frame events
  /* while(!glfwWindowShouldClose()) */
  {
    profilerTimeline->frameAdvance();

    VkCommandBuffer cmd{};  // EX acquire somehow per-frame

    {
      // per-frame sections must be within frameBegin/frameEnd
      // and are NOT thread-safe with respect to the timeline.

      auto profiledSection = gpuTimer.cmdFrameSection(cmd, "processing");

      // do some work
    }

    // submit command buffer

    // get snapshot for visualization, printing etc.
    std::vector<nvutils::ProfilerTimeline::Snapshot> frameSnapshots;
    std::vector<nvutils::ProfilerTimeline::Snapshot> asyncSnapshots;

    profilerManager.getSnapshots(frameSnapshots, asyncSnapshots);
  }

  profilerManager.destroyTimeline(profilerTimeline);
}
