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

#pragma once

#include <atomic>

#include <vulkan/vulkan_core.h>
#include <nvutils/profiler.hpp>

namespace nvvk {
class ProfilerGpuTimer
{
public:
  ProfilerGpuTimer() = default;
  ~ProfilerGpuTimer();

  // `profilerTimeline` pointer is copied and must be kept alive during this class lifetime
  void init(nvutils::ProfilerTimeline* profilerTimeline, VkDevice device, VkPhysicalDevice physicalDevice, int queueFamilyIndex, bool useLabels);
  void deinit();

  nvutils::ProfilerTimeline*       getProfilerTimeline() { return m_profilerTimeline; }
  const nvutils::ProfilerTimeline* getProfilerTimeline() const { return m_profilerTimeline; }

  // not thread-safe
  nvutils::ProfilerTimeline::FrameSectionID cmdFrameBeginSection(VkCommandBuffer cmd, const std::string& name);
  void cmdFrameEndSection(VkCommandBuffer cmd, nvutils::ProfilerTimeline::FrameSectionID slot);

  // thread-safe
  nvutils::ProfilerTimeline::AsyncSectionID cmdAsyncBeginSection(VkCommandBuffer cmd, const std::string& name);
  void cmdAsyncEndSection(VkCommandBuffer cmd, nvutils::ProfilerTimeline::AsyncSectionID slot);

  //////////////////////////////////////////////////////////////////////////

  // utility class to call begin/end within local scope
  class FrameSection
  {
  public:
    FrameSection(ProfilerGpuTimer& profilerGpuTimer, nvutils::ProfilerTimeline::FrameSectionID id, VkCommandBuffer cmd)
        : m_profilerGpuTimer(profilerGpuTimer)
        , m_cmd(cmd)
        , m_id(id) {};
    ~FrameSection() { m_profilerGpuTimer.cmdFrameEndSection(m_cmd, m_id); }

  private:
    ProfilerGpuTimer&                         m_profilerGpuTimer;
    VkCommandBuffer                           m_cmd;
    nvutils::ProfilerTimeline::FrameSectionID m_id;
  };

  // frame section must be within beginFrame/endFrame
  // not thread-safe
  FrameSection cmdFrameSection(VkCommandBuffer cmd, const std::string& name)
  {
    return FrameSection(*this, cmdFrameBeginSection(cmd, name), cmd);
  }

  // utility class to call begin/end within local scope
  class AsyncSection
  {
  public:
    AsyncSection(ProfilerGpuTimer& profilerGpuTimer, nvutils::ProfilerTimeline::AsyncSectionID id, VkCommandBuffer cmd)
        : m_profilerGpuTimer(profilerGpuTimer)
        , m_cmd(cmd)
        , m_id(id) {};
    ~AsyncSection() { m_profilerGpuTimer.cmdAsyncEndSection(m_cmd, m_id); }

  private:
    ProfilerGpuTimer&                         m_profilerGpuTimer;
    VkCommandBuffer                           m_cmd;
    nvutils::ProfilerTimeline::AsyncSectionID m_id;
  };

  // thread-safe
  AsyncSection cmdAsyncSection(VkCommandBuffer cmd, const std::string& name)
  {
    return AsyncSection(*this, cmdAsyncBeginSection(cmd, name), cmd);
  }

  //////////////////////////////////////////////////////////////////////////

protected:
  struct PoolContainer
  {
    // to keep the thread-safe locking operations quick,
    // we have a virtual array of queries distributed over
    // N VkQueryPools. Each query pool contains POOL_QUERY_COUNT
    // many queries.

    static constexpr uint32_t POOL_QUERY_COUNT = 1024;

    // each VkQueryPool contains POOL_BATCH_SIZE size many queries
    std::vector<VkQueryPool> queryPools;
    uint32_t                 queryPoolSize = 0;
  };

  bool provideTime(const PoolContainer& container, uint32_t idx, double& time) const;

  VkQueryPool getPool(PoolContainer& container, uint32_t idx, uint32_t& idxInPool);
  VkQueryPool getPool(const PoolContainer& container, uint32_t idx, uint32_t& idxInPool) const;
  void        resizePool(PoolContainer& container, uint32_t requiredSize);

  nvutils::ProfilerTimeline*                 m_profilerTimeline{};
  nvutils::ProfilerTimeline::GpuTimeProvider m_timeProvider;

  VkDevice m_device          = {VK_NULL_HANDLE};
  bool     m_useLabels       = false;
  float    m_frequency       = 1.0f;
  uint64_t m_queueFamilyMask = ~0;

  PoolContainer      m_frame;
  PoolContainer      m_async;
  mutable std::mutex m_asyncMutex;
};

}  // namespace nvvk
