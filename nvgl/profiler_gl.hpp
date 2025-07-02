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
* SPDX-FileCopyrightText: Copyright (c) 2014-2025, NVIDIA CORPORATION.
* SPDX-License-Identifier: Apache-2.0
*/

#pragma once

#include <atomic>
#include <memory>
#include <nvutils/profiler.hpp>

#include "extensions.hpp"

namespace nvgl {
class ProfilerGpuTimer
{
public:
  ProfilerGpuTimer() = default;
  ~ProfilerGpuTimer();

  // `profilerTimeline` pointer is copied and must be kept alive during this class lifetime
  void init(nvutils::ProfilerTimeline* profilerTimeline);
  void deinit();

  // not thread-safe
  nvutils::ProfilerTimeline::FrameSectionID frameBeginSection(const std::string& name);
  void                                      frameEndSection(nvutils::ProfilerTimeline::FrameSectionID slot);

  // thread-safe
  nvutils::ProfilerTimeline::AsyncSectionID asyncBeginSection(const std::string& name);
  void                                      asyncEndSection(nvutils::ProfilerTimeline::AsyncSectionID slot);

  void frameAccumulationSplit();

  //////////////////////////////////////////////////////////////////////////

  // utility class to call begin/end within local scope
  class FrameSection
  {
  public:
    FrameSection(ProfilerGpuTimer& profilerGpuTimer, nvutils::ProfilerTimeline::FrameSectionID id)
        : m_profilerGpuTimer(profilerGpuTimer)
        , m_id(id) {};
    ~FrameSection() { m_profilerGpuTimer.frameEndSection(m_id); }

  private:
    ProfilerGpuTimer&                         m_profilerGpuTimer;
    nvutils::ProfilerTimeline::FrameSectionID m_id;
  };

  // frame section must be within beginFrame/endFrame
  // not thread-safe
  FrameSection frameSection(const std::string& name) { return FrameSection(*this, frameBeginSection(name)); }

  // utility class to call begin/end within local scope
  class AsyncSection
  {
  public:
    AsyncSection(ProfilerGpuTimer& profilerGpuTimer, nvutils::ProfilerTimeline::AsyncSectionID id)
        : m_profilerGpuTimer(profilerGpuTimer)
        , m_id(id) {};
    ~AsyncSection() { m_profilerGpuTimer.asyncEndSection(m_id); }

  private:
    ProfilerGpuTimer&                         m_profilerGpuTimer;
    nvutils::ProfilerTimeline::AsyncSectionID m_id;
  };

  // thread-safe
  AsyncSection asyncSection(const std::string& name) { return AsyncSection(*this, asyncBeginSection(name)); }

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
    std::vector<std::vector<GLuint>> queryPools;
    uint32_t                         queryPoolSize = 0;
  };

  bool provideTime(const PoolContainer& container, uint32_t idx, double& time) const;

  const GLuint* getPoolBegin(PoolContainer& container, uint32_t idx, uint32_t& idxInPool);
  const GLuint* getPoolEnd(const PoolContainer& container, uint32_t idx, uint32_t& idxInPool) const;
  void          resizePool(PoolContainer& container, uint32_t requiredSize);

  nvutils::ProfilerTimeline*                 m_profilerTimeline{};
  nvutils::ProfilerTimeline::GpuTimeProvider m_timeProvider;

  PoolContainer      m_frame;
  PoolContainer      m_async;
  mutable std::mutex m_asyncMutex;
};

}  // namespace nvgl
