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

#include <cassert>

#include "profiler_gl.hpp"

namespace nvgl {

ProfilerGpuTimer::~ProfilerGpuTimer()
{
  assert(m_profilerTimeline == nullptr && "Missing deinit()");
}

void ProfilerGpuTimer::init(nvutils::ProfilerTimeline* profilerTimeline)
{
  assert(m_profilerTimeline == nullptr);

  m_profilerTimeline = profilerTimeline;


  m_timeProvider.apiName       = "GL";
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
}

void ProfilerGpuTimer::deinit()
{
  if(!m_profilerTimeline)
    return;

  for(auto& it : m_frame.queryPools)
  {
    glDeleteQueries(PoolContainer::POOL_QUERY_COUNT, it.data());
  }
  for(auto& it : m_async.queryPools)
  {
    glDeleteQueries(PoolContainer::POOL_QUERY_COUNT, it.data());
  }

  m_frame.queryPools = {};
  m_async.queryPools = {};

  m_profilerTimeline = nullptr;
}

nvutils::ProfilerTimeline::FrameSectionID ProfilerGpuTimer::frameBeginSection(const std::string& name)
{
  nvutils::ProfilerTimeline::FrameSectionID sec = m_profilerTimeline->frameBeginSection(name, &m_timeProvider);
  uint32_t                                  idx = nvutils::ProfilerTimeline::GpuTimeProvider::getTimerBaseIdx(sec);

  uint32_t      idxInPool;
  const GLuint* queryPool = getPoolBegin(m_frame, idx, idxInPool);

  glPushDebugGroup(GL_DEBUG_SOURCE_APPLICATION, 0, -1, name.c_str());
  glQueryCounter(queryPool[idxInPool], GL_TIMESTAMP);

  return sec;
}

void ProfilerGpuTimer::frameEndSection(nvutils::ProfilerTimeline::FrameSectionID sec)
{
  uint32_t idx = nvutils::ProfilerTimeline::GpuTimeProvider::getTimerBaseIdx(sec) + 1;

  uint32_t      idxInPool;
  const GLuint* queryPool = getPoolEnd(m_frame, idx, idxInPool);

  glQueryCounter(queryPool[idxInPool], GL_TIMESTAMP);
  glPopDebugGroup();

  m_profilerTimeline->frameEndSection(sec);
}

nvutils::ProfilerTimeline::AsyncSectionID ProfilerGpuTimer::asyncBeginSection(const std::string& name)
{
  nvutils::ProfilerTimeline::AsyncSectionID sec = m_profilerTimeline->asyncBeginSection(name, &m_timeProvider);
  uint32_t                                  idx = nvutils::ProfilerTimeline::GpuTimeProvider::getTimerBaseIdx(sec);

  uint32_t      idxInPool;
  const GLuint* queryPool;
  {
    std::lock_guard lock(m_asyncMutex);
    queryPool = getPoolBegin(m_async, idx, idxInPool);
  }

  glPushDebugGroup(GL_DEBUG_SOURCE_APPLICATION, 0, -1, name.c_str());
  glQueryCounter(queryPool[idxInPool], GL_TIMESTAMP);

  return sec;
}

void ProfilerGpuTimer::asyncEndSection(nvutils::ProfilerTimeline::AsyncSectionID sec)
{
  uint32_t idx = nvutils::ProfilerTimeline::GpuTimeProvider::getTimerBaseIdx(sec) + 1;

  uint32_t      idxInPool;
  const GLuint* queryPool;
  {
    std::lock_guard lock(m_asyncMutex);
    queryPool = getPoolEnd(m_async, idx, idxInPool);
  }

  glQueryCounter(queryPool[idxInPool], GL_TIMESTAMP);
  glPopDebugGroup();

  m_profilerTimeline->asyncEndSection(sec);
}

void ProfilerGpuTimer::frameAccumulationSplit()
{
  m_profilerTimeline->frameAccumulationSplit();
}


bool ProfilerGpuTimer::provideTime(const PoolContainer& container, uint32_t idxBegin, double& gpuTime) const
{
  uint32_t        idxInPool;
  const uint32_t* queryPool = getPoolEnd(container, idxBegin, idxInPool);

  GLint available = 0;
  glGetQueryObjectiv(queryPool[idxInPool + 1], GL_QUERY_RESULT_AVAILABLE, &available);

  if(available)
  {
    GLuint64 beginTime;
    GLuint64 endTime;
    glGetQueryObjectui64v(queryPool[idxInPool + 0], GL_QUERY_RESULT, &beginTime);
    glGetQueryObjectui64v(queryPool[idxInPool + 1], GL_QUERY_RESULT, &endTime);

    gpuTime = double(endTime - beginTime) / double(1000);

    return true;
  }
  else
  {
    return false;
  }
}

const uint32_t* ProfilerGpuTimer::getPoolBegin(PoolContainer& container, uint32_t idx, uint32_t& idxInPool)
{
  idxInPool = idx % PoolContainer::POOL_QUERY_COUNT;

  // early out
  if(idx <= container.queryPoolSize)
  {
    return container.queryPools[idx / PoolContainer::POOL_QUERY_COUNT].data();
  }

  resizePool(container, idx);
  return container.queryPools[idx / PoolContainer::POOL_QUERY_COUNT].data();
}

const uint32_t* ProfilerGpuTimer::getPoolEnd(const PoolContainer& container, uint32_t idx, uint32_t& idxInPool) const
{
  idxInPool = idx % PoolContainer::POOL_QUERY_COUNT;

  return container.queryPools[idx / PoolContainer::POOL_QUERY_COUNT].data();
}

void ProfilerGpuTimer::resizePool(PoolContainer& container, uint32_t requiredSize)
{

  uint32_t oldCount = container.queryPoolSize / PoolContainer::POOL_QUERY_COUNT;
  uint32_t newCount = (requiredSize + PoolContainer::POOL_QUERY_COUNT - 1) / PoolContainer::POOL_QUERY_COUNT;

  container.queryPools.resize(newCount);

  for(uint32_t i = oldCount; i < newCount; i++)
  {
    container.queryPools[i].resize(PoolContainer::POOL_QUERY_COUNT);
    glGenQueries(PoolContainer::POOL_QUERY_COUNT, container.queryPools[i].data());
  }

  container.queryPoolSize = newCount * PoolContainer::POOL_QUERY_COUNT;
}
}  // namespace nvgl

//--------------------------------------------------------------------------------------------------
// Usage example
//--------------------------------------------------------------------------------------------------
[[maybe_unused]] static void usage_ProfilerGpuTimer()
{
  nvutils::ProfilerManager profilerManager;

  // in a typical single-threaded main loop we have one timeline on which we submit commands.
  nvutils::ProfilerTimeline* profilerTimeline = profilerManager.createTimeline({"primary"});

  // the above timeline will represent a VkQueue
  uint32_t queueFamilyIndex = 0;

  nvgl::ProfilerGpuTimer gpuTimer;
  gpuTimer.init(profilerTimeline);

  // or re-occurring per-frame events
  /* while(!glfwWindowShouldClose()) */
  {
    profilerTimeline->frameAdvance();

    {
      // per-frame sections must be within frameBegin/frameEnd
      // and are NOT thread-safe with respect to the timeline.

      auto profiledSection = gpuTimer.frameSection("processing");

      // do some work
    }
  }
  profilerManager.destroyTimeline(profilerTimeline);
}
