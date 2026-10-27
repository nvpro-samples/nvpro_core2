/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "profiler_dx12.hpp"

#include <atomic>
#include <cassert>
#include <cstdint>

#include <d3dx12.h>

#include "check_error.hpp"
#include "debug_util.hpp"

namespace nvdx12 {

// TIMESTAMP query heaps cannot be written on copy lists (those need
// COPY_QUEUE_TIMESTAMP heaps): sections recorded there report zero.
static bool canWriteTimestamps(ID3D12GraphicsCommandList* cmd)
{
  return cmd->GetType() != D3D12_COMMAND_LIST_TYPE_COPY;
}

ProfilerGpuTimer::~ProfilerGpuTimer()
{
  assert(m_device == nullptr && "Missing deinit()");
}

void ProfilerGpuTimer::init(nvutils::ProfilerTimeline* profilerTimeline, ID3D12Device* device, ID3D12CommandQueue* commandQueue, bool useLabels)
{
  assert(m_device == nullptr);

  m_profilerTimeline = profilerTimeline;
  m_device           = device;

  m_frequency = 1;
  if(commandQueue != nullptr)
  {
    assert(commandQueue->GetDesc().Type != D3D12_COMMAND_LIST_TYPE_COPY && "TIMESTAMP query heaps: direct or compute queue");
    NVDX12_CHECK(commandQueue->GetTimestampFrequency(&m_frequency));
  }

  // The resolve tags need WriteBufferImmediate on the list types the sections are recorded on.
  m_writeImmediateSupport = 0;
  D3D12_FEATURE_DATA_D3D12_OPTIONS3 options3{};
  if(SUCCEEDED(device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS3, &options3, sizeof(options3))))
  {
    m_writeImmediateSupport = static_cast<uint32_t>(options3.WriteBufferImmediateSupportFlags);
  }

  m_timeProvider.apiName       = "DX12";
  m_timeProvider.frameFunction = [&](nvutils::ProfilerTimeline::FrameSectionID sec, double& gpuTime) {
    uint32_t idx = nvutils::ProfilerTimeline::GpuTimeProvider::getTimerBaseIdx(sec);
    return provideTime(m_frame, idx, gpuTime, true);
  };
  m_timeProvider.asyncFunction = [&](nvutils::ProfilerTimeline::AsyncSectionID sec, double& gpuTime) {
    std::lock_guard lock(m_asyncMutex);
    uint32_t        idx = nvutils::ProfilerTimeline::GpuTimeProvider::getTimerBaseIdx(sec);
    return provideTime(m_async, idx, gpuTime, false);
  };
  // Not consumed: the timeline asks again while the section stays pending, and stops once it reported.
  m_timeProvider.updateFunction = [&](nvutils::ProfilerTimeline::UpdateSectionID sec, double& gpuTime) {
    uint32_t idx = nvutils::ProfilerTimeline::GpuTimeProvider::getTimerBaseIdx(sec);
    return provideTime(m_update, idx, gpuTime, false);
  };

  resizePool(m_frame, PoolContainer::POOL_QUERY_COUNT);
  resizePool(m_async, PoolContainer::POOL_QUERY_COUNT);
  resizePool(m_update, PoolContainer::POOL_QUERY_COUNT);

  m_useLabels = useLabels;
}

void ProfilerGpuTimer::deinit()
{
  if(!m_device)
    return;

  m_frame  = {};
  m_async  = {};
  m_update = {};

  m_device           = nullptr;
  m_profilerTimeline = nullptr;
}

nvutils::ProfilerTimeline::FrameSectionID ProfilerGpuTimer::cmdFrameBeginSection(ID3D12GraphicsCommandList* cmd,
                                                                                 const std::string&         name)
{
  nvutils::ProfilerTimeline::FrameSectionID sec = m_profilerTimeline->frameBeginSection(name, &m_timeProvider);
  uint32_t                                  idx = nvutils::ProfilerTimeline::GpuTimeProvider::getTimerBaseIdx(sec);

  uint32_t         idxInPool;
  ID3D12Resource*  readback  = nullptr;
  ID3D12QueryHeap* queryHeap = getPool(m_frame, idx, idxInPool, readback);
  // Nothing to report until this use resolved (nvvk resets the queries here).
  m_frame.expectedTags[idx / 2] = kNotPending;

  if(m_useLabels)
  {
    cmdBeginLabel(cmd, name.c_str());
  }

  m_profilerTimeline->frameResetCpuBegin(sec);

  // D3D12 has no BeginQuery for timestamps; EndQuery writes the timestamp at the index.
  if(canWriteTimestamps(cmd))
  {
    cmd->EndQuery(queryHeap, D3D12_QUERY_TYPE_TIMESTAMP, idxInPool);
  }

  return sec;
}

void ProfilerGpuTimer::cmdFrameEndSection(ID3D12GraphicsCommandList* cmd, nvutils::ProfilerTimeline::FrameSectionID sec)
{
  cmdEnd(m_frame, cmd, nvutils::ProfilerTimeline::GpuTimeProvider::getTimerBaseIdx(sec) + 1);

  if(m_useLabels)
  {
    cmdEndLabel(cmd);
  }
  m_profilerTimeline->frameEndSection(sec);
}

nvutils::ProfilerTimeline::AsyncSectionID ProfilerGpuTimer::cmdAsyncBeginSection(ID3D12GraphicsCommandList* cmd,
                                                                                 const std::string&         name)
{
  nvutils::ProfilerTimeline::AsyncSectionID sec = m_profilerTimeline->asyncBeginSection(name, &m_timeProvider);
  uint32_t                                  idx = nvutils::ProfilerTimeline::GpuTimeProvider::getTimerBaseIdx(sec);

  uint32_t         idxInPool;
  ID3D12Resource*  readback  = nullptr;
  ID3D12QueryHeap* queryHeap = nullptr;
  {
    std::lock_guard lock(m_asyncMutex);
    queryHeap                     = getPool(m_async, idx, idxInPool, readback);
    m_async.expectedTags[idx / 2] = kNotPending;
  }

  if(m_useLabels)
  {
    cmdBeginLabel(cmd, name.c_str());
  }

  m_profilerTimeline->asyncResetCpuBegin(sec);

  if(canWriteTimestamps(cmd))
  {
    cmd->EndQuery(queryHeap, D3D12_QUERY_TYPE_TIMESTAMP, idxInPool);
  }

  return sec;
}

void ProfilerGpuTimer::cmdAsyncEndSection(ID3D12GraphicsCommandList* cmd, nvutils::ProfilerTimeline::AsyncSectionID sec)
{
  {
    std::lock_guard lock(m_asyncMutex);
    cmdEnd(m_async, cmd, nvutils::ProfilerTimeline::GpuTimeProvider::getTimerBaseIdx(sec) + 1);
  }

  if(m_useLabels)
  {
    cmdEndLabel(cmd);
  }
  m_profilerTimeline->asyncEndSection(sec);
}

nvutils::ProfilerTimeline::UpdateSectionID ProfilerGpuTimer::cmdUpdateBeginSection(ID3D12GraphicsCommandList* cmd,
                                                                                   const std::string&         name)
{
  nvutils::ProfilerTimeline::UpdateSectionID sec = m_profilerTimeline->updateBeginSection(name, &m_timeProvider);
  uint32_t                                   idx = nvutils::ProfilerTimeline::GpuTimeProvider::getTimerBaseIdx(sec);

  uint32_t         idxInPool;
  ID3D12Resource*  readback      = nullptr;
  ID3D12QueryHeap* queryHeap     = getPool(m_update, idx, idxInPool, readback);
  m_update.expectedTags[idx / 2] = kNotPending;

  if(m_useLabels)
  {
    cmdBeginLabel(cmd, name.c_str());
  }

  m_profilerTimeline->updateResetCpuBegin(sec);

  if(canWriteTimestamps(cmd))
  {
    cmd->EndQuery(queryHeap, D3D12_QUERY_TYPE_TIMESTAMP, idxInPool);
  }

  return sec;
}

void ProfilerGpuTimer::cmdUpdateEndSection(ID3D12GraphicsCommandList* cmd, nvutils::ProfilerTimeline::UpdateSectionID sec)
{
  cmdEnd(m_update, cmd, nvutils::ProfilerTimeline::GpuTimeProvider::getTimerBaseIdx(sec) + 1);

  if(m_useLabels)
  {
    cmdEndLabel(cmd);
  }
  m_profilerTimeline->updateEndSection(sec);
}

void ProfilerGpuTimer::cmdEnd(PoolContainer& container, ID3D12GraphicsCommandList* cmd, uint32_t idxEnd)
{
  if(!canWriteTimestamps(cmd))
  {
    container.expectedTags[idxEnd / 2] = kNoTimestamps;
    return;
  }

  uint32_t         idxEndInPool;
  ID3D12Resource*  readback  = nullptr;
  ID3D12QueryHeap* queryHeap = getPool(container, idxEnd, idxEndInPool, readback);

  const uint32_t idxBeginInPool = idxEndInPool - 1;

  cmd->EndQuery(queryHeap, D3D12_QUERY_TYPE_TIMESTAMP, idxEndInPool);

  // Resolve the begin/end pair into the matching readback buffer. Begin and
  // end always live in the same pool because POOL_QUERY_COUNT is even.
  cmd->ResolveQueryData(queryHeap, D3D12_QUERY_TYPE_TIMESTAMP, idxBeginInPool, 2, readback, idxBeginInPool * sizeof(uint64_t));

  // Tag the resolve: the GPU writes the generation once the resolve (and all
  // the work before it) completed, so a matching tag means the timestamps
  // are those of this use. The readback buffer stays in COPY_DEST, the state
  // WriteBufferImmediate writes in.
  uint32_t                           tag = kUntagged;
  ComPtr<ID3D12GraphicsCommandList2> cmd2;
  if((m_writeImmediateSupport & (1u << cmd->GetType())) != 0 && SUCCEEDED(cmd->QueryInterface(IID_PPV_ARGS(&cmd2))))
  {
    // Generations skip 0 and the reserved tags.
    container.generation = (container.generation + 1 >= kNoTimestamps) ? 1 : container.generation + 1;
    tag                  = container.generation;
    const D3D12_WRITEBUFFERIMMEDIATE_PARAMETER param{
        readback->GetGPUVirtualAddress() + PoolContainer::TAG_OFFSET + (idxBeginInPool / 2) * sizeof(uint32_t), tag};
    const D3D12_WRITEBUFFERIMMEDIATE_MODE mode = D3D12_WRITEBUFFERIMMEDIATE_MODE_MARKER_OUT;
    cmd2->WriteBufferImmediate(1, &param, &mode);
  }
  container.expectedTags[idxEnd / 2] = tag;
}

bool ProfilerGpuTimer::provideTime(PoolContainer& container, uint32_t idxBegin, double& gpuTime, bool consume)
{
  if(idxBegin >= container.queryPoolSize)
  {
    return false;
  }
  const uint32_t expected = container.expectedTags[idxBegin / 2];
  if(expected == kNotPending)
  {
    return false;  // never recorded, begun but not ended, or already read
  }
  if(consume)
  {
    container.expectedTags[idxBegin / 2] = kNotPending;
  }
  if(expected == kNoTimestamps)
  {
    gpuTime = 0.0;
    return true;
  }

  uint32_t        idxInPool;
  ID3D12Resource* readback = nullptr;
  // Only the readback buffer and the in-pool offset matter here.
  (void)getPool(static_cast<const PoolContainer&>(container), idxBegin, idxInPool, readback);

  // Map the whole buffer, read range limited to the two timestamps and the
  // tag: the read range lets the runtime invalidate the matching CPU cache
  // lines, so the resolved values are seen.
  const SIZE_T      byteBegin = static_cast<SIZE_T>(idxInPool) * sizeof(uint64_t);
  const SIZE_T      tagByte   = static_cast<SIZE_T>(PoolContainer::TAG_OFFSET + (idxInPool / 2) * sizeof(uint32_t));
  const D3D12_RANGE readRange{byteBegin, tagByte + sizeof(uint32_t)};

  void*         mappedPtr = nullptr;
  const HRESULT hr        = readback->Map(0, &readRange, &mappedPtr);
  if(FAILED(hr) || mappedPtr == nullptr)
  {
    return false;
  }

  const uint8_t* bytes = static_cast<const uint8_t*>(mappedPtr);
  // The tag first: written after the timestamps, it validates them.
  const uint32_t tag = *reinterpret_cast<const volatile uint32_t*>(bytes + tagByte);
  std::atomic_thread_fence(std::memory_order_acquire);
  const volatile uint64_t* timestamps = reinterpret_cast<const volatile uint64_t*>(bytes + byteBegin);
  const uint64_t           t0         = timestamps[0];
  const uint64_t           t1         = timestamps[1];

  // Empty written range: the CPU did not write anything.
  const D3D12_RANGE writeRange{0, 0};
  readback->Unmap(0, &writeRange);

  if(m_frequency == 0)
  {
    return false;
  }
  if(expected == kUntagged)
  {
    // No tag on this list type: the `t1 > t0` check filters the zeros of a
    // section never resolved, not the values of an older use.
    if(t1 <= t0)
    {
      return false;
    }
  }
  else if(tag != expected || t1 < t0)
  {
    return false;  // resolve not completed yet (an older use's values, if any)
  }

  gpuTime = static_cast<double>(t1 - t0) * 1'000'000.0 / static_cast<double>(m_frequency);
  return true;
}

ID3D12QueryHeap* ProfilerGpuTimer::getPool(PoolContainer& container, uint32_t idx, uint32_t& idxInPool, ID3D12Resource*& readbackBuffer)
{
  idxInPool = idx % PoolContainer::POOL_QUERY_COUNT;

  if(idx >= container.queryPoolSize)
  {
    resizePool(container, idx + 1);
  }

  const uint32_t poolIdx = idx / PoolContainer::POOL_QUERY_COUNT;
  readbackBuffer         = container.readbackBuffers[poolIdx].Get();
  return container.queryHeaps[poolIdx].Get();
}

ID3D12QueryHeap* ProfilerGpuTimer::getPool(const PoolContainer& container, uint32_t idx, uint32_t& idxInPool, ID3D12Resource*& readbackBuffer) const
{
  assert(idx < container.queryPoolSize);

  idxInPool = idx % PoolContainer::POOL_QUERY_COUNT;

  const uint32_t poolIdx = idx / PoolContainer::POOL_QUERY_COUNT;
  readbackBuffer         = container.readbackBuffers[poolIdx].Get();
  return container.queryHeaps[poolIdx].Get();
}

void ProfilerGpuTimer::resizePool(PoolContainer& container, uint32_t requiredSize)
{
  const uint32_t oldCount = container.queryPoolSize / PoolContainer::POOL_QUERY_COUNT;
  const uint32_t newCount = (requiredSize + PoolContainer::POOL_QUERY_COUNT - 1) / PoolContainer::POOL_QUERY_COUNT;

  container.queryHeaps.resize(newCount);
  container.readbackBuffers.resize(newCount);
  container.expectedTags.resize(newCount * PoolContainer::POOL_QUERY_COUNT / 2, kNotPending);

  for(uint32_t i = oldCount; i < newCount; i++)
  {
    const D3D12_QUERY_HEAP_DESC heapDesc{
        .Type     = D3D12_QUERY_HEAP_TYPE_TIMESTAMP,
        .Count    = PoolContainer::POOL_QUERY_COUNT,
        .NodeMask = 0,
    };
    NVDX12_CHECK(m_device->CreateQueryHeap(&heapDesc, IID_PPV_ARGS(&container.queryHeaps[i])));
    NVDX12_DBG_NAME(container.queryHeaps[i]);

    const CD3DX12_HEAP_PROPERTIES heapProperties(D3D12_HEAP_TYPE_READBACK);
    const CD3DX12_RESOURCE_DESC   bufferDesc = CD3DX12_RESOURCE_DESC::Buffer(PoolContainer::READBACK_SIZE);
    NVDX12_CHECK(m_device->CreateCommittedResource(&heapProperties, D3D12_HEAP_FLAG_NONE, &bufferDesc, D3D12_RESOURCE_STATE_COPY_DEST,
                                                   nullptr, IID_PPV_ARGS(&container.readbackBuffers[i])));
    NVDX12_DBG_NAME(container.readbackBuffers[i]);
  }

  container.queryPoolSize = newCount * PoolContainer::POOL_QUERY_COUNT;
}
}  // namespace nvdx12

//--------------------------------------------------------------------------------------------------
// Usage example
//--------------------------------------------------------------------------------------------------
[[maybe_unused]] static void usage_ProfilerGpuTimer()
{
  ID3D12Device*       device       = nullptr;  // EX: from nvdx12::Context::getDevice()
  ID3D12CommandQueue* commandQueue = nullptr;  // EX: the queue the profiled lists execute on

  nvutils::ProfilerManager profilerManager;

  // in a typical single-threaded main loop we have one timeline on which we submit commands.
  nvutils::ProfilerTimeline* profilerTimeline = profilerManager.createTimeline({"primary"});

  nvdx12::ProfilerGpuTimer gpuTimer;
  gpuTimer.init(profilerTimeline, device, commandQueue, true);

  // per-frame events
  /* while(!glfwWindowShouldClose()) */
  {
    profilerTimeline->frameAdvance();

    ID3D12GraphicsCommandList* cmd{};  // EX acquire somehow per-frame

    {
      // per-frame sections must be within frameBegin/frameEnd
      // and are NOT thread-safe with respect to the timeline.

      auto profiledSection = gpuTimer.cmdFrameSection(cmd, "processing");

      // do some work
    }

    {
      // async sections are thread-safe and independent of the frame loop,
      // for work recorded on other threads (executed on the timer's queue).
      auto asyncSection = gpuTimer.cmdAsyncSection(cmd, "upload");
    }

    {
      // update sections measure work that does not run every frame (uploads,
      // acceleration structure builds); an idle one reports 0, or its last
      // measurement without ProfilerTimeline::CreateInfo::updateResetWhenIdle.
      auto updateSection = gpuTimer.cmdUpdateSection(cmd, "rebuild TLAS");
    }

    // submit command list

    // get snapshot for visualization, printing etc.
    std::vector<nvutils::ProfilerTimeline::Snapshot> frameSnapshots;
    std::vector<nvutils::ProfilerTimeline::Snapshot> asyncSnapshots;
    std::vector<nvutils::ProfilerTimeline::Snapshot> updateSnapshots;

    profilerManager.getSnapshots(frameSnapshots, asyncSnapshots, updateSnapshots);
  }

  gpuTimer.deinit();
  profilerManager.destroyTimeline(profilerTimeline);
}
