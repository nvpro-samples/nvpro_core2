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

#pragma once

// basic_types.hpp sets NOMINMAX before Windows.h; it has to come before
// <nvutils/profiler.hpp>, which uses std::numeric_limits<>::min/max().
#include "basic_types.hpp"

#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

#include <nvutils/profiler.hpp>

namespace nvdx12 {

/*-------------------------------------------------------------------------------------------------
# class nvdx12::ProfilerGpuTimer

> GPU timestamps for nvutils::ProfilerTimeline sections, the D3D12
> counterpart of nvvk::ProfilerGpuTimer.

Implements nvutils::ProfilerTimeline::GpuTimeProvider with TIMESTAMP query
heaps, each paired with a readback buffer the pair of timestamps of a
section is resolved into. Frame and update sections belong to the frame loop
of the timeline and are not thread-safe; async sections are. With `useLabels`
every section is also a PIX event (debug_util.hpp). The timer belongs to one
queue: the timestamp frequency is read from the queue given to `init`, and
the command lists of the sections are expected to execute on it. The queue
is a direct or compute queue: the query heaps are TIMESTAMP heaps, which
copy queues cannot write (they need COPY_QUEUE_TIMESTAMP heaps). A section
recorded on a copy list writes no timestamps and reports a GPU time of zero.

Result validity. A section reports a GPU time only once the resolve of the
use it was last recorded for has completed on the GPU, never a value left
in the readback buffer by an older use of its queries:
- Every section end tags its resolve with a generation number that the GPU
  writes next to the timestamps (WriteBufferImmediate, MARKER_OUT: after the
  resolve completed). The timer reads the timestamps only when the tag
  matches the generation it expects, the counterpart of the availability
  status of vkGetQueryPoolResults (nvvk resets the queries when a section
  begins, so they read as unavailable until the GPU wrote them).
- A section that begins is unavailable until its end resolved.
- Frame sections are read once: the timeline reads the queries of a frame
  `frameDelay` frames later, after which they no longer report. A section
  skipped in a later frame using the same queries (same section index and
  sub-frame) therefore reports nothing instead of the older frame's time, and
  a frame whose resolve did not complete when the timeline read it is
  dropped, not reported later. Update sections are read while the timeline
  keeps them pending: a read before the resolve completed reports nothing
  and the timeline asks again in a later frame. Async sections report their
  latest completed use on every read, until the next use begins.
- On command list types without WriteBufferImmediate support
  (D3D12_FEATURE_D3D12_OPTIONS3) there is no tag: frame sections are still
  read once, and a section reports when its end timestamp is greater than
  its begin one, which cannot tell a resolve still in flight from the
  previous use of the queries.

Differences with nvvk that follow from D3D12:
- `init` takes the command queue in place of the physical device and queue
  family index: D3D12 reads the timestamp frequency from the queue
  (QueueTimeline::getQueue or QueueInfo::queue), where Vulkan reads the
  timestamp period and valid bits from the physical device.
- Timestamps are resolved into readback buffers paired with the query
  heaps, as D3D12 has no vkGetQueryPoolResults; availability is the
  generation tag above.

```cpp
nvdx12::ProfilerGpuTimer gpuTimer;
gpuTimer.init(profilerTimeline, device, queue, true);
...
{
  auto section = gpuTimer.cmdFrameSection(cmd, "Render");
  // record commands
}
```

See usage_ProfilerGpuTimer() in profiler_dx12.cpp.
-------------------------------------------------------------------------------------------------*/

class ProfilerGpuTimer
{
public:
  ProfilerGpuTimer() = default;
  ~ProfilerGpuTimer();
  // The time provider's callbacks hold `this`.
  ProfilerGpuTimer(const ProfilerGpuTimer&)            = delete;
  ProfilerGpuTimer& operator=(const ProfilerGpuTimer&) = delete;

  // `profilerTimeline` pointer is copied and must be kept alive during this class lifetime.
  // `commandQueue` is only used here to obtain the timestamp frequency.
  void init(nvutils::ProfilerTimeline* profilerTimeline, ID3D12Device* device, ID3D12CommandQueue* commandQueue, bool useLabels);
  void deinit();

  nvutils::ProfilerTimeline*       getProfilerTimeline() { return m_profilerTimeline; }
  const nvutils::ProfilerTimeline* getProfilerTimeline() const { return m_profilerTimeline; }

  // not thread-safe
  nvutils::ProfilerTimeline::FrameSectionID cmdFrameBeginSection(ID3D12GraphicsCommandList* cmd, const std::string& name);
  void cmdFrameEndSection(ID3D12GraphicsCommandList* cmd, nvutils::ProfilerTimeline::FrameSectionID slot);

  // thread-safe
  nvutils::ProfilerTimeline::AsyncSectionID cmdAsyncBeginSection(ID3D12GraphicsCommandList* cmd, const std::string& name);
  void cmdAsyncEndSection(ID3D12GraphicsCommandList* cmd, nvutils::ProfilerTimeline::AsyncSectionID slot);

  // not thread-safe
  nvutils::ProfilerTimeline::UpdateSectionID cmdUpdateBeginSection(ID3D12GraphicsCommandList* cmd, const std::string& name);
  void cmdUpdateEndSection(ID3D12GraphicsCommandList* cmd, nvutils::ProfilerTimeline::UpdateSectionID slot);

  //////////////////////////////////////////////////////////////////////////

  // utility class to call begin/end within local scope
  class FrameSection
  {
  public:
    FrameSection(ProfilerGpuTimer& profilerGpuTimer, nvutils::ProfilerTimeline::FrameSectionID id, ID3D12GraphicsCommandList* cmd)
        : m_profilerGpuTimer(profilerGpuTimer)
        , m_cmd(cmd)
        , m_id(id) {};
    ~FrameSection() { m_profilerGpuTimer.cmdFrameEndSection(m_cmd, m_id); }

  private:
    ProfilerGpuTimer&                         m_profilerGpuTimer;
    ID3D12GraphicsCommandList*                m_cmd;
    nvutils::ProfilerTimeline::FrameSectionID m_id;
  };

  // frame section must be within beginFrame/endFrame
  // not thread-safe
  FrameSection cmdFrameSection(ID3D12GraphicsCommandList* cmd, const std::string& name)
  {
    return FrameSection(*this, cmdFrameBeginSection(cmd, name), cmd);
  }

  // utility class to call begin/end within local scope
  class AsyncSection
  {
  public:
    AsyncSection(ProfilerGpuTimer& profilerGpuTimer, nvutils::ProfilerTimeline::AsyncSectionID id, ID3D12GraphicsCommandList* cmd)
        : m_profilerGpuTimer(profilerGpuTimer)
        , m_cmd(cmd)
        , m_id(id) {};
    ~AsyncSection() { m_profilerGpuTimer.cmdAsyncEndSection(m_cmd, m_id); }

  private:
    ProfilerGpuTimer&                         m_profilerGpuTimer;
    ID3D12GraphicsCommandList*                m_cmd;
    nvutils::ProfilerTimeline::AsyncSectionID m_id;
  };

  // thread-safe
  AsyncSection cmdAsyncSection(ID3D12GraphicsCommandList* cmd, const std::string& name)
  {
    return AsyncSection(*this, cmdAsyncBeginSection(cmd, name), cmd);
  }

  // utility class to call begin/end within local scope
  class UpdateSection
  {
  public:
    UpdateSection(ProfilerGpuTimer& profilerGpuTimer, nvutils::ProfilerTimeline::UpdateSectionID id, ID3D12GraphicsCommandList* cmd)
        : m_profilerGpuTimer(profilerGpuTimer)
        , m_cmd(cmd)
        , m_id(id) {};
    ~UpdateSection() { m_profilerGpuTimer.cmdUpdateEndSection(m_cmd, m_id); }

  private:
    ProfilerGpuTimer&                          m_profilerGpuTimer;
    ID3D12GraphicsCommandList*                 m_cmd;
    nvutils::ProfilerTimeline::UpdateSectionID m_id;
  };

  // update section must be within beginFrame/endFrame
  // not thread-safe
  UpdateSection cmdUpdateSection(ID3D12GraphicsCommandList* cmd, const std::string& name)
  {
    return UpdateSection(*this, cmdUpdateBeginSection(cmd, name), cmd);
  }

  //////////////////////////////////////////////////////////////////////////

protected:
  struct PoolContainer
  {
    // A virtual array of queries distributed over N query heaps of
    // POOL_QUERY_COUNT timestamps, each paired with a readback buffer of
    // POOL_QUERY_COUNT timestamps (uint64_t) followed by the generation
    // tags of the POOL_QUERY_COUNT / 2 sections (uint32_t).
    static constexpr uint32_t POOL_QUERY_COUNT = 1024;
    static constexpr uint64_t TAG_OFFSET       = POOL_QUERY_COUNT * sizeof(uint64_t);
    static constexpr uint64_t READBACK_SIZE    = TAG_OFFSET + POOL_QUERY_COUNT / 2 * sizeof(uint32_t);

    std::vector<ComPtr<ID3D12QueryHeap>> queryHeaps;
    std::vector<ComPtr<ID3D12Resource>>  readbackBuffers;
    uint32_t                             queryPoolSize = 0;

    // Per section (pair of queries): the generation its last resolve tags,
    // kNotPending when there is nothing to report.
    std::vector<uint32_t> expectedTags;
    uint32_t              generation = 0;  // last generation handed out
  };

  static constexpr uint32_t kNotPending   = 0;                 // no use to report
  static constexpr uint32_t kUntagged     = ~uint32_t(0);      // resolved without a tag (no WriteBufferImmediate)
  static constexpr uint32_t kNoTimestamps = ~uint32_t(0) - 1;  // recorded on a copy list: a GPU time of zero

  // Reads the GPU time of the section whose queries start at `idx`; with
  // `consume`, the section no longer reports afterwards (frame sections).
  bool provideTime(PoolContainer& container, uint32_t idx, double& time, bool consume);
  // Records the end timestamp, the resolve and its tag (m_asyncMutex held for m_async).
  void cmdEnd(PoolContainer& container, ID3D12GraphicsCommandList* cmd, uint32_t idxEnd);

  ID3D12QueryHeap* getPool(PoolContainer& container, uint32_t idx, uint32_t& idxInPool, ID3D12Resource*& readbackBuffer);
  ID3D12QueryHeap* getPool(const PoolContainer& container, uint32_t idx, uint32_t& idxInPool, ID3D12Resource*& readbackBuffer) const;
  void resizePool(PoolContainer& container, uint32_t requiredSize);

  nvutils::ProfilerTimeline*                 m_profilerTimeline{};
  nvutils::ProfilerTimeline::GpuTimeProvider m_timeProvider;

  ID3D12Device* m_device{nullptr};
  bool          m_useLabels{false};
  uint64_t      m_frequency{1};  // ticks per second (ID3D12CommandQueue::GetTimestampFrequency)
  // D3D12_COMMAND_LIST_SUPPORT_FLAGS of WriteBufferImmediate (bit 1 << list type).
  uint32_t m_writeImmediateSupport{0};

  PoolContainer      m_frame;
  PoolContainer      m_async;
  mutable std::mutex m_asyncMutex;
  PoolContainer      m_update;
};

}  // namespace nvdx12
