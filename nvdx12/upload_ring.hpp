/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cassert>
#include <cstdint>
#include <deque>
#include <atomic>
#include <mutex>

#include "basic_types.hpp"
#include "pipeline.hpp"  // BindPoint, kPushConstantsMaxBytes
#include "queue.hpp"

namespace nvdx12 {

/*-------------------------------------------------------------------------------------------------
# class nvdx12::UploadRing

> Persistently mapped upload memory recycled by fence value, for push data
> and inline buffer updates.

No nvvk counterpart: it backs push data, which Vulkan has natively
(vkCmdPushDataEXT), and the inline updates of cmdUpdateBuffer, which
vkCmdUpdateBuffer records without user memory. Its recycling resembles
nvvk::BufferCircularAllocator (a ring retired by semaphore values), whose
names it follows where they apply (DEFAULT_ALIGNMENT).

Push data is the second tier of pushing: constants within
kPushConstantsMaxBytes go through cmdPushConstants (pipeline.hpp) and need
no ring; a larger block is copied here and bound as the root SRV, the
counterpart of vkCmdPushDataEXT.

The ring serves one QueueTimeline, given to `init`: every signal of the
timeline marks the allocations made so far as in flight until that fence
value, and completed values hand their memory back. `allocate` returns a
span valid until the next signal has completed, so every allocation must
belong to a command list submitted with (or before) that next signal. A
full ring waits for the oldest signaled allocations; what was allocated
since the last signal is never recycled, so the ring must hold everything
the command lists recorded between two signals allocate (an allocation that
does not fit returns an empty span).

A command list may be recorded while other lists are submitted on the same
queue, such as a frame list while temporary lists are submitted and waited
for. `beginDeferral(list)` covers its recording: the signals before the
submission that executes `list` leave the allocations made since
`beginDeferral` in flight, and that submission marks them and ends the
deferral under the timeline's lock, so no other submission comes in
between. The ring must hold everything allocated meanwhile;
`cancelDeferral` ends a deferral whose list is not submitted.

`cmdPushData` copies the data into the ring and binds it as the root SRV
of the bindless root signature in pipeline.hpp, for the compute, graphics
or both bind points.

See usage_UploadRing() in upload_ring.cpp.
-------------------------------------------------------------------------------------------------*/

// Sub-allocation from the ring: a persistently mapped, GPU-visible span.
struct UploadSpan
{
  uint8_t*                  cpu{nullptr};
  D3D12_GPU_VIRTUAL_ADDRESS gpu{0};
  ID3D12Resource*           resource{nullptr};
  uint64_t                  offset{0};  // byte offset inside `resource`
  uint64_t                  size{0};

  explicit operator bool() const { return cpu != nullptr; }
};

class UploadRing
{
public:
  static constexpr uint64_t MIN_SIZE = 1ull << 20;  // init raises `bytes` to this, then to a multiple of 64 KiB
  static constexpr uint64_t DEFAULT_ALIGNMENT = 256;

  UploadRing()                             = default;
  UploadRing(const UploadRing&)            = delete;
  UploadRing& operator=(const UploadRing&) = delete;
  ~UploadRing() { assert(!m_resource && "Missing deinit()?"); }

  // `bytes` of upload memory (at least MIN_SIZE) recycled by the signals of
  // `queue`, which must outlive the ring; the device is the queue's.
  HRESULT init(QueueTimeline& queue, uint64_t bytes);
  // The GPU must have finished every command list that used the ring.
  void deinit();

  // Thread-safe; `alignment` is a power of two. Empty span (cpu == nullptr)
  // when the request does not fit beside the unsignaled allocations.
  UploadSpan allocate(uint64_t size, uint64_t alignment = DEFAULT_ALIGNMENT);

  // Frees the spans whose fence value has completed. Every signal and every
  // full ring do this already; a call frees memory earlier.
  void retire();

  // The allocations from here on belong to `list`, submitted later to the
  // ring's queue: the signals before that submission do not mark them, the
  // submission does and ends the deferral. Not nested.
  void beginDeferral(ID3D12CommandList* list);
  // Ends a deferral whose list is not submitted; the next signal marks its allocations.
  void cancelDeferral();

  using BindPoint = nvdx12::BindPoint;  // which root signature's argument a push binds

  // Copies `size` bytes into the ring and binds them to the root SRV
  // `rootParameterIndex` of the chosen bind point(s); the arguments of
  // cmdPushConstants (pipeline.hpp). Returns false, binding nothing, when the
  // ring has no room; the previous push data then stays bound.
  bool cmdPushData(ID3D12GraphicsCommandList* cmd, const void* data, uint32_t size, BindPoint bindPoint, uint32_t rootParameterIndex = kRootParamPushData);

  uint64_t size() const { return m_size; }
  // The timeline whose signals recycle the ring.
  QueueTimeline* getQueueTimeline() const { return m_queue; }
  // Bytes allocated and not yet retired.
  uint64_t usedBytes() const
  {
    std::lock_guard lock(m_mutex);
    return m_head - m_tail;
  }

private:
  // Everything allocated so far, or before beginDeferral while deferring,
  // completes with `fenceValue` (signal listener).
  void markSubmitted(uint64_t fenceValue);
  // The submission of the deferred list completes everything allocated so
  // far and ends the deferral (submit listener, after the signal listener).
  void markListSubmitted(std::span<ID3D12CommandList* const> lists, uint64_t fenceValue);
  // Frees the spans of the submissions completed by `completedValue`; the
  // caller holds m_mutex. False when nothing was freed.
  bool retireLocked(uint64_t completedValue);

  struct Retire
  {
    uint64_t fenceValue;
    uint64_t head;  // the position marked at markSubmitted
  };

  ComPtr<ID3D12Resource>        m_resource;
  uint8_t*                      m_mapping{nullptr};
  uint64_t                      m_size{0};
  uint64_t                      m_head{0};                // next free byte (monotonic, wraps modulo size)
  uint64_t                      m_tail{0};                // oldest byte still in flight
  uint64_t                      m_deferredHead{0};        // m_head at beginDeferral
  ID3D12CommandList*            m_deferredList{nullptr};  // its submission ends the deferral; null when not deferring
  std::deque<Retire>            m_retires;                // submissions in fence order
  std::atomic<bool>             m_fullLogged{false};      // read and set outside m_mutex
  QueueTimeline*                m_queue{nullptr};         // the timeline served; not owned
  QueueTimeline::ListenerHandle m_signalListener{0};
  QueueTimeline::ListenerHandle m_submitListener{0};
  mutable std::mutex            m_mutex;
};

}  // namespace nvdx12
