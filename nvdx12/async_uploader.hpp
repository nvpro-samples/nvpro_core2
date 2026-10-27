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

#include <atomic>
#include <functional>
#include <mutex>
#include <string>
#include <vector>

#include <nvutils/id_pool.hpp>

#include "buffer_suballocator.hpp"
#include "command_pools.hpp"
#include "queue.hpp"
#include "resources.hpp"
#include "semaphore.hpp"
#include "staging.hpp"
#include "uploader_interface.hpp"

namespace nvdx12 {

/*-------------------------------------------------------------------------------------------------
# class nvdx12::AsyncUploader

> Collects uploads, typically from background threads, and submits them in
> batches on a dedicated transfer queue, the D3D12 counterpart of
> nvvk::AsyncUploader.

Staging memory comes from a thread-safe BufferSubAllocator of UPLOAD-heap
blocks. Appends go into a StagingCopyBatch that is submitted on
`transferQueue` once `flushSize` bytes are pending, or by `flushPending`.
Each batch is a command list of ManagedCommandPools, submitted through a
QueueTimeline of the uploader's own on the transfer queue: its fence values
are the SemaphoreStates the appends hand out, which tell the host (testSignaled,
wait) or another queue (ID3D12CommandQueue::Wait on the state's fence and
value) that the uploads completed. Staging memory is reclaimed once its
batch's state is signaled (`releaseCompletedAllocations`).

The transfer queue can be a COPY queue (typical for background streaming),
a COMPUTE queue or a DIRECT queue; `targetQueue` is where the resources are
used. The uploader needs exclusive submit access to the transfer queue
while it is in use. Setting both to the same queue gives a synchronous
uploader (see usage_AsyncUploader). Uploads from several threads are best
written in place: `acquireMapping`, fill, then `append*Mappings`.

Differences with nvvk that follow from D3D12:
- See uploader_interface.hpp for the interface itself (HRESULTs,
  ImageSubresourceLayers, D3D12_BARRIER_LAYOUT, placed-footprint mappings).
- No queue-family ownership transfer. Where nvvk releases buffers and images
  on the transfer queue and acquires them on the target queue
  (`enableOwnerBarriers`), D3D12 only needs the fence: buffers have no
  layout, and writes are visible to other queues once the
  ExecuteCommandLists that made them completed. Buffers therefore need no
  barriers at all; the target queue only waits for the returned state.
- Images are different: a texture used by several queue types must be in
  the COMMON layout when it changes queues, and a COPY queue cannot use any
  other layout. When the transfer queue is another queue than the target
  queue, or a COPY queue, images are therefore copied in COMMON (an
  UNDEFINED image is transitioned to COMMON on the transfer queue; an image
  in any other layout is rejected with E_INVALIDARG: transition it to
  COMMON on its queue first). The transition from COMMON to `newLayout` is
  what nvvk's acquire barrier becomes: `cmdDrainOwnershipBarriers` records
  it on the target queue for every batch that completed. `Image::layout`
  is set to the final layout when the upload is appended, as nvvk sets
  `descriptor.imageLayout`; until the drain recorded the transition, the
  image is still in COMMON on the GPU. Uploading again to an image whose
  transition was not drained yet is fine: the pending transition moves to
  the newer batch. Uploading to an image whose transition was drained (it is
  in use on the target queue) needs the image back in COMMON.
  When transfer and target queue are the same DIRECT or COMPUTE queue, the
  layout barriers are recorded in the batch (COPY_DEST, then `newLayout`)
  and nothing is left to drain.
- The uploader owns a QueueTimeline on `transferQueue` (its own fence) and
  submits through it; `waitForCompletion` waits on that fence. The
  `onSubmitCallback` receives the ID3D12GraphicsCommandList7 of the batch.
- Upload-heap memory is always coherent: no `forceCoherentMapping` and no
  flush ranges (nvvk's MAX_FLUSH_BLOCKS / RANGES_PER_BLOCK bookkeeping).
- Buffer destinations must be in a heap the GPU can write (DEFAULT or
  READBACK).

```cpp
nvdx12::AsyncUploader uploader;
uploader.init({.allocator = &allocator, .transferQueue = copyQueueInfo, .targetQueue = directQueueInfo});
// worker threads
nvdx12::SemaphoreState done;
uploader.appendBuffer(buffer, 0, std::span(data), &done);
uploader.flushPending();
// main thread, every frame
uploader.flushPending();
uploader.releaseCompletedAllocations();
const bool ready = done.testSignaled();  // before the drain, see cmdDrainOwnershipBarriers
uploader.cmdDrainOwnershipBarriers(cmd);
if(ready) { ... use buffer ... }
```

See usage_AsyncUploader() in async_uploader.cpp.
-------------------------------------------------------------------------------------------------*/

class AsyncUploader : public UploaderInterface
{
public:
  AsyncUploader() = default;
  ~AsyncUploader();
  AsyncUploader(const AsyncUploader&)            = delete;
  AsyncUploader& operator=(const AsyncUploader&) = delete;

  struct InitInfo
  {
    ResourceAllocator* allocator{};
    // the uploader must have exclusive submit access to this queue,
    // while being used. A COPY, COMPUTE or DIRECT queue.
    QueueInfo transferQueue{};

    // The target queue for where the resources are used at the end.
    // One must call `cmdDrainOwnershipBarriers` on this queue, prior
    // using images that completed their transfer.
    //
    // If targetQueue.queue and transferQueue.queue match (and are not a
    // COPY queue), the layout transitions are recorded with the copies.
    QueueInfo targetQueue{};

    // how many command lists can be in flight
    uint32_t commandPoolSize = 3;

    // after how much appended staging memory we trigger a flush
    uint32_t flushSize = 64 * 1024 * 1024;

    uint64_t blockSize = 128 * 1024 * 1024;

    // keeping at least one block is recommended for persistent usage
    uint32_t keepBlockCount = 1;

    // number of copy operations per block
    // number of mappings in flight
    uint32_t    maxOperations = 0xFFFF;
    std::string debugName{};

    // called with `begin == true` before the copies are recorded into `cmd`, and with
    // `begin == false` after them; the first call's return value is passed as `payload`
    // to the second.
    std::function<void*(ID3D12GraphicsCommandList7* cmd, uint32_t submitCount, SemaphoreState& sem, bool begin, void* payload)> onSubmitCallback =
        nullptr;
  };

  HRESULT init(const InitInfo& info);
  void    deinit();

  void setFlushSize(uint32_t flushSize)
  {
    std::lock_guard lock(m_appendLock);
    m_info.flushSize = flushSize;
  }

  uint32_t                   getSubmitCount() const { return m_submitCount.load(); }
  uint32_t                   getWaitCmdCount() const { return m_waitCmdCount.load(); }
  BufferSubAllocator::Report getAllocationReport() const;
  // true while image layout transitions wait for `cmdDrainOwnershipBarriers`
  bool hasOwnershipBarriers() const;

  uint32_t getAllocationCount() const { return uint32_t(m_allocationCount.load()); }

  // should be called at end of multiple upload append operations,
  // and/or once every frame
  // `semaphoreState` can be used to wait for anything up to this point
  // if there is no pending work does same as `releaseCompletedAllocations`
  // on failure `outSemaphoreState` is left untouched, as no signal of the
  // batch completes it.
  HRESULT flushPending(SemaphoreState* outSemaphoreState = nullptr);

  // waits for completion of last submit
  HRESULT waitForCompletion();

  // should be called once every frame
  // Only reclaims allocations of batches that were submitted and completed. Allocations of the
  // batch that is still being appended to are reclaimed once that batch was submitted and
  // completed as well, so an idle uploader holds on to its last staging memory until
  // `flushPending` was called.
  void releaseCompletedAllocations();

  // utility helper
  void flushAndWait()
  {
    flushPending();
    waitForCompletion();
    releaseCompletedAllocations();
  }

  // should be called after upload appends were done, to get a rough idea
  // what to wait for.
  // if there are pending operations left, returns next submit state
  // if flushed returns previous submit
  // if nothing was submitted yet, returns an invalid state
  //
  // due to async nature when you query after an appended uploaded, you might
  // not get exactly the submit your append was part of (as another thread could trigger a flush),
  // but typically this is good enough / guaranteed conservative.
  SemaphoreState getSubmitState();

  // Apply this on the target queue, prior using any uploaded images, typically once per frame.
  // Records the COMMON -> newLayout transitions of the images whose batch has been signaled
  // (the D3D12 replacement of nvvk's ownership acquisition barriers, see the class documentation).
  //
  // Example order of operations; the state is tested before the drain, so
  // that a batch completing in between still has its transition recorded
  // - worker threads
  //    - uploader.appendImage(myImage, ... &mySemaphoreState);
  // - main thread / target queue (in parallel)
  //    - uploader.flushPending
  //    - const bool ready = mySemaphoreState.testSignaled();
  //    - uploader.cmdDrainOwnershipBarriers(cmdFromTargetQueue);
  //    - if(ready) {... safe to use myImage }
  void cmdDrainOwnershipBarriers(ID3D12GraphicsCommandList7* cmd);

  // UploaderInterface, see `uploader_interface.hpp`

  using UploaderInterface::appendBufferRange;
  using UploaderInterface::appendImageSub;

  // acquire mapping space to be used to write data for uploads.
  HRESULT acquireMapping(size_t dataSize, BufferRange& mappingSpace, MappingHandle& mappingHandle) override;
  // Only call when not making use of `commitMapping_` in the `append*Mappings` functions.
  // This means no other operation can use that mappingHandle afterwards, but that it may have been used before.
  void commitMapping(MappingHandle& mappingHandle) override;
  // Only call manually when aborting the use of the mapping space.
  // This means no other operation can use that mappingHandle afterwards, and it must not have been used before.
  void releaseMapping(MappingHandle& mappingHandle) override;

  HRESULT appendBufferRange(const BufferRange& buffer, const void* data, SemaphoreState* outSemaphoreState) override;
  HRESULT appendBufferRangeMappings(size_t                    rangeCount,
                                    const MappedBufferRanges* ranges,
                                    MappingHandle&            mappingHandle,
                                    bool                      commitMapping_    = true,
                                    SemaphoreState*           outSemaphoreState = nullptr) override;

  HRESULT appendImageSub(Image&                        image,
                         const Offset3D&               offset,
                         const Extent3D&               extent,
                         const ImageSubresourceLayers& subresource,
                         size_t                        dataSize,
                         const void*                   data,
                         D3D12_BARRIER_LAYOUT          newLayout         = D3D12_BARRIER_LAYOUT_UNDEFINED,
                         SemaphoreState*               outSemaphoreState = nullptr) override;
  HRESULT appendImageSubMappings(Image&                 image,
                                 size_t                 imageSubCount,
                                 const MappedImageSubs* imageSubs,
                                 MappingHandle&         mappingHandle,
                                 bool                   commitMapping_    = true,
                                 D3D12_BARRIER_LAYOUT   newLayout         = D3D12_BARRIER_LAYOUT_UNDEFINED,
                                 SemaphoreState*        outSemaphoreState = nullptr) override;

protected:
  union MappingHandleDetail
  {
    uint64_t handle;
    struct
    {
      uint8_t  valid;
      uint8_t  used;
      uint32_t mappingIndex;
    };
  };

  // requires m_allocLock
  HRESULT acquireStagingSpace(BufferSubAllocation& subAllocation, BufferRange& stagingSpace, size_t dataSize, uint32_t alignment);

  // requires m_appendLock
  void addStagingAllocation(const BufferSubAllocation& subAllocation);
  // immediate free, only legal when nothing references the allocation
  // requires m_allocLock
  void freeStagingAllocation(BufferSubAllocation& subAllocation);
  void releaseStagingAllocations(bool forceAll);

  // require m_appendLock
  HRESULT submitBatch();
  HRESULT submitBatchChecked(size_t addedSize);

  // A COMMON -> layout transition taken out of its record by reclaimPendingTransition,
  // with the image's tracked state it replaced.
  struct PendingTransition
  {
    D3D12_TEXTURE_BARRIER barrier{};  // pResource is null if nothing was pending
    SemaphoreState        semaphoreState;
    D3D12_BARRIER_SYNC    stageMask{};
    D3D12_BARRIER_ACCESS  accessMask{};

    explicit operator bool() const { return barrier.pResource != nullptr; }
  };

  // When the queue changes: an image whose COMMON -> layout transition of an
  // earlier batch was not drained yet is still in COMMON on the GPU. Moves it
  // out of its record (the current batch adds a new one) and resets the
  // tracked layout to COMMON. `barrier.LayoutAfter` of the result is the
  // layout the image was going to. Requires m_appendLock.
  PendingTransition reclaimPendingTransition(Image& image);
  // Puts a reclaimed transition back when the append added no copy of the
  // image to the batch, so that cmdDrainOwnershipBarriers still records it.
  // Requires m_appendLock.
  void restorePendingTransition(Image& image, const PendingTransition& pending);

  // on success invalidates `handle`
  bool commitMappingHandle(MappingHandle& handle, BufferSubAllocation& subAllocation);

  // State of the last successful submit. Returns an invalid state when nothing was
  // submitted yet, as timeline value 0 is not a legal `SemaphoreState`.
  inline SemaphoreState getLastSubmitState() const
  {
    const uint64_t lastValue = m_transferTimeline.lastSignaledValue();
    if(lastValue == 0)
    {
      return {};
    }
    return SemaphoreState::makeFixed(m_transferTimeline.getFence(), lastValue);
  }

  inline void getOutSemaphoreState(SemaphoreState* out)
  {
    if(out != nullptr)
    {
      *out = m_transferSubmitState;
    }
  }

  struct StagingAllocation
  {
    BufferSubAllocation subAllocation;
    SemaphoreState      semaphoreState;
  };

  struct TargetBarriers
  {
    BarrierContainer ownerAcquisitionBarriers;
    SemaphoreState   semaphoreState;
  };

  // staging alignment of buffer copies, and of image footprints / mappings
  static constexpr uint32_t BUFFER_ALIGNMENT = 16;
  static constexpr uint32_t IMAGE_ALIGNMENT  = D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT;
  static constexpr uint32_t MAX_BLOCKS       = 64;

  InitInfo         m_info;
  StagingCopyBatch m_batch;

  // statistics
  // how often we had to wait for command list
  std::atomic_uint32_t m_waitCmdCount{0};
  // how many submits
  std::atomic_uint32_t m_submitCount{0};

  mutable std::mutex  m_allocLock;
  BufferSubAllocator  m_bufferSubAllocator;
  std::atomic_int32_t m_allocationCount{0};

  std::mutex                       m_mappingLock;
  nvutils::IDPool                  m_mappingPool;
  std::vector<BufferSubAllocation> m_mappingAllocations;

  std::mutex                     m_appendLock;
  std::vector<StagingAllocation> m_stagingAllocations;

  SemaphoreState      m_transferSubmitState;
  ManagedCommandPools m_transferCmdPool;
  QueueTimeline       m_transferTimeline;

  mutable std::mutex          m_targetLock;
  std::vector<TargetBarriers> m_targetBarriers;
};

}  // namespace nvdx12
