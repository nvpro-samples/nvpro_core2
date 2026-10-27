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

#include <string>
#include <vector>

#include "buffer_circular_allocator.hpp"
#include "resources.hpp"
#include "semaphore.hpp"
#include "staging.hpp"
#include "uploader_interface.hpp"

namespace nvdx12 {

/*-------------------------------------------------------------------------------------------------
# class nvdx12::FrameUploader

> Per-frame staged uploads recorded into the frame's command list, the
> D3D12 counterpart of nvvk::FrameUploader.

Meant for smaller data transfers that are done every frame;
nvdx12::AsyncUploader handles large transfers submitted on a queue of their
own. Staging memory is bump-allocated from UPLOAD-heap blocks of a
BufferCircularAllocator and retired in the same order once the frame's
SemaphoreState (`updateFrameSemaphoreState`) is signaled
(`releaseCompletedAllocations`). `cmdUploadAppended` records the copies and
the image layout transitions (StagingCopyBatch with layout barriers) into
the caller's command list. A mapping handle belongs to the frame of its
`acquireMapping`: after the next `updateFrameSemaphoreState` its staging may
be recycled, and `append*Mappings` rejects it with E_INVALIDARG.

Differences with nvvk that follow from D3D12:
- See uploader_interface.hpp for the interface itself (HRESULTs,
  ImageSubresourceLayers, D3D12_BARRIER_LAYOUT, placed-footprint mappings).
- Upload-heap memory is always coherent: no `forceCoherentMapping`, nothing
  is flushed, and nvvk's flush-block bookkeeping (MAX_FLUSH_BLOCKS) has no
  counterpart; the block count is still capped at 64 blocks (`maxAllocatedSize`).
- Image layouts follow StagingCopyBatch: whole-image transitions to
  COPY_DEST and to the final layout, tracked in nvdx12::Image.
- As nvvk sets `descriptor.imageLayout`, an appended Image tracks its final
  layout from the append on, before the barriers are recorded: do not
  transition it until `cmdUploadAppended` has recorded them.

```cpp
nvdx12::FrameUploader uploader;
uploader.init({.allocator = &allocator, .debugName = "frameUploads"});
// every frame
uploader.releaseCompletedAllocations();
uploader.updateFrameSemaphoreState(frameState);  // e.g. queue.createNextFixedSemaphoreState()
uploader.appendBuffer(buffer, 0, std::span(data));
uploader.cmdUploadAppended(cmd);
```

See usage_FrameUploader() in frame_uploader.cpp.
-------------------------------------------------------------------------------------------------*/

class FrameUploader : public CmdUploaderInterface
{
public:
  FrameUploader() = default;
  ~FrameUploader();
  FrameUploader(const FrameUploader&)            = delete;
  FrameUploader& operator=(const FrameUploader&) = delete;

  struct InitInfo
  {
    // resource allocator
    ResourceAllocator* allocator{};
    // we always keep at least 1 block around
    uint64_t blockSize = 4 * 1024 * 1024;
    // debug prefix for D3D12 resources
    std::string debugName{};
  };

  struct Stats
  {
    size_t copyCount{};
    size_t copyBytes{};
  };

  HRESULT init(const InitInfo& info);
  void    deinit();

  void updateFrameSemaphoreState(const SemaphoreState& semaphoreState);
  void releaseCompletedAllocations(bool forceAll = false);

  // copies of the last `cmdUploadAppended`
  const Stats& getStats() const { return m_stats; }

  //////////////////////////////////////////////////////////////////////////
  // UploaderInterface

  using UploaderInterface::appendBufferRange;
  using UploaderInterface::appendImageSub;

  HRESULT acquireMapping(size_t dataSize, BufferRange& mappingSpace, MappingHandle& mappingHandle) override;
  void    commitMapping(MappingHandle& mappingHandle) override;
  void    releaseMapping(MappingHandle& mappingHandle) override;

  HRESULT appendBufferRange(const BufferRange& buffer, const void* data, SemaphoreState* outSemaphoreState) override;
  HRESULT appendBufferRangeMappings(size_t                    mappingCount,
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

  //////////////////////////////////////////////////////////////////////////
  // CmdUploaderInterface

  void cmdUploadAppended(ID3D12GraphicsCommandList7* cmd) override;


protected:
  union MappingHandleDetail
  {
    uint64_t handle;
    struct
    {
      uint8_t  valid;
      uint8_t  used;
      uint32_t frameIndex;
    };
  };

  HRESULT acquireStagingSpace(BufferCircularAllocation& subAllocation, BufferRange& stagingSpace, size_t dataSize, uint32_t alignment);

  inline void getOutSemaphoreState(SemaphoreState* out)
  {
    if(out != nullptr)
    {
      *out = m_semaphoreState;
    }
  }

  struct StagingAllocation
  {
    BufferCircularAllocation subAllocation;
    SemaphoreState           semaphoreState;
  };

  static constexpr uint32_t MAX_BLOCKS = 64;

  uint32_t                       m_frameIndex{0};
  ResourceAllocator*             m_resourceAllocator = nullptr;
  StagingCopyBatch               m_batch;
  BufferCircularAllocator        m_bufferCircularAllocator;
  Stats                          m_stats;
  SemaphoreState                 m_semaphoreState;
  std::vector<StagingAllocation> m_stagingAllocations;
};

}  // namespace nvdx12
