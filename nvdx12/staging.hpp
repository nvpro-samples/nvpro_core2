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

#include <cstdint>
#include <vector>

#include "barriers.hpp"
#include "resources.hpp"
#include "uploader_interface.hpp"

namespace nvdx12 {

/*-------------------------------------------------------------------------------------------------
# class nvdx12::StagingCopyBatch

> Batches staging copies and the image layout barriers around them, the
> D3D12 counterpart of nvvk::StagingCopyBatch. Fully public so classes that
> manage uploads (FrameUploader, AsyncUploader) or downloads can use it
> directly.

Buffer copies are CopyBufferRegion calls; image copies are CopyTextureRegion
calls from a placed footprint (makeImageSubFootprint) in the staging
resource, one per array layer.

Differences with nvvk that follow from D3D12:
- Layout barriers cover the whole image and are recorded once per image and
  batch, because nvdx12::Image tracks one layout for all subresources: the
  first copy into an image adds its `pre` transition (from the tracked
  layout and scope to COPY_DEST), later copies into the same image only
  update the final layout. The image's tracked layout and scope are updated
  when the copy is appended.
- No queue-family ownership: `enableOwnerBarriers`, the queue family indices
  and `addOwnerBufferBarrier` are replaced by `enableQueueTransfer`, for
  copies that execute on another queue than the one using the images, or on
  a COPY queue. D3D12 has no ownership transfer, but a texture written on
  one queue and used on another must be in the COMMON layout across the
  queue switch, and a COPY queue accepts no other layout. In that mode the
  image is copied in COMMON (UNDEFINED images get an UNDEFINED -> COMMON
  `pre` barrier; any other layout is rejected, transition it to COMMON
  first) and its transition to the final layout lands in `acquire`, which
  the using queue records after waiting for the copies. Buffers have no
  layout and need nothing: a fence wait orders the queues, and writes are
  visible at ExecuteCommandLists boundaries.
- `enableLayoutBarrierMask` and its masks are left out. Queue-transfer mode
  (`enableQueueTransfer`) is the one for copies on a COPY queue. Layout-barrier
  mode (`enableLayoutBarriers`) waits for the image's tracked scope (or its
  layout's) and leaves it with its final layout's scope, so it must run on
  the queue that uses the image.
- `addImageCopy` returns an HRESULT (E_INVALIDARG for a copy `checkImageCopy`
  rejects or a staging footprint that is unaligned or too small).
- `checkImageCopy` and `checkImageSubCopies` are additions: they run the
  validation of `addImageCopy` without appending, so that an uploader can
  reject a whole append before adding any copy.
- nvvk::StagingUploader, which shares nvvk/staging.hpp, has no counterpart:
  FrameUploader and AsyncUploader replace it (uploader_interface.hpp).

See usage_StagingCopyBatch() in staging.cpp.
-------------------------------------------------------------------------------------------------*/
class StagingCopyBatch
{
public:
  //////////////////////////////////////////////////////////////////////////
  // persistent state

  // enqueues layout transitions for images
  // `pre` contains the transitions to a copyable layout and `post` the
  // transitions to the final layout (the original one, or `newLayout`).
  bool enableLayoutBarriers = false;

  // the copies run on another queue than the one using the images (or on a
  // COPY queue): images are copied in COMMON and the transitions to their
  // final layouts go to `acquire` (see the class documentation).
  bool enableQueueTransfer = false;

  //////////////////////////////////////////////////////////////////////////
  // these states are reset/cleared on `reset` and/or `cmdCopyAppended`
  uint64_t stagingSize = 0;

  struct BufferCopy
  {
    ID3D12Resource* srcBuffer{};
    uint64_t        srcOffset{};
    ID3D12Resource* dstBuffer{};
    uint64_t        dstOffset{};
    uint64_t        size{};
  };
  struct ImageCopy
  {
    D3D12_TEXTURE_COPY_LOCATION dst{};  // subresource index
    D3D12_TEXTURE_COPY_LOCATION src{};  // placed footprint
    Offset3D                    offset{};
  };

  // These copy operations are triggered during `cmdCopyAppended`.
  std::vector<BufferCopy> copyBufferRegions;
  std::vector<ImageCopy>  copyBufferImageRegions;

  // barriers performed on the provided command list during `cmdCopyAppended`
  BarrierContainer pre;
  BarrierContainer post;
  // with `enableQueueTransfer`: transitions the using queue must record
  // after the copies completed, gathered before `cmdCopyAppended`
  BarrierContainer acquire;

  // nothing in the batch
  bool isAppendedEmpty() const { return copyBufferRegions.empty() && copyBufferImageRegions.empty(); }

  // check if we should flush because already enough traffic caused by the batch
  inline bool checkAppendedSize(size_t limitInBytes, size_t addedSize = 0) const
  {
    return stagingSize && (stagingSize + addedSize) > limitInBytes;
  }

  // Records pending copy and barrier operations into `cmd` and then clears the batch state (`reset`).
  // Warning: Staging resource ownership and lifetime must be managed by caller.
  void cmdCopyAppended(ID3D12GraphicsCommandList7* cmd);

  // Records the pending copy and barrier operations into `cmd` and keeps the
  // batch state, for a caller that calls `reset` once `cmd` closed.
  void cmdRecordAppended(ID3D12GraphicsCommandList7* cmd) const;

  // resets all vectors and intermediate `stagingSize`
  void reset();

  // adds a buffer copy
  void addBufferCopy(ID3D12Resource* stagingBuffer, uint64_t stagingOffset, ID3D12Resource* buffer, uint64_t bufferOffset, uint64_t dataSize);

  // Validates an image region copy: E_INVALIDARG with a log if `image` is null,
  // depth-stencil or multisampled, its format is unsupported, the region is
  // outside its mip and layers or not block aligned (offsets on block
  // boundaries, extents whole blocks or reaching the edge of the mip), or, with
  // `enableQueueTransfer`, an image new to the batch is neither UNDEFINED nor COMMON.
  HRESULT checkImageCopy(const Image& image, const ImageSubresourceLayers& subresource, const Offset3D& offset, const Extent3D& extent) const;

  // checkImageCopy for every sub of an appendImageSubMappings call, plus
  // their staging placement (512-byte aligned, holding the footprint); the
  // uploaders call it before appending any sub, so a failure appends nothing.
  HRESULT checkImageSubCopies(const Image& image, size_t imageSubCount, const UploaderInterface::MappedImageSubs* imageSubs) const;

  // adds the copies of an image region from staging laid out as
  // makeImageSubFootprint(image.format, extent, subresource.layerCount) at
  // `stagingOffset` (a multiple of 512), and the layout barriers (see the
  // class documentation). `newLayout` UNDEFINED keeps the original layout.
  HRESULT addImageCopy(ID3D12Resource*               stagingBuffer,
                       uint64_t                      stagingOffset,
                       Image&                        image,
                       D3D12_BARRIER_LAYOUT          newLayout,
                       size_t                        dataSize,
                       const ImageSubresourceLayers& subresource,
                       const Offset3D&               offset,
                       const Extent3D&               extent);

protected:
  // one entry per image of the batch: where its final transition lives
  struct ImageState
  {
    ID3D12Resource*      resource{};
    size_t               barrierIndex{~size_t(0)};  // into post (or acquire with enableQueueTransfer), ~0 if none
    D3D12_BARRIER_LAYOUT copyLayout{};              // layout the copies run in
  };
  std::vector<ImageState> m_imageStates;
};

}  // namespace nvdx12
