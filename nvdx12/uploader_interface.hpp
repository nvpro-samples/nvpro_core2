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

#include <cassert>
#include <cstdint>
#include <cstring>
#include <span>

#include "formats.hpp"
#include "resources.hpp"
#include "semaphore.hpp"

namespace nvdx12 {

/*-------------------------------------------------------------------------------------------------
# class nvdx12::UploaderInterface / nvdx12::CmdUploaderInterface

> The common interface of the uploaders, the D3D12 counterpart of
> nvvk/uploader_interface.hpp: nvdx12::FrameUploader (CmdUploaderInterface,
> the caller records the copies) and nvdx12::AsyncUploader (submits on its
> own queue).

The interface yields control over the actual submits of the copies to the
implementation, as well as the SemaphoreState management. Each append can
query the SemaphoreState (`outSemaphoreState`) that tracks its completion.
Staging space can be acquired first (`acquireMapping`), filled, possibly from
several threads, and then handed to `append*Mappings`.

Differences with nvvk that follow from D3D12:
- Results are HRESULTs. Buffers and ranges are nvdx12::Buffer / BufferRange
  (resources.hpp), images nvdx12::Image, whose tracked `layout` plays the
  part of `descriptor.imageLayout`.
- `ImageSubresourceLayers` (below) replaces VkImageSubresourceLayers: one
  mip level and a range of array layers; there is no aspect (single-plane
  color formats only).
- Image data: `appendImageSub` / `appendImage` take the region tightly
  packed (rows of texel blocks, then depth slices, then layers) and the
  uploader re-pitches it into the 256-byte aligned rows D3D12 copies need.
  The mapping variant cannot re-pitch: `MappedImageSubs::mappingSpace` must
  hold the region in D3D12 placed-footprint layout, described by
  `makeImageSubFootprint` (`writeImageSubFootprint` fills it from tightly
  packed data), and start at a multiple of
  D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT (512) inside the staging resource.
  `acquireMapping` always returns such an aligned start.
- `newLayout` is a D3D12_BARRIER_LAYOUT. As in nvvk,
  D3D12_BARRIER_LAYOUT_UNDEFINED (the default) keeps the layout the image had
  before the upload; an image that had none (UNDEFINED, freshly created)
  stays in the layout of the copy: COPY_DEST, or COMMON for an AsyncUploader
  on another queue (see async_uploader.hpp).
- Buffer destinations must be in a heap the GPU can write (DEFAULT or
  READBACK): D3D12 cannot copy into an UPLOAD-heap buffer.
- `appendLargeBuffer*` are left out: D3D12 has no large (sparse) buffers.
-------------------------------------------------------------------------------------------------*/

// One mip level and a range of array layers of an image, the counterpart of
// VkImageSubresourceLayers (no aspect: single-plane color formats).
struct ImageSubresourceLayers
{
  uint32_t mipLevel{0};
  uint32_t baseArrayLayer{0};
  uint32_t layerCount{1};
};

// The D3D12 placed-footprint layout of an image region in staging memory:
// what CopyTextureRegion reads and what `MappedImageSubs::mappingSpace` holds.
// Each layer starts at `layerPitch * layer` (a multiple of 512); inside a
// layer, rows of texel blocks are `footprint.RowPitch` apart (a multiple of
// 256) and depth slices follow each other (`rowCount` rows each).
struct ImageSubFootprint
{
  D3D12_SUBRESOURCE_FOOTPRINT footprint{};    // Width / Height cover whole blocks
  uint64_t                    rowSize{0};     // bytes of one tightly packed row of blocks
  uint32_t                    rowCount{0};    // rows of blocks per depth slice
  uint64_t                    layerPitch{0};  // bytes between the layers' footprints
  uint64_t                    size{0};        // staging bytes for all layers
  uint64_t                    tightSize{0};   // tightly packed bytes for all layers (the `dataSize` of appendImageSub)

  explicit operator bool() const { return rowSize != 0; }
};

// Footprint of an `extent` region (texels) of `layerCount` layers of
// `format`; returns an empty footprint for an unsupported format.
inline ImageSubFootprint makeImageSubFootprint(DXGI_FORMAT format, const Extent3D& extent, uint32_t layerCount = 1)
{
  ImageSubFootprint     result;
  const FormatBlockInfo block = formatBlockInfo(format);
  if(block.bytes == 0 || extent.width == 0 || extent.height == 0 || extent.depth == 0 || layerCount == 0)
  {
    return result;
  }
  const uint32_t blocksX    = (extent.width + block.width - 1) / block.width;
  const uint32_t rows       = (extent.height + block.height - 1) / block.height;
  result.rowSize            = uint64_t(blocksX) * block.bytes;
  result.rowCount           = rows;
  result.footprint.Format   = format;
  result.footprint.Width    = blocksX * block.width;
  result.footprint.Height   = rows * block.height;
  result.footprint.Depth    = extent.depth;
  result.footprint.RowPitch = static_cast<UINT>((result.rowSize + D3D12_TEXTURE_DATA_PITCH_ALIGNMENT - 1)
                                                & ~uint64_t(D3D12_TEXTURE_DATA_PITCH_ALIGNMENT - 1));
  const uint64_t layerBytes = uint64_t(result.footprint.RowPitch) * rows * extent.depth;
  result.layerPitch =
      (layerBytes + D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT - 1) & ~uint64_t(D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT - 1);
  result.size      = result.layerPitch * (layerCount - 1) + layerBytes;
  result.tightSize = result.rowSize * rows * extent.depth * layerCount;
  return result;
}

// Copies tightly packed region data (`footprint.tightSize` bytes, layers
// after each other) into `dst`, laid out as `footprint` describes.
inline void writeImageSubFootprint(const ImageSubFootprint& footprint, void* dst, const void* tightData, uint32_t layerCount = 1)
{
  const uint8_t* src          = static_cast<const uint8_t*>(tightData);
  const uint32_t rowsPerLayer = footprint.rowCount * footprint.footprint.Depth;
  for(uint32_t layer = 0; layer < layerCount; layer++)
  {
    uint8_t* layerDst = static_cast<uint8_t*>(dst) + footprint.layerPitch * layer;
    for(uint32_t row = 0; row < rowsPerLayer; row++)
    {
      memcpy(layerDst + uint64_t(row) * footprint.footprint.RowPitch, src, footprint.rowSize);
      src += footprint.rowSize;
    }
  }
}

// This interface yields control over the actual submits of copying to the implementation,
// as well as the SemaphoreState management. However, each appended upload can query for
// the SemaphoreState to track its completion, if required.
//
// Some uploaders don't submit on their own, but record their copies into a command
// list the caller provides, see `CmdUploaderInterface`.
class UploaderInterface
{
public:
  // intentionally not providing a virtual destructor, because the interface is only meant
  // for operations, not for managing the lifetime of derived classes, as those classes
  // bring their own implementation details.

  // Opaque handle to an acquired mapping; a zero / default-constructed handle is invalid.
  struct MappingHandle
  {
    uint64_t value = 0;
    explicit operator bool() const { return value != 0; }
    bool     operator==(const MappingHandle& other) const = default;
  };

  struct MappedBufferRanges
  {
    BufferRange bufferRange;
    BufferRange mappingRange;
  };

  struct MappedImageSubs
  {
    Offset3D               offset{};
    Extent3D               extent{};
    ImageSubresourceLayers subresource{};
    // holds the region in the layout of makeImageSubFootprint(image.format, extent, subresource.layerCount),
    // starting at a multiple of 512 inside the staging resource
    BufferRange mappingSpace;
  };

  // This interface allows decoupling acquiring the staging space and writing to it
  // from the submit of the copy command later. In a multi-threaded scenario each
  // thread should acquire a bit of a bigger chunk that it fills and then uses for uploads.
  // The returned `mappingSpace.offset` is a multiple of 512 (D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT).
  virtual HRESULT acquireMapping(size_t dataSize, BufferRange& mappingSpace, MappingHandle& mappingHandle) = 0;

  // Call when transferring ownership of the used mapping to the uploader.
  // Normally use `commitMapping_ = true` in the last
  // use of a mapping handle in the later `append*Mapping` functions,
  // or call this after all such uses were appended.
  // After a handle was committed it cannot be used again.
  virtual void commitMapping(MappingHandle& mappingHandle) = 0;

  // Call to abort the use of a mapping that was not part of any append
  // operation. The handle is invalid afterwards; the memory is freed as the
  // implementation retires its staging: at once by AsyncUploader, with the
  // frame it belongs to by FrameUploader.
  virtual void releaseMapping(MappingHandle& mappingHandle) = 0;


  //////////////////////////////////////////////////////////////////////////

  // core interface functions

  // if `outSemaphoreState` is provided it will store the semaphore state that can be
  // used to track completion
  // if `commitMapping_ == true`, then we transfer ownership of the mapping into the uploader,
  // at that point don't call commitMapping manually.

  virtual HRESULT appendBufferRange(const BufferRange& bufferRange, const void* data, SemaphoreState* outSemaphoreState) = 0;
  virtual HRESULT appendBufferRangeMappings(size_t                    rangeCount,
                                            const MappedBufferRanges* ranges,
                                            MappingHandle&            mappingHandle,
                                            bool                      commitMapping_    = true,
                                            SemaphoreState*           outSemaphoreState = nullptr) = 0;

  // `data` holds the region tightly packed (see the class documentation).
  virtual HRESULT appendImageSub(Image&                        image,
                                 const Offset3D&               offset,
                                 const Extent3D&               extent,
                                 const ImageSubresourceLayers& subresource,
                                 size_t                        dataSize,
                                 const void*                   data,
                                 D3D12_BARRIER_LAYOUT          newLayout         = D3D12_BARRIER_LAYOUT_UNDEFINED,
                                 SemaphoreState*               outSemaphoreState = nullptr)         = 0;
  virtual HRESULT appendImageSubMappings(Image&                 image,
                                         size_t                 imageSubCount,
                                         const MappedImageSubs* imageSubs,
                                         MappingHandle&         mappingHandle,
                                         bool                   commitMapping_    = true,
                                         D3D12_BARRIER_LAYOUT   newLayout         = D3D12_BARRIER_LAYOUT_UNDEFINED,
                                         SemaphoreState*        outSemaphoreState = nullptr) = 0;

  //////////////////////////////////////////////////////////////////////////

  // utility wrappers

  // `offset` is relative to the start of the Buffer.
  inline HRESULT appendBuffer(const Buffer& buffer, size_t offset, size_t dataSize, const void* data, SemaphoreState* outSemaphoreState = nullptr)
  {
    return appendBufferRange(makeBufferRange(buffer, offset, dataSize), data, outSemaphoreState);
  }

  inline HRESULT appendBufferMapping(const Buffer&      buffer,
                                     size_t             offset,
                                     size_t             dataSize,
                                     const BufferRange& mappingSpace,
                                     MappingHandle&     mappingHandle,
                                     bool               commitMapping_    = true,
                                     SemaphoreState*    outSemaphoreState = nullptr)
  {
    MappedBufferRanges mappedRange;
    mappedRange.bufferRange  = makeBufferRange(buffer, offset, dataSize);
    mappedRange.mappingRange = mappingSpace;

    return appendBufferRangeMappings(1, &mappedRange, mappingHandle, commitMapping_, outSemaphoreState);
  }

  template <typename T>
  inline HRESULT appendBufferRange(const BufferRange& bufferRange, std::span<T> data, SemaphoreState* outSemaphoreState = nullptr)
  {
    assert(bufferRange.range == data.size_bytes());
    return appendBufferRange(bufferRange, data.data(), outSemaphoreState);
  }

  template <typename T>
  inline HRESULT appendBuffer(const Buffer& buffer, size_t offset, std::span<T> data, SemaphoreState* outSemaphoreState = nullptr)
  {
    return appendBuffer(buffer, offset, data.size_bytes(), data.data(), outSemaphoreState);
  }

  template <typename T>
  inline HRESULT appendImageSub(Image&                        image,
                                const Offset3D&               offset,
                                const Extent3D&               extent,
                                const ImageSubresourceLayers& subresource,
                                std::span<T>                  data,
                                D3D12_BARRIER_LAYOUT          newLayout         = D3D12_BARRIER_LAYOUT_UNDEFINED,
                                SemaphoreState*               outSemaphoreState = nullptr)
  {
    return appendImageSub(image, offset, extent, subresource, data.size_bytes(), data.data(), newLayout, outSemaphoreState);
  }

  // Uploads mip 0 / layer 0.
  inline HRESULT appendImage(Image&               image,
                             size_t               dataSize,
                             const void*          data,
                             D3D12_BARRIER_LAYOUT newLayout         = D3D12_BARRIER_LAYOUT_UNDEFINED,
                             SemaphoreState*      outSemaphoreState = nullptr)
  {
    return appendImageSub(image, {0, 0, 0}, image.extent, {.mipLevel = 0, .baseArrayLayer = 0, .layerCount = 1},
                          dataSize, data, newLayout, outSemaphoreState);
  }

  template <typename T>
  inline HRESULT appendImage(Image&               image,
                             std::span<T>         data,
                             D3D12_BARRIER_LAYOUT newLayout         = D3D12_BARRIER_LAYOUT_UNDEFINED,
                             SemaphoreState*      outSemaphoreState = nullptr)
  {
    return appendImage(image, data.size_bytes(), data.data(), newLayout, outSemaphoreState);
  }
};

// Interface for uploaders that do not own a queue and therefore cannot submit by themselves.
// The caller decides where the copies land in its command stream, and is responsible for
// their submit and synchronization.
class CmdUploaderInterface : public UploaderInterface
{
public:
  // Records the appended operations (copy & relevant layout transitions) into the command list.
  virtual void cmdUploadAppended(ID3D12GraphicsCommandList7* cmd) = 0;
};

}  // namespace nvdx12
