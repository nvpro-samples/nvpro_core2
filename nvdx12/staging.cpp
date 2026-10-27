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

#include "staging.hpp"

#include <algorithm>
#include <cstring>
#include <tuple>
#include <vector>

#include <nvutils/logger.hpp>

#include "barriers.hpp"
#include "check_error.hpp"
#include "formats.hpp"
#include "resource_allocator.hpp"

namespace nvdx12 {

// Extent of a mip level (`mipLevel` < image.mipLevels); depth is 1 for anything but 3D images.
static Extent3D mipExtent(const Image& image, uint32_t mipLevel)
{
  const bool is3D = image.dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D;
  return {std::max(1u, image.extent.width >> mipLevel), std::max(1u, image.extent.height >> mipLevel),
          is3D ? std::max(1u, image.extent.depth >> mipLevel) : 1u};
}

// True if the region lies inside mip `mipLevel`, whose extent is rounded up
// to whole blocks (the last mips of a compressed format still hold one block),
// and is block aligned: offsets on block boundaries, extents whole blocks or
// reaching the edge of the mip (D3D12 rejects other copies of compressed formats).
static bool regionInsideLevel(const Image& image, const FormatBlockInfo& block, uint32_t mipLevel, const Offset3D& offset, const Extent3D& extent)
{
  if(mipLevel >= image.mipLevels)
  {
    return false;
  }
  const Extent3D level       = mipExtent(image, mipLevel);
  const uint64_t levelWidth  = uint64_t(level.width + block.width - 1) / block.width * block.width;
  const uint64_t levelHeight = uint64_t(level.height + block.height - 1) / block.height * block.height;
  if(!(offset.x >= 0 && offset.y >= 0 && offset.z >= 0 && extent.width && extent.height && extent.depth
       && uint64_t(offset.x) + extent.width <= levelWidth && uint64_t(offset.y) + extent.height <= levelHeight
       && uint64_t(offset.z) + extent.depth <= level.depth))
  {
    return false;
  }
  return uint32_t(offset.x) % block.width == 0 && uint32_t(offset.y) % block.height == 0
         && (extent.width % block.width == 0 || uint64_t(offset.x) + extent.width == level.width)
         && (extent.height % block.height == 0 || uint64_t(offset.y) + extent.height == level.height);
}

//--------------------------------------------------------------------------------------------------
// StagingCopyBatch

void StagingCopyBatch::reset()
{
  // clear rather than {} to keep the allocations
  copyBufferRegions.clear();
  copyBufferImageRegions.clear();
  pre.clear();
  post.clear();
  acquire.clear();
  m_imageStates.clear();
  stagingSize = 0;
}

void StagingCopyBatch::cmdCopyAppended(ID3D12GraphicsCommandList7* cmd)
{
  cmdRecordAppended(cmd);
  reset();
}

void StagingCopyBatch::cmdRecordAppended(ID3D12GraphicsCommandList7* cmd) const
{
  if(!pre.imageBarriers.empty() || !pre.bufferBarriers.empty() || !pre.memoryBarriers.empty())
  {
    pre.cmdPipelineBarrier(cmd);
  }
  for(size_t i = 0; i < copyBufferRegions.size(); i++)
  {
    const BufferCopy& copy = copyBufferRegions[i];
    cmd->CopyBufferRegion(copy.dstBuffer, copy.dstOffset, copy.srcBuffer, copy.srcOffset, copy.size);
  }
  for(size_t i = 0; i < copyBufferImageRegions.size(); i++)
  {
    const ImageCopy& copy = copyBufferImageRegions[i];
    cmd->CopyTextureRegion(&copy.dst, UINT(copy.offset.x), UINT(copy.offset.y), UINT(copy.offset.z), &copy.src, nullptr);
  }
  if(!post.imageBarriers.empty() || !post.bufferBarriers.empty() || !post.memoryBarriers.empty())
  {
    post.cmdPipelineBarrier(cmd);
  }
}

void StagingCopyBatch::addBufferCopy(ID3D12Resource* stagingBuffer, uint64_t stagingOffset, ID3D12Resource* buffer, uint64_t bufferOffset, uint64_t dataSize)
{
  stagingSize += dataSize;
  copyBufferRegions.push_back({stagingBuffer, stagingOffset, buffer, bufferOffset, dataSize});
}

HRESULT StagingCopyBatch::checkImageCopy(const Image&                  image,
                                         const ImageSubresourceLayers& subresource,
                                         const Offset3D&               offset,
                                         const Extent3D&               extent) const
{
  const FormatBlockInfo block = formatBlockInfo(image.format);
  if(!image.image || block.bytes == 0)
  {
    LOGE("StagingCopyBatch: null image or unsupported format %d\n", int(image.format));
    return E_INVALIDARG;
  }
  // Buffer-to-image copies take single-plane color images: no depth-stencil
  // (whole subresources, planes) and no multisampled destinations.
  if(image.sampleCount > 1 || (image.flags & D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL) != 0)
  {
    LOGE("StagingCopyBatch: depth-stencil and multisampled images are not copy destinations\n");
    return E_INVALIDARG;
  }
  if(subresource.layerCount == 0 || subresource.baseArrayLayer >= image.arrayLayers
     || subresource.layerCount > image.arrayLayers - subresource.baseArrayLayer
     || !regionInsideLevel(image, block, subresource.mipLevel, offset, extent))
  {
    LOGE("StagingCopyBatch: region (%d, %d, %d) + %ux%ux%u outside mip %u / layers [%u, %u)\n", offset.x, offset.y,
         offset.z, extent.width, extent.height, extent.depth, subresource.mipLevel, subresource.baseArrayLayer,
         subresource.baseArrayLayer + subresource.layerCount);
    return E_INVALIDARG;
  }
  const bool inBatch = std::any_of(m_imageStates.begin(), m_imageStates.end(),
                                   [&](const ImageState& state) { return state.resource == image.image.Get(); });
  if(enableQueueTransfer && !inBatch && image.layout != D3D12_BARRIER_LAYOUT_UNDEFINED && image.layout != D3D12_BARRIER_LAYOUT_COMMON)
  {
    LOGE("StagingCopyBatch: a copy on another queue needs the image in COMMON or UNDEFINED layout (it is in %d)\n",
         int(image.layout));
    return E_INVALIDARG;
  }
  return S_OK;
}

HRESULT StagingCopyBatch::checkImageSubCopies(const Image& image, size_t imageSubCount, const UploaderInterface::MappedImageSubs* imageSubs) const
{
  for(size_t i = 0; i < imageSubCount; i++)
  {
    const UploaderInterface::MappedImageSubs& sub = imageSubs[i];
    if(sub.mappingSpace.range == 0)
    {
      continue;
    }
    const ImageSubFootprint footprint = makeImageSubFootprint(image.format, sub.extent, sub.subresource.layerCount);
    if(!footprint || sub.mappingSpace.range < footprint.size || sub.mappingSpace.offset % D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT != 0)
    {
      LOGE("StagingCopyBatch: image sub %zu has an unaligned staging offset or holds less than its footprint (%llu bytes)\n",
           i, static_cast<unsigned long long>(footprint.size));
      return E_INVALIDARG;
    }
    NVDX12_FAIL_RETURN(checkImageCopy(image, sub.subresource, sub.offset, sub.extent));
  }
  return S_OK;
}

HRESULT StagingCopyBatch::addImageCopy(ID3D12Resource*               stagingBuffer,
                                       uint64_t                      stagingOffset,
                                       Image&                        image,
                                       D3D12_BARRIER_LAYOUT          newLayout,
                                       size_t                        dataSize,
                                       const ImageSubresourceLayers& subresource,
                                       const Offset3D&               offset,
                                       const Extent3D&               extent)
{
  HRESULT hr = checkImageCopy(image, subresource, offset, extent);
  if(FAILED(hr))
  {
    return hr;
  }
  const ImageSubFootprint footprint = makeImageSubFootprint(image.format, extent, subresource.layerCount);
  if(stagingOffset % D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT != 0 || dataSize < footprint.size)
  {
    LOGE("StagingCopyBatch: image staging at offset %llu with %llu bytes, needs 512-byte alignment and %llu bytes\n",
         static_cast<unsigned long long>(stagingOffset), static_cast<unsigned long long>(dataSize),
         static_cast<unsigned long long>(footprint.size));
    return E_INVALIDARG;
  }

  ID3D12Resource* resource = image.image.Get();
  auto            it       = std::find_if(m_imageStates.begin(), m_imageStates.end(),
                                          [&](const ImageState& state) { return state.resource == resource; });
  const bool      first    = it == m_imageStates.end();
  if(first)
  {
    ImageState state{.resource = resource};
    if(enableQueueTransfer)
    {
      // Copies on another queue run in COMMON, the only layout every queue
      // type shares; an UNDEFINED image gets there without prior access.
      state.copyLayout = D3D12_BARRIER_LAYOUT_COMMON;
      if(image.layout == D3D12_BARRIER_LAYOUT_UNDEFINED)
      {
        pre.imageBarriers.push_back({.SyncBefore   = D3D12_BARRIER_SYNC_NONE,
                                     .SyncAfter    = D3D12_BARRIER_SYNC_COPY,
                                     .AccessBefore = D3D12_BARRIER_ACCESS_NO_ACCESS,
                                     .AccessAfter  = D3D12_BARRIER_ACCESS_COPY_DEST,
                                     .LayoutBefore = D3D12_BARRIER_LAYOUT_UNDEFINED,
                                     .LayoutAfter  = D3D12_BARRIER_LAYOUT_COMMON,
                                     .pResource    = resource,
                                     .Subresources = DEFAULT_D3D12_BARRIER_SUBRESOURCE_RANGE,
                                     .Flags        = D3D12_TEXTURE_BARRIER_FLAG_NONE});
        image.layout = D3D12_BARRIER_LAYOUT_COMMON;
      }
    }
    else if(enableLayoutBarriers)
    {
      state.copyLayout = D3D12_BARRIER_LAYOUT_COPY_DEST;
      // Source scope: the tracked one, or the layout's when nothing is tracked.
      auto [syncBefore, accessBefore] = imageSourceScope(image);
      if(image.layout == D3D12_BARRIER_LAYOUT_UNDEFINED)
      {
        syncBefore   = D3D12_BARRIER_SYNC_NONE;
        accessBefore = D3D12_BARRIER_ACCESS_NO_ACCESS;
      }
      pre.imageBarriers.push_back({.SyncBefore   = syncBefore,
                                   .SyncAfter    = D3D12_BARRIER_SYNC_COPY,
                                   .AccessBefore = accessBefore,
                                   .AccessAfter  = D3D12_BARRIER_ACCESS_COPY_DEST,
                                   .LayoutBefore = image.layout,
                                   .LayoutAfter  = D3D12_BARRIER_LAYOUT_COPY_DEST,
                                   .pResource    = resource,
                                   .Subresources = DEFAULT_D3D12_BARRIER_SUBRESOURCE_RANGE,
                                   .Flags = image.layout == D3D12_BARRIER_LAYOUT_UNDEFINED ? D3D12_TEXTURE_BARRIER_FLAG_DISCARD :
                                                                                             D3D12_TEXTURE_BARRIER_FLAG_NONE});
    }
    else
    {
      // the caller put the image in a layout that allows the copy
      state.copyLayout = image.layout;
    }
    m_imageStates.push_back(state);
    it = m_imageStates.end() - 1;
  }

  // Final layout: `newLayout`, else the layout before the batch (which the
  // first append resolved into image.layout already), else the copy layout.
  D3D12_BARRIER_LAYOUT finalLayout = newLayout != D3D12_BARRIER_LAYOUT_UNDEFINED ? newLayout : image.layout;
  if(finalLayout == D3D12_BARRIER_LAYOUT_UNDEFINED)
  {
    finalLayout = it->copyLayout;
  }

  if(enableQueueTransfer || enableLayoutBarriers)
  {
    BarrierContainer& target = enableQueueTransfer ? acquire : post;
    // The transfer's acquire runs on the using queue after the fence wait:
    // nothing on that queue touched the image before.
    const D3D12_BARRIER_SYNC syncBefore = enableQueueTransfer ? D3D12_BARRIER_SYNC_NONE : D3D12_BARRIER_SYNC_COPY;
    const D3D12_BARRIER_ACCESS accessBefore = enableQueueTransfer ? D3D12_BARRIER_ACCESS_NO_ACCESS : D3D12_BARRIER_ACCESS_COPY_DEST;
    const auto [syncAfter, accessAfter] = inferPipelineStageAccessTuple(finalLayout);
    const D3D12_TEXTURE_BARRIER barrier{.SyncBefore   = syncBefore,
                                        .SyncAfter    = syncAfter,
                                        .AccessBefore = accessBefore,
                                        .AccessAfter  = accessAfter,
                                        .LayoutBefore = it->copyLayout,
                                        .LayoutAfter  = finalLayout,
                                        .pResource    = resource,
                                        .Subresources = DEFAULT_D3D12_BARRIER_SUBRESOURCE_RANGE,
                                        .Flags        = D3D12_TEXTURE_BARRIER_FLAG_NONE};
    if(finalLayout != it->copyLayout)
    {
      if(it->barrierIndex == ~size_t(0))
      {
        it->barrierIndex = target.imageBarriers.size();
        target.imageBarriers.push_back(barrier);
      }
      else
      {
        target.imageBarriers[it->barrierIndex] = barrier;
      }
      image.stageMask  = syncAfter;
      image.accessMask = accessAfter;
    }
    else
    {
      if(it->barrierIndex != ~size_t(0))
      {
        // a later append keeps the copy layout: drop the earlier transition
        const size_t erased = it->barrierIndex;
        target.imageBarriers.erase(target.imageBarriers.begin() + erased);
        it->barrierIndex = ~size_t(0);
        for(size_t i = 0; i < m_imageStates.size(); i++)
        {
          ImageState& state = m_imageStates[i];
          if(state.barrierIndex != ~size_t(0) && state.barrierIndex > erased)
          {
            state.barrierIndex--;
          }
        }
      }
      image.stageMask  = enableQueueTransfer ? D3D12_BARRIER_SYNC_NONE : D3D12_BARRIER_SYNC_COPY;
      image.accessMask = enableQueueTransfer ? D3D12_BARRIER_ACCESS_NO_ACCESS : D3D12_BARRIER_ACCESS_COPY_DEST;
    }
    image.layout = finalLayout;
  }

  // One copy per layer; the layers' footprints follow each other.
  for(uint32_t layer = 0; layer < subresource.layerCount; layer++)
  {
    ImageCopy copy;
    copy.dst.pResource        = resource;
    copy.dst.Type             = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    copy.dst.SubresourceIndex = image.subresourceIndex(subresource.mipLevel, subresource.baseArrayLayer + layer);
    copy.src.pResource        = stagingBuffer;
    copy.src.Type             = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    copy.src.PlacedFootprint  = {stagingOffset + footprint.layerPitch * layer, footprint.footprint};
    copy.offset               = offset;
    copyBufferImageRegions.push_back(copy);
  }
  stagingSize += footprint.size;
  return S_OK;
}

}  // namespace nvdx12

//--------------------------------------------------------------------------------------------------
// Usage example
//--------------------------------------------------------------------------------------------------
[[maybe_unused]] static void usage_StagingCopyBatch()
{
  nvdx12::ResourceAllocator   allocator;      // EX: initialized with the device and the descriptor heap
  ID3D12GraphicsCommandList7* cmd = nullptr;  // EX: a command list in the recording state
  nvdx12::Buffer              buffer;         // EX: a device buffer of 1024 bytes
  nvdx12::Image               image;          // EX: 64x64 DXGI_FORMAT_R8G8B8A8_UNORM
  std::vector<uint8_t>        bufferData(1024), texels(64 * 64 * 4);

  // Staging owned by the caller: the buffer data, then the image region in
  // placed-footprint layout at a 512-byte aligned offset.
  const nvdx12::ImageSubFootprint footprint   = nvdx12::makeImageSubFootprint(image.format, {64, 64, 1});
  const uint64_t                  imageOffset = D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT * 2;
  nvdx12::Buffer                  staging;
  NVDX12_CHECK(allocator.createBuffer(staging, imageOffset + footprint.size, D3D12_RESOURCE_FLAG_NONE, nvdx12::MemoryUsage::eHostUpload));
  memcpy(staging.mapping, bufferData.data(), bufferData.size());
  nvdx12::writeImageSubFootprint(footprint, staging.mapping + imageOffset, texels.data());

  // The batch records the copies and the layout transitions around them.
  nvdx12::StagingCopyBatch batch;
  batch.enableLayoutBarriers = true;
  batch.addBufferCopy(staging.buffer.Get(), 0, buffer.buffer.Get(), 0, bufferData.size());
  NVDX12_CHECK(batch.addImageCopy(staging.buffer.Get(), imageOffset, image, D3D12_BARRIER_LAYOUT_SHADER_RESOURCE,
                                  footprint.size, {}, {}, {64, 64, 1}));
  batch.cmdCopyAppended(cmd);

  // EX: submit cmd and wait for the GPU before freeing the staging
  allocator.destroyBuffer(staging);
}
