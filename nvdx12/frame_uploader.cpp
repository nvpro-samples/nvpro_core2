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

#include "frame_uploader.hpp"

#include <cassert>
#include <cstring>

#include <nvutils/logger.hpp>

#include "check_error.hpp"
#include "queue.hpp"

namespace nvdx12 {

// buffer copies only need the copy alignment; image footprints and mappings
// (which may hold footprints) need the placement alignment
static constexpr uint32_t kBufferStagingAlignment = 16;
static constexpr uint32_t kImageStagingAlignment  = D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT;

FrameUploader::~FrameUploader()
{
  assert(m_resourceAllocator == nullptr && "Missing deinit()");
}

HRESULT FrameUploader::init(const InitInfo& info)
{
  assert(m_resourceAllocator == nullptr && "Missing deinit()");
  assert(info.allocator);

  // FrameUploader allocates staging strictly linearly per frame and retires it
  // in the same order once the frame's GPU work completed, which is exactly the
  // access pattern a circular allocator is built for.
  BufferCircularAllocator::InitInfo circInitInfo;
  circInitInfo.resourceAllocator = info.allocator;
  circInitInfo.debugName         = info.debugName;
  circInitInfo.usageFlags        = D3D12_RESOURCE_FLAG_NONE;  // copy source only, no descriptor
  circInitInfo.memoryUsage       = MemoryUsage::eHostUpload;
  circInitInfo.blockSize         = info.blockSize;
  circInitInfo.maxAllocatedSize  = info.blockSize * MAX_BLOCKS;

  HRESULT result = NVDX12_FAIL_REPORT(m_bufferCircularAllocator.init(circInitInfo));
  if(FAILED(result))
  {
    return result;
  }

  m_resourceAllocator          = info.allocator;
  m_batch.enableLayoutBarriers = true;

  return S_OK;
}

void FrameUploader::deinit()
{
  if(m_resourceAllocator == nullptr)
    return;

  releaseCompletedAllocations(true);

  m_batch.reset();
  m_bufferCircularAllocator.deinit();
  m_stagingAllocations = {};
  m_semaphoreState     = {};
  m_stats              = {};
  m_resourceAllocator  = nullptr;
}

void FrameUploader::updateFrameSemaphoreState(const SemaphoreState& semaphoreState)
{
  m_semaphoreState = semaphoreState;
  m_frameIndex++;
}

HRESULT FrameUploader::acquireStagingSpace(BufferCircularAllocation& subAllocation, BufferRange& stagingSpace, size_t dataSize, uint32_t alignment)
{
  HRESULT result = m_bufferCircularAllocator.subAllocate(subAllocation, dataSize, alignment);
  if(SUCCEEDED(result))
  {
    stagingSpace                        = m_bufferCircularAllocator.subRange(subAllocation);
    StagingAllocation stagingAllocation = {};
    stagingAllocation.semaphoreState    = m_semaphoreState;
    stagingAllocation.subAllocation     = subAllocation;
    m_stagingAllocations.push_back(stagingAllocation);
  }
  return result;
}

HRESULT FrameUploader::acquireMapping(size_t dataSize, BufferRange& mappingSpace, MappingHandle& mappingHandle)
{
  mappingHandle = {};

  if(dataSize == 0)
  {
    mappingSpace = {};
    return S_OK;
  }
  BufferCircularAllocation subAllocation;
  NVDX12_FAIL_RETURN(acquireStagingSpace(subAllocation, mappingSpace, dataSize, kImageStagingAlignment));
  // the allocation is rounded up to 16 bytes: hand out the requested size, so that
  // the mapping matches a `dataSize` destination range (appendBufferMapping)
  mappingSpace.range = dataSize;

  MappingHandleDetail detail{};
  detail.valid        = 1;
  detail.frameIndex   = m_frameIndex;
  mappingHandle.value = detail.handle;

  return S_OK;
}

void FrameUploader::commitMapping(MappingHandle& mappingHandle)
{
  // a zero-size `acquireMapping` legitimately yields an invalid handle
  if(!mappingHandle)
  {
    return;
  }

  // Intentionally does not free the staging slice. `m_bufferCircularAllocator` requires
  // strict FIFO frees, and so we ignore the mappingHandle which may come in out of order.
  [[maybe_unused]] MappingHandleDetail detail{};
  detail.handle = mappingHandle.value;
  assert(detail.valid != 0);
  assert(detail.frameIndex == m_frameIndex);

  // the handle must not be used again
  mappingHandle = {};
}

void FrameUploader::releaseMapping(MappingHandle& mappingHandle)
{
  // a zero-size `acquireMapping` legitimately yields an invalid handle
  if(!mappingHandle)
  {
    return;
  }

  // Intentionally does not free the staging slice. `m_bufferCircularAllocator` requires
  // strict FIFO frees, and so we ignore the mappingHandle which may come in out of order.
  [[maybe_unused]] MappingHandleDetail detail{};
  detail.handle = mappingHandle.value;
  assert(detail.valid != 0 && !detail.used);
  assert(detail.frameIndex == m_frameIndex);

  // the handle must not be used again
  mappingHandle = {};
}

HRESULT FrameUploader::appendBufferRange(const BufferRange& buffer, const void* data, SemaphoreState* outSemaphoreState)
{
  if(buffer.range == 0)
  {
    if(outSemaphoreState)
      *outSemaphoreState = {};
    return S_OK;
  }

  assert(buffer.buffer);
  assert(data);
  if(!buffer.buffer || !data)
  {
    return E_INVALIDARG;
  }

  BufferCircularAllocation subAllocation;
  BufferRange              stagingSpace;
  NVDX12_FAIL_RETURN(acquireStagingSpace(subAllocation, stagingSpace, buffer.range, kBufferStagingAlignment));
  memcpy(stagingSpace.mapping, data, buffer.range);

  m_batch.addBufferCopy(stagingSpace.buffer, stagingSpace.offset, buffer.buffer, buffer.offset, buffer.range);
  getOutSemaphoreState(outSemaphoreState);

  return S_OK;
}

HRESULT FrameUploader::appendBufferRangeMappings(size_t                    mappingCount,
                                                 const MappedBufferRanges* ranges,
                                                 MappingHandle&            mappingHandle,
                                                 bool                      commitMapping_,
                                                 SemaphoreState*           outSemaphoreState)
{
  assert(mappingHandle);
  if(!mappingHandle)
  {
    return E_INVALIDARG;
  }

  MappingHandleDetail detail{};
  detail.handle = mappingHandle.value;
  // a handle from an earlier frame refers to staging that may already be recycled
  assert(detail.valid != 0 && detail.frameIndex == m_frameIndex);
  if(detail.valid == 0 || detail.frameIndex != m_frameIndex)
  {
    return E_INVALIDARG;
  }

  // the staging slice itself is always retired per-frame, not per-mapping (see commitMapping()),
  // the commit only invalidates the handle
  if(commitMapping_)
  {
    mappingHandle = {};
  }
  else
  {
    // tag as used, the mapping is now referenced by the batch and must not be released
    detail.used         = 1;
    mappingHandle.value = detail.handle;
  }

  for(size_t i = 0; i < mappingCount; i++)
  {
    const MappedBufferRanges& range = ranges[i];
    if(range.bufferRange.range == 0)
      continue;

    assert(range.bufferRange.buffer);
    assert(range.mappingRange.range == range.bufferRange.range);

    m_batch.addBufferCopy(range.mappingRange.buffer, range.mappingRange.offset, range.bufferRange.buffer,
                          range.bufferRange.offset, range.bufferRange.range);
  }

  getOutSemaphoreState(outSemaphoreState);
  return S_OK;
}

HRESULT FrameUploader::appendImageSub(Image&                        image,
                                      const Offset3D&               offset,
                                      const Extent3D&               extent,
                                      const ImageSubresourceLayers& subresource,
                                      size_t                        dataSize,
                                      const void*                   data,
                                      D3D12_BARRIER_LAYOUT          newLayout,
                                      SemaphoreState*               outSemaphoreState)
{
  if(dataSize == 0)
  {
    if(outSemaphoreState)
      *outSemaphoreState = {};
    return S_OK;
  }

  assert(image.image);
  assert(data);

  if(!data)
  {
    LOGE("FrameUploader::appendImageSub: null data\n");
    return E_INVALIDARG;
  }
  const ImageSubFootprint footprint = makeImageSubFootprint(image.format, extent, subresource.layerCount);
  if(!footprint)
  {
    LOGE("FrameUploader::appendImageSub: unsupported format %d or empty region\n", int(image.format));
    return E_INVALIDARG;
  }
  if(dataSize < footprint.tightSize)
  {
    LOGE("FrameUploader::appendImageSub: %zu bytes provided, %llu needed\n", dataSize,
         static_cast<unsigned long long>(footprint.tightSize));
    return E_INVALIDARG;
  }
  NVDX12_FAIL_RETURN(m_batch.checkImageCopy(image, subresource, offset, extent));

  // the staging holds the region re-pitched into D3D12's placed footprint
  BufferCircularAllocation subAllocation;
  BufferRange              stagingSpace;
  NVDX12_FAIL_RETURN(acquireStagingSpace(subAllocation, stagingSpace, footprint.size, kImageStagingAlignment));
  writeImageSubFootprint(footprint, stagingSpace.mapping, data, subresource.layerCount);

  NVDX12_FAIL_RETURN(m_batch.addImageCopy(stagingSpace.buffer, stagingSpace.offset, image, newLayout,
                                          stagingSpace.range, subresource, offset, extent));
  getOutSemaphoreState(outSemaphoreState);

  return S_OK;
}

HRESULT FrameUploader::appendImageSubMappings(Image&                 image,
                                              size_t                 imageSubCount,
                                              const MappedImageSubs* imageSubs,
                                              MappingHandle&         mappingHandle,
                                              bool                   commitMapping_,
                                              D3D12_BARRIER_LAYOUT   newLayout,
                                              SemaphoreState*        outSemaphoreState)
{
  assert(image.image);
  assert(mappingHandle);
  if(!mappingHandle)
  {
    return E_INVALIDARG;
  }

  MappingHandleDetail detail{};
  detail.handle = mappingHandle.value;
  // a handle from an earlier frame refers to staging that may already be recycled
  assert(detail.valid != 0 && detail.frameIndex == m_frameIndex);
  if(detail.valid == 0 || detail.frameIndex != m_frameIndex)
  {
    return E_INVALIDARG;
  }

  // validate every sub first, so that a failure appends nothing and leaves the handle as it was
  NVDX12_FAIL_RETURN(m_batch.checkImageSubCopies(image, imageSubCount, imageSubs));

  // the staging slice itself is always retired per-frame, not per-mapping (see commitMapping()),
  // the commit only invalidates the handle
  if(commitMapping_)
  {
    mappingHandle = {};
  }
  else
  {
    // tag as used, the mapping is now referenced by the batch and must not be released
    detail.used         = 1;
    mappingHandle.value = detail.handle;
  }

  // StagingCopyBatch transitions the whole image once per batch, so every
  // sub-image shares the barrier and ends in the same layout.
  for(size_t i = 0; i < imageSubCount; i++)
  {
    const MappedImageSubs& sub = imageSubs[i];
    if(sub.mappingSpace.range == 0)
      continue;

    NVDX12_FAIL_RETURN(m_batch.addImageCopy(sub.mappingSpace.buffer, sub.mappingSpace.offset, image, newLayout,
                                            sub.mappingSpace.range, sub.subresource, sub.offset, sub.extent));
  }

  getOutSemaphoreState(outSemaphoreState);
  return S_OK;
}

void FrameUploader::cmdUploadAppended(ID3D12GraphicsCommandList7* cmd)
{
  if(m_batch.isAppendedEmpty())
  {
    m_stats = {};
    return;
  }

  m_stats.copyBytes = m_batch.stagingSize;
  m_stats.copyCount = m_batch.copyBufferImageRegions.size() + m_batch.copyBufferRegions.size();

  m_batch.cmdCopyAppended(cmd);
}

void FrameUploader::releaseCompletedAllocations(bool forceAll)
{
  if(m_resourceAllocator == nullptr)
    return;

  SemaphoreStateSignalCache semaphoreCache;

  // The circular allocator frees in allocation order: stop at the first
  // allocation still in use, even if a later one (another fence, or no
  // state) could go.
  size_t freed = 0;
  for(; freed < m_stagingAllocations.size(); freed++)
  {
    StagingAllocation& allocation = m_stagingAllocations[freed];
    if(!forceAll && allocation.semaphoreState.isValid() && !semaphoreCache.testSignaled(allocation.semaphoreState))
    {
      break;
    }
    m_bufferCircularAllocator.subFree(allocation.subAllocation);
  }
  m_stagingAllocations.erase(m_stagingAllocations.begin(), m_stagingAllocations.begin() + freed);
}

}  // namespace nvdx12

//--------------------------------------------------------------------------------------------------
// Usage example
//--------------------------------------------------------------------------------------------------
[[maybe_unused]] static void usage_FrameUploader()
{
  // FrameUploader batches staging copies and records them into the same command list
  // used for rendering. Uploads are tied to the frame's SemaphoreState.

  nvdx12::ResourceAllocator resourceAllocator;  // EX. initialize somehow
  nvdx12::QueueTimeline*    queue{};            // EX. the frame's queue
  nvdx12::Buffer            deviceBuffer{};     // EX. create GPU buffer to upload into
  nvdx12::Image             deviceImage{};      // EX. create a 64x64 RGBA8 GPU image to upload into

  nvdx12::FrameUploader frameUploader;
  frameUploader.init({.allocator = &resourceAllocator, .debugName = "frameUploads"});

  uint8_t uploadData[64 * 64 * 4]{};
  size_t  uploadSize = 4096;

  while(true)
  {
    ID3D12GraphicsCommandList7* cmd{};  // per-frame command list, recording

    // recycle staging memory from previous frames once GPU is done
    frameUploader.releaseCompletedAllocations();

    // the state this frame's submit completes
    nvdx12::SemaphoreState frameSemState = queue->createNextFixedSemaphoreState();
    frameUploader.updateFrameSemaphoreState(frameSemState);

    // direct upload: copies data into internal staging, queued for cmdUploadAppended
    frameUploader.appendBuffer(deviceBuffer, 0, uploadSize, uploadData);

    // mapping upload: acquire staging, fill on CPU, then append copies without extra memcpy
    {
      nvdx12::BufferRange                      mappingSpace{};
      nvdx12::UploaderInterface::MappingHandle mappingHandle{};
      if(frameUploader.acquireMapping(uploadSize, mappingSpace, mappingHandle) == S_OK)
      {
        memcpy(mappingSpace.mapping, uploadData, uploadSize);
        frameUploader.appendBufferMapping(deviceBuffer, 0, uploadSize, mappingSpace, mappingHandle);
      }
    }

    // image upload via direct data pointer (tightly packed)
    {
      nvdx12::Offset3D               offset{};
      nvdx12::Extent3D               extent{64, 64, 1};
      nvdx12::ImageSubresourceLayers subresource{.mipLevel = 0, .baseArrayLayer = 0, .layerCount = 1};
      frameUploader.appendImageSub(deviceImage, offset, extent, subresource, sizeof(uploadData), uploadData,
                                   D3D12_BARRIER_LAYOUT_SHADER_RESOURCE);
    }

    // mapping-based image upload: the mapping holds D3D12's placed footprint
    {
      nvdx12::Extent3D          extent{64, 64, 1};
      nvdx12::ImageSubFootprint footprint = nvdx12::makeImageSubFootprint(deviceImage.format, extent);

      nvdx12::BufferRange                      mappingSpace{};
      nvdx12::UploaderInterface::MappingHandle mappingHandle{};
      if(frameUploader.acquireMapping(footprint.size, mappingSpace, mappingHandle) == S_OK)
      {
        nvdx12::writeImageSubFootprint(footprint, mappingSpace.mapping, uploadData);

        nvdx12::UploaderInterface::MappedImageSubs imageSub{};
        imageSub.offset       = {};
        imageSub.extent       = extent;
        imageSub.subresource  = {.mipLevel = 0, .baseArrayLayer = 0, .layerCount = 1};
        imageSub.mappingSpace = mappingSpace;

        frameUploader.appendImageSubMappings(deviceImage, 1, &imageSub, mappingHandle, true, D3D12_BARRIER_LAYOUT_SHADER_RESOURCE);
      }
    }

    // record all appended copies into the frame command list
    frameUploader.cmdUploadAppended(cmd);

    // EX. close and submit cmd with frameSemState: queue->submit(cmd, frameSemState)
  }

  // call frameUploader.deinit() on shutdown, after the GPU finished
}
