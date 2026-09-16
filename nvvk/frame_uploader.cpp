/*
* Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
* SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
* SPDX-License-Identifier: Apache-2.0
*/

#include <cassert>

#include <volk.h>

#include <nvvk/check_error.hpp>
#include <nvvk/debug_util.hpp>

#include "frame_uploader.hpp"

namespace nvvk {

FrameUploader::~FrameUploader()
{
  assert(m_resourceAllocator == nullptr && "Missing deinit()");
}

VkResult FrameUploader::init(const InitInfo& info)
{
  assert(m_resourceAllocator == nullptr && "Missing deinit()");
  assert(info.allocator);

  VkResult result;

  m_forceCoherentMapping = info.forceCoherentMapping;

  // FrameUploader allocates staging strictly linearly per frame and retires it
  // in the same order once the frame's GPU work completed, which is exactly the
  // access pattern a circular allocator is built for.
  nvvk::BufferCircularAllocator::InitInfo circInitInfo;
  circInitInfo.allocationFlags = VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
  circInitInfo.memoryUsage = info.forceCoherentMapping ? VMA_MEMORY_USAGE_CPU_ONLY : VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
  circInitInfo.usageFlags  = VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT;
  circInitInfo.resourceAllocator = info.allocator;
  circInitInfo.debugName         = info.debugName;
  circInitInfo.blockSize         = info.blockSize;
  circInitInfo.maxAllocatedSize  = info.blockSize * MAX_FLUSH_BLOCKS;

  result = NVVK_FAIL_REPORT(m_bufferCircularAllocator.init(circInitInfo));
  if(result != VK_SUCCESS)
    return result;

  m_resourceAllocator          = info.allocator;
  m_batch.enableLayoutBarriers = true;

  return VK_SUCCESS;
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
  m_resourceAllocator  = nullptr;
}

void FrameUploader::updateFrameSemaphoreState(const nvvk::SemaphoreState& semaphoreState)
{
  m_semaphoreState = semaphoreState;
  m_frameIndex++;
}

void FrameUploader::markBlockForFlush(uint32_t blockIndex)
{
  if(!m_forceCoherentMapping && blockIndex < MAX_FLUSH_BLOCKS)
  {
    m_batchFlushBlockMask |= (1ull << blockIndex);
  }
}

void FrameUploader::flushBatchBlocks()
{
  if(m_batchFlushBlockMask == 0)
  {
    return;
  }

  for(uint32_t blockIndex = 0; blockIndex < MAX_FLUSH_BLOCKS; blockIndex++)
  {
    if(m_batchFlushBlockMask & (1ull << blockIndex))
    {
      const nvvk::Buffer& blockBuffer = m_bufferCircularAllocator.getBlockBuffer(blockIndex);
      m_resourceAllocator->autoFlushBuffer(blockBuffer);
    }
  }
  m_batchFlushBlockMask = 0;
}

VkResult FrameUploader::acquireStagingSpace(nvvk::BufferCircularAllocation& subAllocation,
                                            nvvk::BufferRange&              stagingSpace,
                                            size_t                          dataSize,
                                            const void*                     data)
{
  VkResult result = m_bufferCircularAllocator.subAllocate(subAllocation, dataSize);
  if(result == VK_SUCCESS)
  {
    stagingSpace                        = m_bufferCircularAllocator.subRange(subAllocation);
    StagingAllocation stagingAllocation = {};
    stagingAllocation.semaphoreState    = m_semaphoreState;
    stagingAllocation.subAllocation     = subAllocation;
    m_stagingAllocations.push_back(stagingAllocation);
    if(data)
    {
      memcpy(stagingSpace.mapping, data, dataSize);
      markBlockForFlush(subAllocation.getBlockIndex());
    }
  }

  return result;
}

VkResult FrameUploader::acquireMapping(size_t dataSize, nvvk::BufferRange& mappingSpace, MappingHandle& mappingHandle)
{
  mappingHandle = {};

  if(dataSize == 0)
  {
    mappingSpace = {};
    return VK_SUCCESS;
  }
  nvvk::BufferCircularAllocation subAllocation;
  NVVK_FAIL_RETURN(acquireStagingSpace(subAllocation, mappingSpace, dataSize, nullptr));

  if(subAllocation.getBlockIndex() >= MAX_FLUSH_BLOCKS)
  {
    assert(0 && "FrameUploader supports at most 64 blocks");
    return VK_ERROR_OUT_OF_POOL_MEMORY;
  }

  MappingHandleDetail detail{};
  detail.valid        = 1;
  detail.frameIndex   = m_frameIndex;
  detail.blockIndex   = subAllocation.getBlockIndex();
  mappingHandle.value = detail.handle;

  return VK_SUCCESS;
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
  MappingHandleDetail detail{};
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
  MappingHandleDetail detail{};
  detail.handle = mappingHandle.value;
  assert(detail.valid != 0 && !detail.used);
  assert(detail.frameIndex == m_frameIndex);

  // the handle must not be used again
  mappingHandle = {};
}

VkResult FrameUploader::appendBufferRange(const nvvk::BufferRange& buffer, const void* data, nvvk::SemaphoreState* outSemaphoreState)
{
  if(buffer.range == 0)
  {
    if(outSemaphoreState)
      *outSemaphoreState = {};
    return VK_SUCCESS;
  }

  assert(buffer.buffer);
  assert(data);

  nvvk::BufferCircularAllocation subAllocation;
  nvvk::BufferRange              stagingSpace;
  NVVK_FAIL_RETURN(acquireStagingSpace(subAllocation, stagingSpace, buffer.range, data));

  m_batch.addBufferCopy(stagingSpace.buffer, stagingSpace.offset, buffer.buffer, buffer.offset, buffer.range, true);
  getOutSemaphoreState(outSemaphoreState);

  return VK_SUCCESS;
}

VkResult FrameUploader::appendBufferRangeMappings(size_t                    mappingCount,
                                                  const MappedBufferRanges* ranges,
                                                  MappingHandle&            mappingHandle,
                                                  bool                      commitMapping_,
                                                  nvvk::SemaphoreState*     outSemaphoreState)
{
  assert(mappingHandle);
  if(!mappingHandle)
  {
    return VK_ERROR_UNKNOWN;
  }

  MappingHandleDetail detail{};
  detail.handle = mappingHandle.value;
  assert(detail.valid != 0);
  assert(detail.frameIndex == m_frameIndex);

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

  markBlockForFlush(detail.blockIndex);

  getOutSemaphoreState(outSemaphoreState);
  return VK_SUCCESS;
}

VkResult FrameUploader::appendImageSub(nvvk::Image&                    image,
                                       const VkOffset3D&               offset,
                                       const VkExtent3D&               extent,
                                       const VkImageSubresourceLayers& subresource,
                                       size_t                          dataSize,
                                       const void*                     data,
                                       VkImageLayout                   newLayout,
                                       nvvk::SemaphoreState*           outSemaphoreState)
{
  if(dataSize == 0)
  {
    if(outSemaphoreState)
      *outSemaphoreState = {};
    return VK_SUCCESS;
  }

  assert(image.image);
  assert(data);

  nvvk::BufferCircularAllocation subAllocation;
  nvvk::BufferRange              stagingSpace;
  NVVK_FAIL_RETURN(acquireStagingSpace(subAllocation, stagingSpace, dataSize, data));

  const VkImageSubresourceRange subresourceRange{subresource.aspectMask, subresource.mipLevel, 1,
                                                 subresource.baseArrayLayer, subresource.layerCount};

  m_batch.addImageCopy(stagingSpace.buffer, stagingSpace.offset, image.image, image.descriptor.imageLayout, newLayout,
                       dataSize, subresource, offset, extent, &subresourceRange);
  getOutSemaphoreState(outSemaphoreState);

  return VK_SUCCESS;
}

VkResult FrameUploader::appendImageSubMappings(nvvk::Image&           image,
                                               size_t                 imageSubCount,
                                               const MappedImageSubs* imageSubs,
                                               MappingHandle&         mappingHandle,
                                               bool                   commitMapping_,
                                               VkImageLayout          newLayout,
                                               nvvk::SemaphoreState*  outSemaphoreState)
{
  assert(image.image);
  assert(mappingHandle);
  if(!mappingHandle)
  {
    return VK_ERROR_UNKNOWN;
  }

  MappingHandleDetail detail{};
  detail.handle = mappingHandle.value;
  assert(detail.valid != 0);
  assert(detail.frameIndex == m_frameIndex);

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

  VkImageLayout imageLayout = image.descriptor.imageLayout;
  for(size_t i = 0; i < imageSubCount; i++)
  {
    const MappedImageSubs& sub = imageSubs[i];
    if(sub.mappingSpace.range == 0)
      continue;

    // reset with each sub image to original
    image.descriptor.imageLayout = imageLayout;

    const VkImageSubresourceRange subresourceRange{sub.subresource.aspectMask, sub.subresource.mipLevel, 1,
                                                   sub.subresource.baseArrayLayer, sub.subresource.layerCount};

    m_batch.addImageCopy(sub.mappingSpace.buffer, sub.mappingSpace.offset, image.image, image.descriptor.imageLayout,
                         newLayout, sub.mappingSpace.range, sub.subresource, sub.offset, sub.extent, &subresourceRange);
  }

  markBlockForFlush(detail.blockIndex);

  getOutSemaphoreState(outSemaphoreState);
  return VK_SUCCESS;
}

void FrameUploader::cmdUploadAppended(VkCommandBuffer cmd)
{
  if(m_batch.isAppendedEmpty())
  {
    m_stats = {};
    return;
  }

  if(!m_forceCoherentMapping)
  {
    flushBatchBlocks();
  }

  m_stats.copyBytes = m_batch.stagingSize;
  m_stats.copyCount = 0;
  m_stats.copyCount += m_batch.copyBufferImageRegions.size();
  m_stats.copyCount += m_batch.copyBufferRegions.size();

  m_batch.cmdCopyAppended(cmd);
}

void FrameUploader::releaseCompletedAllocations(bool forceAll)
{
  if(m_resourceAllocator == nullptr)
    return;

  VkDevice device = m_resourceAllocator->getDevice();

  nvvk::SemaphoreStateSignalCache semaphoreCache;

  size_t writeIndex = 0;

  for(size_t readIndex = 0; readIndex < m_stagingAllocations.size(); readIndex++)
  {
    StagingAllocation& allocation = m_stagingAllocations[readIndex];

    if(forceAll || !allocation.semaphoreState.isValid() || (semaphoreCache.testSignaled(device, allocation.semaphoreState)))
    {
      m_bufferCircularAllocator.subFree(allocation.subAllocation);
    }
    else if(readIndex != writeIndex)
    {
      m_stagingAllocations[writeIndex++] = allocation;
    }
    else
    {
      writeIndex++;
    }
  }

  m_stagingAllocations.resize(writeIndex);
}


//--------------------------------------------------------------------------------------------------
// Usage example
//--------------------------------------------------------------------------------------------------
[[maybe_unused]] static void usage_FrameUploader()
{
  // FrameUploader batches staging copies and records them into the same command buffer
  // used for rendering. Uploads are tied to the frame's signal semaphore.

  nvvk::ResourceAllocator resourceAllocator;  // EX. initialize somehow
  nvvk::Buffer            deviceBuffer{};     // EX. create GPU buffer to upload into
  nvvk::Image             deviceImage{};      // EX. create GPU image to upload into

  FrameUploader frameUploader;
  frameUploader.init({.allocator = &resourceAllocator, .debugName = "frameUploads"});

  uint8_t uploadData[4096]{};
  size_t  uploadSize = sizeof(uploadData);

  while(true)
  {
    VkCommandBuffer cmd{};  // per-frame command buffer, setup state etc.

    // recycle staging memory from previous frames once GPU is done
    frameUploader.releaseCompletedAllocations();

    nvvk::SemaphoreInfo  frameSemInfo{};  // EX. from swapchain / application frame signal
    nvvk::SemaphoreState frameSemState = nvvk::SemaphoreState::makeFixed(frameSemInfo);
    frameUploader.updateFrameSemaphoreState(frameSemState);

    // direct upload: copies data into internal staging, queued for cmdUploadAppended
    frameUploader.appendBuffer(deviceBuffer, 0, uploadSize, uploadData);

    // mapping upload: acquire staging, fill on CPU, then append copies without extra memcpy
    {
      nvvk::BufferRange                mappingSpace{};
      UploaderInterface::MappingHandle mappingHandle{};
      if(frameUploader.acquireMapping(uploadSize, mappingSpace, mappingHandle) == VK_SUCCESS)
      {
        memcpy(mappingSpace.mapping, uploadData, uploadSize);
        frameUploader.appendBufferMapping(deviceBuffer, 0, uploadSize, mappingSpace, mappingHandle);
      }
    }

    // image upload via direct data pointer
    {
      VkOffset3D               offset{};
      VkExtent3D               extent{64, 64, 1};
      VkImageSubresourceLayers subresource{VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
      frameUploader.appendImageSub(deviceImage, offset, extent, subresource, uploadSize, uploadData,
                                   VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    }

    // mapping-based image upload
    {
      nvvk::BufferRange                mappingSpace{};
      UploaderInterface::MappingHandle mappingHandle{};
      if(frameUploader.acquireMapping(uploadSize, mappingSpace, mappingHandle) == VK_SUCCESS)
      {
        memcpy(mappingSpace.mapping, uploadData, uploadSize);

        VkOffset3D                         offset{};
        VkExtent3D                         extent{64, 64, 1};
        VkImageSubresourceLayers           subresource{VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        UploaderInterface::MappedImageSubs imageSub{};
        imageSub.offset       = offset;
        imageSub.extent       = extent;
        imageSub.subresource  = subresource;
        imageSub.mappingSpace = mappingSpace;

        frameUploader.appendImageSubMappings(deviceImage, 1, &imageSub, mappingHandle, true, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
      }
    }

    // record all appended copies into the frame command buffer
    frameUploader.cmdUploadAppended(cmd);

    // optional barrier before using uploaded resources on GPU
  }

  // call frameUploader.deinit() on shutdown
}

}  // namespace nvvk
