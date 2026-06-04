/*
* Copyright (c) 2025-2026, NVIDIA CORPORATION.  All rights reserved.
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
* SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
* SPDX-License-Identifier: Apache-2.0
*/

#include "staging.hpp"
#include "barriers.hpp"
#include "check_error.hpp"
#include "debug_util.hpp"

namespace nvvk {

namespace {

inline bool imageLayoutAllowsCopy(VkImageLayout layout)
{
  return layout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL || layout == VK_IMAGE_LAYOUT_GENERAL
         || layout == VK_IMAGE_LAYOUT_SHARED_PRESENT_KHR;
}

}  // namespace


//////////////////////////////////////////////////////////////////////////


bool StagingCopyBatch::isAppendedEmpty() const
{
  return copyBufferImageInfos.empty() && copyBufferInfos.empty();
}

void StagingCopyBatch::cmdCopyAppended(VkCommandBuffer cmd)
{
  if(enableLayoutBarriers)
  {
    pre.cmdPipelineBarrier(cmd, 0);
  }

  size_t regionOffset = 0;
  for(size_t i = 0; i < copyBufferInfos.size(); i++)
  {
    copyBufferInfos[i].pRegions = &copyBufferRegions[regionOffset];
    regionOffset += copyBufferInfos[i].regionCount;
    vkCmdCopyBuffer2(cmd, &copyBufferInfos[i]);
  }
  regionOffset = 0;
  for(size_t i = 0; i < copyBufferImageInfos.size(); i++)
  {
    copyBufferImageInfos[i].pRegions = &copyBufferImageRegions[regionOffset];
    regionOffset += copyBufferImageInfos[i].regionCount;
    vkCmdCopyBufferToImage2(cmd, &copyBufferImageInfos[i]);
  }

  if(enableLayoutBarriers || enableOwnerBarriers)
  {
    post.cmdPipelineBarrier(cmd, 0);
  }

  reset();
}

void StagingCopyBatch::reset()
{
  // let's use clear rather than {};
  // to avoid heap allocations
  copyBufferImageInfos.clear();
  copyBufferImageRegions.clear();
  copyBufferInfos.clear();
  copyBufferRegions.clear();
  pre.clear();
  post.clear();
  acquire.clear();
  stagingSize = 0;
}

void StagingCopyBatch::addBufferCopy(VkBuffer     stagingBuffer,
                                     VkDeviceSize stagingOffset,
                                     VkBuffer     buffer,
                                     VkDeviceSize bufferOffset,
                                     VkDeviceSize dataSize,
                                     bool         allowOwnerBarrier)
{
  VkBufferCopy2 copyRegionInfo{
      .sType     = VK_STRUCTURE_TYPE_BUFFER_COPY_2,
      .srcOffset = stagingOffset,
      .dstOffset = bufferOffset,
      .size      = dataSize,
  };

  VkCopyBufferInfo2 copyBufferInfo{
      .sType       = VK_STRUCTURE_TYPE_COPY_BUFFER_INFO_2,
      .srcBuffer   = stagingBuffer,
      .dstBuffer   = buffer,
      .regionCount = 1,
      .pRegions    = nullptr,  // set when calling `cmdCopyAppended`
  };

  stagingSize += dataSize;
  copyBufferRegions.push_back(copyRegionInfo);
  copyBufferInfos.push_back(copyBufferInfo);

  if(allowOwnerBarrier && enableOwnerBarriers)
  {
    addOwnerBufferBarrier(buffer, bufferOffset, dataSize);
  }
}

void StagingCopyBatch::addOwnerBufferBarrier(VkBuffer buffer, VkDeviceSize offset, VkDeviceSize size)
{
  VkBufferMemoryBarrier2 bufferBarrier = {
      .sType               = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
      .srcStageMask        = VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT,
      .srcAccessMask       = VK_ACCESS_2_TRANSFER_WRITE_BIT,
      .srcQueueFamilyIndex = srcQueueFamilyIndex,
      .dstQueueFamilyIndex = dstQueueFamilyIndex,
      .buffer              = buffer,
      .offset              = offset,
      .size                = size,
  };

  // release barrier
  post.bufferBarriers.push_back(bufferBarrier);

  bufferBarrier.srcStageMask  = VK_PIPELINE_STAGE_2_NONE;
  bufferBarrier.srcAccessMask = VK_ACCESS_2_NONE;
  bufferBarrier.dstStageMask  = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
  bufferBarrier.dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT;

  // acquire barrier
  acquire.bufferBarriers.push_back(bufferBarrier);
}

void StagingCopyBatch::addImageCopy(VkBuffer                        stagingBuffer,
                                    VkDeviceSize                    stagingOffset,
                                    VkImage                         image,
                                    VkImageLayout&                  imageLayout,
                                    VkImageLayout                   newLayout,
                                    size_t                          dataSize,
                                    const VkImageSubresourceLayers& subresource,
                                    const VkOffset3D&               offset,
                                    const VkExtent3D&               extent,
                                    const VkImageSubresourceRange*  subresourceRange)
{
  const VkImageLayout dstImageLayout = handlePreImageBarrier(image, imageLayout, newLayout, subresourceRange);

  const VkBufferImageCopy2 copyBufferImageRegion{
      .sType             = VK_STRUCTURE_TYPE_BUFFER_IMAGE_COPY_2,
      .bufferOffset      = stagingOffset,
      .bufferRowLength   = 0,  // tightly packed
      .bufferImageHeight = 0,  // tightly packed
      .imageSubresource  = subresource,
      .imageOffset       = offset,
      .imageExtent       = extent,
  };

  VkCopyBufferToImageInfo2 copyBufferToImageInfo{
      .sType          = VK_STRUCTURE_TYPE_COPY_BUFFER_TO_IMAGE_INFO_2,
      .srcBuffer      = stagingBuffer,
      .dstImage       = image,
      .dstImageLayout = dstImageLayout,
      .regionCount    = 1,
      .pRegions       = nullptr,  // set when calling `cmdCopyAppended`
  };

  stagingSize += dataSize;
  copyBufferImageRegions.emplace_back(copyBufferImageRegion);
  copyBufferImageInfos.emplace_back(copyBufferToImageInfo);

  imageLayout = handlePostImageBarrier(image, dstImageLayout, imageLayout, newLayout, subresourceRange);
}

VkImageLayout StagingCopyBatch::handlePreImageBarrier(VkImage                        image,
                                                      VkImageLayout                  currentLayout,
                                                      VkImageLayout                  newLayout,
                                                      const VkImageSubresourceRange* subresourceRange)
{
  const bool currentAllowsCopy = imageLayoutAllowsCopy(currentLayout);
  const bool newAllowsCopy     = imageLayoutAllowsCopy(newLayout);

  VkImageLayout copyImageLayout = currentLayout;

  if(enableLayoutBarriers && !currentAllowsCopy)
  {
    copyImageLayout = newAllowsCopy ? newLayout : VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;

    ImageMemoryBarrierParams params{.image = image, .oldLayout = currentLayout, .newLayout = copyImageLayout};
    if(subresourceRange)
    {
      params.subresourceRange = *subresourceRange;
    }

    VkImageMemoryBarrier2 barrier = makeImageMemoryBarrier(params);
    if(enableLayoutBarrierMask)
    {
      applyImageBarrierMask(barrier);
    }
    pre.imageBarriers.push_back(barrier);
  }

  return copyImageLayout;
}

void StagingCopyBatch::applyImageBarrierMask(VkImageMemoryBarrier2& barrier)
{
  barrier.dstAccessMask &= layoutBarrierAccessMask;
  barrier.srcAccessMask &= layoutBarrierAccessMask;
  barrier.dstStageMask &= layoutBarrierPipelineStageMask;
  barrier.srcStageMask &= layoutBarrierPipelineStageMask;
}

VkImageLayout StagingCopyBatch::handlePostImageBarrier(VkImage                        image,
                                                       VkImageLayout                  currentLayout,
                                                       VkImageLayout                  originalLayout,
                                                       VkImageLayout                  newLayout,
                                                       const VkImageSubresourceRange* subresourceRange)
{
  newLayout = newLayout != VK_IMAGE_LAYOUT_UNDEFINED ? newLayout : originalLayout;

  // If our format was not copyable, we temporarily transitioned into a copyable format, and
  // may need to transition back to original.
  // Or we have a new layout that we want to transition to, and we haven't transitioned into it already.
  const bool requiresLayoutTransition = (enableLayoutBarriers && (newLayout != currentLayout));

  if(enableOwnerBarriers || requiresLayoutTransition)
  {

    ImageMemoryBarrierParams params{.image = image, .oldLayout = currentLayout, .newLayout = newLayout};
    if(subresourceRange)
    {
      params.subresourceRange = *subresourceRange;
    }

    VkImageMemoryBarrier2 barrier = makeImageMemoryBarrier(params);
    if(enableOwnerBarriers)
    {
      barrier.srcQueueFamilyIndex = srcQueueFamilyIndex;
      barrier.dstQueueFamilyIndex = dstQueueFamilyIndex;

      VkImageMemoryBarrier2 releaseBarrier = barrier;
      VkImageMemoryBarrier2 acquireBarrier = barrier;

      releaseBarrier.srcStageMask  = VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT;
      releaseBarrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
      releaseBarrier.dstStageMask  = VK_PIPELINE_STAGE_2_NONE;
      releaseBarrier.dstAccessMask = VK_ACCESS_2_NONE;
      // handle layout change at acquisition / on destination queue
      releaseBarrier.newLayout = releaseBarrier.oldLayout;

      acquireBarrier.srcStageMask  = VK_PIPELINE_STAGE_2_NONE;
      acquireBarrier.srcAccessMask = VK_ACCESS_2_NONE;

      post.imageBarriers.push_back(releaseBarrier);
      acquire.imageBarriers.push_back(acquireBarrier);
    }
    else
    {
      if(enableLayoutBarrierMask)
      {
        applyImageBarrierMask(barrier);
      }

      post.imageBarriers.push_back(barrier);
    }
  }

  return newLayout;
}

//////////////////////////////////////////////////////////////////////////

StagingUploader::StagingUploader(StagingUploader&& other) noexcept
{
  {
    std::swap(m_batch, other.m_batch);
    std::swap(m_stagingResourcesSize, other.m_stagingResourcesSize);
    std::swap(m_resourceAllocator, other.m_resourceAllocator);
    std::swap(m_stagingResources, other.m_stagingResources);
    std::swap(m_batchStagingCount, other.m_batchStagingCount);
    std::swap(m_batchRequiresFlush, other.m_batchRequiresFlush);
    std::swap(m_forceCoherentMapping, other.m_forceCoherentMapping);
  }
}

nvvk::StagingUploader& StagingUploader::operator=(StagingUploader&& other) noexcept
{
  if(this != &other)
  {
    assert(m_resourceAllocator == nullptr && "deinit not called prior move assignment");

    std::swap(m_batch, other.m_batch);
    std::swap(m_stagingResourcesSize, other.m_stagingResourcesSize);
    std::swap(m_resourceAllocator, other.m_resourceAllocator);
    std::swap(m_stagingResources, other.m_stagingResources);
    std::swap(m_batchStagingCount, other.m_batchStagingCount);
    std::swap(m_batchRequiresFlush, other.m_batchRequiresFlush);
    std::swap(m_forceCoherentMapping, other.m_forceCoherentMapping);
  }
  return *this;
}

void StagingUploader::init(ResourceAllocator* resourceAllocator, bool enableLayoutBarriers, bool forceCoherentMapping)
{
  assert(m_resourceAllocator == nullptr);
  m_resourceAllocator          = resourceAllocator;
  m_batch.enableLayoutBarriers = enableLayoutBarriers;
  m_forceCoherentMapping       = forceCoherentMapping;
  m_batchRequiresFlush         = false;
}

void StagingUploader::deinit()
{
  if(m_resourceAllocator != nullptr)
  {
    releaseStaging(true);
    assert(m_stagingResources.empty() && m_stagingResourcesSize == 0);  // must have released all staged uploads
  }
  m_resourceAllocator = nullptr;
  m_batchStagingCount = 0;
  m_batch.reset();
}

void StagingUploader::setEnableLayoutBarriers(bool enableLayoutBarriers)
{
  m_batch.enableLayoutBarriers = enableLayoutBarriers;
}

void StagingUploader::setEnableOwnerBarriers(bool enableOwnerBarriers, uint32_t srcQueueFamilyIndex, uint32_t dstQueueFamilyIndex)
{
  m_batch.enableOwnerBarriers = enableOwnerBarriers;
  m_batch.srcQueueFamilyIndex = srcQueueFamilyIndex;
  m_batch.dstQueueFamilyIndex = dstQueueFamilyIndex;
}

nvvk::ResourceAllocator* StagingUploader::getResourceAllocator()
{
  assert(m_resourceAllocator);
  return m_resourceAllocator;
}

VkResult StagingUploader::acquireStagingSpace(BufferRange& stagingSpace, size_t dataSize, const void* data, const SemaphoreState& semaphoreState)
{
  StagingResource stagingResource;
  stagingResource.semaphoreState = semaphoreState;

  // VMA_MEMORY_USAGE_CPU_ONLY staging memory is meant to not cost additional device memory
  //
  // VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT staging memory is filled sequentially
  // VMA_ALLOCATION_CREATE_MAPPED_BIT staging memory is filled through pointer access
  //
  // VK_MEMORY_PROPERTY_HOST_COHERENT_BIT we want to avoid having to call "vkFlushMappedMemoryRanges"
  //
  // As of writing VMA doesn't have a simple usage that guarantees VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
  // (only the deprecated VMA_MEMORY_USAGE_CPU_ONLY did)

  VmaAllocationCreateInfo allocInfo = {
      .flags         = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT,
      .usage         = m_forceCoherentMapping ? VMA_MEMORY_USAGE_CPU_ONLY : VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
      .requiredFlags = m_forceCoherentMapping ? VK_MEMORY_PROPERTY_HOST_COHERENT_BIT : 0u,
  };

  const VkBufferUsageFlags2CreateInfo bufferUsageFlags2CreateInfo{
      .sType = VK_STRUCTURE_TYPE_BUFFER_USAGE_FLAGS_2_CREATE_INFO,
      .usage = VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT_KHR | VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT,
  };

  const VkBufferCreateInfo bufferInfo{
      .sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .pNext       = &bufferUsageFlags2CreateInfo,
      .size        = dataSize,
      .usage       = 0,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
  };

  // Create a staging buffer
  NVVK_FAIL_RETURN(m_resourceAllocator->createBuffer(stagingResource.buffer, bufferInfo, allocInfo));
  NVVK_DBG_NAME(stagingResource.buffer.buffer);

  if(!stagingResource.buffer.mapping)
  {
    m_resourceAllocator->destroyBuffer(stagingResource.buffer);
    return VK_ERROR_MEMORY_MAP_FAILED;
  }

  if(data)
  {
    memcpy(stagingResource.buffer.mapping, data, dataSize);
  }

  m_stagingResourcesSize += dataSize;
  m_stagingResources.emplace_back(stagingResource);
  m_batchStagingCount++;

  if(!m_forceCoherentMapping && m_resourceAllocator->isNonCoherentlyMapped(stagingResource.buffer))
  {
    m_batchRequiresFlush = true;
  }

  stagingSpace.buffer  = stagingResource.buffer.buffer;
  stagingSpace.offset  = 0;
  stagingSpace.range   = dataSize;
  stagingSpace.address = stagingResource.buffer.address;
  stagingSpace.mapping = stagingResource.buffer.mapping;

  m_batch.layoutBarrierAccessMask        = VK_ACCESS_2_TRANSFER_WRITE_BIT | VK_ACCESS_2_TRANSFER_READ_BIT;
  m_batch.layoutBarrierPipelineStageMask = VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT;

  return VK_SUCCESS;
}

bool StagingUploader::isAppendedEmpty() const
{
  return m_batch.isAppendedEmpty();
}

void StagingUploader::beginTransferOnly()
{
  m_batch.enableLayoutBarrierMask = true;
}

VkResult StagingUploader::appendBuffer(const nvvk::Buffer&   buffer,
                                       VkDeviceSize          bufferOffset,
                                       VkDeviceSize          dataSize,
                                       const void*           data,
                                       const SemaphoreState& semaphoreState)
{
  // allow empty without throwing error
  if(dataSize == 0)
  {
    return VK_SUCCESS;
  }

  if(dataSize == VK_WHOLE_SIZE)
  {
    dataSize = buffer.bufferSize;
  }

  assert(data);
  assert(bufferOffset + dataSize <= buffer.bufferSize);
  assert(buffer.buffer);

  if(buffer.mapping)
  {
    memcpy(buffer.mapping + bufferOffset, data, dataSize);
  }
  else
  {
    BufferRange stagingSpace;
    NVVK_FAIL_RETURN(acquireStagingSpace(stagingSpace, dataSize, data, semaphoreState));

    m_batch.addBufferCopy(stagingSpace.buffer, stagingSpace.offset, buffer.buffer, bufferOffset, dataSize);
  }

  return VK_SUCCESS;
}

VkResult StagingUploader::appendBufferRange(const nvvk::BufferRange& bufferRange, const void* data, const SemaphoreState& semaphoreState)
{
  // allow empty without throwing error
  if(bufferRange.range == 0)
  {
    return VK_SUCCESS;
  }

  assert(data);
  assert(bufferRange.buffer);

  if(bufferRange.mapping)
  {
    memcpy(bufferRange.mapping, data, bufferRange.range);
  }
  else
  {
    BufferRange stagingSpace;
    NVVK_FAIL_RETURN(acquireStagingSpace(stagingSpace, bufferRange.range, data, semaphoreState));

    m_batch.addBufferCopy(stagingSpace.buffer, stagingSpace.offset, bufferRange.buffer, bufferRange.offset, bufferRange.range);
  }

  return VK_SUCCESS;
}

VkResult StagingUploader::appendBufferMapping(const nvvk::Buffer&   buffer,
                                              VkDeviceSize          bufferOffset,
                                              VkDeviceSize          dataSize,
                                              void*&                uploadMapping,
                                              const SemaphoreState& semaphoreState)
{
  uploadMapping = nullptr;

  // allow empty without throwing error
  if(dataSize == 0)
  {
    return VK_SUCCESS;
  }

  if(dataSize == VK_WHOLE_SIZE)
  {
    dataSize = buffer.bufferSize;
  }

  assert(buffer.buffer);
  assert(bufferOffset + dataSize <= buffer.bufferSize);

  if(buffer.mapping)
  {
    uploadMapping = buffer.mapping + bufferOffset;

    return VK_SUCCESS;
  }
  else
  {
    BufferRange stagingSpace;
    NVVK_FAIL_RETURN(acquireStagingSpace(stagingSpace, dataSize, nullptr, semaphoreState));

    uploadMapping = stagingSpace.mapping;

    m_batch.addBufferCopy(stagingSpace.buffer, stagingSpace.offset, buffer.buffer, bufferOffset, dataSize);

    return VK_SUCCESS;
  }
}

VkResult StagingUploader::appendBufferRangeMapping(const nvvk::BufferRange& bufferRange, void*& uploadMapping, const SemaphoreState& semaphoreState)
{
  uploadMapping = nullptr;

  // allow empty without throwing error
  if(bufferRange.range == 0)
  {
    return VK_SUCCESS;
  }

  assert(bufferRange.buffer);

  if(bufferRange.mapping)
  {
    uploadMapping = bufferRange.mapping;

    return VK_SUCCESS;
  }
  else
  {
    BufferRange stagingSpace;
    NVVK_FAIL_RETURN(acquireStagingSpace(stagingSpace, bufferRange.range, nullptr, semaphoreState));

    uploadMapping = stagingSpace.mapping;

    m_batch.addBufferCopy(stagingSpace.buffer, stagingSpace.offset, bufferRange.buffer, bufferRange.offset, bufferRange.range);

    return VK_SUCCESS;
  }
}

VkResult StagingUploader::appendLargeBuffer(const nvvk::LargeBuffer& buffer,
                                            VkDeviceSize             bufferOffset,
                                            VkDeviceSize             dataSize,
                                            const void*              data,
                                            const SemaphoreState&    semaphoreState,
                                            VkDeviceSize             chunkSize)
{
  if(dataSize == 0)
  {
    return VK_SUCCESS;
  }

  assert(data);
  const uint8_t* srcData   = static_cast<const uint8_t*>(data);
  VkDeviceSize   remaining = dataSize;
  VkDeviceSize   srcOffset = 0;
  VkDeviceSize   dstOffset = bufferOffset;

  // Upload in chunks (default 256MB) to avoid staging buffer allocation issues
  while(remaining > 0)
  {
    VkDeviceSize currentChunkSize = std::min(remaining, chunkSize);

    nvvk::BufferRange stagingSpace;
    NVVK_FAIL_RETURN(acquireStagingSpace(stagingSpace, currentChunkSize, srcData + srcOffset, semaphoreState));

    m_batch.addBufferCopy(stagingSpace.buffer, stagingSpace.offset, buffer.buffer, dstOffset, currentChunkSize, false);

    srcOffset += currentChunkSize;
    dstOffset += currentChunkSize;
    remaining -= currentChunkSize;
  }

  if(m_batch.enableOwnerBarriers)
  {
    m_batch.addOwnerBufferBarrier(buffer.buffer, bufferOffset, dataSize);
  }

  return VK_SUCCESS;
}

VkResult StagingUploader::appendLargeBufferMapping(const nvvk::LargeBuffer& buffer,
                                                   VkDeviceSize             bufferOffset,
                                                   VkDeviceSize             dataSize,
                                                   void*&                   uploadMapping,
                                                   const SemaphoreState&    semaphoreState)
{
  uploadMapping = nullptr;

  if(dataSize == 0)
  {
    return VK_SUCCESS;
  }

  nvvk::BufferRange stagingSpace;
  NVVK_FAIL_RETURN(acquireStagingSpace(stagingSpace, dataSize, nullptr, semaphoreState));

  uploadMapping = stagingSpace.mapping;

  m_batch.addBufferCopy(stagingSpace.buffer, stagingSpace.offset, buffer.buffer, bufferOffset, dataSize, false);

  if(m_batch.enableOwnerBarriers)
  {
    m_batch.addOwnerBufferBarrier(buffer.buffer, bufferOffset, dataSize);
  }

  return VK_SUCCESS;
}

VkResult StagingUploader::appendImage(nvvk::Image& image, size_t dataSize, const void* data, VkImageLayout newLayout, const SemaphoreState& semaphoreState)
{
  if(dataSize == 0)
  {
    return VK_SUCCESS;
  }

  assert(data);

  BufferRange stagingSpace;
  NVVK_FAIL_RETURN(acquireStagingSpace(stagingSpace, dataSize, data, semaphoreState));

  m_batch.addImageCopy(stagingSpace.buffer, stagingSpace.offset, image.image, image.descriptor.imageLayout, newLayout,
                       dataSize, {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .layerCount = 1}, {0, 0, 0}, image.extent);

  return VK_SUCCESS;
}

VkResult StagingUploader::appendImageMapping(nvvk::Image&          image,
                                             size_t                dataSize,
                                             void*&                uploadMapping,
                                             VkImageLayout         newLayout /*= VK_IMAGE_LAYOUT_UNDEFINED*/,
                                             const SemaphoreState& semaphoreState /*= {}*/)
{
  uploadMapping = nullptr;

  if(dataSize == 0)
  {
    return VK_SUCCESS;
  }

  BufferRange stagingSpace;
  NVVK_FAIL_RETURN(acquireStagingSpace(stagingSpace, dataSize, nullptr, semaphoreState));

  uploadMapping = stagingSpace.mapping;

  m_batch.addImageCopy(stagingSpace.buffer, stagingSpace.offset, image.image, image.descriptor.imageLayout, newLayout,
                       dataSize, {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .layerCount = 1}, {0, 0, 0}, image.extent);

  return VK_SUCCESS;
}

VkResult StagingUploader::appendImageSub(nvvk::Image&                    image,
                                         const VkOffset3D&               offset,
                                         const VkExtent3D&               extent,
                                         const VkImageSubresourceLayers& subresource,
                                         size_t                          dataSize,
                                         const void*                     data,
                                         VkImageLayout                   newLayout /*= VK_IMAGE_LAYOUT_UNDEFINED*/,
                                         const SemaphoreState&           semaphoreState /*= {}*/)
{
  if(dataSize == 0)
  {
    return VK_SUCCESS;
  }

  assert(data);

  BufferRange stagingSpace;
  NVVK_FAIL_RETURN(acquireStagingSpace(stagingSpace, dataSize, data, semaphoreState));

  const VkImageSubresourceRange subresourceRange{subresource.aspectMask, subresource.mipLevel, 1,
                                                 subresource.baseArrayLayer, subresource.layerCount};

  m_batch.addImageCopy(stagingSpace.buffer, stagingSpace.offset, image.image, image.descriptor.imageLayout, newLayout,
                       dataSize, subresource, offset, extent, &subresourceRange);

  return VK_SUCCESS;
}

VkResult StagingUploader::appendImageSubMapping(nvvk::Image&                    image,
                                                const VkOffset3D&               offset,
                                                const VkExtent3D&               extent,
                                                const VkImageSubresourceLayers& subresource,
                                                size_t                          dataSize,
                                                void*&                          uploadMapping,
                                                VkImageLayout         newLayout /*= VK_IMAGE_LAYOUT_UNDEFINED*/,
                                                const SemaphoreState& semaphoreState /*= {}*/)
{
  uploadMapping = nullptr;

  if(dataSize == 0)
  {
    return VK_SUCCESS;
  }

  BufferRange stagingSpace;
  NVVK_FAIL_RETURN(acquireStagingSpace(stagingSpace, dataSize, nullptr, semaphoreState));

  uploadMapping = stagingSpace.mapping;

  const VkImageSubresourceRange subresourceRange{subresource.aspectMask, subresource.mipLevel, 1,
                                                 subresource.baseArrayLayer, subresource.layerCount};

  m_batch.addImageCopy(stagingSpace.buffer, stagingSpace.offset, image.image, image.descriptor.imageLayout, newLayout,
                       dataSize, subresource, offset, extent, &subresourceRange);

  return VK_SUCCESS;
}

bool StagingUploader::checkAppendedSize(size_t limitInBytes, size_t addedSize) const
{
  return m_batch.checkAppendedSize(limitInBytes, addedSize);
}

void StagingUploader::cmdUploadAppended(VkCommandBuffer cmd)
{
  if(m_batchRequiresFlush)
  {
    size_t stagingResourceCount = m_stagingResources.size();
    // we speculatively always flush and not check each resource individually
    for(size_t i = stagingResourceCount - m_batchStagingCount; i < stagingResourceCount; i++)
    {
      StagingResource& staging = m_stagingResources[i];
      m_resourceAllocator->flushBuffer(staging.buffer, 0, VK_WHOLE_SIZE);
    }
  }
  m_batch.cmdCopyAppended(cmd);
  m_batch.enableLayoutBarrierMask = false;
  resetStaging(false);
}

void StagingUploader::cancelAppended()
{
  m_batch.reset();
  m_batch.enableLayoutBarrierMask = false;
  resetStaging(true);
}

void StagingUploader::releaseStaging(bool forceAll)
{
  VkDevice device = m_resourceAllocator->getDevice();

  size_t originalCount = m_stagingResources.size();
  size_t readIdx       = 0;
  size_t writeIdx      = 0;

  // compact as we iterate

  SemaphoreStateSignalCache signalCache;

  for(size_t readIdx = 0; readIdx < m_stagingResources.size(); readIdx++)
  {
    StagingResource& stagingResource = m_stagingResources[readIdx];

    // always release with forceAll,
    // also if semaphoreState is invalid,
    // otherwise test if it was signaled
    bool canRelease = forceAll || (!stagingResource.semaphoreState.isValid())
                      || signalCache.testSignaled(device, stagingResource.semaphoreState);

    if(canRelease)
    {
      m_stagingResourcesSize -= stagingResource.buffer.bufferSize;
      m_resourceAllocator->destroyBuffer(stagingResource.buffer);

      stagingResource.semaphoreState = {};
    }
    else if(readIdx != writeIdx)
    {
      m_stagingResources[writeIdx++] = std::move(stagingResource);
    }
    else
    {
      writeIdx++;
    }
  }

  m_stagingResources.resize(writeIdx);
}

void StagingUploader::resetStaging(bool isCancel)
{
  if(isCancel)
  {
    size_t count = m_stagingResources.size();

    // Resources are always appended to end of vector,
    // therefore we only need to remove last.
    for(size_t i = 0; i < m_batchStagingCount; i++)
    {
      StagingResource& stagingResource = m_stagingResources[count - 1 - i];

      m_stagingResourcesSize -= stagingResource.buffer.bufferSize;
      m_resourceAllocator->destroyBuffer(stagingResource.buffer);

      stagingResource.semaphoreState = {};
    };

    m_stagingResources.resize(count - m_batchStagingCount);
  }
  m_batchStagingCount  = 0;
  m_batchRequiresFlush = false;
}

}  // namespace nvvk

//--------------------------------------------------------------------------------------------------
// Usage example
//--------------------------------------------------------------------------------------------------
[[maybe_unused]] static void usage_StagingUploader()
{
  VkDevice                device{};
  VkResult                result{};
  nvvk::ResourceAllocator resourceAllocator{};

  nvvk::StagingUploader stagingUploader;
  stagingUploader.init(&resourceAllocator);

  //////////////////////////////////////////////////////////////////////////
  // simple example, relying on device wait idle, not using SemaphoreState
  {
    // Create buffer
    nvvk::Buffer buffer;

    // we prefer device memory and set a few bits that allow device-mappable memory to be used, but doesn't enforce it
    resourceAllocator.createBuffer(buffer, 256, VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
                                   VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_ALLOW_TRANSFER_INSTEAD_BIT
                                       | VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);

    // Upload data
    std::vector<float> data     = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f};
    size_t             dataSize = data.size() * sizeof(float);
    uint32_t           offset   = 0;

    // The stagingUploader will detect if the buffer was mappable directly then copy there directly,
    // otherwise copy through a temporary staging buffer.
    //
    // Note we consume the provided pointers directly here, but underlying vulkan commands are "appended"
    // and need to be executed later.
    stagingUploader.appendBuffer(buffer, offset, dataSize, data.data());

    // Execute the upload of all previously appended data to the GPU (if necessary)
    VkCommandBuffer cmd = nullptr;  // EX: create a command buffer
    stagingUploader.cmdUploadAppended(cmd);

    // submit command buffer
    //vkQueueSubmit(...);

    // device wait idle ensures cmd has completed
    vkDeviceWaitIdle(device);

    // safe to release everything after
    stagingUploader.releaseStaging();
  }

  //////////////////////////////////////////////////////////////////////////
  // batched,  relying on device wait idle for release, not using SemaphoreState
  {
    // get command buffer somehow
    VkCommandBuffer cmd{};

    // in this scenario we want to upload a lot of stuff, but we want to keep
    // an upper bound to how much temporary staging memory we use (1 GiB here).


    // we want the staging uploader to manage image layout transitions
    stagingUploader.setEnableLayoutBarriers(true);

    std::vector<nvvk::Image>        myImageTextures;
    std::vector<std::span<uint8_t>> myImageDatas;
    for(size_t i = 0; i < myImageTextures.size(); i++)
    {
      bool isLast = i == myImageTextures.size() - 1;

      // we are not using the semaphoreState in this loop because we intend to
      // use device wait idle and submit in multiple batches anyway


      // this will handle the transition from the current `myImageTextures[i].descriptor.imageLayout` to `VK_IMAGE_LAYOUT_READ_ONLY_OPTIMAL`
      // as well as the intermediate transition to `VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL`
      result = stagingUploader.appendImage(myImageTextures[i], myImageDatas[i], VK_IMAGE_LAYOUT_READ_ONLY_OPTIMAL);

      // for handling the transition with multiple mips, special care must be taken.
      VkImageLayout originalLayout = myImageTextures[i].descriptor.imageLayout;
      for(uint32_t m = 0; m < myImageTextures[i].mipLevels; m++)
      {
        std::vector<std::span<uint8_t>> imageMipDatas;
        VkImageSubresourceLayers        subResource{
                   .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                   .mipLevel   = m,
                   .layerCount = 1,
        };
        VkOffset3D offset{};
        VkExtent3D extent{};

        // reset the imageLayout with each mip, given the automatic transition logic looks only at this single value for
        // all mips.
        myImageTextures[i].descriptor.imageLayout = originalLayout;
        result = stagingUploader.appendImageSub(myImageTextures[i], offset, extent, subResource, imageMipDatas[m],
                                                VK_IMAGE_LAYOUT_READ_ONLY_OPTIMAL);
      }

      // flush upload if we reached a gigabyte or if we are the last resource
      if(isLast || stagingUploader.checkAppendedSize(1024 * 1024 * 1024))
      {
        // handles transfers and barriers for layout transitions
        stagingUploader.cmdUploadAppended(cmd);

        // submit cmd buffer to queue
        cmd;

        vkDeviceWaitIdle(device);

        stagingUploader.releaseStaging();

        // get a new command buffer
        cmd;
      }
    }
  }


  //////////////////////////////////////////////////////////////////////////
  // using semaphore state to track deletion of temporary resources

  {
    // imagine we are updating this buffer every frame
    // create buffer
    std::vector<float> myData;
    nvvk::Buffer       myBuffer;
    VkResult result = resourceAllocator.createBuffer(myBuffer, std::span(myData).size_bytes(), VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT);

    // track its completion through a timeline semaphore
    VkSemaphore timelineSemaphore{};
    uint64_t    timelineValue = 1;

    // frame loop
    while(true)
    {
      // release staging resources from past frames based on their SemaphoreState
      stagingUploader.releaseStaging();

      // get command buffer somehow
      VkCommandBuffer cmd{};

      nvvk::SemaphoreState semaphoreState = nvvk::SemaphoreState::makeFixed(timelineSemaphore, timelineValue);

      // the staging uploader provides two ways to fill a buffer:
      if(true)
      {
        // either providing the data in full
        // and we copy in full via memcpy
        result = stagingUploader.appendBuffer(myBuffer, 0, std::span(myData), semaphoreState);
      }
      else
      {
        // or get a pointer
        float* mappingPointer = nullptr;
        result = stagingUploader.appendBufferMapping(myBuffer, 0, std::span(myData).size_bytes(), mappingPointer, semaphoreState);
        // manually fill the pointer with sequential writes
      }

      // record all potential copy and barrier operations
      stagingUploader.cmdUploadAppended(cmd);
      // submit cmd buffer to queue signaling the timelineValue
      cmd;

      // next frame uses new timelineValue
      timelineValue++;
    }
  }

  //////////////////////////////////////////////////////////////////////////
  // queue ownership transfer

  {
    // We want to do an async transfer on a dedicated transfer queue.
    uint32_t transferQueueFamilyIndex = 0;
    uint32_t graphicsQueueFamilyIndex = 1;

    // And our resources are using the default `VK_SHARING_MODE_EXCLUSIVE`
    nvvk::Buffer          myBuffer;
    std::vector<uint32_t> myData;

    // Track transfer completion through a timeline semaphore.
    // This semaphore state would be signaled by the transfer queue.
    VkSemaphore transferTimelineSemaphore{};
    uint64_t    transferTimelineValue = 1;
    nvvk::SemaphoreState transferSemaphoreState = nvvk::SemaphoreState::makeFixed(transferTimelineSemaphore, transferTimelineValue);

    // by usual means acquire the command buffers.
    VkCommandBuffer transferCmd{};
    VkCommandBuffer graphicsCmd{};

    // we want the uploader to handle the required barriers for us.
    stagingUploader.setEnableLayoutBarriers(true);
    stagingUploader.setEnableOwnerBarriers(true, transferQueueFamilyIndex, graphicsQueueFamilyIndex);
    // also some automatism so it knows we are on a transfer queue (only relevant to image layout transition handling)
    stagingUploader.beginTransferOnly();

    result = stagingUploader.appendBuffer(myBuffer, 0, std::span(myData), transferSemaphoreState);

    // The graphics queue acquires the ownership of myBuffer.
    stagingUploader.getOwnerAcquisitionBarriers().cmdPipelineBarrier(graphicsCmd, 0);

    // Enqueues the actual copy commands (if applicable).
    // this command resets the staging uploader's internal state,
    //
    stagingUploader.cmdUploadAppended(transferCmd);

    // must ensure the following conditions:
    // - the queue submit of `transferCmd` signals `transferSemaphoreState`
    // - the submit of `graphicsCmd` waits for `transferSemaphoreState`
  }
}
