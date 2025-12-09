/*
* Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
* SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
* SPDX-License-Identifier: Apache-2.0
*/

#include "staging.hpp"
#include "barriers.hpp"
#include "check_error.hpp"
#include "debug_util.hpp"

namespace nvvk {

StagingUploader::StagingUploader(StagingUploader&& other) noexcept
{
  {
    std::swap(m_batch, other.m_batch);
    std::swap(m_resourceAllocator, other.m_resourceAllocator);
    std::swap(m_stagingResourcesSize, other.m_stagingResourcesSize);
    std::swap(m_stagingResources, other.m_stagingResources);
  }
}

nvvk::StagingUploader& StagingUploader::operator=(StagingUploader&& other) noexcept
{
  if(this != &other)
  {
    assert(m_resourceAllocator == nullptr && "deinit not called prior move assignment");

    std::swap(m_batch, other.m_batch);
    std::swap(m_resourceAllocator, other.m_resourceAllocator);
    std::swap(m_stagingResourcesSize, other.m_stagingResourcesSize);
    std::swap(m_stagingResources, other.m_stagingResources);
  }
  return *this;
}

void StagingUploader::init(ResourceAllocator* resourceAllocator, bool enableLayoutBarriers)
{
  assert(m_resourceAllocator == nullptr);
  m_resourceAllocator    = resourceAllocator;
  m_enableLayoutBarriers = enableLayoutBarriers;
}

void StagingUploader::deinit()
{
  if(m_resourceAllocator != nullptr)
  {
    releaseStaging(true);
    assert(m_stagingResources.empty() && m_stagingResourcesSize == 0);  // must have released all staged uploads
  }
  m_resourceAllocator = nullptr;
}

void StagingUploader::setEnableLayoutBarriers(bool enableLayoutBarriers)
{
  m_enableLayoutBarriers = enableLayoutBarriers;
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

  // VMA_MEMORY_USAGE_AUTO_PREFER_HOST staging memory is meant to not cost additional device memory
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
      .usage         = VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
      .requiredFlags = VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
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

  stagingSpace.buffer  = stagingResource.buffer.buffer;
  stagingSpace.offset  = 0;
  stagingSpace.range   = dataSize;
  stagingSpace.address = stagingResource.buffer.address;
  stagingSpace.mapping = stagingResource.buffer.mapping;

  return VK_SUCCESS;
}

void StagingUploader::cancelAppended()
{
  // let's use clear rather than m_batch = {};
  // to avoid heap allocations

  m_batch.copyBufferImageInfos.clear();
  m_batch.copyBufferImageRegions.clear();
  m_batch.copyBufferInfos.clear();
  m_batch.copyBufferRegions.clear();
  m_batch.pre.clear();
  m_batch.post.clear();
  m_batch.stagingSize  = 0;
  m_batch.transferOnly = false;
}

bool StagingUploader::isAppendedEmpty() const
{
  return m_batch.copyBufferImageInfos.empty() && m_batch.copyBufferInfos.empty();
}

void StagingUploader::beginTransferOnly()
{
  m_batch.transferOnly = true;
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

    VkBufferCopy2 copyRegionInfo{
        .sType     = VK_STRUCTURE_TYPE_BUFFER_COPY_2,
        .srcOffset = stagingSpace.offset,
        .dstOffset = bufferOffset,
        .size      = dataSize,
    };

    VkCopyBufferInfo2 copyBufferInfo{
        .sType       = VK_STRUCTURE_TYPE_COPY_BUFFER_INFO_2,
        .srcBuffer   = stagingSpace.buffer,
        .dstBuffer   = buffer.buffer,
        .regionCount = 1,
        .pRegions    = nullptr,  // set when calling `cmdUploadAppended`
    };

    m_batch.stagingSize += dataSize;
    m_batch.copyBufferRegions.emplace_back(copyRegionInfo);
    m_batch.copyBufferInfos.emplace_back(copyBufferInfo);
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

    VkBufferCopy2 copyRegionInfo{
        .sType     = VK_STRUCTURE_TYPE_BUFFER_COPY_2,
        .srcOffset = stagingSpace.offset,
        .dstOffset = bufferRange.offset,
        .size      = bufferRange.range,
    };

    VkCopyBufferInfo2 copyBufferInfo{
        .sType       = VK_STRUCTURE_TYPE_COPY_BUFFER_INFO_2,
        .srcBuffer   = stagingSpace.buffer,
        .dstBuffer   = bufferRange.buffer,
        .regionCount = 1,
        .pRegions    = nullptr,  // set when calling `cmdUploadAppended`
    };

    m_batch.stagingSize += bufferRange.range;
    m_batch.copyBufferRegions.emplace_back(copyRegionInfo);
    m_batch.copyBufferInfos.emplace_back(copyBufferInfo);
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

    VkBufferCopy2 copyRegionInfo{
        .sType     = VK_STRUCTURE_TYPE_BUFFER_COPY_2,
        .srcOffset = stagingSpace.offset,
        .dstOffset = bufferOffset,
        .size      = dataSize,
    };

    VkCopyBufferInfo2 copyBufferInfo{
        .sType       = VK_STRUCTURE_TYPE_COPY_BUFFER_INFO_2,
        .srcBuffer   = stagingSpace.buffer,
        .dstBuffer   = buffer.buffer,
        .regionCount = 1,
        .pRegions    = nullptr,  // set when calling `cmdUploadAppended`
    };

    m_batch.stagingSize += dataSize;
    m_batch.copyBufferRegions.emplace_back(copyRegionInfo);
    m_batch.copyBufferInfos.emplace_back(copyBufferInfo);

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

    VkBufferCopy2 copyRegionInfo{
        .sType     = VK_STRUCTURE_TYPE_BUFFER_COPY_2,
        .srcOffset = stagingSpace.offset,
        .dstOffset = bufferRange.offset,
        .size      = bufferRange.range,
    };

    VkCopyBufferInfo2 copyBufferInfo{
        .sType       = VK_STRUCTURE_TYPE_COPY_BUFFER_INFO_2,
        .srcBuffer   = stagingSpace.buffer,
        .dstBuffer   = bufferRange.buffer,
        .regionCount = 1,
        .pRegions    = nullptr,  // set when calling `cmdUploadAppended`
    };

    m_batch.stagingSize += bufferRange.range;
    m_batch.copyBufferRegions.emplace_back(copyRegionInfo);
    m_batch.copyBufferInfos.emplace_back(copyBufferInfo);

    return VK_SUCCESS;
  }
}

VkResult StagingUploader::appendImage(nvvk::Image& image, size_t dataSize, const void* data, VkImageLayout newLayout, const SemaphoreState& semaphoreState)
{
  BufferRange stagingSpace;
  NVVK_FAIL_RETURN(acquireStagingSpace(stagingSpace, dataSize, data, semaphoreState));

  bool layoutAllowsCopy = image.descriptor.imageLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL
                          || image.descriptor.imageLayout == VK_IMAGE_LAYOUT_GENERAL
                          || image.descriptor.imageLayout == VK_IMAGE_LAYOUT_SHARED_PRESENT_KHR;

  VkImageLayout dstImageLayout = image.descriptor.imageLayout;

  if(m_enableLayoutBarriers && !layoutAllowsCopy)
  {
    VkImageMemoryBarrier2 barrier = makeImageMemoryBarrier(
        {.image = image.image, .oldLayout = image.descriptor.imageLayout, .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL});
    modifyImageBarrier(barrier);

    dstImageLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;

    m_batch.pre.imageBarriers.push_back(barrier);
  }

  // Copy buffer data to the image
  const VkBufferImageCopy2 copyBufferImageRegion{
      .sType             = VK_STRUCTURE_TYPE_BUFFER_IMAGE_COPY_2,
      .bufferOffset      = stagingSpace.offset,
      .bufferRowLength   = 0,  // tightly packed
      .bufferImageHeight = 0,  // tightly packed
      .imageSubresource  = {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .layerCount = 1},
      .imageOffset       = {0, 0, 0},
      .imageExtent       = image.extent,
  };

  VkCopyBufferToImageInfo2 copyBufferToImageInfo{
      .sType          = VK_STRUCTURE_TYPE_COPY_BUFFER_TO_IMAGE_INFO_2,
      .srcBuffer      = stagingSpace.buffer,
      .dstImage       = image.image,
      .dstImageLayout = dstImageLayout,
      .regionCount    = 1,
      .pRegions       = nullptr,  // set when calling `cmdUploadAppended`
  };

  m_batch.stagingSize += dataSize;
  m_batch.copyBufferImageRegions.emplace_back(copyBufferImageRegion);
  m_batch.copyBufferImageInfos.emplace_back(copyBufferToImageInfo);

  if(m_enableLayoutBarriers && (!layoutAllowsCopy || newLayout != VK_IMAGE_LAYOUT_UNDEFINED))
  {
    if(newLayout != VK_IMAGE_LAYOUT_UNDEFINED)
    {
      image.descriptor.imageLayout = newLayout;
    }

    VkImageMemoryBarrier2 barrier =
        makeImageMemoryBarrier({.image = image.image, .oldLayout = dstImageLayout, .newLayout = image.descriptor.imageLayout});

    modifyImageBarrier(barrier);

    m_batch.post.imageBarriers.push_back(barrier);
  }

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
  BufferRange stagingSpace;
  NVVK_FAIL_RETURN(acquireStagingSpace(stagingSpace, dataSize, data, semaphoreState));

  bool layoutAllowsCopy = image.descriptor.imageLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL
                          || image.descriptor.imageLayout == VK_IMAGE_LAYOUT_GENERAL
                          || image.descriptor.imageLayout == VK_IMAGE_LAYOUT_SHARED_PRESENT_KHR;

  VkImageLayout dstImageLayout = image.descriptor.imageLayout;

  if(m_enableLayoutBarriers && !layoutAllowsCopy)
  {
    VkImageMemoryBarrier2 barrier = makeImageMemoryBarrier(
        {.image = image.image, .oldLayout = image.descriptor.imageLayout, .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL});
    modifyImageBarrier(barrier);

    dstImageLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;

    m_batch.pre.imageBarriers.push_back(barrier);
  }

  // Copy buffer data to the image
  const VkBufferImageCopy2 copyBufferImageRegion{
      .sType             = VK_STRUCTURE_TYPE_BUFFER_IMAGE_COPY_2,
      .bufferOffset      = stagingSpace.offset,
      .bufferRowLength   = 0,  // tightly packed
      .bufferImageHeight = 0,  // tightly packed
      .imageSubresource  = subresource,
      .imageOffset       = offset,
      .imageExtent       = extent,
  };

  VkCopyBufferToImageInfo2 copyBufferToImageInfo{
      .sType          = VK_STRUCTURE_TYPE_COPY_BUFFER_TO_IMAGE_INFO_2,
      .srcBuffer      = stagingSpace.buffer,
      .dstImage       = image.image,
      .dstImageLayout = dstImageLayout,
      .regionCount    = 1,
      .pRegions       = nullptr,  // set when calling `cmdUploadAppended`
  };

  m_batch.stagingSize += dataSize;
  m_batch.copyBufferImageRegions.emplace_back(copyBufferImageRegion);
  m_batch.copyBufferImageInfos.emplace_back(copyBufferToImageInfo);

  if(m_enableLayoutBarriers && (!layoutAllowsCopy || newLayout != VK_IMAGE_LAYOUT_UNDEFINED))
  {
    if(newLayout != VK_IMAGE_LAYOUT_UNDEFINED)
    {
      image.descriptor.imageLayout = newLayout;
    }

    VkImageMemoryBarrier2 barrier =
        makeImageMemoryBarrier({.image = image.image, .oldLayout = dstImageLayout, .newLayout = image.descriptor.imageLayout});

    modifyImageBarrier(barrier);

    m_batch.post.imageBarriers.push_back(barrier);
  }

  return VK_SUCCESS;
}

bool StagingUploader::checkAppendedSize(size_t limitInBytes, size_t addedSize) const
{
  return m_batch.stagingSize && (m_batch.stagingSize + addedSize) > limitInBytes;
}

void StagingUploader::cmdUploadAppended(VkCommandBuffer cmd)
{
  if(m_enableLayoutBarriers)
  {
    m_batch.pre.cmdPipelineBarrier(cmd, 0);
  }

  for(size_t i = 0; i < m_batch.copyBufferInfos.size(); i++)
  {
    m_batch.copyBufferInfos[i].pRegions = &m_batch.copyBufferRegions[i];
    vkCmdCopyBuffer2(cmd, &m_batch.copyBufferInfos[i]);
  }

  for(size_t i = 0; i < m_batch.copyBufferImageInfos.size(); i++)
  {
    m_batch.copyBufferImageInfos[i].pRegions = &m_batch.copyBufferImageRegions[i];
    vkCmdCopyBufferToImage2(cmd, &m_batch.copyBufferImageInfos[i]);
  }

  if(m_enableLayoutBarriers)
  {
    m_batch.post.cmdPipelineBarrier(cmd, 0);
  }

  // reset
  cancelAppended();
}

void StagingUploader::modifyImageBarrier(VkImageMemoryBarrier2& barrier)
{
  if(m_batch.transferOnly)
  {
    barrier.dstAccessMask &= (VK_ACCESS_2_TRANSFER_WRITE_BIT | VK_ACCESS_2_TRANSFER_READ_BIT);
    barrier.srcAccessMask &= (VK_ACCESS_2_TRANSFER_WRITE_BIT | VK_ACCESS_2_TRANSFER_READ_BIT);
    barrier.dstStageMask &= VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT;
    barrier.srcStageMask &= VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT;
    assert(barrier.dstAccessMask && barrier.srcStageMask);
  }
}

void StagingUploader::releaseStaging(bool forceAll)
{
  VkDevice device = m_resourceAllocator->getDevice();

  size_t originalCount = m_stagingResources.size();
  size_t readIdx       = 0;
  size_t writeIdx      = 0;

  // compact as we iterate

  for(size_t readIdx = 0; readIdx < m_stagingResources.size(); readIdx++)
  {
    StagingResource& stagingResource = m_stagingResources[readIdx];

    // always release with forceAll,
    // also if semaphoreState is invalid,
    // otherwise test if it was signaled
    bool canRelease =
        forceAll || (!stagingResource.semaphoreState.isValid()) || stagingResource.semaphoreState.testSignaled(device);

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
    else if(!forceAll)
    {
      writeIdx++;
    }
  }

  m_stagingResources.resize(writeIdx);
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
}
