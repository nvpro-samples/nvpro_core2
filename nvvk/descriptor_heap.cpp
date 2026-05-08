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
* SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION
* SPDX-License-Identifier: Apache-2.0
*/

#include <volk.h>

#include "descriptor_heap.hpp"

#include <algorithm>
#include <cassert>
#include <limits>
#include <vector>

#include <nvutils/logger.hpp>

#include "check_error.hpp"
#include "resource_allocator.hpp"
#include "staging.hpp"

namespace nvvk {


//--------------------------------------------------------------------------------------------------
VkResult DescriptorHeap::init(VkPhysicalDevice physicalDevice, VkDevice device)
{
  deinit();

  NVVK_FAIL_RETURN(m_info.init(physicalDevice, device));

  // Find the available capacity for resource descriptors
  const VkDeviceSize maxHeapBytes = m_info.maxResourceHeapSize();
  const VkDeviceSize minReserved  = m_info.minResourceHeapReservedRange();
  m_availableCapacity             = (maxHeapBytes >= minReserved) ? (maxHeapBytes - minReserved) : 0;

  // Max capacities if the entire resource heap were one type, ignoring the reserved range and per-type alignment for mixed layouts.
  m_maxImageCapacity =
      static_cast<uint32_t>((m_info.imageDescriptorSize() > 0) ? (m_availableCapacity / m_info.imageDescriptorSize()) : 0);
  m_maxBufferCapacity =
      static_cast<uint32_t>((m_info.bufferDescriptorSize() > 0) ? (m_availableCapacity / m_info.bufferDescriptorSize()) : 0);

  m_maxSamplerCapacity = static_cast<uint32_t>((m_info.maxSamplerHeapSize() - m_info.minSamplerHeapReservedRange())
                                               / m_info.samplerDescriptorSize());

  return VK_SUCCESS;
}

//--------------------------------------------------------------------------------------------------
void DescriptorHeap::deinit()
{
  m_info.deinit();
  m_info                     = {};
  m_samplerHeapSize          = {};
  m_resourceHeapSize         = {};
  m_maxSamplers              = {};
  m_maxSamplerCapacity       = {};
  m_maxImages                = {};
  m_maxBuffers               = {};
  m_maxImageCapacity         = {};
  m_maxBufferCapacity        = {};
  m_availableCapacity        = {};
  m_imageRegionStartBytes    = {};
  m_bufferRegionStartBytes   = {};
  m_reservedRangeOffsetBytes = {};
  m_resourceImageDirtyMin    = std::numeric_limits<VkDeviceSize>::max();
  m_resourceImageDirtyMax    = {};
  m_resourceBufferDirtyMin   = std::numeric_limits<VkDeviceSize>::max();
  m_resourceBufferDirtyMax   = {};
  m_samplerDirtyMin          = std::numeric_limits<uint32_t>::max();
  m_samplerDirtyMax          = {};
  m_samplerMap.clear();
  m_samplerIndexToState.clear();
  m_freeSamplerSlots.clear();
  m_nextSamplerSlot = 0;
}

//--------------------------------------------------------------------------------------------------
uint32_t DescriptorHeap::imageShaderIndexBase() const
{
  const VkDeviceSize sz = m_info.imageDescriptorSize();
  return (sz > 0) ? static_cast<uint32_t>(m_imageRegionStartBytes / sz) : 0;
}

//--------------------------------------------------------------------------------------------------
uint32_t DescriptorHeap::bufferShaderIndexBase() const
{
  const VkDeviceSize sz = m_info.bufferDescriptorSize();
  return (sz > 0) ? static_cast<uint32_t>(m_bufferRegionStartBytes / sz) : 0;
}

//--------------------------------------------------------------------------------------------------
VkDeviceSize DescriptorHeap::setupSamplerHeap(uint32_t maxSamplers)
{
  assert(m_info.isInitialized() && maxSamplers > 0);
  if(maxSamplers > m_maxSamplerCapacity)
  {
    LOGE("DescriptorHeap: maxSamplers (%u) exceeds hardware capacity (%u)\n", maxSamplers, m_maxSamplerCapacity);
    return 0;
  }

  VkDeviceSize offset = 0;
  m_info.appendSamplerDescriptors(offset, maxSamplers);
  m_info.appendSamplerReservedRange(offset);
  m_samplerHeapSize = m_info.alignToSamplerHeap(offset);

  m_maxSamplers     = maxSamplers;
  m_samplerDirtyMin = std::numeric_limits<uint32_t>::max();
  m_samplerDirtyMax = {};
  m_samplerMap.clear();
  m_samplerIndexToState.clear();
  m_freeSamplerSlots.clear();
  m_nextSamplerSlot = 0;
  return m_samplerHeapSize;
}

//--------------------------------------------------------------------------------------------------
VkDeviceSize DescriptorHeap::setupResourceHeap(uint32_t maxImages, uint32_t maxBuffers)
{
  assert(m_info.isInitialized());
  if(maxImages == 0 && maxBuffers == 0)
  {
    LOGE("DescriptorHeap: setupResourceHeap requires maxImages > 0 or maxBuffers > 0\n");
    return 0;
  }

  VkDeviceSize offset        = 0;
  m_imageRegionStartBytes    = m_info.appendImageDescriptors(offset, maxImages);
  m_bufferRegionStartBytes   = m_info.appendBufferDescriptors(offset, maxBuffers);
  m_reservedRangeOffsetBytes = m_info.appendResourceReservedRange(offset);
  // Accurate byte usage for this (images, buffers) layout: packing through append calls, then resource heap alignment.
  const VkDeviceSize packedResourceHeapBytes = m_info.alignToResourceHeap(offset);

  if(packedResourceHeapBytes > m_info.maxResourceHeapSize())
  {
    LOGE(
        "DescriptorHeap: packed resource heap (%llu bytes) exceeds maxResourceHeapSize (%llu) "
        "(images=%u, buffers=%u; includes per-type alignment between regions, reserved range, heap alignment). "
        "Loose limits from availableCapacity() do not apply to mixed layouts.\n",
        packedResourceHeapBytes, m_info.maxResourceHeapSize(), maxImages, maxBuffers);
    return 0;
  }

  m_resourceHeapSize       = packedResourceHeapBytes;
  m_maxImages              = maxImages;
  m_maxBuffers             = maxBuffers;
  m_resourceImageDirtyMin  = std::numeric_limits<VkDeviceSize>::max();
  m_resourceImageDirtyMax  = 0;
  m_resourceBufferDirtyMin = std::numeric_limits<VkDeviceSize>::max();
  m_resourceBufferDirtyMax = 0;
  return m_resourceHeapSize;
}

//--------------------------------------------------------------------------------------------------
bool DescriptorHeap::testMixedCapacity(uint32_t maxImages, uint32_t maxBuffers)
{
  assert(m_info.isInitialized());
  if(maxImages == 0 && maxBuffers == 0)
    return false;

  VkDeviceSize offset = 0;
  m_info.appendImageDescriptors(offset, maxImages);
  m_info.appendBufferDescriptors(offset, maxBuffers);
  m_info.appendResourceReservedRange(offset);
  const VkDeviceSize packedResourceHeapBytes = m_info.alignToResourceHeap(offset);

  const VkDeviceSize maxHeap = m_info.maxResourceHeapSize();
  if(packedResourceHeapBytes > maxHeap)
  {
    LOGI("DescriptorHeap::testMixedCapacity: images=%u buffers=%u pack to %llu bytes, above maxResourceHeapSize (%llu).\n", maxImages,
         maxBuffers, static_cast<unsigned long long>(packedResourceHeapBytes), static_cast<unsigned long long>(maxHeap));
    return false;
  }
  return true;
}

//--------------------------------------------------------------------------------------------------
uint32_t DescriptorHeap::acquireSamplerDescriptor(const VkSamplerCreateInfo& samplerCreateInfo, void* samplerHeapBase)
{
  SamplerState state = SamplerState::fromCreateInfo(samplerCreateInfo);

  if(auto it = m_samplerMap.find(state); it != m_samplerMap.end())
  {
    it->second.refCount++;
    return it->second.heapIndex;
  }

  bool     fromFreeList = false;
  uint32_t index{};
  if(!m_freeSamplerSlots.empty())
  {
    index = m_freeSamplerSlots.back();
    m_freeSamplerSlots.pop_back();
    fromFreeList = true;
  }
  else
  {
    if(m_nextSamplerSlot >= m_maxSamplers)
    {
      LOGE("DescriptorHeap: sampler heap full (%u / %u slots used)\n", m_nextSamplerSlot + 1, m_maxSamplers);
      return ~0u;
    }
    index = m_nextSamplerSlot++;
  }

  VkResult wr = writeSamplerDescriptor(index, samplerCreateInfo, samplerHeapBase);
  if(wr != VK_SUCCESS)
  {
    LOGE("DescriptorHeap: writeSamplerDescriptor failed (VkResult %d)\n", static_cast<int>(wr));
    if(fromFreeList)
      m_freeSamplerSlots.push_back(index);
    else
      m_nextSamplerSlot--;
    return ~0u;
  }

  m_samplerMap[state]          = {index, 1};
  m_samplerIndexToState[index] = state;
  return index;
}

//--------------------------------------------------------------------------------------------------
void DescriptorHeap::releaseSamplerDescriptor(uint32_t index)
{
  auto stateIt = m_samplerIndexToState.find(index);
  if(stateIt == m_samplerIndexToState.end())
  {
    assert(false && "Releasing unknown sampler descriptor index");
    return;
  }

  auto mapIt = m_samplerMap.find(stateIt->second);
  assert(mapIt != m_samplerMap.end());

  mapIt->second.refCount--;
  if(mapIt->second.refCount == 0)
  {
    const uint32_t heapIndex = mapIt->second.heapIndex;
    m_samplerMap.erase(mapIt);
    m_samplerIndexToState.erase(stateIt);
    m_freeSamplerSlots.push_back(heapIndex);
  }
}

//--------------------------------------------------------------------------------------------------
VkResult DescriptorHeap::writeSamplerDescriptor(uint32_t index, const VkSamplerCreateInfo& samplerCreateInfo, void* samplerHeapBase)
{
  assert(m_info.isInitialized());
  assert(index < m_maxSamplers);
  assert(samplerHeapBase != nullptr);

  if(!m_info.isInitialized())
    return VK_ERROR_INITIALIZATION_FAILED;
  if(index >= m_maxSamplers || samplerHeapBase == nullptr)
    return VK_ERROR_VALIDATION_FAILED_EXT;

  void*    dst    = static_cast<uint8_t*>(samplerHeapBase) + m_info.samplerOffset(index);
  VkResult result = m_info.writeSamplerDescriptor(samplerCreateInfo, dst);
  if(result != VK_SUCCESS)
    return result;

  m_samplerDirtyMin = std::min(m_samplerDirtyMin, index);
  m_samplerDirtyMax = std::max(m_samplerDirtyMax, index);
  return VK_SUCCESS;
}

//--------------------------------------------------------------------------------------------------
void DescriptorHeap::markResourceImageDirty(VkDeviceSize byteOffset, VkDeviceSize descriptorByteSize)
{
  const VkDeviceSize lastByte = byteOffset + descriptorByteSize - 1;
  m_resourceImageDirtyMin     = std::min(m_resourceImageDirtyMin, byteOffset);
  m_resourceImageDirtyMax     = std::max(m_resourceImageDirtyMax, lastByte);
}

//--------------------------------------------------------------------------------------------------
void DescriptorHeap::markResourceBufferDirty(VkDeviceSize byteOffset, VkDeviceSize descriptorByteSize)
{
  const VkDeviceSize lastByte = byteOffset + descriptorByteSize - 1;
  m_resourceBufferDirtyMin    = std::min(m_resourceBufferDirtyMin, byteOffset);
  m_resourceBufferDirtyMax    = std::max(m_resourceBufferDirtyMax, lastByte);
}

//--------------------------------------------------------------------------------------------------
VkResult DescriptorHeap::writeImageDescriptor(uint32_t        localImageIndex,
                                              VkImage         image,
                                              VkFormat        format,
                                              VkImageLayout   layout,
                                              void*           resourceHeapBase,
                                              VkImageViewType viewType)
{
  assert(m_info.isInitialized());
  assert(localImageIndex < m_maxImages && "localImageIndex out of range");
  assert(resourceHeapBase != nullptr);

  if(!m_info.isInitialized())
    return VK_ERROR_INITIALIZATION_FAILED;
  if(m_maxImages == 0 || localImageIndex >= m_maxImages || resourceHeapBase == nullptr)
    return VK_ERROR_VALIDATION_FAILED_EXT;

  const VkDeviceSize imgSize = m_info.imageDescriptorSize();
  void*              dst     = static_cast<uint8_t*>(resourceHeapBase) + m_imageRegionStartBytes
              + static_cast<size_t>(localImageIndex) * static_cast<size_t>(imgSize);
  VkResult result = m_info.writeImageDescriptor(image, format, layout, dst, viewType);
  if(result != VK_SUCCESS)
    return result;

  markResourceImageDirty(m_imageRegionStartBytes + static_cast<VkDeviceSize>(localImageIndex) * imgSize, imgSize);
  return VK_SUCCESS;
}

//--------------------------------------------------------------------------------------------------
VkResult DescriptorHeap::writeBufferDescriptor(uint32_t         localBufferIndex,
                                               VkDeviceAddress  bufferAddress,
                                               VkDeviceSize     bufferSize,
                                               VkDescriptorType type,
                                               void*            resourceHeapBase)
{
  assert(m_info.isInitialized());
  assert(localBufferIndex < m_maxBuffers && "localBufferIndex out of range");
  assert(resourceHeapBase != nullptr);

  if(!m_info.isInitialized())
    return VK_ERROR_INITIALIZATION_FAILED;
  if(m_maxBuffers == 0 || localBufferIndex >= m_maxBuffers || resourceHeapBase == nullptr)
    return VK_ERROR_VALIDATION_FAILED_EXT;

  const VkDeviceSize bufSize = m_info.bufferDescriptorSize();
  void*              dst     = static_cast<uint8_t*>(resourceHeapBase) + m_bufferRegionStartBytes
              + static_cast<size_t>(localBufferIndex) * static_cast<size_t>(bufSize);
  VkResult result = m_info.writeBufferDescriptor(bufferAddress, bufferSize, type, dst);
  if(result != VK_SUCCESS)
    return result;

  markResourceBufferDirty(m_bufferRegionStartBytes + static_cast<VkDeviceSize>(localBufferIndex) * bufSize, bufSize);
  return VK_SUCCESS;
}

//--------------------------------------------------------------------------------------------------
DescriptorHeap::DirtyRange DescriptorHeap::getSamplerDirtyRange() const
{
  if(m_samplerDirtyMin > m_samplerDirtyMax)
    return {0, 0};
  VkDeviceSize descSize = m_info.samplerDescriptorSize();
  VkDeviceSize offset   = static_cast<VkDeviceSize>(m_samplerDirtyMin) * descSize;
  VkDeviceSize size     = static_cast<VkDeviceSize>(m_samplerDirtyMax - m_samplerDirtyMin + 1) * descSize;
  return {offset, size};
}

//--------------------------------------------------------------------------------------------------
DescriptorHeap::DirtyRange DescriptorHeap::getResourceImageDirtyRange() const
{
  if(m_resourceImageDirtyMin > m_resourceImageDirtyMax)
    return {0, 0};
  VkDeviceSize offset = m_resourceImageDirtyMin;
  VkDeviceSize size   = m_resourceImageDirtyMax - m_resourceImageDirtyMin + 1;
  return {offset, size};
}

//--------------------------------------------------------------------------------------------------
DescriptorHeap::DirtyRange DescriptorHeap::getResourceBufferDirtyRange() const
{
  if(m_resourceBufferDirtyMin > m_resourceBufferDirtyMax)
    return {0, 0};
  VkDeviceSize offset = m_resourceBufferDirtyMin;
  VkDeviceSize size   = m_resourceBufferDirtyMax - m_resourceBufferDirtyMin + 1;
  return {offset, size};
}

//--------------------------------------------------------------------------------------------------
void DescriptorHeap::clearSamplerDirty()
{
  m_samplerDirtyMin = std::numeric_limits<uint32_t>::max();
  m_samplerDirtyMax = {};
}

//--------------------------------------------------------------------------------------------------
void DescriptorHeap::clearResourceDirty()
{
  m_resourceImageDirtyMin  = std::numeric_limits<VkDeviceSize>::max();
  m_resourceImageDirtyMax  = 0;
  m_resourceBufferDirtyMin = std::numeric_limits<VkDeviceSize>::max();
  m_resourceBufferDirtyMax = 0;
}

//--------------------------------------------------------------------------------------------------
void DescriptorHeap::cmdBindHeaps(VkCommandBuffer cmd, VkDeviceAddress samplerHeapAddr, VkDeviceAddress resourceHeapAddr) const
{
  if(samplerHeapAddr != 0 && m_maxSamplers > 0)
  {
    VkBindHeapInfoEXT samplerBind{
        .sType               = VK_STRUCTURE_TYPE_BIND_HEAP_INFO_EXT,
        .heapRange           = {samplerHeapAddr, m_samplerHeapSize},
        .reservedRangeOffset = m_info.samplerDescriptorSize() * m_maxSamplers,
        .reservedRangeSize   = m_info.minSamplerHeapReservedRange(),
    };
    vkCmdBindSamplerHeapEXT(cmd, &samplerBind);
  }

  if(resourceHeapAddr != 0 && (m_maxImages > 0 || m_maxBuffers > 0))
  {
    VkBindHeapInfoEXT resourceBind{
        .sType               = VK_STRUCTURE_TYPE_BIND_HEAP_INFO_EXT,
        .heapRange           = {resourceHeapAddr, m_resourceHeapSize},
        .reservedRangeOffset = m_reservedRangeOffsetBytes,
        .reservedRangeSize   = m_info.minResourceHeapReservedRange(),
    };
    vkCmdBindResourceHeapEXT(cmd, &resourceBind);
  }
}

//--------------------------------------------------------------------------------------------------
[[maybe_unused]] static void usage_DescriptorHeap()
{
  VkPhysicalDevice   physicalDevice{};
  VkDevice           device{};
  ResourceAllocator* allocator{};
  VkCommandBuffer    cmd{};
  VkImage            image{};
  VkFormat           format{VK_FORMAT_R8G8B8A8_UNORM};
  nvvk::Buffer       sceneUBO{};
  nvvk::Buffer       storageBuffer{};

  DescriptorHeap heap;

  // =====================================================================
  // 1. Init — query hardware properties
  // =====================================================================
  NVVK_CHECK(heap.init(physicalDevice, device));

  // Packed sizing for a candidate mixed heap (e.g. loose single-type ceilings from init often exceed this together).
  uint32_t numImages  = std::min(4U, heap.maxImageCapacity());
  uint32_t numBuffers = std::min(3U, heap.maxBufferCapacity());
  if(heap.testMixedCapacity(numImages, numBuffers) == false)
  {
    LOGE("DescriptorHeap: candidate mixed heap (images=%u, buffers=%u) exceeds capacity; adjust counts or check hardware limits.\n",
         numImages, numBuffers);
    return;
  }

  // =====================================================================
  // 2. Setup — compute heap sizes (images first, then buffers; local indices from 0)
  // =====================================================================
  VkDeviceSize samplerBufSize  = heap.setupSamplerHeap(2);
  VkDeviceSize resourceBufSize = heap.setupResourceHeap(4, 3);  // 4 images + 3 buffers (indices 0..2)

  // =====================================================================
  // 3. Create GPU heap buffers
  // =====================================================================
  // Either heap can be device-local (write via staging) or host-mapped (write directly).
  // Pick per memory characteristics; both paths shown below for illustration.
  const VkBufferUsageFlags2KHR heapUsage = DescriptorHeap::getRequiredBufferUsage();
  nvvk::Buffer                 samplerHeapBuffer{};
  nvvk::Buffer                 resourceHeapBuffer{};

  // Sampler heap: device-local (staging upload path — see step 4a)
  NVVK_CHECK(allocator->createBuffer(samplerHeapBuffer, samplerBufSize, heapUsage, VMA_MEMORY_USAGE_AUTO, {},
                                     heap.getSamplerHeapAlignment()));

  // Resource heap: persistently mapped (direct-write path, e.g. ReBAR — see step 4b)
  NVVK_CHECK(allocator->createBuffer(resourceHeapBuffer, resourceBufSize, heapUsage, VMA_MEMORY_USAGE_AUTO,
                                     VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT,
                                     heap.getResourceHeapAlignment()));

  // =====================================================================
  // 4a. Write sampler descriptors — staging upload (Method A)
  // =====================================================================
  // appendBufferMapping returns a pointer into the staging buffer;
  // vkWriteSamplerDescriptorsEXT writes land there directly.
  // Alternative: if the sampler heap is host-mapped (like the resource heap below),
  // pass samplerHeapBuffer.mapping to acquireSamplerDescriptor and skip the upload.
  StagingUploader uploader;
  uploader.init(allocator);

  void* smpMapping = nullptr;
  NVVK_CHECK(uploader.appendBufferMapping(samplerHeapBuffer, 0, samplerBufSize, smpMapping));

  VkSamplerCreateInfo linearCI{
      .sType     = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
      .magFilter = VK_FILTER_LINEAR,
      .minFilter = VK_FILTER_LINEAR,
  };
  VkSamplerCreateInfo nearestCI{
      .sType     = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
      .magFilter = VK_FILTER_NEAREST,
      .minFilter = VK_FILTER_NEAREST,
  };
  uint32_t linearIdx  = heap.acquireSamplerDescriptor(linearCI, smpMapping);
  uint32_t nearestIdx = heap.acquireSamplerDescriptor(nearestCI, smpMapping);

  // GPU DMA: staging → device-local sampler heap buffer
  uploader.cmdUploadAppended(cmd);
  // submit cmd, wait, then:
  uploader.deinit();

  // =====================================================================
  // 4b. Write resource descriptors — persistently mapped (fully zero-copy)
  // =====================================================================
  // Since the resource heap buffer is host-visible, writes go directly
  // into device-visible memory.  No staging upload needed.
  // Caveat: unsynchronized write — the caller must ensure the GPU is not reading the
  // descriptor slots being updated.
  void* resMapping = resourceHeapBuffer.mapping;

  NVVK_CHECK(heap.writeImageDescriptor(0, image, format, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, resMapping));
  // nvvk::Image convenience overload
  nvvk::Image nvImage{};
  nvImage.image                  = image;
  nvImage.format                 = format;
  nvImage.descriptor.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
  NVVK_CHECK(heap.writeImageDescriptor(1, nvImage, resMapping));

  NVVK_CHECK(heap.writeBufferDescriptor(0, sceneUBO, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, resMapping));
  NVVK_CHECK(heap.writeBufferDescriptor(1, storageBuffer, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, resMapping));
  NVVK_CHECK(heap.writeBufferDescriptor(2, storageBuffer.address, storageBuffer.bufferSize,
                                        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, resMapping));

  // =====================================================================
  // 5. Bind heaps and draw
  // =====================================================================
  const uint32_t imgBase = heap.imageShaderIndexBase();
  const uint32_t bufBase = heap.bufferShaderIndexBase();

  heap.cmdBindHeaps(cmd, samplerHeapBuffer.address, resourceHeapBuffer.address);

  // Mirror this layout in Slang push constants; bases combine with local indices for ResourceHeap handles:
  //   Texture2D.Handle(uint2(push.imageShaderIndexBase + localImageIdx, 0)), etc.
  struct DescriptorHeapPushExample
  {
    uint32_t imageShaderIndexBase{};
    uint32_t bufferShaderIndexBase{};
    uint32_t reserved[2]{};
  };
  DescriptorHeapPushExample push{};
  push.imageShaderIndexBase = imgBase;  // Currently always zero since images are first in the resource heap, but use for clarity and future flexibility.
  push.bufferShaderIndexBase = bufBase;  // Base index for buffer descriptors in the shader; add local indices from writeBufferDescriptor calls.
  VkPushDataInfoEXT pushInfo{
      .sType  = VK_STRUCTURE_TYPE_PUSH_DATA_INFO_EXT,
      .offset = 0,
      .data   = {.address = &push, .size = sizeof(push)},
  };
  vkCmdPushDataEXT(cmd, &pushInfo);

  // =====================================================================
  // 6. Incremental update — dirty range tracking
  // =====================================================================
  heap.clearResourceDirty();
  NVVK_CHECK(heap.writeImageDescriptor(2, image, format, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, resMapping));
  NVVK_CHECK(heap.writeBufferDescriptor(1, storageBuffer, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, resMapping));
  DescriptorHeap::DirtyRange dirtyImages  = heap.getResourceImageDirtyRange();
  DescriptorHeap::DirtyRange dirtyBuffers = heap.getResourceBufferDirtyRange();
  (void)dirtyImages;
  (void)dirtyBuffers;
  heap.clearResourceDirty();

  // =====================================================================
  // 7. Resize — grow the resource heap
  // =====================================================================
  VkDeviceSize newSize = heap.setupResourceHeap(8, 8);
  allocator->destroyBuffer(resourceHeapBuffer);
  NVVK_CHECK(allocator->createBuffer(resourceHeapBuffer, newSize, heapUsage, VMA_MEMORY_USAGE_AUTO,
                                     VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT,
                                     heap.getResourceHeapAlignment()));
  // Re-write all descriptors into the new buffer, then rebind.

  // =====================================================================
  // 8. Low-level access via info()
  // =====================================================================
  // For advanced use cases (mixed-type packed layouts, compute shader
  // descriptor manipulation), access the underlying DescriptorHeapWriter:
  const DescriptorHeapWriter& info = heap.info();
  (void)info.imageDescriptorSize();  // per-type sizes differ
  (void)info.bufferDescriptorSize();
  (void)info.imageDescriptorAlignment();
  (void)info.bufferDescriptorAlignment();
  // See usage_DescriptorHeapWriter() in descriptor_heap_writer.cpp for
  // stack-allocated writes, mixed-type append layouts, etc.

  // =====================================================================
  // 9. Cleanup
  // =====================================================================
  heap.releaseSamplerDescriptor(linearIdx);
  heap.releaseSamplerDescriptor(nearestIdx);
  allocator->destroyBuffer(samplerHeapBuffer);
  allocator->destroyBuffer(resourceHeapBuffer);
  heap.deinit();
}

}  // namespace nvvk
