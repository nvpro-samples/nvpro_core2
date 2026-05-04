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
#include <array>
#include <cassert>
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

  m_resourceDescStride = std::max(m_info.imageDescriptorSize(), m_info.bufferDescriptorSize());

  m_maxSamplerCapacity = static_cast<uint32_t>((m_info.maxSamplerHeapSize() - m_info.minSamplerHeapReservedRange())
                                               / m_info.samplerDescriptorSize());
  m_maxResourceCapacity =
      static_cast<uint32_t>((m_info.maxResourceHeapSize() - m_info.minResourceHeapReservedRange()) / m_resourceDescStride);

  return VK_SUCCESS;
}

//--------------------------------------------------------------------------------------------------
void DescriptorHeap::deinit()
{
  m_info.deinit();
  m_info                = {};
  m_resourceDescStride  = {};
  m_samplerHeapSize     = {};
  m_resourceHeapSize    = {};
  m_maxSamplers         = {};
  m_maxResources        = {};
  m_maxSamplerCapacity  = {};
  m_maxResourceCapacity = {};
  m_samplerDirtyMin     = ~0u;
  m_samplerDirtyMax     = 0;
  m_resourceDirtyMin    = ~0u;
  m_resourceDirtyMax    = 0;
  m_samplerMap.clear();
  m_samplerIndexToState.clear();
  m_freeSamplerSlots.clear();
  m_nextSamplerSlot = 0;
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
  m_samplerDirtyMin = ~0u;
  m_samplerDirtyMax = 0;
  m_samplerMap.clear();
  m_samplerIndexToState.clear();
  m_freeSamplerSlots.clear();
  m_nextSamplerSlot = 0;
  return m_samplerHeapSize;
}

//--------------------------------------------------------------------------------------------------
VkDeviceSize DescriptorHeap::setupResourceHeap(uint32_t maxResources)
{
  assert(m_info.isInitialized() && maxResources > 0);
  if(maxResources > m_maxResourceCapacity)
  {
    LOGE("DescriptorHeap: maxResources (%u) exceeds hardware capacity (%u)\n", maxResources, m_maxResourceCapacity);
    return 0;
  }

  VkDeviceSize resourceHeapSize = m_resourceDescStride * maxResources + m_info.minResourceHeapReservedRange();
  resourceHeapSize              = m_info.alignToResourceHeap(resourceHeapSize);

  m_resourceHeapSize = resourceHeapSize;
  m_maxResources     = maxResources;
  m_resourceDirtyMin = ~0u;
  m_resourceDirtyMax = 0;
  return m_resourceHeapSize;
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
VkResult DescriptorHeap::writeImageDescriptor(uint32_t index, VkImage image, VkFormat format, VkImageLayout layout, void* resourceHeapBase, VkImageViewType viewType)
{
  assert(m_info.isInitialized());
  assert(index < m_maxResources);
  assert(resourceHeapBase != nullptr);

  if(!m_info.isInitialized())
    return VK_ERROR_INITIALIZATION_FAILED;
  if(index >= m_maxResources || resourceHeapBase == nullptr)
    return VK_ERROR_VALIDATION_FAILED_EXT;

  void* dst = static_cast<uint8_t*>(resourceHeapBase) + static_cast<size_t>(index) * static_cast<size_t>(m_resourceDescStride);
  VkResult result = m_info.writeImageDescriptor(image, format, layout, dst, viewType);
  if(result != VK_SUCCESS)
    return result;

  m_resourceDirtyMin = std::min(m_resourceDirtyMin, index);
  m_resourceDirtyMax = std::max(m_resourceDirtyMax, index);
  return VK_SUCCESS;
}

//--------------------------------------------------------------------------------------------------
VkResult DescriptorHeap::writeBufferDescriptor(uint32_t         index,
                                               VkDeviceAddress  bufferAddress,
                                               VkDeviceSize     bufferSize,
                                               VkDescriptorType type,
                                               void*            resourceHeapBase)
{
  assert(m_info.isInitialized());
  assert(index < m_maxResources);
  assert(resourceHeapBase != nullptr);

  if(!m_info.isInitialized())
    return VK_ERROR_INITIALIZATION_FAILED;
  if(index >= m_maxResources || resourceHeapBase == nullptr)
    return VK_ERROR_VALIDATION_FAILED_EXT;

  void* dst = static_cast<uint8_t*>(resourceHeapBase) + static_cast<size_t>(index) * static_cast<size_t>(m_resourceDescStride);
  VkResult result = m_info.writeBufferDescriptor(bufferAddress, bufferSize, type, dst);
  if(result != VK_SUCCESS)
    return result;

  m_resourceDirtyMin = std::min(m_resourceDirtyMin, index);
  m_resourceDirtyMax = std::max(m_resourceDirtyMax, index);
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
DescriptorHeap::DirtyRange DescriptorHeap::getResourceDirtyRange() const
{
  if(m_resourceDirtyMin > m_resourceDirtyMax)
    return {0, 0};
  VkDeviceSize offset = static_cast<VkDeviceSize>(m_resourceDirtyMin) * m_resourceDescStride;
  VkDeviceSize size   = static_cast<VkDeviceSize>(m_resourceDirtyMax - m_resourceDirtyMin + 1) * m_resourceDescStride;
  return {offset, size};
}

//--------------------------------------------------------------------------------------------------
void DescriptorHeap::clearSamplerDirty()
{
  m_samplerDirtyMin = ~0u;
  m_samplerDirtyMax = 0;
}

//--------------------------------------------------------------------------------------------------
void DescriptorHeap::clearResourceDirty()
{
  m_resourceDirtyMin = ~0u;
  m_resourceDirtyMax = 0;
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

  if(resourceHeapAddr != 0 && m_maxResources > 0)
  {
    VkBindHeapInfoEXT resourceBind{
        .sType               = VK_STRUCTURE_TYPE_BIND_HEAP_INFO_EXT,
        .heapRange           = {resourceHeapAddr, m_resourceHeapSize},
        .reservedRangeOffset = m_resourceDescStride * m_maxResources,
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

  // =====================================================================
  // 2. Setup — compute heap sizes
  // =====================================================================
  // Images and buffers share the resource heap.  The caller decides the
  // index layout; for example, textures at [0..N), buffers at [N..M).
  VkDeviceSize samplerBufSize  = heap.setupSamplerHeap(2);
  VkDeviceSize resourceBufSize = heap.setupResourceHeap(8);  // 4 images + 4 buffers

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

  // Images at indices [0..4)
  NVVK_CHECK(heap.writeImageDescriptor(0, image, format, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, resMapping));
  // nvvk::Image convenience overload
  nvvk::Image nvImage{};
  nvImage.image                  = image;
  nvImage.format                 = format;
  nvImage.descriptor.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
  NVVK_CHECK(heap.writeImageDescriptor(1, nvImage, resMapping));

  // Buffers at indices [4..8) — same resource heap, different descriptor type
  NVVK_CHECK(heap.writeBufferDescriptor(4, sceneUBO, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, resMapping));
  NVVK_CHECK(heap.writeBufferDescriptor(5, storageBuffer, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, resMapping));
  // Low-level overload: raw address + size
  NVVK_CHECK(heap.writeBufferDescriptor(6, storageBuffer.address, storageBuffer.bufferSize,
                                        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, resMapping));

  // =====================================================================
  // 5. Bind heaps and draw
  // =====================================================================
  heap.cmdBindHeaps(cmd, samplerHeapBuffer.address, resourceHeapBuffer.address);

  // In the shader (Slang):
  //   Texture2D    tex = Texture2D.Handle(uint2(0, 0));           // image at index 0
  //   SamplerState smp = SamplerState.Handle(uint2(linearIdx, 0));
  //   StructuredBuffer<T> buf = StructuredBuffer<T>.Handle(uint2(4, 0)); // buffer at index 4
  //
  // GLSL/SPIR-V paths that map traditional set/binding to heap offsets via
  // VkDescriptorSetAndBindingMappingEXT need the per-heap stride:
  const uint32_t samplerStride  = static_cast<uint32_t>(heap.samplerDescriptorSize());
  const uint32_t resourceStride = static_cast<uint32_t>(heap.resourceDescriptorStride());
  (void)samplerStride;
  (void)resourceStride;

  // In real code this is your application's push-constants struct.
  std::array<float, 4> push{};
  VkPushDataInfoEXT    pushInfo{
         .sType  = VK_STRUCTURE_TYPE_PUSH_DATA_INFO_EXT,
         .offset = 0,
         .data   = {.address = push.data(), .size = sizeof(push)},
  };
  vkCmdPushDataEXT(cmd, &pushInfo);

  // =====================================================================
  // 6. Incremental update — dirty range tracking
  // =====================================================================
  // After initial fill, update individual descriptors (e.g. texture
  // load/unload).  The dirty range tells you what sub-range to upload.
  // With a persistently mapped buffer this is just a barrier concern,
  // but with a device-local buffer you'd upload the dirty sub-range:
  heap.clearResourceDirty();
  NVVK_CHECK(heap.writeImageDescriptor(2, image, format, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, resMapping));
  NVVK_CHECK(heap.writeBufferDescriptor(5, storageBuffer, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, resMapping));
  DescriptorHeap::DirtyRange dirty = heap.getResourceDirtyRange();  // covers indices [2..5]
  (void)dirty;  // with a mapped buffer, data is already visible to the GPU
  heap.clearResourceDirty();

  // =====================================================================
  // 7. Resize — grow the resource heap
  // =====================================================================
  VkDeviceSize newSize = heap.setupResourceHeap(16);
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
