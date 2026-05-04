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
#pragma once

#include <vulkan/vulkan_core.h>

namespace nvvk {

//////////////////////////////////////////////////////////////////////////

// Low-level descriptor heap writer helper (VK_EXT_descriptor_heap)
//
// Queries per-type descriptor sizes, alignments, and heap limits from
// VkPhysicalDeviceDescriptorHeapPropertiesEXT.  Provides:
//
// - Property accessors for all three descriptor types (sampler, image, buffer)
// - Simple index-to-offset conversion for uniform-type heaps
// - Append-style offset builders for mixed-type resource heap layouts
// - Thin VkResult-returning wrappers over vkWrite*DescriptorsEXT
//
// This class is stateless beyond the cached properties — no index allocation,
// no dedup, no dirty tracking.  Multiple callers can build independent layouts
// from the same instance.  For a higher-level helper with sampler dedup,
// dirty tracking, and heap bind commands, see DescriptorHeap.
//
// Usage:
//   see usage_DescriptorHeapWriter() in descriptor_heap_writer.cpp
class DescriptorHeapWriter
{
public:
  // Spec maxima for a single descriptor's byte size — safe upper bounds for stack allocation.
  // Per VK_EXT_descriptor_heap proposal: implementations must report sizes <= these maxima.
  // The actual hardware sizes (queried via *DescriptorSize() at runtime) are often smaller.
  // BUFFER is largest because `bufferDescriptorSize` also covers acceleration structure descriptors.
  // See https://github.com/KhronosGroup/Vulkan-Docs/blob/main/proposals/VK_EXT_descriptor_heap.adoc
  static constexpr size_t SAMPLER_DESCRIPTOR_MAX_BYTE_SIZE = 32;
  static constexpr size_t IMAGE_DESCRIPTOR_MAX_BYTE_SIZE   = 64;
  static constexpr size_t BUFFER_DESCRIPTOR_MAX_BYTE_SIZE  = 128;

  static constexpr VkBufferUsageFlags2KHR getRequiredBufferUsage()
  {
    return VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT_KHR | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT_KHR
           | VK_BUFFER_USAGE_2_DESCRIPTOR_HEAP_BIT_EXT;
  }

  static bool isSupported(VkPhysicalDevice physicalDevice);

  VkResult init(VkPhysicalDevice physicalDevice, VkDevice device);
  void     deinit() { *this = {}; }

  // --- Per-type descriptor sizes ---
  VkDeviceSize samplerDescriptorSize() const { return m_samplerDescSize; }
  VkDeviceSize imageDescriptorSize() const { return m_imageDescSize; }
  VkDeviceSize bufferDescriptorSize() const { return m_bufferDescSize; }

  // --- Per-type descriptor alignments within their heap ---
  VkDeviceSize samplerDescriptorAlignment() const { return m_samplerDescAlignment; }
  VkDeviceSize imageDescriptorAlignment() const { return m_imageDescAlignment; }
  VkDeviceSize bufferDescriptorAlignment() const { return m_bufferDescAlignment; }

  // --- Heap-level properties ---
  VkDeviceSize samplerHeapAlignment() const { return m_samplerHeapAlignment; }
  VkDeviceSize resourceHeapAlignment() const { return m_resourceHeapAlignment; }
  VkDeviceSize maxSamplerHeapSize() const { return m_maxSamplerHeapSize; }
  VkDeviceSize maxResourceHeapSize() const { return m_maxResourceHeapSize; }
  VkDeviceSize minSamplerHeapReservedRange() const { return m_minSamplerReservedRange; }
  VkDeviceSize minResourceHeapReservedRange() const { return m_minResourceReservedRange; }

  // --- Simple offset for uniform-type regions ---
  VkDeviceSize samplerOffset(uint32_t index) const { return index * m_samplerDescSize; }
  VkDeviceSize imageOffset(uint32_t index) const { return index * m_imageDescSize; }
  VkDeviceSize bufferOffset(uint32_t index) const { return index * m_bufferDescSize; }

  // --- Append for mixed-type layouts ---
  // Aligns `offset` to the type's alignment, advances by count * size.
  // Returns the aligned start offset of the appended block.
  VkDeviceSize appendSamplerDescriptors(VkDeviceSize& offset, uint32_t count = 1) const;
  VkDeviceSize appendImageDescriptors(VkDeviceSize& offset, uint32_t count = 1) const;
  VkDeviceSize appendBufferDescriptors(VkDeviceSize& offset, uint32_t count = 1) const;

  // Append the driver-owned reserved range after all descriptors and return its start offset.
  // REQUIRED for any heap that will be bound via vkCmdBind*HeapEXT: the driver writes into
  // this range. Skipping it risks out-of-bounds writes when minXxxHeapReservedRange() > 0.
  VkDeviceSize appendSamplerReservedRange(VkDeviceSize& offset) const;
  VkDeviceSize appendResourceReservedRange(VkDeviceSize& offset) const;

  // Align offset to the heap buffer alignment (call after appending everything,
  // including the reserved range, to get the final heap buffer size).
  VkDeviceSize alignToSamplerHeap(VkDeviceSize offset) const;
  VkDeviceSize alignToResourceHeap(VkDeviceSize offset) const;

  // --- Write raw descriptor bytes to caller memory ---
  // Thin wrappers over vkWrite*DescriptorsEXT; returns its VkResult unchanged.
  // `dst` must point to at least the corresponding *DescriptorSize() bytes
  // (or *_DESCRIPTOR_MAX_BYTE_SIZE for stack-allocated buffers).
  VkResult writeSamplerDescriptor(const VkSamplerCreateInfo& samplerCreateInfo, void* dst) const;
  VkResult writeImageDescriptor(VkImage image, VkFormat format, VkImageLayout layout, void* dst, VkImageViewType viewType = VK_IMAGE_VIEW_TYPE_2D) const;
  VkResult writeBufferDescriptor(VkDeviceAddress bufferAddress, VkDeviceSize bufferSize, VkDescriptorType type, void* dst) const;

  bool isInitialized() const { return m_device != nullptr; }

private:
  VkDevice m_device{};

  VkDeviceSize m_samplerDescSize{};
  VkDeviceSize m_imageDescSize{};
  VkDeviceSize m_bufferDescSize{};
  VkDeviceSize m_samplerDescAlignment{};
  VkDeviceSize m_imageDescAlignment{};
  VkDeviceSize m_bufferDescAlignment{};

  VkDeviceSize m_samplerHeapAlignment{};
  VkDeviceSize m_resourceHeapAlignment{};
  VkDeviceSize m_maxSamplerHeapSize{};
  VkDeviceSize m_maxResourceHeapSize{};
  VkDeviceSize m_minSamplerReservedRange{};
  VkDeviceSize m_minResourceReservedRange{};
};

}  // namespace nvvk
