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

#include <cassert>
#include <unordered_map>
#include <vector>

#include <vulkan/vulkan_core.h>

#include "descriptor_heap_writer.hpp"
#include "resources.hpp"
#include "sampler_pool.hpp"

namespace nvvk {

//////////////////////////////////////////////////////////////////////////

// Higher-level descriptor heap helper (VK_EXT_descriptor_heap)
//
// Builds on DescriptorHeapWriter (low-level sizes, writes) and adds:
// - Uniform-stride resource heap management (index-based)
// - Sampler dedup (SamplerPool-style ref-counted acquire/release)
// - Dirty range tracking for incremental uploads
// - Heap bind commands
//
// Uses `max(imageDescriptorSize, bufferDescriptorSize)` as the uniform
// resource descriptor stride, so all resource slots (images and buffers)
// share the same index space.  For packed mixed-type layouts with
// different strides per type, use DescriptorHeapWriter directly.
//
// Usage:
//   see usage_DescriptorHeap() in descriptor_heap.cpp
class DescriptorHeap
{
public:
  DescriptorHeap()                                 = default;
  DescriptorHeap(const DescriptorHeap&)            = delete;
  DescriptorHeap& operator=(const DescriptorHeap&) = delete;
  DescriptorHeap(DescriptorHeap&&)                 = delete;
  DescriptorHeap& operator=(DescriptorHeap&&)      = delete;

  ~DescriptorHeap() { assert(!m_info.isInitialized() && "deinit() was not called before destruction"); }

  // Queries heap properties (descriptor sizes, alignment, capacity).
  // Does not allocate staging — call setupSamplerHeap / setupResourceHeap next.
  // Returns `VK_ERROR_FEATURE_NOT_PRESENT` if the extension is unavailable.
  VkResult init(VkPhysicalDevice physicalDevice, VkDevice device);
  // Clears staging and state. Safe to call multiple times.
  void deinit();

  // Hardware limits (derived from maxSamplerHeapSize / maxResourceHeapSize).
  // Available after init().
  uint32_t maxSamplerCapacity() const { return m_maxSamplerCapacity; }
  uint32_t maxResourceCapacity() const { return m_maxResourceCapacity; }

  // Computes heap sizes for `maxSamplers` sampler descriptors.
  // Returns the heap buffer size the caller must allocate.
  VkDeviceSize setupSamplerHeap(uint32_t maxSamplers);
  // Computes heap sizes for `maxResources` resource (image) descriptors.
  // Returns the heap buffer size the caller must allocate.
  // Can be called again to resize; the caller must recreate their GPU buffer.
  VkDeviceSize setupResourceHeap(uint32_t maxResources);

  // Low-level info (per-type sizes, alignments, raw write functions).
  const DescriptorHeapWriter& info() const { return m_info; }

  // --- Size / alignment for user buffer creation ---
  VkDeviceSize getSamplerHeapSize() const { return m_samplerHeapSize; }
  VkDeviceSize getResourceHeapSize() const { return m_resourceHeapSize; }
  VkDeviceSize getSamplerHeapAlignment() const { return m_info.samplerHeapAlignment(); }
  VkDeviceSize getResourceHeapAlignment() const { return m_info.resourceHeapAlignment(); }

  static constexpr VkBufferUsageFlags2KHR getRequiredBufferUsage()
  {
    return DescriptorHeapWriter::getRequiredBufferUsage();
  }

  // --- Sampler descriptors: deduplicated acquire/release (primary API) ---
  // Returns the heap index for `samplerCreateInfo`.  Identical create-infos
  // share the same slot (ref-counted).
  // `samplerHeapBase` is the writable base pointer where descriptor data is written.
  uint32_t acquireSamplerDescriptor(const VkSamplerCreateInfo& samplerCreateInfo, void* samplerHeapBase);
  void     releaseSamplerDescriptor(uint32_t index);

  // --- Sampler descriptors: explicit index (low-level) ---
  // `samplerHeapBase` is the writable base pointer for the sampler heap.
  VkResult writeSamplerDescriptor(uint32_t index, const VkSamplerCreateInfo& samplerCreateInfo, void* samplerHeapBase);

  // --- Resource (image) descriptors: explicit index ---
  // `resourceHeapBase` is the writable base pointer for the resource heap.
  VkResult writeImageDescriptor(uint32_t        index,
                                VkImage         image,
                                VkFormat        format,
                                VkImageLayout   layout,
                                void*           resourceHeapBase,
                                VkImageViewType viewType = VK_IMAGE_VIEW_TYPE_2D);
  VkResult writeImageDescriptor(uint32_t index, const Image& image, void* resourceHeapBase, VkImageViewType viewType = VK_IMAGE_VIEW_TYPE_2D)
  {
    return writeImageDescriptor(index, image.image, image.format, image.descriptor.imageLayout, resourceHeapBase, viewType);
  }

  // --- Resource (buffer) descriptors: explicit index ---
  // `type` must be VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER or VK_DESCRIPTOR_TYPE_STORAGE_BUFFER.
  VkResult writeBufferDescriptor(uint32_t index, VkDeviceAddress bufferAddress, VkDeviceSize bufferSize, VkDescriptorType type, void* resourceHeapBase);
  VkResult writeBufferDescriptor(uint32_t index, const Buffer& buffer, VkDescriptorType type, void* resourceHeapBase)
  {
    return writeBufferDescriptor(index, buffer.address, buffer.bufferSize, type, resourceHeapBase);
  }

  // --- Dirty range tracking ---
  // Returns the byte range modified since the last clear (or setup).
  // Offset/size are relative to heap base. Size is 0 if no writes occurred.
  struct DirtyRange
  {
    VkDeviceSize offset{};
    VkDeviceSize size{};
  };
  DirtyRange getSamplerDirtyRange() const;
  DirtyRange getResourceDirtyRange() const;
  void       clearSamplerDirty();
  void       clearResourceDirty();

  // Binds user-owned heaps to `cmd`.  Pass 0 for an address to skip that heap.
  void cmdBindHeaps(VkCommandBuffer cmd, VkDeviceAddress samplerHeapAddr, VkDeviceAddress resourceHeapAddr) const;

  // The uniform stride used for all resource slots: max(imageDescSize, bufferDescSize).
  VkDeviceSize resourceDescriptorStride() const { return m_resourceDescStride; }
  VkDeviceSize samplerDescriptorSize() const { return m_info.samplerDescriptorSize(); }
  uint32_t     maxSamplerCount() const { return m_maxSamplers; }
  uint32_t     maxResourceCount() const { return m_maxResources; }

  bool isInitialized() const { return m_info.isInitialized(); }

  static bool isSupported(VkPhysicalDevice physicalDevice) { return DescriptorHeapWriter::isSupported(physicalDevice); }

private:
  DescriptorHeapWriter m_info;

  // max(imageDescSize, bufferDescSize) — uniform stride for the index-based API
  VkDeviceSize m_resourceDescStride{};

  VkDeviceSize m_samplerHeapSize{};
  VkDeviceSize m_resourceHeapSize{};
  uint32_t     m_maxSamplers{};
  uint32_t     m_maxResources{};
  uint32_t     m_maxSamplerCapacity{};
  uint32_t     m_maxResourceCapacity{};

  // Dirty range tracking (index-based, converted to bytes by getDirtyRange)
  uint32_t m_samplerDirtyMin{~0u};
  uint32_t m_samplerDirtyMax{0};
  uint32_t m_resourceDirtyMin{~0u};
  uint32_t m_resourceDirtyMax{0};

  // Sampler dedup (SamplerPool-style)
  struct SamplerSlot
  {
    uint32_t heapIndex{};
    uint32_t refCount{};
  };
  std::unordered_map<SamplerState, SamplerSlot, SamplerStateHashFn> m_samplerMap;
  std::unordered_map<uint32_t, SamplerState>                        m_samplerIndexToState;
  std::vector<uint32_t>                                             m_freeSamplerSlots;
  uint32_t                                                          m_nextSamplerSlot{};
};

}  // namespace nvvk
