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
#include <limits>
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
// - Resource heap with **two dense regions**: sampled images first, then buffer
//   descriptors (UBO/SSBO), each at the hardware **per-type** stride.
// - Sampler dedup (SamplerPool-style ref-counted acquire/release)
// - Dirty range tracking for incremental uploads
// - Heap bind commands
// - **Shader index bases** (`imageShaderIndexBase`, `bufferShaderIndexBase`) so Slang /
//   SPIR-V bindless paths that index by `OpConstantSizeOfEXT(ResourceType)` align with
//   host slot indices **within each region** (local index 0..N-1 per type).
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


  // Hardware limits (derived from maxResourceHeapSize). Available after init().
  uint32_t maxSamplerCapacity() const { return m_maxSamplerCapacity; }
  // Loose upper bound on image slots if the entire heap were images (from `m_availableCapacity`).
  // Mixed image+buffer heaps use more bytes per slot due to alignment; use `setupResourceHeap` sizing instead.
  uint32_t maxImageCapacity() const { return m_maxImageCapacity; }
  // Loose upper bound on buffer slots if the entire heap were buffers (from `m_availableCapacity`).
  uint32_t maxBufferCapacity() const { return m_maxBufferCapacity; }

  // After init: `maxResourceHeapSize - minResourceHeapReservedRange()` (saturated when max is below min). Loose single-type hints only.
  VkDeviceSize availableCapacity() const { return m_availableCapacity; }

  // After init: returns whether `maxImages` / `maxBuffers` fit the resource heap using the same packed layout as `setupResourceHeap`.
  // Both counts zero yields false. Otherwise compares packed size to `maxResourceHeapSize`.
  bool testMixedCapacity(uint32_t maxImages, uint32_t maxBuffers);

  // Computes heap sizes for `maxSamplers` sampler descriptors.
  // Returns the heap buffer size the caller must allocate.
  VkDeviceSize setupSamplerHeap(uint32_t maxSamplers);

  // Computes heap sizes for `maxImages` sampled-image descriptors followed by
  // `maxBuffers` buffer descriptors (images first). Returns the heap buffer size.
  // Use `(N, 0)` for image-only heaps or `(0, M)` for buffer-only heaps.
  // At least one of maxImages / maxBuffers must be non-zero.
  // Indices passed to writeImageDescriptor / writeBufferDescriptor are **local**:
  // `0 .. maxImages-1` and `0 .. maxBuffers-1` respectively.
  // Capacity is validated using the packed append layout (per-type alignment, reserved range, heap alignment), not `maxImageCapacity`/`maxBufferCapacity`.
  // Can be called again to resize; the caller must recreate their GPU buffer.
  // We recommend not to exceed 1M active descriptors and 2K samplers.
  VkDeviceSize setupResourceHeap(uint32_t maxImages, uint32_t maxBuffers);

  // Returns the base index for the image or buffer region in the resource heap, as visible in shaders:
  // - In Slang: this is the `Handle.x` offset at which images or buffers begin within the heap (divide region byte offset by descriptor size).
  // - In GLSL: this is the starting index for `heapResource[]` access for the given resource type.
  // Indices are already adjusted for per-type descriptor size—compatible with SPIR-V/Slang indexing.
  uint32_t imageShaderIndexBase() const;
  uint32_t bufferShaderIndexBase() const;
  // Byte offset of the image region / buffer region within the bound resource heap buffer.
  VkDeviceSize imageRegionByteOffset() const { return m_imageRegionStartBytes; }
  VkDeviceSize bufferRegionByteOffset() const { return m_bufferRegionStartBytes; }

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

  // --- Resource (image) descriptors: local image index in [0, maxImages) ---
  // `resourceHeapBase` is the writable base pointer for the resource heap.
  VkResult writeImageDescriptor(uint32_t        localImageIndex,
                                VkImage         image,
                                VkFormat        format,
                                VkImageLayout   layout,
                                void*           resourceHeapBase,
                                VkImageViewType viewType = VK_IMAGE_VIEW_TYPE_2D);
  VkResult writeImageDescriptor(uint32_t localImageIndex, const Image& image, void* resourceHeapBase, VkImageViewType viewType = VK_IMAGE_VIEW_TYPE_2D)
  {
    return writeImageDescriptor(localImageIndex, image.image, image.format, image.descriptor.imageLayout, resourceHeapBase, viewType);
  }

  // --- Resource (buffer) descriptors: local buffer index in [0, maxBuffers) ---
  // `type` must be VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER or VK_DESCRIPTOR_TYPE_STORAGE_BUFFER.
  VkResult writeBufferDescriptor(uint32_t         localBufferIndex,
                                 VkDeviceAddress  bufferAddress,
                                 VkDeviceSize     bufferSize,
                                 VkDescriptorType type,
                                 void*            resourceHeapBase);
  VkResult writeBufferDescriptor(uint32_t localBufferIndex, const Buffer& buffer, VkDescriptorType type, void* resourceHeapBase)
  {
    return writeBufferDescriptor(localBufferIndex, buffer.address, buffer.bufferSize, type, resourceHeapBase);
  }

  // --- Dirty range tracking ---
  // Returns byte ranges modified since the last clear (or setup). Offset/size are relative to the heap base.
  // Size is 0 if no writes occurred in that region. Resource heap uses separate image vs buffer ranges so a distant
  // image and buffer edit does not report one spanning rectangle across both regions.
  struct DirtyRange
  {
    VkDeviceSize offset{};
    VkDeviceSize size{};
  };
  DirtyRange getSamplerDirtyRange() const;
  DirtyRange getResourceImageDirtyRange() const;
  DirtyRange getResourceBufferDirtyRange() const;
  void       clearSamplerDirty();
  void       clearResourceDirty();

  // Binds user-owned heaps to `cmd`.  Pass 0 for an address to skip that heap.
  void cmdBindHeaps(VkCommandBuffer cmd, VkDeviceAddress samplerHeapAddr, VkDeviceAddress resourceHeapAddr) const;

  VkDeviceSize samplerDescriptorSize() const { return m_info.samplerDescriptorSize(); }
  uint32_t     maxSamplerCount() const { return m_maxSamplers; }
  uint32_t     maxImageCount() const { return m_maxImages; }
  uint32_t     maxBufferCount() const { return m_maxBuffers; }

  bool isInitialized() const { return m_info.isInitialized(); }

  static bool isSupported(VkPhysicalDevice physicalDevice) { return DescriptorHeapWriter::isSupported(physicalDevice); }

private:
  void markResourceImageDirty(VkDeviceSize byteOffset, VkDeviceSize descriptorByteSize);
  void markResourceBufferDirty(VkDeviceSize byteOffset, VkDeviceSize descriptorByteSize);

  DescriptorHeapWriter m_info;

  VkDeviceSize m_samplerHeapSize{};
  VkDeviceSize m_resourceHeapSize{};

  uint32_t m_maxSamplers{};
  uint32_t m_maxSamplerCapacity{};
  uint32_t m_maxImages{};
  uint32_t m_maxBuffers{};
  // Upper bounds if the heap were filled with only images or only buffers (derived from m_availableCapacity).
  uint32_t m_maxImageCapacity{};
  uint32_t m_maxBufferCapacity{};
  // Set in init to `maxResourceHeapSize() - minResourceHeapReservedRange()` (saturated); not comparable to packed mixed-layout bytes.
  VkDeviceSize m_availableCapacity{};

  VkDeviceSize m_imageRegionStartBytes{};
  VkDeviceSize m_bufferRegionStartBytes{};
  // Byte offset where the driver reserved range begins (`VkBindHeapInfoEXT::reservedRangeOffset`).
  VkDeviceSize m_reservedRangeOffsetBytes{};

  // Dirty range tracking (byte offsets into the resource heap; separate bounds per region)
  VkDeviceSize m_resourceImageDirtyMin{std::numeric_limits<VkDeviceSize>::max()};
  VkDeviceSize m_resourceImageDirtyMax{};
  VkDeviceSize m_resourceBufferDirtyMin{std::numeric_limits<VkDeviceSize>::max()};
  VkDeviceSize m_resourceBufferDirtyMax{};

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

  uint32_t m_samplerDirtyMin{std::numeric_limits<uint32_t>::max()};
  uint32_t m_samplerDirtyMax{};
};

}  // namespace nvvk
