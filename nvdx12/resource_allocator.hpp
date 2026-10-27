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

#pragma once

#include <atomic>
#include <cstdint>

#include "basic_types.hpp"  // first: Windows headers with NOMINMAX

#include <D3D12MemAlloc.h>

#include "descriptor_heap.hpp"
#include "resources.hpp"

namespace nvdx12 {

/*-------------------------------------------------------------------------------------------------
# class nvdx12::ResourceAllocator

> Creates and destroys buffers, images and acceleration structures, the D3D12
> counterpart of nvvk::ResourceAllocator, with the same entry points.

Memory comes from the D3D12 Memory Allocator (third_party/d3d12ma), as
nvvk's comes from VMA: resources are placed in large heaps shared by many
resources, so creating and destroying one is cheap even by the tens of
thousands. Small buffers use tight alignment where the device supports it;
render-target and depth-stencil images stay committed, since a placed one
must be cleared or discarded before its first use. Resources are created
with the enhanced barrier layout model (CreateResource3). Unlike committed
resources, placed ones are not zero-filled: memory reused from a destroyed
resource keeps its contents, so write a resource before reading it.

Buffers receive raw UAV descriptors in the DescriptorHeap and therefore a
shader address (see resources.hpp). Destruction is immediate, as with nvvk:
callers wait for the GPU before destroying resources that may still be in
flight, and destroy every resource before deinit().

Differences with nvvk that follow from D3D12:
- The create infos are D3D12_RESOURCE_DESC1 / D3D12MA::ALLOCATION_DESC
  instead of VkBufferCreateInfo / VmaAllocationCreateInfo; a buffer's
  `usage` is its D3D12_RESOURCE_FLAGS, and images take their initial layout
  separately since D3D12_RESOURCE_DESC1 has none.
- Image views are descriptors: write them with
  DescriptorHeap::writeSampledImageDescriptor / writeStorageImageDescriptor
  (or writeImageSlot for an SRV / UAV pair) after createImage.
- Mapped memory is always coherent, so flushBuffer and friends do nothing.
- There are no large (sparse) buffers: a D3D12 resource is bounded by its
  GPU virtual address range (getMaxMemoryAllocationSize), not by an
  allocation limit like Vulkan's maxMemoryAllocationSize.
- Debug names are set after creation, with NVDX12_DBG_NAME as with
  NVVK_DBG_NAME.

```cpp
nvdx12::ResourceAllocator allocator;
allocator.init({.pDevice = device}, &descriptorHeap);

nvdx12::Buffer buffer;
allocator.createBuffer(buffer, 1024);
NVDX12_DBG_NAME(buffer.buffer);
...
allocator.destroyBuffer(buffer);
allocator.deinit();
```

See also FrameUploader and AsyncUploader for uploading data to the GPU.

See usage_ResourceAllocator() in resource_allocator.cpp.
-------------------------------------------------------------------------------------------------*/

// Size and kind of an acceleration structure, the counterpart of
// VkAccelerationStructureCreateInfoKHR. Top-level structures (TLAS, PTLAS)
// receive an SRV slot so that shaders can trace against them.
struct AccelerationStructureCreateInfo
{
  D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE type{D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL};
  uint64_t                                     size{0};
};

class ResourceAllocator
{
public:
  ResourceAllocator()                                    = default;
  ResourceAllocator(const ResourceAllocator&)            = delete;
  ResourceAllocator& operator=(const ResourceAllocator&) = delete;
  ResourceAllocator(ResourceAllocator&& other) noexcept;
  ResourceAllocator& operator=(ResourceAllocator&& other) noexcept;
  ~ResourceAllocator();

  // D3D12MA places every buffer on at least this many bytes; createBuffer
  // rejects a larger minAlignment, and sub-allocators carry any larger
  // alignment in their offsets.
  static constexpr uint64_t BUFFER_PLACEMENT_ALIGNMENT = D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT;

  operator D3D12MA::Allocator*() const { return m_allocator; }

  // Initialization of the D3D12 Memory Allocator. `allocatorDesc.pDevice` is
  // required (ID3D12Device10 or newer); a null `pAdapter` is found from the
  // device. Buffers get their raw UAVs, and top-level acceleration structures
  // their SRVs, in `descriptorHeap`.
  HRESULT init(D3D12MA::ALLOCATOR_DESC allocatorDesc, DescriptorHeap* descriptorHeap);

  // De-initialization of the D3D12 Memory Allocator.
  void deinit();

  ID3D12Device10* getDevice() const { return m_device.Get(); }
  IDXGIAdapter*   getAdapter() const { return m_adapter.Get(); }  // counterpart of getPhysicalDevice
  uint64_t        getMaxMemoryAllocationSize() const { return m_maxMemoryAllocationSize; }
  DescriptorHeap* getDescriptorHeap() const { return m_heap; }

  //////////////////////////////////////////////////////////////////////////
  // Create a buffer
  //
  //   usage:        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS gives the
  //                 buffer raw UAV descriptors and so a shader address;
  //                 D3D12_RESOURCE_FLAG_RAYTRACING_ACCELERATION_STRUCTURE
  //                 allocates acceleration-structure storage (no UAV).
  //   memoryUsage:  eDevice, or eHostUpload / eHostReadback, persistently
  //                 mapped. UPLOAD and READBACK heaps reject unordered
  //                 access: eHostUpload buffers with a UAV go to the
  //                 GPU_UPLOAD heap when the device has one; otherwise, as
  //                 for eHostReadback, the buffer gets no UAV and its
  //                 `address` stays 0.
  //   flags:        D3D12MA::ALLOCATION_FLAGS, e.g. ALLOCATION_FLAG_COMMITTED.
  //   minAlignment: buffers are placed on at least BUFFER_PLACEMENT_ALIGNMENT
  //                 (256) bytes, which covers every D3D12 requirement;
  //                 larger values are rejected.
  //
  HRESULT createBuffer(Buffer&                   buffer,
                       uint64_t                  size,
                       D3D12_RESOURCE_FLAGS      usage        = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                       MemoryUsage               memoryUsage  = MemoryUsage::eDevice,
                       D3D12MA::ALLOCATION_FLAGS flags        = D3D12MA::ALLOCATION_FLAG_NONE,
                       uint64_t                  minAlignment = 0) const;

  // This allows more fine control. `allocDesc.HeapType` selects the memory
  // usage: DEFAULT, UPLOAD / GPU_UPLOAD or READBACK.
  HRESULT createBuffer(Buffer&                         buffer,
                       const D3D12_RESOURCE_DESC1&     bufferDesc,
                       const D3D12MA::ALLOCATION_DESC& allocDesc,
                       uint64_t                        minAlignment = 0) const;

  // Releases the descriptor slots, unmaps and destroys the resource and its
  // memory.
  void destroyBuffer(Buffer& buffer) const;

  // Creates an image in device memory, in `initialLayout`. MipLevels 0
  // requests the full chain.
  HRESULT createImage(Image&                      image,
                      const D3D12_RESOURCE_DESC1& imageDesc,
                      D3D12_BARRIER_LAYOUT        initialLayout = D3D12_BARRIER_LAYOUT_UNDEFINED) const;

  // Creates an image with provided allocation information.
  HRESULT createImage(Image&                          image,
                      const D3D12_RESOURCE_DESC1&     imageDesc,
                      D3D12_BARRIER_LAYOUT            initialLayout,
                      const D3D12MA::ALLOCATION_DESC& allocDesc) const;

  // Destroys the image and its memory; its descriptors belong to the caller.
  void destroyImage(Image& image) const;

  // Acceleration-structure storage of `accInfo.size` bytes.
  HRESULT createAcceleration(AccelerationStructure& accel, const AccelerationStructureCreateInfo& accInfo) const;
  HRESULT createAcceleration(AccelerationStructure&                 accel,
                             const AccelerationStructureCreateInfo& accInfo,
                             const D3D12MA::ALLOCATION_DESC&        allocDesc) const;
  void    destroyAcceleration(AccelerationStructure& accel) const;

  //////////////////////////////////////////////////////////////////////////

  // When leaks are reported, set the ID of the leak here: the creation that
  // makes allocation "nvdx12AllocID: <id>" breaks into the debugger.
  void setLeakID(uint32_t id);

  // Returns the heap the allocation is placed in, null for a committed
  // resource (counterpart of getDeviceMemory).
  ID3D12Heap* getHeap(D3D12MA::Allocation* allocation) const;

  //////////////////////////////////////////////////////////////////////////
  // Mapped memory. D3D12 upload and readback memory is always coherent: these
  // exist for code written against nvvk and do nothing.

  bool    isNonCoherentlyMapped(const Buffer& /*buffer*/) const { return false; }
  HRESULT flushBuffer(const Buffer& /*buffer*/, uint64_t /*offset*/ = 0, uint64_t /*size*/ = ~0ull) { return S_OK; }
  HRESULT invalidateBuffer(const Buffer& /*buffer*/, uint64_t /*offset*/ = 0, uint64_t /*size*/ = ~0ull)
  {
    return S_OK;
  }
  HRESULT autoFlushBuffer(const Buffer& /*buffer*/, uint64_t /*offset*/ = 0, uint64_t /*size*/ = ~0ull) { return S_OK; }
  HRESULT autoInvalidateBuffer(const Buffer& /*buffer*/, uint64_t /*offset*/ = 0, uint64_t /*size*/ = ~0ull)
  {
    return S_OK;
  }

  //////////////////////////////////////////////////////////////////////////

  // Byte totals of live resources.
  uint64_t allocatedBufferBytes() const { return m_bufferBytes.load(std::memory_order_relaxed); }
  uint64_t allocatedImageBytes() const { return m_imageBytes.load(std::memory_order_relaxed); }

private:
  // Names the allocation "nvdx12AllocID: <n>", the name D3D12MA reports for
  // unfreed allocations, and breaks on the ID set with setLeakID.
  void addLeakDetection(D3D12MA::Allocation* allocation) const;

  // One raw UAV per 4 GiB window of `buffer`; sets uavSlot, uavSlotCount and address.
  HRESULT writeBufferDescriptors(Buffer& buffer) const;

  ComPtr<ID3D12Device10> m_device;
  ComPtr<IDXGIAdapter>   m_adapter;
  D3D12MA::Allocator*    m_allocator{nullptr};
  DescriptorHeap*        m_heap{nullptr};
  uint64_t               m_maxMemoryAllocationSize{0};  // GPU virtual address range of one resource

  // Leak detection, see setLeakID.
  mutable std::atomic<uint32_t> m_allocationCounter{0};
  uint32_t                      m_leakID{~0u};

  mutable std::atomic<uint64_t> m_bufferBytes{0};  // live buffer bytes
  mutable std::atomic<uint64_t> m_imageBytes{0};   // live image bytes
};

}  // namespace nvdx12
