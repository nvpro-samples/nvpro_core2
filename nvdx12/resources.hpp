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

#include <cassert>
#include <cstdint>

#include "basic_types.hpp"

// Forward declared so that users don't need to include D3D12MemAlloc.h.
namespace D3D12MA {
class Allocation;
}

namespace nvdx12 {

/*-------------------------------------------------------------------------------------------------
# Resources

The D3D12 counterparts of nvvk::Buffer, nvvk::Image and
nvvk::AccelerationStructure, created and destroyed by
nvdx12::ResourceAllocator (resource_allocator.hpp). The native object is
an ID3D12Resource, named after the struct as in nvvk (`Buffer::buffer`,
`Image::image`). Unlike nvvk's plain handles it is reference counted: a copy
of a Buffer or Image keeps the resource object alive, though not its memory,
which destroyBuffer / destroyImage free.

Buffers carry two 64-bit addresses:

- `address` is what shaders consume. HLSL has no pointers, so a buffer
  reference is `(descriptorIndex << 32) | byteOffset` over a raw UAV in the
  shader-visible heap (descriptor_heap.hpp): shaders index
  `ResourceDescriptorHeap[uint(address >> 32)]` and load at the low 32 bits
  (shaders/bindless.h). A buffer over 4 GiB owns one consecutive slot per
  4 GiB window, so 64-bit arithmetic on `address` carries into the right
  window; the addr* accessors add their index in 32 bits, within one window.
- `gpuAddress` is the D3D12_GPU_VIRTUAL_ADDRESS the API takes (acceleration
  structure inputs, indirect arguments, shader tables).
-------------------------------------------------------------------------------------------------*/

constexpr uint32_t kInvalidSlot = ~0u;  // descriptor-heap index of "no descriptor"

// The descriptor pair of an image on the heap: SRV over the mip chain at
// `srv`, UAV of one mip at `srv + 1`. Shaders receive `srv`; HEAP_SRV and
// HEAP_UAV in shaders/bindless.h add the one. DescriptorHeap::acquireImageSlot
// hands them out; single descriptors (buffers, SRV-only images) are plain slots.
struct ImageSlot
{
  uint32_t srv{kInvalidSlot};

  uint32_t uav() const { return srv == kInvalidSlot ? kInvalidSlot : srv + 1; }
  bool     valid() const { return srv != kInvalidSlot; }
  bool     operator==(const ImageSlot&) const = default;
};

// Memory class a buffer lives in; the D3D12 heap type is derived from it.
enum class MemoryUsage : uint32_t
{
  eDevice,        // D3D12_HEAP_TYPE_DEFAULT; `mapping` is null
  eHostUpload,    // D3D12_HEAP_TYPE_UPLOAD (GPU_UPLOAD when shaders address it); persistently mapped
  eHostReadback,  // D3D12_HEAP_TYPE_READBACK; persistently mapped
};

//-----------------------------------------------------------------
// A buffer is a region of memory for vertex, index, uniform or any other
// data. `buffer` is the D3D12 object; `address` is what shaders
// dereference and `gpuAddress` what the API takes.
//-----------------------------------------------------------------
struct Buffer
{
  ComPtr<ID3D12Resource> buffer;
  // Memory of the resource, freed by ResourceAllocator::destroyBuffer. Null
  // for buffers the allocator did not create.
  D3D12MA::Allocation* allocation{nullptr};
  uint64_t             bufferSize{0};
  // Shader address: (uavSlot << 32). Store in shader I/O structs.
  uint64_t address{0};
  // GPU virtual address for the D3D12 API.
  D3D12_GPU_VIRTUAL_ADDRESS gpuAddress{0};
  // Persistent mapping for host-visible memory, null for device memory.
  uint8_t* mapping{nullptr};
  // First raw UAV slot and number of consecutive slots (one per 4 GiB window).
  uint32_t    uavSlot{kInvalidSlot};
  uint32_t    uavSlotCount{0};
  MemoryUsage memoryUsage{MemoryUsage::eDevice};
  // Last synchronization scope, maintained by the tracking barrier helpers.
  // A new resource has no prior access; D3D12 requires NO_ACCESS with SYNC_NONE.
  D3D12_BARRIER_SYNC   stageMask{D3D12_BARRIER_SYNC_NONE};
  D3D12_BARRIER_ACCESS accessMask{D3D12_BARRIER_ACCESS_NO_ACCESS};

  explicit operator bool() const { return buffer != nullptr; }

  // Shader address of a byte offset inside the buffer.
  uint64_t addressAt(uint64_t byteOffset) const { return address + byteOffset; }
  // GPU virtual address of a byte offset inside the buffer.
  D3D12_GPU_VIRTUAL_ADDRESS gpuAddressAt(uint64_t byteOffset) const { return gpuAddress + byteOffset; }
};

//-----------------------------------------------------------------
// A BufferTyped allows to specify if a buffer represents only
// data of a certain type. It's interchangeable with regular Buffer struct,
// and mostly meant to aid code readability if buffers are strictly typed.
//-----------------------------------------------------------------
template <class T>
struct BufferTyped : Buffer
{
  static constexpr size_t value_size = sizeof(T);
  typedef T               value_type;

  BufferTyped& operator=(Buffer other)
  {
    static_cast<Buffer&>(*this) = other;
    return *this;
  }

  size_t   size() const { return bufferSize / sizeof(T); }
  const T* data() const { return reinterpret_cast<const T*>(mapping); }
  T*       data() { return reinterpret_cast<T*>(mapping); }

  // Shader address of the element at `startIndex`; `num` is used for range checking.
  uint64_t addressAt(size_t startIndex, size_t num = 1) const
  {
    assert(startIndex + num <= size());
    return address + sizeof(T) * startIndex;
  }
  // GPU virtual address of the element at `startIndex`; `num` is used for range checking.
  D3D12_GPU_VIRTUAL_ADDRESS gpuAddressAt(size_t startIndex, size_t num = 1) const
  {
    assert(startIndex + num <= size());
    return gpuAddress + sizeof(T) * startIndex;
  }
};

//-----------------------------------------------------------------
// A range within a buffer, the counterpart of nvvk::BufferRange
// (VkDescriptorBufferInfo + address + mapping), handed out by the
// sub-allocators (BufferSubAllocator::subRange,
// BufferCircularAllocator::subRange) and taken by the uploaders.
//
// - `buffer` is the resource, not owned: the Buffer the range comes from
//   keeps it alive. `offset` is the byte offset inside that resource, what
//   CopyBufferRegion and views take; `range` is the size in bytes.
// - `address`, `gpuAddress` and `mapping` already contain the offset, as in
//   nvvk. `address` is the shader address of the range's first byte:
//   `Buffer::address` plus the offset from the start of that Buffer, so it
//   keeps its form, `(uavSlot << 32) | byteOffset`. It is 0 when the buffer has no
//   shader address (no UAV, readback memory). `mapping` is null for device
//   memory.
//
// makeBufferRange builds one from a Buffer.
//-----------------------------------------------------------------
struct BufferRange
{
  ID3D12Resource*           buffer{nullptr};   // the resource, not owned
  uint64_t                  offset{0};         // byte offset inside `buffer`
  uint64_t                  range{0};          // size in bytes
  uint64_t                  address{0};        // shader address, contains offset already; 0 if none
  D3D12_GPU_VIRTUAL_ADDRESS gpuAddress{0};     // contains offset already
  uint8_t*                  mapping{nullptr};  // contains offset already; null for device memory
};

// The range [byteOffset, byteOffset + size) of `buffer`; a `size` of ~0 runs
// to the end of the buffer.
inline BufferRange makeBufferRange(const Buffer& buffer, uint64_t byteOffset = 0, uint64_t size = ~0ull)
{
  assert(byteOffset <= buffer.bufferSize);
  BufferRange range;
  range.buffer     = buffer.buffer.Get();
  range.offset     = byteOffset;
  range.range      = size == ~0ull ? buffer.bufferSize - byteOffset : size;
  range.address    = buffer.address ? buffer.addressAt(byteOffset) : 0;
  range.gpuAddress = buffer.gpuAddress ? buffer.gpuAddressAt(byteOffset) : 0;
  range.mapping    = buffer.mapping ? buffer.mapping + byteOffset : nullptr;
  assert(byteOffset + range.range <= buffer.bufferSize);
  return range;
}

// A BufferRangeTyped allows to specify if a BufferRange represents only
// data of a certain type. It's interchangeable with regular BufferRange struct,
// and mostly meant to aid code readability if buffer ranges are strictly typed.
template <class T>
struct BufferRangeTyped : BufferRange
{
  static constexpr size_t value_size = sizeof(T);
  typedef T               value_type;

  BufferRangeTyped& operator=(BufferRange other)
  {
    static_cast<BufferRange&>(*this) = other;
    return *this;
  }

  size_t   size() const { return range / sizeof(T); }
  const T* data() const { return reinterpret_cast<const T*>(mapping); }
  T*       data() { return reinterpret_cast<T*>(mapping); }

  // Shader address of the element at `startIndex`; `num` is used for range checking.
  uint64_t addressAt(size_t startIndex, size_t num = 1) const
  {
    assert(startIndex + num <= size());
    return address + sizeof(T) * startIndex;
  }
  // GPU virtual address of the element at `startIndex`; `num` is used for range checking.
  D3D12_GPU_VIRTUAL_ADDRESS gpuAddressAt(size_t startIndex, size_t num = 1) const
  {
    assert(startIndex + num <= size());
    return gpuAddress + sizeof(T) * startIndex;
  }
};

// Image size in texels.
struct Extent2D
{
  uint32_t width{0};
  uint32_t height{0};
};

// Image size in texels; `depth` is 1 for anything but 3D images.
struct Extent3D
{
  uint32_t width{0};
  uint32_t height{0};
  uint32_t depth{1};
};

// Texel offset inside an image subresource, the counterpart of VkOffset3D.
struct Offset3D
{
  int32_t x{0};
  int32_t y{0};
  int32_t z{0};
};

//-----------------------------------------------------------------
// An image is a region of memory for texture, render-target and other
// image data. Its descriptors live in the DescriptorHeap; `layout` and the
// sync / access masks are maintained by the tracking barrier helpers.
//-----------------------------------------------------------------
struct Image
{
  ComPtr<ID3D12Resource> image;
  // Memory of the resource, freed by ResourceAllocator::destroyImage. Null for
  // images the allocator did not create (swap-chain back buffers).
  D3D12MA::Allocation*     allocation{nullptr};
  D3D12_RESOURCE_DIMENSION dimension{D3D12_RESOURCE_DIMENSION_TEXTURE2D};
  DXGI_FORMAT              format{DXGI_FORMAT_UNKNOWN};
  Extent3D                 extent{};
  uint32_t                 mipLevels{1};
  uint32_t                 arrayLayers{1};
  uint32_t                 sampleCount{1};  // MSAA samples; multisampled images have no UAV
  D3D12_RESOURCE_FLAGS     flags{D3D12_RESOURCE_FLAG_NONE};
  // Current layout of every subresource and the last synchronization scope,
  // maintained by the barrier helpers (no prior access for a new image).
  D3D12_BARRIER_LAYOUT layout{D3D12_BARRIER_LAYOUT_UNDEFINED};
  D3D12_BARRIER_SYNC   stageMask{D3D12_BARRIER_SYNC_NONE};
  D3D12_BARRIER_ACCESS accessMask{D3D12_BARRIER_ACCESS_NO_ACCESS};

  explicit operator bool() const { return image != nullptr; }
  bool     allowsUav() const { return (flags & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS) != 0; }
  // D3D12 subresource index of (mip, layer) for single-plane formats.
  uint32_t subresourceIndex(uint32_t mipLevel, uint32_t arrayLayer = 0) const
  {
    return mipLevel + arrayLayer * mipLevels;
  }
};

//-----------------------------------------------------------------
// A ray-tracing acceleration structure (BLAS, TLAS, OMM array, ...). The
// backing buffer is created with RAYTRACING_ACCELERATION_STRUCTURE and is
// never transitioned. `gpuAddress` is what the API and instance records use;
// `shaderHandle()` is what shaders trace against.
//-----------------------------------------------------------------
struct AccelerationStructure
{
  Buffer                    buffer;
  D3D12_GPU_VIRTUAL_ADDRESS gpuAddress{0};  // == buffer.gpuAddress
  uint64_t                  size{0};
  // SRV slot (RaytracingAccelerationStructure view) of a structure shaders
  // trace against; kInvalidSlot for a BLAS.
  uint32_t srvSlot{kInvalidSlot};

  explicit operator bool() const { return buffer.buffer != nullptr; }
  // Shader handle of a traceable structure: `ResourceDescriptorHeap[uint(h >> 32)]`.
  uint64_t shaderHandle() const { return uint64_t(srvSlot) << 32; }
};

//-----------------------------------------------------------------
// A command queue and where it comes from, the counterpart of
// nvvk::QueueInfo. D3D12 has no queue families: `familyIndex` holds the
// D3D12_COMMAND_LIST_TYPE of the queue (the lists it accepts), and
// `queueIndex` counts the queues of that type in
// nvdx12::ContextInitInfo::queues. Reference counted like Buffer::buffer.
//-----------------------------------------------------------------
struct QueueInfo
{
  uint32_t                   familyIndex = ~0U;  // D3D12_COMMAND_LIST_TYPE of the queue
  uint32_t                   queueIndex  = ~0U;  // Index of the queue among those of its type
  ComPtr<ID3D12CommandQueue> queue{};            // The queue object
};

// A fence and a value on it, the counterpart of nvvk::SemaphoreInfo
// (timeline semaphore + value).
struct SemaphoreInfo
{
  ComPtr<ID3D12Fence> semaphore{};
  uint64_t            value{};
};

}  // namespace nvdx12
