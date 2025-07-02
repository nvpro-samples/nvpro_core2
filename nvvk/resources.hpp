/*
 * Copyright (c) 2023-2025, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cassert>
#include <vector>

#include <vulkan/vulkan_core.h>
// Forward declare VmaAllocation so we don't need to pull in vk_mem_alloc.h
VK_DEFINE_HANDLE(VmaAllocation)

namespace nvvk {

//-----------------------------------------------------------------
// A buffer is a region of memory used to store data.
// It is used to store vertex data, index data, uniform data, and other types of data.
// There is a VkBuffer object that represents the buffer, and a VmaAllocation object that represents the memory allocation.
// The address is used to access the buffer in the shader.
//
struct Buffer
{
  VkBuffer        buffer{};  // Vulkan Buffer
  VkDeviceSize    bufferSize{};
  VkDeviceAddress address{};  // Address of the buffer in the shader
  uint8_t*        mapping{};
  VmaAllocation   allocation{};  // Memory associated with the buffer
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

  typedef T value_type;

  BufferTyped& operator=(Buffer other)
  {
    *(Buffer*)this = other;
    return *this;
  }

  size_t   size() const { return bufferSize / sizeof(T); }
  const T* data() const { return reinterpret_cast<const T*>(mapping); }
  T*       data() { return reinterpret_cast<T*>(mapping); }

  // Retrieve an address of an element starting at the provided `startIndex`.
  // Provide proper `num` for error checking
  VkDeviceAddress addressAt(size_t startIndex, size_t num = 1) const
  {
    assert(startIndex + num <= size());
    return address + sizeof(T) * startIndex;
  }
};

// Allows to represent >= 4 GB buffers using sparse bindings
struct LargeBuffer
{
  VkBuffer                   buffer{};
  VkDeviceSize               bufferSize{};
  VkDeviceAddress            address{};
  std::vector<VmaAllocation> allocations;
};

//-----------------------------------------------------------------
// An image is a region of memory used to store image data.
// It is used to store texture data, framebuffer data, and other types of data.
//-----------------------------------------------------------------
struct Image
{
  // Vulkan Image
  // created/destroyed by `nvvk::ResourceAllocator`
  VkImage image{};
  // Size of the image
  VkExtent3D extent{};
  // Number of mip levels
  uint32_t mipLevels{};
  // number of array layers
  uint32_t arrayLayers{};
  // format of the image itself
  VkFormat format{VK_FORMAT_UNDEFINED};
  // Memory associated with the image
  // managed by `nvvk::ResourceAllocator`
  VmaAllocation allocation{};

  // descriptor.imageLayout represents the current imageLayout
  // descriptor.imageView may exist, created/destroyed by `nvvk::ResourceAllocator`
  // descriptor.sampler may exist, not managed by `nvvk::ResourceAllocator`
  VkDescriptorImageInfo descriptor{};
};

//-----------------------------------------------------------------
// An acceleration structure is a region of memory used to store acceleration structure data.
// It is used to store acceleration structure data for ray tracing.
//-----------------------------------------------------------------
struct AccelerationStructure
{
  VkAccelerationStructureKHR accel{};
  VkDeviceAddress            address{};
  Buffer                     buffer;  // Underlying buffer
};

// allows >= 4GB sizes
struct LargeAccelerationStructure
{
  VkAccelerationStructureKHR accel{};
  VkDeviceAddress            address{};
  LargeBuffer                buffer;  // Underlying buffer
};

// information about a range within a buffer
struct BufferRange : VkDescriptorBufferInfo
{
  VkDeviceAddress address{};  // must contain offset already
  uint8_t*        mapping{};  // must contain offset already
};

// A BufferRangeTyped allows to specify if a BufferRange represents only
// data of a certain type. It's interchangeable with regular BufferRange struct,
// and mostly meant to aid code readability if buffer ranges are strictly typed.
template <class T>
struct BufferRangeTyped : BufferRange
{
  static constexpr size_t value_size = sizeof(T);

  typedef T value_type;

  BufferRangeTyped& operator=(BufferRange other)
  {
    *(BufferRange*)this = other;
    return *this;
  }

  size_t   size() const { return range / sizeof(T); }
  const T* data() const { return static_cast<const T*>(mapping); }
  T*       data() { return static_cast<T*>(mapping); }

  // Retrieve an address of an element starting at the provided `startIndex`.
  // Provide proper `num` for error checking
  VkDeviceAddress addressAt(size_t startIndex, size_t num = 1) const
  {
    assert(startIndex + num <= size());
    return address + sizeof(T) * startIndex;
  }
};

//-----------------------------------------------------------------
// A queue is a sequence of commands that are executed in order.
// The queue is used to submit command buffers to the GPU.
// The family index is used to identify the queue family (graphic, compute, transfer, ...) .
// The queue index is used to identify the queue in the family, multiple queues can be in the same family.
//-----------------------------------------------------------------
struct QueueInfo
{
  uint32_t familyIndex = ~0U;  // Family index of the queue (graphic, compute, transfer, ...)
  uint32_t queueIndex  = ~0U;  // Index of the queue in the family
  VkQueue  queue{};            // The queue object
};


struct SemaphoreInfo
{
  VkSemaphore semaphore{};  // Timeline semaphore
  uint64_t    value{};      // Timeline value
};

}  // namespace nvvk
