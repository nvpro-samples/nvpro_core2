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
#include <array>
#include <cstring>
#include <atomic>
#include <mutex>
#include <span>
#include <vector>

#ifndef VMA_LEAK_LOG_FORMAT
// If there are reported leaks, use `nvvk::ResourceAllocator::setLeakID(uint32_t id)` accordingly
// to find resource creations that lack an appropriate destroy.
// VMA will report allocations as "nvvkAllocID: <uint32_t id>"
#include <nvutils/logger.hpp>
#define VMA_LEAK_LOG_FORMAT(format, ...)                                                                               \
  LOGW(format, __VA_ARGS__);                                                                                           \
  LOGW("\n")
#endif

#ifndef VMA_ASSERT_LEAK
#define VMA_ASSERT_LEAK(expr)                                                                                          \
  VMA_ASSERT((expr) && "Use nvvk::ResourceAllocator::setLeakID(nvvkAllocID) to find the leak")
#endif


#include <vk_mem_alloc.h>
#include <vulkan/vulkan_core.h>

#include "barriers.hpp"
#include "resources.hpp"


//-----------------------------------------------------------------
// Helper classes for allocating Vulkan resources (buffers, images) using VMA (Vulkan Memory Allocator).
//
// Contains two main classes:
// - ResourceAllocator: Basic allocation of device-local resources with automatic memory management
//   and simplified buffer/image creation. Handles alignment, memory types and resource destruction.
//
// - ResourceAllocatorExport: Extended version that adds support for external memory operations,
//   allowing resources to be shared across APIs (Vulkan<->CUDA, Vulkan<->D3D, etc). Includes
//   proper memory flags and handle types for cross-API interop.
//
// Usage:
// nvvk::ResourceAllocator m_allocator{};
//     m_allocator.init({
//        .flags            = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT,
//        .physicalDevice   = app->getPhysicalDevice(),
//        .device           = app->getDevice(),
//        .instance         = app->getInstance(),
//        .vulkanApiVersion = VK_API_VERSION_1_4,
//    });  // Allocator
//
// m_allocator.createBuffer(buffer, size, usage);
// m_allocator.createImage(image, imageInfo);
// m_allocator.destroyBuffer(buffer);
// m_allocator.destroyImage(image);
//
//  See also the staging classes for uploading data to the GPU.
//
//-----------------------------------------------------------------


namespace nvvk {

//--- Resource Allocator ------------------------------------------------------------------------------------------------------------
//
// Vulkan Memory Allocator (VMA) is a library that helps to manage memory in Vulkan.
// This should be used to manage the memory of the resources instead of using the Vulkan API directly.

class ResourceAllocator
{
public:
  static constexpr VkDeviceSize DEFAULT_LARGE_CHUNK_SIZE = VkDeviceSize(2) * 1024ull * 1024ull * 1024ull;

  ResourceAllocator()                                    = default;
  ResourceAllocator(const ResourceAllocator&)            = delete;
  ResourceAllocator& operator=(const ResourceAllocator&) = delete;
  ResourceAllocator(ResourceAllocator&& other) noexcept;
  ResourceAllocator& operator=(ResourceAllocator&& other) noexcept;
  ~ResourceAllocator();

  operator VmaAllocator() const;

  // Initialization of VMA allocator.
  VkResult init(VmaAllocatorCreateInfo allocatorInfo);

  // De-initialization of VMA allocator.
  void deinit();

  VkDevice         getDevice() const { return m_device; }
  VkPhysicalDevice getPhysicalDevice() const { return m_physicalDevice; }
  VkDeviceSize     getMaxMemoryAllocationSize() const { return m_maxMemoryAllocationSize; }

  //////////////////////////////////////////////////////////////////////////

  // Create a VkBuffer
  //
  //        + VMA_MEMORY_USAGE_AUTO
  //        + VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE
  //        + VMA_MEMORY_USAGE_AUTO_PREFER_HOST
  //      ----
  //        + VMA_ALLOCATION_CREATE_MAPPED_BIT // Automatically maps the buffer upon creation
  //        + VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT // If the CPU will sequentially write to the buffer's memory,
  //        + VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT
  //
  VkResult createBuffer(Buffer&                   buffer,
                        VkDeviceSize              size,
                        VkBufferUsageFlags2KHR    usage,
                        VmaMemoryUsage            memoryUsage   = VMA_MEMORY_USAGE_AUTO,
                        VmaAllocationCreateFlags  flags         = {},
                        VkDeviceSize              minAlignment  = 0,
                        std::span<const uint32_t> queueFamilies = {}) const;

  // This allows more fine control
  VkResult createBuffer(Buffer&                        buffer,
                        const VkBufferCreateInfo&      bufferInfo,
                        const VmaAllocationCreateInfo& allocInfo,
                        VkDeviceSize                   minAlignment = 0) const;

  // Destroy the VkBuffer
  void destroyBuffer(Buffer& buffer) const;


  // A large buffer allows sizes > maxMemoryAllocationSize (often around 4 GB)
  // by using sparse binding and multiple smaller allocations.
  // if no fence is provided, a vkQueueWaitIdle will be performed after the binding
  // operation

  VkResult createLargeBuffer(LargeBuffer&              buffer,
                             VkDeviceSize              size,
                             VkBufferUsageFlags2KHR    usage,
                             VkQueue                   sparseBindingQueue,
                             VkFence                   sparseBindingFence = VK_NULL_HANDLE,
                             VkDeviceSize              maxChunkSize       = DEFAULT_LARGE_CHUNK_SIZE,
                             VmaMemoryUsage            memoryUsage        = VMA_MEMORY_USAGE_AUTO,
                             VmaAllocationCreateFlags  flags              = {},
                             VkDeviceSize              minAlignment       = 0,
                             std::span<const uint32_t> queueFamilies      = {}) const;

  VkResult createLargeBuffer(LargeBuffer&                   buffer,
                             const VkBufferCreateInfo&      bufferInfo,
                             const VmaAllocationCreateInfo& allocInfo,
                             VkQueue                        sparseBindingQueue,
                             VkFence                        sparseBindingFence = VK_NULL_HANDLE,
                             VkDeviceSize                   maxChunkSize       = DEFAULT_LARGE_CHUNK_SIZE,
                             VkDeviceSize                   minAlignment       = 0) const;

  void destroyLargeBuffer(LargeBuffer& buffer) const;


  // Creates VkImage in device memory
  VkResult createImage(Image& image, const VkImageCreateInfo& imageInfo) const;

  // Creates VkImage and VkImageView in device memory
  VkResult createImage(Image& image, const VkImageCreateInfo& imageInfo, const VkImageViewCreateInfo& imageViewInfo) const;

  // Creates VkImage with provided allocation information
  VkResult createImage(Image& image, const VkImageCreateInfo& imageInfo, const VmaAllocationCreateInfo& vmaInfo) const;

  // Creates VkImage and VkImageView with provided allocation information
  VkResult createImage(Image&                         image,
                       const VkImageCreateInfo&       imageInfo,
                       const VkImageViewCreateInfo&   imageViewInfo,
                       const VmaAllocationCreateInfo& vmaInfo) const;

  // Destroys VkImage and VkImageView
  void destroyImage(Image& image) const;


  // AcclerationStructure

  VkResult createAcceleration(AccelerationStructure& accel, const VkAccelerationStructureCreateInfoKHR& accInfo) const;

  VkResult createAcceleration(AccelerationStructure&                      accel,
                              const VkAccelerationStructureCreateInfoKHR& accInfo,
                              const VmaAllocationCreateInfo&              vmaInfo,
                              std::span<const uint32_t>                   queueFamilies = {}) const;

  void destroyAcceleration(AccelerationStructure& accel) const;

  // LargeAccelerationStructure

  // if no fence is provided, a vkQueueWaitIdle will be performed after the binding
  // operation

  VkResult createLargeAcceleration(LargeAccelerationStructure&                 accel,
                                   const VkAccelerationStructureCreateInfoKHR& accInfo,
                                   VkQueue                                     sparseBindingQueue,
                                   VkFence                                     sparseBindingFence = VK_NULL_HANDLE,
                                   VkDeviceSize maxChunkSize = DEFAULT_LARGE_CHUNK_SIZE) const;

  VkResult createLargeAcceleration(LargeAccelerationStructure&                 accel,
                                   const VkAccelerationStructureCreateInfoKHR& accInfo,
                                   const VmaAllocationCreateInfo&              vmaInfo,
                                   VkQueue                                     sparseBindingQueue,
                                   VkFence                                     sparseBindingFence = VK_NULL_HANDLE,
                                   VkDeviceSize                                maxChunkSize  = DEFAULT_LARGE_CHUNK_SIZE,
                                   std::span<const uint32_t>                   queueFamilies = {}) const;

  void destroyLargeAcceleration(LargeAccelerationStructure& accel) const;

  //////////////////////////////////////////////////////////////////////////

  // When leak are reported, set the ID of the leak here
  void setLeakID(uint32_t id);

  // Returns the device memory of the VMA allocation
  VkDeviceMemory getDeviceMemory(VmaAllocation allocation) const;

  //////////////////////////////////////////////////////////////////////////

  // Calls `vkFlushMappedMemoryRanges` via VMA for the provided buffer's memory.
  // Is required for non-coherent mapped memory after it was written by cpu.
  VkResult flushBuffer(const nvvk::Buffer& buffer, VkDeviceSize offset = 0, VkDeviceSize size = VK_WHOLE_SIZE);

  // Calls `vkInvalidateMappedMemoryRanges` via VMA for the provided buffer's memory
  // Is required for non-coherent mapped memory prior reading it from cpu.
  VkResult invalidateBuffer(const nvvk::Buffer& buffer, VkDeviceSize offset = 0, VkDeviceSize size = VK_WHOLE_SIZE);

  // Calls `vkFlushMappedMemoryRanges` via VMA, if the buffer's memory is non-coherent, otherwise does nothing and always returns VK_SUCCESS.
  // VMA's heuristic for picking memory types, may make it non-trivial to know in advance if a buffer is coherent or not
  VkResult autoFlushBuffer(const nvvk::Buffer& buffer, VkDeviceSize offset = 0, VkDeviceSize size = VK_WHOLE_SIZE);

  // Calls `vkInvalidateMappedMemoryRanges` via VMA, if the buffer's memory is non-coherent, otherwise does nothing and always returns VK_SUCCESS.
  // VMA's heuristic for picking memory types, may make it non-trivial to know in advance if a buffer is coherent or not
  VkResult autoInvalidateBuffer(const nvvk::Buffer& buffer, VkDeviceSize offset = 0, VkDeviceSize size = VK_WHOLE_SIZE);

private:
  // Adds "nvvkAllocID: <id>" name to the vma allocation, useful for vma's leak detection
  // (see comments around m_leakID)
  void addLeakDetection(VmaAllocation allocation) const;

private:
  VmaAllocator     m_allocator{};
  VkDevice         m_device{};
  VkPhysicalDevice m_physicalDevice{};
  VkDeviceSize     m_maxMemoryAllocationSize = 0;

  // Each vma allocation is named using a global monotonic counter
  mutable std::atomic_uint32_t m_allocationCounter = 0;
  // Throws breakpoint/signal when a resource using "nvvkAllocID: <id>" name was
  // created. Only works if `m_allocationCounter` is used deterministically.
  uint32_t m_leakID = ~0U;
};


// This allocator extends ResourceAllocator to enable interoperability with external APIs like CUDA and OpenGL.
// It adds memory export flags during allocation, allowing resources to be shared across APIs while maintaining
// Vulkan-native performance. This is particularly useful for applications that need to process data with CUDA
// or display results through OpenGL.
class ResourceAllocatorExport : public ResourceAllocator
{
public:
  // This initialization is the same as the ResourceAllocator, but adds the VMA_ALLOCATOR_CREATE_KHR_EXTERNAL_MEMORY_WIN32_BIT flag
  VkResult init(VmaAllocatorCreateInfo allocatorInfo)
  {
#ifdef _WIN32
    allocatorInfo.flags |= VMA_ALLOCATOR_CREATE_KHR_EXTERNAL_MEMORY_WIN32_BIT;
#endif  // _WIN32
    return nvvk::ResourceAllocator::init(allocatorInfo);
  }

  // The de-initialization is the same as the ResourceAllocator, but destroy all the pools that have been created
  void deinit()
  {
    for(auto& p : m_pools)
    {
      if(p != VK_NULL_HANDLE)
      {
        vmaDestroyPool(*this, p);
        p = VK_NULL_HANDLE;
      }
    }

    nvvk::ResourceAllocator::deinit();
  }

  // Same as createBuffer, but with export flag capability
  VkResult createBufferExport(Buffer&                   buffer,
                              VkDeviceSize              size,
                              VkBufferUsageFlags2KHR    usage,
                              VmaMemoryUsage            memoryUsage   = VMA_MEMORY_USAGE_AUTO,
                              VmaAllocationCreateFlags  flags         = {},
                              VkDeviceSize              minAlignment  = 0,
                              std::span<const uint32_t> queueFamilies = {});

  // Same as createImage, but with export flag capability
  VkResult createImageExport(Image& image, const VkImageCreateInfo& imageInfo, const VkImageViewCreateInfo& imageViewInfo);

  // This needs to be called to get the right pool for the allocation (could create a new vmaPool)
  VkResult getAllocInfo(VmaAllocationCreateInfo&  allocCreateInfo,
                        VmaAllocationCreateFlags  flags,
                        VmaMemoryUsage            usage,
                        const VkBufferCreateInfo& bufferInfo);

  // Adding export flag capability to allocation create info (could create a new vmaPool)
  VkResult getAllocInfo(VmaAllocationCreateInfo& allocCreateInfo,
                        VmaAllocationCreateFlags flags,
                        VmaMemoryUsage           usage,
                        const VkImageCreateInfo& imageInfo);

private:
  VkResult getPool(uint32_t memoryTypeIndex, VmaPool& pool);

  // Array of memory pools with the export
  std::array<VmaPool, VK_MAX_MEMORY_TYPES> m_pools{};
  std::mutex                               m_mutex;
};


}  // namespace nvvk
