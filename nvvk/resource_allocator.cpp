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

#include <fmt/format.h>
#include <nvutils/logger.hpp>

#include "resource_allocator.hpp"
#include "check_error.hpp"

#include <volk.h>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <Windows.h>
#include <debugapi.h>
#elif defined(__unix__)
#include <signal.h>
#endif

nvvk::ResourceAllocator::ResourceAllocator(ResourceAllocator&& other) noexcept
{
  std::swap(m_allocator, other.m_allocator);
  std::swap(m_device, other.m_device);
  std::swap(m_physicalDevice, other.m_physicalDevice);
  std::swap(m_leakID, other.m_leakID);
  std::swap(m_maxMemoryAllocationSize, other.m_maxMemoryAllocationSize);
}

nvvk::ResourceAllocator& nvvk::ResourceAllocator::operator=(ResourceAllocator&& other) noexcept
{
  if(this != &other)
  {
    assert(m_allocator == nullptr && "Missing deinit()");

    std::swap(m_allocator, other.m_allocator);
    std::swap(m_device, other.m_device);
    std::swap(m_physicalDevice, other.m_physicalDevice);
    std::swap(m_leakID, other.m_leakID);
    std::swap(m_maxMemoryAllocationSize, other.m_maxMemoryAllocationSize);
  }

  return *this;
}

nvvk::ResourceAllocator::~ResourceAllocator()
{
  assert(m_allocator == nullptr && "Missing deinit()");
}

nvvk::ResourceAllocator::operator VmaAllocator() const
{
  return m_allocator;
}

VkResult nvvk::ResourceAllocator::init(VmaAllocatorCreateInfo allocatorInfo)
{
  assert(m_allocator == nullptr);

  // #TODO : VK_EXT_memory_priority ? VMA_ALLOCATOR_CREATE_EXT_MEMORY_PRIORITY_BIT

  allocatorInfo.flags |= VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;  // allow querying for the GPU address of a buffer
  allocatorInfo.flags |= VMA_ALLOCATOR_CREATE_KHR_MAINTENANCE4_BIT;
  allocatorInfo.flags |= VMA_ALLOCATOR_CREATE_KHR_MAINTENANCE5_BIT;  // allow using VkBufferUsageFlags2CreateInfoKHR

  VkPhysicalDeviceVulkan11Properties props11{
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_PROPERTIES,
  };

  VkPhysicalDeviceProperties2 props = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
      .pNext = &props11,
  };
  vkGetPhysicalDeviceProperties2(allocatorInfo.physicalDevice, &props);
  m_maxMemoryAllocationSize = props11.maxMemoryAllocationSize;

  m_device         = allocatorInfo.device;
  m_physicalDevice = allocatorInfo.physicalDevice;

  // Because we use VMA_DYNAMIC_VULKAN_FUNCTIONS
  const VmaVulkanFunctions functions = {
      .vkGetInstanceProcAddr = vkGetInstanceProcAddr,
      .vkGetDeviceProcAddr   = vkGetDeviceProcAddr,
  };
  allocatorInfo.pVulkanFunctions = &functions;
  return vmaCreateAllocator(&allocatorInfo, &m_allocator);
}

void nvvk::ResourceAllocator::deinit()
{
  if(!m_allocator)
    return;

  vmaDestroyAllocator(m_allocator);
  m_allocator      = nullptr;
  m_device         = nullptr;
  m_physicalDevice = nullptr;
  m_leakID         = ~0;
}

void nvvk::ResourceAllocator::addLeakDetection(VmaAllocation allocation) const
{
  if(m_leakID == m_allocationCounter)
  {
#ifdef _WIN32
    if(IsDebuggerPresent())
    {
      DebugBreak();
    }
#elif defined(__unix__)
    raise(SIGTRAP);
#endif
  }
  std::string nvvkAllocID = fmt::format("nvvkAllocID: {}", m_allocationCounter++);
  vmaSetAllocationName(m_allocator, allocation, nvvkAllocID.c_str());
}

VkResult nvvk::ResourceAllocator::createBuffer(nvvk::Buffer&             buffer,
                                               VkDeviceSize              size,
                                               VkBufferUsageFlags2KHR    usage,
                                               VmaMemoryUsage            memoryUsage,
                                               VmaAllocationCreateFlags  flags,
                                               VkDeviceSize              minAlignment,
                                               std::span<const uint32_t> queueFamilies) const
{
  const VkBufferUsageFlags2CreateInfo bufferUsageFlags2CreateInfo{
      .sType = VK_STRUCTURE_TYPE_BUFFER_USAGE_FLAGS_2_CREATE_INFO,
      .usage = usage | VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT,
  };

  const VkBufferCreateInfo bufferInfo{
      .sType                 = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .pNext                 = &bufferUsageFlags2CreateInfo,
      .size                  = size,
      .usage                 = 0,
      .sharingMode           = queueFamilies.empty() ? VK_SHARING_MODE_EXCLUSIVE : VK_SHARING_MODE_CONCURRENT,
      .queueFamilyIndexCount = static_cast<uint32_t>(queueFamilies.size()),
      .pQueueFamilyIndices   = queueFamilies.data(),
  };

  VmaAllocationCreateInfo allocInfo = {.flags = flags, .usage = memoryUsage};

  return createBuffer(buffer, bufferInfo, allocInfo, minAlignment);
}

VkResult nvvk::ResourceAllocator::createBuffer(nvvk::Buffer&                  resultBuffer,
                                               const VkBufferCreateInfo&      bufferInfo,
                                               const VmaAllocationCreateInfo& allocInfo,
                                               VkDeviceSize                   minAlignment) const
{
  resultBuffer = {};

  // Create the buffer
  VmaAllocationInfo allocInfoOut{};

  VkResult result = vmaCreateBufferWithAlignment(m_allocator, &bufferInfo, &allocInfo, minAlignment,
                                                 &resultBuffer.buffer, &resultBuffer.allocation, &allocInfoOut);

  if(result != VK_SUCCESS)
  {
    // Handle allocation failure
    LOGW("Failed to create buffer");
    return result;
  }

  resultBuffer.bufferSize = bufferInfo.size;
  resultBuffer.mapping    = static_cast<uint8_t*>(allocInfoOut.pMappedData);

  // Get the GPU address of the buffer
  const VkBufferDeviceAddressInfo info = {.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO, .buffer = resultBuffer.buffer};
  resultBuffer.address = vkGetBufferDeviceAddress(m_device, &info);

  addLeakDetection(resultBuffer.allocation);

  return result;
}

void nvvk::ResourceAllocator::destroyBuffer(nvvk::Buffer& buffer) const
{
  vmaDestroyBuffer(m_allocator, buffer.buffer, buffer.allocation);
  buffer = {};
}

VkResult nvvk::ResourceAllocator::createLargeBuffer(LargeBuffer&                   largeBuffer,
                                                    const VkBufferCreateInfo&      bufferInfo,
                                                    const VmaAllocationCreateInfo& allocInfo,
                                                    VkQueue                        sparseBindingQueue,
                                                    VkFence                        sparseBindingFence,
                                                    VkDeviceSize                   maxChunkSize,
                                                    VkDeviceSize                   minAlignment) const
{
  assert(sparseBindingQueue);

  largeBuffer = {};

  maxChunkSize = std::min(m_maxMemoryAllocationSize, maxChunkSize);

  if(bufferInfo.size <= maxChunkSize)
  {
    Buffer buffer;
    NVVK_FAIL_RETURN(createBuffer(buffer, bufferInfo, allocInfo, minAlignment));

    largeBuffer.buffer      = buffer.buffer;
    largeBuffer.bufferSize  = buffer.bufferSize;
    largeBuffer.address     = buffer.address;
    largeBuffer.allocations = {buffer.allocation};

    return VK_SUCCESS;
  }
  else
  {
    VkBufferCreateInfo createInfo = bufferInfo;

    createInfo.flags |= VK_BUFFER_CREATE_SPARSE_BINDING_BIT;

    NVVK_FAIL_RETURN(vkCreateBuffer(m_device, &createInfo, nullptr, &largeBuffer.buffer));

    // Find memory requirements
    VkMemoryRequirements2           memReqs{VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2};
    VkMemoryDedicatedRequirements   dedicatedRegs{VK_STRUCTURE_TYPE_MEMORY_DEDICATED_REQUIREMENTS};
    VkBufferMemoryRequirementsInfo2 bufferReqs{VK_STRUCTURE_TYPE_BUFFER_MEMORY_REQUIREMENTS_INFO_2};

    memReqs.pNext     = &dedicatedRegs;
    bufferReqs.buffer = largeBuffer.buffer;

    vkGetBufferMemoryRequirements2(m_device, &bufferReqs, &memReqs);
    memReqs.memoryRequirements.alignment = std::max(minAlignment, memReqs.memoryRequirements.alignment);

    // align maxChunkSize to required alignment
    size_t pageAlignment = memReqs.memoryRequirements.alignment;
    maxChunkSize         = (maxChunkSize + pageAlignment - 1) & ~(pageAlignment - 1);

    // get chunk count
    size_t fullChunkCount  = bufferInfo.size / maxChunkSize;
    size_t totalChunkCount = (bufferInfo.size + maxChunkSize - 1) / maxChunkSize;

    largeBuffer.allocations.resize(totalChunkCount);
    std::vector<VmaAllocationInfo> allocationInfos(totalChunkCount);

    // full chunks first
    memReqs.memoryRequirements.size = maxChunkSize;
    VkResult result = vmaAllocateMemoryPages(m_allocator, &memReqs.memoryRequirements, &allocInfo, fullChunkCount,
                                             largeBuffer.allocations.data(), allocationInfos.data());
    if(result != VK_SUCCESS)
    {
      vkDestroyBuffer(m_device, largeBuffer.buffer, nullptr);
      largeBuffer = {};
      return result;
    }

    // tail chunk last
    if(fullChunkCount != totalChunkCount)
    {
      memReqs.memoryRequirements.size = createInfo.size - fullChunkCount * maxChunkSize;
      memReqs.memoryRequirements.size = (memReqs.memoryRequirements.size + pageAlignment - 1) & ~(pageAlignment - 1);

      result = vmaAllocateMemoryPages(m_allocator, &memReqs.memoryRequirements, &allocInfo, 1,
                                      largeBuffer.allocations.data() + fullChunkCount, allocationInfos.data() + fullChunkCount);
      if(result != VK_SUCCESS)
      {
        vmaFreeMemoryPages(m_allocator, fullChunkCount, largeBuffer.allocations.data());
        vkDestroyBuffer(m_device, largeBuffer.buffer, nullptr);
        largeBuffer = {};
        return result;
      }
    }

    std::vector<VkSparseMemoryBind> sparseBinds(totalChunkCount);

    for(uint32_t i = 0; i < totalChunkCount; i++)
    {
      VkSparseMemoryBind& sparseBind = sparseBinds[i];
      sparseBind.flags               = 0;
      sparseBind.memory              = allocationInfos[i].deviceMemory;
      sparseBind.memoryOffset        = allocationInfos[i].offset;
      sparseBind.resourceOffset      = i * maxChunkSize;
      sparseBind.size                = std::min(maxChunkSize, createInfo.size - i * maxChunkSize);
      sparseBind.size                = (sparseBind.size + pageAlignment - 1) & ~(pageAlignment - 1);

      addLeakDetection(largeBuffer.allocations[i]);
    }

    VkSparseBufferMemoryBindInfo sparseBufferMemoryBindInfo{};
    sparseBufferMemoryBindInfo.buffer    = largeBuffer.buffer;
    sparseBufferMemoryBindInfo.bindCount = uint32_t(sparseBinds.size());
    sparseBufferMemoryBindInfo.pBinds    = sparseBinds.data();

    VkBindSparseInfo bindSparseInfo{VK_STRUCTURE_TYPE_BIND_SPARSE_INFO};
    bindSparseInfo.bufferBindCount = 1;
    bindSparseInfo.pBufferBinds    = &sparseBufferMemoryBindInfo;

    result = NVVK_FAIL_REPORT(vkQueueBindSparse(sparseBindingQueue, 1, &bindSparseInfo, sparseBindingFence));
    if(result != VK_SUCCESS)
    {
      vkDestroyBuffer(m_device, largeBuffer.buffer, nullptr);
      vmaFreeMemoryPages(m_allocator, largeBuffer.allocations.size(), largeBuffer.allocations.data());
      largeBuffer = {};
      return result;
    }

    if(!sparseBindingFence)
    {
      result = NVVK_FAIL_REPORT(vkQueueWaitIdle(sparseBindingQueue));
      if(result != VK_SUCCESS)
      {
        if(result != VK_ERROR_DEVICE_LOST)
        {
          vkDestroyBuffer(m_device, largeBuffer.buffer, nullptr);
          vmaFreeMemoryPages(m_allocator, largeBuffer.allocations.size(), largeBuffer.allocations.data());
          largeBuffer = {};
        }
        return result;
      }
    }

    // Get the GPU address of the buffer
    const VkBufferDeviceAddressInfo info = {
        .sType  = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
        .buffer = largeBuffer.buffer,
    };
    largeBuffer.address    = vkGetBufferDeviceAddress(m_device, &info);
    largeBuffer.bufferSize = createInfo.size;

    return VK_SUCCESS;
  }
}

VkResult nvvk::ResourceAllocator::createLargeBuffer(LargeBuffer&           largeBuffer,
                                                    VkDeviceSize           size,
                                                    VkBufferUsageFlags2KHR usage,
                                                    VkQueue                sparseBindingQueue,
                                                    VkFence                sparseBindingFence /*= VK_NULL_HANDLE*/,
                                                    VkDeviceSize           maxChunkSize /*= DEFAULT_LARGE_CHUNK_SIZE*/,
                                                    VmaMemoryUsage         memoryUsage /*= VMA_MEMORY_USAGE_AUTO*/,
                                                    VmaAllocationCreateFlags  flags /*= {}*/,
                                                    VkDeviceSize              minAlignment /*= 0*/,
                                                    std::span<const uint32_t> queueFamilies /*= {}*/) const
{
  const VkBufferUsageFlags2CreateInfo bufferUsageFlags2CreateInfo{
      .sType = VK_STRUCTURE_TYPE_BUFFER_USAGE_FLAGS_2_CREATE_INFO,
      .usage = usage | VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT,
  };

  const VkBufferCreateInfo bufferInfo{
      .sType                 = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .pNext                 = &bufferUsageFlags2CreateInfo,
      .size                  = size,
      .usage                 = 0,
      .sharingMode           = queueFamilies.empty() ? VK_SHARING_MODE_EXCLUSIVE : VK_SHARING_MODE_CONCURRENT,
      .queueFamilyIndexCount = static_cast<uint32_t>(queueFamilies.size()),
      .pQueueFamilyIndices   = queueFamilies.data(),
  };

  VmaAllocationCreateInfo allocInfo       = {.flags = flags, .usage = memoryUsage};
  uint32_t                memoryTypeIndex = 0;
  NVVK_FAIL_RETURN(vmaFindMemoryTypeIndexForBufferInfo(m_allocator, &bufferInfo, &allocInfo, &memoryTypeIndex));

  allocInfo.usage          = VMA_MEMORY_USAGE_UNKNOWN;
  allocInfo.memoryTypeBits = 1 << memoryTypeIndex;

  return createLargeBuffer(largeBuffer, bufferInfo, allocInfo, sparseBindingQueue, sparseBindingFence, maxChunkSize, minAlignment);
}

void nvvk::ResourceAllocator::destroyLargeBuffer(LargeBuffer& buffer) const
{
  vkDestroyBuffer(m_device, buffer.buffer, nullptr);
  vmaFreeMemoryPages(m_allocator, buffer.allocations.size(), buffer.allocations.data());
  buffer = {};
}

VkResult nvvk::ResourceAllocator::createImage(nvvk::Image& image, const VkImageCreateInfo& imageInfo, const VmaAllocationCreateInfo& allocInfo) const
{
  image = {};

  VmaAllocationInfo allocInfoOut{};
  VkResult result = vmaCreateImage(m_allocator, &imageInfo, &allocInfo, &image.image, &image.allocation, &allocInfoOut);

  if(result != VK_SUCCESS)
  {
    // Handle allocation failure
    LOGW("Failed to create image\n");
  }

  image.extent                 = imageInfo.extent;
  image.mipLevels              = imageInfo.mipLevels;
  image.arrayLayers            = imageInfo.arrayLayers;
  image.format                 = imageInfo.format;
  image.descriptor.imageLayout = VK_IMAGE_LAYOUT_UNDEFINED;

  addLeakDetection(image.allocation);

  return result;
}

VkResult nvvk::ResourceAllocator::createImage(nvvk::Image& image, const VkImageCreateInfo& imageInfo) const
{
  const VmaAllocationCreateInfo allocInfo{.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};

  return createImage(image, imageInfo, allocInfo);
}

VkResult nvvk::ResourceAllocator::createImage(Image& image, const VkImageCreateInfo& imageInfo, const VkImageViewCreateInfo& imageViewInfo) const
{
  const VmaAllocationCreateInfo allocInfo{.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};

  return createImage(image, imageInfo, imageViewInfo, allocInfo);
}

VkResult nvvk::ResourceAllocator::createImage(Image&                         image,
                                              const VkImageCreateInfo&       _imageInfo,
                                              const VkImageViewCreateInfo&   _imageViewInfo,
                                              const VmaAllocationCreateInfo& vmaInfo) const
{
  VkResult result{};
  // Create image in GPU memory
  VkImageCreateInfo imageInfo = _imageInfo;
  imageInfo.usage |= VK_IMAGE_USAGE_TRANSFER_DST_BIT;  // We will copy data to this image
  result = createImage(image, imageInfo, vmaInfo);
  if(result != VK_SUCCESS)
  {
    return result;
  }

  // Create image view
  VkImageViewCreateInfo viewInfo = _imageViewInfo;
  viewInfo.image                 = image.image;
  viewInfo.format                = _imageInfo.format;
  NVVK_FAIL_RETURN(vkCreateImageView(m_device, &viewInfo, nullptr, &image.descriptor.imageView));

  return VK_SUCCESS;
}

void nvvk::ResourceAllocator::destroyImage(Image& image) const
{
  vkDestroyImageView(m_device, image.descriptor.imageView, nullptr);
  vmaDestroyImage(m_allocator, image.image, image.allocation);
  image = {};
}

VkResult nvvk::ResourceAllocator::createAcceleration(nvvk::AccelerationStructure&                resultAccel,
                                                     const VkAccelerationStructureCreateInfoKHR& accInfo,
                                                     const VmaAllocationCreateInfo&              vmaInfo,
                                                     std::span<const uint32_t>                   queueFamilies) const
{
  resultAccel                                      = {};
  VkAccelerationStructureCreateInfoKHR accelStruct = accInfo;

  const VkBufferUsageFlags2CreateInfo bufferUsageFlags2CreateInfo{
      .sType = VK_STRUCTURE_TYPE_BUFFER_USAGE_FLAGS_2_CREATE_INFO,
      .usage = VK_BUFFER_USAGE_2_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT,
  };

  const VkBufferCreateInfo bufferInfo{
      .sType                 = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .pNext                 = &bufferUsageFlags2CreateInfo,
      .size                  = accelStruct.size,
      .usage                 = 0,
      .sharingMode           = queueFamilies.empty() ? VK_SHARING_MODE_EXCLUSIVE : VK_SHARING_MODE_CONCURRENT,
      .queueFamilyIndexCount = static_cast<uint32_t>(queueFamilies.size()),
      .pQueueFamilyIndices   = queueFamilies.data(),
  };

  // Step 1: Create the buffer to hold the acceleration structure
  VkResult result = createBuffer(resultAccel.buffer, bufferInfo, vmaInfo);

  if(result != VK_SUCCESS)
  {
    return result;
  }

  // Step 2: Create the acceleration structure with the buffer
  accelStruct.buffer = resultAccel.buffer.buffer;
  result             = vkCreateAccelerationStructureKHR(m_device, &accelStruct, nullptr, &resultAccel.accel);

  if(result != VK_SUCCESS)
  {
    destroyBuffer(resultAccel.buffer);
    LOGW("Failed to create acceleration structure");
    return result;
  }

  {
    VkAccelerationStructureDeviceAddressInfoKHR info{.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR,
                                                     .accelerationStructure = resultAccel.accel};
    resultAccel.address = vkGetAccelerationStructureDeviceAddressKHR(m_device, &info);
  }

  return result;
}

VkResult nvvk::ResourceAllocator::createAcceleration(nvvk::AccelerationStructure&                accel,
                                                     const VkAccelerationStructureCreateInfoKHR& inAccInfo) const
{
  const VmaAllocationCreateInfo allocInfo{.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};

  return createAcceleration(accel, inAccInfo, allocInfo);
}

void nvvk::ResourceAllocator::destroyAcceleration(nvvk::AccelerationStructure& accel) const
{
  destroyBuffer(accel.buffer);
  vkDestroyAccelerationStructureKHR(m_device, accel.accel, nullptr);
  accel = {};
}

VkResult nvvk::ResourceAllocator::createLargeAcceleration(LargeAccelerationStructure&                 resultAccel,
                                                          const VkAccelerationStructureCreateInfoKHR& accInfo,
                                                          const VmaAllocationCreateInfo&              vmaInfo,
                                                          VkQueue                   sparseBindingQueue,
                                                          VkFence                   sparseBindingFence,
                                                          VkDeviceSize              maxChunkSize,
                                                          std::span<const uint32_t> queueFamilies) const
{
  resultAccel                                      = {};
  VkAccelerationStructureCreateInfoKHR accelStruct = accInfo;

  const VkBufferUsageFlags2CreateInfo bufferUsageFlags2CreateInfo{
      .sType = VK_STRUCTURE_TYPE_BUFFER_USAGE_FLAGS_2_CREATE_INFO,
      .usage = VK_BUFFER_USAGE_2_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT_KHR,
  };

  const VkBufferCreateInfo bufferInfo{
      .sType                 = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .pNext                 = &bufferUsageFlags2CreateInfo,
      .flags                 = VK_BUFFER_CREATE_SPARSE_BINDING_BIT,
      .size                  = accelStruct.size,
      .usage                 = 0,
      .sharingMode           = queueFamilies.empty() ? VK_SHARING_MODE_EXCLUSIVE : VK_SHARING_MODE_CONCURRENT,
      .queueFamilyIndexCount = static_cast<uint32_t>(queueFamilies.size()),
      .pQueueFamilyIndices   = queueFamilies.data(),
  };

  // Step 1: Create the buffer to hold the acceleration structure
  VkResult result = createLargeBuffer(resultAccel.buffer, bufferInfo, vmaInfo, sparseBindingQueue, sparseBindingFence, maxChunkSize);

  if(result != VK_SUCCESS)
  {
    return result;
  }

  // Step 2: Create the acceleration structure with the buffer
  accelStruct.buffer = resultAccel.buffer.buffer;
  result             = vkCreateAccelerationStructureKHR(m_device, &accelStruct, nullptr, &resultAccel.accel);

  if(result != VK_SUCCESS)
  {
    destroyLargeBuffer(resultAccel.buffer);
    LOGW("Failed to create acceleration structure");
    return result;
  }

  {
    VkAccelerationStructureDeviceAddressInfoKHR info{.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR,
                                                     .accelerationStructure = resultAccel.accel};
    resultAccel.address = vkGetAccelerationStructureDeviceAddressKHR(m_device, &info);
  }

  return result;
}

VkResult nvvk::ResourceAllocator::createLargeAcceleration(LargeAccelerationStructure&                 accel,
                                                          const VkAccelerationStructureCreateInfoKHR& accInfo,
                                                          VkQueue      sparseBindingQueue,
                                                          VkFence      sparseBindingFence,
                                                          VkDeviceSize maxChunkSize) const
{
  VmaAllocationCreateInfo allocInfo = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};

  return createLargeAcceleration(accel, accInfo, allocInfo, sparseBindingQueue, sparseBindingFence, maxChunkSize);
}

void nvvk::ResourceAllocator::destroyLargeAcceleration(LargeAccelerationStructure& accel) const
{
  vkDestroyAccelerationStructureKHR(m_device, accel.accel, nullptr);
  destroyLargeBuffer(accel.buffer);

  accel = {};
}

void nvvk::ResourceAllocator::setLeakID(uint32_t id)
{
  m_leakID = id;
}

VkDeviceMemory nvvk::ResourceAllocator::getDeviceMemory(VmaAllocation allocation) const
{
  VmaAllocationInfo allocationInfo;
  vmaGetAllocationInfo(*this, allocation, &allocationInfo);
  return allocationInfo.deviceMemory;
}

VkResult nvvk::ResourceAllocator::flushBuffer(const nvvk::Buffer& buffer, VkDeviceSize offset /*= 0*/, VkDeviceSize size /*= VK_WHOLE_SIZE*/)
{
  assert(buffer.mapping);
  return vmaFlushAllocation(m_allocator, buffer.allocation, offset, size);
}

VkResult nvvk::ResourceAllocator::invalidateBuffer(const nvvk::Buffer& buffer, VkDeviceSize offset /*= 0*/, VkDeviceSize size /*= VK_WHOLE_SIZE*/)
{
  assert(buffer.mapping);
  return vmaInvalidateAllocation(m_allocator, buffer.allocation, offset, size);
}

VkResult nvvk::ResourceAllocator::autoFlushBuffer(const nvvk::Buffer& buffer, VkDeviceSize offset /*= 0*/, VkDeviceSize size /*= VK_WHOLE_SIZE*/)
{
  assert(buffer.mapping);
  VkMemoryPropertyFlags memFlags{};
  vmaGetAllocationMemoryProperties(m_allocator, buffer.allocation, &memFlags);
  if(!(memFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT))
  {
    return vmaFlushAllocation(m_allocator, buffer.allocation, offset, size);
  }
  else
  {
    return VK_SUCCESS;
  }
}

VkResult nvvk::ResourceAllocator::autoInvalidateBuffer(const nvvk::Buffer& buffer, VkDeviceSize offset /*= 0*/, VkDeviceSize size /*= VK_WHOLE_SIZE*/)
{
  assert(buffer.mapping);
  VkMemoryPropertyFlags memFlags{};
  vmaGetAllocationMemoryProperties(m_allocator, buffer.allocation, &memFlags);
  if(!(memFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT))
  {
    return vmaInvalidateAllocation(m_allocator, buffer.allocation, offset, size);
  }
  else
  {
    return VK_SUCCESS;
  }
}


//-------------------------------------------------------------------------------------------------------------------------
// Export allocator

// This creates the buffer with the export flag
VkResult nvvk::ResourceAllocatorExport::createBufferExport(Buffer&                   buffer,
                                                           VkDeviceSize              size,
                                                           VkBufferUsageFlags2KHR    usage,
                                                           VmaMemoryUsage            memoryUsage,
                                                           VmaAllocationCreateFlags  flags,
                                                           VkDeviceSize              minAlignment,
                                                           std::span<const uint32_t> queueFamilies)
{
  const VkBufferUsageFlags2CreateInfo bufferUsageFlags2CreateInfo{
      .sType = VK_STRUCTURE_TYPE_BUFFER_USAGE_FLAGS_2_CREATE_INFO,
      .usage = usage | VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT,
  };

  // Structure for buffer creation with export flag capability
  const VkExternalMemoryBufferCreateInfo externalMemBufCreateInfo{
      .sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO,
      .pNext = &bufferUsageFlags2CreateInfo,
#ifdef _WIN32
      .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT,
#else
      .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT,
#endif  // _WIN32
  };

  // Adding export flag capability to buffer create info
  const VkBufferCreateInfo bufferInfo{
      .sType                 = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .pNext                 = &externalMemBufCreateInfo,
      .size                  = size,
      .usage                 = 0,
      .sharingMode           = queueFamilies.empty() ? VK_SHARING_MODE_EXCLUSIVE : VK_SHARING_MODE_CONCURRENT,
      .queueFamilyIndexCount = static_cast<uint32_t>(queueFamilies.size()),
      .pQueueFamilyIndices   = queueFamilies.data(),
  };

  // Getting the allocation create info with the export flag capability
  VmaAllocationCreateInfo allocCreateInfo = {};

  NVVK_FAIL_RETURN(getAllocInfo(allocCreateInfo, flags, memoryUsage, bufferInfo));

  return createBuffer(buffer, bufferInfo, allocCreateInfo, minAlignment);
}

VkResult nvvk::ResourceAllocatorExport::createImageExport(Image& image, const VkImageCreateInfo& imageInfo, const VkImageViewCreateInfo& imageViewInfo)
{
  // Structure for image creation with export flag capability
  const VkExternalMemoryImageCreateInfo externalMemCreateInfo{
      .sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO,
#ifdef _WIN32
      .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT,
#else
      .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT,
#endif  // _WIN32
  };

  // Adding export flag capability to image create info
  VkImageCreateInfo imageInfoCpy = imageInfo;
  imageInfoCpy.pNext             = &externalMemCreateInfo;

  // Getting the allocation create info with the export flag capability
  VmaAllocationCreateInfo allocInfo{};
  NVVK_FAIL_RETURN(getAllocInfo(allocInfo, 0, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE, imageInfoCpy));

  return createImage(image, imageInfoCpy, imageViewInfo, allocInfo);
}

// Returns the VmaAllocationCreateInfo with the export flag and using the export memory pool
VkResult nvvk::ResourceAllocatorExport::getAllocInfo(VmaAllocationCreateInfo&  allocCreateInfo,
                                                     VmaAllocationCreateFlags  flags,
                                                     VmaMemoryUsage            usage,
                                                     const VkBufferCreateInfo& bufferInfo)
{
  VkResult result       = VK_SUCCESS;
  uint32_t memTypeIndex = UINT32_MAX;

  // Populate the flags and usage before vmaFindMemoryTypeIndexForBufferInfo since it uses them to query memTypeIndex.
  allocCreateInfo.flags = flags;
  allocCreateInfo.usage = usage;

  NVVK_FAIL_RETURN(vmaFindMemoryTypeIndexForBufferInfo(*this, &bufferInfo, &allocCreateInfo, &memTypeIndex));
  NVVK_FAIL_RETURN(getPool(memTypeIndex, allocCreateInfo.pool));

  return VK_SUCCESS;
}

// Returns the VmaAllocationCreateInfo with the export flag and using the export memory pool
VkResult nvvk::ResourceAllocatorExport::getAllocInfo(VmaAllocationCreateInfo& allocCreateInfo,
                                                     VmaAllocationCreateFlags flags,
                                                     VmaMemoryUsage           usage,
                                                     const VkImageCreateInfo& imageInfo)
{
  VkResult result       = VK_SUCCESS;
  uint32_t memTypeIndex = UINT32_MAX;

  // Populate the flags and usage before vmaFindMemoryTypeIndexForImageInfo since it uses them to query memTypeIndex.
  allocCreateInfo.flags = flags;
  allocCreateInfo.usage = usage;

  NVVK_FAIL_RETURN(vmaFindMemoryTypeIndexForImageInfo(*this, &imageInfo, &allocCreateInfo, &memTypeIndex));
  NVVK_FAIL_RETURN(getPool(memTypeIndex, allocCreateInfo.pool));

  return VK_SUCCESS;
}

// Returns the VmaPool with the export flag and using the export memory pool
VkResult nvvk::ResourceAllocatorExport::getPool(uint32_t memoryTypeIndex, VmaPool& pool)
{
  constexpr static VkExportMemoryAllocateInfo exportMemAllocInfo{
      .sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO_KHR,
      .pNext = nullptr,
#ifdef _WIN32
      .handleTypes = {VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT},
#else
      .handleTypes = {VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT},
#endif  // _WIN32
  };

  std::lock_guard<std::mutex> lock(m_mutex);
  if(m_pools[memoryTypeIndex] == VK_NULL_HANDLE)
  {

    VmaPoolCreateInfo poolCreateInfo   = {};
    poolCreateInfo.memoryTypeIndex     = memoryTypeIndex;
    poolCreateInfo.pMemoryAllocateNext = (void*)&exportMemAllocInfo;

    vmaCreatePool(*this, &poolCreateInfo, &m_pools[memoryTypeIndex]);
  }
  pool = m_pools[memoryTypeIndex];
  return VK_SUCCESS;
}

//--------------------------------------------------------------------------------------------------
// Usage example
//--------------------------------------------------------------------------------------------------
[[maybe_unused]] static void usage_ResourceAllocator()
{
  VkResult         result{};
  VkDevice         device{};
  VkPhysicalDevice physicalDevice{};
  VkInstance       instance{};

  nvvk::ResourceAllocator m_allocator{};
  result = m_allocator.init({
      .flags            = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT,
      .physicalDevice   = physicalDevice,
      .device           = device,
      .instance         = instance,
      .vulkanApiVersion = VK_API_VERSION_1_4,
  });  // Allocator

  nvvk::Buffer buffer;
  result = m_allocator.createBuffer(buffer, 1024, VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);

  nvvk::Image       image;
  VkImageCreateInfo imageInfo{};
  result = m_allocator.createImage(image, imageInfo);

  // destroy later
  m_allocator.destroyBuffer(buffer);
  m_allocator.destroyImage(image);
}
