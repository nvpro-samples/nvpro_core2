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
#include <cassert>

#include "descriptor_heap_writer.hpp"
#include "check_error.hpp"

#include <nvutils/alignment.hpp>

namespace {

bool formatHasDepth(VkFormat format)
{
  switch(format)
  {
    case VK_FORMAT_D16_UNORM:
    case VK_FORMAT_X8_D24_UNORM_PACK32:
    case VK_FORMAT_D32_SFLOAT:
    case VK_FORMAT_D24_UNORM_S8_UINT:
    case VK_FORMAT_D16_UNORM_S8_UINT:
    case VK_FORMAT_D32_SFLOAT_S8_UINT:
      return true;
    default:
      return false;
  }
}

bool formatHasStencil(VkFormat format)
{
  switch(format)
  {
    case VK_FORMAT_S8_UINT:
    case VK_FORMAT_D24_UNORM_S8_UINT:
    case VK_FORMAT_D16_UNORM_S8_UINT:
    case VK_FORMAT_D32_SFLOAT_S8_UINT:
      return true;
    default:
      return false;
  }
}

VkImageAspectFlags imageAspectMaskForFormat(VkFormat format)
{
  VkImageAspectFlags mask{};
  if(formatHasDepth(format))
    mask |= VK_IMAGE_ASPECT_DEPTH_BIT;
  if(formatHasStencil(format))
    mask |= VK_IMAGE_ASPECT_STENCIL_BIT;
  if(mask != 0)
    return mask;
  return VK_IMAGE_ASPECT_COLOR_BIT;
}

}  // namespace

namespace nvvk {

//--------------------------------------------------------------------------------------------------
bool DescriptorHeapWriter::isSupported(VkPhysicalDevice physicalDevice)
{
  VkPhysicalDeviceDescriptorHeapPropertiesEXT heapProps{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_HEAP_PROPERTIES_EXT};
  VkPhysicalDeviceProperties2 props2{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, .pNext = &heapProps};
  vkGetPhysicalDeviceProperties2(physicalDevice, &props2);
  return heapProps.samplerDescriptorSize > 0 && heapProps.imageDescriptorSize > 0;
}

//--------------------------------------------------------------------------------------------------
VkResult DescriptorHeapWriter::init(VkPhysicalDevice physicalDevice, VkDevice device)
{
  deinit();

  if(device == nullptr)
    return VK_ERROR_INITIALIZATION_FAILED;

  VkPhysicalDeviceDescriptorHeapPropertiesEXT heapProps{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_HEAP_PROPERTIES_EXT};
  VkPhysicalDeviceProperties2 props2{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, .pNext = &heapProps};
  vkGetPhysicalDeviceProperties2(physicalDevice, &props2);

  if(heapProps.samplerDescriptorSize == 0 || heapProps.imageDescriptorSize == 0)
    return VK_ERROR_FEATURE_NOT_PRESENT;

  m_samplerDescSize      = heapProps.samplerDescriptorSize;
  m_imageDescSize        = heapProps.imageDescriptorSize;
  m_bufferDescSize       = heapProps.bufferDescriptorSize;
  m_samplerDescAlignment = heapProps.samplerDescriptorAlignment;
  m_imageDescAlignment   = heapProps.imageDescriptorAlignment;
  m_bufferDescAlignment  = heapProps.bufferDescriptorAlignment;

  m_samplerHeapAlignment     = heapProps.samplerHeapAlignment;
  m_resourceHeapAlignment    = heapProps.resourceHeapAlignment;
  m_maxSamplerHeapSize       = heapProps.maxSamplerHeapSize;
  m_maxResourceHeapSize      = heapProps.maxResourceHeapSize;
  m_minSamplerReservedRange  = heapProps.minSamplerHeapReservedRange;
  m_minResourceReservedRange = heapProps.minResourceHeapReservedRange;

  m_device = device;
  return VK_SUCCESS;
}

//--------------------------------------------------------------------------------------------------
VkDeviceSize DescriptorHeapWriter::appendSamplerDescriptors(VkDeviceSize& offset, uint32_t count) const
{
  VkDeviceSize start = nvutils::align_up(offset, m_samplerDescAlignment);
  offset             = start + m_samplerDescSize * count;
  return start;
}

//--------------------------------------------------------------------------------------------------
VkDeviceSize DescriptorHeapWriter::appendImageDescriptors(VkDeviceSize& offset, uint32_t count) const
{
  VkDeviceSize start = nvutils::align_up(offset, m_imageDescAlignment);
  offset             = start + m_imageDescSize * count;
  return start;
}

//--------------------------------------------------------------------------------------------------
VkDeviceSize DescriptorHeapWriter::appendBufferDescriptors(VkDeviceSize& offset, uint32_t count) const
{
  VkDeviceSize start = nvutils::align_up(offset, m_bufferDescAlignment);
  offset             = start + m_bufferDescSize * count;
  return start;
}

//--------------------------------------------------------------------------------------------------
VkDeviceSize DescriptorHeapWriter::appendSamplerReservedRange(VkDeviceSize& offset) const
{
  VkDeviceSize start = offset;
  offset             = start + m_minSamplerReservedRange;
  return start;
}

//--------------------------------------------------------------------------------------------------
VkDeviceSize DescriptorHeapWriter::appendResourceReservedRange(VkDeviceSize& offset) const
{
  VkDeviceSize start = offset;
  offset             = start + m_minResourceReservedRange;
  return start;
}

//--------------------------------------------------------------------------------------------------
VkDeviceSize DescriptorHeapWriter::alignToSamplerHeap(VkDeviceSize offset) const
{
  return nvutils::align_up(offset, m_samplerHeapAlignment);
}

//--------------------------------------------------------------------------------------------------
VkDeviceSize DescriptorHeapWriter::alignToResourceHeap(VkDeviceSize offset) const
{
  return nvutils::align_up(offset, m_resourceHeapAlignment);
}

//--------------------------------------------------------------------------------------------------
VkResult DescriptorHeapWriter::writeSamplerDescriptor(const VkSamplerCreateInfo& samplerCreateInfo, void* dst) const
{
  assert(m_device != nullptr && dst != nullptr);

  VkHostAddressRangeEXT hostDst{
      .address = dst,
      .size    = m_samplerDescSize,
  };
  NVVK_FAIL_RETURN(vkWriteSamplerDescriptorsEXT(m_device, 1, &samplerCreateInfo, &hostDst));
  return VK_SUCCESS;
}

//--------------------------------------------------------------------------------------------------
VkResult DescriptorHeapWriter::writeImageDescriptor(VkImage image, VkFormat format, VkImageLayout layout, void* dst, VkImageViewType viewType) const
{
  assert(m_device != nullptr && dst != nullptr);

  VkImageSubresourceRange subresourceRange{
      .aspectMask     = imageAspectMaskForFormat(format),
      .baseMipLevel   = 0,
      .levelCount     = 1,
      .baseArrayLayer = 0,
      .layerCount     = 1,
  };

  VkImageViewCreateInfo viewInfo{
      .sType            = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
      .image            = image,
      .viewType         = viewType,
      .format           = format,
      .subresourceRange = subresourceRange,
  };

  VkImageDescriptorInfoEXT imageDescInfo{
      .sType  = VK_STRUCTURE_TYPE_IMAGE_DESCRIPTOR_INFO_EXT,
      .pView  = &viewInfo,
      .layout = layout,
  };

  VkResourceDescriptorInfoEXT resInfo{
      .sType = VK_STRUCTURE_TYPE_RESOURCE_DESCRIPTOR_INFO_EXT,
      .type  = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
      .data  = {.pImage = &imageDescInfo},
  };

  VkHostAddressRangeEXT hostDst{
      .address = dst,
      .size    = m_imageDescSize,
  };
  NVVK_FAIL_RETURN(vkWriteResourceDescriptorsEXT(m_device, 1, &resInfo, &hostDst));
  return VK_SUCCESS;
}

//--------------------------------------------------------------------------------------------------
VkResult DescriptorHeapWriter::writeBufferDescriptor(VkDeviceAddress bufferAddress, VkDeviceSize bufferSize, VkDescriptorType type, void* dst) const
{
  assert(m_device != nullptr && dst != nullptr);
  assert(type == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER || type == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);

  VkDeviceAddressRangeEXT addressRange{
      .address = bufferAddress,
      .size    = bufferSize,
  };

  VkResourceDescriptorInfoEXT resInfo{
      .sType = VK_STRUCTURE_TYPE_RESOURCE_DESCRIPTOR_INFO_EXT,
      .type  = type,
      .data  = {.pAddressRange = &addressRange},
  };

  VkHostAddressRangeEXT hostDst{
      .address = dst,
      .size    = m_bufferDescSize,
  };
  NVVK_FAIL_RETURN(vkWriteResourceDescriptorsEXT(m_device, 1, &resInfo, &hostDst));
  return VK_SUCCESS;
}

//--------------------------------------------------------------------------------------------------
[[maybe_unused]] static void usage_DescriptorHeapWriter()
{
  VkPhysicalDevice physicalDevice{};
  VkDevice         device{};
  VkImage          image{};
  VkFormat         format{VK_FORMAT_R8G8B8A8_UNORM};
  VkDeviceAddress  bufferAddr{};
  VkDeviceSize     bufferSize{1024};  // size in bytes of the buffer pointed to by bufferAddr

  DescriptorHeapWriter info;
  info.init(physicalDevice, device);

  // =====================================================================
  // Stack-allocated descriptor write
  // =====================================================================
  // Write a sampler descriptor into a stack buffer, then copy wherever needed.
  // The buffer must be at least info.samplerDescriptorSize() bytes
  // (SAMPLER_DESCRIPTOR_MAX_BYTE_SIZE is a safe upper bound for stack allocation).
  char                samplerTemp[DescriptorHeapWriter::SAMPLER_DESCRIPTOR_MAX_BYTE_SIZE];
  VkSamplerCreateInfo samplerCI{
      .sType     = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
      .magFilter = VK_FILTER_LINEAR,
      .minFilter = VK_FILTER_LINEAR,
  };
  NVVK_CHECK(info.writeSamplerDescriptor(samplerCI, samplerTemp));
  // samplerTemp now contains info.samplerDescriptorSize() bytes of opaque descriptor data.
  // Copy to staging, mapped memory, or feed to a compute shader.

  // =====================================================================
  // Uniform-type sampler heap layout
  // =====================================================================
  // All slots same size — simple index * size.
  // The reserved range is mandatory for any heap that will be bound to a command buffer.
  VkDeviceSize offset = 0;
  info.appendSamplerDescriptors(offset, 100);  // 100 sampler slots
  VkDeviceSize reservedStart   = info.appendSamplerReservedRange(offset);
  VkDeviceSize samplerHeapSize = info.alignToSamplerHeap(offset);
  // Create a buffer of `samplerHeapSize` bytes.
  // Bind with reservedRangeOffset = reservedStart, reservedRangeSize = minSamplerHeapReservedRange.
  (void)reservedStart;
  (void)samplerHeapSize;

  // =====================================================================
  // Mixed-type resource heap layout
  // =====================================================================
  // Images and buffers can have different sizes/alignments.
  // The append pattern handles alignment transitions between types.
  // The reserved range is mandatory for any heap that will be bound to a command buffer.
  offset                        = 0;
  VkDeviceSize imgStart         = info.appendImageDescriptors(offset, 1000);  // images at [imgStart..)
  VkDeviceSize bufStart         = info.appendBufferDescriptors(offset, 500);  // buffers at [bufStart..)
  VkDeviceSize resReserved      = info.appendResourceReservedRange(offset);
  VkDeviceSize resourceHeapSize = info.alignToResourceHeap(offset);
  // Create a buffer of `resourceHeapSize` bytes.
  // Bind with reservedRangeOffset = resReserved.

  // Write image descriptor at index 0 within the image region.
  // Stack buffer is sized to the spec max; only info.imageDescriptorSize() bytes are written.
  char imageTemp[DescriptorHeapWriter::IMAGE_DESCRIPTOR_MAX_BYTE_SIZE];
  NVVK_CHECK(info.writeImageDescriptor(image, format, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, imageTemp));
  // Copy info.imageDescriptorSize() bytes to (heapBase + imgStart + 0 * imageDescriptorSize)

  // Write buffer descriptor at index 0 within the buffer region.
  // Stack buffer is sized to the spec max (bufferDescriptorSize covers AS descriptors too);
  // only info.bufferDescriptorSize() bytes are written.
  char bufferTemp[DescriptorHeapWriter::BUFFER_DESCRIPTOR_MAX_BYTE_SIZE];
  NVVK_CHECK(info.writeBufferDescriptor(bufferAddr, bufferSize, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, bufferTemp));
  // Copy info.bufferDescriptorSize() bytes to (heapBase + bufStart + 0 * bufferDescriptorSize)

  // In the shader, the image at index 0 is at byte offset `imgStart`:
  //   Texture2D tex = Texture2D.Handle(uint2(imgStart / imageDescriptorSize, 0));
  // The buffer at index 0 is at byte offset `bufStart`:
  //   StructuredBuffer<T> buf = StructuredBuffer<T>.Handle(uint2(bufStart / bufferDescriptorSize, 0));
  (void)imgStart;
  (void)bufStart;
  (void)resReserved;
  (void)resourceHeapSize;

  info.deinit();
}

}  // namespace nvvk
