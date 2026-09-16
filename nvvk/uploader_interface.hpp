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
* SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
* SPDX-License-Identifier: Apache-2.0
*/

#pragma once

#include <span>

#include <nvvk/resources.hpp>
#include <nvvk/semaphore.hpp>

namespace nvvk {

// This interface will yield control over the actual submits of copying to the implementation.
// As well as the SemaphoreState management. However, each appended upload can query for
// the SempahoreState to track its completion, if required.
//
// The intent of this interface in combination with `nvvk::AsyncUploader` and `nvvk::FrameUploader`
// is to replace the old `nvvk::StagingUploader` which we want to deprecate.
//
// Some uploaders don't submit on their own, but record their copies into a command
// buffer the caller provides, see `CmdUploaderInterface`
class UploaderInterface
{
public:
  // intentionally not providing a virtual destructor, because the interface is only meant
  // for operations, not for managing the lifetime of derived classes, as those classes
  // bring their own implementation details.

  // Opaque handle to an acquired mapping; a zero / default-constructed handle is invalid.
  struct MappingHandle
  {
    uint64_t value = 0;
    explicit operator bool() const { return value != 0; }
    bool     operator==(const MappingHandle& other) const = default;
  };

  struct MappedBufferRanges
  {
    nvvk::BufferRange bufferRange;
    nvvk::BufferRange mappingRange;
  };

  struct MappedImageSubs
  {
    VkOffset3D               offset{};
    VkExtent3D               extent{};
    VkImageSubresourceLayers subresource{};
    nvvk::BufferRange        mappingSpace;
  };

  // This interface allows decoupling acquiring the staging space and writing to it
  // from the submit of the copy command later. In a multi-threaded scenario each
  // thread should acquire a bit of a bigger chunk that it fills and then uses for uploads.

  virtual VkResult acquireMapping(size_t dataSize, nvvk::BufferRange& mappingSpace, MappingHandle& mappingHandle) = 0;

  // Call when transferring ownership of the used mapping to the uploader.
  // Normally use `commitMapping_ = true` in the last
  // use of a mapping handle in the later `append*Mapping` functions,
  // or call this after all such uses were appended.
  // After a handle was consumed it cannot be used again.
  virtual void commitMapping(MappingHandle& mappingHandle) = 0;

  // Call when we aborting the use of the mapping. The mapping is released immediately
  // it must not have been part of any append operations before.
  virtual void releaseMapping(MappingHandle& mappingHandle) = 0;


  //////////////////////////////////////////////////////////////////////////

  // core interface functions

  // if `outSemaphoreState` is provided it will store the semaphore state that can be
  // used to track completion
  // if `consumeMapping_ == true`, then we transfer ownership of the mapping into the uploader,
  // at that point don't call consumeMapping manually.

  virtual VkResult appendBufferRange(const nvvk::BufferRange& bufferRange, const void* data, nvvk::SemaphoreState* outSemaphoreState) = 0;
  virtual VkResult appendBufferRangeMappings(size_t                    rangeCount,
                                             const MappedBufferRanges* ranges,
                                             MappingHandle&            mappingHandle,
                                             bool                      commitMapping_    = true,
                                             nvvk::SemaphoreState*     outSemaphoreState = nullptr) = 0;

  virtual VkResult appendImageSub(nvvk::Image&                    image,
                                  const VkOffset3D&               offset,
                                  const VkExtent3D&               extent,
                                  const VkImageSubresourceLayers& subresource,
                                  size_t                          dataSize,
                                  const void*                     data,
                                  VkImageLayout                   newLayout         = VK_IMAGE_LAYOUT_UNDEFINED,
                                  nvvk::SemaphoreState*           outSemaphoreState = nullptr)         = 0;
  virtual VkResult appendImageSubMappings(nvvk::Image&           image,
                                          size_t                 imageSubCount,
                                          const MappedImageSubs* imageSubs,
                                          MappingHandle&         mappingHandle,
                                          bool                   commitMapping_    = true,
                                          VkImageLayout          newLayout         = VK_IMAGE_LAYOUT_UNDEFINED,
                                          nvvk::SemaphoreState*  outSemaphoreState = nullptr) = 0;

  //////////////////////////////////////////////////////////////////////////

  // utility wrappers

  inline VkResult appendBuffer(const nvvk::Buffer& buffer, size_t offset, size_t dataSize, const void* data, nvvk::SemaphoreState* outSemaphoreState = nullptr)
  {
    nvvk::BufferRange bufferRange;
    bufferRange.buffer  = buffer.buffer;
    bufferRange.offset  = offset;
    bufferRange.range   = dataSize;
    bufferRange.mapping = buffer.mapping ? buffer.mapping + offset : 0;
    bufferRange.address = buffer.address ? buffer.address + offset : 0;

    return appendBufferRange(bufferRange, data, outSemaphoreState);
  }

  inline VkResult appendBufferMapping(const nvvk::Buffer&      buffer,
                                      size_t                   offset,
                                      size_t                   dataSize,
                                      const nvvk::BufferRange& mappingSpace,
                                      MappingHandle&           mappingHandle,
                                      bool                     commitMapping_    = true,
                                      nvvk::SemaphoreState*    outSemaphoreState = nullptr)
  {
    MappedBufferRanges mappedRange;
    mappedRange.bufferRange.buffer  = buffer.buffer;
    mappedRange.bufferRange.offset  = offset;
    mappedRange.bufferRange.range   = dataSize;
    mappedRange.bufferRange.mapping = buffer.mapping ? buffer.mapping + offset : 0;
    mappedRange.bufferRange.address = buffer.address ? buffer.address + offset : 0;
    mappedRange.mappingRange        = mappingSpace;

    return appendBufferRangeMappings(1, &mappedRange, mappingHandle, commitMapping_, outSemaphoreState);
  }

  template <typename T>
  inline VkResult appendBufferRange(const nvvk::BufferRange& bufferRange, std::span<T> data, nvvk::SemaphoreState* outSemaphoreState = nullptr)
  {
    assert(bufferRange.range == data.size_bytes());
    return appendBufferRange(bufferRange, data.data(), outSemaphoreState);
  }

  template <typename T>
  inline VkResult appendBuffer(const nvvk::Buffer& buffer, size_t offset, std::span<T> data, nvvk::SemaphoreState* outSemaphoreState = nullptr)
  {
    return appendBuffer(buffer, offset, data.size_bytes(), data.data(), outSemaphoreState);
  }

  // Split bigger uploads into multiple appends via `offset` / `dataSize`.
  inline VkResult appendLargeBuffer(const nvvk::LargeBuffer& buffer,
                                    size_t                   offset,
                                    size_t                   dataSize,
                                    const void*              data,
                                    nvvk::SemaphoreState*    outSemaphoreState = nullptr)
  {
    nvvk::BufferRange bufferRange;
    bufferRange.buffer  = buffer.buffer;
    bufferRange.offset  = offset;
    bufferRange.range   = dataSize;
    bufferRange.mapping = nullptr;
    bufferRange.address = buffer.address + offset;

    return appendBufferRange(bufferRange, data, outSemaphoreState);
  }

  inline VkResult appendLargeBufferMapping(const nvvk::LargeBuffer& buffer,
                                           size_t                   offset,
                                           size_t                   dataSize,
                                           const nvvk::BufferRange& mappingSpace,
                                           MappingHandle&           mappingHandle,
                                           bool                     commitMapping_    = true,
                                           nvvk::SemaphoreState*    outSemaphoreState = nullptr)
  {
    MappedBufferRanges mappedRange;
    mappedRange.bufferRange.buffer  = buffer.buffer;
    mappedRange.bufferRange.offset  = offset;
    mappedRange.bufferRange.range   = dataSize;
    mappedRange.bufferRange.mapping = nullptr;
    mappedRange.bufferRange.address = buffer.address + offset;
    mappedRange.mappingRange        = mappingSpace;

    return appendBufferRangeMappings(1, &mappedRange, mappingHandle, commitMapping_, outSemaphoreState);
  }

  template <typename T>
  inline VkResult appendImageSub(nvvk::Image&                    image,
                                 const VkOffset3D&               offset,
                                 const VkExtent3D&               extent,
                                 const VkImageSubresourceLayers& subresource,
                                 std::span<T>                    data,
                                 VkImageLayout                   newLayout         = VK_IMAGE_LAYOUT_UNDEFINED,
                                 nvvk::SemaphoreState*           outSemaphoreState = nullptr)
  {
    return appendImageSub(image, offset, extent, subresource, data.size_bytes(), data.data(), newLayout, outSemaphoreState);
  }

  inline VkResult appendImage(nvvk::Image&          image,
                              size_t                dataSize,
                              const void*           data,
                              VkImageLayout         newLayout         = VK_IMAGE_LAYOUT_UNDEFINED,
                              nvvk::SemaphoreState* outSemaphoreState = nullptr)
  {
    return appendImageSub(image, {0, 0, 0}, image.extent,
                          {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .mipLevel = 0, .baseArrayLayer = 0, .layerCount = 1},
                          dataSize, data, newLayout, outSemaphoreState);
  }

  template <typename T>
  inline VkResult appendImage(nvvk::Image&          image,
                              std::span<T>          data,
                              VkImageLayout         newLayout         = VK_IMAGE_LAYOUT_UNDEFINED,
                              nvvk::SemaphoreState* outSemaphoreState = nullptr)
  {
    return appendImage(image, data.size_bytes(), data.data(), newLayout, outSemaphoreState);
  }
};

// Interface for uploaders that do not own a queue and therefore cannot submit by themselves.
// The caller decides where the copies land in its command stream, and is responsible for
// their submit and synchronization.
class CmdUploaderInterface : public UploaderInterface
{
public:
  // Records the appended operations (copy & relevant layout transitions) into the command buffer.
  virtual void cmdUploadAppended(VkCommandBuffer cmd) = 0;
};

}  // namespace nvvk
