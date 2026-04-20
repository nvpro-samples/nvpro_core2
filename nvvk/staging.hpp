/*
* Copyright (c) 2025-2026, NVIDIA CORPORATION.  All rights reserved.
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
* SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
* SPDX-License-Identifier: Apache-2.0
*/

#pragma once

#include <cassert>

#include "semaphore.hpp"
#include "barriers.hpp"
#include "resource_allocator.hpp"

//-----------------------------------------------------------------
// StagingUploader is a class that allows to upload data to the GPU.
//
// Usage:
//      see usage_StagingUploader in staging.cpp
//-----------------------------------------------------------------

namespace nvvk {

class StagingUploader
{
public:
  static constexpr VkDeviceSize DEFAULT_LARGE_CHUNK_SIZE = 256ull * 1024 * 1024;

  StagingUploader()                                  = default;
  StagingUploader(const StagingUploader&)            = delete;
  StagingUploader& operator=(const StagingUploader&) = delete;
  StagingUploader(StagingUploader&& other) noexcept;
  StagingUploader& operator=(StagingUploader&& other) noexcept;
  ~StagingUploader()
  {
    assert(isAppendedEmpty() && "Did you forget cmdUploadAppended() or cancelAppended()");
    assert(m_resourceAllocator == nullptr && "Missing deinit()");
  }

  // explicit lifetime of resourceAllocator must be ensured externally
  void init(ResourceAllocator* resourceAllocator, bool enableLayoutBarriers = false);

  // deinit implicitly calls `releaseStaging(true)`
  void deinit();

  void setEnableLayoutBarriers(bool enableLayoutBarriers);

  ResourceAllocator* getResourceAllocator();

  // All temporary staging resources are associated with the provided SemaphoreState.

  // releases temporary staging resources based on SemaphoreState
  // if `!SemaphoreState.isValid ` then immediately released
  // otherwise runs `SemaphoreState.testSignaled`
  // if `forceAll` is true, then we assume it's safe delete all resources
  // this typically requires a device wait idle in advance.
  // virtual so derived classes can implement staging space management by different means.
  virtual void releaseStaging(bool forceAll = false);

  // get size of all staging resources
  size_t getStagingUsage() const { return m_stagingResourcesSize; }


  // returns staging buffer information that can be used for any manual copy operations.
  // if data is non-null it will be copied to bufferMapping automatically
  // virtual so that derived classes can implement this by different means.
  virtual VkResult acquireStagingSpace(BufferRange& stagingSpace, size_t dataSize, const void* data, const SemaphoreState& semaphoreState = {});

  // clear all pending copy operations.
  // this does NOT release resources of such operations and
  // `releaseStaging` is still required.
  void cancelAppended();

  // check if no operations were appended
  bool isAppendedEmpty() const;

  // start operations that are only on transfer queue
  // state is reset on `cancelAppended` or `cmdUploadAppended`
  void beginTransferOnly();

  // buffer.buffer, buffer.bufferSize and buffer.mapping are used
  // if buffer.mapping is valid, then we directly write to it
  // else staging space is acquired and a copy command appended
  // for later execution via `cmdUploadAppended`.
  // If `dataSize` is `0`, returns VK_SUCCESS
  // `data` must be valid if `dataSize` is non zero, and is copied linearly into staging memory with this call.
  // `semaphoreState` can be used to track the completion of this upload.
  VkResult appendBuffer(const nvvk::Buffer&   buffer,
                        VkDeviceSize          bufferOffset,
                        VkDeviceSize          dataSize,
                        const void*           data,
                        const SemaphoreState& semaphoreState = {});

  template <typename T>
  inline VkResult appendBuffer(const nvvk::Buffer& buffer, size_t bufferOffset, std::span<T> data, const SemaphoreState& semaphoreState = {})
  {
    return appendBuffer(buffer, bufferOffset, data.size_bytes(), data.data(), semaphoreState);
  }

  // same as above but infers `bufferOffset` from `bufferRange.offset` and `dataSize` from `bufferRange.range`
  VkResult appendBufferRange(const nvvk::BufferRange& bufferRange, const void* data, const SemaphoreState& semaphoreState = {});

  template <typename T>
  inline VkResult appendBufferRange(const nvvk::BufferRange& bufferRange, std::span<T> data, const SemaphoreState& semaphoreState = {})
  {
    assert(bufferRange.range == data.size_bytes());
    return appendBufferRange(bufferRange, data.data(), semaphoreState);
  }

  // `buffer.buffer`, `buffer.bufferSize` and `buffer.mapping` are used
  // if `buffer.mapping` is valid, then we return it as `uploadMapping`
  // else staging space is acquired its mapping is returned in `uploadMapping`
  // and a copy command is appended for later execution via `cmdUploadAppended`.
  // The pointer is valid until the next `cmdUploadAppended` or `cancelAppended` call.
  // If `dataSize` is `0`, returns VK_SUCCESS and sets `uploadMapping` to nullptr
  // `semaphoreState` can be used to track the completion of this upload.
  VkResult appendBufferMapping(const nvvk::Buffer&   buffer,
                               VkDeviceSize          bufferOffset,
                               VkDeviceSize          dataSize,
                               void*&                uploadMapping,
                               const SemaphoreState& semaphoreState = {});

  template <typename T>
  inline VkResult appendBufferMapping(const nvvk::Buffer&   buffer,
                                      size_t                bufferOffset,
                                      size_t                dataSize,
                                      T*&                   uploadMapping,
                                      const SemaphoreState& semaphoreState = {})
  {
    return appendBufferMapping(buffer, bufferOffset, dataSize, (void*&)uploadMapping, semaphoreState);
  }

  // same as `appendBufferMapping` but infers `bufferOffset` from `bufferRange.offset` and `dataSize` from `bufferRange.range`
  VkResult appendBufferRangeMapping(const nvvk::BufferRange& bufferRange, void*& uploadMapping, const SemaphoreState& semaphoreState = {});

  template <typename T>
  inline VkResult appendBufferRangeMapping(const nvvk::BufferRange& bufferRange,
                                           T*&                      uploadMapping,
                                           const SemaphoreState&    semaphoreState = {})
  {
    return appendBufferRangeMapping(bufferRange, (void*&)uploadMapping, semaphoreState);
  }


  // For Large buffers the uploads are using chunked staging allocations to avoid 4GB allocation limits.
  // If `dataSize` is `0`, returns VK_SUCCESS
  // `data` must be valid if `dataSize` is non zero, and is copied linearly into staging memory with this call.
  // `semaphoreState` can be used to track the completion of this upload.
  VkResult appendLargeBuffer(const nvvk::LargeBuffer& buffer,
                             VkDeviceSize             bufferOffset,
                             VkDeviceSize             dataSize,
                             const void*              data,
                             const SemaphoreState&    semaphoreState = {},
                             VkDeviceSize             chunkSize      = DEFAULT_LARGE_CHUNK_SIZE);

  template <typename T>
  inline VkResult appendLargeBuffer(const nvvk::LargeBuffer& buffer,
                                    size_t                   bufferOffset,
                                    std::span<T>             data,
                                    const SemaphoreState&    semaphoreState = {},
                                    VkDeviceSize             chunkSize      = DEFAULT_LARGE_CHUNK_SIZE)
  {
    return appendLargeBuffer(buffer, bufferOffset, data.size_bytes(), data.data(), semaphoreState, chunkSize);
  }

  // `dataSize` should be kept <= 1GB to be safe to acquire
  // If `dataSize` is `0`, returns VK_SUCCESS
  // `uploadMapping` is returned on success, user is expected to write the data into the staging buffer
  // linearly. The pointer is valid until the next `cmdUploadAppended` or `cancelAppended` call.
  // `semaphoreState` can be used to track the completion of this upload.
  VkResult appendLargeBufferMapping(const nvvk::LargeBuffer& buffer,
                                    VkDeviceSize             bufferOffset,
                                    VkDeviceSize             dataSize,
                                    void*&                   uploadMapping,
                                    const SemaphoreState&    semaphoreState = {});

  template <typename T>
  VkResult appendLargeBufferMapping(const nvvk::LargeBuffer& buffer,
                                    VkDeviceSize             bufferOffset,
                                    VkDeviceSize             dataSize,
                                    T*&                      uploadMapping,
                                    const SemaphoreState&    semaphoreState = {})
  {
    return appendLargeBufferMapping(buffer, bufferOffset, dataSize, (void*&)uploadMapping, semaphoreState);
  }

  // if the internal state StagingUploader's `enableLayoutBarriers` is true
  // then all appendImage functions may add barriers prior and after
  // the copy operations. These barriers are added
  // if `imageTex.descriptor.imageLayout` is not (VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL or
  //                                              VK_IMAGE_LAYOUT_GENERAL or
  //                                              VK_IMAGE_LAYOUT_SHARED_PRESENT_KHR)
  // then we transition to `VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL` prior any copy operation in
  // the appended batch. After the copy operations a transition is added back to the image's
  // original layout unless a valid `newLayout` is provided.
  // The `imageTex.descriptor.imageLayout` can also be updated
  // by providing `newLayout != VK_IMAGE_LAYOUT_UNDEFINED` through the post operation barriers.


  // All image textures must only use the color aspect.
  // This uploads mip 0/layer 0 only
  // `newLayout` may be transitioned into after the copy operations are completed depending
  // on the behavior the StagingUploader was set to, as described in the comment section above.
  // If `dataSize` is `0`, returns VK_SUCCESS
  // `data` must be valid if `dataSize` is non zero, and is copied linearly into staging memory with this call.
  // `semaphoreState` can be used to track the completion of this upload.
  VkResult appendImage(nvvk::Image&          image,
                       size_t                dataSize,
                       const void*           data,
                       VkImageLayout         newLayout      = VK_IMAGE_LAYOUT_UNDEFINED,
                       const SemaphoreState& semaphoreState = {});

  template <typename T>
  inline VkResult appendImage(nvvk::Image&          image,
                              std::span<T>          data,
                              VkImageLayout         newLayout      = VK_IMAGE_LAYOUT_UNDEFINED,
                              const SemaphoreState& semaphoreState = {})
  {
    return appendImage(image, data.size_bytes(), data.data(), newLayout, semaphoreState);
  }

  // Similar as function above.
  // `uploadMapping` is returned on success, user is expected to write the data into the staging buffer
  // linearly. The pointer is valid until the next `cmdUploadAppended` or `cancelAppended` call.
  VkResult appendImageMapping(nvvk::Image&          image,
                              size_t                dataSize,
                              void*&                uploadMapping,
                              VkImageLayout         newLayout      = VK_IMAGE_LAYOUT_UNDEFINED,
                              const SemaphoreState& semaphoreState = {});

  // Upload partial image data within an individual mip and/or multiple layers
  // `offset` within the target subresource
  // `extent` within the target subresource
  // `subresource` target within the image
  // `newLayout` may be transitioned into after the copy operations are completed depending
  // on the behavior the StagingUploader was set to, as described in the comment section above.
  // If `dataSize` is `0`, returns VK_SUCCESS
  // `data` must be valid if `dataSize` is non zero, and is copied linearly into staging memory with this call.
  // `semaphoreState` can be used to track the completion of this upload.
  VkResult appendImageSub(nvvk::Image&                    image,
                          const VkOffset3D&               offset,
                          const VkExtent3D&               extent,
                          const VkImageSubresourceLayers& subresource,
                          size_t                          dataSize,
                          const void*                     data,
                          VkImageLayout                   newLayout      = VK_IMAGE_LAYOUT_UNDEFINED,
                          const SemaphoreState&           semaphoreState = {});

  template <typename T>
  inline VkResult appendImageSub(nvvk::Image&                    image,
                                 const VkOffset3D&               offset,
                                 const VkExtent3D&               extent,
                                 const VkImageSubresourceLayers& subresource,
                                 std::span<T>                    data,
                                 VkImageLayout                   newLayout      = VK_IMAGE_LAYOUT_UNDEFINED,
                                 const SemaphoreState&           semaphoreState = {})
  {
    return appendImageSub(image, offset, extent, subresource, data.size_bytes(), data.data(), newLayout, semaphoreState);
  }

  // Variant of `appendImageSub`, in which we retrieve a pointer to write the image data to.
  // `uploadMapping` is returned on success, user is expected to write the data into the staging buffer
  // linearly. The pointer is valid until the next `cmdUploadAppended` or `cancelAppended` call.
  VkResult appendImageSubMapping(nvvk::Image&                    image,
                                 const VkOffset3D&               offset,
                                 const VkExtent3D&               extent,
                                 const VkImageSubresourceLayers& subresource,
                                 size_t                          dataSize,
                                 void*&                          uploadMapping,
                                 VkImageLayout                   newLayout      = VK_IMAGE_LAYOUT_UNDEFINED,
                                 const SemaphoreState&           semaphoreState = {});

  // returns true if the sum of staging resources used in pending operations
  // and the added size is beyond the limit
  bool checkAppendedSize(size_t limitInBytes, size_t addedSize = 0) const;

  // records pending operations (copy & relevant layout transitions) into the command buffer
  // and then resets the internal state for appended.
  void cmdUploadAppended(VkCommandBuffer cmd);

protected:
  void modifyImageBarrier(VkImageMemoryBarrier2& barrier);

  struct Batch
  {
    bool   transferOnly = false;
    size_t stagingSize  = 0;

    std::vector<VkBufferCopy2>            copyBufferRegions;
    std::vector<VkCopyBufferInfo2>        copyBufferInfos;
    std::vector<VkBufferImageCopy2>       copyBufferImageRegions;
    std::vector<VkCopyBufferToImageInfo2> copyBufferImageInfos;
    BarrierContainer                      pre;
    BarrierContainer                      post;
  };

  struct StagingResource
  {
    nvvk::Buffer   buffer;
    SemaphoreState semaphoreState;
  };

  ResourceAllocator* m_resourceAllocator    = nullptr;
  size_t             m_stagingResourcesSize = 0;
  bool               m_enableLayoutBarriers = false;

  std::vector<StagingResource> m_stagingResources;
  Batch                        m_batch{};
};

}  // namespace nvvk
