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
// StagingUploader is a class that allows to upload data to the GPU,
// it implements the StagingUploaderBase interface.
//
// Usage:
//      see usage_StagingUploader in staging.cpp
//-----------------------------------------------------------------

namespace nvvk {

// Utility class to batch copy operations and related barriers.
// Fully public so classes that manage uploads or downloads can use it directly
// with full flexibility
class StagingCopyBatch
{
public:
  //////////////////////////////////////////////////////////////////////////
  // persistent state

  // enqueues layout transitions for images
  // `pre` barrier will contain a barrier transition to a copyable state if it's not already,
  // and `post` barrier will contain back to original or the optional `newLayout`
  bool enableLayoutBarriers = false;

  // relevant if `enableLayoutBarriers` are enabled.
  // Triggers `applyImageBarrierMask` which allows disabling bits that
  // are not supported by the vulkan queue that this batch is executed on.
  // Example: for a transfer only queue, would use
  // VK_ACCESS_2_TRANSFER_WRITE_BIT | VK_ACCESS_2_TRANSFER_READ_BIT accessMask bits
  // as well as VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT stageMask
  bool                     enableLayoutBarrierMask        = false;
  VkAccessFlagBits2        layoutBarrierAccessMask        = ~0ULL;
  VkPipelineStageFlagBits2 layoutBarrierPipelineStageMask = ~0ULL;

  // enqueues resource ownership barriers
  bool enableOwnerBarriers = false;
  // used for ownership barriers
  uint32_t srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  uint32_t dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;

  //////////////////////////////////////////////////////////////////////////
  // these states are reset/cleared on `reset` and/or `cmdCopyAppended`
  size_t stagingSize = 0;

  // These copy operations are triggered during `cmdCopyAppended`.
  // The `copyInfo.pRegions` are computed from the tightly packed
  // regions using a prefix sum over `copyInfo.regionCount`
  std::vector<VkBufferCopy2>            copyBufferRegions;
  std::vector<VkCopyBufferInfo2>        copyBufferInfos;
  std::vector<VkBufferImageCopy2>       copyBufferImageRegions;
  std::vector<VkCopyBufferToImageInfo2> copyBufferImageInfos;

  // barriers performed on the provided command buffer during `cmdCopyAppended`
  BarrierContainer pre;
  BarrierContainer post;
  // external barriers must be gathered externally before `cmdCopyAppended`
  BarrierContainer acquire;

  // nothing in the batch
  bool isAppendedEmpty() const;

  // check if we should flush because already enough traffic caused by the batch
  inline bool checkAppendedSize(size_t limitInBytes, size_t addedSize = 0) const
  {
    return stagingSize && (stagingSize + addedSize) > limitInBytes;
  }

  // Records pending copy and barrier operations into `cmd` and then clears the batch state.
  // Warning: Staging resource ownership and lifetime must be managed by caller.
  // calls `reset` after enqueuing the commands.
  void cmdCopyAppended(VkCommandBuffer cmd);

  // resets all vectors and intermediate `stagingSize`
  void reset();

  // adds a buffer copy
  // calls `addOwnerBufferBarrier` if `allowOwnerBarrier` and `enableOwnerBarriers` are true
  // `allowOwnerBarrier` is optional for scenarios where we might copy to a big buffer a few times
  // and then trigger `addOwnerBufferBarrier` manually for the full range.
  void addBufferCopy(VkBuffer     stagingBuffer,
                     VkDeviceSize stagingOffset,
                     VkBuffer     buffer,
                     VkDeviceSize bufferOffset,
                     VkDeviceSize dataSize,
                     bool         allowOwnerBarrier = true);

  // adds an image subresource copy
  // calls `handlePreImageBarrier` and `handlePostImageBarrier`
  void addImageCopy(VkBuffer                        stagingBuffer,
                    VkDeviceSize                    stagingOffset,
                    VkImage                         image,
                    VkImageLayout&                  imageLayout,
                    VkImageLayout                   newLayout,
                    size_t                          dataSize,
                    const VkImageSubresourceLayers& subresource,
                    const VkOffset3D&               offset,
                    const VkExtent3D&               extent,
                    const VkImageSubresourceRange*  subresourceRange = nullptr);

  // ownership barrier management for buffers.
  // adds `post` and `acquire` barrier
  void addOwnerBufferBarrier(VkBuffer buffer, VkDeviceSize offset, VkDeviceSize size);

  // modifies automatic image memory barrier,
  // to remove pipelines/access bits that aren't supported on
  // the queue that this batch is executed on
  void applyImageBarrierMask(VkImageMemoryBarrier2& barrier);

  // may add `pre` barrier for layout transition prior copy.
  // `currentLayout` layout the image is in before copy operation
  // `newLayout` layout that is desired after copy operation (if not VK_IMAGE_LAYOUT_UNDEFINED)
  // returns the layout in which the copy is done in.
  // may call `applyImageBarrierMask`
  [[nodiscard]] VkImageLayout handlePreImageBarrier(VkImage                        image,
                                                    VkImageLayout                  currentLayout,
                                                    VkImageLayout                  newLayout,
                                                    const VkImageSubresourceRange* subresourceRange = nullptr);

  // may add `post` barrier for layout transition after copy and/or ownership transfer.
  // may add `acquire` barrier for ownership transfer.
  // may call `applyImageBarrierMask`
  // `currentLayout` layout in which copy is done
  // `originalLayout` before operations
  // `newLayout` desired after copy operation (if not VK_IMAGE_LAYOUT_UNDEFINED)
  // returns either `originalLayout` or `newLayout`
  [[nodiscard]] VkImageLayout handlePostImageBarrier(VkImage                        image,
                                                     VkImageLayout                  currentLayout,
                                                     VkImageLayout                  originalLayout,
                                                     VkImageLayout                  newLayout,
                                                     const VkImageSubresourceRange* subresourceRange = nullptr);
};

// A basic implementation of a StagingUploader that uses
// the provided resource allocator to acquire staging resources
// as individual buffers.
// Does take care of non-coherent memory flushes and various
// barrier management.
// We do recommend more sophisticated designs than this, that
// are tuned for the application need, like per-frame submissions
// or asynchronous submissions.
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
  // `enableLayoutBarriers` is passed to `setEnableLayoutBarriers`
  // `forceCoherentMapping` means we use memory for staging buffers that is guaranteed to have
  // VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, avoiding the need to call vkFlushMappedMemoryRanges.
  void init(ResourceAllocator* resourceAllocator, bool enableLayoutBarriers = false, bool forceCoherentMapping = true);

  // deinit implicitly calls `releaseStaging(true)`
  void deinit();

  ResourceAllocator* getResourceAllocator();

  //////////////////////////////////////////////////////////////////////////

  // Returns staging buffer information that can be used for any manual copy operations.
  // If data is non-null it will be copied to bufferMapping automatically.
  // The returned `stagingSpace` is valid until `cmdUploadAppended` or `cancelAppended`.
  virtual VkResult acquireStagingSpace(BufferRange& stagingSpace, size_t dataSize, const void* data, const SemaphoreState& semaphoreState = {});

  // Releases temporary staging resources based on SemaphoreState.
  // If a resources `!SemaphoreState.isValid()` then it is immediately released,
  // otherwise runs `SemaphoreState.testSignaled`.
  // If `forceAll` is true, then we assume it's safe delete all resources, which
  // typically requires a device wait idle in advance.
  virtual void releaseStaging(bool forceAll = false);

  // Records pending operations (copy & relevant layout transitions) into the command buffer
  // and then resets the internal state for appended / acquired staging space.
  // When ownership transfers are active, don't forget to copy/run the `getOwnerDestinationBarriers`
  // at the acquiring queue before calling this function.
  // Triggers `m_batch.reset(); resetStaging(false);`
  // This call may also flush non-coherently mapped staging buffers when applicable
  virtual void cmdUploadAppended(VkCommandBuffer cmd);

  // Clears all pending copy operations and their
  // acquired staging spaces.
  // Triggers `m_batch.reset(); resetStaging(true);`
  void cancelAppended();

  //////////////////////////////////////////////////////////////////////////

  // start operations that are only on transfer queue
  // state is set to false on `cancelAppended` or `cmdUploadAppended`
  void beginTransferOnly();

  // Handles layout transitions for images on a VkImageSubresourceRange level.
  // Note that image sub updates that don't expand the full sub resource are not allowed.
  void setEnableLayoutBarriers(bool enableLayoutBarriers);

  // Enables resource ownership transfer barriers.
  // Implicitly enables the "post" image layout barriers.
  void setEnableOwnerBarriers(bool enableOwnerBarriers, uint32_t srcQueueFamilyIndex, uint32_t dstQueueFamilyIndex);

  // Must execute these barriers on the destination queue.
  // Must copy / execute content of this container before calling `cmdUploadAppended`.
  const BarrierContainer& getOwnerAcquisitionBarriers() const { return m_batch.acquire; }

  //////////////////////////////////////////////////////////////////////////

  // check if no operations were appended
  bool isAppendedEmpty() const;

  // get size of all staging resources
  size_t getStagingUsage() const { return m_stagingResourcesSize; }

  // returns true if the sum of staging resources used in pending operations
  // and the added size is beyond the limit
  bool checkAppendedSize(size_t limitInBytes, size_t addedSize = 0) const;

  //////////////////////////////////////////////////////////////////////////

  // All temporary staging resources are associated with the provided SemaphoreState.
  // When mapped pointers are used, their use is only valid until the next
  // `cmdUploadAppended` or `cancelAppended` call.

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
  // Note:
  // `image.descriptor.imageLayout` is tested when automatic layout barriers are enabled. However,
  // as mips have their own layout states, we advise resetting `image.descriptor.imageLayout`
  // before each mip upload when uploading many mips. See `usage_StagingUploader`.
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

protected:
  virtual void resetStaging(bool isCancel);

  struct StagingResource
  {
    nvvk::Buffer   buffer;
    SemaphoreState semaphoreState;
  };

  ResourceAllocator*           m_resourceAllocator = nullptr;
  std::vector<StagingResource> m_stagingResources;
  size_t                       m_stagingResourcesSize = 0;
  StagingCopyBatch             m_batch;
  size_t                       m_batchStagingCount    = 0;
  bool                         m_batchRequiresFlush   = false;
  bool                         m_forceCoherentMapping = true;
};

}  // namespace nvvk
