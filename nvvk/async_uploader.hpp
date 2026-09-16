/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <nvvk/resources.hpp>
#include <nvvk/semaphore.hpp>
#include <nvvk/staging.hpp>
#include <nvvk/buffer_suballocator.hpp>
#include <nvvk/command_pools.hpp>
#include <nvutils/id_pool.hpp>
#include <functional>
#include <array>

#include <nvvk/uploader_interface.hpp>

namespace nvvk {

// This class is mostly designed to collect uploads from
// background threads to buffers and images and issue them on a
// dedicated `transferQueue`. The queue access must be exclusive
// during operations of this class.
// The `targetQueue` would then later use the resources. There
// are some utility functions to handle resource ownership transfers
// or layout transitions.
// A developer would typically track the returned `nvvk::SemaphoreState`
// for completion on the host and then enqueue new vulkan work using that
// resource depending on the outcome.
//
// One can use the class synchronously on a local scope as well,
// by setting target and transfer queue to be equal and then completing
// all operations.
//
// When uploading from multiple threads we recommend the use
// of `acquireMapping` and then fill the data directly there.
//
// Have a look at `usage_AsyncUploader` in the cpp file for examples.
//
// The intent of this class and `nvvk::FrameUploader` is to replace the
// old `nvvk::StagingUploader` which we want to deprecate.
class AsyncUploader : public UploaderInterface
{
public:
  AsyncUploader() = default;
  ~AsyncUploader();
  AsyncUploader(const AsyncUploader&)            = delete;
  AsyncUploader& operator=(const AsyncUploader&) = delete;

  struct InitInfo
  {
    nvvk::ResourceAllocator* allocator{};
    // the uploader must have exclusive submit access to this queue,
    // while being used.
    nvvk::QueueInfo transferQueue{};

    // The target queue for where the resources are used at the end.
    // One must call `cmdDrainOwnershipBarriers` on this queue, prior
    // using resources that completed their transfer.
    //
    // If targetQueue.queue and transferQueue.queue match, the ownership
    // transfer is skipped.
    nvvk::QueueInfo targetQueue{};

    // how many command buffers can be in flight
    uint32_t commandPoolSize = 3;

    // after how much appended staging memory we trigger a flush
    uint32_t flushSize = 64 * 1024 * 1024;

    VkDeviceSize blockSize = 128 * 1024 * 1024;

    // keeping at least one block is recommended for persistent usage
    uint32_t keepBlockCount = 1;

    // number of copy operations per block
    // number of mappings in flight
    uint32_t    maxOperations = 0xFFFF;
    std::string debugName{};

    // if true, staging memory is guaranteed to have VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
    // if false, non-coherent staging memory may be used and is flushed before GPU copy
    bool forceCoherentMapping = true;

    std::function<void*(VkCommandBuffer cmd, uint32_t submitCount, nvvk::SemaphoreState& sem, bool begin, void* payload)> onSubmitCallback =
        nullptr;
  };

  VkResult init(const InitInfo& info);
  void     deinit();

  void setFlushSize(uint32_t flushSize)
  {
    std::lock_guard lock(m_appendLock);
    m_info.flushSize = flushSize;
  }

  uint32_t                         getSubmitCount() const { return m_submitCount; }
  uint32_t                         getWaitCmdCount() const { return m_waitCmdCount; }
  nvvk::BufferSubAllocator::Report getAllocationReport() const;
  bool                             hasOwnershipBarriers() const;

  uint32_t getAllocationCount() const { return uint32_t(m_allocationCount.load()); }

  // should be called at end of multiple upload append operations,
  // and/or once every frame
  // `semaphoreState` can be used to wait for anything up to this point
  // if there is no pending work does same as `releaseSubmitted`
  // on failure `outSemaphoreState` is left untouched, as the submit that
  // would signal it did not happen.
  VkResult flushPending(nvvk::SemaphoreState* outSemaphoreState = nullptr);

  // waits for completion of last submit
  VkResult waitForCompletion();

  // should be called once every frame
  // Only reclaims allocations of batches that were submitted and completed. Allocations of the
  // batch that is still being appended to are reclaimed once that batch was submitted and
  // completed as well, so an idle uploader holds on to its last staging memory until
  // `flushPending` was called.
  void releaseCompletedAllocations();

  // utility helper
  void flushAndWait()
  {
    flushPending();
    waitForCompletion();
    releaseCompletedAllocations();
  }

  // should be called after upload appends were done, to get a rough idea
  // what to wait for.
  // if there are pending operations left, returns next submit state
  // if flushed returns previous submit
  // if nothing was submitted yet, returns an invalid state
  //
  // due to async nature when you query after an appended uploaded, you might
  // not get exactly the submit your append was part of (as another thread could trigger a flush),
  // but typically this is good enough / guaranteed conservative.
  nvvk::SemaphoreState getSubmitState();

  // Apply this on destination graphics queue, prior using any resources, typically once per frame.
  // It adds the barriers by querying if the semaphore state of each resource has been signaled.
  //
  // Example order of operations
  // - worker threads
  //    - uploader.appendBuffer(myBuffer, ... &mySemaphoreState);
  // - main thread / target queue (in parallel)
  //    - uploader.flushPending
  //    - uploader.cmdDrainOwnershipBarriers(cmdFromTargetQueue);
  //    - if(mySemaphoreState.testSignaled(device)) {... safe to use myBuffer }
  void cmdDrainOwnershipBarriers(VkCommandBuffer cmd);

  // UploaderInterface, see `uploader_interface.hpp`

  // acquire mapping space to be used to write data for uploads.
  VkResult acquireMapping(size_t dataSize, nvvk::BufferRange& mappingSpace, MappingHandle& mappingHandle) override;
  // Only call when not making use of `commitMapping_` in the `append*Mappings` functions.
  // This means no other operation can use that mappingHandle afterwards, but that it may have been used before.
  void commitMapping(MappingHandle& mappingHandle) override;
  // Only call manually when aborting the use of the mapping space.
  // This means no other operation can use that mappingHandle afterwards, and it must not have been used before.
  void releaseMapping(MappingHandle& mappingHandle) override;

  VkResult appendBufferRange(const nvvk::BufferRange& buffer, const void* data, nvvk::SemaphoreState* outSemaphoreState) override;
  VkResult appendBufferRangeMappings(size_t                    rangeCount,
                                     const MappedBufferRanges* ranges,
                                     MappingHandle&            mappingHandle,
                                     bool                      commitMapping_    = true,
                                     nvvk::SemaphoreState*     outSemaphoreState = nullptr) override;

  VkResult appendImageSub(nvvk::Image&                    image,
                          const VkOffset3D&               offset,
                          const VkExtent3D&               extent,
                          const VkImageSubresourceLayers& subresource,
                          size_t                          dataSize,
                          const void*                     data,
                          VkImageLayout                   newLayout         = VK_IMAGE_LAYOUT_UNDEFINED,
                          nvvk::SemaphoreState*           outSemaphoreState = nullptr) override;
  VkResult appendImageSubMappings(nvvk::Image&           image,
                                  size_t                 imageSubCount,
                                  const MappedImageSubs* imageSubs,
                                  MappingHandle&         mappingHandle,
                                  bool                   commitMapping_    = true,
                                  VkImageLayout          newLayout         = VK_IMAGE_LAYOUT_UNDEFINED,
                                  nvvk::SemaphoreState*  outSemaphoreState = nullptr) override;

protected:
  union MappingHandleDetail
  {
    uint64_t handle;
    struct
    {
      uint8_t  valid;
      uint8_t  used;
      uint16_t blockIndex;
      uint32_t mappingIndex;
    };
  };

  VkResult acquireStagingSpace(nvvk::BufferSubAllocation& subAllocation, nvvk::BufferRange& stagingSpace, size_t dataSize);

  // requires m_appendLock
  void addStagingAllocation(const nvvk::BufferSubAllocation& subAllocation);
  // immediate free, only legal when nothing references the allocation
  // requires m_allocLock
  void freeStagingAllocation(nvvk::BufferSubAllocation& subAllocation);
  void releaseStagingAllocations(bool forceAll);

  VkResult submitBatch();
  VkResult submitBatchChecked(size_t addedSize);

  // on success invalidates `handle`
  bool commitMappingHandle(MappingHandle& handle, nvvk::BufferSubAllocation& subAllocation);

  void markRangeForFlush(uint32_t blockIndex, VkDeviceSize offset, VkDeviceSize size);
  void markForFlush(const nvvk::BufferSubAllocation& subAllocation);
  void flushBatchBlocks();

  // State of the last successful submit. Returns an invalid state when nothing was
  // submitted yet, as timeline value 0 is not a legal `SemaphoreState`.
  inline nvvk::SemaphoreState getLastSubmitState() const
  {
    if(m_transferTimelineValue <= 1)
    {
      return {};
    }
    // current value is always for next submit, hence -1
    return nvvk::SemaphoreState::makeFixed(m_transferTimelineSemaphore, m_transferTimelineValue - 1);
  }

  inline void getOutSemaphoreState(nvvk::SemaphoreState* out)
  {
    if(out != nullptr)
    {
      *out = m_transferSubmitState;
    }
  }

  struct StagingAllocation
  {
    nvvk::BufferSubAllocation subAllocation;
    nvvk::SemaphoreState      semaphoreState;
  };

  struct TargetBarriers
  {
    nvvk::BarrierContainer ownerAcquisitionBarriers;
    nvvk::SemaphoreState   semaphoreState;
  };

  static constexpr uint32_t MAX_FLUSH_BLOCKS = 64;
  static constexpr uint32_t RANGES_PER_BLOCK = 64;
  static constexpr uint64_t ALL_RANGE_BITS   = ~0ull;

  InitInfo               m_info;
  nvvk::StagingCopyBatch m_batch;
  // non-coherent flushing is done in ranges based on this granularity
  size_t m_flushRangeGranularity = 128;
  // a bit mask of which ranges we already flushed for each block
  std::array<uint64_t, MAX_FLUSH_BLOCKS> m_batchFlushRangeMasks{};

  // statistics
  // how often we had to wait for command buffer
  uint32_t m_waitCmdCount{0};
  // how many submits
  uint32_t m_submitCount{0};

  mutable std::mutex       m_allocLock;
  nvvk::BufferSubAllocator m_bufferSubAllocator;
  std::atomic_int32_t      m_allocationCount{0};

  std::mutex                             m_mappingLock;
  nvutils::IDPool                        m_mappingPool;
  std::vector<nvvk::BufferSubAllocation> m_mappingAllocations;

  std::mutex                     m_appendLock;
  std::vector<StagingAllocation> m_stagingAllocations;

  nvvk::SemaphoreState      m_transferSubmitState;
  nvvk::ManagedCommandPools m_transferCmdPool;
  VkSemaphore               m_transferTimelineSemaphore{};
  uint64_t                  m_transferTimelineValue = 1;

  mutable std::mutex          m_targetLock;
  std::vector<TargetBarriers> m_targetBarriers;
};

}  // namespace nvvk
