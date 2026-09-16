/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cassert>

#include <volk.h>

#include <nvvk/check_error.hpp>
#include <nvvk/debug_util.hpp>

#include "async_uploader.hpp"

namespace nvvk {

namespace {

// returns a mask with all bits from `firstBit` to `lastBit` set, both inclusive
uint64_t setInclusiveBitRange(uint32_t firstBit, uint32_t lastBit)
{
  assert(firstBit <= lastBit && lastBit < 64);
  // both shift counts stay < 64, hence no undefined behavior for a full range
  return (~0ull >> (63 - lastBit)) & (~0ull << firstBit);
}

}  // namespace


AsyncUploader::~AsyncUploader()
{
  assert(m_info.allocator == nullptr && "Missing deinit()");
}

VkResult AsyncUploader::init(const InitInfo& info)
{
  assert(m_info.allocator == nullptr && "Missing deinit()");
  assert(info.allocator);

  VkResult result;
  VkDevice device = info.allocator->getDevice();

  result = NVVK_FAIL_REPORT(nvvk::createTimelineSemaphore(device, 0, m_transferTimelineSemaphore));
  if(result != VK_SUCCESS)
  {
    return result;
  }

  result = NVVK_FAIL_REPORT(m_transferCmdPool.init(device, info.transferQueue.familyIndex, nvvk::ManagedCommandPools::Mode::SEMAPHORE_STATE,
                                                   VK_COMMAND_POOL_CREATE_TRANSIENT_BIT, info.commandPoolSize));
  if(result != VK_SUCCESS)
  {
    vkDestroySemaphore(device, m_transferTimelineSemaphore, nullptr);
    m_transferTimelineSemaphore = nullptr;
    return result;
  }

  nvvk::BufferSubAllocator::InitInfo subInitInfo;
  subInitInfo.allocationFlags = VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
  subInitInfo.memoryUsage = info.forceCoherentMapping ? VMA_MEMORY_USAGE_CPU_ONLY : VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
  subInitInfo.usageFlags  = VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT;
  subInitInfo.resourceAllocator     = info.allocator;
  subInitInfo.debugName             = info.debugName;
  subInitInfo.blockSize             = info.blockSize;
  subInitInfo.maxAllocatedSize      = info.blockSize * MAX_FLUSH_BLOCKS;
  subInitInfo.perBlockAllocations   = info.maxOperations;
  subInitInfo.keepBlockCount        = info.keepBlockCount;
  subInitInfo.threadSafeBlockAccess = true;

  m_flushRangeGranularity = info.blockSize / RANGES_PER_BLOCK;
  assert(info.blockSize % RANGES_PER_BLOCK == 0);

  result = NVVK_FAIL_REPORT(m_bufferSubAllocator.init(subInitInfo));
  if(result != VK_SUCCESS)
  {
    vkDestroySemaphore(device, m_transferTimelineSemaphore, nullptr);
    m_transferCmdPool.deinit();
    m_transferTimelineSemaphore = nullptr;
    return result;
  }

  m_batch.enableLayoutBarriers = true;
  m_batch.enableOwnerBarriers  = info.transferQueue.queue != info.targetQueue.queue;
  m_batch.dstQueueFamilyIndex  = info.targetQueue.familyIndex;
  m_batch.srcQueueFamilyIndex  = info.transferQueue.familyIndex;

  // When we own a dedicated transfer queue, the layout barriers are recorded on it, so they
  // must not use stages/accesses that a transfer-only queue family does not support.
  m_batch.enableLayoutBarrierMask        = m_batch.enableOwnerBarriers;
  m_batch.layoutBarrierAccessMask        = VK_ACCESS_2_TRANSFER_WRITE_BIT | VK_ACCESS_2_TRANSFER_READ_BIT;
  m_batch.layoutBarrierPipelineStageMask = VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT;

  m_transferSubmitState = nvvk::SemaphoreState::makeFixed(m_transferTimelineSemaphore, m_transferTimelineValue);

  m_mappingAllocations.resize(info.maxOperations);
  m_mappingPool.init(info.maxOperations);

  m_info = info;

  return VK_SUCCESS;
}

void AsyncUploader::deinit()
{
  if(m_info.allocator == nullptr)
    return;

  VkDevice device = m_info.allocator->getDevice();
  NVVK_CHECK(vkQueueWaitIdle(m_info.transferQueue.queue));

  releaseStagingAllocations(true);

  for(size_t i = 0; i < m_mappingAllocations.size(); i++)
  {
    if(m_mappingAllocations[i])
    {
      freeStagingAllocation(m_mappingAllocations[i]);
      m_mappingPool.destroyID(uint32_t(i));
    }
  }

  // everything the user handed us must have been given back by now
  assert(m_stagingAllocations.empty() && m_allocationCount == 0);
  assert(m_bufferSubAllocator.getReport().requestedSize == 0);
  m_batch.reset();
  vkDestroySemaphore(device, m_transferTimelineSemaphore, nullptr);
  m_transferTimelineSemaphore = nullptr;
  m_transferCmdPool.deinit();
  m_bufferSubAllocator.deinit();
  m_info               = {};
  m_mappingAllocations = {};
  m_mappingPool.deinit();
  m_transferSubmitState   = {};
  m_transferTimelineValue = 1;
  m_submitCount           = 0;
  m_waitCmdCount          = 0;
  m_targetBarriers.clear();
  m_batchFlushRangeMasks = {};
  m_allocationCount      = 0;
}

VkResult AsyncUploader::submitBatch()
{
  VkCommandBuffer cmd;
  NVVK_FAIL_RETURN(m_transferCmdPool.acquireCommandBuffer(m_transferSubmitState, cmd));
  NVVK_DBG_CUSTOM_NAME(cmd, "AsyncUploader:cmd:" + std::to_string(m_transferTimelineValue));

  releaseStagingAllocations(false);

  // Snapshot the ownership-acquisition barriers before cmdCopyAppended() consumes the batch. The
  // record is only published to m_targetBarriers after a successful submit (below), so a failed
  // vkQueueSubmit2() leaves no record whose semaphore would never be signaled.
  TargetBarriers record{
      .ownerAcquisitionBarriers = m_batch.acquire,
      .semaphoreState           = m_transferSubmitState,
  };

  VkCommandBufferBeginInfo cmdBegin{
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
      .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
  };
  NVVK_FAIL_RETURN(vkBeginCommandBuffer(cmd, &cmdBegin));

  void* payload = nullptr;
  if(m_info.onSubmitCallback)
  {
    payload = m_info.onSubmitCallback(cmd, m_submitCount, m_transferSubmitState, true, nullptr);
  }
  if(!m_info.forceCoherentMapping)
  {
    flushBatchBlocks();
  }
  m_batch.cmdCopyAppended(cmd);
  if(m_info.onSubmitCallback)
  {
    m_info.onSubmitCallback(cmd, m_submitCount, m_transferSubmitState, false, payload);
  }
  NVVK_FAIL_RETURN(vkEndCommandBuffer(cmd));
  m_waitCmdCount = m_transferCmdPool.getWaitCount();
  m_submitCount++;

  VkCommandBufferSubmitInfo cmdSubmitInfo = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO};
  cmdSubmitInfo.commandBuffer             = cmd;

  VkSemaphoreSubmitInfo semSubmitInfo = nvvk::makeSemaphoreSubmitInfo(m_transferSubmitState, VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT);

  // prepare actual submit
  VkSubmitInfo2 submitInfo2            = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
  submitInfo2.commandBufferInfoCount   = 1;
  submitInfo2.pCommandBufferInfos      = &cmdSubmitInfo;
  submitInfo2.signalSemaphoreInfoCount = 1;
  submitInfo2.pSignalSemaphoreInfos    = &semSubmitInfo;

  // submit to queue
  NVVK_FAIL_RETURN(vkQueueSubmit2(m_info.transferQueue.queue, 1, &submitInfo2, VK_NULL_HANDLE));

  // submit succeeded: now retain the ownership-barrier record for cmdDrainOwnershipBarriers()
  {
    std::lock_guard lock(m_targetLock);
    m_targetBarriers.push_back(record);
  }

  m_transferTimelineValue++;
  m_transferSubmitState = nvvk::SemaphoreState::makeFixed(m_transferTimelineSemaphore, m_transferTimelineValue);

  return VK_SUCCESS;
}

void AsyncUploader::releaseStagingAllocations(bool forceAll)
{
  VkDevice device = m_info.allocator->getDevice();

  nvvk::SemaphoreStateSignalCache semaphoreCache;

  size_t writeIndex = 0;

  for(size_t readIndex = 0; readIndex < m_stagingAllocations.size(); readIndex++)
  {
    StagingAllocation& allocation = m_stagingAllocations[readIndex];

    if(forceAll || !allocation.semaphoreState.isValid() || (semaphoreCache.testSignaled(device, allocation.semaphoreState)))
    {
      std::lock_guard lock(m_allocLock);

      freeStagingAllocation(allocation.subAllocation);
    }
    else if(readIndex != writeIndex)
    {
      m_stagingAllocations[writeIndex++] = allocation;
    }
    else
    {
      writeIndex++;
    }
  }
  m_stagingAllocations.resize(writeIndex);
}

VkResult AsyncUploader::submitBatchChecked(size_t addedSize)
{
  if(m_batch.checkAppendedSize(m_info.flushSize, addedSize))
  {
    return submitBatch();
  }
  return VK_SUCCESS;
}

bool AsyncUploader::commitMappingHandle(MappingHandle& handle, nvvk::BufferSubAllocation& subAllocation)
{
  MappingHandleDetail detail{};
  detail.handle        = handle.value;
  const uint32_t index = detail.mappingIndex;

  assert(detail.valid != 0);
  if(detail.valid == 0)
  {
    // never touch the slot of an invalid handle, index 0 may well belong to someone else
    return false;
  }

  // copy subAllocation before we destroy the ID, as otherwise
  // someone else can overwrite the slot
  subAllocation               = m_mappingAllocations[index];
  m_mappingAllocations[index] = {};

  bool result;
  {
    std::lock_guard lock(m_mappingLock);
    result = m_mappingPool.destroyID(index);
  }
  if(!result)
    return false;

  // the handle must not be used again
  handle = {};

  return true;
}

void AsyncUploader::markRangeForFlush(uint32_t blockIndex, VkDeviceSize offset, VkDeviceSize size)
{
  if(m_info.forceCoherentMapping || size == 0)
  {
    return;
  }

  assert(blockIndex < MAX_FLUSH_BLOCKS && "AsyncUploader supports at most 64 blocks");


  if(offset + size > m_flushRangeGranularity * RANGES_PER_BLOCK)
  {
    m_batchFlushRangeMasks[blockIndex] = ALL_RANGE_BITS;
    return;
  }

  const VkDeviceSize rangeSize = m_flushRangeGranularity;
  const uint32_t     startBit  = static_cast<uint32_t>(offset / rangeSize);
  const uint32_t     endBit    = static_cast<uint32_t>((offset + size - 1) / rangeSize);

  assert(endBit < RANGES_PER_BLOCK);
  m_batchFlushRangeMasks[blockIndex] |= setInclusiveBitRange(startBit, endBit);
}

void AsyncUploader::markForFlush(const nvvk::BufferSubAllocation& subAllocation)
{
  if(m_info.forceCoherentMapping || !subAllocation)
  {
    return;
  }

  markRangeForFlush(subAllocation.getBlockIndex(), subAllocation.getOffset(m_bufferSubAllocator.getOffsetUnitSize()),
                    subAllocation.getSize());
}

void AsyncUploader::flushBatchBlocks()
{
  for(uint32_t blockIndex = 0; blockIndex < MAX_FLUSH_BLOCKS; blockIndex++)
  {
    const uint64_t rangeMask = m_batchFlushRangeMasks[blockIndex];
    if(rangeMask == 0)
    {
      continue;
    }

    const nvvk::Buffer& blockBuffer = m_bufferSubAllocator.getBlockBuffer(static_cast<uint16_t>(blockIndex));

    if(rangeMask == ALL_RANGE_BITS)
    {
      m_info.allocator->autoFlushBuffer(blockBuffer);
      continue;
    }

    // must match the granularity that `markRangeForFlush` used to set the bits
    const VkDeviceSize rangeSize = m_flushRangeGranularity;

    for(uint32_t bit = 0; bit < RANGES_PER_BLOCK; bit++)
    {
      if((rangeMask & (1ull << bit)) == 0)
      {
        continue;
      }

      const uint32_t startBit = bit;
      while(bit + 1 < RANGES_PER_BLOCK && (rangeMask & (1ull << (bit + 1))))
      {
        bit++;
      }

      const VkDeviceSize flushOffset = VkDeviceSize(startBit) * rangeSize;
      const VkDeviceSize flushSize   = VkDeviceSize(bit - startBit + 1) * rangeSize;
      m_info.allocator->autoFlushBuffer(blockBuffer, flushOffset, flushSize);
    }
  }

  m_batchFlushRangeMasks = {};
}

VkResult AsyncUploader::acquireStagingSpace(nvvk::BufferSubAllocation& subAllocation, nvvk::BufferRange& stagingSpace, size_t dataSize)
{
  if(dataSize > m_bufferSubAllocator.getMaxAllocationSize())
  {
    // we don't chunk uploads, and BufferSubAllocation tracks the size in 32-bit,
    // so bail out rather than let the size truncate silently
    assert(0 && "AsyncUploader: upload is bigger than the maximum staging allocation");
    return VK_ERROR_OUT_OF_DEVICE_MEMORY;
  }

  VkResult result = m_bufferSubAllocator.subAllocate(subAllocation, dataSize);
  if(result == VK_SUCCESS)
  {
    if(subAllocation.getBlockIndex() >= MAX_FLUSH_BLOCKS)
    {
      assert(0 && "AsyncUploader supports at most 64 blocks");
      m_bufferSubAllocator.subFree(subAllocation);
      return VK_ERROR_OUT_OF_POOL_MEMORY;
    }

    stagingSpace = m_bufferSubAllocator.subRange(subAllocation);
    m_allocationCount++;
  }

  return result;
}

void AsyncUploader::freeStagingAllocation(nvvk::BufferSubAllocation& subAllocation)
{
  m_bufferSubAllocator.subFree(subAllocation);
  m_allocationCount--;
}

void AsyncUploader::addStagingAllocation(const nvvk::BufferSubAllocation& subAllocation)
{
  StagingAllocation stagingAllocation = {};
  stagingAllocation.semaphoreState    = m_transferSubmitState;
  stagingAllocation.subAllocation     = subAllocation;
  m_stagingAllocations.push_back(stagingAllocation);
}

VkResult AsyncUploader::acquireMapping(size_t dataSize, nvvk::BufferRange& mappingSpace, MappingHandle& mappingHandle)
{
  mappingHandle = {};

  if(dataSize == 0)
  {
    mappingSpace = {};
    return VK_SUCCESS;
  }
  uint32_t mappingIndex;
  {
    std::lock_guard lock(m_mappingLock);
    if(!m_mappingPool.createID(mappingIndex))
      return VK_ERROR_UNKNOWN;
  }

  VkResult result;
  {
    std::lock_guard lock(m_allocLock);
    result = acquireStagingSpace(m_mappingAllocations[mappingIndex], mappingSpace, dataSize);
  }

  if(result == VK_SUCCESS)
  {
    MappingHandleDetail detail{};
    detail.valid        = 1;
    detail.mappingIndex = mappingIndex;
    detail.blockIndex   = m_mappingAllocations[mappingIndex].getBlockIndex();
    mappingHandle.value = detail.handle;
    return VK_SUCCESS;
  }

  {
    std::lock_guard lock(m_mappingLock);
    m_mappingPool.destroyID(mappingIndex);
  }
  return result;
}

void AsyncUploader::commitMapping(MappingHandle& mappingHandle)
{
  if(!mappingHandle)
  {
    return;
  }
  nvvk::BufferSubAllocation subAllocation;
  if(!commitMappingHandle(mappingHandle, subAllocation))
  {
    assert(0 && "could not commit mapping handle");
    return;
  }

  std::lock_guard lock(m_appendLock);
  addStagingAllocation(subAllocation);
}

void AsyncUploader::releaseMapping(MappingHandle& mappingHandle)
{
  if(!mappingHandle)
  {
    return;
  }

  MappingHandleDetail detail{};
  detail.handle               = mappingHandle.value;
  const uint32_t mappingIndex = detail.mappingIndex;
  assert(detail.valid != 0 && !detail.used);

  {
    std::lock_guard lock(m_allocLock);

    freeStagingAllocation(m_mappingAllocations[mappingIndex]);
    m_mappingAllocations[mappingIndex] = {};
  }
  {
    std::lock_guard lock(m_mappingLock);

    m_mappingPool.destroyID(mappingIndex);
  }

  // the handle must not be used again
  mappingHandle = {};
}

VkResult AsyncUploader::appendBufferRange(const nvvk::BufferRange& buffer, const void* data, nvvk::SemaphoreState* outSemaphoreState)
{
  if(buffer.range == 0)
  {
    if(outSemaphoreState)
      *outSemaphoreState = {};
    return VK_SUCCESS;
  }

  assert(buffer.buffer);
  assert(data);

  nvvk::BufferSubAllocation subAllocation;
  nvvk::BufferRange         stagingSpace;
  {
    std::lock_guard lock(m_allocLock);
    NVVK_FAIL_RETURN(acquireStagingSpace(subAllocation, stagingSpace, buffer.range));
  }

  memcpy(stagingSpace.mapping, data, buffer.range);

  {
    std::lock_guard lock(m_appendLock);

    const VkResult result = NVVK_FAIL_REPORT(submitBatchChecked(buffer.range));
    if(result != VK_SUCCESS)
    {
      // nothing references this staging space yet, so it is safe to hand it back
      std::lock_guard allocLock(m_allocLock);

      freeStagingAllocation(subAllocation);
      return result;
    }

    m_batch.addBufferCopy(stagingSpace.buffer, stagingSpace.offset, buffer.buffer, buffer.offset, buffer.range, true);
    addStagingAllocation(subAllocation);

    markForFlush(subAllocation);

    getOutSemaphoreState(outSemaphoreState);
  }

  return VK_SUCCESS;
}

VkResult AsyncUploader::appendBufferRangeMappings(size_t                    rangeCount,
                                                  const MappedBufferRanges* ranges,
                                                  MappingHandle&            mappingHandle,
                                                  bool                      commitMapping_,
                                                  nvvk::SemaphoreState*     outSemaphoreState)
{
  nvvk::BufferSubAllocation subAllocation{};

  assert(mappingHandle);
  if(!mappingHandle)
  {
    return VK_ERROR_UNKNOWN;
  }

  // read the details before a commit invalidates the handle
  MappingHandleDetail detail{};
  detail.handle = mappingHandle.value;

  if(commitMapping_)
  {
    if(!commitMappingHandle(mappingHandle, subAllocation))
    {
      return VK_ERROR_UNKNOWN;
    }
  }
  else
  {
    // tag as used
    detail.used         = 1;
    mappingHandle.value = detail.handle;
  }

  size_t totalSize = 0;
  for(size_t i = 0; i < rangeCount; i++)
  {
    const MappedBufferRanges& range = ranges[i];
    totalSize += range.bufferRange.range;
  }

  {
    std::lock_guard lock(m_appendLock);

    if(totalSize > 0)
    {
      const VkResult result = NVVK_FAIL_REPORT(submitBatchChecked(totalSize));
      if(result != VK_SUCCESS)
      {
        // the mapping may already be referenced by earlier appends, so it must not be
        // freed here, only deferred to the current batch
        if(commitMapping_)
        {
          addStagingAllocation(subAllocation);
        }
        return result;
      }
    }

    const uint32_t mappingBlockIndex = commitMapping_ ? subAllocation.getBlockIndex() : detail.blockIndex;

    for(size_t i = 0; i < rangeCount; i++)
    {
      const MappedBufferRanges& range = ranges[i];
      if(range.bufferRange.range == 0)
        continue;

      assert(range.bufferRange.buffer);
      assert(range.mappingRange.range == range.bufferRange.range);

      markRangeForFlush(mappingBlockIndex, range.mappingRange.offset, range.mappingRange.range);

      m_batch.addBufferCopy(range.mappingRange.buffer, range.mappingRange.offset, range.bufferRange.buffer,
                            range.bufferRange.offset, range.bufferRange.range);
    }

    if(commitMapping_)
    {
      // Always defer the release to the current batch, even if this call appended nothing.
      // The mapping may have been used by earlier append calls whose copies are still
      // pending in m_batch, so freeing it here could hand the staging space to someone
      // else before the GPU has read it.
      addStagingAllocation(subAllocation);
    }

    getOutSemaphoreState(outSemaphoreState);
  }

  return VK_SUCCESS;
}

VkResult AsyncUploader::appendImageSub(nvvk::Image&                    image,
                                       const VkOffset3D&               offset,
                                       const VkExtent3D&               extent,
                                       const VkImageSubresourceLayers& subresource,
                                       size_t                          dataSize,
                                       const void*                     data,
                                       VkImageLayout                   newLayout,
                                       nvvk::SemaphoreState*           outSemaphoreState)
{
  if(dataSize == 0)
  {
    if(outSemaphoreState)
      *outSemaphoreState = {};
    return VK_SUCCESS;
  }

  assert(image.image);
  assert(data);

  nvvk::BufferSubAllocation subAllocation;
  nvvk::BufferRange         stagingSpace;
  {
    std::lock_guard lock(m_allocLock);
    NVVK_FAIL_RETURN(acquireStagingSpace(subAllocation, stagingSpace, dataSize));
  }

  memcpy(stagingSpace.mapping, data, dataSize);

  const VkImageSubresourceRange subresourceRange{subresource.aspectMask, subresource.mipLevel, 1,
                                                 subresource.baseArrayLayer, subresource.layerCount};

  {
    std::lock_guard lock(m_appendLock);

    const VkResult result = NVVK_FAIL_REPORT(submitBatchChecked(dataSize));
    if(result != VK_SUCCESS)
    {
      // nothing references this staging space yet, so it is safe to hand it back
      std::lock_guard allocLock(m_allocLock);

      freeStagingAllocation(subAllocation);
      return result;
    }

    m_batch.addImageCopy(stagingSpace.buffer, stagingSpace.offset, image.image, image.descriptor.imageLayout, newLayout,
                         dataSize, subresource, offset, extent, &subresourceRange);
    addStagingAllocation(subAllocation);
    markForFlush(subAllocation);

    getOutSemaphoreState(outSemaphoreState);
  }

  return VK_SUCCESS;
}

VkResult AsyncUploader::appendImageSubMappings(nvvk::Image&           image,
                                               size_t                 imageSubCount,
                                               const MappedImageSubs* imageSubs,
                                               MappingHandle&         mappingHandle,
                                               bool                   commitMapping_,
                                               VkImageLayout          newLayout,
                                               nvvk::SemaphoreState*  outSemaphoreState)
{
  assert(image.image);
  assert(mappingHandle);
  if(!mappingHandle)
  {
    return VK_ERROR_UNKNOWN;
  }

  nvvk::BufferSubAllocation subAllocation{};

  // read the details before a commit invalidates the handle
  MappingHandleDetail detail{};
  detail.handle = mappingHandle.value;

  if(commitMapping_)
  {
    if(!commitMappingHandle(mappingHandle, subAllocation))
    {
      return VK_ERROR_UNKNOWN;
    }
  }
  else
  {
    // tag as used
    detail.used         = 1;
    mappingHandle.value = detail.handle;
  }

  size_t totalSize = 0;
  for(size_t i = 0; i < imageSubCount; i++)
  {
    const MappedImageSubs& sub = imageSubs[i];
    totalSize += sub.mappingSpace.range;
  }

  {
    std::lock_guard lock(m_appendLock);

    const VkImageLayout imageLayout = image.descriptor.imageLayout;

    if(totalSize > 0)
    {
      const VkResult result = NVVK_FAIL_REPORT(submitBatchChecked(totalSize));
      if(result != VK_SUCCESS)
      {
        // the mapping may already be referenced by earlier appends, so it must not be
        // freed here, only deferred to the current batch
        if(commitMapping_)
        {
          addStagingAllocation(subAllocation);
        }
        return result;
      }
    }

    const uint32_t mappingBlockIndex = commitMapping_ ? subAllocation.getBlockIndex() : detail.blockIndex;

    for(size_t i = 0; i < imageSubCount; i++)
    {
      const MappedImageSubs& sub = imageSubs[i];
      if(sub.mappingSpace.range == 0)
        continue;

      markRangeForFlush(mappingBlockIndex, sub.mappingSpace.offset, sub.mappingSpace.range);

      // reset imageLayout with each subImage
      image.descriptor.imageLayout = imageLayout;

      const VkImageSubresourceRange subresourceRange{sub.subresource.aspectMask, sub.subresource.mipLevel, 1,
                                                     sub.subresource.baseArrayLayer, sub.subresource.layerCount};

      m_batch.addImageCopy(sub.mappingSpace.buffer, sub.mappingSpace.offset, image.image, image.descriptor.imageLayout,
                           newLayout, sub.mappingSpace.range, sub.subresource, sub.offset, sub.extent, &subresourceRange);
    }

    if(commitMapping_)
    {
      // Always defer the release to the current batch, even if this call appended nothing.
      // The mapping may have been used by earlier append calls whose copies are still
      // pending in m_batch, so freeing it here could hand the staging space to someone
      // else before the GPU has read it.
      addStagingAllocation(subAllocation);
    }

    getOutSemaphoreState(outSemaphoreState);
  }

  return VK_SUCCESS;
}

nvvk::BufferSubAllocator::Report AsyncUploader::getAllocationReport() const
{
  std::lock_guard lock(m_allocLock);

  return m_bufferSubAllocator.getReport();
}

bool AsyncUploader::hasOwnershipBarriers() const
{
  std::lock_guard lock(m_targetLock);

  return !m_targetBarriers.empty();
}

VkResult AsyncUploader::flushPending(nvvk::SemaphoreState* outSemaphoreState)
{
  std::lock_guard lock(m_appendLock);

  if(!m_batch.isAppendedEmpty())
  {
    // Capture the state of the batch we are about to submit; a successful submitBatch()
    // advances m_transferSubmitState to the next batch. On failure the timeline value is
    // never signaled, so we must not hand it out - the caller would wait on it forever.
    const nvvk::SemaphoreState submittedState = m_transferSubmitState;

    NVVK_FAIL_RETURN(submitBatch());

    if(outSemaphoreState)
    {
      *outSemaphoreState = submittedState;
    }
    return VK_SUCCESS;
  }
  else
  {
    if(outSemaphoreState)
    {
      *outSemaphoreState = getLastSubmitState();
    }
    releaseStagingAllocations(false);
    return VK_SUCCESS;
  }
}

VkResult AsyncUploader::waitForCompletion()
{
  if(m_transferTimelineValue == 1)
    return VK_SUCCESS;

  // current value is always for next submit, hence -1
  nvvk::SemaphoreState sem = nvvk::SemaphoreState::makeFixed(m_transferTimelineSemaphore, m_transferTimelineValue - 1);
  return sem.wait(m_info.allocator->getDevice(), ~0ull);
}

void AsyncUploader::releaseCompletedAllocations()
{
  std::lock_guard lock(m_appendLock);
  releaseStagingAllocations(false);
}

nvvk::SemaphoreState AsyncUploader::getSubmitState()
{
  std::lock_guard lock(m_appendLock);

  if(!m_batch.isAppendedEmpty())
  {
    return m_transferSubmitState;
  }

  // invalid when nothing was submitted yet, there is nothing to wait for then
  return getLastSubmitState();
}

void AsyncUploader::cmdDrainOwnershipBarriers(VkCommandBuffer cmd)
{
  std::lock_guard lock(m_targetLock);

  if(m_targetBarriers.empty())
    return;

  VkDevice device = m_info.allocator->getDevice();

  nvvk::SemaphoreStateSignalCache semaphoreCache;

  size_t writeIndex = 0;

  for(size_t readIndex = 0; readIndex < m_targetBarriers.size(); readIndex++)
  {
    TargetBarriers& barriers = m_targetBarriers[readIndex];

    if(!barriers.semaphoreState.isValid() || (semaphoreCache.testSignaled(device, barriers.semaphoreState)))
    {
      barriers.ownerAcquisitionBarriers.cmdPipelineBarrier(cmd, 0);
      barriers.ownerAcquisitionBarriers.clear();
    }
    else if(readIndex != writeIndex)
    {
      m_targetBarriers[writeIndex++] = std::move(barriers);
    }
    else
    {
      writeIndex++;
    }
  }
  m_targetBarriers.resize(writeIndex);
}


//--------------------------------------------------------------------------------------------------
// Usage example
//--------------------------------------------------------------------------------------------------
[[maybe_unused]] static void usage_AsyncUploader()
{

  //////////////////////////////////////////////////////////////////////////
  // asynchronous example
  {
    // AsyncUploader submits copy batches on a dedicated transfer queue.
    // Before using uploaded resources on the graphics queue, call cmdDrainOwnershipBarriers().

    // We show

    VkDevice                device{};
    nvvk::ResourceAllocator resourceAllocator;  // EX. initialize somehow
    nvvk::QueueInfo         transferQueue{};    // EX. dedicated transfer queue with exclusive submit access
    nvvk::QueueInfo         graphicsQueue{};    // EX. queue where uploaded resources are consumed
    nvvk::Buffer            deviceBuffer{};     // EX. create GPU buffer to upload into
    nvvk::Image             deviceImage{};      // EX. create GPU image to upload into
    nvvk::SemaphoreState    uploadSemState{};

    AsyncUploader asyncUploader;
    asyncUploader.init({
        .allocator     = &resourceAllocator,
        .transferQueue = transferQueue,
        .targetQueue   = graphicsQueue,
        .debugName     = "asyncUploads",
    });

    // WORKER THREAD SCOPE
    {
      // This scope shows typical per-thread work.
      // The threads only produce the data that is uploaded and track
      // the completion state somewhere.


      uint8_t uploadData[4096]{};
      size_t  uploadSize = sizeof(uploadData);

      // direct buffer upload
      {
        // we pass in a semaphore state that we can later query if the upload completed.
        asyncUploader.appendBuffer(deviceBuffer, 0, uploadSize, uploadData, &uploadSemState);

        // The appends are thread-safe, and while the returned semaphore state can be different
        // for each resource, they are monotonic. Meaning on the same thread it's enough to
        // track the last semaphore state of several uploads to have the guarantee that all
        // previous are covered with it.
      }


      // direct image upload
      {
        VkOffset3D               offset{};
        VkExtent3D               extent{64, 64, 1};
        VkImageSubresourceLayers subresource{VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        asyncUploader.appendImageSub(deviceImage, offset, extent, subresource, uploadSize, uploadData,
                                     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
      }

      // mapping upload: acquire staging, fill on CPU, append, then flush batch to transfer queue
      {
        nvvk::BufferRange                mappingSpace{};
        UploaderInterface::MappingHandle mappingHandle{};
        if(asyncUploader.acquireMapping(uploadSize, mappingSpace, mappingHandle) == VK_SUCCESS)
        {
          memcpy(mappingSpace.mapping, uploadData, uploadSize);

          UploaderInterface::MappedBufferRanges mappedRange{};
          mappedRange.bufferRange.buffer = deviceBuffer.buffer;
          mappedRange.bufferRange.offset = 0;
          mappedRange.bufferRange.range  = uploadSize;
          mappedRange.mappingRange       = mappingSpace;

          asyncUploader.appendBufferRangeMappings(1, &mappedRange, mappingHandle, true);
        }
      }

      // cancel path: release the mapping if it was acquired but never appended
      {
        // EX. CPU-side work that fills the mapping and can fail
        auto produceUploadData = [&](const nvvk::BufferRange& space) { return space.mapping != nullptr; };

        nvvk::BufferRange                mappingSpace{};
        UploaderInterface::MappingHandle mappingHandle{};
        // a valid handle, not `VK_SUCCESS`, tells us that staging space was acquired
        asyncUploader.acquireMapping(uploadSize, mappingSpace, mappingHandle);

        if(mappingHandle && !produceUploadData(mappingSpace))
        {
          asyncUploader.releaseMapping(mappingHandle);
        }
      }


      // Manually trigger a flush.
      // Can be useful when we have a thread that is doing a lot of uploads and we
      // want to enforce a submit at the end of its run.
      nvvk::SemaphoreState flushSem{};
      asyncUploader.flushPending(&flushSem);
    }

    // MAIN THREAD SCOPE
    while(true)
    {
      // this scope reflects the main rendering loop.

      VkCommandBuffer cmd{};  // per-frame graphics command buffer

      // if you are not flushing at the end of major upload sections in your code then do it per-frame
      asyncUploader.flushPending();
      // also release completed allocations
      asyncUploader.releaseCompletedAllocations();

      // apply ownership barriers before using async-uploaded resources
      asyncUploader.cmdDrainOwnershipBarriers(cmd);

      // check if upload completed
      if(uploadSemState.testSignaled(device))
      {
        // now it's safe to use this buffer
        vkCmdBindIndexBuffer(cmd, deviceBuffer.buffer, 0, VK_INDEX_TYPE_UINT16);
      }
    }

    // call asyncUploader.waitForCompletion() and asyncUploader.deinit() on shutdown
  }

  //////////////////////////////////////////////////////////////////////////
  // synchronous example
  {
    // In this example we use the uploader in a synchronous way on the
    // main queue.

    nvvk::ResourceAllocator resourceAllocator;  // EX. initialize somehow
    nvvk::QueueInfo         graphicsQueue{};    // EX. queue where uploaded resources are consumed
    nvvk::Buffer            deviceBuffer{};     // EX. create GPU buffer to upload into

    // initialize once
    AsyncUploader syncUploader;
    syncUploader.init({
        .allocator = &resourceAllocator,
        // we use the same queue for both, this will
        // remove need for ownership barriers
        .transferQueue = graphicsQueue,
        .targetQueue   = graphicsQueue,
        .debugName     = "asyncUploads",
    });


    // whenever doing uploads use the interface functions

    uint8_t uploadData[4096]{};
    size_t  uploadSize = sizeof(uploadData);

    syncUploader.appendBuffer(deviceBuffer, 0, uploadSize, uploadData);

    // At the end call these functions.
    // There is no need for ownership transfer barriers,
    // as we only used one queue.
    syncUploader.flushAndWait();
  }
}

}  // namespace nvvk
