/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "async_uploader.hpp"

#include <cassert>
#include <cstring>

#include <nvutils/logger.hpp>

#include "check_error.hpp"
#include "debug_util.hpp"

namespace nvdx12 {

AsyncUploader::~AsyncUploader()
{
  assert(m_info.allocator == nullptr && "Missing deinit()");
}

HRESULT AsyncUploader::init(const InitInfo& info)
{
  assert(m_info.allocator == nullptr && "Missing deinit()");
  assert(info.allocator);
  assert(info.transferQueue.queue && info.targetQueue.queue);

  HRESULT           result;
  ID3D12Device*     device = info.allocator->getDevice();
  const std::string name   = info.debugName.empty() ? std::string("nvdx12::AsyncUploader") : info.debugName;

  // The uploader's own fence on the transfer queue: its values are the batches.
  result = NVDX12_FAIL_REPORT(m_transferTimeline.init(device, info.transferQueue, name.c_str()));
  if(FAILED(result))
  {
    return result;
  }

  result = NVDX12_FAIL_REPORT(m_transferCmdPool.init(device, info.transferQueue.familyIndex, ManagedCommandPools::Mode::SEMAPHORE_STATE,
                                                     D3D12_COMMAND_LIST_FLAG_NONE, info.commandPoolSize));
  if(FAILED(result))
  {
    m_transferTimeline.deinit();
    return result;
  }

  BufferSubAllocator::InitInfo subInitInfo;
  subInitInfo.resourceAllocator     = info.allocator;
  subInitInfo.debugName             = info.debugName;
  subInitInfo.usageFlags            = D3D12_RESOURCE_FLAG_NONE;  // copy source only, no descriptor
  subInitInfo.memoryUsage           = MemoryUsage::eHostUpload;
  subInitInfo.blockSize             = info.blockSize;
  subInitInfo.maxAllocatedSize      = info.blockSize * MAX_BLOCKS;
  subInitInfo.perBlockAllocations   = info.maxOperations;
  subInitInfo.keepBlockCount        = info.keepBlockCount;
  subInitInfo.threadSafeBlockAccess = true;

  result = NVDX12_FAIL_REPORT(m_bufferSubAllocator.init(subInitInfo));
  if(FAILED(result))
  {
    m_transferCmdPool.deinit();
    m_transferTimeline.deinit();
    return result;
  }

  // Same DIRECT / COMPUTE queue: layout barriers are recorded with the copies.
  // Another queue, or a COPY queue (which only knows COMMON): the images are
  // copied in COMMON and the target queue records the transition, see
  // cmdDrainOwnershipBarriers. This is where nvvk's ownership transfer goes.
  m_batch.enableLayoutBarriers = true;
  m_batch.enableQueueTransfer  = info.transferQueue.queue != info.targetQueue.queue
                                || info.transferQueue.familyIndex == D3D12_COMMAND_LIST_TYPE_COPY;

  m_transferSubmitState = m_transferTimeline.createNextFixedSemaphoreState();

  m_mappingAllocations.resize(info.maxOperations);
  m_mappingPool.init(info.maxOperations);

  m_info = info;

  return S_OK;
}

void AsyncUploader::deinit()
{
  if(m_info.allocator == nullptr)
    return;

  m_transferTimeline.waitIdle();

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
  m_transferCmdPool.deinit();
  m_transferTimeline.deinit();
  m_bufferSubAllocator.deinit();
  m_info               = {};
  m_mappingAllocations = {};
  m_mappingPool.deinit();
  m_transferSubmitState = {};
  m_submitCount         = 0;
  m_waitCmdCount        = 0;
  m_targetBarriers.clear();
  m_allocationCount = 0;
}

HRESULT AsyncUploader::submitBatch()
{
  ID3D12GraphicsCommandList7* cmd = nullptr;
  HRESULT                     hr  = m_transferCmdPool.acquireCommandBuffer(m_transferSubmitState, cmd);
  if(hr != S_OK)
  {
    return FAILED(hr) ? hr : E_FAIL;
  }
  setDebugName(cmd, "AsyncUploader:cmd:" + std::to_string(m_transferTimeline.lastSignaledValue() + 1));

  releaseStagingAllocations(false);

  void* payload = nullptr;
  if(m_info.onSubmitCallback)
  {
    payload = m_info.onSubmitCallback(cmd, m_submitCount, m_transferSubmitState, true, nullptr);
  }
  // m_batch keeps the copies until the list closed, so a failed Close loses
  // nothing and the state handed out for them (m_transferSubmitState) is only
  // signaled once they are submitted.
  m_batch.cmdRecordAppended(cmd);
  if(m_info.onSubmitCallback)
  {
    m_info.onSubmitCallback(cmd, m_submitCount, m_transferSubmitState, false, payload);
  }
  hr = cmd->Close();
  if(FAILED(hr))
  {
    NVDX12_FAIL_REPORT(hr);
    m_transferCmdPool.releaseCommandBuffer(cmd);
    return hr;
  }
  // The layout transitions for the target queue are published to
  // m_targetBarriers only after Close succeeded, so a failed Close leaves no
  // record whose fence value would never be signaled.
  TargetBarriers record{
      .ownerAcquisitionBarriers = std::move(m_batch.acquire),
      .semaphoreState           = m_transferSubmitState,
  };
  m_batch.reset();
  m_waitCmdCount = m_transferCmdPool.getWaitCount();
  m_submitCount++;

  // submit to queue; the fixed state holds the value this submit signals.
  // A rejected signal (0) still executed the copies, and the queue's next
  // signal takes that value, so the record and the next state stay valid.
  const uint64_t submittedValue = m_transferTimeline.submit(cmd, m_transferSubmitState);

  // retain the transitions for cmdDrainOwnershipBarriers()
  if(!record.ownerAcquisitionBarriers.imageBarriers.empty())
  {
    std::lock_guard lock(m_targetLock);
    m_targetBarriers.push_back(std::move(record));
  }

  m_transferSubmitState = m_transferTimeline.createNextFixedSemaphoreState();

  return submittedValue != 0 ? S_OK : E_FAIL;
}

void AsyncUploader::releaseStagingAllocations(bool forceAll)
{
  SemaphoreStateSignalCache semaphoreCache;

  size_t writeIndex = 0;

  for(size_t readIndex = 0; readIndex < m_stagingAllocations.size(); readIndex++)
  {
    StagingAllocation& allocation = m_stagingAllocations[readIndex];

    if(forceAll || !allocation.semaphoreState.isValid() || (semaphoreCache.testSignaled(allocation.semaphoreState)))
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

HRESULT AsyncUploader::submitBatchChecked(size_t addedSize)
{
  if(m_batch.checkAppendedSize(m_info.flushSize, addedSize))
  {
    return submitBatch();
  }
  return S_OK;
}

AsyncUploader::PendingTransition AsyncUploader::reclaimPendingTransition(Image& image)
{
  PendingTransition pending;
  if(!m_batch.enableQueueTransfer || !image.image)
  {
    return pending;
  }
  std::lock_guard lock(m_targetLock);
  for(size_t i = 0; i < m_targetBarriers.size(); i++)
  {
    TargetBarriers&                     record   = m_targetBarriers[i];
    std::vector<D3D12_TEXTURE_BARRIER>& barriers = record.ownerAcquisitionBarriers.imageBarriers;
    for(auto it = barriers.begin(); it != barriers.end(); ++it)
    {
      if(it->pResource == image.image.Get())
      {
        // The earlier batch copied the image in COMMON and nothing transitioned it since.
        assert(it->LayoutBefore == D3D12_BARRIER_LAYOUT_COMMON);
        pending.barrier        = *it;
        pending.semaphoreState = record.semaphoreState;
        pending.stageMask      = image.stageMask;
        pending.accessMask     = image.accessMask;
        barriers.erase(it);
        image.layout     = D3D12_BARRIER_LAYOUT_COMMON;
        image.stageMask  = D3D12_BARRIER_SYNC_NONE;
        image.accessMask = D3D12_BARRIER_ACCESS_NO_ACCESS;
        return pending;
      }
    }
  }
  return pending;
}

void AsyncUploader::restorePendingTransition(Image& image, const PendingTransition& pending)
{
  if(!pending)
  {
    return;
  }
  // The record may have been drained (emptied) meanwhile, so the transition gets
  // a record of its own; it is drained once its batch is signaled.
  TargetBarriers record{.semaphoreState = pending.semaphoreState};
  record.ownerAcquisitionBarriers.imageBarriers.push_back(pending.barrier);
  {
    std::lock_guard lock(m_targetLock);
    m_targetBarriers.push_back(std::move(record));
  }
  image.layout     = pending.barrier.LayoutAfter;
  image.stageMask  = pending.stageMask;
  image.accessMask = pending.accessMask;
}

bool AsyncUploader::commitMappingHandle(MappingHandle& handle, BufferSubAllocation& subAllocation)
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

HRESULT AsyncUploader::acquireStagingSpace(BufferSubAllocation& subAllocation, BufferRange& stagingSpace, size_t dataSize, uint32_t alignment)
{
  if(dataSize > m_bufferSubAllocator.getMaxAllocationSize())
  {
    // we don't chunk uploads, and BufferSubAllocation tracks the size in 32-bit,
    // so bail out rather than let the size truncate silently
    assert(0 && "AsyncUploader: upload is bigger than the maximum staging allocation");
    return E_OUTOFMEMORY;
  }

  HRESULT result = m_bufferSubAllocator.subAllocate(subAllocation, dataSize, alignment);
  if(SUCCEEDED(result))
  {
    stagingSpace = m_bufferSubAllocator.subRange(subAllocation);
    m_allocationCount++;
  }

  return result;
}

void AsyncUploader::freeStagingAllocation(BufferSubAllocation& subAllocation)
{
  m_bufferSubAllocator.subFree(subAllocation);
  m_allocationCount--;
}

void AsyncUploader::addStagingAllocation(const BufferSubAllocation& subAllocation)
{
  StagingAllocation stagingAllocation = {};
  // With nothing pending, every copy that reads this staging was submitted
  // already (a mapping committed after its batch was flushed): the last submit
  // covers it, or, before any submit, nothing reads it. The pending batch's
  // state would otherwise hold it until some later append is flushed.
  stagingAllocation.semaphoreState = m_batch.isAppendedEmpty() ? getLastSubmitState() : m_transferSubmitState;
  stagingAllocation.subAllocation  = subAllocation;
  m_stagingAllocations.push_back(stagingAllocation);
}

HRESULT AsyncUploader::acquireMapping(size_t dataSize, BufferRange& mappingSpace, MappingHandle& mappingHandle)
{
  mappingHandle = {};

  if(dataSize == 0)
  {
    mappingSpace = {};
    return S_OK;
  }
  uint32_t mappingIndex;
  {
    std::lock_guard lock(m_mappingLock);
    if(!m_mappingPool.createID(mappingIndex))
      return E_OUTOFMEMORY;
  }

  HRESULT result;
  {
    std::lock_guard lock(m_allocLock);
    // placement alignment, so that the mapping can hold image footprints
    result = acquireStagingSpace(m_mappingAllocations[mappingIndex], mappingSpace, dataSize, IMAGE_ALIGNMENT);
  }

  if(SUCCEEDED(result))
  {
    MappingHandleDetail detail{};
    detail.valid        = 1;
    detail.mappingIndex = mappingIndex;
    mappingHandle.value = detail.handle;
    return S_OK;
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
  BufferSubAllocation subAllocation;
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

HRESULT AsyncUploader::appendBufferRange(const BufferRange& buffer, const void* data, SemaphoreState* outSemaphoreState)
{
  if(buffer.range == 0)
  {
    if(outSemaphoreState)
      *outSemaphoreState = {};
    return S_OK;
  }

  assert(buffer.buffer);
  assert(data);
  if(!buffer.buffer || !data)
  {
    return E_INVALIDARG;
  }

  BufferSubAllocation subAllocation;
  BufferRange         stagingSpace;
  {
    std::lock_guard lock(m_allocLock);
    NVDX12_FAIL_RETURN(acquireStagingSpace(subAllocation, stagingSpace, buffer.range, BUFFER_ALIGNMENT));
  }

  memcpy(stagingSpace.mapping, data, buffer.range);

  {
    std::lock_guard lock(m_appendLock);

    const HRESULT result = NVDX12_FAIL_REPORT(submitBatchChecked(buffer.range));
    if(FAILED(result))
    {
      // nothing references this staging space yet, so it is safe to hand it back
      std::lock_guard allocLock(m_allocLock);

      freeStagingAllocation(subAllocation);
      return result;
    }

    m_batch.addBufferCopy(stagingSpace.buffer, stagingSpace.offset, buffer.buffer, buffer.offset, buffer.range);
    addStagingAllocation(subAllocation);

    getOutSemaphoreState(outSemaphoreState);
  }

  return S_OK;
}

HRESULT AsyncUploader::appendBufferRangeMappings(size_t                    rangeCount,
                                                 const MappedBufferRanges* ranges,
                                                 MappingHandle&            mappingHandle,
                                                 bool                      commitMapping_,
                                                 SemaphoreState*           outSemaphoreState)
{
  BufferSubAllocation subAllocation{};

  assert(mappingHandle);
  if(!mappingHandle)
  {
    return E_INVALIDARG;
  }

  size_t totalSize = 0;
  for(size_t i = 0; i < rangeCount; i++)
  {
    const MappedBufferRanges& range = ranges[i];
    totalSize += range.bufferRange.range;
  }

  {
    std::lock_guard lock(m_appendLock);

    // a failed flush appends nothing and leaves the handle as it was
    if(totalSize > 0)
    {
      NVDX12_FAIL_RETURN(submitBatchChecked(totalSize));
    }

    if(commitMapping_)
    {
      if(!commitMappingHandle(mappingHandle, subAllocation))
      {
        return E_INVALIDARG;
      }
    }
    else
    {
      // tag as used
      MappingHandleDetail detail{};
      detail.handle       = mappingHandle.value;
      detail.used         = 1;
      mappingHandle.value = detail.handle;
    }

    for(size_t i = 0; i < rangeCount; i++)
    {
      const MappedBufferRanges& range = ranges[i];
      if(range.bufferRange.range == 0)
        continue;

      assert(range.bufferRange.buffer);
      assert(range.mappingRange.range == range.bufferRange.range);

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

  return S_OK;
}

HRESULT AsyncUploader::appendImageSub(Image&                        image,
                                      const Offset3D&               offset,
                                      const Extent3D&               extent,
                                      const ImageSubresourceLayers& subresource,
                                      size_t                        dataSize,
                                      const void*                   data,
                                      D3D12_BARRIER_LAYOUT          newLayout,
                                      SemaphoreState*               outSemaphoreState)
{
  if(dataSize == 0)
  {
    if(outSemaphoreState)
      *outSemaphoreState = {};
    return S_OK;
  }

  assert(image.image);
  assert(data);

  if(!image.image || !data)
  {
    LOGE("AsyncUploader::appendImageSub: null image or data\n");
    return E_INVALIDARG;
  }
  const ImageSubFootprint footprint = makeImageSubFootprint(image.format, extent, subresource.layerCount);
  if(!footprint)
  {
    LOGE("AsyncUploader::appendImageSub: unsupported format %d or empty region\n", int(image.format));
    return E_INVALIDARG;
  }
  if(dataSize < footprint.tightSize)
  {
    LOGE("AsyncUploader::appendImageSub: %zu bytes provided, %llu needed\n", dataSize,
         static_cast<unsigned long long>(footprint.tightSize));
    return E_INVALIDARG;
  }

  // the staging holds the region re-pitched into D3D12's placed footprint
  BufferSubAllocation subAllocation;
  BufferRange         stagingSpace;
  {
    std::lock_guard lock(m_allocLock);
    NVDX12_FAIL_RETURN(acquireStagingSpace(subAllocation, stagingSpace, footprint.size, IMAGE_ALIGNMENT));
  }

  writeImageSubFootprint(footprint, stagingSpace.mapping, data, subresource.layerCount);

  {
    std::lock_guard lock(m_appendLock);

    HRESULT result = NVDX12_FAIL_REPORT(submitBatchChecked(footprint.size));
    if(SUCCEEDED(result))
    {
      // UNDEFINED keeps the layout the image was going to, as nvvk keeps the tracked one
      const PendingTransition pending = reclaimPendingTransition(image);
      if(newLayout == D3D12_BARRIER_LAYOUT_UNDEFINED && pending)
      {
        newLayout = pending.barrier.LayoutAfter;
      }
      result = m_batch.addImageCopy(stagingSpace.buffer, stagingSpace.offset, image, newLayout, footprint.size,
                                    subresource, offset, extent);
      if(FAILED(result))
      {
        restorePendingTransition(image, pending);
      }
    }
    if(FAILED(result))
    {
      // nothing references this staging space yet, so it is safe to hand it back
      std::lock_guard allocLock(m_allocLock);

      freeStagingAllocation(subAllocation);
      return result;
    }

    addStagingAllocation(subAllocation);

    getOutSemaphoreState(outSemaphoreState);
  }

  return S_OK;
}

HRESULT AsyncUploader::appendImageSubMappings(Image&                 image,
                                              size_t                 imageSubCount,
                                              const MappedImageSubs* imageSubs,
                                              MappingHandle&         mappingHandle,
                                              bool                   commitMapping_,
                                              D3D12_BARRIER_LAYOUT   newLayout,
                                              SemaphoreState*        outSemaphoreState)
{
  assert(image.image);
  assert(mappingHandle);
  if(!mappingHandle)
  {
    return E_INVALIDARG;
  }

  BufferSubAllocation subAllocation{};

  size_t totalSize = 0;
  for(size_t i = 0; i < imageSubCount; i++)
  {
    const MappedImageSubs& sub = imageSubs[i];
    totalSize += sub.mappingSpace.range;
  }

  {
    std::lock_guard lock(m_appendLock);

    // a failed flush or validation appends nothing and leaves the handle as it was
    if(totalSize > 0)
    {
      NVDX12_FAIL_RETURN(submitBatchChecked(totalSize));
    }

    // StagingCopyBatch transitions the whole image once per batch, so every
    // sub-image shares the barrier and ends in the same layout.
    // UNDEFINED keeps the layout the image was going to, as nvvk keeps the tracked one
    const PendingTransition pending = reclaimPendingTransition(image);
    if(newLayout == D3D12_BARRIER_LAYOUT_UNDEFINED && pending)
    {
      newLayout = pending.barrier.LayoutAfter;
    }
    // validate every sub first (against the reclaimed layout), so that a failure appends nothing
    HRESULT result = m_batch.checkImageSubCopies(image, imageSubCount, imageSubs);
    if(SUCCEEDED(result) && commitMapping_ && !commitMappingHandle(mappingHandle, subAllocation))
    {
      result = E_INVALIDARG;
    }
    if(FAILED(result))
    {
      restorePendingTransition(image, pending);
      return result;
    }
    if(!commitMapping_)
    {
      // tag as used
      MappingHandleDetail detail{};
      detail.handle       = mappingHandle.value;
      detail.used         = 1;
      mappingHandle.value = detail.handle;
    }

    bool addedCopy = false;
    for(size_t i = 0; i < imageSubCount && SUCCEEDED(result); i++)
    {
      const MappedImageSubs& sub = imageSubs[i];
      if(sub.mappingSpace.range == 0)
        continue;

      result = m_batch.addImageCopy(sub.mappingSpace.buffer, sub.mappingSpace.offset, image, newLayout,
                                    sub.mappingSpace.range, sub.subresource, sub.offset, sub.extent);
      // checkImageSubCopies ran every check addImageCopy makes: no sub fails after another was appended.
      assert(SUCCEEDED(result) && "addImageCopy failed after checkImageSubCopies");
      addedCopy = addedCopy || SUCCEEDED(result);
    }
    if(!addedCopy)
    {
      // nothing of the image went into the batch: the reclaimed transition stays where it was
      restorePendingTransition(image, pending);
    }

    if(commitMapping_)
    {
      // Always defer the release to the current batch, even if this call appended nothing
      // or failed. The mapping may have been used by earlier append calls whose copies are
      // still pending in m_batch, so freeing it here could hand the staging space to someone
      // else before the GPU has read it.
      addStagingAllocation(subAllocation);
    }
    if(FAILED(result))
    {
      return result;
    }

    getOutSemaphoreState(outSemaphoreState);
  }

  return S_OK;
}

BufferSubAllocator::Report AsyncUploader::getAllocationReport() const
{
  std::lock_guard lock(m_allocLock);

  return m_bufferSubAllocator.getReport();
}

bool AsyncUploader::hasOwnershipBarriers() const
{
  std::lock_guard lock(m_targetLock);

  for(size_t i = 0; i < m_targetBarriers.size(); i++)
  {
    if(!m_targetBarriers[i].ownerAcquisitionBarriers.imageBarriers.empty())
    {
      return true;
    }
  }
  return false;
}

HRESULT AsyncUploader::flushPending(SemaphoreState* outSemaphoreState)
{
  std::lock_guard lock(m_appendLock);

  if(!m_batch.isAppendedEmpty())
  {
    // The state of the batch about to be submitted; submitBatch() moves m_transferSubmitState
    // to the next batch. On failure no signal of this batch completes it, so it is not handed out.
    const SemaphoreState submittedState = m_transferSubmitState;

    NVDX12_FAIL_RETURN(submitBatch());

    if(outSemaphoreState)
    {
      *outSemaphoreState = submittedState;
    }
    return S_OK;
  }
  else
  {
    if(outSemaphoreState)
    {
      *outSemaphoreState = getLastSubmitState();
    }
    releaseStagingAllocations(false);
    return S_OK;
  }
}

HRESULT AsyncUploader::waitForCompletion()
{
  SemaphoreState sem;
  {
    std::lock_guard lock(m_appendLock);
    sem = getLastSubmitState();
  }
  if(!sem.isValid())
    return S_OK;

  return sem.wait(~0ull);
}

void AsyncUploader::releaseCompletedAllocations()
{
  std::lock_guard lock(m_appendLock);
  releaseStagingAllocations(false);
}

SemaphoreState AsyncUploader::getSubmitState()
{
  std::lock_guard lock(m_appendLock);

  if(!m_batch.isAppendedEmpty())
  {
    return m_transferSubmitState;
  }

  // invalid when nothing was submitted yet, there is nothing to wait for then
  return getLastSubmitState();
}

void AsyncUploader::cmdDrainOwnershipBarriers(ID3D12GraphicsCommandList7* cmd)
{
  std::lock_guard lock(m_targetLock);

  if(m_targetBarriers.empty())
    return;

  SemaphoreStateSignalCache semaphoreCache;

  // gather the transitions of every completed batch into one Barrier call
  BarrierContainer drained;

  size_t writeIndex = 0;

  for(size_t readIndex = 0; readIndex < m_targetBarriers.size(); readIndex++)
  {
    TargetBarriers& barriers = m_targetBarriers[readIndex];

    if(!barriers.semaphoreState.isValid() || (semaphoreCache.testSignaled(barriers.semaphoreState)))
    {
      const std::vector<D3D12_TEXTURE_BARRIER>& imageBarriers = barriers.ownerAcquisitionBarriers.imageBarriers;
      drained.imageBarriers.insert(drained.imageBarriers.end(), imageBarriers.begin(), imageBarriers.end());
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

  if(!drained.imageBarriers.empty())
  {
    drained.cmdPipelineBarrier(cmd);
  }
}

}  // namespace nvdx12

//--------------------------------------------------------------------------------------------------
// Usage example
//--------------------------------------------------------------------------------------------------
[[maybe_unused]] static void usage_AsyncUploader()
{

  //////////////////////////////////////////////////////////////////////////
  // asynchronous example
  {
    // AsyncUploader submits copy batches on a dedicated transfer queue (here a COPY queue).
    // Before using uploaded images on the direct queue, call cmdDrainOwnershipBarriers().

    nvdx12::ResourceAllocator resourceAllocator;  // EX. initialize somehow
    nvdx12::QueueInfo         transferQueue{};    // EX. D3D12_COMMAND_LIST_TYPE_COPY queue with exclusive submit access
    nvdx12::QueueInfo         graphicsQueue{};    // EX. queue where uploaded resources are consumed
    nvdx12::Buffer            deviceBuffer{};     // EX. create GPU buffer to upload into
    nvdx12::Image             deviceImage{};  // EX. create a 64x64 RGBA8 GPU image to upload into (UNDEFINED or COMMON)
    nvdx12::SemaphoreState    uploadSemState{};

    nvdx12::AsyncUploader asyncUploader;
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

      uint8_t uploadData[64 * 64 * 4]{};
      size_t  uploadSize = 4096;

      // direct buffer upload
      {
        // we pass in a semaphore state that we can later query if the upload completed.
        asyncUploader.appendBuffer(deviceBuffer, 0, uploadSize, uploadData, &uploadSemState);

        // The appends are thread-safe, and while the returned semaphore state can be different
        // for each resource, they are monotonic. Meaning on the same thread it's enough to
        // track the last semaphore state of several uploads to have the guarantee that all
        // previous are covered with it.
      }

      // direct image upload: copied in COMMON on the copy queue, transitioned to
      // SHADER_RESOURCE on the graphics queue by cmdDrainOwnershipBarriers
      {
        nvdx12::Offset3D               offset{};
        nvdx12::Extent3D               extent{64, 64, 1};
        nvdx12::ImageSubresourceLayers subresource{.mipLevel = 0, .baseArrayLayer = 0, .layerCount = 1};
        asyncUploader.appendImageSub(deviceImage, offset, extent, subresource, sizeof(uploadData), uploadData,
                                     D3D12_BARRIER_LAYOUT_SHADER_RESOURCE);
      }

      // mapping upload: acquire staging, fill on CPU, append, then flush batch to transfer queue
      {
        nvdx12::BufferRange                      mappingSpace{};
        nvdx12::UploaderInterface::MappingHandle mappingHandle{};
        if(asyncUploader.acquireMapping(uploadSize, mappingSpace, mappingHandle) == S_OK)
        {
          memcpy(mappingSpace.mapping, uploadData, uploadSize);

          nvdx12::UploaderInterface::MappedBufferRanges mappedRange{};
          mappedRange.bufferRange  = nvdx12::makeBufferRange(deviceBuffer, 0, uploadSize);
          mappedRange.mappingRange = mappingSpace;

          asyncUploader.appendBufferRangeMappings(1, &mappedRange, mappingHandle, true);
        }
      }

      // cancel path: release the mapping if it was acquired but never appended
      {
        // EX. CPU-side work that fills the mapping and can fail
        auto produceUploadData = [&](const nvdx12::BufferRange& space) { return space.mapping != nullptr; };

        nvdx12::BufferRange                      mappingSpace{};
        nvdx12::UploaderInterface::MappingHandle mappingHandle{};
        // a valid handle, not `S_OK`, tells us that staging space was acquired
        asyncUploader.acquireMapping(uploadSize, mappingSpace, mappingHandle);

        if(mappingHandle && !produceUploadData(mappingSpace))
        {
          asyncUploader.releaseMapping(mappingHandle);
        }
      }

      // Manually trigger a flush.
      // Can be useful when we have a thread that is doing a lot of uploads and we
      // want to enforce a submit at the end of its run.
      nvdx12::SemaphoreState flushSem{};
      asyncUploader.flushPending(&flushSem);
    }

    // MAIN THREAD SCOPE
    while(true)
    {
      // this scope reflects the main rendering loop.

      ID3D12GraphicsCommandList7* cmd{};  // per-frame graphics command list

      // if you are not flushing at the end of major upload sections in your code then do it per-frame
      asyncUploader.flushPending();
      // also release completed allocations
      asyncUploader.releaseCompletedAllocations();

      // check if upload completed, before the drain records the transitions of the completed batches
      const bool uploaded = uploadSemState.testSignaled();

      // apply the image layout transitions before using async-uploaded images
      asyncUploader.cmdDrainOwnershipBarriers(cmd);

      if(uploaded)
      {
        // now it's safe to use this buffer
        D3D12_INDEX_BUFFER_VIEW view{deviceBuffer.gpuAddress, uint32_t(deviceBuffer.bufferSize), DXGI_FORMAT_R16_UINT};
        cmd->IASetIndexBuffer(&view);
      }

      // Alternatively the GPU waits instead of the host, for buffers:
      // graphicsQueue.queue->Wait(uploadSemState.getSemaphore(), uploadSemState.getTimelineValue());
    }

    // call asyncUploader.waitForCompletion() and asyncUploader.deinit() on shutdown
  }

  //////////////////////////////////////////////////////////////////////////
  // synchronous example
  {
    // In this example we use the uploader in a synchronous way on the
    // main queue.

    nvdx12::ResourceAllocator resourceAllocator;  // EX. initialize somehow
    nvdx12::QueueInfo         graphicsQueue{};    // EX. queue where uploaded resources are consumed
    nvdx12::Buffer            deviceBuffer{};     // EX. create GPU buffer to upload into

    // initialize once
    nvdx12::AsyncUploader syncUploader;
    syncUploader.init({
        .allocator = &resourceAllocator,
        // we use the same queue for both, the layout transitions are then
        // recorded with the copies
        .transferQueue = graphicsQueue,
        .targetQueue   = graphicsQueue,
        .debugName     = "asyncUploads",
    });

    // whenever doing uploads use the interface functions

    uint8_t uploadData[4096]{};
    size_t  uploadSize = sizeof(uploadData);

    syncUploader.appendBuffer(deviceBuffer, 0, uploadSize, uploadData);

    // At the end call these functions.
    // There are no transitions to drain, as we only used one queue.
    syncUploader.flushAndWait();
  }
}
