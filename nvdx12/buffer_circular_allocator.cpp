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


#include "buffer_circular_allocator.hpp"

#include <algorithm>
#include <cassert>
#include <typeinfo>

#include "check_error.hpp"
#include "debug_util.hpp"

namespace nvdx12 {

BufferCircularAllocator::~BufferCircularAllocator()
{
  assert(m_info.resourceAllocator == nullptr && "Missing deinit()");
}

BufferCircularAllocator::BufferCircularAllocator(BufferCircularAllocator&& other) noexcept
{
  // this is freshly default-constructed; swapping all members transfers other's
  // full state here and leaves other empty (deinit-safe: resourceAllocator == null).
  std::swap(m_info, other.m_info);
  std::swap(m_state, other.m_state);
  std::swap(m_blocks, other.m_blocks);
}

BufferCircularAllocator& BufferCircularAllocator::operator=(BufferCircularAllocator&& other) noexcept
{
  if(this != &other)
  {
    assert(m_info.resourceAllocator == nullptr && "Missing deinit()");

    std::swap(m_info, other.m_info);
    std::swap(m_state, other.m_state);
    std::swap(m_blocks, other.m_blocks);
  }

  return *this;
}

HRESULT BufferCircularAllocator::init(const InitInfo& info)
{
  assert(m_info.resourceAllocator == nullptr);

  assert(info.resourceAllocator != nullptr);
  assert(!info.threadSafeBlockAccess || info.maxAllocatedSize != 0);
  if(info.minAlignment < MIN_ALIGNMENT || (info.minAlignment & (info.minAlignment - 1)) != 0)
  {
    return E_INVALIDARG;
  }

  // the packed 16-byte-unit offset field bounds the largest representable block;
  // reject at runtime too, so release builds fail cleanly instead of corrupting.
  // Reservations round up to MIN_ALIGNMENT, so the block size must be a multiple of it.
  assert(info.blockSize <= BufferCircularAllocation::MAX_BLOCK_SIZE && "blockSize too large for packed offset field");
  if(info.blockSize == 0 || info.blockSize > BufferCircularAllocation::MAX_BLOCK_SIZE || info.blockSize % MIN_ALIGNMENT != 0)
  {
    return E_INVALIDARG;
  }
  const uint64_t maxAllocatedSize = info.maxAllocatedSize ? info.maxAllocatedSize : info.blockSize * MAX_TOTAL_BLOCKS;
  const uint64_t maxBlocks        = (maxAllocatedSize + info.blockSize - 1) / info.blockSize;
  if(maxBlocks > MAX_TOTAL_BLOCKS)
  {
    return E_INVALIDARG;
  }

  m_info                  = info;
  m_info.maxAllocatedSize = maxAllocatedSize;

  // clamp allocations and block size to what the packed size/offset fields can address
  uint64_t addressableMax =
      std::min<uint64_t>(BufferCircularAllocation::MAX_SIZE, info.resourceAllocator->getMaxMemoryAllocationSize());
  m_state.maxAllocationSize = std::min<uint64_t>(info.blockSize, addressableMax);

  m_state.maxBlocks = static_cast<uint32_t>(maxBlocks);

  // Pre-size the block array so it never reallocates. This keeps block indices
  // and buffer payloads stable, which is what makes `subRange` / `getBlockBuffer`
  // safe to call from other threads while allocations happen.
  if(m_info.threadSafeBlockAccess)
  {
    m_blocks.resize(m_state.maxBlocks);

    // chain every slot into the free-slot list
    m_state.freeSlotIndex = 0;
    for(uint32_t i = 0; i < m_state.maxBlocks; i++)
    {
      m_blocks[i].nextFree = (i + 1 < m_state.maxBlocks) ? (i + 1) : INVALID_BLOCK_INDEX;
    }
  }

  // Pre-allocate one block and keep it as the active read/write block.
  if(m_info.keepBlockCount)
  {
    uint32_t      blockIndex;
    const HRESULT result = NVDX12_FAIL_REPORT(acquireBlock(blockIndex));
    if(FAILED(result))
    {
      // back to the uninitialized state, as BufferSubAllocator::init
      m_info  = {};
      m_state = {};
      m_blocks.clear();
      return result;
    }
    m_state.readBlockIndex   = blockIndex;
    m_state.writeBlockIndex  = blockIndex;
    m_state.activeBlockCount = 1;
  }

  return S_OK;
}

void BufferCircularAllocator::deinit()
{
  if(!m_info.resourceAllocator)
    return;

  for(size_t i = 0; i < m_blocks.size(); i++)
  {
    m_info.resourceAllocator->destroyBuffer(m_blocks[i].buffer);
  }

  m_info  = {};
  m_state = {};
  m_blocks.clear();
  m_blocks.shrink_to_fit();
}

BufferCircularAllocator::Report BufferCircularAllocator::getReport() const
{
  Report report;
  report.requestedSize = m_state.allocatedSize;
  // every block in the active ring is committed and unavailable for other data
  report.reservedSize  = uint64_t(m_state.activeBlockCount) * m_info.blockSize;
  report.allocatedSize = uint64_t(m_state.blockBufferCount) * m_info.blockSize;
  report.freeSize      = report.allocatedSize - report.reservedSize;
  return report;
}

HRESULT BufferCircularAllocator::acquireBlock(uint32_t& blockIndex)
{
  if(m_state.reserveBlockIndex != INVALID_BLOCK_INDEX)
  {
    // reuse a reserved block (its buffer already exists)
    blockIndex                = m_state.reserveBlockIndex;
    m_state.reserveBlockIndex = m_blocks[blockIndex].nextFree;
    m_state.reserveCount--;
  }
  else
  {
    // we need a fresh buffer, check the block budget first
    if(m_state.blockBufferCount >= m_state.maxBlocks)
    {
      return E_OUTOFMEMORY;  // not reported: the caller decides
    }

    if(m_state.freeSlotIndex != INVALID_BLOCK_INDEX)
    {
      // reuse a recycled, buffer-less slot
      blockIndex            = m_state.freeSlotIndex;
      m_state.freeSlotIndex = m_blocks[blockIndex].nextFree;
    }
    else
    {
      // grow the pool (only reachable when !threadSafeBlockAccess)
      blockIndex = uint32_t(m_blocks.size());
      m_blocks.push_back({});
    }

    const HRESULT result = createNewBuffer(m_blocks[blockIndex].buffer, blockIndex);  // reports failures
    if(FAILED(result))
    {
      // hand the buffer-less slot back
      m_blocks[blockIndex]          = {};
      m_blocks[blockIndex].nextFree = m_state.freeSlotIndex;
      m_state.freeSlotIndex         = blockIndex;
      return result;
    }
    m_state.blockBufferCount++;
  }

  // reset the block to a clean state (buffer is left untouched)
  Block& block      = m_blocks[blockIndex];
  block.writeOffset = 0;
  block.readOffset  = 0;
  block.liveCount   = 0;

  block.nextActive = INVALID_BLOCK_INDEX;
  block.nextFree   = INVALID_BLOCK_INDEX;

  return S_OK;
}

HRESULT BufferCircularAllocator::appendWriteBlock()
{
  uint32_t      blockIndex;
  const HRESULT result = acquireBlock(blockIndex);
  if(FAILED(result))
  {
    return result;
  }

  // link the new block at the tail (write side) of the ordered list
  m_blocks[m_state.writeBlockIndex].nextActive = blockIndex;
  m_state.writeBlockIndex                      = blockIndex;
  m_state.activeBlockCount++;

  return S_OK;
}

void BufferCircularAllocator::recycleBlock(uint32_t blockIndex)
{
  if(m_state.reserveCount < m_info.keepBlockCount)
  {
    // keep the buffer around in reserve to avoid re-creation churn
    Block& block      = m_blocks[blockIndex];
    block.writeOffset = 0;
    block.readOffset  = 0;
    block.liveCount   = 0;

    block.nextActive          = INVALID_BLOCK_INDEX;
    block.nextFree            = m_state.reserveBlockIndex;
    m_state.reserveBlockIndex = blockIndex;
    m_state.reserveCount++;
  }
  else
  {
    // destroy the buffer, keep only the empty slot for later reuse
    m_info.resourceAllocator->destroyBuffer(m_blocks[blockIndex].buffer);
    m_state.blockBufferCount--;

    m_blocks[blockIndex]          = {};
    m_blocks[blockIndex].nextFree = m_state.freeSlotIndex;
    m_state.freeSlotIndex         = blockIndex;
  }
}

HRESULT BufferCircularAllocator::subAllocate(BufferCircularAllocation& subAllocation, uint64_t size, uint32_t alignment)
{
  subAllocation = {};

  assert((alignment & (alignment - 1)) == 0 && "alignment must be power-of-two");
  assert(size <= m_state.maxAllocationSize && "size must be <= blockSize");

  // minAlignment is the guaranteed floor; a higher per-call alignment is honored
  // by simply bumping the write cursor further.
  alignment = std::max(alignment, m_info.minAlignment);

  if(size == 0)
  {
    // a zero-sized request yields an invalid allocation
    return S_OK;
  }
  if(size > m_state.maxAllocationSize)
  {
    // release builds: a request larger than a block would be placed past the
    // block's end (the assert above catches it in debug builds)
    return E_INVALIDARG;
  }

  // Reserve the requested size rounded up to the 16-byte offset granularity.
  // Since offsets are 16-byte aligned the next allocation starts on a 16-byte
  // boundary anyway, so this consumes no extra space; it just lets the packed
  // size field be stored in 16-byte units too.
  const uint32_t allocSize   = static_cast<uint32_t>(size);
  const uint32_t reserveSize = (allocSize + (MIN_ALIGNMENT - 1)) & ~(MIN_ALIGNMENT - 1);

  if(uint64_t(reserveSize) + m_state.allocatedSize > m_info.maxAllocatedSize)
  {
    return E_OUTOFMEMORY;
  }

  // make sure there is an active write block
  if(m_state.writeBlockIndex == INVALID_BLOCK_INDEX)
  {
    uint32_t      blockIndex;
    const HRESULT result = acquireBlock(blockIndex);
    if(FAILED(result))
    {
      return result;  // out of block budget, or createNewBuffer failed (reported)
    }
    m_state.writeBlockIndex  = blockIndex;
    m_state.readBlockIndex   = blockIndex;
    m_state.activeBlockCount = 1;
  }

  // does it fit into the remaining space of the current write block?
  {
    Block&         block  = m_blocks[m_state.writeBlockIndex];
    const uint32_t offset = (block.writeOffset + alignment - 1) & ~(alignment - 1);

    if(uint64_t(offset) + reserveSize > m_info.blockSize)
    {
      // Not enough space left, the remaining tail of the block is abandoned.
      if(block.liveCount == 0)
      {
        // The write block has no live allocations, so we can reset its cursors and
        // reuse it from the start (the abandoned tail is simply skipped).
        block.writeOffset = 0;
        block.readOffset  = 0;
      }
      else
      {
        // There are still live allocations in this block (its read side has not
        // caught up), so we must keep it and continue in a fresh write block.
        const HRESULT result = appendWriteBlock();
        if(FAILED(result))
        {
          return result;  // out of block budget, or createNewBuffer failed (reported)
        }
      }
    }
  }

  // allocate from the current write block (guaranteed to fit now)
  Block&         block  = m_blocks[m_state.writeBlockIndex];
  const uint32_t offset = (block.writeOffset + alignment - 1) & ~(alignment - 1);
  assert(uint64_t(offset) + reserveSize <= m_info.blockSize);

  block.writeOffset = offset + reserveSize;
  block.liveCount++;

  m_state.allocatedSize += reserveSize;

  subAllocation.set(m_state.writeBlockIndex, offset, reserveSize);

  return S_OK;
}

void BufferCircularAllocator::subFree(BufferCircularAllocation& subAllocation)
{
  // make it legal to pass unset allocations
  if(!subAllocation)
  {
    return;
  }

  const uint32_t blockIndex = subAllocation.getBlockIndex();
  const uint32_t offset     = subAllocation.getOffset();
  const uint32_t size       = subAllocation.getSize();

  // frees must happen in allocation order: always the oldest block first
  assert(blockIndex == m_state.readBlockIndex && "frees must happen in allocation order (oldest block first)");

  Block& block = m_blocks[blockIndex];

  assert(offset >= block.readOffset && "frees must happen in allocation order within a block");
  assert(block.liveCount > 0);

  block.readOffset = offset + size;
  block.liveCount--;
  m_state.allocatedSize -= size;

  if(block.liveCount == 0)
  {
    if(m_state.readBlockIndex != m_state.writeBlockIndex)
    {
      // the oldest block fully drained: advance the read side and recycle it
      uint32_t nextIndex     = m_blocks[m_state.readBlockIndex].nextActive;
      m_state.readBlockIndex = nextIndex;
      m_state.activeBlockCount--;
      recycleBlock(blockIndex);
    }
    else if(m_info.keepBlockCount > 0)
    {
      // sole active block just drained: keep it and wrap the cursors in place
      block.writeOffset = 0;
      block.readOffset  = 0;
    }
    else
    {
      // sole active block and nothing should be kept around: release it
      m_state.readBlockIndex  = INVALID_BLOCK_INDEX;
      m_state.writeBlockIndex = INVALID_BLOCK_INDEX;
      m_state.activeBlockCount--;
      recycleBlock(blockIndex);
    }
  }

  subAllocation = {};
}

BufferRange BufferCircularAllocator::subRange(const BufferCircularAllocation& subAllocation) const
{
  // make it legal to pass unset allocations
  if(!subAllocation)
  {
    return {};
  }

  // the range covers [offset, offset + size) of the block in every address form
  return makeBufferRange(m_blocks[subAllocation.getBlockIndex()].buffer, subAllocation.getOffset(), subAllocation.getSize());
}

const Buffer& BufferCircularAllocator::getBlockBuffer(const uint32_t blockIndex) const
{
  assert(isBlockValid(blockIndex) && "invalid blockIndex");
  return m_blocks[blockIndex].buffer;
}

bool BufferCircularAllocator::isBlockValid(uint32_t blockIndex) const
{
  if(blockIndex >= m_blocks.size())
    return false;

  return m_blocks[blockIndex].buffer.buffer != nullptr;
}

uint32_t BufferCircularAllocator::getBlockCount() const
{
  return uint32_t(m_blocks.size());
}

HRESULT BufferCircularAllocator::createNewBuffer(Buffer& buffer, uint32_t blockIndex)
{
  // offsets inside the block carry any alignment above the buffer placement
  NVDX12_FAIL_RETURN(m_info.resourceAllocator->createBuffer(
      buffer, m_info.blockSize, m_info.usageFlags, m_info.memoryUsage, m_info.allocationFlags,
      std::min<uint64_t>(m_info.minAlignment, ResourceAllocator::BUFFER_PLACEMENT_ALIGNMENT)));

  NVDX12_DBG_CUSTOM_NAME(buffer.buffer.Get(),
                         std::string(typeid(*this).name()) + "::" + m_info.debugName + "_" + std::to_string(blockIndex));
  return S_OK;
}

}  // namespace nvdx12


//--------------------------------------------------------------------------------------------------
// Usage example
//--------------------------------------------------------------------------------------------------
[[maybe_unused]] static void usage_BufferCircularAllocator()
{
  // A circular allocator suits transient, per-frame linear streaming where the
  // allocations of a frame are made in order and retired together in the same
  // order once the GPU signaled that the frame is done.

  nvdx12::ResourceAllocator resourceAllocator;  // EX. initialize somehow

  nvdx12::BufferCircularAllocator circularAllocator;
  circularAllocator.init({.resourceAllocator = &resourceAllocator,
                          .debugName         = "streaming",
                          .usageFlags        = D3D12_RESOURCE_FLAG_NONE,          // copy source only
                          .memoryUsage       = nvdx12::MemoryUsage::eHostUpload,  // persistently mapped
                          .blockSize         = 4 * 1024 * 1024});

  // A simple ring of in-flight frames. Each frame remembers the allocations it
  // made so they can be freed - in order - once that frame completed.
  struct FrameAllocations
  {
    std::vector<nvdx12::BufferCircularAllocation> allocations;
  };
  std::vector<FrameAllocations> framesInFlight(3);

  uint64_t frameCounter = 0;
  while(true)
  {
    FrameAllocations& frame = framesInFlight[frameCounter % framesInFlight.size()];

    // Retire the previous use of this ring slot (EX: after waiting for its
    // fence value). Because frames complete in order, freeing its allocations
    // front-to-back keeps the strict FIFO order the allocator requires.
    for(size_t i = 0; i < frame.allocations.size(); i++)
    {
      circularAllocator.subFree(frame.allocations[i]);
    }
    frame.allocations.clear();

    // allocate some transient staging space for this frame
    for(uint32_t i = 0; i < 8; i++)
    {
      nvdx12::BufferCircularAllocation allocation;
      if(circularAllocator.subAllocate(allocation, 64 * 1024) == S_OK)
      {
        nvdx12::BufferRange range = circularAllocator.subRange(allocation);
        (void)range;  // EX. memcpy into range.mapping, CopyBufferRegion from range.buffer / range.offset ...
        frame.allocations.push_back(allocation);
      }
    }

    frameCounter++;
  }

  // call circularAllocator.deinit() on shutdown
}
