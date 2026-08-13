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

#include <string>
#include <vector>

#include "buffer_circular_allocation.hpp"
#include "resource_allocator.hpp"

namespace nvvk {

// Allocates fixed-size block buffers and bump-allocates from them in a strictly
// circular (FIFO) fashion. Like `BufferSubAllocator` it controls block size, block
// count, buffer usage etc. and reuses its block array, but allocations and frees
// MUST happen in the same order (first allocated is first freed). That strict
// linearity drops the generic free-space book-keeping and fits per-frame linear
// streaming (e.g. staging uploads) retired together once the GPU is done.
//
// Active blocks form an ordered list from oldest (the "read" side, where frees
// happen) to newest (the "write" side, where allocations happen), threaded through
// the block array by index. Blocks are addressed by a stable index, and with
// threadSafeBlockAccess the array is pre-sized so it never reallocates.
//
// When a request does not fit the current write block, the rest of that block is
// abandoned and a fresh block is started. A block is recycled once all of its
// allocations are freed (kept in reserve or destroyed, per `keepBlockCount`).
class BufferCircularAllocator
{
public:
  // offsets are always at least 16-byte aligned so they fit the packed
  // 16-byte-unit offset field of BufferCircularAllocation
  static constexpr uint32_t MIN_ALIGNMENT     = BufferCircularAllocation::OFFSET_ALIGNMENT;  // 16
  static constexpr uint32_t DEFAULT_ALIGNMENT = 16;
  // limited by the 10-bit block index field of BufferCircularAllocation
  static constexpr uint32_t     MAX_TOTAL_BLOCKS   = BufferCircularAllocation::MAX_BLOCK_INDEX + 1;  // 1023
  static constexpr VkDeviceSize DEFAULT_BLOCK_SIZE = VkDeviceSize(128) * 1024 * 1024;

  BufferCircularAllocator() = default;
  ~BufferCircularAllocator();

  // Delete copy constructor and copy assignment operator
  BufferCircularAllocator(const BufferCircularAllocator&)            = delete;
  BufferCircularAllocator& operator=(const BufferCircularAllocator&) = delete;

  // Allow move constructor and move assignment operator (moved-from is left deinit-safe;
  // move assignment requires the target to be deinit'd).
  BufferCircularAllocator(BufferCircularAllocator&& other) noexcept;
  BufferCircularAllocator& operator=(BufferCircularAllocator&& other) noexcept;

  struct InitInfo
  {
    ResourceAllocator* resourceAllocator{};

    std::string debugName{};

    // properties of the internal buffer allocation
    VkBufferUsageFlags2KHR   usageFlags      = VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT;
    VmaMemoryUsage           memoryUsage     = VMA_MEMORY_USAGE_AUTO;
    VmaAllocationCreateFlags allocationFlags = {};
    std::vector<uint32_t>    queueFamilies   = {};

    // minimum (guaranteed) alignment for every allocation; block buffers are
    // created with it. A `subAllocate` call may request a higher alignment.
    // must be power-of-two and >= MIN_ALIGNMENT.
    uint32_t minAlignment = DEFAULT_ALIGNMENT;

    // size of a single block buffer; this is also the maximum size of a single
    // allocation. must be <= min(4 GB - 1, VkPhysicalDeviceVulkan11Properties::maxMemoryAllocationSize)
    VkDeviceSize blockSize = DEFAULT_BLOCK_SIZE;

    // 0 will default to blockSize * MAX_TOTAL_BLOCKS
    VkDeviceSize maxAllocatedSize = 0;

    // Set this to avoid destroying and re-creating block buffers in succession.
    // If greater than 0, one block is pre-allocated at init time.
    // When an emptied block would otherwise be destroyed, its buffer is kept in
    // reserve as long as the number of reserved buffers stays <= keepBlockCount.
    uint32_t keepBlockCount = 1;

    // allows thread-safe access to `subRange` and `getBlockBuffer`.
    // warning: requires a reasonably low non-zero `maxAllocatedSize`, as the
    // internal block array is pre-sized for the worst case so it never reallocates.
    bool threadSafeBlockAccess = false;
  };

  VkResult     init(const InitInfo& createInfo);
  void         deinit();
  VkDeviceSize getMaxAllocationSize() const { return m_state.maxAllocationSize; }

  struct Report
  {
    // sum of requests made by the user that are still live
    VkDeviceSize requestedSize{};
    // memory tied up in the active ring (all committed blocks)
    VkDeviceSize reservedSize{};
    // memory held in reserve (recycled empty blocks kept around)
    VkDeviceSize freeSize{};
    // total block buffer memory currently allocated
    VkDeviceSize allocatedSize{};
  };

  // current report on memory consumption
  Report getReport() const;

  // circular sub allocation.
  // size must be > 0 and <= blockSize; a zero size yields an invalid allocation.
  // alignment must be power-of-two; the effective alignment is max(alignment, minAlignment).
  VkResult subAllocate(BufferCircularAllocation& subAllocation, VkDeviceSize size, uint32_t alignment = DEFAULT_ALIGNMENT);

  // Free a sub allocation.
  // Frees MUST occur in the same order the allocations were made (FIFO).
  // Passing an invalid allocation (bool(subAllocation) == false) is valid and a no-op.
  void subFree(BufferCircularAllocation& subAllocation);

  // Get information about buffer/binding etc.
  // Passing an invalid allocation (bool(subAllocation) == false) is valid
  // and will just return a zeroed output.
  BufferRange subRange(const BufferCircularAllocation& subAllocation) const;

  // get full block buffer
  const nvvk::Buffer& getBlockBuffer(const uint32_t blockIndex) const;

  // offsets within `BufferCircularAllocation` are stored directly in bytes.
  // provided for API parity with `BufferSubAllocator`.
  uint32_t getOffsetUnitSize() const { return 1; }

  // test if a blockIndex is valid / and getBlockBuffer could be called
  bool isBlockValid(uint32_t blockIndex) const;

  // returns total number of internal block slots, not all may be valid
  // use `isBlockValid` to test individually
  uint32_t getBlockCount() const;

protected:
  static constexpr uint32_t INVALID_BLOCK_INDEX = BufferCircularAllocation::INVALID_BLOCK_INDEX;

  VkResult createNewBuffer(nvvk::Buffer& buffer, uint32_t blockIndex);

  // acquire a block slot with a ready-to-use buffer, reset to a clean state
  VkResult acquireBlock(uint32_t& blockIndex);
  // append a fresh block at the write side of the ordered list (abandons the
  // remaining tail of the current write block)
  VkResult appendWriteBlock();
  // recycle a drained block (already unlinked from the ordered list); keeps the
  // buffer in reserve or destroys it depending on `keepBlockCount`
  void recycleBlock(uint32_t blockIndex);

  // stored block, addressed by a stable block index
  struct Block
  {
    // the block's buffer; set once on creation, immutable until destroyed
    nvvk::Buffer buffer;
    // bump cursor: next byte offset to allocate from within this block
    uint32_t writeOffset = 0;
    // next byte offset expected to be freed (used for strict-order asserts and
    // to know when the block has fully drained)
    uint32_t readOffset = 0;
    // number of not-yet-freed allocations in this block
    uint32_t liveCount = 0;
    // ordered ring list, from oldest (read) to newest (write)
    uint32_t nextActive = INVALID_BLOCK_INDEX;
    // reserve list (buffer kept) or free-slot list (no buffer); a block is in at most one
    uint32_t nextFree = INVALID_BLOCK_INDEX;
  };

  struct State
  {
    // = blockSize, capped to the addressable range; the max single allocation
    VkDeviceSize maxAllocationSize{};
    // adjusted max number of blocks based on maxAllocatedSize
    uint32_t maxBlocks{};

    // statistics: sum of live requested sizes
    VkDeviceSize allocatedSize{};

    // ordered list of active blocks
    uint32_t readBlockIndex   = INVALID_BLOCK_INDEX;  // oldest, list head, frees happen here
    uint32_t writeBlockIndex  = INVALID_BLOCK_INDEX;  // newest, list tail, allocations happen here
    uint32_t activeBlockCount = 0;

    // reserve list of recycled blocks that keep their buffer (LIFO)
    uint32_t reserveBlockIndex = INVALID_BLOCK_INDEX;
    uint32_t reserveCount      = 0;
    // free-slot list of recycled block slots without a buffer (LIFO)
    uint32_t freeSlotIndex = INVALID_BLOCK_INDEX;

    // total number of block buffers alive (active ring + reserve)
    uint32_t blockBufferCount = 0;
  };

  InitInfo           m_info;
  State              m_state;
  std::vector<Block> m_blocks;
};

}  // namespace nvvk
