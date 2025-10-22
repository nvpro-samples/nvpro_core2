/*
* Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
* SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
* SPDX-License-Identifier: Apache-2.0
*/

#pragma once

#include <vector>
#include <string>
#include <memory>

#include <offsetallocator/offsetAllocator.hpp>
#include <vulkan/vulkan_core.h>

#include "resource_allocator.hpp"

namespace nvvk {

class BufferSubAllocation
{
public:
  BufferSubAllocation() = default;

  operator bool() const { return allocation.offset != OffsetAllocator::Allocation::NO_SPACE; }

private:
  friend class BufferSubAllocator;

  // the allocation.offset is in units of BufferSubAllocator's minAlignment
  OffsetAllocator::Allocation allocation;

  // original requested allocation size
  // the OffsetAllocator's size may be bigger given its internal free space search
  uint32_t size{};

  // original requested alignment
  // This alignment may need to be applied when converting the allocation.offset back
  // to actual byte offset
  uint16_t alignmentMinusOne{};

  uint16_t block{};
#ifndef NDEBUG
  class BufferSubAllocator* allocator{};
#endif
};

// Allocates blocks of buffers that one can
// sub allocate from.
// If a requested allocation size is bigger than the
// block size, a dedicated block/buffer will be used.
class BufferSubAllocator
{
public:
  static constexpr uint32_t     MIN_ALIGNMENT      = 4;
  static constexpr uint32_t     MAX_ALIGNMENT      = 1 << (sizeof(BufferSubAllocation::alignmentMinusOne) * 8);
  static constexpr uint32_t     MAX_TOTAL_BLOCKS   = 1 << (sizeof(BufferSubAllocation::block) * 8);
  static constexpr VkDeviceSize DEFAULT_BLOCK_SIZE = VkDeviceSize(128) * 1024 * 1024;
  static constexpr uint32_t     DEFAULT_ALIGNMENT  = 16;

  BufferSubAllocator() = default;
  ~BufferSubAllocator();

  // Delete copy constructor and copy assignment operator
  BufferSubAllocator(const BufferSubAllocator&)            = delete;
  BufferSubAllocator& operator=(const BufferSubAllocator&) = delete;

  // Allow move constructor and move assignment operator
  BufferSubAllocator(BufferSubAllocator&& other) noexcept;
  BufferSubAllocator& operator=(BufferSubAllocator&& other) noexcept;

  struct InitInfo
  {
    ResourceAllocator* resourceAllocator{};

    std::string debugName{};

    // properties of the internal buffer allocation
    VkBufferUsageFlagBits2    usageFlags      = VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT;
    VmaMemoryUsage            memoryUsage     = VMA_MEMORY_USAGE_AUTO;
    VmaAllocationCreateFlags  allocationFlags = {};
    std::span<const uint32_t> queueFamilies;

    // must be power-of-two
    uint32_t minAlignment = DEFAULT_ALIGNMENT;

    // a single block's OffsetAllocator can track this many sub-allocations
    uint32_t perBlockAllocations = 128 * 1024;

    // must be <= min(4GB, VkPhysicalDeviceVulkan11Properties::maxMemoryAllocationSize)
    VkDeviceSize blockSize = DEFAULT_BLOCK_SIZE;

    // 0 will default to blockSize * MAX_TOTAL_BLOCKS
    VkDeviceSize maxAllocatedSize = 0;

    // to avoid freeing and allocating blocks in succession
    bool keepLastBlock = true;
  };

  VkResult     init(const InitInfo& createInfo);
  void         deinit();
  VkDeviceSize getMaxAllocationSize() const { return m_state.maxAllocationSize; }

  struct Report
  {
    // sum of requests made by user
    VkDeviceSize requestedSize{};
    // internal usage, can be greater than requestedSize
    VkDeviceSize reservedSize{};
    // what is available within internal usage
    VkDeviceSize freeSize{};
  };

  // current report on memory consumption
  Report getReport() const;

  // sub allocate
  // alignment must fulfill MIN_ALIGNMENT and MAX_ALIGNMENT
  // alignment is legal to be non-power-of-two, but must be divisible by MIN_ALIGNMENT, then the returned offsets will be a multiple of the alignment
  // size must be <= min(4GB, VkPhysicalDeviceVulkan11Properties::maxMemoryAllocationSize)
  VkResult subAllocate(BufferSubAllocation& subAllocation, VkDeviceSize size, uint32_t alignment = DEFAULT_ALIGNMENT);

  // Free sub allocation
  // Passing an invalid suballocation (bool(subAllocation) == false) is valid
  void subFree(BufferSubAllocation& subAllocation);

  // Get information about buffer/binding etc.
  // Passing an invalid suballocation (bool(subAllocation) == false) is valid
  // and will just return a zeroed output
  BufferRange subRange(const BufferSubAllocation& subAllocation) const;

protected:
  static constexpr uint32_t INVALID_BLOCK_INDEX = ~0u;

  VkResult createNewBuffer(nvvk::Buffer& buffer, VkDeviceSize size, uint32_t alignment, uint32_t blockIndex);

  uint32_t acquireBlockIndex();

  struct Block
  {
    // can be null for dedicated block that have only a single big allocation > m_info.blockSize
    std::unique_ptr<OffsetAllocator::Allocator> offsetAllocator;
    // can be null if block was fully deallocated
    nvvk::Buffer buffer;
    // continuation of single linked list of blocks that were deallocated completely
    uint32_t nextFreeIndex = INVALID_BLOCK_INDEX;
    // continuation of double linked list of blocks that have OffsetAllocators
    uint32_t nextActiveIndex = INVALID_BLOCK_INDEX;
    uint32_t prevActiveIndex = INVALID_BLOCK_INDEX;
  };

  struct State
  {
    // adjusted size based on config
    VkDeviceSize maxAllocationSize{};
    // adjusted size as the offset allocator operates in units of m_info.minAlignment
    uint32_t internalBlockUnits{};
    // adjusted max blocks based on m_info.totalSize
    uint32_t maxBlocks{};

    // statistics
    VkDeviceSize allocatedSize{};

    // single linked list of blocks that were deallocated completely
    // list head
    uint32_t freeBlockIndex = INVALID_BLOCK_INDEX;

    // active blocks are blocks that have OffsetAllocators (i.e. not dedicated to a single allocation)
    uint32_t activeBlockCount = 0;

    // double linked list of blocks that are active
    // list head
    uint32_t activeBlockIndex = INVALID_BLOCK_INDEX;
  };

  InitInfo           m_info;
  State              m_state;
  std::vector<Block> m_blocks;
};

}  // namespace nvvk
