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

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "buffer_suballocation.hpp"
#include "resource_allocator.hpp"

namespace nvdx12 {

/*-------------------------------------------------------------------------------------------------
# class nvdx12::BufferSubAllocator

> Allocates blocks of buffers that one can sub allocate from, the D3D12
> counterpart of nvvk::BufferSubAllocator.

Blocks are nvdx12::Buffers created by the ResourceAllocator, so each block
has its own resource, raw UAV (shader address), GPU virtual address and,
for host memory, persistent mapping. Sub-allocations are tracked per block by
an OffsetAllocator; the allocator grows by adding blocks and gives empty
blocks back, keeping `keepBlockCount` of them. If a requested allocation
size is bigger than the block size, a dedicated block/buffer will be used.
`subRange` resolves a BufferSubAllocation to a BufferRange: the block's
resource, the byte offset in it, and the shader address, GPU address and
mapping of the range (resources.hpp).

Every block has its own descriptor: a range's `address` is
`(block uavSlot << 32) | offset`.

Differences with nvvk that follow from D3D12:
- InitInfo takes the buffer creation parameters of
  nvdx12::ResourceAllocator::createBuffer: `usageFlags` is the
  D3D12_RESOURCE_FLAGS (ALLOW_UNORDERED_ACCESS, the default, gives blocks a
  shader address), `memoryUsage` an nvdx12::MemoryUsage (eDevice by
  default; eHostUpload / eHostReadback blocks are mapped) and
  `allocationFlags` D3D12MA::ALLOCATION_FLAGS. No `queueFamilies`: D3D12
  resources have no queue-family sharing mode.
- Results are HRESULTs (E_OUTOFMEMORY where nvvk returns
  VK_ERROR_OUT_OF_DEVICE_MEMORY).
- Block resources are placed on at least 256 bytes (the ResourceAllocator
  rejects more): offsets inside a block honor any alignment up to
  MAX_ALIGNMENT, but a range's `gpuAddress` is only guaranteed to be
  aligned to min(alignment, 256), which covers every D3D12 requirement
  (constant buffers, acceleration structures, raw views).
- A request below the block size that an empty block cannot take (its
  alignment margin, or the OffsetAllocator's bin rounding, makes it too
  large) also gets a dedicated block; nvvk returns an out-of-memory error
  for it after adding a block that stays empty.
- Blocks and dedicated allocations stay below 4 GiB (sizes are 32 bits as in
  nvvk), so every range is reached through the block's first UAV window.

```cpp
nvdx12::BufferSubAllocator subAllocator;
subAllocator.init({.resourceAllocator = &allocator, .debugName = "meshes", .blockSize = 64 << 20});
nvdx12::BufferSubAllocation allocation;
subAllocator.subAllocate(allocation, bytes);
nvdx12::BufferRange range = subAllocator.subRange(allocation);  // range.buffer / offset / address
...
subAllocator.subFree(allocation);
subAllocator.deinit();
```

See usage_BufferSubAllocator() in buffer_suballocator.cpp.
-------------------------------------------------------------------------------------------------*/

class BufferSubAllocator
{
public:
  static constexpr uint32_t MIN_ALIGNMENT      = 4;
  static constexpr uint32_t MAX_ALIGNMENT      = 1 << (sizeof(BufferSubAllocation::m_alignmentMinusOne) * 8);
  static constexpr uint32_t MAX_TOTAL_BLOCKS   = (1 << (sizeof(BufferSubAllocation::m_block) * 8)) - 1;
  static constexpr uint64_t DEFAULT_BLOCK_SIZE = uint64_t(128) * 1024 * 1024;
  static constexpr uint32_t DEFAULT_ALIGNMENT  = 16;

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

    // properties of the internal buffer allocation (ResourceAllocator::createBuffer)
    D3D12_RESOURCE_FLAGS      usageFlags      = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    MemoryUsage               memoryUsage     = MemoryUsage::eDevice;
    D3D12MA::ALLOCATION_FLAGS allocationFlags = D3D12MA::ALLOCATION_FLAG_NONE;

    // must be power-of-two, >= MIN_ALIGNMENT and <= MAX_ALIGNMENT
    uint32_t minAlignment = DEFAULT_ALIGNMENT;

    // a single block's OffsetAllocator can track this many sub-allocations
    uint32_t perBlockAllocations = 128 * 1024;

    // must be <= 4GB - 1
    uint64_t blockSize = DEFAULT_BLOCK_SIZE;

    // 0 will default to blockSize * MAX_TOTAL_BLOCKS
    uint64_t maxAllocatedSize = 0;

    // Set this to avoid freeing and allocating blocks in succession.
    // If greater than 0, one block is pre-allocated at init time.
    // When an empty block would otherwise be destroyed, it is kept as long as
    // the total active block count does not exceed this value.
    // Allocations that are greater than the blockSize will get their own allocation
    // that isn't counted against this limit.
    uint32_t keepBlockCount = 1;
    // allows thread-safe access to `subRange` and `getBlockBuffer`
    // warning requires reasonable low non-zero `maxAllocatedSize`
    // as we allocate internal arrays for worst-case
    bool threadSafeBlockAccess = false;
  };

  HRESULT  init(const InitInfo& createInfo);
  void     deinit();
  uint64_t getMaxAllocationSize() const { return m_state.maxAllocationSize; }

  struct Report
  {
    // sum of requests made by user
    uint64_t requestedSize{};
    // internal usage, can be greater than requestedSize
    uint64_t reservedSize{};
    // what is available within internal usage
    uint64_t freeSize{};
    // sum of allocations
    uint64_t allocatedSize{};
  };

  // current report on memory consumption
  Report getReport() const;

  // sub allocate
  // alignment must fulfill MIN_ALIGNMENT and MAX_ALIGNMENT
  // alignment is legal to be non-power-of-two, but must be divisible by MIN_ALIGNMENT, then the returned offsets will be a multiple of the alignment
  // size must be > 0 and <= 4GB - 1 (E_INVALIDARG otherwise)
  HRESULT subAllocate(BufferSubAllocation& subAllocation, uint64_t size, uint32_t alignment = DEFAULT_ALIGNMENT);

  // Free sub allocation
  // Passing an invalid suballocation (bool(subAllocation) == false) is valid
  void subFree(BufferSubAllocation& subAllocation);

  // Get information about buffer/binding etc.
  // Passing an invalid suballocation (bool(subAllocation) == false) is valid
  // and will just return a zeroed output
  BufferRange subRange(const BufferSubAllocation& subAllocation) const;

  // get full block buffer
  const Buffer& getBlockBuffer(const uint32_t blockIndex) const;

  // get offset unit byte size, as used within `BufferSubAllocation`
  uint32_t getOffsetUnitSize() const { return m_info.minAlignment; }

  // test if a blockIndex is valid / and getBlockBuffer could be called
  bool isBlockValid(uint32_t blockIndex) const;

  // returns total number of internal blocks, not all may be valid
  // use `isBlockValid` to test individually
  uint32_t getBlockCount() const;

protected:
  static constexpr uint32_t INVALID_BLOCK_INDEX = ~0u;

  HRESULT createNewBuffer(Buffer& buffer, uint64_t size, uint32_t alignment, uint32_t blockIndex);

  uint32_t acquireBlockIndex();
  // returns a block that has no buffer / allocations back to the free list
  void releaseBlockIndex(uint32_t blockIndex);

  struct Block
  {
    // can be null for dedicated block that have only a single big allocation > m_info.blockSize
    std::unique_ptr<OffsetAllocator::Allocator> offsetAllocator;
    // can be null if block was fully deallocated
    Buffer buffer;
    // continuation of single linked list of blocks that were deallocated completely
    uint32_t nextFreeIndex = INVALID_BLOCK_INDEX;
    // continuation of double linked list of blocks that have OffsetAllocators
    uint32_t nextActiveIndex = INVALID_BLOCK_INDEX;
    uint32_t prevActiveIndex = INVALID_BLOCK_INDEX;
  };

  struct State
  {
    // adjusted size based on config
    uint64_t maxAllocationSize{};
    // adjusted size as the offset allocator operates in units of m_info.minAlignment
    uint32_t internalBlockUnits{};
    // largest request (in units) the OffsetAllocator of an empty block accepts:
    // it bins free space rounding down and requests rounding up, so this can be
    // below internalBlockUnits; larger requests get a dedicated block
    uint32_t maxBlockUnits{};
    // adjusted max blocks based on m_info.maxAllocatedSize
    uint32_t maxBlocks{};

    // statistics
    uint64_t allocatedSize{};

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

}  // namespace nvdx12
