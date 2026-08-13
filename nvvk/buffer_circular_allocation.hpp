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

#include <cassert>
#include <cstdint>

namespace nvvk {
class BufferCircularAllocator;

// A single allocation handed out by `BufferCircularAllocator`, packed into 64 bits:
//   [ blockIndex : 10 ][ offset / 16 : 27 ][ size / 16 : 27 ]
//
// Offsets are always 16-byte aligned and sizes rounded up to 16 bytes, so both fit
// their 16-byte-unit fields (the rounding reserves no extra space, as the next
// offset is 16-byte aligned regardless). The all-ones blockIndex marks an invalid
// allocation. Supports blocks up to 2 GiB and allocations up to just under 2 GiB.
class BufferCircularAllocation
{
public:
  static constexpr uint32_t OFFSET_ALIGNMENT = 16;

  static constexpr uint32_t BLOCK_BITS  = 10;
  static constexpr uint32_t OFFSET_BITS = 27;
  static constexpr uint32_t SIZE_BITS   = 27;
  static_assert(BLOCK_BITS + OFFSET_BITS + SIZE_BITS == 64, "must pack into 64 bits");

  // all-ones block index marks an invalid allocation
  static constexpr uint32_t INVALID_BLOCK_INDEX = (1u << BLOCK_BITS) - 1;  // 1023
  // largest usable block index
  static constexpr uint32_t MAX_BLOCK_INDEX = INVALID_BLOCK_INDEX - 1;  // 1022
  // largest single allocation size in bytes (16-byte-unit size field)
  static constexpr uint64_t MAX_SIZE = (((uint64_t(1) << SIZE_BITS) - 1) * OFFSET_ALIGNMENT);
  // largest supported block size in bytes (16-byte-unit offset field)
  static constexpr uint64_t MAX_BLOCK_SIZE = uint64_t(OFFSET_ALIGNMENT) << OFFSET_BITS;  // 2 GiB

  // an allocation is valid when it points at a real block
  operator bool() const { return m_blockIndex != INVALID_BLOCK_INDEX; }

  // convenience accessors mirroring `nvvk::BufferSubAllocation`
  uint32_t getBlockIndex() const { return uint32_t(m_blockIndex); }
  // byte offset (already includes the requested alignment)
  uint32_t getOffset() const { return uint32_t(m_offset16) * OFFSET_ALIGNMENT; }
  // reserved byte size, rounded up to a 16-byte multiple (>= the requested size)
  uint32_t getSize() const { return uint32_t(m_size16) * OFFSET_ALIGNMENT; }

private:
  friend class BufferCircularAllocator;

  // offset and size must already be multiples of OFFSET_ALIGNMENT
  void set(uint32_t blockIndex, uint32_t offset, uint32_t size)
  {
    assert(blockIndex <= INVALID_BLOCK_INDEX);
    assert((offset % OFFSET_ALIGNMENT) == 0 && "offsets must be 16-byte aligned");
    assert((size % OFFSET_ALIGNMENT) == 0 && "sizes must be reserved in 16-byte units");
    assert(offset / OFFSET_ALIGNMENT < (1u << OFFSET_BITS));
    assert(size / OFFSET_ALIGNMENT < (1u << SIZE_BITS));
    m_blockIndex = blockIndex;
    m_offset16   = offset / OFFSET_ALIGNMENT;
    m_size16     = size / OFFSET_ALIGNMENT;
  }

  // default is invalid: blockIndex all-ones, offset/size 0
  uint64_t m_blockIndex : BLOCK_BITS = INVALID_BLOCK_INDEX;
  uint64_t m_offset16 : OFFSET_BITS  = 0;
  uint64_t m_size16 : SIZE_BITS      = 0;
};

static_assert(sizeof(BufferCircularAllocation) == 8, "BufferCircularAllocation must be 8 bytes");

}  // namespace nvvk
