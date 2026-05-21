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

#include <cstdint>

#include <nvutils/offset_allocator.hpp>

namespace nvvk {
class BufferSubAllocator;

class BufferSubAllocation
{
public:
  BufferSubAllocation() = default;

  operator bool() const { return allocation.offset != OffsetAllocator::Allocation::NO_SPACE; }

  // useful for sorting by buffer binds
  uint16_t getBlockIndex() const { return block; }

private:
  friend class BufferSubAllocator;

  // the allocation.offset is in units of BufferSubAllocator's minAlignment
  OffsetAllocator::Allocation allocation{};

  // original requested allocation size
  // the OffsetAllocator's size may be bigger given its internal free space search
  uint32_t size{};

  // original requested alignment
  // This alignment may need to be applied when converting the allocation.offset back
  // to actual byte offset
  uint16_t alignmentMinusOne{};

  uint16_t block{};
#if !defined(NDEBUG) && !defined(NVVK_DISABLE_BUFFER_SUB_ALLOCATOR_DEBUG_POINTER)
  class BufferSubAllocator* allocator{};
#endif
};
}  // namespace nvvk
