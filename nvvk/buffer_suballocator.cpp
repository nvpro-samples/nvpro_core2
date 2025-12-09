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

#include <cassert>

#include "check_error.hpp"
#include "debug_util.hpp"
#include "buffer_suballocator.hpp"

namespace nvvk {

BufferSubAllocator::~BufferSubAllocator()
{
  assert(m_info.resourceAllocator == nullptr && "Missing deinit()");
}

BufferSubAllocator::BufferSubAllocator(BufferSubAllocator&& other) noexcept
{
  std::swap(m_state.allocatedSize, other.m_state.allocatedSize);
  std::swap(m_blocks, other.m_blocks);
  std::swap(m_info, other.m_info);
  std::swap(m_state.internalBlockUnits, other.m_state.internalBlockUnits);
  std::swap(m_state.maxBlocks, other.m_state.maxBlocks);
}

BufferSubAllocator& BufferSubAllocator::operator=(BufferSubAllocator&& other) noexcept
{
  if(this != &other)
  {
    assert(m_info.resourceAllocator == nullptr && "Missing deinit()");

    std::swap(m_state.allocatedSize, other.m_state.allocatedSize);
    std::swap(m_blocks, other.m_blocks);
    std::swap(m_info, other.m_info);
    std::swap(m_state.internalBlockUnits, other.m_state.internalBlockUnits);
    std::swap(m_state.maxBlocks, other.m_state.maxBlocks);
  }

  return *this;
}

VkResult BufferSubAllocator::init(const InitInfo& info)
{
  assert(m_info.resourceAllocator == nullptr);

  assert(info.resourceAllocator != nullptr);
  assert(info.minAlignment <= MAX_ALIGNMENT);
  assert(info.minAlignment >= MIN_ALIGNMENT);

  m_state.maxAllocationSize = std::min((VkDeviceSize(1) << (sizeof(BufferSubAllocation::size)) * 8) - 1,
                                       info.resourceAllocator->getMaxMemoryAllocationSize());

  assert(info.blockSize <= m_state.maxAllocationSize);

  m_info = info;
  if(!m_info.maxAllocatedSize)
  {
    m_info.maxAllocatedSize = info.blockSize * MAX_TOTAL_BLOCKS;
  }

  size_t maxBlocks = (m_info.maxAllocatedSize + m_info.blockSize - 1) / m_info.blockSize;
  assert(maxBlocks <= MAX_TOTAL_BLOCKS);

  m_state.maxBlocks = static_cast<uint32_t>(maxBlocks);
  m_state.internalBlockUnits = static_cast<uint32_t>((m_info.blockSize + m_info.minAlignment - 1) / m_info.minAlignment);

  if(m_info.keepLastBlock)
  {
    Block block;
    block.offsetAllocator = std::make_unique<OffsetAllocator::Allocator>(m_state.internalBlockUnits, m_info.perBlockAllocations);
    NVVK_FAIL_RETURN(createNewBuffer(block.buffer, VkDeviceSize(m_state.internalBlockUnits) * m_info.minAlignment,
                                     m_info.minAlignment, 0));

    m_blocks.push_back(std::move(block));

    m_state.activeBlockCount = 1;
    m_state.activeBlockIndex = 0;
  }

  return VK_SUCCESS;
}

void BufferSubAllocator::deinit()
{
  if(!m_info.resourceAllocator)
    return;

  for(Block& it : m_blocks)
  {
    m_info.resourceAllocator->destroyBuffer(it.buffer);
  }

  m_info  = {};
  m_state = {};
  m_blocks.clear();
  m_blocks.shrink_to_fit();
}

BufferSubAllocator::Report BufferSubAllocator::getReport() const
{
  BufferSubAllocator::Report report;

  for(size_t i = 0; i < m_blocks.size(); i++)
  {
    const Block& block = m_blocks[i];

    const OffsetAllocator::Allocator* offsetAllocator = block.offsetAllocator.get();

    if(offsetAllocator)
    {
      OffsetAllocator::StorageReport storageReport = offsetAllocator->storageReport();
      report.reservedSize += VkDeviceSize(m_state.internalBlockUnits - storageReport.totalFreeSpace) * m_info.minAlignment;
      report.freeSize = VkDeviceSize(storageReport.totalFreeSpace) * m_info.minAlignment;
    }
    else
    {
      // dedicated blocks
      report.reservedSize += block.buffer.bufferSize;
    }
  }

  report.requestedSize = m_state.allocatedSize;

  return report;
}

uint32_t BufferSubAllocator::acquireBlockIndex()
{
  uint32_t freeBlockIndex = m_state.freeBlockIndex;
  if(freeBlockIndex != INVALID_BLOCK_INDEX)
  {
    m_state.freeBlockIndex = m_blocks[m_state.freeBlockIndex].nextFreeIndex;
  }
  else
  {
    freeBlockIndex = uint32_t(m_blocks.size());
    m_blocks.push_back({});
  }

  return freeBlockIndex;
}

VkResult BufferSubAllocator::subAllocate(BufferSubAllocation& subAllocation, VkDeviceSize size, uint32_t alignment)
{
  subAllocation = {};

  assert(alignment % MIN_ALIGNMENT == 0);
  assert(alignment >= MIN_ALIGNMENT && alignment <= MAX_ALIGNMENT);
  assert(size <= m_state.maxAllocationSize);

  bool alignmentIsPowerOfTwo = (alignment & (alignment - 1)) == 0;

  if(size + m_state.allocatedSize > m_info.maxAllocatedSize)
  {
    return VK_ERROR_OUT_OF_DEVICE_MEMORY;
  }

  m_state.allocatedSize += size;

  // if large use a dedicated block
  if(size >= m_info.blockSize)
  {
    if(m_state.freeBlockIndex == INVALID_BLOCK_INDEX && m_blocks.size() == size_t(m_state.maxBlocks))
    {
      return VK_ERROR_OUT_OF_DEVICE_MEMORY;
    }

    // recycle a block or new one
    uint32_t freeBlockIndex = acquireBlockIndex();

    if(!alignmentIsPowerOfTwo)
    {
      // find largest power of 2 that fits into alignment
      uint32_t newAlignment = MIN_ALIGNMENT;
      for(uint32_t searchAlignment = MIN_ALIGNMENT; searchAlignment <= MAX_ALIGNMENT; searchAlignment *= 2)
      {
        if((alignment & (searchAlignment - 1)) == 0)
        {
          newAlignment = searchAlignment;
        }
        else
        {
          break;
        }
      }
      alignment = newAlignment;
    }

    Block& block = m_blocks[freeBlockIndex];
    NVVK_FAIL_RETURN(createNewBuffer(block.buffer, size, std::max(m_info.minAlignment, alignment), freeBlockIndex));

    subAllocation.allocation.offset   = 0;
    subAllocation.allocation.metadata = OffsetAllocator::Allocation::NO_SPACE;
    subAllocation.size                = static_cast<uint32_t>(size);
    subAllocation.alignmentMinusOne   = alignment - 1;
    subAllocation.block               = uint16_t(freeBlockIndex);
#ifndef NDEBUG
    subAllocation.allocator = this;
#endif

    // dedicated blocks are _not_ thrown into the active block list (m_activeBlockIndex)

    return VK_SUCCESS;
  }


  // else try to find a sub allocation

  // adjust the size to account for local alignment
  VkDeviceSize sizeAllocate = size;

  // for non power of two, always add extra space to return a proper offset
  if(!alignmentIsPowerOfTwo || alignment > m_info.minAlignment)
  {
    // adjust for requested alignment and add safety margin to size.
    // The offset returned from OffsetAllocator will only be aligned to m_info.minAlignment.
    // With the extra safety margin space, we can later adjust the returned offset to alignment,
    // see logic in `subRange`.
    sizeAllocate = (sizeAllocate + alignment - 1);
  }

  // offset allocator works in units of `m_info.minAlignment`
  uint32_t allocatorUnits = static_cast<uint32_t>((sizeAllocate + m_info.minAlignment - 1) / m_info.minAlignment);


  // iterate over active blocks to find allocation

  uint32_t activeBlockIndex = m_state.activeBlockIndex;

  while(activeBlockIndex != INVALID_BLOCK_INDEX)
  {
    Block& block = m_blocks[activeBlockIndex];

    // attempt to sub allocate from active blocks

    OffsetAllocator::Allocation allocation = block.offsetAllocator->allocate(allocatorUnits);

    if(allocation.offset != OffsetAllocator::Allocation::NO_SPACE)
    {
      subAllocation.allocation        = allocation;
      subAllocation.size              = static_cast<uint32_t>(size);
      subAllocation.alignmentMinusOne = alignment - 1;
      subAllocation.block             = uint16_t(activeBlockIndex);
#ifndef NDEBUG
      subAllocation.allocator = this;
#endif

      return VK_SUCCESS;
    }

    activeBlockIndex = block.nextActiveIndex;
  }

  // could not find anything

  // if we reached the limit for blocks, bail out
  if(m_state.freeBlockIndex == INVALID_BLOCK_INDEX && m_blocks.size() == size_t(m_state.maxBlocks))
  {
    return VK_ERROR_OUT_OF_DEVICE_MEMORY;
  }

  {
    // add new block

    uint32_t freeBlockIndex = acquireBlockIndex();

    Block& block = m_blocks[freeBlockIndex];
    block.offsetAllocator = std::make_unique<OffsetAllocator::Allocator>(m_state.internalBlockUnits, m_info.perBlockAllocations);
    NVVK_FAIL_RETURN(createNewBuffer(block.buffer, VkDeviceSize(m_state.internalBlockUnits) * m_info.minAlignment,
                                     m_info.minAlignment, freeBlockIndex));

    // insert block into active block list
    if(m_state.activeBlockIndex != INVALID_BLOCK_INDEX)
    {
      m_blocks[m_state.activeBlockIndex].prevActiveIndex = freeBlockIndex;
    }

    block.nextActiveIndex = m_state.activeBlockIndex;

    // make it new list head
    m_state.activeBlockIndex = freeBlockIndex;
    m_state.activeBlockCount++;


    // sub allocate from new block

    OffsetAllocator::Allocation allocation = block.offsetAllocator->allocate(allocatorUnits);

    if(allocation.offset != OffsetAllocator::Allocation::NO_SPACE)
    {
      subAllocation.allocation        = allocation;
      subAllocation.size              = static_cast<uint32_t>(size);
      subAllocation.alignmentMinusOne = alignment - 1;
      subAllocation.block             = uint16_t(freeBlockIndex);
#ifndef NDEBUG
      subAllocation.allocator = this;
#endif

      return VK_SUCCESS;
    }
    else
    {
      return VK_ERROR_OUT_OF_DEVICE_MEMORY;
    }
  }
}

void BufferSubAllocator::subFree(BufferSubAllocation& subAllocation)
{
  // make it legal to pass unset ranges
  if(!subAllocation)
  {
    return;
  }

#ifndef NDEBUG
  assert(subAllocation.allocator == this);
#endif

  OffsetAllocator::Allocator* offsetAllocator = m_blocks[subAllocation.block].offsetAllocator.get();

  // dedicated blocks might not have an offset allocator
  if(offsetAllocator)
  {
    offsetAllocator->free(subAllocation.allocation);
  }

  m_state.allocatedSize -= subAllocation.size;

  // check if dedicated block or empty
  if(!offsetAllocator || offsetAllocator->storageReport().totalFreeSpace == m_state.internalBlockUnits)
  {
    // always free if dedicated
    // and maybe depending if we are the last one
    if(!offsetAllocator || (m_state.activeBlockCount > 1 || !m_info.keepLastBlock))
    {
      m_info.resourceAllocator->destroyBuffer(m_blocks[subAllocation.block].buffer);

      // blocks with OffsetAllocators are counted to active blocks
      if(offsetAllocator)
      {
        m_state.activeBlockCount--;

        // need to remove from linked list of active blocks

        uint32_t selfActiveIndex = subAllocation.block;
        uint32_t prevActiveIndex = m_blocks[selfActiveIndex].prevActiveIndex;
        uint32_t nextActiveIndex = m_blocks[selfActiveIndex].nextActiveIndex;
        if(prevActiveIndex != INVALID_BLOCK_INDEX)
        {
          // set previous's next to self next
          m_blocks[prevActiveIndex].nextActiveIndex = nextActiveIndex;
        }
        if(nextActiveIndex != INVALID_BLOCK_INDEX)
        {
          // set next's previous to self previous
          m_blocks[nextActiveIndex].prevActiveIndex = prevActiveIndex;
        }
        if(m_state.activeBlockIndex == selfActiveIndex)
        {
          m_state.activeBlockIndex = nextActiveIndex;
        }
      }

      // nuke it completely
      m_blocks[subAllocation.block] = {};

      // chain into linked list of empty blocks
      m_blocks[subAllocation.block].nextFreeIndex = m_state.freeBlockIndex;
      m_state.freeBlockIndex                      = subAllocation.block;
    }
  }

  subAllocation = {};
}

BufferRange BufferSubAllocator::subRange(const BufferSubAllocation& subAllocation) const
{
  // make it legal to pass unset ranges
  if(!subAllocation)
  {
    return {};
  }

#ifndef NDEBUG
  assert(subAllocation.allocator == this);
#endif

  BufferRange info;
  info.buffer  = m_blocks[subAllocation.block].buffer.buffer;
  info.address = m_blocks[subAllocation.block].buffer.address;
  info.mapping = m_blocks[subAllocation.block].buffer.mapping;

  info.range = subAllocation.size;

  // OffsetAllocator's offset is in units of `m_info.minAlignment`
  info.offset = subAllocation.allocation.offset * m_info.minAlignment;

  // The original requested alignment might have been greater than the minAlignment,
  // or might be non-power-of-two.
  // In that case we need to re-adjust the offset, which is safe to work as we
  // allocated a safety margin.
  uint32_t alignment = uint32_t(subAllocation.alignmentMinusOne) + 1;

  // allow non-power-of-two alignments
  uint32_t rest = info.offset % alignment;
  if(rest != 0)
  {
    info.offset += alignment - rest;
  }

  // apply offset to address
  info.address += info.offset;
  if(info.mapping)
  {
    // and mapping if applicable
    info.mapping += info.offset;
  }

  return info;
}

VkResult BufferSubAllocator::createNewBuffer(nvvk::Buffer& buffer, VkDeviceSize size, uint32_t alignment, uint32_t blockIndex)
{
  NVVK_FAIL_RETURN(m_info.resourceAllocator->createBuffer(buffer, size, m_info.usageFlags, m_info.memoryUsage,
                                                          m_info.allocationFlags, alignment, m_info.queueFamilies));

  nvvk::DebugUtil::getInstance().setObjectName(buffer.buffer, std::string(typeid(*this).name()) + "::" + m_info.debugName
                                                                  + "_" + std::to_string(blockIndex));
  return VK_SUCCESS;
}

}  // namespace nvvk


//--------------------------------------------------------------------------------------------------
// Usage example
//--------------------------------------------------------------------------------------------------
[[maybe_unused]] static void usage_BufferSubAllocator()
{
  // imagine we have a scene with lots of meshes that are typically not very big
  struct Mesh
  {
    size_t vertexSize;
    size_t indexSize;

    nvvk::BufferSubAllocation vertex;
    nvvk::BufferSubAllocation index;
  };

  std::vector<Mesh> meshes;

  nvvk::ResourceAllocator resourceAllocator;  // EX. initialize somehow

  // create the buffer sub allocator covering the buffers
  // note the
  nvvk::BufferSubAllocator bufferSubAllocator;
  bufferSubAllocator.init({.debugName   = "meshes",
                           .usageFlags  = VK_BUFFER_USAGE_2_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_2_VERTEX_BUFFER_BIT,
                           .memoryUsage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
                           .blockSize   = 64 * 1024 * 1024});  // allocations greater blockSize get their own block

  for(Mesh& mesh : meshes)
  {
    bufferSubAllocator.subAllocate(mesh.vertex, mesh.vertexSize);
    bufferSubAllocator.subAllocate(mesh.index, mesh.indexSize);

    // upload somehow
  }

  // later when drawing

  while(true)
  {
    VkCommandBuffer cmd{};  // per-frame command buffer, setup state etc.

    VkBuffer lastVertexBuffer = {};
    VkBuffer lastIndexBuffer  = {};

    for(Mesh& mesh : meshes)
    {
      nvvk::BufferRange vertexRange = bufferSubAllocator.subRange(mesh.vertex);
      nvvk::BufferRange indexRange  = bufferSubAllocator.subRange(mesh.index);

      // given we sub-allocate there is a higher chance we use the same buffers
      // so don't always bind per-mesh

      if(vertexRange.buffer != lastVertexBuffer)
      {
        VkDeviceSize offset = {0};
        VkDeviceSize size   = {VK_WHOLE_SIZE};
        VkDeviceSize stride = {sizeof(float) * 4};
        vkCmdBindVertexBuffers2(cmd, 0, 1, &vertexRange.buffer, &offset, &size, &stride);

        lastVertexBuffer = vertexRange.buffer;
      }

      if(indexRange.buffer != lastIndexBuffer)
      {
        vkCmdBindIndexBuffer(cmd, indexRange.buffer, 0, VK_INDEX_TYPE_UINT16);

        lastIndexBuffer = indexRange.buffer;
      }

      // need to apply the buffer offsets to the draw given we bound the full buffer
      vkCmdDrawIndexed(cmd, uint32_t(mesh.indexSize / sizeof(uint16_t)), 1U, uint32_t(indexRange.offset / sizeof(uint16_t)),
                       int32_t(vertexRange.offset / (sizeof(float) * 4)), 0U);
    }
  }
}
