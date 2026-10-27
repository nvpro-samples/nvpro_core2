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

#include "buffer_suballocator.hpp"

#include <algorithm>
#include <cassert>
#include <typeinfo>

#include "check_error.hpp"
#include "debug_util.hpp"

namespace nvdx12 {

BufferSubAllocator::~BufferSubAllocator()
{
  assert(m_info.resourceAllocator == nullptr && "Missing deinit()");
}

BufferSubAllocator::BufferSubAllocator(BufferSubAllocator&& other) noexcept
{
  // this is freshly default-constructed; swapping all members transfers other's
  // full state here and leaves other empty (deinit-safe: resourceAllocator == null).
  std::swap(m_info, other.m_info);
  std::swap(m_state, other.m_state);
  std::swap(m_blocks, other.m_blocks);
}

BufferSubAllocator& BufferSubAllocator::operator=(BufferSubAllocator&& other) noexcept
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

HRESULT BufferSubAllocator::init(const InitInfo& info)
{
  assert(m_info.resourceAllocator == nullptr);

  assert(info.resourceAllocator != nullptr);
  if(info.minAlignment < MIN_ALIGNMENT || info.minAlignment > MAX_ALIGNMENT || (info.minAlignment & (info.minAlignment - 1)) != 0)
  {
    return E_INVALIDARG;
  }

  m_state.maxAllocationSize = std::min((uint64_t(1) << (sizeof(BufferSubAllocation::m_size) * 8)) - 1,
                                       info.resourceAllocator->getMaxMemoryAllocationSize());

  assert(info.blockSize <= m_state.maxAllocationSize);
  assert(!info.threadSafeBlockAccess || info.maxAllocatedSize != 0);
  const uint64_t maxAllocatedSize = info.maxAllocatedSize ? info.maxAllocatedSize : info.blockSize * MAX_TOTAL_BLOCKS;
  if(info.blockSize == 0 || info.blockSize > m_state.maxAllocationSize
     || (maxAllocatedSize + info.blockSize - 1) / info.blockSize > MAX_TOTAL_BLOCKS)
  {
    m_state = {};
    return E_INVALIDARG;
  }

  m_info                  = info;
  m_info.maxAllocatedSize = maxAllocatedSize;

  const uint64_t maxBlocks = (maxAllocatedSize + info.blockSize - 1) / info.blockSize;
  m_state.maxBlocks        = static_cast<uint32_t>(maxBlocks);
  m_state.internalBlockUnits = static_cast<uint32_t>((m_info.blockSize + m_info.minAlignment - 1) / m_info.minAlignment);

  // The largest request an empty block accepts (binary search on a probe
  // allocator: acceptance is monotonic in the size). Requests above it, though
  // below blockSize, get a dedicated block instead of failing after adding a
  // block they cannot use.
  {
    OffsetAllocator::Allocator probe(m_state.internalBlockUnits, 4);
    uint32_t                   lo = 0;
    uint32_t                   hi = m_state.internalBlockUnits;
    while(lo < hi)
    {
      const uint32_t              mid        = lo + (hi - lo + 1) / 2;
      OffsetAllocator::Allocation allocation = probe.allocate(mid);
      if(allocation.offset != OffsetAllocator::Allocation::NO_SPACE)
      {
        probe.free(allocation);
        lo = mid;
      }
      else
      {
        hi = mid - 1;
      }
    }
    m_state.maxBlockUnits = lo;
  }

  if(m_info.keepBlockCount)
  {
    Block block;
    block.offsetAllocator = std::make_unique<OffsetAllocator::Allocator>(m_state.internalBlockUnits, m_info.perBlockAllocations);
    const HRESULT result = NVDX12_FAIL_REPORT(
        createNewBuffer(block.buffer, uint64_t(m_state.internalBlockUnits) * m_info.minAlignment, m_info.minAlignment, 0));
    if(FAILED(result))
    {
      m_info  = {};
      m_state = {};
      return result;
    }

    m_blocks.push_back(std::move(block));

    m_state.activeBlockCount = 1;
    m_state.activeBlockIndex = 0;
  }

  if(m_info.threadSafeBlockAccess && m_blocks.size() < m_state.maxBlocks)
  {
    size_t oldSize = m_blocks.size();

    m_blocks.resize(m_state.maxBlocks);
    m_state.freeBlockIndex = uint32_t(oldSize);
    for(size_t i = oldSize; i < m_state.maxBlocks; i++)
    {
      m_blocks[i].nextFreeIndex = i < (m_state.maxBlocks - 1) ? uint32_t(i + 1) : INVALID_BLOCK_INDEX;
    }
  }

  return S_OK;
}

void BufferSubAllocator::deinit()
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
      report.reservedSize += uint64_t(m_state.internalBlockUnits - storageReport.totalFreeSpace) * m_info.minAlignment;
      report.freeSize += uint64_t(storageReport.totalFreeSpace) * m_info.minAlignment;
    }
    else
    {
      // dedicated blocks
      report.reservedSize += block.buffer.bufferSize;
    }
    report.allocatedSize += block.buffer.bufferSize;
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

  assert(freeBlockIndex < MAX_TOTAL_BLOCKS);

  return freeBlockIndex;
}

void BufferSubAllocator::releaseBlockIndex(uint32_t blockIndex)
{
  // nuke it completely and chain into the linked list of empty blocks
  m_blocks[blockIndex]               = {};
  m_blocks[blockIndex].nextFreeIndex = m_state.freeBlockIndex;
  m_state.freeBlockIndex             = blockIndex;
}

HRESULT BufferSubAllocator::subAllocate(BufferSubAllocation& subAllocation, uint64_t size, uint32_t alignment)
{
  subAllocation = {};

  assert(alignment % MIN_ALIGNMENT == 0);
  assert(alignment >= MIN_ALIGNMENT && alignment <= MAX_ALIGNMENT);
  assert(size <= m_state.maxAllocationSize);

  if(size == 0 || size > m_state.maxAllocationSize)
  {
    return E_INVALIDARG;
  }

  bool alignmentIsPowerOfTwo = (alignment & (alignment - 1)) == 0;

  if(size + m_state.allocatedSize > m_info.maxAllocatedSize)
  {
    return E_OUTOFMEMORY;
  }

  // note: m_state.allocatedSize is only advanced on the success paths below,
  // so a failed allocation does not permanently eat into m_info.maxAllocatedSize

  // adjust the size to account for local alignment
  uint64_t sizeAllocate = size;

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
  const uint64_t allocatorUnits64 = (sizeAllocate + m_info.minAlignment - 1) / m_info.minAlignment;
  const uint32_t allocatorUnits   = static_cast<uint32_t>(std::min<uint64_t>(allocatorUnits64, ~0u));

  // if large use a dedicated block; also when an empty block could not take
  // the request (alignment margin or OffsetAllocator bin rounding), which
  // would otherwise add a block that stays empty and return E_OUTOFMEMORY
  if(size >= m_info.blockSize || allocatorUnits64 > m_state.maxBlockUnits)
  {
    if(m_state.freeBlockIndex == INVALID_BLOCK_INDEX && m_blocks.size() == size_t(m_state.maxBlocks))
    {
      return E_OUTOFMEMORY;
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

    Block&        block = m_blocks[freeBlockIndex];
    const HRESULT result =
        NVDX12_FAIL_REPORT(createNewBuffer(block.buffer, size, std::max(m_info.minAlignment, alignment), freeBlockIndex));
    if(FAILED(result))
    {
      // createBuffer leaves the buffer empty on failure; releaseBlockIndex drops the Block
      releaseBlockIndex(freeBlockIndex);
      return result;
    }

    subAllocation.m_allocation.offset   = 0;
    subAllocation.m_allocation.metadata = OffsetAllocator::Allocation::NO_SPACE;
    subAllocation.m_size                = static_cast<uint32_t>(size);
    subAllocation.m_alignmentMinusOne   = uint16_t(alignment - 1);
    subAllocation.m_block               = uint16_t(freeBlockIndex);
#if !defined(NDEBUG) && !defined(NVDX12_DISABLE_BUFFER_SUB_ALLOCATOR_DEBUG_POINTER)
    subAllocation.m_allocator = this;
#endif

    m_state.allocatedSize += size;

    // dedicated blocks are _not_ thrown into the active block list (m_state.activeBlockIndex)

    return S_OK;
  }


  // else try to find a sub allocation


  // iterate over active blocks to find allocation

  uint32_t activeBlockIndex = m_state.activeBlockIndex;

  while(activeBlockIndex != INVALID_BLOCK_INDEX)
  {
    Block& block = m_blocks[activeBlockIndex];

    // attempt to sub allocate from active blocks

    OffsetAllocator::Allocation allocation = block.offsetAllocator->allocate(allocatorUnits);

    if(allocation.offset != OffsetAllocator::Allocation::NO_SPACE)
    {
      subAllocation.m_allocation        = allocation;
      subAllocation.m_size              = static_cast<uint32_t>(size);
      subAllocation.m_alignmentMinusOne = uint16_t(alignment - 1);
      subAllocation.m_block             = uint16_t(activeBlockIndex);
#if !defined(NDEBUG) && !defined(NVDX12_DISABLE_BUFFER_SUB_ALLOCATOR_DEBUG_POINTER)
      subAllocation.m_allocator = this;
#endif

      m_state.allocatedSize += size;

      return S_OK;
    }

    activeBlockIndex = block.nextActiveIndex;
  }

  // could not find anything

  // if we reached the limit for blocks, bail out
  if(m_state.freeBlockIndex == INVALID_BLOCK_INDEX && m_blocks.size() == size_t(m_state.maxBlocks))
  {
    return E_OUTOFMEMORY;
  }

  {
    // add new block

    uint32_t freeBlockIndex = acquireBlockIndex();

    Block& block = m_blocks[freeBlockIndex];
    block.offsetAllocator = std::make_unique<OffsetAllocator::Allocator>(m_state.internalBlockUnits, m_info.perBlockAllocations);
    const HRESULT result =
        NVDX12_FAIL_REPORT(createNewBuffer(block.buffer, uint64_t(m_state.internalBlockUnits) * m_info.minAlignment,
                                           m_info.minAlignment, freeBlockIndex));
    if(FAILED(result))
    {
      // createBuffer leaves the buffer empty on failure, and the block is not in
      // the active block list yet: handing the index back is enough
      releaseBlockIndex(freeBlockIndex);
      return result;
    }

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
      subAllocation.m_allocation        = allocation;
      subAllocation.m_size              = static_cast<uint32_t>(size);
      subAllocation.m_alignmentMinusOne = uint16_t(alignment - 1);
      subAllocation.m_block             = uint16_t(freeBlockIndex);
#if !defined(NDEBUG) && !defined(NVDX12_DISABLE_BUFFER_SUB_ALLOCATOR_DEBUG_POINTER)
      subAllocation.m_allocator = this;
#endif

      m_state.allocatedSize += size;

      return S_OK;
    }
    else
    {
      return E_OUTOFMEMORY;
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

#if !defined(NDEBUG) && !defined(NVDX12_DISABLE_BUFFER_SUB_ALLOCATOR_DEBUG_POINTER)
  assert(subAllocation.m_allocator == this);
#endif

  OffsetAllocator::Allocator* offsetAllocator = m_blocks[subAllocation.m_block].offsetAllocator.get();

  // dedicated blocks might not have an offset allocator
  if(offsetAllocator)
  {
    offsetAllocator->free(subAllocation.m_allocation);
  }

  m_state.allocatedSize -= subAllocation.m_size;

  // check if dedicated block or empty
  if(!offsetAllocator || offsetAllocator->storageReport().totalFreeSpace == m_state.internalBlockUnits)
  {
    // always free if dedicated
    // and maybe depending on how many blocks to keep
    if(!offsetAllocator || (m_state.activeBlockCount > m_info.keepBlockCount))
    {
      m_info.resourceAllocator->destroyBuffer(m_blocks[subAllocation.m_block].buffer);

      // blocks with OffsetAllocators are counted to active blocks
      if(offsetAllocator)
      {
        m_state.activeBlockCount--;

        // need to remove from linked list of active blocks

        uint32_t selfActiveIndex = subAllocation.m_block;
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

      // nuke it completely and chain into linked list of empty blocks
      releaseBlockIndex(subAllocation.m_block);
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

#if !defined(NDEBUG) && !defined(NVDX12_DISABLE_BUFFER_SUB_ALLOCATOR_DEBUG_POINTER)
  assert(subAllocation.m_allocator == this);
#endif

  // OffsetAllocator's offset is in units of `m_info.minAlignment`; the range
  // covers [offset, offset + size) of the block in every address form.
  return makeBufferRange(m_blocks[subAllocation.m_block].buffer, subAllocation.getOffset(m_info.minAlignment),
                         subAllocation.m_size);
}

const Buffer& BufferSubAllocator::getBlockBuffer(const uint32_t blockIndex) const
{
  assert(isBlockValid(blockIndex) && "invalid blockIndex");
  return m_blocks[blockIndex].buffer;
}

bool BufferSubAllocator::isBlockValid(uint32_t blockIndex) const
{
  if(blockIndex >= m_blocks.size())
    return false;

  return m_blocks[blockIndex].buffer.buffer != nullptr;
}

uint32_t BufferSubAllocator::getBlockCount() const
{
  return uint32_t(m_blocks.size());
}

HRESULT BufferSubAllocator::createNewBuffer(Buffer& buffer, uint64_t size, uint32_t alignment, uint32_t blockIndex)
{
  NVDX12_FAIL_RETURN(
      m_info.resourceAllocator->createBuffer(buffer, size, m_info.usageFlags, m_info.memoryUsage, m_info.allocationFlags,
                                             std::min<uint64_t>(alignment, ResourceAllocator::BUFFER_PLACEMENT_ALIGNMENT)));

  NVDX12_DBG_CUSTOM_NAME(buffer.buffer.Get(),
                         std::string(typeid(*this).name()) + "::" + m_info.debugName + "_" + std::to_string(blockIndex));
  return S_OK;
}

}  // namespace nvdx12


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

    nvdx12::BufferSubAllocation vertex;
    nvdx12::BufferSubAllocation index;
  };

  std::vector<Mesh> meshes;

  nvdx12::ResourceAllocator resourceAllocator;  // EX. initialize somehow

  // create the buffer sub allocator covering the buffers
  nvdx12::BufferSubAllocator bufferSubAllocator;
  bufferSubAllocator.init({.resourceAllocator = &resourceAllocator,
                           .debugName         = "meshes",
                           .usageFlags        = D3D12_RESOURCE_FLAG_NONE,  // vertex / index data, no shader address
                           .memoryUsage       = nvdx12::MemoryUsage::eDevice,
                           .blockSize = 64 * 1024 * 1024});  // requests of blockSize or more get their own block

  for(size_t i = 0; i < meshes.size(); i++)
  {
    Mesh& mesh = meshes[i];
    bufferSubAllocator.subAllocate(mesh.vertex, mesh.vertexSize);
    bufferSubAllocator.subAllocate(mesh.index, mesh.indexSize);

    // upload somehow, e.g. a CopyBufferRegion into subRange(...).buffer at subRange(...).offset
  }

  // later when drawing

  while(true)
  {
    ID3D12GraphicsCommandList7* cmd{};  // per-frame command list, setup state etc.

    for(size_t i = 0; i < meshes.size(); i++)
    {
      Mesh&               mesh        = meshes[i];
      nvdx12::BufferRange vertexRange = bufferSubAllocator.subRange(mesh.vertex);
      nvdx12::BufferRange indexRange  = bufferSubAllocator.subRange(mesh.index);

      // D3D12 binds vertex and index buffers by GPU virtual address, so the
      // range is bound directly (no rebinding of the whole block as in Vulkan)
      D3D12_VERTEX_BUFFER_VIEW vbv{vertexRange.gpuAddress, UINT(vertexRange.range), UINT(sizeof(float) * 4)};
      D3D12_INDEX_BUFFER_VIEW  ibv{indexRange.gpuAddress, UINT(indexRange.range), DXGI_FORMAT_R16_UINT};
      cmd->IASetVertexBuffers(0, 1, &vbv);
      cmd->IASetIndexBuffer(&ibv);

      cmd->DrawIndexedInstanced(uint32_t(mesh.indexSize / sizeof(uint16_t)), 1U, 0U, 0, 0U);
    }
  }
}
