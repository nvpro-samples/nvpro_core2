/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "commands.hpp"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <string>

#include "barriers.hpp"
#include "check_error.hpp"
#include "debug_util.hpp"
#include "formats.hpp"
#include "queue.hpp"

#include <nvutils/logger.hpp>

namespace nvdx12 {

//--------------------------------------------------------------------------------------------------
// Dispatch and buffers

void cmdDispatchIndirect(ID3D12GraphicsCommandList* cmd, ID3D12CommandSignature* signature, const Buffer& args, uint64_t byteOffset)
{
  cmd->ExecuteIndirect(signature, 1, args.buffer.Get(), byteOffset, nullptr, 0);
}

bool cmdUpdateBuffer(ID3D12GraphicsCommandList* cmd, UploadRing& ring, const Buffer& dst, uint64_t dstOffset, uint64_t size, const void* data)
{
  if(!dst.buffer)
  {
    return false;
  }
  if(size == 0)
  {
    return true;
  }
  UploadSpan span = ring.allocate(size, 16);
  if(!span)
  {
    return false;
  }
  memcpy(span.cpu, data, size);
  cmd->CopyBufferRegion(dst.buffer.Get(), dstOffset, span.resource, span.offset, size);
  return true;
}

void cmdCopyBuffer(ID3D12GraphicsCommandList* cmd, const Buffer& src, uint64_t srcOffset, const Buffer& dst, uint64_t dstOffset, uint64_t size)
{
  if(size == 0 || !src.buffer || !dst.buffer)
  {
    return;
  }
  cmd->CopyBufferRegion(dst.buffer.Get(), dstOffset, src.buffer.Get(), srcOffset, size);
}

//--------------------------------------------------------------------------------------------------
// Images

static D3D12_TEXTURE_COPY_LOCATION subresourceLocation(ID3D12Resource* resource, uint32_t subresource)
{
  D3D12_TEXTURE_COPY_LOCATION location{};
  location.pResource        = resource;
  location.Type             = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
  location.SubresourceIndex = subresource;
  return location;
}

static D3D12_TEXTURE_COPY_LOCATION footprintLocation(ID3D12Resource* resource, const D3D12_PLACED_SUBRESOURCE_FOOTPRINT& footprint)
{
  D3D12_TEXTURE_COPY_LOCATION location{};
  location.pResource       = resource;
  location.Type            = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
  location.PlacedFootprint = footprint;
  return location;
}

// Depth-stencil and multisampled images only copy whole subresources: no
// source box, a destination offset of 0 and the same sample count on both sides.
static bool copiesWholeSubresources(const Image& image)
{
  return image.sampleCount > 1 || (image.flags & D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL) != 0;
}

void cmdCopyImage(ID3D12GraphicsCommandList7* cmd, Image& src, Image& dst)
{
  if(!src.image || !dst.image)
  {
    return;
  }
  const bool identical = src.format == dst.format && src.extent.width == dst.extent.width
                         && src.extent.height == dst.extent.height && src.extent.depth == dst.extent.depth
                         && src.mipLevels == dst.mipLevels && src.arrayLayers == dst.arrayLayers
                         && src.dimension == dst.dimension && src.sampleCount == dst.sampleCount;
  const uint32_t mips   = std::min(src.mipLevels, dst.mipLevels);
  const uint32_t layers = std::min(src.arrayLayers, dst.arrayLayers);

  // The region both levels hold: a null box would copy the whole source
  // level, out of bounds of a smaller destination.
  const auto levelExtent = [](uint32_t size, uint32_t mip) { return std::max(1u, size >> mip); };
  const auto commonBox   = [&](uint32_t mip) {
    return D3D12_BOX{
          .left   = 0,
          .top    = 0,
          .front  = 0,
          .right  = std::min(levelExtent(src.extent.width, mip), levelExtent(dst.extent.width, mip)),
          .bottom = std::min(levelExtent(src.extent.height, mip), levelExtent(dst.extent.height, mip)),
          .back   = std::min(levelExtent(src.extent.depth, mip), levelExtent(dst.extent.depth, mip)),
    };
  };
  const auto coversSourceLevel = [&](const D3D12_BOX& box, uint32_t mip) {
    return box.right == levelExtent(src.extent.width, mip) && box.bottom == levelExtent(src.extent.height, mip)
           && box.back == levelExtent(src.extent.depth, mip);
  };
  if(!identical && (copiesWholeSubresources(src) || copiesWholeSubresources(dst)))
  {
    bool whole = src.sampleCount == dst.sampleCount;
    for(uint32_t mip = 0; mip < mips && whole; mip++)
    {
      whole = coversSourceLevel(commonBox(mip), mip);
    }
    if(!whole)
    {
      LOGE(
          "cmdCopyImage: depth-stencil and multisampled images copy whole subresources; the destination levels "
          "must hold the source's, with the same sample count. Nothing copied\n");
      return;
    }
  }

  cmdImageBarrier(cmd, src, D3D12_BARRIER_LAYOUT_COPY_SOURCE);
  cmdImageBarrier(cmd, dst, D3D12_BARRIER_LAYOUT_COPY_DEST);
  if(identical)
  {
    cmd->CopyResource(dst.image.Get(), src.image.Get());
    return;
  }
  for(uint32_t mip = 0; mip < mips; mip++)
  {
    // A box that covers the whole source level is passed as null, which
    // depth-stencil and multisampled copies require (they take no box).
    const D3D12_BOX box         = commonBox(mip);
    const bool      wholeSource = coversSourceLevel(box, mip);
    for(uint32_t layer = 0; layer < layers; layer++)
    {
      const D3D12_TEXTURE_COPY_LOCATION srcLoc = subresourceLocation(src.image.Get(), src.subresourceIndex(mip, layer));
      const D3D12_TEXTURE_COPY_LOCATION dstLoc = subresourceLocation(dst.image.Get(), dst.subresourceIndex(mip, layer));
      cmd->CopyTextureRegion(&dstLoc, 0, 0, 0, &srcLoc, wholeSource ? nullptr : &box);
    }
  }
}

void cmdCopyImageRegion(ID3D12GraphicsCommandList7* cmd,
                        Image&                      src,
                        uint32_t                    srcMip,
                        Image&                      dst,
                        uint32_t                    dstMip,
                        const D3D12_BOX*            srcBox,
                        uint32_t                    dstX,
                        uint32_t                    dstY,
                        uint32_t                    dstZ)
{
  if(!src.image || !dst.image)
  {
    return;
  }
  if((copiesWholeSubresources(src) || copiesWholeSubresources(dst))
     && (srcBox != nullptr || dstX != 0 || dstY != 0 || dstZ != 0 || src.sampleCount != dst.sampleCount))
  {
    LOGE(
        "cmdCopyImageRegion: depth-stencil and multisampled images copy whole subresources (null srcBox, "
        "destination offset 0, same sample count). Nothing copied\n");
    return;
  }
  cmdImageBarrier(cmd, src, D3D12_BARRIER_LAYOUT_COPY_SOURCE);
  cmdImageBarrier(cmd, dst, D3D12_BARRIER_LAYOUT_COPY_DEST);
  const D3D12_TEXTURE_COPY_LOCATION srcLoc = subresourceLocation(src.image.Get(), src.subresourceIndex(srcMip));
  const D3D12_TEXTURE_COPY_LOCATION dstLoc = subresourceLocation(dst.image.Get(), dst.subresourceIndex(dstMip));
  cmd->CopyTextureRegion(&dstLoc, dstX, dstY, dstZ, &srcLoc, srcBox);
}

void cmdCopyImageToBuffer(ID3D12GraphicsCommandList7* cmd, Image& src, const Buffer& dst, uint64_t dstOffset, uint32_t mipLevel, uint32_t arrayLayer)
{
  if(!src.image || !dst.buffer)
  {
    return;
  }
  assert((dstOffset % D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT) == 0 && "buffer offset must be 512-byte aligned");
  cmdImageBarrier(cmd, src, D3D12_BARRIER_LAYOUT_COPY_SOURCE);
  const SubresourceFootprint fp = imageSubresourceFootprint(src, mipLevel, arrayLayer, dstOffset);
  const D3D12_TEXTURE_COPY_LOCATION srcLoc = subresourceLocation(src.image.Get(), src.subresourceIndex(mipLevel, arrayLayer));
  const D3D12_TEXTURE_COPY_LOCATION dstLoc = footprintLocation(dst.buffer.Get(), fp.footprint);
  cmd->CopyTextureRegion(&dstLoc, 0, 0, 0, &srcLoc, nullptr);
}

void cmdCopyBufferToImage(ID3D12GraphicsCommandList7* cmd, const Buffer& src, uint64_t srcOffset, Image& dst, uint32_t mipLevel, uint32_t arrayLayer)
{
  if(!src.buffer || !dst.image)
  {
    return;
  }
  assert((srcOffset % D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT) == 0 && "buffer offset must be 512-byte aligned");
  cmdImageBarrier(cmd, dst, D3D12_BARRIER_LAYOUT_COPY_DEST);
  const SubresourceFootprint        fp     = imageSubresourceFootprint(dst, mipLevel, arrayLayer, srcOffset);
  const D3D12_TEXTURE_COPY_LOCATION srcLoc = footprintLocation(src.buffer.Get(), fp.footprint);
  const D3D12_TEXTURE_COPY_LOCATION dstLoc = subresourceLocation(dst.image.Get(), dst.subresourceIndex(mipLevel, arrayLayer));
  cmd->CopyTextureRegion(&dstLoc, 0, 0, 0, &srcLoc, nullptr);
}

//--------------------------------------------------------------------------------------------------
// Command contexts

HRESULT CommandContext::init(ID3D12Device* device, const char* name, D3D12_COMMAND_LIST_TYPE type)
{
  NVDX12_FAIL_RETURN(device->CreateCommandAllocator(type, IID_PPV_ARGS(&allocator)));
  ComPtr<ID3D12Device4> device4;
  NVDX12_FAIL_RETURN(device->QueryInterface(IID_PPV_ARGS(&device4)));
  NVDX12_FAIL_RETURN(device4->CreateCommandList1(0, type, D3D12_COMMAND_LIST_FLAG_NONE, IID_PPV_ARGS(&list)));
  if(name)
  {
    setDebugName(allocator.Get(), std::string(name) + " allocator");
    setDebugName(list.Get(), name);
  }
  return S_OK;
}

void CommandContext::deinit()
{
  list.Reset();
  allocator.Reset();
}

HRESULT CommandContext::begin(ID3D12PipelineState* initialState)
{
  NVDX12_FAIL_RETURN(allocator->Reset());
  NVDX12_FAIL_RETURN(list->Reset(allocator.Get(), initialState));
  return S_OK;
}

HRESULT CommandContext::close()
{
  return NVDX12_FAIL_REPORT(list->Close());
}

ComPtr<ID3D12CommandAllocator> createTransientCommandPool(ID3D12Device* device, D3D12_COMMAND_LIST_TYPE type)
{
  ComPtr<ID3D12CommandAllocator> allocator;
  if(FAILED(NVDX12_FAIL_REPORT(device->CreateCommandAllocator(type, IID_PPV_ARGS(&allocator)))))
  {
    return nullptr;
  }
  NVDX12_DBG_NAME(allocator);
  return allocator;
}

HRESULT beginSingleTimeCommands(ComPtr<ID3D12GraphicsCommandList7>& cmd, ID3D12Device* device, ID3D12CommandAllocator* allocator, D3D12_COMMAND_LIST_TYPE type)
{
  ComPtr<ID3D12Device4> device4;
  NVDX12_FAIL_RETURN(device->QueryInterface(IID_PPV_ARGS(&device4)));
  NVDX12_FAIL_RETURN(device4->CreateCommandList1(0, type, D3D12_COMMAND_LIST_FLAG_NONE, IID_PPV_ARGS(&cmd)));
  NVDX12_FAIL_RETURN(cmd->Reset(allocator, nullptr));
  return S_OK;
}

ComPtr<ID3D12GraphicsCommandList7> createSingleTimeCommands(ID3D12Device* device, ID3D12CommandAllocator* allocator, D3D12_COMMAND_LIST_TYPE type)
{
  ComPtr<ID3D12GraphicsCommandList7> cmd;
  NVDX12_CHECK(beginSingleTimeCommands(cmd, device, allocator, type));
  return cmd;
}

HRESULT endSingleTimeCommands(ID3D12GraphicsCommandList* cmd, ID3D12Device* device, ID3D12CommandAllocator* allocator, ID3D12CommandQueue* queue)
{
  NVDX12_FAIL_RETURN(cmd->Close());

  // A fence of its own, as nvvk's VkFence: the queue may have no QueueTimeline.
  ComPtr<ID3D12Fence> fence;
  NVDX12_FAIL_RETURN(createTimelineSemaphore(device, 0, fence));

  ID3D12CommandList* lists[] = {cmd};
  queue->ExecuteCommandLists(1, lists);
  NVDX12_FAIL_RETURN(queue->Signal(fence.Get(), 1));
  NVDX12_FAIL_RETURN(SemaphoreState::makeFixed(fence, 1).wait(UINT64_MAX));
  // The list completed: what it recorded can go, as vkFreeCommandBuffers returns the buffer to its pool.
  return NVDX12_FAIL_REPORT(allocator->Reset());
}

ScopeCommandList::ScopeCommandList(ID3D12Device* device, SubmitAndWait submitAndWait, const DescriptorHeap* heap)
    : m_submitAndWait(std::move(submitAndWait))
{
  if(FAILED(NVDX12_FAIL_REPORT(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&m_allocator)))))
  {
    return;
  }
  setDebugName(m_allocator.Get(), "nvdx12::ScopeCommandList allocator");
  if(FAILED(beginSingleTimeCommands(m_cmd, device, m_allocator.Get())))
  {
    m_cmd.Reset();
    return;
  }
  setDebugName(m_cmd.Get(), "nvdx12::ScopeCommandList");
  if(heap)
  {
    heap->cmdBind(m_cmd.Get());
  }
}

ScopeCommandList::~ScopeCommandList()
{
  if(!m_cmd)
  {
    return;
  }
  // A list that failed to close cannot be executed.
  if(SUCCEEDED(NVDX12_FAIL_REPORT(m_cmd->Close())) && m_submitAndWait)
  {
    m_submitAndWait(m_cmd.Get());
  }
}

}  // namespace nvdx12

//--------------------------------------------------------------------------------------------------
// Usage example
//--------------------------------------------------------------------------------------------------
[[maybe_unused]] static void usage_Commands()
{
  ID3D12Device*          device = nullptr;  // EX: from nvdx12::Context::getDevice()
  nvdx12::QueueTimeline* queue  = nullptr;  // EX: the context's direct queue
  nvdx12::DescriptorHeap heap;              // EX: initialized by the application
  nvdx12::UploadRing     ring;              // EX: initialized by the application
  nvdx12::Buffer         deviceBuffer, readback;
  nvdx12::Image          image, copy;

  // Per-frame recording: one CommandContext per frame in flight.
  nvdx12::CommandContext frameCmd;
  NVDX12_CHECK(frameCmd.init(device, "frame"));
  NVDX12_CHECK(frameCmd.begin());
  heap.cmdBind(frameCmd.list.Get());
  const uint32_t header[4] = {1, 2, 3, 4};
  nvdx12::cmdUpdateBuffer(frameCmd.list.Get(), ring, deviceBuffer, 0, sizeof(header), header);
  nvdx12::cmdCopyImage(frameCmd.list.Get(), image, copy);
  NVDX12_CHECK(frameCmd.close());
  // ... queue->submit(frameCmd.list.Get()) ...

  // Single-time commands: the list closes, executes and completes when `cmd`
  // leaves the scope.
  {
    nvdx12::ScopeCommandList cmd(device, [queue](ID3D12CommandList* list) { queue->submitAndWait(list); }, &heap);
    nvdx12::cmdCopyBuffer(cmd, deviceBuffer, readback);
    nvdx12::cmdCopyImageToBuffer(cmd, image, readback, 0);
  }
  // ... read `readback.mapping` ...

  // The same on a caller-owned allocator and queue, as with nvvk.
  {
    nvdx12::ComPtr<ID3D12CommandAllocator> transientPool =
        nvdx12::createTransientCommandPool(device, D3D12_COMMAND_LIST_TYPE_DIRECT);
    nvdx12::ComPtr<ID3D12GraphicsCommandList7> cmd = nvdx12::createSingleTimeCommands(device, transientPool.Get());
    nvdx12::cmdCopyBuffer(cmd.Get(), deviceBuffer, readback);
    NVDX12_CHECK(nvdx12::endSingleTimeCommands(cmd.Get(), device, transientPool.Get(), queue->getQueue()));
  }

  frameCmd.deinit();
}
