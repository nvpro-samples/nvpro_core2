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

#include "barriers.hpp"

#include <array>

namespace nvdx12 {

//--------------------------------------------------------------------------------------------------
// Single-barrier groups

static void cmdBarrier(ID3D12GraphicsCommandList7* cmd, const D3D12_TEXTURE_BARRIER& barrier)
{
  const D3D12_BARRIER_GROUP group{.Type = D3D12_BARRIER_TYPE_TEXTURE, .NumBarriers = 1, .pTextureBarriers = &barrier};
  cmd->Barrier(1, &group);
}

static void cmdBarrier(ID3D12GraphicsCommandList7* cmd, const D3D12_BUFFER_BARRIER& barrier)
{
  const D3D12_BARRIER_GROUP group{.Type = D3D12_BARRIER_TYPE_BUFFER, .NumBarriers = 1, .pBufferBarriers = &barrier};
  cmd->Barrier(1, &group);
}

static void cmdBarrier(ID3D12GraphicsCommandList7* cmd, const D3D12_GLOBAL_BARRIER& barrier)
{
  const D3D12_BARRIER_GROUP group{.Type = D3D12_BARRIER_TYPE_GLOBAL, .NumBarriers = 1, .pGlobalBarriers = &barrier};
  cmd->Barrier(1, &group);
}

//--------------------------------------------------------------------------------------------------
// Image barriers

void cmdImageMemoryBarrier(ID3D12GraphicsCommandList7* cmd, const ImageMemoryBarrierParams& params)
{
  cmdBarrier(cmd, makeImageMemoryBarrier(params));
}

void cmdImageMemoryBarrier(ID3D12GraphicsCommandList7* cmd, Image& image, const ImageMemoryBarrierParams& params)
{
  ImageMemoryBarrierParams localParams = params;
  localParams.image                    = image.image.Get();
  localParams.oldLayout                = image.layout;
  // The image's source scope, unless the caller gives one (as nvvk uses params).
  if(params.srcStageMask == INFER_BARRIER_PARAMS)
  {
    const auto [srcStageMask, srcAccessMask] = imageSourceScope(image);
    localParams.srcStageMask                 = srcStageMask;
    localParams.srcAccessMask                = srcAccessMask;
  }

  const D3D12_TEXTURE_BARRIER barrier = makeImageMemoryBarrier(localParams);
  cmdBarrier(cmd, barrier);

  image.layout     = barrier.LayoutAfter;
  image.stageMask  = barrier.SyncAfter;
  image.accessMask = barrier.AccessAfter;
}

void cmdImageBarrier(ID3D12GraphicsCommandList7* cmd, Image& image, D3D12_BARRIER_LAYOUT newLayout)
{
  if(!image.image || image.layout == newLayout)
  {
    return;
  }
  const D3D12_TEXTURE_BARRIER barrier = makeImageBarrier(image, newLayout);
  cmdBarrier(cmd, barrier);
  image.layout     = barrier.LayoutAfter;
  image.stageMask  = barrier.SyncAfter;
  image.accessMask = barrier.AccessAfter;
}

//--------------------------------------------------------------------------------------------------
// Buffer barriers

void cmdBufferMemoryBarrier(ID3D12GraphicsCommandList7* cmd, const BufferMemoryBarrierParams& params)
{
  cmdBarrier(cmd, makeBufferMemoryBarrier(params));
}

void cmdBufferMemoryBarrier(ID3D12GraphicsCommandList7* cmd, Buffer& buffer, const BufferMemoryBarrierParams& params)
{
  BufferMemoryBarrierParams localParams = params;
  localParams.buffer                    = buffer.buffer.Get();
  // The caller's source scope, else the tracked one. With neither, the
  // buffer may still have been written by code that does not track (such as
  // cmdBufferBarrier), so every prior access is waited for.
  if(params.srcStageMask == D3D12_BARRIER_SYNC_NONE)
  {
    const bool tracked        = buffer.stageMask != D3D12_BARRIER_SYNC_NONE;
    localParams.srcStageMask  = tracked ? buffer.stageMask : D3D12_BARRIER_SYNC_ALL;
    localParams.srcAccessMask = tracked ? uint64_t(buffer.accessMask) : uint64_t(D3D12_BARRIER_ACCESS_COMMON);
  }

  const D3D12_BUFFER_BARRIER barrier = makeBufferMemoryBarrier(localParams);
  cmdBarrier(cmd, barrier);

  buffer.stageMask  = barrier.SyncAfter;
  buffer.accessMask = barrier.AccessAfter;
}

void cmdBufferBarrier(ID3D12GraphicsCommandList7* cmd, const Buffer& buffer)
{
  if(!buffer.buffer)
  {
    return;
  }
  cmdBarrier(cmd, makeBufferBarrier(buffer));
}

//--------------------------------------------------------------------------------------------------
// Global barriers

void cmdMemoryBarrier(ID3D12GraphicsCommandList7* cmd,
                      D3D12_BARRIER_SYNC          srcStageMask,
                      D3D12_BARRIER_SYNC          dstStageMask,
                      uint64_t                    srcAccessMask /* = INFER_BARRIER_PARAMS */,
                      uint64_t                    dstAccessMask /* = INFER_BARRIER_PARAMS */)
{
  cmdBarrier(cmd, makeMemoryBarrier(srcStageMask, dstStageMask, srcAccessMask, dstAccessMask));
}

void cmdMemoryBarrier(ID3D12GraphicsCommandList7* cmd)
{
  cmdBarrier(cmd, makeMemoryBarrier(D3D12_BARRIER_SYNC_ALL, D3D12_BARRIER_SYNC_ALL, D3D12_BARRIER_ACCESS_COMMON,
                                    D3D12_BARRIER_ACCESS_COMMON));
}

//--------------------------------------------------------------------------------------------------
// BarrierContainer

void BarrierContainer::cmdPipelineBarrier(ID3D12GraphicsCommandList7* cmd) const
{
  std::array<D3D12_BARRIER_GROUP, 3> groups{};
  uint32_t                           numGroups = 0;

  if(!memoryBarriers.empty())
  {
    groups[numGroups++] = D3D12_BARRIER_GROUP{.Type            = D3D12_BARRIER_TYPE_GLOBAL,
                                              .NumBarriers     = static_cast<UINT32>(memoryBarriers.size()),
                                              .pGlobalBarriers = memoryBarriers.data()};
  }
  if(!bufferBarriers.empty())
  {
    groups[numGroups++] = D3D12_BARRIER_GROUP{.Type            = D3D12_BARRIER_TYPE_BUFFER,
                                              .NumBarriers     = static_cast<UINT32>(bufferBarriers.size()),
                                              .pBufferBarriers = bufferBarriers.data()};
  }
  if(!imageBarriers.empty())
  {
    groups[numGroups++] = D3D12_BARRIER_GROUP{.Type             = D3D12_BARRIER_TYPE_TEXTURE,
                                              .NumBarriers      = static_cast<UINT32>(imageBarriers.size()),
                                              .pTextureBarriers = imageBarriers.data()};
  }
  if(numGroups > 0)
  {
    cmd->Barrier(numGroups, groups.data());
  }
}

void BarrierContainer::appendOptionalLayoutTransition(Image& image, D3D12_TEXTURE_BARRIER imageBarrier)
{
  if(image.layout == imageBarrier.LayoutAfter)
  {
    return;
  }
  imageBarrier.pResource = image.image.Get();
  imageBarriers.push_back(imageBarrier);

  image.layout     = imageBarrier.LayoutAfter;
  image.stageMask  = imageBarrier.SyncAfter;
  image.accessMask = imageBarrier.AccessAfter;
}

void BarrierContainer::clear()
{
  memoryBarriers.clear();
  bufferBarriers.clear();
  imageBarriers.clear();
}

}  // namespace nvdx12

//--------------------------------------------------------------------------------------------------
// Usage example
//--------------------------------------------------------------------------------------------------
[[maybe_unused]] static void usage_Barriers()
{
  ID3D12GraphicsCommandList7* cmd = nullptr;  // EX: the frame's command list
  nvdx12::Image               image;   // EX: created by nvdx12::ResourceAllocator, layout tracked in image.layout
  nvdx12::Buffer              buffer;  // EX: written by one dispatch, read by the next

  // Short forms: the tracked scope before, the new layout's whole scope after.
  nvdx12::cmdImageBarrier(cmd, image, D3D12_BARRIER_LAYOUT_UNORDERED_ACCESS);
  // ... dispatch writing `image` and `buffer` ...
  nvdx12::cmdBufferBarrier(cmd, buffer);  // UAV write -> UAV read of the next dispatch
  nvdx12::cmdImageBarrier(cmd, image, D3D12_BARRIER_LAYOUT_SHADER_RESOURCE);
  // ... dispatch, DispatchRays or draw sampling `image` ...
  nvdx12::cmdMemoryBarrier(cmd);  // everything before visible to everything after

  // Explicit scopes with inferred access masks, as in nvvk.
  nvdx12::cmdMemoryBarrier(cmd, D3D12_BARRIER_SYNC_COMPUTE_SHADING, D3D12_BARRIER_SYNC_COPY);
  nvdx12::cmdImageMemoryBarrier(cmd, image, {.newLayout = D3D12_BARRIER_LAYOUT_COPY_SOURCE});
  nvdx12::cmdBufferMemoryBarrier(
      cmd, buffer, {.dstStageMask = D3D12_BARRIER_SYNC_RAYTRACING, .dstAccessMask = D3D12_BARRIER_ACCESS_SHADER_RESOURCE});

  // Several barriers in one call.
  nvdx12::BarrierContainer barriers;
  barriers.memoryBarriers.push_back(nvdx12::makeMemoryBarrier(D3D12_BARRIER_SYNC_COPY, D3D12_BARRIER_SYNC_ALL_SHADING));
  barriers.appendOptionalLayoutTransition(
      image, nvdx12::makeImageMemoryBarrier({.oldLayout = image.layout, .newLayout = D3D12_BARRIER_LAYOUT_SHADER_RESOURCE}));
  barriers.cmdPipelineBarrier(cmd);
  barriers.clear();
}
