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

#pragma once

#include <cassert>
#include <cstdint>
#include <tuple>
#include <vector>

#include "default_structs.hpp"
#include "resources.hpp"

namespace nvdx12 {

/*-------------------------------------------------------------------------------------------------
# Barriers

> Enhanced-barrier helpers (ID3D12GraphicsCommandList7::Barrier), the D3D12
> counterpart of nvvk/barriers.hpp.

- `makeImageMemoryBarrier`, `makeBufferMemoryBarrier` and `makeMemoryBarrier`
  fill a D3D12_TEXTURE_BARRIER, D3D12_BUFFER_BARRIER or D3D12_GLOBAL_BARRIER
  in one line. Sync and access masks left at `INFER_BARRIER_PARAMS` are
  derived from the layouts and stages, as in nvvk.
- `cmdImageMemoryBarrier`, `cmdBufferMemoryBarrier` and `cmdMemoryBarrier`
  record the same barriers. The overloads taking an nvdx12::Image or Buffer
  read the resource and its previous scope from it and track the new one.
- `BarrierContainer` batches barriers of all three kinds into one call.
- `cmdImageBarrier`, `cmdBufferBarrier` and `cmdMemoryBarrier(cmd)` are the
  short forms (D3D12 extras), for direct and compute lists and any pipeline
  stage: an image barrier waits for the image's tracked scope (or the scope
  of its layout) and makes the image available to every stage its new
  layout can be used by, graphics included; see makeImageBarrier for the
  exact barrier. Copy queues take neither layout transitions nor these
  scopes (only COMMON, the copy accesses and SYNC_ALL / SYNC_COPY).

Differences with nvvk that follow from D3D12:
- Barriers are D3D12_TEXTURE_BARRIER / D3D12_BUFFER_BARRIER /
  D3D12_GLOBAL_BARRIER with D3D12_BARRIER_SYNC / D3D12_BARRIER_ACCESS masks
  in place of the Vk*Barrier2 structs and stage / access flags; D3D12
  layouts imply narrower scopes than Vulkan's (see
  inferPipelineStageAccessTuple).
- nvdx12::Image tracks its layout and last scope (`layout`, `stageMask`,
  `accessMask`); the Image overload of cmdImageMemoryBarrier uses the
  tracked source scope unless `params.srcStageMask` is given, and records
  the new one.
- Left out: queue family indices (D3D12 resources have no queue ownership;
  queue-specific layouts such as DIRECT_QUEUE_* express the same intent),
  dependency flags (BarrierContainer::cmdPipelineBarrier takes none), and
  the buffer barrier offset / size (D3D12 requires the whole resource).
- Buffer barriers infer no acceleration-structure access: D3D12 reserves it
  for resources created as acceleration structures, which take explicit
  masks or cmdAccelerationStructureBarrier.

Constexpr functions live in this header so barrier structs can be built at
compile time; the `cmd*` functions are in the source file.

See usage_Barriers() in barriers.cpp.
-------------------------------------------------------------------------------------------------*/

//-----------------------------------------------------------------------------
// Inference

// Access mask implied by a sync mask, for reads (`read`) or writes. Split
// barriers and the shading-rate source need explicit masks.
[[nodiscard]] constexpr D3D12_BARRIER_ACCESS inferAccessMaskFromStage(D3D12_BARRIER_SYNC stage, bool read)
{
  if(stage == D3D12_BARRIER_SYNC_NONE)
  {
    return D3D12_BARRIER_ACCESS_NO_ACCESS;
  }

  const UINT sync = static_cast<UINT>(stage);

  // Both halves of a split barrier carry identical access masks, which
  // inference cannot produce.
  assert((sync & static_cast<UINT>(D3D12_BARRIER_SYNC_SPLIT)) == 0 && "Split barriers require explicit access masks");

  if((sync & (D3D12_BARRIER_SYNC_ALL | D3D12_BARRIER_SYNC_DRAW)) != 0)
  {
    return D3D12_BARRIER_ACCESS_COMMON;
  }

  UINT access = 0;

  constexpr UINT nonRtShaderSync = D3D12_BARRIER_SYNC_VERTEX_SHADING | D3D12_BARRIER_SYNC_PIXEL_SHADING | D3D12_BARRIER_SYNC_COMPUTE_SHADING
                                   | D3D12_BARRIER_SYNC_ALL_SHADING | D3D12_BARRIER_SYNC_NON_PIXEL_SHADING;
  constexpr UINT shaderSync = nonRtShaderSync | D3D12_BARRIER_SYNC_RAYTRACING;
  if((sync & shaderSync) != 0)
    access |= read ? (D3D12_BARRIER_ACCESS_SHADER_RESOURCE | D3D12_BARRIER_ACCESS_UNORDERED_ACCESS) :
                     D3D12_BARRIER_ACCESS_UNORDERED_ACCESS;

  // Traces read acceleration structures, as nvvk infers for the ray-tracing
  // stage (global barriers; texture and buffer barriers mask it out).
  if((sync & D3D12_BARRIER_SYNC_RAYTRACING) != 0 && read)
    access |= D3D12_BARRIER_ACCESS_RAYTRACING_ACCELERATION_STRUCTURE_READ;

  // CONSTANT_BUFFER is not compatible with SYNC_RAYTRACING alone. With
  // SYNC_NON_PIXEL_SHADING the runtime allows it (the legacy
  // NON_PIXEL_SHADER_RESOURCE state maps to exactly this combination).
  if((sync & nonRtShaderSync) != 0 && read)
    access |= D3D12_BARRIER_ACCESS_CONSTANT_BUFFER;

  if((sync & (D3D12_BARRIER_SYNC_VERTEX_SHADING | D3D12_BARRIER_SYNC_ALL_SHADING | D3D12_BARRIER_SYNC_NON_PIXEL_SHADING)) != 0 && read)
    access |= D3D12_BARRIER_ACCESS_VERTEX_BUFFER;

  if((sync & D3D12_BARRIER_SYNC_INDEX_INPUT) != 0)
    access |= D3D12_BARRIER_ACCESS_INDEX_BUFFER;

  if((sync & D3D12_BARRIER_SYNC_EXECUTE_INDIRECT) != 0)
    access |= D3D12_BARRIER_ACCESS_INDIRECT_ARGUMENT;

  if((sync & D3D12_BARRIER_SYNC_DEPTH_STENCIL) != 0)
    access |= read ? D3D12_BARRIER_ACCESS_DEPTH_STENCIL_READ : D3D12_BARRIER_ACCESS_DEPTH_STENCIL_WRITE;

  if((sync & D3D12_BARRIER_SYNC_RENDER_TARGET) != 0)
    access |= D3D12_BARRIER_ACCESS_RENDER_TARGET;

  if((sync & D3D12_BARRIER_SYNC_COPY) != 0)
    access |= read ? D3D12_BARRIER_ACCESS_COPY_SOURCE : D3D12_BARRIER_ACCESS_COPY_DEST;
  if((sync & D3D12_BARRIER_SYNC_RESOLVE) != 0)
    access |= read ? D3D12_BARRIER_ACCESS_RESOLVE_SOURCE : D3D12_BARRIER_ACCESS_RESOLVE_DEST;

  if((sync & D3D12_BARRIER_SYNC_CLEAR_UNORDERED_ACCESS_VIEW) != 0)
    access |= D3D12_BARRIER_ACCESS_UNORDERED_ACCESS;

  if((sync & D3D12_BARRIER_SYNC_VIDEO_DECODE) != 0)
    access |= read ? D3D12_BARRIER_ACCESS_VIDEO_DECODE_READ : D3D12_BARRIER_ACCESS_VIDEO_DECODE_WRITE;
  if((sync & D3D12_BARRIER_SYNC_VIDEO_PROCESS) != 0)
    access |= read ? D3D12_BARRIER_ACCESS_VIDEO_PROCESS_READ : D3D12_BARRIER_ACCESS_VIDEO_PROCESS_WRITE;
  if((sync & D3D12_BARRIER_SYNC_VIDEO_ENCODE) != 0)
    access |= read ? D3D12_BARRIER_ACCESS_VIDEO_ENCODE_READ : D3D12_BARRIER_ACCESS_VIDEO_ENCODE_WRITE;

  if((sync & (D3D12_BARRIER_SYNC_BUILD_RAYTRACING_ACCELERATION_STRUCTURE | D3D12_BARRIER_SYNC_COPY_RAYTRACING_ACCELERATION_STRUCTURE)) != 0)
  {
    access |= read ? D3D12_BARRIER_ACCESS_RAYTRACING_ACCELERATION_STRUCTURE_READ :
                     D3D12_BARRIER_ACCESS_RAYTRACING_ACCELERATION_STRUCTURE_WRITE;
  }
  // Builds also read their inputs (vertices, indices, instances) as shader
  // resources and access the scratch as unordered access, which a later
  // build reusing the scratch must wait for.
  if((sync & D3D12_BARRIER_SYNC_BUILD_RAYTRACING_ACCELERATION_STRUCTURE) != 0)
    access |= read ? (D3D12_BARRIER_ACCESS_SHADER_RESOURCE | D3D12_BARRIER_ACCESS_UNORDERED_ACCESS) :
                     D3D12_BARRIER_ACCESS_UNORDERED_ACCESS;

  if((sync & D3D12_BARRIER_SYNC_EMIT_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO) != 0)
    access |= read ? D3D12_BARRIER_ACCESS_RAYTRACING_ACCELERATION_STRUCTURE_READ : D3D12_BARRIER_ACCESS_UNORDERED_ACCESS;

  // ConvertLinearAlgebraMatrix reads its source as a shader resource and
  // writes its destination as an unordered access.
  if((sync & D3D12_BARRIER_SYNC_CONVERT_LINEAR_ALGEBRA_MATRIX) != 0)
    access |= read ? D3D12_BARRIER_ACCESS_SHADER_RESOURCE : D3D12_BARRIER_ACCESS_UNORDERED_ACCESS;

  assert((access != 0 || (sync & ~static_cast<UINT>(D3D12_BARRIER_SYNC_SPLIT)) == 0) && "Missing stage implementation");

  return static_cast<D3D12_BARRIER_ACCESS>(access);
}

// Destination access of an inferred barrier: the reads and the writes of
// `stage`, so that both reads and writes after the barrier see earlier writes.
// COMMON and NO_ACCESS take no other bits; both sides infer them alike.
[[nodiscard]] constexpr D3D12_BARRIER_ACCESS inferDstAccessMaskFromStage(D3D12_BARRIER_SYNC stage)
{
  const D3D12_BARRIER_ACCESS reads = inferAccessMaskFromStage(stage, true);
  if(reads == D3D12_BARRIER_ACCESS_COMMON || reads == D3D12_BARRIER_ACCESS_NO_ACCESS)
  {
    return reads;
  }
  return static_cast<D3D12_BARRIER_ACCESS>(static_cast<UINT>(reads) | static_cast<UINT>(inferAccessMaskFromStage(stage, false)));
}

// Sync and access scope implied by a layout, for all queue types and
// pipeline stages. D3D12 layouts name their use, so most cases carry one
// access bit; COMMON / PRESENT and GENERIC_READ (and their queue-scoped
// variants) return ACCESS_COMMON, every access the layout permits.
[[nodiscard]] constexpr std::tuple<D3D12_BARRIER_SYNC, D3D12_BARRIER_ACCESS> inferPipelineStageAccessTuple(D3D12_BARRIER_LAYOUT state)
{
  switch(state)
  {
    // Contents are meaningless, so the access is NO_ACCESS. The sync stays
    // SYNC_ALL rather than SYNC_NONE: SYNC_NONE promises that nothing else
    // touches the resource in the same ExecuteCommandLists scope, which
    // discarding a texture mid-frame violates. Callers that know the barrier
    // is the resource's first use in the scope pass SYNC_NONE explicitly.
    case D3D12_BARRIER_LAYOUT_UNDEFINED:
      return std::make_tuple(D3D12_BARRIER_SYNC_ALL, D3D12_BARRIER_ACCESS_NO_ACCESS);

    // COMMON and PRESENT share the value 0, so a swapchain image about to be
    // presented cannot be narrowed to the render-target scope.
    case D3D12_BARRIER_LAYOUT_COMMON:  // == D3D12_BARRIER_LAYOUT_PRESENT
    case D3D12_BARRIER_LAYOUT_DIRECT_QUEUE_COMMON:
    case D3D12_BARRIER_LAYOUT_COMPUTE_QUEUE_COMMON:
      return std::make_tuple(D3D12_BARRIER_SYNC_ALL, D3D12_BARRIER_ACCESS_COMMON);

    // Readable by anything, like VK_IMAGE_LAYOUT_GENERAL without the writes.
    case D3D12_BARRIER_LAYOUT_GENERIC_READ:
    case D3D12_BARRIER_LAYOUT_DIRECT_QUEUE_GENERIC_READ:
    case D3D12_BARRIER_LAYOUT_COMPUTE_QUEUE_GENERIC_READ:
    case D3D12_BARRIER_LAYOUT_DIRECT_QUEUE_GENERIC_READ_COMPUTE_QUEUE_ACCESSIBLE:
      return std::make_tuple(D3D12_BARRIER_SYNC_ALL, D3D12_BARRIER_ACCESS_COMMON);

    // Like VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL.
    case D3D12_BARRIER_LAYOUT_RENDER_TARGET:
      return std::make_tuple(D3D12_BARRIER_SYNC_RENDER_TARGET, D3D12_BARRIER_ACCESS_RENDER_TARGET);

    // The closest thing to a writable VK_IMAGE_LAYOUT_GENERAL.
    // ClearUnorderedAccessView* writes through this layout too, from its own sync scope.
    case D3D12_BARRIER_LAYOUT_UNORDERED_ACCESS:
    case D3D12_BARRIER_LAYOUT_DIRECT_QUEUE_UNORDERED_ACCESS:
    case D3D12_BARRIER_LAYOUT_COMPUTE_QUEUE_UNORDERED_ACCESS:
      return std::make_tuple(D3D12_BARRIER_SYNC_ALL_SHADING | D3D12_BARRIER_SYNC_CLEAR_UNORDERED_ACCESS_VIEW,
                             D3D12_BARRIER_ACCESS_UNORDERED_ACCESS);

    // Like VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL: depth/stencil
    // tests read in this layout as well, so both access bits are included.
    case D3D12_BARRIER_LAYOUT_DEPTH_STENCIL_WRITE:
      return std::make_tuple(D3D12_BARRIER_SYNC_DEPTH_STENCIL,
                             D3D12_BARRIER_ACCESS_DEPTH_STENCIL_WRITE | D3D12_BARRIER_ACCESS_DEPTH_STENCIL_READ);
    // Like VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL.
    case D3D12_BARRIER_LAYOUT_DEPTH_STENCIL_READ:
      return std::make_tuple(D3D12_BARRIER_SYNC_DEPTH_STENCIL, D3D12_BARRIER_ACCESS_DEPTH_STENCIL_READ);

    // Like VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL.
    case D3D12_BARRIER_LAYOUT_SHADER_RESOURCE:
    case D3D12_BARRIER_LAYOUT_DIRECT_QUEUE_SHADER_RESOURCE:
    case D3D12_BARRIER_LAYOUT_COMPUTE_QUEUE_SHADER_RESOURCE:
      return std::make_tuple(D3D12_BARRIER_SYNC_ALL_SHADING, D3D12_BARRIER_ACCESS_SHADER_RESOURCE);

    // Like VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL and VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL.
    case D3D12_BARRIER_LAYOUT_COPY_SOURCE:
    case D3D12_BARRIER_LAYOUT_DIRECT_QUEUE_COPY_SOURCE:
    case D3D12_BARRIER_LAYOUT_COMPUTE_QUEUE_COPY_SOURCE:
      return std::make_tuple(D3D12_BARRIER_SYNC_COPY, D3D12_BARRIER_ACCESS_COPY_SOURCE);
    case D3D12_BARRIER_LAYOUT_COPY_DEST:
    case D3D12_BARRIER_LAYOUT_DIRECT_QUEUE_COPY_DEST:
    case D3D12_BARRIER_LAYOUT_COMPUTE_QUEUE_COPY_DEST:
      return std::make_tuple(D3D12_BARRIER_SYNC_COPY, D3D12_BARRIER_ACCESS_COPY_DEST);

    // MSAA resolves; Vulkan runs these through the transfer stage and has no
    // layouts of its own for them.
    case D3D12_BARRIER_LAYOUT_RESOLVE_SOURCE:
      return std::make_tuple(D3D12_BARRIER_SYNC_RESOLVE, D3D12_BARRIER_ACCESS_RESOLVE_SOURCE);
    case D3D12_BARRIER_LAYOUT_RESOLVE_DEST:
      return std::make_tuple(D3D12_BARRIER_SYNC_RESOLVE, D3D12_BARRIER_ACCESS_RESOLVE_DEST);

    // Like VK_IMAGE_LAYOUT_FRAGMENT_SHADING_RATE_ATTACHMENT_OPTIMAL_KHR; the
    // pair inferAccessMaskFromStage cannot produce, since SYNC_PIXEL_SHADING
    // on its own means an ordinary shader read.
    case D3D12_BARRIER_LAYOUT_SHADING_RATE_SOURCE:
      return std::make_tuple(D3D12_BARRIER_SYNC_PIXEL_SHADING, D3D12_BARRIER_ACCESS_SHADING_RATE_SOURCE);

    case D3D12_BARRIER_LAYOUT_VIDEO_DECODE_READ:
      return std::make_tuple(D3D12_BARRIER_SYNC_VIDEO_DECODE, D3D12_BARRIER_ACCESS_VIDEO_DECODE_READ);
    case D3D12_BARRIER_LAYOUT_VIDEO_DECODE_WRITE:
      return std::make_tuple(D3D12_BARRIER_SYNC_VIDEO_DECODE, D3D12_BARRIER_ACCESS_VIDEO_DECODE_WRITE);
    case D3D12_BARRIER_LAYOUT_VIDEO_PROCESS_READ:
      return std::make_tuple(D3D12_BARRIER_SYNC_VIDEO_PROCESS, D3D12_BARRIER_ACCESS_VIDEO_PROCESS_READ);
    case D3D12_BARRIER_LAYOUT_VIDEO_PROCESS_WRITE:
      return std::make_tuple(D3D12_BARRIER_SYNC_VIDEO_PROCESS, D3D12_BARRIER_ACCESS_VIDEO_PROCESS_WRITE);
    case D3D12_BARRIER_LAYOUT_VIDEO_ENCODE_READ:
      return std::make_tuple(D3D12_BARRIER_SYNC_VIDEO_ENCODE, D3D12_BARRIER_ACCESS_VIDEO_ENCODE_READ);
    case D3D12_BARRIER_LAYOUT_VIDEO_ENCODE_WRITE:
      return std::make_tuple(D3D12_BARRIER_SYNC_VIDEO_ENCODE, D3D12_BARRIER_ACCESS_VIDEO_ENCODE_WRITE);

    default: {
      assert(false && "Unsupported layout transition!");
      return std::make_tuple(D3D12_BARRIER_SYNC_ALL, D3D12_BARRIER_ACCESS_COMMON);
    }
  }
}

// Access of a texture barrier side whose stage is given: the stage's access
// (inferAccessMaskFromStage for the source side, inferDstAccessMaskFromStage
// for the destination side `dst`) restricted to what `layout` permits
// (inferPipelineStageAccessTuple), the layout's access when they share no
// bit, ACCESS_COMMON for the layouts that permit every access (COMMON,
// GENERIC_READ), and NO_ACCESS for UNDEFINED or SYNC_NONE.
[[nodiscard]] constexpr D3D12_BARRIER_ACCESS inferTextureAccessMaskFromStage(D3D12_BARRIER_SYNC stage, D3D12_BARRIER_LAYOUT layout, bool dst)
{
  if(stage == D3D12_BARRIER_SYNC_NONE || layout == D3D12_BARRIER_LAYOUT_UNDEFINED)
  {
    return D3D12_BARRIER_ACCESS_NO_ACCESS;
  }
  const D3D12_BARRIER_ACCESS layoutAccess = std::get<1>(inferPipelineStageAccessTuple(layout));
  if(layoutAccess == D3D12_BARRIER_ACCESS_COMMON)
  {
    return D3D12_BARRIER_ACCESS_COMMON;
  }
  const D3D12_BARRIER_ACCESS stageAccess = dst ? inferDstAccessMaskFromStage(stage) : inferAccessMaskFromStage(stage, false);
  const UINT access = static_cast<UINT>(stageAccess) & static_cast<UINT>(layoutAccess);
  return access != 0 ? static_cast<D3D12_BARRIER_ACCESS>(access) : layoutAccess;
}

// Access of a buffer barrier side whose stage is given: the stage's access
// (inferAccessMaskFromStage for the source side, inferDstAccessMaskFromStage
// for the destination side `dst`) without the acceleration-structure bits,
// which D3D12 reserves for resources created as acceleration structures;
// NO_ACCESS when nothing else remains (a copy of acceleration structures does
// not touch the buffer).
[[nodiscard]] constexpr D3D12_BARRIER_ACCESS inferBufferAccessMaskFromStage(D3D12_BARRIER_SYNC stage, bool dst)
{
  const D3D12_BARRIER_ACCESS access = dst ? inferDstAccessMaskFromStage(stage) : inferAccessMaskFromStage(stage, false);
  if(access == D3D12_BARRIER_ACCESS_COMMON || access == D3D12_BARRIER_ACCESS_NO_ACCESS)
  {
    return access;
  }
  constexpr UINT accelerationStructure = D3D12_BARRIER_ACCESS_RAYTRACING_ACCELERATION_STRUCTURE_READ
                                         | D3D12_BARRIER_ACCESS_RAYTRACING_ACCELERATION_STRUCTURE_WRITE;
  const UINT rest = static_cast<UINT>(access) & ~accelerationStructure;
  return rest != 0 ? static_cast<D3D12_BARRIER_ACCESS>(rest) : D3D12_BARRIER_ACCESS_NO_ACCESS;
}

//-----------------------------------------------------------------------------
// Image (texture) barriers

constexpr uint64_t INFER_BARRIER_PARAMS = ~0ULL;

struct ImageMemoryBarrierParams
{
  ID3D12Resource*                 image{};
  D3D12_BARRIER_LAYOUT            oldLayout        = D3D12_BARRIER_LAYOUT_UNDEFINED;
  D3D12_BARRIER_LAYOUT            newLayout        = D3D12_BARRIER_LAYOUT_UNDEFINED;
  D3D12_BARRIER_SUBRESOURCE_RANGE subresourceRange = DEFAULT_D3D12_BARRIER_SUBRESOURCE_RANGE;
  // A D3D12_BARRIER_SYNC each, or INFER_BARRIER_PARAMS
  uint64_t srcStageMask = INFER_BARRIER_PARAMS;  // infer from oldLayout
  uint64_t dstStageMask = INFER_BARRIER_PARAMS;  // infer from newLayout
  // A D3D12_BARRIER_ACCESS each, or INFER_BARRIER_PARAMS
  uint64_t                    srcAccessMask = INFER_BARRIER_PARAMS;  // infer from stage or layout
  uint64_t                    dstAccessMask = INFER_BARRIER_PARAMS;  // infer from stage or layout
  D3D12_TEXTURE_BARRIER_FLAGS flags         = D3D12_TEXTURE_BARRIER_FLAG_NONE;
};

// Texture barrier for a layout transition; scopes left at
// INFER_BARRIER_PARAMS come from the layouts (both masks) or the stage and
// layout (access only, inferTextureAccessMaskFromStage). For the common
// whole-image case, cmdImageBarrier is shorter.
[[nodiscard]] constexpr D3D12_TEXTURE_BARRIER makeImageMemoryBarrier(const ImageMemoryBarrierParams& params)
{
  // An access mask without its stage mask leaves the INFER sentinel in the sync field.
  assert(params.srcStageMask != INFER_BARRIER_PARAMS || params.srcAccessMask == INFER_BARRIER_PARAMS);
  assert(params.dstStageMask != INFER_BARRIER_PARAMS || params.dstAccessMask == INFER_BARRIER_PARAMS);

  D3D12_TEXTURE_BARRIER barrier{.SyncBefore   = static_cast<D3D12_BARRIER_SYNC>(params.srcStageMask),
                                .SyncAfter    = static_cast<D3D12_BARRIER_SYNC>(params.dstStageMask),
                                .AccessBefore = static_cast<D3D12_BARRIER_ACCESS>(params.srcAccessMask),
                                .AccessAfter  = static_cast<D3D12_BARRIER_ACCESS>(params.dstAccessMask),
                                .LayoutBefore = params.oldLayout,
                                .LayoutAfter  = params.newLayout,
                                .pResource    = params.image,
                                .Subresources = params.subresourceRange,
                                .Flags        = params.flags};

  if(params.srcStageMask == INFER_BARRIER_PARAMS && params.srcAccessMask == INFER_BARRIER_PARAMS)
  {
    const auto [srcStageMask, srcAccessMask] = inferPipelineStageAccessTuple(params.oldLayout);
    barrier.SyncBefore                       = srcStageMask;
    barrier.AccessBefore                     = srcAccessMask;
  }
  else if(params.srcAccessMask == INFER_BARRIER_PARAMS)
  {
    barrier.AccessBefore =
        inferTextureAccessMaskFromStage(static_cast<D3D12_BARRIER_SYNC>(params.srcStageMask), params.oldLayout, false);
  }

  if(params.dstStageMask == INFER_BARRIER_PARAMS && params.dstAccessMask == INFER_BARRIER_PARAMS)
  {
    const auto [dstStageMask, dstAccessMask] = inferPipelineStageAccessTuple(params.newLayout);
    barrier.SyncAfter                        = dstStageMask;
    barrier.AccessAfter                      = dstAccessMask;
  }
  else if(params.dstAccessMask == INFER_BARRIER_PARAMS)
  {
    barrier.AccessAfter =
        inferTextureAccessMaskFromStage(static_cast<D3D12_BARRIER_SYNC>(params.dstStageMask), params.newLayout, true);
  }

  return barrier;
}

// Records the texture barrier of makeImageMemoryBarrier.
void cmdImageMemoryBarrier(ID3D12GraphicsCommandList7* cmd, const ImageMemoryBarrierParams& params);

// Source scope of the next barrier on `image`: the tracked one, or the scope
// of its layout (inferPipelineStageAccessTuple) when nothing is tracked but
// the layout is defined, as for an image written by code that does not
// track or created in a defined layout.
[[nodiscard]] inline std::tuple<D3D12_BARRIER_SYNC, D3D12_BARRIER_ACCESS> imageSourceScope(const Image& image)
{
  if(image.stageMask == D3D12_BARRIER_SYNC_NONE && image.layout != D3D12_BARRIER_LAYOUT_UNDEFINED)
  {
    return inferPipelineStageAccessTuple(image.layout);
  }
  return std::make_tuple(image.stageMask, image.accessMask);
}

// Same, taking the resource and old layout from `image`, and the source
// scopes too (imageSourceScope) unless `params.srcStageMask` is given;
// `image` then tracks the new layout and scopes. params.image and params.oldLayout are ignored.
// The image tracks one layout, so `params.subresourceRange` must cover all of it.
void cmdImageMemoryBarrier(ID3D12GraphicsCommandList7* cmd, Image& image, const ImageMemoryBarrierParams& params);

// Destination scope of the short forms: the scope of `layout`
// (inferPipelineStageAccessTuple: every stage and access the layout can be
// used by, e.g. SYNC_ALL_SHADING for SHADER_RESOURCE, which covers the
// vertex, pixel, compute and ray-tracing stages). UNDEFINED gives SYNC_ALL /
// NO_ACCESS: SYNC_NONE would forbid any later barrier on the image in the
// same ExecuteCommandLists scope.
[[nodiscard]] constexpr std::tuple<D3D12_BARRIER_SYNC, D3D12_BARRIER_ACCESS> layoutDestinationScope(D3D12_BARRIER_LAYOUT layout)
{
  return inferPipelineStageAccessTuple(layout);
}

// The barrier cmdImageBarrier records (D3D12 extra), every subresource of
// `image` from image.layout to `newLayout`:
// - SyncBefore / AccessBefore: imageSourceScope(image), i.e. the scope the
//   image tracks (the SyncAfter / AccessAfter of its previous barrier,
//   recorded by any helper of this file, the uploaders and the built-in
//   kernels), else the full scope of image.layout, else (UNDEFINED and
//   untracked) SYNC_NONE / NO_ACCESS;
// - SyncAfter / AccessAfter: layoutDestinationScope(newLayout);
// - Flags: DISCARD when leaving UNDEFINED (no content to keep).
[[nodiscard]] inline D3D12_TEXTURE_BARRIER makeImageBarrier(const Image& image, D3D12_BARRIER_LAYOUT newLayout)
{
  const auto [syncBefore, accessBefore] = imageSourceScope(image);
  const auto [syncAfter, accessAfter]   = layoutDestinationScope(newLayout);
  return D3D12_TEXTURE_BARRIER{
      .SyncBefore   = syncBefore,
      .SyncAfter    = syncAfter,
      .AccessBefore = accessBefore,
      .AccessAfter  = accessAfter,
      .LayoutBefore = image.layout,
      .LayoutAfter  = newLayout,
      .pResource    = image.image.Get(),
      .Subresources = DEFAULT_D3D12_BARRIER_SUBRESOURCE_RANGE,
      .Flags = image.layout == D3D12_BARRIER_LAYOUT_UNDEFINED ? D3D12_TEXTURE_BARRIER_FLAG_DISCARD : D3D12_TEXTURE_BARRIER_FLAG_NONE,
  };
}

// Short form (D3D12 extra): records makeImageBarrier(image, newLayout) and
// tracks its LayoutAfter / SyncAfter / AccessAfter in `image`. Because the
// destination scope is the whole of the new layout's, any later use the
// layout allows (a pixel shader as well as a dispatch or DispatchRays
// reading SHADER_RESOURCE) is ordered after it, and the next short-form
// transition waits for all of them.
// Records nothing when `image` already is in `newLayout` (or is null): a
// hazard inside one layout (UAV write then UAV read or write) needs
// cmdMemoryBarrier or cmdImageMemoryBarrier instead.
void cmdImageBarrier(ID3D12GraphicsCommandList7* cmd, Image& image, D3D12_BARRIER_LAYOUT newLayout);

//-----------------------------------------------------------------------------
// Buffer barriers

struct BufferMemoryBarrierParams
{
  ID3D12Resource*    buffer{};
  D3D12_BARRIER_SYNC srcStageMask  = D3D12_BARRIER_SYNC_NONE;
  D3D12_BARRIER_SYNC dstStageMask  = D3D12_BARRIER_SYNC_NONE;
  uint64_t           srcAccessMask = INFER_BARRIER_PARAMS;  // a D3D12_BARRIER_ACCESS, or inferred from srcStageMask
  uint64_t           dstAccessMask = INFER_BARRIER_PARAMS;  // a D3D12_BARRIER_ACCESS, or inferred from dstStageMask
};

// Buffer barrier over the whole resource (D3D12 requires Offset 0 and Size
// UINT64_MAX); access masks left at INFER_BARRIER_PARAMS come from the stages
// (inferBufferAccessMaskFromStage).
[[nodiscard]] constexpr D3D12_BUFFER_BARRIER makeBufferMemoryBarrier(const BufferMemoryBarrierParams& params)
{
  return D3D12_BUFFER_BARRIER{
      .SyncBefore = params.srcStageMask,
      .SyncAfter  = params.dstStageMask,
      .AccessBefore = params.srcAccessMask != INFER_BARRIER_PARAMS ? static_cast<D3D12_BARRIER_ACCESS>(params.srcAccessMask) :
                                                                     inferBufferAccessMaskFromStage(params.srcStageMask, false),
      .AccessAfter = params.dstAccessMask != INFER_BARRIER_PARAMS ? static_cast<D3D12_BARRIER_ACCESS>(params.dstAccessMask) :
                                                                    inferBufferAccessMaskFromStage(params.dstStageMask, true),
      .pResource = params.buffer,
      .Offset    = 0,
      .Size      = UINT64_MAX,
  };
}

// Records the buffer barrier of makeBufferMemoryBarrier.
void cmdBufferMemoryBarrier(ID3D12GraphicsCommandList7* cmd, const BufferMemoryBarrierParams& params);

// Same, taking the resource from `buffer` and the source scopes too unless
// `params.srcStageMask` is given (SYNC_ALL / ACCESS_COMMON when `buffer`
// tracks none); `buffer` then tracks the new scopes. params.buffer is ignored.
void cmdBufferMemoryBarrier(ID3D12GraphicsCommandList7* cmd, Buffer& buffer, const BufferMemoryBarrierParams& params);

// Unordered-access scope of the short buffer barrier: every shader stage
// (SYNC_ALL_SHADING: vertex, pixel, compute, ray tracing) and
// ClearUnorderedAccessView*.
constexpr D3D12_BARRIER_SYNC kUnorderedAccessSync = D3D12_BARRIER_SYNC_ALL_SHADING | D3D12_BARRIER_SYNC_CLEAR_UNORDERED_ACCESS_VIEW;

// The barrier cmdBufferBarrier records (D3D12 extra): UNORDERED_ACCESS
// writes and reads in kUnorderedAccessSync before, the same after.
[[nodiscard]] inline D3D12_BUFFER_BARRIER makeBufferBarrier(const Buffer& buffer)
{
  return makeBufferMemoryBarrier({.buffer        = buffer.buffer.Get(),
                                  .srcStageMask  = kUnorderedAccessSync,
                                  .dstStageMask  = kUnorderedAccessSync,
                                  .srcAccessMask = D3D12_BARRIER_ACCESS_UNORDERED_ACCESS,
                                  .dstAccessMask = D3D12_BARRIER_ACCESS_UNORDERED_ACCESS});
}

// Short form (D3D12 extra): UAV-to-UAV ordering of one buffer between
// shader invocations of any stage (makeBufferBarrier). It does not cover
// copies, indirect arguments or acceleration-structure builds (use
// cmdBufferMemoryBarrier or cmdMemoryBarrier), and it neither reads nor
// updates the buffer's tracked scope.
void cmdBufferBarrier(ID3D12GraphicsCommandList7* cmd, const Buffer& buffer);

//-----------------------------------------------------------------------------
// Global (memory) barriers

// Global barrier; access masks left at INFER_BARRIER_PARAMS come from the stages.
[[nodiscard]] constexpr D3D12_GLOBAL_BARRIER makeMemoryBarrier(D3D12_BARRIER_SYNC srcStageMask,
                                                               D3D12_BARRIER_SYNC dstStageMask,
                                                               uint64_t srcAccessMask = INFER_BARRIER_PARAMS,  // a D3D12_BARRIER_ACCESS, or inferred
                                                               uint64_t dstAccessMask = INFER_BARRIER_PARAMS  // a D3D12_BARRIER_ACCESS, or inferred
)
{
  return D3D12_GLOBAL_BARRIER{
      .SyncBefore   = srcStageMask,
      .SyncAfter    = dstStageMask,
      .AccessBefore = srcAccessMask != INFER_BARRIER_PARAMS ? static_cast<D3D12_BARRIER_ACCESS>(srcAccessMask) :
                                                              inferAccessMaskFromStage(srcStageMask, false),
      .AccessAfter  = dstAccessMask != INFER_BARRIER_PARAMS ? static_cast<D3D12_BARRIER_ACCESS>(dstAccessMask) :
                                                              inferDstAccessMaskFromStage(dstStageMask),
  };
}

// Records the global barrier of makeMemoryBarrier.
void cmdMemoryBarrier(ID3D12GraphicsCommandList7* cmd,
                      D3D12_BARRIER_SYNC          srcStageMask,
                      D3D12_BARRIER_SYNC          dstStageMask,
                      uint64_t srcAccessMask = INFER_BARRIER_PARAMS,  // a D3D12_BARRIER_ACCESS, or inferred
                      uint64_t dstAccessMask = INFER_BARRIER_PARAMS   // a D3D12_BARRIER_ACCESS, or inferred
);

// Short form (D3D12 extra): every prior access on the queue visible to
// every later one (SYNC_ALL / ACCESS_COMMON on both sides). It changes no
// layout and no tracked scope; it is the barrier for hazards inside one
// layout, which cmdImageBarrier skips.
void cmdMemoryBarrier(ID3D12GraphicsCommandList7* cmd);

//-----------------------------------------------------------------------------
// Batching

class BarrierContainer
{
public:
  std::vector<D3D12_GLOBAL_BARRIER>  memoryBarriers;
  std::vector<D3D12_BUFFER_BARRIER>  bufferBarriers;
  std::vector<D3D12_TEXTURE_BARRIER> imageBarriers;

  // Records every barrier in one Barrier call. Does not clear the vectors.
  void cmdPipelineBarrier(ID3D12GraphicsCommandList7* cmd) const;

  // Appends `imageBarrier` with pResource set to `image.image` and
  // tracks the new layout, unless `image` already is in LayoutAfter. The
  // skip can hide a hazard when `image` was just written and is read or
  // written again in the same layout.
  void appendOptionalLayoutTransition(Image& image, D3D12_TEXTURE_BARRIER imageBarrier);

  // Clears all vectors.
  void clear();
};

}  // namespace nvdx12
