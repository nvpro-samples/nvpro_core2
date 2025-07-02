/*
 * Copyright (c) 2023-2025, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <vector>
#include <tuple>
#include <cassert>

#include <vulkan/vulkan_core.h>

#include "resources.hpp"

//-----------------------------------------------------------------------------
// Helper functions for more concise Vulkan barriers.
//
// * makeImageMemoryBarrier, makeBufferMemoryBarrier, and makeMemoryBarrier
// create corresponding Vk*Barrier2 structs in a single line. When the exact
// masks aren't critical, you can replace stage and access flags with
// nvvk::INFER_BARRIER_PARAMS to infer them from the layout and stage,
// respectively.
//
// * cmdImageMemoryBarrier, cmdBufferMemoryBarrier, and cmdMemoryBarrier
// do the above but also records a vkCmdPipelineBarrier2 call at the same time.
//
// * BarrierContainer can be used to batch together multiple pipeline barriers,
// or to also automatically update nvvk::Image::descriptor::imageLayout.
//
// Constexpr functions are in this header file so that structs can be
// determined at compile-time; `cmd` functions are in the source file.

namespace nvvk {

//--
// Automatically infers appropriate access masks from pipeline stage flags
// `read` parameter determines read (true) or write (false) operations
// Used to simplify barrier creation when exact access masks aren't critical
//
[[nodiscard]] constexpr VkAccessFlags2 inferAccessMaskFromStage(VkPipelineStageFlags2 stage, bool read)
{
  VkAccessFlags2 access = 0;

  if((stage & (VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT | VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT)) != 0)
    access |= read ? VK_ACCESS_2_MEMORY_READ_BIT : VK_ACCESS_2_MEMORY_WRITE_BIT;

  // Handle each possible stage bit
  if((stage
      & (VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT
         | VK_PIPELINE_STAGE_2_MESH_SHADER_BIT_EXT | VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR | VK_PIPELINE_STAGE_2_TASK_SHADER_BIT_EXT
         | VK_PIPELINE_STAGE_2_PRE_RASTERIZATION_SHADERS_BIT | VK_PIPELINE_STAGE_2_TESSELLATION_CONTROL_SHADER_BIT
         | VK_PIPELINE_STAGE_2_TESSELLATION_EVALUATION_SHADER_BIT | VK_PIPELINE_STAGE_2_GEOMETRY_SHADER_BIT))
     != 0)
  {
    access |= read ? VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_UNIFORM_READ_BIT : VK_ACCESS_2_SHADER_WRITE_BIT;
  }

  if((stage & VK_PIPELINE_STAGE_2_HOST_BIT) != 0)
    access |= read ? VK_ACCESS_2_HOST_READ_BIT : VK_ACCESS_2_HOST_WRITE_BIT;

  if((stage & VK_PIPELINE_STAGE_2_TRANSFER_BIT) != 0)
    access |= read ? VK_ACCESS_2_TRANSFER_READ_BIT : VK_ACCESS_2_TRANSFER_WRITE_BIT;

  if((stage & VK_PIPELINE_STAGE_2_VERTEX_ATTRIBUTE_INPUT_BIT) != 0)
    access |= VK_ACCESS_2_VERTEX_ATTRIBUTE_READ_BIT;
  if((stage & VK_PIPELINE_STAGE_2_INDEX_INPUT_BIT) != 0)
    access |= VK_ACCESS_2_INDEX_READ_BIT;
  if((stage & VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT) != 0)
    access |= VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT;

  if((stage & (VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT)) != 0)
    access |= read ? VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT : VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
  if((stage & VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT) != 0)
    access |= read ? VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT : VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;

  if((stage & VK_PIPELINE_STAGE_2_COMMAND_PREPROCESS_BIT_EXT) != 0)
    access |= read ? VK_ACCESS_2_COMMAND_PREPROCESS_READ_BIT_EXT : VK_ACCESS_2_COMMAND_PREPROCESS_WRITE_BIT_EXT;

  if((stage & VK_PIPELINE_STAGE_2_FRAGMENT_SHADING_RATE_ATTACHMENT_BIT_KHR) != 0)
    access |= VK_ACCESS_2_FRAGMENT_SHADING_RATE_ATTACHMENT_READ_BIT_KHR;

  if((stage & VK_PIPELINE_STAGE_2_VIDEO_DECODE_BIT_KHR) != 0)
    access |= read ? VK_ACCESS_2_VIDEO_DECODE_READ_BIT_KHR : VK_ACCESS_2_VIDEO_DECODE_WRITE_BIT_KHR;
  if((stage & VK_PIPELINE_STAGE_2_VIDEO_ENCODE_BIT_KHR) != 0)
    access |= read ? VK_ACCESS_2_VIDEO_ENCODE_READ_BIT_KHR : VK_ACCESS_2_VIDEO_ENCODE_WRITE_BIT_KHR;

  if((stage & VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR) != 0)
    access |= read ? VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR : VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
  if((stage & VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_COPY_BIT_KHR) != 0)
    access |= VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT;

  if((stage & VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR) != 0)
  {
    access |= read ? VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR : 0;
  }

  assert((access != 0 || stage == VK_PIPELINE_STAGE_2_NONE) && "Missing stage implementation");
  return access;
}

//
// Maps image layouts to appropriate pipeline stages and access flags
// Used for synchronizing image state transitions in the pipeline
//
[[nodiscard]] constexpr std::tuple<VkPipelineStageFlags2, VkAccessFlags2> inferPipelineStageAccessTuple(VkImageLayout state)
{
  switch(state)
  {
    case VK_IMAGE_LAYOUT_UNDEFINED:
      return std::make_tuple(VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE);
    case VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL:
      return std::make_tuple(VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                             VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
    case VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL:
      return std::make_tuple(VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT
                                 | VK_PIPELINE_STAGE_2_PRE_RASTERIZATION_SHADERS_BIT | VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                             VK_ACCESS_2_SHADER_READ_BIT);
    case VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL:
      return std::make_tuple(VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
    case VK_IMAGE_LAYOUT_GENERAL:
      return std::make_tuple(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_PRE_RASTERIZATION_SHADERS_BIT
                                 | VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                             VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT | VK_ACCESS_2_SHADER_WRITE_BIT
                                 | VK_ACCESS_2_TRANSFER_WRITE_BIT);
    case VK_IMAGE_LAYOUT_PRESENT_SRC_KHR:
      return std::make_tuple(VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_NONE);

    case VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL:
      return std::make_tuple(VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);

    case VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL:
    case VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL:
    case VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL:
      return std::make_tuple(VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
                             VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT);
    default: {
      assert(false && "Unsupported layout transition!");
      return std::make_tuple(VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT);
    }
  }
}

constexpr uint64_t INFER_BARRIER_PARAMS = ~0ULL;

struct ImageMemoryBarrierParams
{
  VkImage       image     = VK_NULL_HANDLE;
  VkImageLayout oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  VkImageLayout newLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  VkImageSubresourceRange subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS};
  VkPipelineStageFlags2 srcStageMask        = INFER_BARRIER_PARAMS;  // infer from oldLayout
  VkPipelineStageFlags2 dstStageMask        = INFER_BARRIER_PARAMS;  // infer from newLayout
  VkAccessFlags2        srcAccessMask       = INFER_BARRIER_PARAMS;  // infer from stage or layout
  VkAccessFlags2        dstAccessMask       = INFER_BARRIER_PARAMS;  // infer from stage or layout
  uint32_t              srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  uint32_t              dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
};

//
// Creates a standardized image memory barrier for layout transitions
// Handles the complex mapping of layouts to appropriate stage/access flags
// For common cases, consider using cmdTransitionImageLayout instead
//
[[nodiscard]] constexpr VkImageMemoryBarrier2 makeImageMemoryBarrier(const ImageMemoryBarrierParams& params)
{
  VkImageMemoryBarrier2 barrier{
      .sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
      .srcStageMask        = params.srcStageMask,
      .srcAccessMask       = params.srcAccessMask,
      .dstStageMask        = params.dstStageMask,
      .dstAccessMask       = params.dstAccessMask,
      .oldLayout           = params.oldLayout,
      .newLayout           = params.newLayout,
      .srcQueueFamilyIndex = params.srcQueueFamilyIndex,
      .dstQueueFamilyIndex = params.dstQueueFamilyIndex,
      .image               = params.image,
      .subresourceRange    = params.subresourceRange,
  };

  if(params.srcStageMask == INFER_BARRIER_PARAMS && params.srcAccessMask == INFER_BARRIER_PARAMS)
  {
    const auto [srcStageMask, srcAccessMask] = inferPipelineStageAccessTuple(params.oldLayout);
    barrier.srcStageMask                     = srcStageMask;
    barrier.srcAccessMask                    = srcAccessMask;
  }
  else if(params.srcAccessMask == INFER_BARRIER_PARAMS)
  {
    assert(params.srcStageMask != INFER_BARRIER_PARAMS);
    barrier.srcAccessMask = inferAccessMaskFromStage(params.srcStageMask, false);
  }

  if(params.dstStageMask == INFER_BARRIER_PARAMS && params.dstAccessMask == INFER_BARRIER_PARAMS)
  {
    const auto [dstStageMask, dstAccessMask] = inferPipelineStageAccessTuple(params.newLayout);
    barrier.dstStageMask                     = dstStageMask;
    barrier.dstAccessMask                    = dstAccessMask;
  }
  else if(params.dstAccessMask == INFER_BARRIER_PARAMS)
  {
    assert(params.dstStageMask != INFER_BARRIER_PARAMS);
    barrier.dstAccessMask = inferAccessMaskFromStage(params.dstStageMask, false);
  }

  return barrier;
}

//--
// A helper function to transition an image from one layout to another.
// In the pipeline, the image must be in the correct layout to be used, and this function is used to transition the image to the correct layout.
//
void cmdImageMemoryBarrier(VkCommandBuffer cmd, const ImageMemoryBarrierParams& params);

//--
// A helper function to transition an image from one layout to another. Will
// ignore params.image and params.oldLayout and instead use image.image and
// image.descriptor.imageLayout, respectively. params.newLayout will be written
// to image.descriptor.imageLayout
void cmdImageMemoryBarrier(VkCommandBuffer cmd, nvvk::Image& image, const ImageMemoryBarrierParams& params);

struct BufferMemoryBarrierParams
{
  VkBuffer              buffer              = VK_NULL_HANDLE;
  VkPipelineStageFlags2 srcStageMask        = VK_PIPELINE_STAGE_2_NONE;
  VkPipelineStageFlags2 dstStageMask        = VK_PIPELINE_STAGE_2_NONE;
  VkDeviceSize          offset              = 0;
  VkDeviceSize          size                = VK_WHOLE_SIZE;
  VkAccessFlags2        srcAccessMask       = INFER_BARRIER_PARAMS;  // infers from srcStageMask
  VkAccessFlags2        dstAccessMask       = INFER_BARRIER_PARAMS;  // infers from dstStageMask
  uint32_t              srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  uint32_t              dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
};

//--
// Simplifies buffer synchronization by optionally inferring access masks
// Particularly useful for compute/shader to transfer synchronization
// Provide explicit access masks for fine-grained control, or use INFER_BARRIER_PARAMS for automatic inference
//
[[nodiscard]] constexpr VkBufferMemoryBarrier2 makeBufferMemoryBarrier(const BufferMemoryBarrierParams& params)
{
  const VkBufferMemoryBarrier2 bufferBarrier{
      .sType               = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
      .srcStageMask        = params.srcStageMask,
      .srcAccessMask       = params.srcAccessMask != INFER_BARRIER_PARAMS ? params.srcAccessMask :
                                                                            inferAccessMaskFromStage(params.srcStageMask, false),
      .dstStageMask        = params.dstStageMask,
      .dstAccessMask       = params.dstAccessMask != INFER_BARRIER_PARAMS ? params.dstAccessMask :
                                                                            inferAccessMaskFromStage(params.dstStageMask, true),
      .srcQueueFamilyIndex = params.srcQueueFamilyIndex,
      .dstQueueFamilyIndex = params.dstQueueFamilyIndex,
      .buffer              = params.buffer,
      .offset              = params.offset,
      .size                = params.size,
  };

  return bufferBarrier;
}

void cmdBufferMemoryBarrier(VkCommandBuffer commandBuffer, const BufferMemoryBarrierParams& params);

[[nodiscard]] constexpr VkMemoryBarrier2 makeMemoryBarrier(VkPipelineStageFlags2 srcStageMask,
                                                           VkPipelineStageFlags2 dstStageMask,
                                                           VkAccessFlags2 srcAccessMask = INFER_BARRIER_PARAMS,  // Default to infer if not provided
                                                           VkAccessFlags2 dstAccessMask = INFER_BARRIER_PARAMS  // Default to infer if not provided
)
{
  const VkMemoryBarrier2 memoryBarrier{
      .sType        = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
      .srcStageMask = srcStageMask,
      .srcAccessMask = (srcAccessMask != INFER_BARRIER_PARAMS) ? srcAccessMask : inferAccessMaskFromStage(srcStageMask, false),
      .dstStageMask = dstStageMask,
      .dstAccessMask = (dstAccessMask != INFER_BARRIER_PARAMS) ? dstAccessMask : inferAccessMaskFromStage(dstStageMask, true)};
  return memoryBarrier;
}

void cmdMemoryBarrier(VkCommandBuffer       cmd,
                      VkPipelineStageFlags2 srcStageMask,
                      VkPipelineStageFlags2 dstStageMask,
                      VkAccessFlags2        srcAccessMask = INFER_BARRIER_PARAMS,  // Default to infer if not provided
                      VkAccessFlags2        dstAccessMask = INFER_BARRIER_PARAMS   // Default to infer if not provided
);

class BarrierContainer
{
public:
  std::vector<VkMemoryBarrier2>       memoryBarriers;
  std::vector<VkBufferMemoryBarrier2> bufferBarriers;
  std::vector<VkImageMemoryBarrier2>  imageBarriers;

  // Submits all barriers. Does not clear vectors.
  void cmdPipelineBarrier(VkCommandBuffer cmd, VkDependencyFlags dependencyFlags);

  // Overwrites imageBarrier::image with `image.image`.
  // Detects if imageBarrier.newLayout is actually new, skips if not; note that
  // this skip can cause synchronization issues if `image` has just been
  // written and is about to be read or written with the same layout.
  void appendOptionalLayoutTransition(nvvk::Image& image, VkImageMemoryBarrier2 imageBarrier);

  // Clears all vectors.
  void clear();
};

}  // namespace nvvk
