/*
 * Copyright (c) 2014-2025, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2014-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "barriers.hpp"
#include "helpers.hpp"
#include "mipmaps.hpp"

#include <volk.h>

// This mipmap generation relies on blitting.
// For a more sophisticated version using compute shaders that can use fewer
// passes and barriers, please see the nvpro_pyramid library and
// vk_compute_mipmaps sample:
// https://github.com/nvpro-samples/vk_compute_mipmaps/tree/main/nvpro_pyramid

void nvvk::cmdGenerateMipmaps(VkCommandBuffer   cmd,
                              VkImage           image,
                              const VkExtent2D& size,
                              uint32_t          levelCount,
                              uint32_t          layerCount /*= 1*/,
                              VkImageLayout     currentLayout /*= VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL*/)
{
  VkImageSubresourceRange subresourceRange{
      .aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT,
      .baseMipLevel   = 0,
      .levelCount     = 1,
      .baseArrayLayer = 0,
      .layerCount     = layerCount,
  };

  // Transfer the top level image to a layout a Src
  VkImageMemoryBarrier2 barrier = nvvk::makeImageMemoryBarrier(
      {.image = image, .oldLayout = currentLayout, .newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, .subresourceRange = subresourceRange});

  const VkDependencyInfo depInfo{
      .sType                   = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
      .imageMemoryBarrierCount = 1,
      .pImageMemoryBarriers    = &barrier,
  };
  vkCmdPipelineBarrier2(cmd, &depInfo);

  if(levelCount > 1)
  {
    // Transfer remaining mips to DST optimal (Src -> Dst)
    barrier.newLayout                     = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier.dstAccessMask                 = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    barrier.subresourceRange.baseMipLevel = 1;
    barrier.subresourceRange.levelCount   = VK_REMAINING_MIP_LEVELS;

    vkCmdPipelineBarrier2(cmd, &depInfo);
  };

  // Starting mip dimensions
  int32_t mipWidth  = size.width;
  int32_t mipHeight = size.height;

  // Blit region - will be adjusted in the loop
  VkImageBlit2 blitRegion{
      .sType          = VK_STRUCTURE_TYPE_IMAGE_BLIT_2,
      .srcSubresource = {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .layerCount = layerCount},
      .dstSubresource = {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .layerCount = layerCount},
  };

  VkBlitImageInfo2 blitImageInfo{
      .sType          = VK_STRUCTURE_TYPE_BLIT_IMAGE_INFO_2,
      .srcImage       = image,
      .srcImageLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
      .dstImage       = image,
      .dstImageLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
      .regionCount    = 1,
      .pRegions       = &blitRegion,
      .filter         = VK_FILTER_LINEAR,
  };

  // Generating the mip-maps
  for(uint32_t i = 1; i < levelCount; i++)
  {
    blitRegion.srcSubresource.mipLevel = i - 1;
    blitRegion.srcOffsets[1]           = {mipWidth, mipHeight, 1};
    blitRegion.dstSubresource.mipLevel = i;
    blitRegion.dstOffsets[1]           = {mipWidth > 1 ? mipWidth / 2 : 1, mipHeight > 1 ? mipHeight / 2 : 1, 1};
    // Blit from current dimension to half of it
    vkCmdBlitImage2(cmd, &blitImageInfo);

    // Next mip-level
    {
      // Transition the current mip-level into a eTransferSrcOptimal layout, to be used as the source for the next one.
      barrier.subresourceRange.baseMipLevel = i;
      barrier.subresourceRange.levelCount   = 1;
      barrier.oldLayout                     = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
      barrier.newLayout                     = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
      barrier.srcStageMask                  = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
      barrier.srcAccessMask                 = VK_ACCESS_2_TRANSFER_WRITE_BIT;
      barrier.dstStageMask                  = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
      barrier.dstAccessMask                 = VK_ACCESS_2_TRANSFER_READ_BIT;
      vkCmdPipelineBarrier2(cmd, &depInfo);
    }

    // Reducing dimension at each iteration
    if(mipWidth > 1)
    {
      mipWidth /= 2;
    }
    if(mipHeight > 1)
    {
      mipHeight /= 2;
    }
  }

  // Transition all mip-levels (now in VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL) back to currentLayout
  std::tuple<VkPipelineStageFlags2, VkAccessFlags2> transition = nvvk::inferPipelineStageAccessTuple(currentLayout);

  barrier.subresourceRange.baseMipLevel = 0;
  barrier.subresourceRange.levelCount   = VK_REMAINING_MIP_LEVELS;
  barrier.oldLayout                     = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
  barrier.newLayout                     = currentLayout;
  barrier.srcStageMask                  = VK_PIPELINE_STAGE_TRANSFER_BIT;
  barrier.srcAccessMask                 = VK_ACCESS_2_TRANSFER_READ_BIT;
  barrier.dstStageMask                  = std::get<0>(transition);
  barrier.dstAccessMask                 = std::get<1>(transition);
  vkCmdPipelineBarrier2(cmd, &depInfo);
}
