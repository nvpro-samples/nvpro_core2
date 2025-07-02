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

#include "barriers.hpp"

#include <volk.h>

namespace nvvk {

//-----------------------------------------------------------------------------
// Non-constexpr functions
// This separation also allows us to avoid including volk.h in the header.

void cmdImageMemoryBarrier(VkCommandBuffer cmd, const ImageMemoryBarrierParams& params)
{
  VkImageMemoryBarrier2 barrier = makeImageMemoryBarrier(params);

  const VkDependencyInfo depInfo{.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &barrier};

  vkCmdPipelineBarrier2(cmd, &depInfo);
}

void cmdImageMemoryBarrier(VkCommandBuffer cmd, nvvk::Image& image, const ImageMemoryBarrierParams& params)
{
  ImageMemoryBarrierParams localParams = params;
  localParams.image                    = image.image;
  localParams.oldLayout                = image.descriptor.imageLayout;

  cmdImageMemoryBarrier(cmd, localParams);

  image.descriptor.imageLayout = params.newLayout;
}

void cmdBufferMemoryBarrier(VkCommandBuffer commandBuffer, const BufferMemoryBarrierParams& params)
{
  VkBufferMemoryBarrier2 bufferBarrier = makeBufferMemoryBarrier(params);
  const VkDependencyInfo depInfo{
      .sType                    = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
      .bufferMemoryBarrierCount = 1,
      .pBufferMemoryBarriers    = &bufferBarrier,
  };

  vkCmdPipelineBarrier2(commandBuffer, &depInfo);
}

void cmdMemoryBarrier(VkCommandBuffer       cmd,
                      VkPipelineStageFlags2 srcStageMask,
                      VkPipelineStageFlags2 dstStageMask,
                      VkAccessFlags2        srcAccessMask /* = INFER_BARRIER_PARAMS */,
                      VkAccessFlags2        dstAccessMask /* = INFER_BARRIER_PARAMS */)
{
  const VkMemoryBarrier2 memoryBarrier = makeMemoryBarrier(srcStageMask, dstStageMask, srcAccessMask, dstAccessMask);

  const VkDependencyInfo depInfo{
      .sType              = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
      .memoryBarrierCount = 1,
      .pMemoryBarriers    = &memoryBarrier,
  };

  vkCmdPipelineBarrier2(cmd, &depInfo);
}

//-----------------------------------------------------------------------------
// BarrierContainer implementation

void BarrierContainer::cmdPipelineBarrier(VkCommandBuffer cmd, VkDependencyFlags dependencyFlags)
{
  if(memoryBarriers.empty() && bufferBarriers.empty() && imageBarriers.empty())
    return;

  const VkDependencyInfo depInfo{
      .sType                    = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
      .dependencyFlags          = dependencyFlags,
      .memoryBarrierCount       = static_cast<uint32_t>(memoryBarriers.size()),
      .pMemoryBarriers          = memoryBarriers.data(),
      .bufferMemoryBarrierCount = static_cast<uint32_t>(bufferBarriers.size()),
      .pBufferMemoryBarriers    = bufferBarriers.data(),
      .imageMemoryBarrierCount  = static_cast<uint32_t>(imageBarriers.size()),
      .pImageMemoryBarriers     = imageBarriers.data(),
  };
  vkCmdPipelineBarrier2(cmd, &depInfo);
}

void BarrierContainer::appendOptionalLayoutTransition(nvvk::Image& image, VkImageMemoryBarrier2 imageBarrier)
{
  if(image.descriptor.imageLayout == imageBarrier.newLayout)
  {
    return;
  }

  imageBarrier.image = image.image;
  imageBarriers.push_back(imageBarrier);

  image.descriptor.imageLayout = imageBarrier.newLayout;
}

void BarrierContainer::clear()
{
  memoryBarriers.clear();
  bufferBarriers.clear();
  imageBarriers.clear();
}

}  // namespace nvvk
