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

#pragma once

#include <cmath>
#include <algorithm>

#include "vulkan/vulkan_core.h"


namespace nvvk {

inline uint32_t mipLevels(uint32_t extent)
{
  return static_cast<uint32_t>(std::floor(std::log2(extent))) + 1;
}

inline uint32_t mipLevels(VkExtent2D extent)
{
  return static_cast<uint32_t>(std::floor(std::log2(std::max(extent.width, extent.height)))) + 1;
}

inline uint32_t mipLevels(VkExtent3D extent)
{
  return static_cast<uint32_t>(std::floor(std::log2(std::max(std::max(extent.width, extent.height), extent.depth)))) + 1;
}

// To get the number of mip levels, use the helper functions
//      uint32_t levelCount = nvvk::mipLevels(extent);
//
// The current layout of the image is the layout of the image before the mipmaps are generated.
void cmdGenerateMipmaps(VkCommandBuffer   cmd,                // Command buffer to record the command
                        VkImage           image,              // Image to generate mipmaps for
                        const VkExtent2D& size,               // Size of the image
                        uint32_t          levelCount,         // Level is the mip level to generate
                        uint32_t          layerCount    = 1,  // Number of layers in the image (2D: 1, 3D: Z, Cube: 6)
                        VkImageLayout     currentLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

}  // namespace nvvk