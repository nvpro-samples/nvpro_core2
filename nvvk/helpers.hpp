/*
 * Copyright (c) 2023-2026, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <filesystem>

#include <vulkan/vulkan_core.h>


namespace nvvk {

//-----------------------------------
// Image helpers

// Convert a tiled image to a linear image through vkCmdBlitImage.
// `srcLayout` must match the image's actual current layout when the command buffer is recorded.
// During the blit operation we will temporarily change the layout to VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL
// and then back to the provided layout.
// The `dstImage` and `dstImageMemory` must be cleaned up manually afterwards.
VkResult imageToLinear(VkCommandBuffer  cmd,
                       VkDevice         device,
                       VkPhysicalDevice physicalDevice,
                       VkImage          srcImage,
                       VkExtent2D       size,
                       VkImage&         dstImage,
                       VkDeviceMemory&  dstImageMemory,
                       VkFormat         dstFormat,
                       VkImageLayout    srcLayout = VK_IMAGE_LAYOUT_GENERAL);

// Stores image to a file.
// Extensions must be: .png, .jpg, .jpeg, .bmp (rgba8) or .hdr (rgba32f).
// Undetected extension is treated as PNG.
// Also calls VkMapMemory & vkMapMemory on dstImageMemory during the process.
VkResult saveImageToFile(VkDevice                     device,
                         VkImage                      dstImage,
                         VkDeviceMemory               dstImageMemory,
                         VkExtent2D                   size,
                         const std::filesystem::path& filename,
                         int                          quality = 100);

}  // namespace nvvk