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

//--------------------------------------------------------------------------------------------------
// Default Structs
//
// Usage:
//   VkImageCreateInfo imageCreateInfo = DEFAULT_VkImageCreateInfo;
//   VkImageViewCreateInfo imageViewCreateInfo = DEFAULT_VkImageViewCreateInfo;
//

constexpr VkImageCreateInfo DEFAULT_VkImageCreateInfo{
    .sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
    .pNext         = nullptr,
    .imageType     = VK_IMAGE_TYPE_2D,
    .format        = VK_FORMAT_B8G8R8A8_UNORM,
    .extent        = {1, 1, 1},
    .mipLevels     = 1,
    .arrayLayers   = 1,
    .samples       = VK_SAMPLE_COUNT_1_BIT,
    .tiling        = VK_IMAGE_TILING_OPTIMAL,
    .usage         = {},
    .sharingMode   = VK_SHARING_MODE_EXCLUSIVE,
    .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
};

constexpr VkImageSubresourceRange DEFAULT_VkImageSubresourceRange{
    .aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT,
    .baseMipLevel   = 0,
    .levelCount     = VK_REMAINING_MIP_LEVELS,
    .baseArrayLayer = 0,
    .layerCount     = VK_REMAINING_ARRAY_LAYERS,
};

constexpr VkImageViewCreateInfo DEFAULT_VkImageViewCreateInfo{
    .sType            = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
    .pNext            = nullptr,
    .flags            = {},
    .image            = {},
    .viewType         = VK_IMAGE_VIEW_TYPE_2D,
    .format           = {},
    .components       = {},
    .subresourceRange = DEFAULT_VkImageSubresourceRange,
};

constexpr VkClearDepthStencilValue DEFAULT_VkClearDepthStencilValue{
    .depth   = 1.0f,
    .stencil = 0,
};

constexpr VkRenderingAttachmentInfo DEFAULT_VkRenderingAttachmentInfo{
    .sType              = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
    .pNext              = nullptr,
    .imageView          = VK_NULL_HANDLE,
    .imageLayout        = VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL,
    .resolveMode        = VK_RESOLVE_MODE_NONE,
    .resolveImageView   = VK_NULL_HANDLE,
    .resolveImageLayout = VK_IMAGE_LAYOUT_UNDEFINED,
    .loadOp             = VK_ATTACHMENT_LOAD_OP_CLEAR,
    .storeOp            = VK_ATTACHMENT_STORE_OP_STORE,
    .clearValue         = {},
};

constexpr VkRect2D DEFAULT_VkRect2D(const VkExtent2D& extent)
{
  return {
      .offset = {0, 0},
      .extent = extent,
  };
};

constexpr VkRenderingInfo DEFAULT_VkRenderingInfo{
    .sType                = VK_STRUCTURE_TYPE_RENDERING_INFO,
    .pNext                = nullptr,
    .flags                = {},
    .renderArea           = {},
    .layerCount           = 1,
    .viewMask             = 0,
    .colorAttachmentCount = {},
    .pColorAttachments    = {},
    .pDepthAttachment     = {},
    .pStencilAttachment   = {},
};

constexpr VkSamplerCreateInfo DEFAULT_VkSamplerCreateInfo{
    .sType                   = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
    .pNext                   = nullptr,
    .flags                   = 0,
    .magFilter               = VK_FILTER_LINEAR,
    .minFilter               = VK_FILTER_LINEAR,
    .mipmapMode              = VK_SAMPLER_MIPMAP_MODE_LINEAR,
    .addressModeU            = VK_SAMPLER_ADDRESS_MODE_REPEAT,
    .addressModeV            = VK_SAMPLER_ADDRESS_MODE_REPEAT,
    .addressModeW            = VK_SAMPLER_ADDRESS_MODE_REPEAT,
    .mipLodBias              = 0.0f,
    .anisotropyEnable        = VK_FALSE,
    .maxAnisotropy           = 0.0f,
    .compareEnable           = VK_FALSE,
    .compareOp               = VK_COMPARE_OP_NEVER,
    .minLod                  = 0.0f,
    .maxLod                  = VK_LOD_CLAMP_NONE,
    .borderColor             = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK,
    .unnormalizedCoordinates = VK_FALSE,
};
