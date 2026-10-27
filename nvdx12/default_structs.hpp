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

#pragma once

#include "basic_types.hpp"

/*-------------------------------------------------------------------------------------------------
# Default structs

The D3D12 counterparts of nvvk/default_structs.hpp: fully initialized
descriptions to copy and adjust.

```cpp
D3D12_SAMPLER_DESC desc = DEFAULT_D3D12_SAMPLER_DESC;
desc.Filter            = D3D12_FILTER_MIN_MAG_MIP_POINT;

D3D12_RESOURCE_DESC1 imageDesc = DEFAULT_D3D12_RESOURCE_DESC1;
imageDesc.Width                = width;
imageDesc.Height               = height;
imageDesc.Flags                = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
```

DEFAULT_D3D12_SAMPLER_DESC matches DEFAULT_VkSamplerCreateInfo (trilinear,
wrap, whole mip chain); nvvk::SamplerPool's own default differs (see
sampler_pool.hpp).

Differences with nvvk that follow from D3D12:
- DEFAULT_D3D12_RECT takes the width and height (D3D12 has no extent type)
  and returns a D3D12_RECT, the scissor and clear rectangle type.
- Left out: DEFAULT_VkImageSubresourceRange and DEFAULT_VkImageViewCreateInfo
  (views are descriptors written by DescriptorHeap; barriers use
  DEFAULT_D3D12_BARRIER_SUBRESOURCE_RANGE), DEFAULT_VkRenderingAttachmentInfo
  and DEFAULT_VkRenderingInfo (D3D12 renders to RTV / DSV handles set with
  OMSetRenderTargets or BeginRenderPass, see nvdx12::RenderTargetState).
-------------------------------------------------------------------------------------------------*/

// A 1x1 2D texture with one mip and one layer, as DEFAULT_VkImageCreateInfo;
// the description nvdx12::ResourceAllocator::createImage takes.
constexpr D3D12_RESOURCE_DESC1 DEFAULT_D3D12_RESOURCE_DESC1{
    .Dimension                = D3D12_RESOURCE_DIMENSION_TEXTURE2D,
    .Alignment                = 0,
    .Width                    = 1,
    .Height                   = 1,
    .DepthOrArraySize         = 1,
    .MipLevels                = 1,
    .Format                   = DXGI_FORMAT_B8G8R8A8_UNORM,
    .SampleDesc               = {1, 0},
    .Layout                   = D3D12_TEXTURE_LAYOUT_UNKNOWN,
    .Flags                    = D3D12_RESOURCE_FLAG_NONE,
    .SamplerFeedbackMipRegion = {},
};

// Trilinear, wrapping, no anisotropy, the whole mip chain: the sampler
// nvdx12::SamplerPool::acquireSampler hands out without a description.
constexpr D3D12_SAMPLER_DESC DEFAULT_D3D12_SAMPLER_DESC{
    .Filter         = D3D12_FILTER_MIN_MAG_MIP_LINEAR,
    .AddressU       = D3D12_TEXTURE_ADDRESS_MODE_WRAP,
    .AddressV       = D3D12_TEXTURE_ADDRESS_MODE_WRAP,
    .AddressW       = D3D12_TEXTURE_ADDRESS_MODE_WRAP,
    .MipLODBias     = 0.0f,
    .MaxAnisotropy  = 1,
    .ComparisonFunc = D3D12_COMPARISON_FUNC_NEVER,
    .BorderColor    = {0.0f, 0.0f, 0.0f, 0.0f},
    .MinLOD         = 0.0f,
    .MaxLOD         = D3D12_FLOAT32_MAX,
};

// Depth 1, stencil 0: the clear value of a depth buffer, as DEFAULT_VkClearDepthStencilValue.
constexpr D3D12_DEPTH_STENCIL_VALUE DEFAULT_D3D12_DEPTH_STENCIL_VALUE{
    .Depth   = 1.0f,
    .Stencil = 0,
};

// The rectangle (0, 0) - (width, height), as DEFAULT_VkRect2D.
constexpr D3D12_RECT DEFAULT_D3D12_RECT(UINT width, UINT height)
{
  return {
      .left   = 0,
      .top    = 0,
      .right  = LONG(width),
      .bottom = LONG(height),
  };
}

// Every mip, layer and plane of a texture, for D3D12_TEXTURE_BARRIER::Subresources.
constexpr D3D12_BARRIER_SUBRESOURCE_RANGE DEFAULT_D3D12_BARRIER_SUBRESOURCE_RANGE{
    .IndexOrFirstMipLevel = 0xFFFFFFFFu,
    .NumMipLevels         = 0,
    .FirstArraySlice      = 0,
    .NumArraySlices       = 0,
    .FirstPlane           = 0,
    .NumPlanes            = 0,
};
