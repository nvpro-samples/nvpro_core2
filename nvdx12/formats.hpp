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

#include <cstdint>
#include <vector>

#include "resources.hpp"

namespace nvdx12 {

/*-------------------------------------------------------------------------------------------------
# Format and footprint helpers

The D3D12 counterparts of nvvk/formats.hpp and nvvk/mipmaps.hpp: format
queries, texel sizes, mip counts and the placed footprints image <-> buffer
copies use.

Differences with nvvk that follow from D3D12:
- `findSupportedFormat` takes the ID3D12Device and D3D12_FORMAT_SUPPORT1
  bits instead of the physical device, an image tiling and
  VkFormatFeatureFlags2 (D3D12 has one optimal tiling).
- `mipLevels` lives here rather than in a mipmaps header; mip generation is
  BuiltinKernels::cmdGenerateMipmaps (builtin_kernels.hpp), since D3D12 has
  no blit command.
- D3D12 extras: texel and block sizes (`formatBytesPerPixel`,
  `formatBlockInfo`), `imageDataSize` and the placed footprints of image /
  buffer copies, which Vulkan copies describe with regions instead.

See usage_Formats() in formats.cpp.
-------------------------------------------------------------------------------------------------*/

//-----------------------------------------------------------------------------
// Format support

// First format of `candidates` the device supports with every bit of
// `features` (D3D12_FEATURE_FORMAT_SUPPORT); DXGI_FORMAT_UNKNOWN when none does.
DXGI_FORMAT findSupportedFormat(ID3D12Device* device, const std::vector<DXGI_FORMAT>& candidates, D3D12_FORMAT_SUPPORT1 features);

// A depth, or depth-stencil, format usable as a depth-stencil target.
DXGI_FORMAT findDepthFormat(ID3D12Device* device);
DXGI_FORMAT findDepthStencilFormat(ID3D12Device* device);

//-----------------------------------------------------------------------------
// Sizes

// Bytes per texel of the uncompressed formats, including their typeless
// and depth-stencil variants; 0 for block-compressed, planar and video formats.
uint32_t formatBytesPerPixel(DXGI_FORMAT format);

// Texel block of a format: 1x1 and formatBytesPerPixel bytes for the
// uncompressed formats, 4x4 and 8 (BC1, BC4) or 16 bytes (BC2, BC3, BC5 to
// BC7) for the block-compressed ones, 2x1 and 4 bytes for R8G8_B8G8 and
// G8R8_G8B8; `bytes` is 0 for planar and video formats.
struct FormatBlockInfo
{
  uint32_t width{1};
  uint32_t height{1};
  uint32_t bytes{0};
};
FormatBlockInfo formatBlockInfo(DXGI_FORMAT format);

// Total bytes of the mip chain of `image` assuming tight packing (memory
// estimates, readback sizes); block-compressed mips count whole blocks.
uint64_t imageDataSize(const Image& image);

// Number of mip levels of a full chain (nvvk/mipmaps.hpp).
uint32_t        mipLevels(uint32_t width, uint32_t height = 1, uint32_t depth = 1);
inline uint32_t mipLevels(const Extent2D& extent)
{
  return mipLevels(extent.width, extent.height);
}
inline uint32_t mipLevels(const Extent3D& extent)
{
  return mipLevels(extent.width, extent.height, extent.depth);
}

//-----------------------------------------------------------------------------
// Subresource footprints

// Row pitch (256-byte aligned) and total size of one mip level as laid out
// by GetCopyableFootprints; used by image <-> buffer copies and readbacks.
struct SubresourceFootprint
{
  D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
  uint32_t                           rowCount{0};
  uint64_t                           rowSizeInBytes{0};  // bytes of texel data per row, before padding
  uint64_t totalBytes{0};  // pTotalBytes of GetCopyableFootprints (the buffer bytes to reserve)
};

// Footprint of (`mipLevel`, `arrayLayer`) placed at `baseOffset` in a
// buffer; `baseOffset` is a multiple of D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT (512).
// Block-compressed formats report rows of blocks (rowCount = block rows).
// An out-of-range mip or layer returns a zero footprint.
SubresourceFootprint imageSubresourceFootprint(const Image& image, uint32_t mipLevel, uint32_t arrayLayer = 0, uint64_t baseOffset = 0);

}  // namespace nvdx12
