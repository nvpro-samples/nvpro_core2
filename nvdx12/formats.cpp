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

#include "formats.hpp"

#include <algorithm>
#include <vector>

namespace nvdx12 {

//--------------------------------------------------------------------------------------------------
// Format support

DXGI_FORMAT findSupportedFormat(ID3D12Device* device, const std::vector<DXGI_FORMAT>& candidates, D3D12_FORMAT_SUPPORT1 features)
{
  for(size_t i = 0; i < candidates.size(); i++)
  {
    const DXGI_FORMAT                 format = candidates[i];
    D3D12_FEATURE_DATA_FORMAT_SUPPORT support{.Format = format};
    if(SUCCEEDED(device->CheckFeatureSupport(D3D12_FEATURE_FORMAT_SUPPORT, &support, sizeof(support)))
       && (support.Support1 & features) == features)
    {
      return format;
    }
  }
  return DXGI_FORMAT_UNKNOWN;
}

DXGI_FORMAT findDepthFormat(ID3D12Device* device)
{
  return findSupportedFormat(device, {DXGI_FORMAT_D32_FLOAT, DXGI_FORMAT_D24_UNORM_S8_UINT, DXGI_FORMAT_D32_FLOAT_S8X24_UINT, DXGI_FORMAT_D16_UNORM},
                             D3D12_FORMAT_SUPPORT1_DEPTH_STENCIL);
}

DXGI_FORMAT findDepthStencilFormat(ID3D12Device* device)
{
  return findSupportedFormat(device, {DXGI_FORMAT_D24_UNORM_S8_UINT, DXGI_FORMAT_D32_FLOAT_S8X24_UINT},
                             D3D12_FORMAT_SUPPORT1_DEPTH_STENCIL);
}

//--------------------------------------------------------------------------------------------------
// Sizes

uint32_t formatBytesPerPixel(DXGI_FORMAT format)
{
  switch(format)
  {
    case DXGI_FORMAT_R8_TYPELESS:
    case DXGI_FORMAT_R8_UNORM:
    case DXGI_FORMAT_R8_SNORM:
    case DXGI_FORMAT_R8_UINT:
    case DXGI_FORMAT_R8_SINT:
    case DXGI_FORMAT_A8_UNORM:
      return 1;
    case DXGI_FORMAT_R8G8_TYPELESS:
    case DXGI_FORMAT_R8G8_UNORM:
    case DXGI_FORMAT_R8G8_SNORM:
    case DXGI_FORMAT_R8G8_UINT:
    case DXGI_FORMAT_R8G8_SINT:
    case DXGI_FORMAT_R16_TYPELESS:
    case DXGI_FORMAT_R16_UNORM:
    case DXGI_FORMAT_R16_SNORM:
    case DXGI_FORMAT_R16_UINT:
    case DXGI_FORMAT_R16_SINT:
    case DXGI_FORMAT_R16_FLOAT:
    case DXGI_FORMAT_D16_UNORM:
    case DXGI_FORMAT_B5G6R5_UNORM:
    case DXGI_FORMAT_B5G5R5A1_UNORM:
    case DXGI_FORMAT_B4G4R4A4_UNORM:
    case DXGI_FORMAT_A4B4G4R4_UNORM:
      return 2;
    case DXGI_FORMAT_R8G8B8A8_TYPELESS:
    case DXGI_FORMAT_R8G8B8A8_UNORM:
    case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB:
    case DXGI_FORMAT_R8G8B8A8_SNORM:
    case DXGI_FORMAT_R8G8B8A8_UINT:
    case DXGI_FORMAT_R8G8B8A8_SINT:
    case DXGI_FORMAT_B8G8R8A8_TYPELESS:
    case DXGI_FORMAT_B8G8R8A8_UNORM:
    case DXGI_FORMAT_B8G8R8A8_UNORM_SRGB:
    case DXGI_FORMAT_B8G8R8X8_TYPELESS:
    case DXGI_FORMAT_B8G8R8X8_UNORM:
    case DXGI_FORMAT_B8G8R8X8_UNORM_SRGB:
    case DXGI_FORMAT_R16G16_TYPELESS:
    case DXGI_FORMAT_R16G16_UNORM:
    case DXGI_FORMAT_R16G16_SNORM:
    case DXGI_FORMAT_R16G16_UINT:
    case DXGI_FORMAT_R16G16_SINT:
    case DXGI_FORMAT_R16G16_FLOAT:
    case DXGI_FORMAT_R32_TYPELESS:
    case DXGI_FORMAT_R32_UINT:
    case DXGI_FORMAT_R32_SINT:
    case DXGI_FORMAT_R32_FLOAT:
    case DXGI_FORMAT_D32_FLOAT:
    case DXGI_FORMAT_R24G8_TYPELESS:
    case DXGI_FORMAT_D24_UNORM_S8_UINT:
    case DXGI_FORMAT_R24_UNORM_X8_TYPELESS:
    case DXGI_FORMAT_X24_TYPELESS_G8_UINT:
    case DXGI_FORMAT_R10G10B10A2_TYPELESS:
    case DXGI_FORMAT_R10G10B10A2_UNORM:
    case DXGI_FORMAT_R10G10B10A2_UINT:
    case DXGI_FORMAT_R10G10B10_XR_BIAS_A2_UNORM:
    case DXGI_FORMAT_R11G11B10_FLOAT:
    case DXGI_FORMAT_R9G9B9E5_SHAREDEXP:
      return 4;
    case DXGI_FORMAT_R16G16B16A16_TYPELESS:
    case DXGI_FORMAT_R16G16B16A16_UNORM:
    case DXGI_FORMAT_R16G16B16A16_SNORM:
    case DXGI_FORMAT_R16G16B16A16_UINT:
    case DXGI_FORMAT_R16G16B16A16_SINT:
    case DXGI_FORMAT_R16G16B16A16_FLOAT:
    case DXGI_FORMAT_R32G32_TYPELESS:
    case DXGI_FORMAT_R32G32_UINT:
    case DXGI_FORMAT_R32G32_SINT:
    case DXGI_FORMAT_R32G32_FLOAT:
    case DXGI_FORMAT_R32G8X24_TYPELESS:
    case DXGI_FORMAT_D32_FLOAT_S8X24_UINT:
    case DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS:
    case DXGI_FORMAT_X32_TYPELESS_G8X24_UINT:
      return 8;
    case DXGI_FORMAT_R32G32B32_TYPELESS:
    case DXGI_FORMAT_R32G32B32_UINT:
    case DXGI_FORMAT_R32G32B32_SINT:
    case DXGI_FORMAT_R32G32B32_FLOAT:
      return 12;
    case DXGI_FORMAT_R32G32B32A32_TYPELESS:
    case DXGI_FORMAT_R32G32B32A32_UINT:
    case DXGI_FORMAT_R32G32B32A32_SINT:
    case DXGI_FORMAT_R32G32B32A32_FLOAT:
      return 16;
    default:
      return 0;
  }
}

FormatBlockInfo formatBlockInfo(DXGI_FORMAT format)
{
  switch(format)
  {
    case DXGI_FORMAT_BC1_TYPELESS:
    case DXGI_FORMAT_BC1_UNORM:
    case DXGI_FORMAT_BC1_UNORM_SRGB:
    case DXGI_FORMAT_BC4_TYPELESS:
    case DXGI_FORMAT_BC4_UNORM:
    case DXGI_FORMAT_BC4_SNORM:
      return {4, 4, 8};
    case DXGI_FORMAT_BC2_TYPELESS:
    case DXGI_FORMAT_BC2_UNORM:
    case DXGI_FORMAT_BC2_UNORM_SRGB:
    case DXGI_FORMAT_BC3_TYPELESS:
    case DXGI_FORMAT_BC3_UNORM:
    case DXGI_FORMAT_BC3_UNORM_SRGB:
    case DXGI_FORMAT_BC5_TYPELESS:
    case DXGI_FORMAT_BC5_UNORM:
    case DXGI_FORMAT_BC5_SNORM:
    case DXGI_FORMAT_BC6H_TYPELESS:
    case DXGI_FORMAT_BC6H_UF16:
    case DXGI_FORMAT_BC6H_SF16:
    case DXGI_FORMAT_BC7_TYPELESS:
    case DXGI_FORMAT_BC7_UNORM:
    case DXGI_FORMAT_BC7_UNORM_SRGB:
      return {4, 4, 16};
    case DXGI_FORMAT_R8G8_B8G8_UNORM:
    case DXGI_FORMAT_G8R8_G8B8_UNORM:
      return {2, 1, 4};
    default:
      return {1, 1, formatBytesPerPixel(format)};
  }
}

uint64_t imageDataSize(const Image& image)
{
  const FormatBlockInfo block = formatBlockInfo(image.format);
  uint64_t              total = 0;
  for(uint32_t mip = 0; mip < image.mipLevels; mip++)
  {
    const uint64_t w = std::max(1u, image.extent.width >> mip);
    const uint64_t h = std::max(1u, image.extent.height >> mip);
    const uint64_t d = image.dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D ? std::max(1u, image.extent.depth >> mip) : 1;
    const uint64_t blocksX = (w + block.width - 1) / block.width;
    const uint64_t blocksY = (h + block.height - 1) / block.height;
    total += blocksX * blocksY * d * block.bytes;
  }
  return total * image.arrayLayers;
}

uint32_t mipLevels(uint32_t width, uint32_t height, uint32_t depth)
{
  uint32_t largest = std::max(width, std::max(height, depth));
  uint32_t levels  = 1;
  while(largest > 1)
  {
    largest >>= 1;
    levels++;
  }
  return levels;
}

//--------------------------------------------------------------------------------------------------
// Subresource footprints

SubresourceFootprint imageSubresourceFootprint(const Image& image, uint32_t mipLevel, uint32_t arrayLayer, uint64_t baseOffset)
{
  SubresourceFootprint result{};
  if(!image.image || mipLevel >= image.mipLevels || arrayLayer >= image.arrayLayers)
  {
    return result;
  }
  ComPtr<ID3D12Device> device;
  if(FAILED(image.image->GetDevice(IID_PPV_ARGS(&device))))
  {
    return result;
  }
  const D3D12_RESOURCE_DESC desc     = image.image->GetDesc();
  UINT                      rows     = 0;
  UINT64                    rowBytes = 0;
  UINT64                    total    = 0;
  device->GetCopyableFootprints(&desc, image.subresourceIndex(mipLevel, arrayLayer), 1, baseOffset, &result.footprint,
                                &rows, &rowBytes, &total);
  result.rowCount       = rows;
  result.rowSizeInBytes = rowBytes;
  result.totalBytes     = total;
  return result;
}

}  // namespace nvdx12

//--------------------------------------------------------------------------------------------------
// Usage example
//--------------------------------------------------------------------------------------------------
[[maybe_unused]] static void usage_Formats()
{
  ID3D12Device* device = nullptr;  // EX: from nvdx12::Context::getDevice()

  // Image creation: a full mip chain, and the depth format of the device.
  nvdx12::Image image;  // EX: a D3D12_RESOURCE_DESC1 filled the same way, then ResourceAllocator::createImage
  image.extent    = {1024, 512, 1};
  image.mipLevels = nvdx12::mipLevels(image.extent);
  image.format    = nvdx12::findDepthFormat(device);

  // The first candidate with typed UAV support.
  [[maybe_unused]] const DXGI_FORMAT storageFormat =
      nvdx12::findSupportedFormat(device, {DXGI_FORMAT_R11G11B10_FLOAT, DXGI_FORMAT_R16G16B16A16_FLOAT},
                                  D3D12_FORMAT_SUPPORT1_TYPED_UNORDERED_ACCESS_VIEW);

  // Readback: a buffer sized for the padded footprint of mip 0 (see
  // cmdCopyImageToBuffer), read with the 256-byte aligned row pitch.
  const nvdx12::SubresourceFootprint footprint = nvdx12::imageSubresourceFootprint(image, 0);
  std::vector<uint8_t>               readback(footprint.totalBytes);  // EX: the mapping of a readback buffer
  const uint32_t                     x = 3, y = 7;
  [[maybe_unused]] const uint8_t*    texel = readback.data() + footprint.footprint.Offset
                                          + uint64_t(y) * footprint.footprint.Footprint.RowPitch
                                          + uint64_t(x) * nvdx12::formatBytesPerPixel(image.format);
  // ... decode `texel` ...

  // Block-compressed texture: sizes count whole 4x4 blocks, so a row of the
  // tightly packed mip 0 is blocksPerRow * bytesPerBlock and the smallest
  // mips still take one block each.
  nvdx12::Image bcImage;  // EX: DXGI_FORMAT_BC7_UNORM_SRGB, created by ResourceAllocator::createImage
  bcImage.extent                        = {1024, 1024, 1};
  bcImage.mipLevels                     = nvdx12::mipLevels(bcImage.extent);
  bcImage.format                        = DXGI_FORMAT_BC7_UNORM_SRGB;
  const nvdx12::FormatBlockInfo   block = nvdx12::formatBlockInfo(bcImage.format);  // 4x4, 16 bytes
  [[maybe_unused]] const uint64_t mip0RowBytes =
      uint64_t((bcImage.extent.width + block.width - 1) / block.width) * block.bytes;
  [[maybe_unused]] const uint64_t chainBytes = nvdx12::imageDataSize(bcImage);  // every mip, every layer
}
