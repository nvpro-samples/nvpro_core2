/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "sky_dx12.hpp"

#include <d3dx12.h>

#include <nvdx12/barriers.hpp>
#include <nvutils/logger.hpp>

// DXIL compiled by nvshaders_host/CMakeLists.txt (compile_slang TARGET dxil ENTRY_POINTS main)
#include "sky_physical_main.slang.h"
#include "sky_simple_main.slang.h"

namespace nvshaders {

// The matrix offsets the shaders' reflection reports (slangc -reflection-json, sm_6_6)
static_assert(SkySimpleDx12::kMatrixOffset == 112);
static_assert(SkyPhysicalDx12::kMatrixOffset == 96);

HRESULT createSkyRootSignature(nvdx12::ComPtr<ID3D12RootSignature>& rootSignature, ID3D12Device* device, uint32_t constantCount)
{
  rootSignature.Reset();

  // One-entry table into the application's heap; the entry may be rewritten
  // (image resize) while a list that used it is in flight, hence volatile.
  const CD3DX12_DESCRIPTOR_RANGE1 uavRange(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 0, 0, D3D12_DESCRIPTOR_RANGE_FLAG_DESCRIPTORS_VOLATILE);

  CD3DX12_ROOT_PARAMETER1 params[2];
  params[0].InitAsConstants(constantCount, 0);
  params[1].InitAsDescriptorTable(1, &uavRange);

  CD3DX12_VERSIONED_ROOT_SIGNATURE_DESC desc;
  desc.Init_1_1(2, params);

  nvdx12::ComPtr<ID3DBlob> serialized;
  nvdx12::ComPtr<ID3DBlob> error;
  const HRESULT hr = D3DX12SerializeVersionedRootSignature(&desc, D3D_ROOT_SIGNATURE_VERSION_1_1, &serialized, &error);
  if(FAILED(hr))
  {
    LOGE("Sky root signature serialization failed: %s\n",
         error ? static_cast<const char*>(error->GetBufferPointer()) : nvdx12::hresultToString(hr).c_str());
    return hr;
  }
  return NVDX12_FAIL_REPORT(device->CreateRootSignature(0, serialized->GetBufferPointer(), serialized->GetBufferSize(),
                                                        IID_PPV_ARGS(&rootSignature)));
}

template <>
std::span<const uint8_t> skyDxil<shaderio::SkySimpleParameters>()
{
  return sky_simple_main_slang;
}

template <>
std::span<const uint8_t> skyDxil<shaderio::SkyPhysicalParameters>()
{
  return sky_physical_main_slang;
}

}  // namespace nvshaders

//--------------------------------------------------------------------------------------------------
// Usage example
//--------------------------------------------------------------------------------------------------
[[maybe_unused]] static void usage_SkyDx12()
{
  nvdx12::ResourceAllocator*  alloc = nullptr;  // EX: the application's allocator
  nvdx12::DescriptorHeap*     heap  = nullptr;  // EX: the heap the image's slot pair lives in
  ID3D12GraphicsCommandList7* cmd   = nullptr;
  nvdx12::Image               skyImage;  // EX: RGBA32F storage image on the pair `skySlot`
  nvdx12::ImageSlot           skySlot;   // EX: heap->acquireImageSlot(), heap->writeImageSlot(skySlot, skyImage)
  glm::uvec2                  size{1920, 1080};
  glm::mat4                   view{1.0f};
  glm::mat4                   proj{1.0f};

  nvshaders::SkyPhysicalDx12 sky;  // or SkySimpleDx12
  NVDX12_CHECK(sky.init(alloc, heap));  // or init(alloc, heap, dxil) with your own compile_slang(... TARGET dxil ENTRY_POINTS main) output

  // Per frame
  shaderio::SkyPhysicalParameters params{};  // EX: from the UI
  nvdx12::cmdImageBarrier(cmd, skyImage, D3D12_BARRIER_LAYOUT_UNORDERED_ACCESS);
  sky.runCompute(cmd, size, view, proj, params, skySlot);
  heap->cmdBind(cmd);  // Back to the application's root signature

  sky.deinit();
}
