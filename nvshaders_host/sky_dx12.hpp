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

#pragma once

#include <nvdx12/basic_types.hpp>

#include <array>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <span>

#include <glm/glm.hpp>

#include <nvdx12/check_error.hpp>
#include <nvdx12/debug_util.hpp>
#include <nvdx12/descriptor_heap.hpp>
#include <nvdx12/pipeline.hpp>
#include <nvdx12/resource_allocator.hpp>
#include <nvshaders/sky_io.h.slang>

namespace nvshaders {

/*-------------------------------------------------------------------------------------------------
# class nvshaders::SkyBaseDx12

> Renders the procedural skies of nvshaders/sky_simple.slang and
> sky_physical.slang into a storage image on Direct3D 12, the counterpart of
> nvshaders::SkyBase. SkySimpleDx12 and SkyPhysicalDx12 are its two instances,
> sharing the parameter structs of nvshaders/sky_io.h.slang with the Vulkan
> classes.

The shader compiled to DXIL binds `b0` (the parameters followed by the inverse
view-projection matrix) and `u0` (the output image). The host's root signature
carries them as 32-bit root constants and one descriptor table into the
application's heap, the UAV of the `outputImageSlot` pair.
The matrix starts on the 16-byte boundary after the parameters, as in the
shader's constant buffer (byte 96 for sky_physical, whose parameters take 88).

runCompute binds the heap and sets this root signature and pipeline; the
caller rebinds its own afterwards (nvdx12::DescriptorHeap::cmdBind) and has
the image in UNORDERED_ACCESS layout.

The DXIL comes from compile_slang(... TARGET dxil ENTRY_POINTS main) or from the copies
nvshaders_host compiles at build time (the init overload without shaders).

Usage:
  see usage_SkyDx12() in sky_dx12.cpp
-------------------------------------------------------------------------------------------------*/

// Root signature of the skies: `constantCount` 32-bit constants at b0 and a one-entry UAV table at u0.
HRESULT createSkyRootSignature(nvdx12::ComPtr<ID3D12RootSignature>& rootSignature, ID3D12Device* device, uint32_t constantCount);

// DXIL of the sky shaders compiled by nvshaders_host at build time, per parameter struct.
template <typename SkyParams>
std::span<const uint8_t> skyDxil();
template <>
std::span<const uint8_t> skyDxil<shaderio::SkySimpleParameters>();
template <>
std::span<const uint8_t> skyDxil<shaderio::SkyPhysicalParameters>();

template <typename SkyParams>
class SkyBaseDx12
{
public:
  SkyBaseDx12() = default;
  virtual ~SkyBaseDx12() { assert(m_device == nullptr); }  // "Missing to call deinit"

  // Root signature and compute pipeline; `alloc` provides the device.
  HRESULT init(nvdx12::ResourceAllocator* alloc, nvdx12::DescriptorHeap* heap, std::span<const uint8_t> dxil)
  {
    assert(!m_device);
    m_heap   = heap;
    m_device = alloc->getDevice();

    NVDX12_FAIL_RETURN(createSkyRootSignature(m_rootSignature, m_device, kConstantCount));
    NVDX12_DBG_NAME(m_rootSignature);
    NVDX12_FAIL_RETURN(nvdx12::createComputePipeline(m_pipeline, m_device, m_rootSignature.Get(), dxil));
    NVDX12_DBG_NAME(m_pipeline);
    return S_OK;
  }

  // Same, with the DXIL nvshaders_host compiles at build time.
  HRESULT init(nvdx12::ResourceAllocator* alloc, nvdx12::DescriptorHeap* heap)
  {
    return init(alloc, heap, skyDxil<SkyParams>());
  }

  void deinit()
  {
    m_pipeline.Reset();
    m_rootSignature.Reset();
    m_device = nullptr;
    m_heap   = nullptr;
  }

  // Writes the sky seen through the camera into the UAV of `outputImageSlot`.
  // Leaves the heap, the host's root signature and pipeline bound.
  void runCompute(ID3D12GraphicsCommandList7* cmd,
                  const glm::uvec2&           size,
                  const glm::mat4&            viewMatrix,
                  const glm::mat4&            projMatrix,
                  const SkyParams&            skyParam,
                  nvdx12::ImageSlot           outputImageSlot)
  {
    NVDX12_DBG_SCOPE(cmd);  // <-- Helps to debug in NSight

    // Remove the translation from the view matrix
    glm::mat4 viewNoTrans = viewMatrix;
    viewNoTrans[3]        = {0.0f, 0.0f, 0.0f, 1.0f};
    glm::mat4 mvp = glm::inverse(projMatrix * viewNoTrans);  // This will be to have a world direction vector pointing to the pixel

    // Root constants: the parameters, then the matrix at the shader's offset
    std::array<uint32_t, kConstantCount> constants{};
    std::memcpy(constants.data(), &skyParam, sizeof(SkyParams));
    std::memcpy(reinterpret_cast<uint8_t*>(constants.data()) + kMatrixOffset, &mvp, sizeof(glm::mat4));

    m_heap->cmdBind(cmd);  // the descriptor table points into it
    cmd->SetComputeRootSignature(m_rootSignature.Get());
    cmd->SetComputeRoot32BitConstants(kRootConstants, kConstantCount, constants.data(), 0);
    cmd->SetComputeRootDescriptorTable(kRootOutImage, m_heap->gpuHandle(outputImageSlot.uav()));
    cmd->SetPipelineState(m_pipeline.Get());

    // Dispatching the compute job
    cmd->Dispatch(nvdx12::getGroupCounts(size.x, 16), nvdx12::getGroupCounts(size.y, 16), 1);
  }

  // Constant-buffer layout of the shader: the float4x4 follows the parameters on a 16-byte boundary
  static constexpr uint32_t kMatrixOffset  = (uint32_t(sizeof(SkyParams)) + 15) / 16 * 16;
  static constexpr uint32_t kConstantCount = (kMatrixOffset + uint32_t(sizeof(glm::mat4))) / uint32_t(sizeof(uint32_t));

protected:
  static constexpr uint32_t kRootConstants = 0;  // b0
  static constexpr uint32_t kRootOutImage  = 1;  // u0

  nvdx12::DescriptorHeap*             m_heap{nullptr};  // The output image's slot pair lives here
  ID3D12Device*                       m_device{nullptr};
  nvdx12::ComPtr<ID3D12RootSignature> m_rootSignature;
  nvdx12::ComPtr<ID3D12PipelineState> m_pipeline;
};

// Define specific types
using SkySimpleDx12   = SkyBaseDx12<shaderio::SkySimpleParameters>;
using SkyPhysicalDx12 = SkyBaseDx12<shaderio::SkyPhysicalParameters>;

}  // namespace nvshaders
