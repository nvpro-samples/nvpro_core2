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

#include <cassert>
#include <cstdint>
#include <span>

#include <glm/glm.hpp>

#include <nvdx12/descriptor_heap.hpp>
#include <nvdx12/resource_allocator.hpp>
#include <nvshaders/tonemap_io.h.slang>
#include <nvutils/timers.hpp>

namespace nvshaders {

/*-------------------------------------------------------------------------------------------------
# class nvshaders::TonemapperDx12

> Runs nvshaders/tonemapper.slang on Direct3D 12: HDR to LDR with the optional
> histogram auto-exposure. The counterpart of nvshaders::Tonemapper, taking the
> settings struct of nvshaders/tonemap_io.h.slang and the descriptor heap slots
> of the images instead of VkDescriptorImageInfo.

The shader compiled to DXIL binds at fixed registers, not through
ResourceDescriptorHeap[]: `b0` TonemapperData, `t0` the input image, `u0` the
output image, `u1` the histogram and `u2` the luminance (the Slang declaration
order). The host owns a root signature for them: 32-bit root constants for the
settings, one descriptor table per image into the application's heap (SRV at
`inputImageSlot`, UAV of the `outputImageSlot` pair) and
root UAVs for the two buffers it creates.

The settings travel as root constants in the constant-buffer layout of the
shader, which differs from the C++ struct (16-byte rows for the matrix, float3
members that do not cross a 16-byte boundary); runCompute repacks them.

runCompute binds the heap and sets this root signature and pipeline; the
caller rebinds its own afterwards (nvdx12::DescriptorHeap::cmdBind). The
input image is read in SHADER_RESOURCE layout and the output written in
UNORDERED_ACCESS layout; the caller records those barriers.

The DXIL comes from compile_slang(... TARGET dxil ENTRY_POINTS Tonemap Histogram
AutoExposure) or from the copies nvshaders_host compiles at build time (the
init overload without shaders).

Usage:
  see usage_TonemapperDx12() in tonemapper_dx12.cpp
-------------------------------------------------------------------------------------------------*/

// The three compute entry points of tonemapper.slang compiled to DXIL, one blob each.
struct TonemapperDxil
{
  std::span<const uint8_t> tonemap;
  std::span<const uint8_t> histogram;
  std::span<const uint8_t> autoExposure;
};

class TonemapperDx12
{
public:
  TonemapperDx12() = default;
  ~TonemapperDx12() { assert(m_device == nullptr); }  // "Missing to call deinit"

  // Root signature, the three pipelines and the histogram and luminance buffers.
  HRESULT init(nvdx12::ResourceAllocator* alloc, nvdx12::DescriptorHeap* heap, const TonemapperDxil& dxil);
  // Same, with the DXIL nvshaders_host compiles at build time.
  HRESULT init(nvdx12::ResourceAllocator* alloc, nvdx12::DescriptorHeap* heap);
  void    deinit();

  // Tonemaps the SRV at `inputImageSlot` into the UAV of `outputImageSlot`,
  // after the histogram and exposure passes when auto-exposure is on. Leaves
  // the heap, the host's root signature and pipeline bound.
  void runCompute(ID3D12GraphicsCommandList7*     cmd,
                  const glm::uvec2&               size,
                  const shaderio::TonemapperData& settings,
                  uint32_t                        inputImageSlot,
                  nvdx12::ImageSlot               outputImageSlot);

private:
  // Histogram-based auto-exposure, run with the root arguments runCompute set
  void    runAutoExposureHistogram(ID3D12GraphicsCommandList7* cmd, const glm::uvec2& size);
  void    runAutoExposure(ID3D12GraphicsCommandList7* cmd);
  void    clearHistogram(ID3D12GraphicsCommandList7* cmd);
  HRESULT createRootSignature();

  nvdx12::ResourceAllocator* m_alloc{nullptr};
  nvdx12::DescriptorHeap*    m_heap{nullptr};  // The images' slots live here
  ID3D12Device*              m_device{nullptr};

  nvdx12::ComPtr<ID3D12RootSignature> m_rootSignature;
  nvdx12::ComPtr<ID3D12PipelineState> m_tonemapPipeline;
  nvdx12::ComPtr<ID3D12PipelineState> m_histogramPipeline;
  nvdx12::ComPtr<ID3D12PipelineState> m_exposurePipeline;

  nvutils::PerformanceTimer m_timer;                    // Scales the auto-exposure speed by the frame time
  bool                      m_histogramCleared{false};  // clearHistogram ran before the first auto-exposure

  // Auto-Exposure
  nvdx12::Buffer m_exposureBuffer;   // float: the adapted luminance
  nvdx12::Buffer m_histogramBuffer;  // uint[EXPOSURE_HISTOGRAM_SIZE]
};

}  // namespace nvshaders
