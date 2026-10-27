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

#include "tonemapper_dx12.hpp"

#include <array>
#include <cstddef>
#include <iterator>

#include <d3dx12.h>

#include <nvdx12/barriers.hpp>
#include <nvdx12/check_error.hpp>
#include <nvdx12/debug_util.hpp>
#include <nvdx12/pipeline.hpp>
#include <nvutils/logger.hpp>

#include "nvshaders/tonemap_functions.h.slang"

// DXIL compiled by nvshaders_host/CMakeLists.txt (compile_slang TARGET dxil)
#include "tonemapper_AutoExposure.slang.h"
#include "tonemapper_Histogram.slang.h"
#include "tonemapper_Tonemap.slang.h"

namespace nvshaders {

namespace {

// Root parameters, in the registers Slang assigned to tonemapper.slang
enum RootParam : uint32_t
{
  eRootConstants = 0,  // b0: TonemapperData
  eRootInputImage,     // t0: descriptor table, SRV of the input image
  eRootOutputImage,    // u0: descriptor table, UAV of the output image
  eRootHistogram,      // u1: root UAV
  eRootLuminance,      // u2: root UAV
};

// TonemapperData in the constant-buffer layout of the shader: the float3x3
// takes three 16-byte rows (shader row j is glm column j, the row-major
// reading of the column-major matrix the Vulkan push constants rely on) and
// a float3 moves to the next row rather than crossing one.
struct TonemapperConstants
{
  int32_t   isActive;
  int32_t   method;
  float     exposure;
  float     temperature;
  float     tint;
  float     pad0[3];
  glm::vec3 inputMatrixRow0;
  float     pad1;
  glm::vec3 inputMatrixRow1;
  float     pad2;
  glm::vec3 inputMatrixRow2;
  float     contrast;
  float     brightness;
  float     saturation;
  float     vignette;
  float     vibrance;
  float     shadowBias;
  float     midtoneBias;
  float     highlightBias;
  float     pad3;
  glm::vec3 coolColor;
  float     pad4;
  glm::vec3 warmColor;
  float     splitBalance;
  int32_t   autoExposure;
  float     autoExposureSpeed;
  float     evMinValue;
  float     evMaxValue;
  uint32_t  enableCenterMetering;
  float     centerMeteringSize;
  uint32_t  averageMode;
  int32_t   dither;
};
// The offsets the shader's reflection reports (slangc -reflection-json, sm_6_6)
static_assert(sizeof(TonemapperConstants) == 176);
static_assert(offsetof(TonemapperConstants, tint) == 16);
static_assert(offsetof(TonemapperConstants, inputMatrixRow0) == 32);
static_assert(offsetof(TonemapperConstants, contrast) == 76);
static_assert(offsetof(TonemapperConstants, coolColor) == 112);
static_assert(offsetof(TonemapperConstants, warmColor) == 128);
static_assert(offsetof(TonemapperConstants, autoExposure) == 144);
static_assert(offsetof(TonemapperConstants, dither) == 172);
constexpr uint32_t kConstantCount = sizeof(TonemapperConstants) / sizeof(uint32_t);

TonemapperConstants packConstants(const shaderio::TonemapperData& d)
{
  TonemapperConstants c{};
  c.isActive             = d.isActive;
  c.method               = d.method;
  c.exposure             = d.exposure;
  c.temperature          = d.temperature;
  c.tint                 = d.tint;
  c.inputMatrixRow0      = d.inputMatrix[0];
  c.inputMatrixRow1      = d.inputMatrix[1];
  c.inputMatrixRow2      = d.inputMatrix[2];
  c.contrast             = d.contrast;
  c.brightness           = d.brightness;
  c.saturation           = d.saturation;
  c.vignette             = d.vignette;
  c.vibrance             = d.vibrance;
  c.shadowBias           = d.shadowBias;
  c.midtoneBias          = d.midtoneBias;
  c.highlightBias        = d.highlightBias;
  c.coolColor            = d.coolColor;
  c.warmColor            = d.warmColor;
  c.splitBalance         = d.splitBalance;
  c.autoExposure         = d.autoExposure;
  c.autoExposureSpeed    = d.autoExposureSpeed;
  c.evMinValue           = d.evMinValue;
  c.evMaxValue           = d.evMaxValue;
  c.enableCenterMetering = d.enableCenterMetering;
  c.centerMeteringSize   = d.centerMeteringSize;
  c.averageMode          = d.averageMode;
  c.dither               = d.dither;
  return c;
}

}  // namespace

HRESULT TonemapperDx12::init(nvdx12::ResourceAllocator* alloc, nvdx12::DescriptorHeap* heap, const TonemapperDxil& dxil)
{
  assert(!m_device);
  m_alloc  = alloc;
  m_heap   = heap;
  m_device = alloc->getDevice();

  // Create buffers
  NVDX12_FAIL_RETURN(alloc->createBuffer(m_exposureBuffer, sizeof(float)));
  NVDX12_DBG_NAME(m_exposureBuffer.buffer);
  NVDX12_FAIL_RETURN(alloc->createBuffer(m_histogramBuffer, sizeof(uint32_t) * EXPOSURE_HISTOGRAM_SIZE));
  NVDX12_DBG_NAME(m_histogramBuffer.buffer);

  NVDX12_FAIL_RETURN(createRootSignature());
  NVDX12_DBG_NAME(m_rootSignature);

  // One compute pipeline per entry point
  NVDX12_FAIL_RETURN(nvdx12::createComputePipeline(m_tonemapPipeline, m_device, m_rootSignature.Get(), dxil.tonemap));
  NVDX12_DBG_NAME(m_tonemapPipeline);
  NVDX12_FAIL_RETURN(nvdx12::createComputePipeline(m_histogramPipeline, m_device, m_rootSignature.Get(), dxil.histogram));
  NVDX12_DBG_NAME(m_histogramPipeline);
  NVDX12_FAIL_RETURN(nvdx12::createComputePipeline(m_exposurePipeline, m_device, m_rootSignature.Get(), dxil.autoExposure));
  NVDX12_DBG_NAME(m_exposurePipeline);

  return S_OK;
}

HRESULT TonemapperDx12::init(nvdx12::ResourceAllocator* alloc, nvdx12::DescriptorHeap* heap)
{
  return init(alloc, heap, {.tonemap = tonemapper_Tonemap_slang, .histogram = tonemapper_Histogram_slang, .autoExposure = tonemapper_AutoExposure_slang});
}

void TonemapperDx12::deinit()
{
  if(!m_device)
    return;

  m_alloc->destroyBuffer(m_exposureBuffer);
  m_alloc->destroyBuffer(m_histogramBuffer);

  m_tonemapPipeline.Reset();
  m_histogramPipeline.Reset();
  m_exposurePipeline.Reset();
  m_rootSignature.Reset();

  m_histogramCleared = false;
  m_device           = nullptr;
  m_heap             = nullptr;
  m_alloc            = nullptr;
}

HRESULT TonemapperDx12::createRootSignature()
{
  // One-entry tables into the application's heap; the entries may be rewritten
  // (image resize) while a list that used them is in flight, hence volatile.
  const CD3DX12_DESCRIPTOR_RANGE1 srvRange(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0, 0, D3D12_DESCRIPTOR_RANGE_FLAG_DESCRIPTORS_VOLATILE);
  const CD3DX12_DESCRIPTOR_RANGE1 uavRange(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 0, 0, D3D12_DESCRIPTOR_RANGE_FLAG_DESCRIPTORS_VOLATILE);

  CD3DX12_ROOT_PARAMETER1 params[5];
  params[eRootConstants].InitAsConstants(kConstantCount, 0);
  params[eRootInputImage].InitAsDescriptorTable(1, &srvRange);
  params[eRootOutputImage].InitAsDescriptorTable(1, &uavRange);
  params[eRootHistogram].InitAsUnorderedAccessView(1);
  params[eRootLuminance].InitAsUnorderedAccessView(2);

  CD3DX12_VERSIONED_ROOT_SIGNATURE_DESC desc;
  desc.Init_1_1(uint32_t(std::size(params)), params);

  nvdx12::ComPtr<ID3DBlob> serialized;
  nvdx12::ComPtr<ID3DBlob> error;
  const HRESULT hr = D3DX12SerializeVersionedRootSignature(&desc, D3D_ROOT_SIGNATURE_VERSION_1_1, &serialized, &error);
  if(FAILED(hr))
  {
    LOGE("Tonemapper root signature serialization failed: %s\n",
         error ? static_cast<const char*>(error->GetBufferPointer()) : nvdx12::hresultToString(hr).c_str());
    return hr;
  }
  return NVDX12_FAIL_REPORT(m_device->CreateRootSignature(0, serialized->GetBufferPointer(),
                                                          serialized->GetBufferSize(), IID_PPV_ARGS(&m_rootSignature)));
}

//----------------------------------
// Run the tonemapper compute shader
//
void TonemapperDx12::runCompute(ID3D12GraphicsCommandList7*     cmd,
                                const glm::uvec2&               size,
                                const shaderio::TonemapperData& settings,
                                uint32_t                        inputImageSlot,
                                nvdx12::ImageSlot               outputImageSlot)
{
  NVDX12_DBG_SCOPE(cmd);  // <-- Helps to debug in NSight

  // Root constants
  shaderio::TonemapperData tonemapperData = settings;
  tonemapperData.autoExposureSpeed *= float(m_timer.getSeconds());
  tonemapperData.inputMatrix =
      shaderio::getColorCorrectionMatrix(tonemapperData.exposure, settings.temperature, settings.tint);
  m_timer.reset();
  const TonemapperConstants constants = packConstants(tonemapperData);

  // Root arguments shared by the three passes; the descriptor tables point into the heap
  m_heap->cmdBind(cmd);
  cmd->SetComputeRootSignature(m_rootSignature.Get());
  cmd->SetComputeRoot32BitConstants(eRootConstants, kConstantCount, &constants, 0);
  cmd->SetComputeRootDescriptorTable(eRootInputImage, m_heap->gpuHandle(inputImageSlot));
  cmd->SetComputeRootDescriptorTable(eRootOutputImage, m_heap->gpuHandle(outputImageSlot.uav()));
  cmd->SetComputeRootUnorderedAccessView(eRootHistogram, m_histogramBuffer.gpuAddress);
  cmd->SetComputeRootUnorderedAccessView(eRootLuminance, m_exposureBuffer.gpuAddress);

  // Run auto-exposure histogram/exposure if enabled
  if(settings.isActive && settings.autoExposure)
  {
    if(!m_histogramCleared)
    {
      clearHistogram(cmd);
      m_histogramCleared = true;
    }

    runAutoExposureHistogram(cmd, size);
    runAutoExposure(cmd);
  }

  // Run tonemapper compute shader
  cmd->SetPipelineState(m_tonemapPipeline.Get());
  const nvdx12::Extent2D groups = nvdx12::getGroupCounts(nvdx12::Extent2D{size.x, size.y}, TONEMAP_WORKGROUP_SIZE);
  cmd->Dispatch(groups.width, groups.height, 1);
}

void TonemapperDx12::runAutoExposureHistogram(ID3D12GraphicsCommandList7* cmd, const glm::uvec2& size)
{
  NVDX12_DBG_SCOPE(cmd);  // <-- Helps to debug in NSight

  cmd->SetPipelineState(m_histogramPipeline.Get());
  const nvdx12::Extent2D groups = nvdx12::getGroupCounts(nvdx12::Extent2D{size.x, size.y}, TONEMAP_WORKGROUP_SIZE);
  cmd->Dispatch(groups.width, groups.height, 1);
  nvdx12::cmdBufferBarrier(cmd, m_histogramBuffer);  // The exposure pass reads the bins
}

void TonemapperDx12::runAutoExposure(ID3D12GraphicsCommandList7* cmd)
{
  NVDX12_DBG_SCOPE(cmd);  // <-- Helps to debug in NSight

  cmd->SetPipelineState(m_exposurePipeline.Get());
  cmd->Dispatch(1, 1, 1);
  nvdx12::cmdBufferBarrier(cmd, m_exposureBuffer);  // The tonemap pass reads the luminance
}

void TonemapperDx12::clearHistogram(ID3D12GraphicsCommandList7* cmd)
{
  // Immediate writes of the bins and of the adapted luminance (the vkCmdUpdateBuffer and
  // vkCmdFillBuffer of the Vulkan host), ordered before the histogram pass. The exposure pass
  // blends with the previous luminance, so it must not start from uninitialized memory.
  std::array<D3D12_WRITEBUFFERIMMEDIATE_PARAMETER, EXPOSURE_HISTOGRAM_SIZE + 1> writes{};
  for(uint32_t i = 0; i < EXPOSURE_HISTOGRAM_SIZE; i++)
  {
    writes[i] = {.Dest = m_histogramBuffer.gpuAddressAt(sizeof(uint32_t) * i), .Value = 0};
  }
  writes[EXPOSURE_HISTOGRAM_SIZE] = {.Dest = m_exposureBuffer.gpuAddress, .Value = 0};  // 0.0f
  cmd->WriteBufferImmediate(uint32_t(writes.size()), writes.data(), nullptr);

  for(ID3D12Resource* buffer : {m_histogramBuffer.buffer.Get(), m_exposureBuffer.buffer.Get()})
  {
    nvdx12::cmdBufferMemoryBarrier(cmd, {.buffer        = buffer,
                                         .srcStageMask  = D3D12_BARRIER_SYNC_ALL,
                                         .dstStageMask  = D3D12_BARRIER_SYNC_COMPUTE_SHADING,
                                         .srcAccessMask = D3D12_BARRIER_ACCESS_COMMON,
                                         .dstAccessMask = D3D12_BARRIER_ACCESS_UNORDERED_ACCESS});
  }
}

}  // namespace nvshaders

//--------------------------------------------------------------------------------------------------
// Usage example
//--------------------------------------------------------------------------------------------------
[[maybe_unused]] static void usage_TonemapperDx12()
{
  nvdx12::ResourceAllocator*  alloc = nullptr;  // EX: the application's allocator
  nvdx12::DescriptorHeap*     heap  = nullptr;  // EX: the heap the images' slot pairs live in
  ID3D12GraphicsCommandList7* cmd   = nullptr;
  nvdx12::Image               hdrImage;  // EX: RGBA32F render target on the pair `hdrSlot`
  nvdx12::Image               ldrImage;  // EX: RGBA8 image ImGui shows, on the pair `ldrSlot`
  nvdx12::ImageSlot           hdrSlot;   // EX: heap->acquireImageSlot(), heap->writeImageSlot(hdrSlot, hdrImage)
  nvdx12::ImageSlot           ldrSlot;
  glm::uvec2                  size{1920, 1080};

  nvshaders::TonemapperDx12 tonemapper;
  NVDX12_CHECK(tonemapper.init(alloc, heap));  // or init(alloc, heap, {tonemapDxil, histogramDxil, autoExposureDxil})

  // Per frame: the input is sampled, the output written
  shaderio::TonemapperData settings{};  // EX: from the UI; autoExposure = 1 adds the histogram passes
  nvdx12::cmdImageBarrier(cmd, hdrImage, D3D12_BARRIER_LAYOUT_SHADER_RESOURCE);
  nvdx12::cmdImageBarrier(cmd, ldrImage, D3D12_BARRIER_LAYOUT_UNORDERED_ACCESS);
  tonemapper.runCompute(cmd, size, settings, hdrSlot.srv, ldrSlot);
  heap->cmdBind(cmd);  // Back to the application's root signature

  tonemapper.deinit();
}
