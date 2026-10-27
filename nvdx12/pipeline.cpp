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

#include "pipeline.hpp"

#include <cassert>

#include <d3dx12.h>

#include <nvutils/logger.hpp>

#include "check_error.hpp"
#include "debug_util.hpp"

namespace nvdx12 {

HRESULT createBindlessRootSignature(ComPtr<ID3D12RootSignature>& rootSignature, ID3D12Device* device, uint32_t pushConstantDwords)
{
  rootSignature.Reset();
  assert(pushConstantDwords <= kPushConstantsMaxDwords);

  CD3DX12_ROOT_PARAMETER1 params[2];
  params[kRootParamPushData].InitAsShaderResourceView(0, 0, D3D12_ROOT_DESCRIPTOR_FLAG_NONE, D3D12_SHADER_VISIBILITY_ALL);
  params[kRootParamPushConstants].InitAsConstants(pushConstantDwords, 0, 0, D3D12_SHADER_VISIBILITY_ALL);
  CD3DX12_VERSIONED_ROOT_SIGNATURE_DESC desc;
  desc.Init_1_1(pushConstantDwords > 0 ? 2 : 1, params, 0, nullptr,
                D3D12_ROOT_SIGNATURE_FLAG_CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED | D3D12_ROOT_SIGNATURE_FLAG_SAMPLER_HEAP_DIRECTLY_INDEXED);

  ComPtr<ID3DBlob> serialized;
  ComPtr<ID3DBlob> error;
  HRESULT hr = D3DX12SerializeVersionedRootSignature(&desc, D3D_ROOT_SIGNATURE_VERSION_1_1, &serialized, &error);
  if(FAILED(hr))
  {
    LOGE("Root signature serialization failed: %s\n",
         error ? static_cast<const char*>(error->GetBufferPointer()) : hresultToString(hr).c_str());
    return hr;
  }
  return NVDX12_FAIL_REPORT(device->CreateRootSignature(0, serialized->GetBufferPointer(), serialized->GetBufferSize(),
                                                        IID_PPV_ARGS(&rootSignature)));
}

void cmdPushConstants(ID3D12GraphicsCommandList* cmd, const void* data, uint32_t size, BindPoint bindPoint, uint32_t rootParameterIndex)
{
  assert(size % sizeof(uint32_t) == 0 && size <= kPushConstantsMaxBytes);
  const uint32_t count = size / sizeof(uint32_t);
  if(bindPoint != BindPoint::eGraphics)
  {
    cmd->SetComputeRoot32BitConstants(rootParameterIndex, count, data, 0);
  }
  if(bindPoint != BindPoint::eCompute)
  {
    cmd->SetGraphicsRoot32BitConstants(rootParameterIndex, count, data, 0);
  }
}

HRESULT createDispatchIndirectSignature(ComPtr<ID3D12CommandSignature>& signature, ID3D12Device* device)
{
  signature.Reset();

  D3D12_INDIRECT_ARGUMENT_DESC argument{};
  argument.Type = D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH;
  D3D12_COMMAND_SIGNATURE_DESC desc{};
  desc.ByteStride       = sizeof(D3D12_DISPATCH_ARGUMENTS);
  desc.NumArgumentDescs = 1;
  desc.pArgumentDescs   = &argument;
  return NVDX12_FAIL_REPORT(device->CreateCommandSignature(&desc, nullptr, IID_PPV_ARGS(&signature)));
}

HRESULT createComputePipeline(ComPtr<ID3D12PipelineState>& pipeline,
                              ID3D12Device*                device,
                              ID3D12RootSignature*         rootSignature,
                              const ShaderBlob&            blob,
                              D3D12_PIPELINE_STATE_FLAGS   flags)
{
  if(!blob.valid())
  {
    LOGE("Compute pipeline: shader blob is not valid\n%s\n", blob.errors.c_str());
    pipeline.Reset();
    return E_INVALIDARG;
  }
  return createComputePipeline(pipeline, device, rootSignature, blob.dxil, flags);
}

HRESULT createComputePipeline(ComPtr<ID3D12PipelineState>& pipeline,
                              ID3D12Device*                device,
                              ID3D12RootSignature*         rootSignature,
                              std::span<const uint8_t>     dxil,
                              D3D12_PIPELINE_STATE_FLAGS   flags)
{
  pipeline.Reset();

  D3D12_COMPUTE_PIPELINE_STATE_DESC desc{};
  desc.pRootSignature = rootSignature;
  desc.CS             = D3D12_SHADER_BYTECODE{dxil.data(), dxil.size()};
  desc.Flags          = flags;
  return NVDX12_FAIL_REPORT(device->CreateComputePipelineState(&desc, IID_PPV_ARGS(&pipeline)));
}

}  // namespace nvdx12

//--------------------------------------------------------------------------------------------------
// Usage example
//--------------------------------------------------------------------------------------------------
[[maybe_unused]] static void usage_Pipeline()
{
  ID3D12Device*              device = nullptr;
  ID3D12GraphicsCommandList* cmd    = nullptr;
  nvdx12::ShaderCompiler     compiler;  // see usage_ShaderCompiler()

  // One root signature for every compute PSO, graphics PSO and ray-tracing state object;
  // the creation helpers leave naming to the caller.
  nvdx12::ComPtr<ID3D12RootSignature> rootSignature;
  NVDX12_CHECK(nvdx12::createBindlessRootSignature(rootSignature, device));
  NVDX12_DBG_NAME(rootSignature);

  // Compute PSO from a runtime-compiled shader.
  nvdx12::ShaderBlob                  blob = compiler.compileFile("fill.comp.hlsl", "cs_6_6");
  nvdx12::ComPtr<ID3D12PipelineState> pipeline;
  NVDX12_CHECK(nvdx12::createComputePipeline(pipeline, device, rootSignature.Get(), blob));
  NVDX12_DBG_CUSTOM_NAME(pipeline, "fill");

  // Dispatch: the heap bind (DescriptorHeap::cmdBind) set the root signature;
  // the arguments go in root constants, PUSH_CONSTANTS(FillArgs, args) in the
  // shader, then one group per 256 elements. Data beyond kPushConstantsMaxBytes
  // goes through UploadRing::cmdPushData instead.
  struct FillArgs
  {
    uint64_t bufferAddress;  // nvdx12::Buffer::address
    uint32_t elementCount;
    uint32_t value;
  } args{0, 100'000, 0xFFFFFFFFu};
  cmd->SetPipelineState(pipeline.Get());
  nvdx12::cmdPushConstants(cmd, args, nvdx12::BindPoint::eCompute);
  cmd->Dispatch(nvdx12::getGroupCounts(args.elementCount, 256), 1, 1);

  // 2D: one group per 16x16 pixels of a 1920x1080 image.
  const nvdx12::Extent2D groups = nvdx12::getGroupCounts(nvdx12::Extent2D{1920, 1080}, 16);
  cmd->Dispatch(groups.width, groups.height, 1);

  // Indirect dispatch needs the shared DISPATCH command signature (cmdDispatchIndirect in commands.hpp).
  nvdx12::ComPtr<ID3D12CommandSignature> dispatchSignature;
  NVDX12_CHECK(nvdx12::createDispatchIndirectSignature(dispatchSignature, device));
  NVDX12_DBG_NAME(dispatchSignature);
}
