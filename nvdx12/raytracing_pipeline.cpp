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

#include "raytracing_pipeline.hpp"

#include <d3dx12.h>

#include <nvutils/logger.hpp>

#include "check_error.hpp"
#include "debug_util.hpp"

namespace nvdx12 {

RayTracingPipelineCreateInfo::Library& RayTracingPipelineCreateInfo::addLibrary(const ShaderBlob& blob,
                                                                                std::initializer_list<std::wstring> exports)
{
  Library& library = libraries.emplace_back();
  library.blob     = &blob;
  library.exports.assign(exports.begin(), exports.end());
  return library;
}

RayTracingPipelineCreateInfo::HitGroup& RayTracingPipelineCreateInfo::addHitGroup(const std::wstring& name,
                                                                                  const std::wstring& closestHit,
                                                                                  const std::wstring& anyHit,
                                                                                  const std::wstring& intersection)
{
  return hitGroups.emplace_back(HitGroup{name, closestHit, anyHit, intersection});
}

HRESULT createRayTracingPipeline(ComPtr<ID3D12StateObject>& pipeline, ID3D12Device5* device, const RayTracingPipelineCreateInfo& createInfo)
{
  pipeline.Reset();

  CD3DX12_STATE_OBJECT_DESC stateObject(D3D12_STATE_OBJECT_TYPE_RAYTRACING_PIPELINE);

  for(size_t i = 0; i < createInfo.libraries.size(); i++)
  {
    const RayTracingPipelineCreateInfo::Library& library = createInfo.libraries[i];
    if(!library.blob || !library.blob->valid())
    {
      LOGE("Ray tracing pipeline: a DXIL library blob is not valid\n");
      return E_INVALIDARG;
    }
    // The subobject copies the bytecode descriptor; the DXIL stays in the blob.
    const D3D12_SHADER_BYTECODE bytecode  = library.blob->bytecode();
    auto*                       subobject = stateObject.CreateSubobject<CD3DX12_DXIL_LIBRARY_SUBOBJECT>();
    subobject->SetDXILLibrary(&bytecode);
    for(size_t j = 0; j < library.exports.size(); j++)
      subobject->DefineExport(library.exports[j].c_str());
  }

  for(size_t i = 0; i < createInfo.hitGroups.size(); i++)
  {
    const RayTracingPipelineCreateInfo::HitGroup& hitGroup = createInfo.hitGroups[i];

    auto* subobject = stateObject.CreateSubobject<CD3DX12_HIT_GROUP_SUBOBJECT>();
    subobject->SetHitGroupExport(hitGroup.name.c_str());
    subobject->SetHitGroupType(hitGroup.intersection.empty() ? D3D12_HIT_GROUP_TYPE_TRIANGLES : D3D12_HIT_GROUP_TYPE_PROCEDURAL_PRIMITIVE);
    if(!hitGroup.closestHit.empty())
      subobject->SetClosestHitShaderImport(hitGroup.closestHit.c_str());
    if(!hitGroup.anyHit.empty())
      subobject->SetAnyHitShaderImport(hitGroup.anyHit.c_str());
    if(!hitGroup.intersection.empty())
      subobject->SetIntersectionShaderImport(hitGroup.intersection.c_str());
  }

  auto* shaderConfig = stateObject.CreateSubobject<CD3DX12_RAYTRACING_SHADER_CONFIG_SUBOBJECT>();
  shaderConfig->Config(createInfo.maxPipelineRayPayloadSize, createInfo.maxPipelineRayHitAttributeSize);

  // PIPELINE_CONFIG1 (pipeline flags) needs ray tracing tier 1.1; without flags the tier 1.0 subobject does.
  if(createInfo.flags == D3D12_RAYTRACING_PIPELINE_FLAG_NONE)
  {
    auto* pipelineConfig = stateObject.CreateSubobject<CD3DX12_RAYTRACING_PIPELINE_CONFIG_SUBOBJECT>();
    pipelineConfig->Config(createInfo.maxPipelineRayRecursionDepth);
  }
  else
  {
    auto* pipelineConfig = stateObject.CreateSubobject<CD3DX12_RAYTRACING_PIPELINE_CONFIG1_SUBOBJECT>();
    pipelineConfig->Config(createInfo.maxPipelineRayRecursionDepth, createInfo.flags);
  }

  auto* rootSignature = stateObject.CreateSubobject<CD3DX12_GLOBAL_ROOT_SIGNATURE_SUBOBJECT>();
  rootSignature->SetRootSignature(createInfo.globalRootSignature);

  return NVDX12_FAIL_REPORT(device->CreateStateObject(stateObject, IID_PPV_ARGS(&pipeline)));
}

}  // namespace nvdx12

//--------------------------------------------------------------------------------------------------
// Usage example
//--------------------------------------------------------------------------------------------------
[[maybe_unused]] static void usage_RayTracingPipeline()
{
  ID3D12Device5*         device        = nullptr;
  ID3D12RootSignature*   rootSignature = nullptr;  // createBindlessRootSignature() in pipeline.hpp
  nvdx12::ShaderCompiler compiler;                 // see usage_ShaderCompiler()

  // One DXIL library per stage, compiled as lib_6_x; a library may export several entry points.
  nvdx12::ShaderBlob raygen = compiler.compileFile("pathtrace.rgen.hlsl", "lib_6_6");
  nvdx12::ShaderBlob miss   = compiler.compileFile("pathtrace.rmiss.hlsl", "lib_6_6");
  nvdx12::ShaderBlob hit    = compiler.compileFile("pathtrace.rchit.hlsl", "lib_6_6");

  nvdx12::RayTracingPipelineCreateInfo createInfo;
  createInfo.addLibrary(raygen, {L"rayGenMain"});
  createInfo.addLibrary(miss, {L"missMain", L"shadowMissMain"});
  createInfo.addLibrary(hit, {L"closestHitMain", L"anyHitMain"});
  createInfo.addHitGroup(L"HitGroup", L"closestHitMain", L"anyHitMain");
  createInfo.globalRootSignature          = rootSignature;
  createInfo.maxPipelineRayPayloadSize    = 48;
  createInfo.maxPipelineRayRecursionDepth = 2;

  nvdx12::ComPtr<ID3D12StateObject> pipeline;
  NVDX12_CHECK(nvdx12::createRayTracingPipeline(pipeline, device, createInfo));
  NVDX12_DBG_CUSTOM_NAME(pipeline, "path tracer");

  // The shader binding table takes the exports by name (see usage_SBTGenerator()).

  // Hot reload: build the replacement first, then swap, so a failed compile keeps the old state object.
  nvdx12::ComPtr<ID3D12StateObject> replacement;
  if(SUCCEEDED(nvdx12::createRayTracingPipeline(replacement, device, createInfo)))
  {
    pipeline = std::move(replacement);
  }
}
