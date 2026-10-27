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
#include <initializer_list>
#include <string>
#include <vector>

#include "basic_types.hpp"
#include "shader_compiler.hpp"

namespace nvdx12 {

/*-------------------------------------------------------------------------------------------------
# `nvdx12::createRayTracingPipeline`

> Ray-tracing state object from DXIL libraries and hit groups, the D3D12
> counterpart of vkCreateRayTracingPipelinesKHR with a
> VkRayTracingPipelineCreateInfoKHR, in the style of createComputePipeline.

`RayTracingPipelineCreateInfo` lists the libraries with the entry points each
exports (the shader stages), the hit groups that reference those entry
points (the shader groups), the payload / attribute sizes
(VkRayTracingPipelineInterfaceCreateInfoKHR), the recursion depth, the
pipeline flags and the global root signature (the pipeline layout). As with
the other creation helpers, the caller names the object (NVDX12_DBG_NAME).

Differences with Vulkan that follow from D3D12:
- Shaders and hit groups are identified by export name, not by group index:
  SBTGenerator (sbt_generator.hpp) takes the names and reads the identifiers
  from the state object.
- The state object uses one global root signature and no local root
  signatures, so a shader record is the 32-byte identifier alone (no
  shader record data).
- Ray-tracing pipeline libraries (D3D12_STATE_OBJECT_TYPE_COLLECTION),
  pipeline caches and dynamic stack sizes are not exposed.

See usage_RayTracingPipeline() in raytracing_pipeline.cpp.
-------------------------------------------------------------------------------------------------*/

struct RayTracingPipelineCreateInfo
{
  // A DXIL library (shader stages); `blob` is only read during createRayTracingPipeline.
  struct Library
  {
    const ShaderBlob*         blob{nullptr};
    std::vector<std::wstring> exports;  // entry points exported from this DXIL library
  };
  // A hit group (shader group) referencing exported entry points.
  struct HitGroup
  {
    std::wstring name;
    std::wstring closestHit;    // empty = none
    std::wstring anyHit;        // empty = none
    std::wstring intersection;  // empty = triangles, set = procedural primitive
  };

  std::vector<Library>  libraries;
  std::vector<HitGroup> hitGroups;
  ID3D12RootSignature*  globalRootSignature{nullptr};
  uint32_t              maxPipelineRayPayloadSize{64};
  uint32_t              maxPipelineRayHitAttributeSize{8};  // float2 barycentrics
  uint32_t              maxPipelineRayRecursionDepth{1};
  // The flags the traced geometry needs, such as ALLOW_OPACITY_MICROMAPS for OMM geometry.
  D3D12_RAYTRACING_PIPELINE_FLAGS flags{D3D12_RAYTRACING_PIPELINE_FLAG_NONE};

  // Convenience for filling the two lists. The library keeps a pointer to
  // `blob`, which must outlive createRayTracingPipeline: temporaries are rejected.
  Library&  addLibrary(const ShaderBlob& blob, std::initializer_list<std::wstring> exports);
  Library&  addLibrary(ShaderBlob&& blob, std::initializer_list<std::wstring> exports) = delete;
  HitGroup& addHitGroup(const std::wstring& name,
                        const std::wstring& closestHit,
                        const std::wstring& anyHit       = {},
                        const std::wstring& intersection = {});
};

// Creates the state object (D3D12_STATE_OBJECT_TYPE_RAYTRACING_PIPELINE);
// `pipeline` is reset first and stays null on failure.
HRESULT createRayTracingPipeline(ComPtr<ID3D12StateObject>& pipeline, ID3D12Device5* device, const RayTracingPipelineCreateInfo& createInfo);

}  // namespace nvdx12
