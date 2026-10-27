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

#include <cassert>
#include <cstdint>
#include <span>

#include "basic_types.hpp"
#include "resources.hpp"
#include "shader_compiler.hpp"

namespace nvdx12 {

/*-------------------------------------------------------------------------------------------------
# Compute pipelines

> Compute pipeline states and dispatch sizes, the D3D12 counterpart of
> nvvk/compute_pipeline.hpp (getGroupCounts) plus the compute-pipeline creation
> the Vulkan samples write with vkCreateComputePipelines.

- `createComputePipeline`: compute PSO from compiled DXIL on a root signature
  (pipeline.hpp, createBindlessRootSignature), the caller names it.
- `createDispatchIndirectSignature`: the command signature ExecuteIndirect
  needs for indirect dispatches (vkCmdDispatchIndirect has none).
- `getGroupCounts`: thread groups covering a 1D, 2D or 3D size, as nvvk's.

Differences with nvvk that follow from D3D12:
- A pipeline state takes the root signature (the pipeline layout) and DXIL
  directly: there is no shader module, and the flags carry the pre-SM 6.9
  RayQuery opt-ins instead of VkPipelineCreateFlags.
- No specialization constants (nvvk::Specialization): DXIL has none, use
  preprocessor defines (ShaderDefine) when compiling instead.

The definitions are in pipeline.cpp; pipeline.hpp includes this header.
-------------------------------------------------------------------------------------------------*/

// Command signature with a single DISPATCH argument (D3D12_DISPATCH_ARGUMENTS), for cmdDispatchIndirect.
HRESULT createDispatchIndirectSignature(ComPtr<ID3D12CommandSignature>& signature, ID3D12Device* device);

// Compute PSO from compiled DXIL. `flags` carries the pre-SM 6.9 RayQuery
// opt-ins (opacity micromaps) when the shader needs them.
HRESULT createComputePipeline(ComPtr<ID3D12PipelineState>& pipeline,
                              ID3D12Device*                device,
                              ID3D12RootSignature*         rootSignature,
                              const ShaderBlob&            blob,
                              D3D12_PIPELINE_STATE_FLAGS   flags = D3D12_PIPELINE_STATE_FLAG_NONE);
HRESULT createComputePipeline(ComPtr<ID3D12PipelineState>& pipeline,
                              ID3D12Device*                device,
                              ID3D12RootSignature*         rootSignature,
                              std::span<const uint8_t>     dxil,
                              D3D12_PIPELINE_STATE_FLAGS   flags = D3D12_PIPELINE_STATE_FLAG_NONE);

// Return the number of thread groups needed for a given size, for Dispatch
// (nvvk::getGroupCounts). Workgroup sizes are the shader's thread-group sizes, never 0.
inline Extent2D getGroupCounts(const Extent2D& size, uint32_t workgroupSize)
{
  assert(workgroupSize > 0);
  return Extent2D{(size.width + (workgroupSize - 1)) / workgroupSize, (size.height + (workgroupSize - 1)) / workgroupSize};
}

inline Extent2D getGroupCounts(const Extent2D& size, const Extent2D& workgroupSize)
{
  assert(workgroupSize.width > 0 && workgroupSize.height > 0);
  return Extent2D{(size.width + (workgroupSize.width - 1)) / workgroupSize.width,
                  (size.height + (workgroupSize.height - 1)) / workgroupSize.height};
}

inline Extent3D getGroupCounts(const Extent3D& size, const Extent3D& workgroupSize)
{
  assert(workgroupSize.width > 0 && workgroupSize.height > 0 && workgroupSize.depth > 0);
  return Extent3D{(size.width + (workgroupSize.width - 1)) / workgroupSize.width,
                  (size.height + (workgroupSize.height - 1)) / workgroupSize.height,
                  (size.depth + (workgroupSize.depth - 1)) / workgroupSize.depth};
}

inline uint32_t getGroupCounts(uint32_t size, uint32_t workgroupSize)
{
  assert(workgroupSize > 0);
  return (size + (workgroupSize - 1)) / workgroupSize;
}

}  // namespace nvdx12
