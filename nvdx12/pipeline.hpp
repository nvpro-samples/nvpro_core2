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

#include "basic_types.hpp"
#include "compute_pipeline.hpp"  // createComputePipeline, getGroupCounts (nvvk/compute_pipeline.hpp)

namespace nvdx12 {

/*-------------------------------------------------------------------------------------------------
# Root signature and push constants

> The bindless root signature of a descriptor-heap renderer, the D3D12
> counterpart of a VK_EXT_descriptor_heap pipeline layout with push constants
> and push data. nvvk/pipeline.hpp holds the pipeline debugging dumps; the
> compute pipeline creation and getGroupCounts (nvvk/compute_pipeline.hpp)
> are in compute_pipeline.hpp, which this header includes.

Every compute PSO, ray-tracing state object and graphics PSO
(graphics_pipeline.hpp) shares one root signature:

- root parameter `kRootParamPushConstants` holds root constants (`b0`), the
  counterpart of Vulkan push constants: up to kPushConstantsMaxBytes set
  with cmdPushConstants, no upload memory involved,
- root parameter `kRootParamPushData` is a root SRV (`t0`) for push data
  beyond that size, copied into an UploadRing (UploadRing::cmdPushData),
- the CBV_SRV_UAV and sampler heaps are indexed directly from shaders
  (`ResourceDescriptorHeap[]` / `SamplerDescriptorHeap[]`).

A root signature holds 64 DWORDs; the root descriptor takes two, the
root constants 60 by default (240 bytes, where Vulkan
guarantees 128 and NVIDIA offers 256). Root constants use constant-buffer
packing, the push data natural layout; shaders/bindless.h declares both
(`PUSH_CONSTANTS`, `PUSH_DATA`).

DescriptorHeap::cmdBind sets the signature as the compute root signature on
direct and compute lists and as the graphics root signature on direct lists;
cmdPushConstants and UploadRing::cmdPushData bind their argument for the
bind point the caller names (a draw pushes to the graphics one). Graphics
pipelines pull their vertices through shader addresses; an input layout
would need the D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT
flag, which this signature does not set.

Differences with nvvk that follow from D3D12:
- The root-signature helpers (createBindlessRootSignature, cmdPushConstants,
  BindPoint) are the D3D12 counterpart of the pipeline layout and
  vkCmdPushConstants that nvvk leaves to the Vulkan API.
- Left out: dumpPipelineStats / dumpPipelineInternals. D3D12 has no
  pipeline-executable properties (VK_KHR_pipeline_executable_properties);
  use the vendor tools (Nsight Graphics, PIX) to inspect compiled shaders.

See usage_Pipeline() in pipeline.cpp.
-------------------------------------------------------------------------------------------------*/

constexpr uint32_t kRootParamPushData      = 0;
constexpr uint32_t kRootParamPushConstants = 1;

// Root constants within the 64-DWORD budget, beside the push-data root descriptor.
constexpr uint32_t kPushConstantsMaxDwords = 60;
constexpr uint32_t kPushConstantsMaxBytes  = kPushConstantsMaxDwords * sizeof(uint32_t);

// Which root signature's arguments a push addresses; DescriptorHeap::cmdBind
// sets both root signatures on a direct list.
enum class BindPoint : uint32_t
{
  eCompute,
  eGraphics,
  eBoth,
};

// The root signature described above, with `pushConstantDwords` root
// constants (none with 0). As with the nvvk creation helpers, the caller
// names the object (NVDX12_DBG_NAME).
HRESULT createBindlessRootSignature(ComPtr<ID3D12RootSignature>& rootSignature,
                                    ID3D12Device*                device,
                                    uint32_t                     pushConstantDwords = kPushConstantsMaxDwords);

// Root constants, the counterpart of vkCmdPushConstants: `size` bytes (a
// multiple of four, at most what the root signature was created with) land
// in `b0` of the chosen bind point(s), read through PUSH_CONSTANTS(T, name).
void cmdPushConstants(ID3D12GraphicsCommandList* cmd, const void* data, uint32_t size, BindPoint bindPoint, uint32_t rootParameterIndex = kRootParamPushConstants);

template <class T>
void cmdPushConstants(ID3D12GraphicsCommandList* cmd, const T& data, BindPoint bindPoint, uint32_t rootParameterIndex = kRootParamPushConstants)
{
  static_assert(sizeof(T) % sizeof(uint32_t) == 0 && sizeof(T) <= kPushConstantsMaxBytes,
                "push constants are DWORDs, at most kPushConstantsMaxBytes; larger data goes through UploadRing::cmdPushData");
  cmdPushConstants(cmd, &data, uint32_t(sizeof(T)), bindPoint, rootParameterIndex);
}

}  // namespace nvdx12
