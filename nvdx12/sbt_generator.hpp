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

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "basic_types.hpp"

namespace nvdx12 {

/*-------------------------------------------------------------------------------------------------
# class nvdx12::SBTGenerator

> Shader binding table layout and contents for a ray-tracing state object
> (createRayTracingPipeline), the D3D12 counterpart of nvvk::SBTGenerator.

The state object has no local root arguments, so every record is the
32-byte shader identifier of an export. Records of each group type are
laid out in `addExport` order, and the hit-group index an instance selects
(InstanceContributionToHitGroupIndex) is the index of its addHitGroup call.
Raygen records use a D3D12_RAYTRACING_SHADER_TABLE_BYTE_ALIGNMENT (64-byte)
stride, so any raygen record can start RayGenerationShaderRecord; the other
groups use a 32-byte stride, and their tables start on a 64-byte boundary.

## Usage
- Register the exports per group type (`addRaygen`, `addMiss`, `addHitGroup`, `addCallable`)
- After the pipeline creation, call `calculateSBTBufferSize(pipeline)` which
  returns the size of the buffer to create
- Create a host-visible buffer of that size (alignment `getBufferAlignment()`),
  then call `populateSBTBuffer()` to write the identifiers
- Use `getSBTRegions()` or `getDispatchRaysDesc()` for DispatchRays

Rebuild (calculate and populate again) after every pipeline recompile: the
identifiers belong to one state object.

Differences with nvvk that follow from D3D12:
- D3D12 identifies shaders and hit groups by export name, not by group
  index, so the records are registered by name (`addExport` and the D3D12
  extras `addRaygen` / `addMiss` / `addHitGroup` / `addCallable`) instead of
  being read from the pipeline create info: there is no `addIndices` /
  `addIndex`, and calculateSBTBufferSize takes only the pipeline.
- The identifier size and alignments are D3D12 constants rather than device
  properties, so there is no `init` / `deinit`.
- No `addData`: the state objects have no local root signatures, hence no
  shader record data.
- The pipeline is stored without a reference, as nvvk stores the VkPipeline:
  it must stay alive until populateSBTBuffer.
- `getDispatchRaysDesc` (D3D12 extra) returns the regions plus the launch size.

See usage_SBTGenerator() in sbt_generator.cpp.
-------------------------------------------------------------------------------------------------*/

class SBTGenerator
{
public:
  enum GroupType
  {
    eRaygen,
    eMiss,
    eHit,
    eCallable
  };

  // Address regions of all groups, in the form D3D12_DISPATCH_RAYS_DESC takes.
  struct Regions
  {
    D3D12_GPU_VIRTUAL_ADDRESS_RANGE            raygen{};
    D3D12_GPU_VIRTUAL_ADDRESS_RANGE_AND_STRIDE miss{};
    D3D12_GPU_VIRTUAL_ADDRESS_RANGE_AND_STRIDE hit{};
    D3D12_GPU_VIRTUAL_ADDRESS_RANGE_AND_STRIDE callable{};
  };

  // Appends a record for `exportName` (shader entry point or hit-group name) to a group.
  void addExport(GroupType type, const std::wstring& exportName) { m_exports[type].push_back(exportName); }
  void addRaygen(const std::wstring& exportName) { addExport(eRaygen, exportName); }
  void addMiss(const std::wstring& exportName) { addExport(eMiss, exportName); }
  void addHitGroup(const std::wstring& hitGroupName) { addExport(eHit, hitGroupName); }
  void addCallable(const std::wstring& exportName) { addExport(eCallable, exportName); }

  // Lays out the groups for `rayPipeline` (kept for populateSBTBuffer) and
  // returns the size of the buffer to create.
  size_t calculateSBTBufferSize(ID3D12StateObject* rayPipeline);

  // Required alignment of the buffer address (D3D12_RAYTRACING_SHADER_TABLE_BYTE_ALIGNMENT).
  uint32_t getBufferAlignment() const { return D3D12_RAYTRACING_SHADER_TABLE_BYTE_ALIGNMENT; }

  // Writes the identifiers of the pipeline given to calculateSBTBufferSize
  // into `bufferData`, the mapping of a buffer of at least
  // calculateSBTBufferSize() bytes at `bufferAddress` (aligned to getBufferAlignment()).
  // E_INVALIDARG when an export has no identifier or the buffer is too small.
  HRESULT populateSBTBuffer(D3D12_GPU_VIRTUAL_ADDRESS bufferAddress, size_t bufferSize, void* bufferData);

  // After populateSBTBuffer: the region of one group, starting at record `indexOffset`.
  // For miss, hit and callable groups the start must stay 64-byte aligned
  // (even `indexOffset`); an empty region is returned otherwise.
  D3D12_GPU_VIRTUAL_ADDRESS_RANGE_AND_STRIDE getSBTRegion(GroupType type, uint32_t indexOffset = 0) const;

  // The regions of all groups; `rayGenIndexOffset` selects which raygen record to launch.
  Regions getSBTRegions(uint32_t rayGenIndexOffset = 0) const;

  // The regions plus the launch size, ready for DispatchRays.
  D3D12_DISPATCH_RAYS_DESC getDispatchRaysDesc(uint32_t width, uint32_t height, uint32_t depth = 1, uint32_t rayGenIndexOffset = 0) const;

  // Forgets the exports, the pipeline and the buffer; the generator can be filled again.
  void reset();
  // Forgets the buffer layout, addresses and pipeline, keeping the exports
  // (calculateSBTBufferSize starts again from there).
  void resetBuffer();

private:
  uint32_t getGroupIndexCount(GroupType type) const { return uint32_t(m_exports[type].size()); }
  // Identifier only; raygen records each start a table, hence the table alignment.
  uint32_t getGroupStride(GroupType type) const
  {
    return type == eRaygen ? D3D12_RAYTRACING_SHADER_TABLE_BYTE_ALIGNMENT : D3D12_RAYTRACING_SHADER_RECORD_BYTE_ALIGNMENT;
  }
  uint32_t getGroupSize(GroupType type) const { return getGroupStride(type) * getGroupIndexCount(type); }

  std::array<std::vector<std::wstring>, 4> m_exports;            // per group type, the export names in record order
  std::array<uint64_t, 4>                  m_groupOffsets{};     // byte offset of each group inside the buffer
  std::array<D3D12_GPU_VIRTUAL_ADDRESS, 4> m_bufferAddresses{};  // GPU address of each group after populateSBTBuffer
  size_t                                   m_bufferSize{0};      // result of calculateSBTBufferSize
  ID3D12StateObject*                       m_pipeline{nullptr};  // from calculateSBTBufferSize, not referenced
};

}  // namespace nvdx12
