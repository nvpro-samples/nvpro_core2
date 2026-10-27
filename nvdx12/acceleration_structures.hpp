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
#include <queue>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "basic_types.hpp"  // first: Windows headers with NOMINMAX

#include <glm/glm.hpp>

#include "resource_allocator.hpp"
#include "uploader_interface.hpp"

namespace nvdx12 {

/*-------------------------------------------------------------------------------------------------
# Acceleration structures

> Build-input helpers for bottom- and top-level acceleration structures,
> opacity micromap arrays (DXR 1.2) and compaction, the D3D12 counterparts
> of nvvk/acceleration_structures.hpp and the Vulkan micromap builds.

# `nvdx12::makeTriangleGeometry` / `nvdx12::makeOmmLinkage`
Fill a triangle geometry desc and the opacity-micromap linkage of one.

# `nvdx12::toTransformMatrix`
Converts a `glm::mat4` to the row-major 3x4 matrix of
D3D12_RAYTRACING_INSTANCE_DESC::Transform (nvvk::toTransformMatrixKHR).

# `nvdx12::AccelerationStructureBuildData`
Collects the geometries of a BLAS or the instance array of a TLAS, queries
the prebuild sizes and records builds and refits, as nvvk's:
- `addGeometry` / `addOmmGeometry`: adds triangles, with or without opacity micromaps (BLAS).
- `setInstances`: the instance array (TLAS).
- `finalizeGeometry`: fills `buildInfo` and returns the size information (also kept in `sizeInfo`).
- `makeCreateInfo`: type and size for ResourceAllocator::createAcceleration.
- `cmdBuildAccelerationStructure` / `cmdUpdateAccelerationStructure`: records the build or refit.
- `hasCompactFlag`.

# `nvdx12::OpacityMicromapBuildData`
The same steps (`setInputs`, `finalize`, `makeCreateInfo`, `cmdBuild`) for an opacity micromap array.

# Compaction
`cmdEmitCompactedSizes` writes the compacted sizes, `cmdCopyCompacted` copies
a structure into its compacted storage. `cmdAccelerationStructureBarrier`
orders builds against later builds, copies and traces, including the reuse
of the scratch memory by a later build.

Storage comes from ResourceAllocator::createAcceleration (kAccelerationStructureAlignment)
and plain device buffers for scratch (kScratchAlignment).

# class `nvdx12::AccelerationStructureBuilder`
Builds a set of BLAS in batches that fit a scratch buffer and a size budget,
and compacts them, as nvvk's:
- `getScratchSize`: scratch size for a budget (all at once if they fit, at least the largest).
- `cmdCreateBlas`: creates and builds as many BLAS as `hintMaxBudget` allows,
  reusing the scratch buffer in sub-batches separated by barriers; S_FALSE
  (VK_INCOMPLETE) asks to be called again after the submission.
- `cmdCompactBlas`: after the build submission completed, replaces the
  compactable BLAS of the oldest pending batch by compacted copies.
- `destroyNonCompactedBlas`: frees the originals once the copies completed.
- `getStatistics`: original vs compacted bytes.

# class `nvdx12::AccelerationStructureHelper`
Submit-and-wait BLAS sets (built and compacted through the builder) and a
TLAS over a D3D12_RAYTRACING_INSTANCE_DESC array uploaded through a
CmdUploaderInterface, with in-place updates (`tlasSubmitUpdateAndWait`).

Differences with nvvk that follow from D3D12:
- Acceleration structures are addressed by their GPU virtual address, not
  by a VkAccelerationStructureKHR handle, and the build inputs are one
  D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS (`buildInfo`) over
  D3D12_RAYTRACING_GEOMETRY_DESC (`asGeometry`): there are no separate
  build ranges (asBuildRangeInfo, AccelerationStructureGeometryInfo).
- A TLAS takes its instance array in the inputs rather than as an instance
  geometry, so `setInstances` replaces makeInstanceGeometry + addGeometry.
- The builds do not record a barrier after themselves, unlike nvvk's: the
  barrier needed depends on what consumes the result (a later build, a
  copy, a trace or a compute pass), and independent builds recorded back to
  back (BLAS with their own scratch) may then overlap on the GPU. Record
  `cmdAccelerationStructureBarrier` (nvvk::accelerationStructureBarrier,
  named after the nvdx12 cmd* recording convention) where needed.
- D3D12 extras: opacity micromap geometries and arrays (DXR 1.2), the
  compaction helpers, kScratchAlignment.
- AccelerationStructureBuilder: the results are HRESULTs, S_OK for
  VK_SUCCESS and S_FALSE for VK_INCOMPLETE. There are no query pools: the
  compacted sizes are written by EmitRaytracingAccelerationStructurePostbuildInfo
  into a device buffer and copied to a readback buffer in the build command
  list; the builder owns both (one UINT64 per BLAS, indexed like the BLAS)
  and frees them in deinit. Only BLAS built with ALLOW_COMPACTION get a
  size and a compacted copy (nvvk queries every BLAS of a batch once one is
  compactable). The builder records the barriers its own work needs (after
  each scratch sub-batch, before the size emission, after the compaction
  copies), so the compacted BLAS can be consumed in the same command list.
  The scratch alignment is the constant kScratchAlignment (no device query).
  A BLAS whose scratch exceeds `scratchSize` fails with E_INVALIDARG instead
  of looping.
- AccelerationStructureHelper: `blasSubmitBuildAndWait` takes one
  D3D12_RAYTRACING_GEOMETRY_DESC per BLAS (no AccelerationStructureGeometryInfo);
  queues are QueueInfo with the D3D12_COMMAND_LIST_TYPE in `familyIndex`, and
  the transient pool is an ID3D12CommandAllocator. `tlasSubmitUpdateAndWait`
  with a different instance count rebuilds as nvvk does, but also updates the
  instance count of the build inputs and grows the instance buffer, the TLAS
  and the scratch when needed (D3D12 inputs carry the count). Growing the
  TLAS destroys and recreates it (ResourceAllocator::createAcceleration):
  `tlas.gpuAddress`, its SRV slot and `tlas.shaderHandle()` may all change,
  as `tlasInstancesBuffer.gpuAddress` does when the instance buffer grows.
  Callers that stored any of them (descriptor writes, push constants,
  shader tables) must read them again after each tlasSubmitUpdateAndWait
  that changed the instance count. The helper takes a CmdUploaderInterface
  (FrameUploader) where nvvk takes a StagingUploader, which nvdx12 does not
  have: the instance upload is recorded into the helper's command list, which
  it waits for, and the uploader's owner releases the staging afterwards
  (FrameUploader::releaseCompletedAllocations).

See usage_AccelerationStructures() and usage_AccelerationStructureBuilder()
in acceleration_structures.cpp.
-------------------------------------------------------------------------------------------------*/

constexpr uint64_t kAccelerationStructureAlignment = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BYTE_ALIGNMENT;  // 256
constexpr uint64_t kScratchAlignment               = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BYTE_ALIGNMENT;

// Triangle geometry with `vertexFormat` positions. `indexFormat` is
// DXGI_FORMAT_R32_UINT / R16_UINT, or UNKNOWN for non-indexed triangles.
D3D12_RAYTRACING_GEOMETRY_DESC makeTriangleGeometry(D3D12_GPU_VIRTUAL_ADDRESS vertexAddress,
                                                    uint32_t                  vertexStride,
                                                    uint32_t                  vertexCount,
                                                    D3D12_GPU_VIRTUAL_ADDRESS indexAddress,
                                                    DXGI_FORMAT               indexFormat,
                                                    uint32_t                  indexCount,
                                                    D3D12_RAYTRACING_GEOMETRY_FLAGS flags = D3D12_RAYTRACING_GEOMETRY_FLAG_NONE,
                                                    DXGI_FORMAT vertexFormat = DXGI_FORMAT_R32G32B32_FLOAT);

// OMM linkage for a triangle geometry. The per-triangle OMM index buffer
// uses `indexFormat` R32_UINT / R16_UINT / R8_UINT and may hold the
// special indices (D3D12_RAYTRACING_OPACITY_MICROMAP_SPECIAL_INDEX_*).
D3D12_RAYTRACING_GEOMETRY_OMM_LINKAGE_DESC makeOmmLinkage(D3D12_GPU_VIRTUAL_ADDRESS ommArrayAddress,
                                                          D3D12_GPU_VIRTUAL_ADDRESS ommIndexAddress,
                                                          uint32_t                  ommIndexStride,
                                                          DXGI_FORMAT               indexFormat,
                                                          uint32_t                  baseLocation = 0);

// Row-major 3x4 instance transform from a column-major glm::mat4, the
// D3D12 counterpart of nvvk::toTransformMatrixKHR: fills
// D3D12_RAYTRACING_INSTANCE_DESC::Transform.
void toTransformMatrix(const glm::mat4& matrix, FLOAT out[3][4]);

// Build inputs of one acceleration structure. OMM geometries point into the
// owned `ommTriangles` / `ommLinkages` storage, which every call re-links,
// so the object may be copied or moved. The calls read the descriptions on
// the CPU; only the GPU buffers they reference must live until the build ran.
struct AccelerationStructureBuildData
{
  D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE asType{D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL};
  // Collection of geometries for the acceleration structure (BLAS).
  std::vector<D3D12_RAYTRACING_GEOMETRY_DESC> asGeometry;
  // Build information required for the acceleration structure.
  D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS buildInfo{};
  // Size information for the acceleration structure build resources.
  D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO sizeInfo{};
  // Owned storage referenced by OMM geometries.
  std::vector<D3D12_RAYTRACING_GEOMETRY_TRIANGLES_DESC>   ommTriangles;
  std::vector<D3D12_RAYTRACING_GEOMETRY_OMM_LINKAGE_DESC> ommLinkages;

  // BLAS: plain triangles.
  void addGeometry(const D3D12_RAYTRACING_GEOMETRY_DESC& geometry);
  // BLAS: triangles with opacity micromaps (D3D12_RAYTRACING_GEOMETRY_TYPE_OMM_TRIANGLES).
  void addOmmGeometry(const D3D12_RAYTRACING_GEOMETRY_TRIANGLES_DESC&   triangles,
                      const D3D12_RAYTRACING_GEOMETRY_OMM_LINKAGE_DESC& linkage,
                      D3D12_RAYTRACING_GEOMETRY_FLAGS                   flags = D3D12_RAYTRACING_GEOMETRY_FLAG_NONE);
  // TLAS: `instanceCount` D3D12_RAYTRACING_INSTANCE_DESC at `instanceAddress`,
  // the counterpart of nvvk's makeInstanceGeometry + addGeometry. Sets `asType` to TOP_LEVEL.
  void setInstances(D3D12_GPU_VIRTUAL_ADDRESS instanceAddress, uint32_t instanceCount);

  // Configures `buildInfo` with the geometry and `flags`, queries the prebuild sizes and returns them.
  D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO finalizeGeometry(ID3D12Device5* device,
                                                                         D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAGS flags);

  // Type and size (sizeInfo.ResultDataMaxSizeInBytes) for ResourceAllocator::createAcceleration.
  AccelerationStructureCreateInfo makeCreateInfo() const;

  // Records the build into `accelerationStructure` (makeCreateInfo().size
  // bytes) using `scratchAddress` (sizeInfo.ScratchDataSizeInBytes bytes).
  // Records no barrier: see cmdAccelerationStructureBarrier.
  void cmdBuildAccelerationStructure(ID3D12GraphicsCommandList4* cmd,
                                     D3D12_GPU_VIRTUAL_ADDRESS   accelerationStructure,
                                     D3D12_GPU_VIRTUAL_ADDRESS   scratchAddress);
  // In-place refit of `accelerationStructure` (built with ALLOW_UPDATE) using
  // `scratchAddress` (sizeInfo.UpdateScratchDataSizeInBytes bytes). Records no barrier.
  void cmdUpdateAccelerationStructure(ID3D12GraphicsCommandList4* cmd,
                                      D3D12_GPU_VIRTUAL_ADDRESS   accelerationStructure,
                                      D3D12_GPU_VIRTUAL_ADDRESS   scratchAddress);

  // Checks if the compact flag is set for the build.
  bool hasCompactFlag() const
  {
    return (buildInfo.Flags & D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_COMPACTION) != 0;
  }
};

// Largest build scratch size over a set of finalized build data, for a shared scratch buffer.
uint64_t getMaxScratchSize(std::span<const AccelerationStructureBuildData> buildData);

// Opacity micromap array (DXR 1.2). `inputData` holds the packed opacity
// bits, `perOmmDescs` the D3D12_RAYTRACING_OPACITY_MICROMAP_DESC array (byte
// offset, subdivision level, format), `histogram` the per (subdivision,
// format) counts, which also give the number of OMMs: the build inputs
// describe the whole array with one desc (NumDescs 1).
struct OpacityMicromapBuildData
{
  std::vector<D3D12_RAYTRACING_OPACITY_MICROMAP_HISTOGRAM_ENTRY> histogram;
  D3D12_RAYTRACING_OPACITY_MICROMAP_ARRAY_DESC                   arrayDesc{};
  D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS           buildInfo{};
  D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO          sizeInfo{};

  void setInputs(D3D12_GPU_VIRTUAL_ADDRESS inputData, D3D12_GPU_VIRTUAL_ADDRESS perOmmDescs, uint32_t perOmmDescStride);
  // Fills `buildInfo` from the histogram and `flags`, queries the prebuild sizes and returns them.
  const D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO& finalize(ID3D12Device5* device,
                                                                        D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAGS flags);
  void cmdBuild(ID3D12GraphicsCommandList4* cmd, D3D12_GPU_VIRTUAL_ADDRESS destination, D3D12_GPU_VIRTUAL_ADDRESS scratch);

  // Type (OPACITY_MICROMAP_ARRAY) and size (sizeInfo.ResultDataMaxSizeInBytes) for createAcceleration.
  AccelerationStructureCreateInfo makeCreateInfo() const;
};

// Writes the compacted size (one UINT64 each) of `sources` to `destination`,
// a default-heap buffer the caller copies to a readback buffer afterwards.
void cmdEmitCompactedSizes(ID3D12GraphicsCommandList4*                cmd,
                           std::span<const D3D12_GPU_VIRTUAL_ADDRESS> sources,
                           D3D12_GPU_VIRTUAL_ADDRESS                  destination);

// Copies `source` into `destination` in compacted form.
void cmdCopyCompacted(ID3D12GraphicsCommandList4* cmd, D3D12_GPU_VIRTUAL_ADDRESS destination, D3D12_GPU_VIRTUAL_ADDRESS source);

// Global barrier from acceleration-structure writes (builds, copies) to
// `dstSync`: later builds, copies, compacted-size emissions and traces by
// default (nvvk::accelerationStructureBarrier). Later builds and copies may
// write the structures again (an update in place); when `dstSync` includes
// builds, the barrier also orders the scratch (UAV) accesses, so a later
// build may reuse the scratch memory.
void cmdAccelerationStructureBarrier(ID3D12GraphicsCommandList7* cmd,
                                     D3D12_BARRIER_SYNC dstSync = D3D12_BARRIER_SYNC_BUILD_RAYTRACING_ACCELERATION_STRUCTURE
                                                                  | D3D12_BARRIER_SYNC_COPY_RAYTRACING_ACCELERATION_STRUCTURE
                                                                  | D3D12_BARRIER_SYNC_EMIT_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO
                                                                  | D3D12_BARRIER_SYNC_RAYTRACING | D3D12_BARRIER_SYNC_COMPUTE_SHADING);

static_assert(sizeof(D3D12_RAYTRACING_INSTANCE_DESC) == 64, "D3D12_RAYTRACING_INSTANCE_DESC is 64 bytes");

/*-------------------------------------------------------------------------------------------------
 Manages the construction and compaction of bottom-level acceleration structures (BLAS)

 Usage (as nvvk's):
 - Initialize the builder with a resource allocator via `init`
 - Query the scratch buffer size via `getScratchSize` and create the scratch buffer
 Within a loop
   - Call `cmdCreateBlas` to create all or a range of the BLAS
   - Submit the command list and wait
   - Call `cmdCompactBlas` to compact the BLAS built thus far, submit and wait
   - Call `destroyNonCompactedBlas` to destroy the original BLAS that were compacted
   - Continue the loop while `cmdCreateBlas` returned S_FALSE
 - Call `getStatistics` for the compaction results, `deinit` to clean up.
--------------------------------------------------------------------------------------------------- */
class AccelerationStructureBuilder
{
public:
  AccelerationStructureBuilder()                                               = default;
  AccelerationStructureBuilder(const AccelerationStructureBuilder&)            = delete;
  AccelerationStructureBuilder& operator=(const AccelerationStructureBuilder&) = delete;
  AccelerationStructureBuilder(AccelerationStructureBuilder&& other) noexcept { *this = std::move(other); }
  AccelerationStructureBuilder& operator=(AccelerationStructureBuilder&& other) noexcept;

  ~AccelerationStructureBuilder() { assert(!m_queryPool.sizes && "Missing deinit()"); }

  void init(ResourceAllocator* allocator);
  void deinit();

  struct Stats
  {
    uint64_t totalOriginalSize = 0;
    uint64_t totalCompactSize  = 0;

    std::string toString() const;
  };

  // Creates the BLAS of `blasBuildData` (finalized) into `blasAccel` (same
  // size), building as many in parallel as `scratchSize` and `hintMaxBudget`
  // (sum of the structure sizes built by this call) allow.
  // Returns S_OK when the entire input was processed, S_FALSE when this
  // function must be called again (after submitting `cmd`); other results are errors.
  HRESULT cmdCreateBlas(ID3D12GraphicsCommandList7*                cmd,
                        std::span<AccelerationStructureBuildData>& blasBuildData,  // List of the BLAS to build
                        std::span<AccelerationStructure>&          blasAccel,  // List of the acceleration structures
                        D3D12_GPU_VIRTUAL_ADDRESS                  scratchAddress,  // Address of the scratch buffer
                        uint64_t                                   scratchSize,     // Size of the scratch buffer
                        uint64_t hintMaxBudget = 512'000'000  // Ceiling for sum of acceleration structure sizes built
  );

  // Compacts the BLAS of the oldest batch built by cmdCreateBlas, whose
  // command list must have completed. Returns S_FALSE while batches remain.
  HRESULT cmdCompactBlas(ID3D12GraphicsCommandList7*                cmd,
                         std::span<AccelerationStructureBuildData>& blasBuildData,
                         std::span<AccelerationStructure>&          blasAccel);

  // Destroys the original BLAS that were compacted (once the copies completed).
  void destroyNonCompactedBlas();

  // Statistics about the compacted BLAS.
  Stats getStatistics() const { return m_stats; };

  // Scratch size strategy, as nvvk's: the accumulated (aligned) size if it
  // fits `hintMaxBudget`, else the larger of the budget and the largest size.
  uint64_t getScratchSize(uint64_t hintMaxBudget, const std::span<AccelerationStructureBuildData>& buildData) const;

  // Alignment of each BLAS's scratch range.
  uint64_t getScratchAlignment() const { return m_scratchAlignment; }

private:
  // The compacted-size "query pool": one UINT64 per BLAS.
  struct QueryPool
  {
    Buffer   sizes;     // default heap, written by EmitRaytracingAccelerationStructurePostbuildInfo
    Buffer   readback;  // readback heap copy of `sizes`
    uint32_t count{0};
  };

  void    destroy();
  void    destroyQueryPool();
  HRESULT initializeQueryPoolIfNeeded(const std::span<AccelerationStructureBuildData>& blasBuildData);
  HRESULT cmdBuildAccelerationStructures(ID3D12GraphicsCommandList7* cmd,
                                         uint64_t&                   budgetUsed,  // in/out: budget used so far
                                         std::span<AccelerationStructureBuildData>& blasBuildData,
                                         std::span<AccelerationStructure>&          blasAccel,
                                         D3D12_GPU_VIRTUAL_ADDRESS                  scratchAddress,
                                         D3D12_GPU_VIRTUAL_ADDRESS                  scratchAddressEnd,
                                         uint64_t                                   hintMaxBudget,
                                         QueryPool*                                 queryPool);

  struct CompactBatchInfo
  {
    uint32_t startIdx{};
    uint32_t endIdx{};
  };

  ResourceAllocator*           m_alloc{};             // Allocator for the creation of acceleration structures
  QueryPool                    m_queryPool{};         // Compacted sizes for BLAS compaction
  uint32_t                     m_currentBlasIdx{};    // Index to the current BLAS being processed
  uint64_t                     m_scratchAlignment{};  // Alignment of the scratch buffer
  std::queue<CompactBatchInfo> m_batches;             // Queue of compact batches to be processed

  std::vector<AccelerationStructure> m_cleanupBlasAccel;  // List of BLAS to be cleaned up

  Stats m_stats;  // Statistics about the compacted BLAS
};

// Helper for building BLAS sets and a TLAS with submit-and-wait calls,
// compacting and updating them. For more control use
// AccelerationStructureBuilder and AccelerationStructureBuildData.
class AccelerationStructureHelper
{
public:
  ~AccelerationStructureHelper() { assert(!m_transientPool && "deinit missing"); }

  // `queueInfo.familyIndex` is the D3D12_COMMAND_LIST_TYPE of `queueInfo.queue`.
  void init(ResourceAllocator*    alloc,
            CmdUploaderInterface* uploader,
            QueueInfo             queueInfo,
            uint64_t              hintMaxAccelerationStructureSize = 512'000'000,
            uint64_t              hintMaxScratchStructureSize      = 128'000'000);

  // Releases the transient pool; call deinitAccelerationStructures first.
  void deinit(void);

  // Frees both the TLAS and the BLAS set created by the build methods; new
  // builds can be invoked afterwards.
  void deinitAccelerationStructures(void);

public:
  // BLAS related

  std::vector<AccelerationStructureBuildData> blasBuildData;
  std::vector<AccelerationStructure>          blasSet{};  // Bottom-level AS set
  AccelerationStructureBuilder::Stats         blasBuildStatistics;
  Buffer                                      blasScratchBuffer{};

  // Builds one BLAS per geometry of `asGeoInfoSet`, with the same `buildFlags`
  // each (compacted when they include ALLOW_COMPACTION).
  void blasSubmitBuildAndWait(const std::vector<D3D12_RAYTRACING_GEOMETRY_DESC>&  asGeoInfoSet,
                              D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAGS buildFlags);

public:
  // TLAS related

  AccelerationStructureBuildData tlasBuildData{};
  AccelerationStructure          tlas{};  // Top-level AS
  Buffer                         tlasInstancesBuffer{};
  Buffer                         tlasScratchBuffer{};
  size_t                         tlasSize{0};

  // Builds the TLAS from a list of instances. Add ALLOW_UPDATE to
  // `buildFlags` to use tlasSubmitUpdateAndWait.
  void tlasSubmitBuildAndWait(const std::vector<D3D12_RAYTRACING_INSTANCE_DESC>&  tlasInstances,
                              D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAGS buildFlags);

  // Updates the TLAS with a new list of instances; rebuilds it when the
  // instance count changed. The TLAS must have been built with ALLOW_UPDATE.
  // A rebuild that needs a larger TLAS recreates it: refresh any stored
  // `tlas.gpuAddress`, SRV slot or `tlas.shaderHandle()` afterwards.
  void tlasSubmitUpdateAndWait(const std::vector<D3D12_RAYTRACING_INSTANCE_DESC>& tlasInstances);

private:
  QueueInfo                      m_queueInfo;
  ResourceAllocator*             m_alloc{nullptr};
  CmdUploaderInterface*          m_uploader{};
  uint64_t                       m_blasAccelerationStructureBudget{};
  uint64_t                       m_blasScratchBudget{};
  ComPtr<ID3D12CommandAllocator> m_transientPool{};
};

}  // namespace nvdx12
