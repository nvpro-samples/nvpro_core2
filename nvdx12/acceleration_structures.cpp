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

#include "acceleration_structures.hpp"

#include <algorithm>
#include <cassert>
#include <cstring>

#include <fmt/format.h>
#include <nvutils/logger.hpp>

#include "barriers.hpp"
#include "check_error.hpp"
#include "commands.hpp"
#include "debug_util.hpp"
#include "frame_uploader.hpp"
#include "resource_allocator.hpp"

namespace nvdx12 {

namespace {

// The OMM geometries point into the owned vectors, which may have moved
// since the geometry was added; refresh the pointers before any API call.
void repointOmmGeometries(AccelerationStructureBuildData& data)
{
  size_t ommIndex = 0;
  for(size_t i = 0; i < data.asGeometry.size(); i++)
  {
    D3D12_RAYTRACING_GEOMETRY_DESC& geometry = data.asGeometry[i];
    if(geometry.Type != D3D12_RAYTRACING_GEOMETRY_TYPE_OMM_TRIANGLES)
      continue;
    geometry.OmmTriangles.pTriangles  = &data.ommTriangles[ommIndex];
    geometry.OmmTriangles.pOmmLinkage = &data.ommLinkages[ommIndex];
    ommIndex++;
  }
}

}  // namespace

D3D12_RAYTRACING_GEOMETRY_DESC makeTriangleGeometry(D3D12_GPU_VIRTUAL_ADDRESS       vertexAddress,
                                                    uint32_t                        vertexStride,
                                                    uint32_t                        vertexCount,
                                                    D3D12_GPU_VIRTUAL_ADDRESS       indexAddress,
                                                    DXGI_FORMAT                     indexFormat,
                                                    uint32_t                        indexCount,
                                                    D3D12_RAYTRACING_GEOMETRY_FLAGS flags,
                                                    DXGI_FORMAT                     vertexFormat)
{
  D3D12_RAYTRACING_GEOMETRY_DESC desc{};
  desc.Type                                 = D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES;
  desc.Flags                                = flags;
  desc.Triangles.Transform3x4               = 0;
  desc.Triangles.VertexFormat               = vertexFormat;
  desc.Triangles.VertexCount                = vertexCount;
  desc.Triangles.VertexBuffer.StartAddress  = vertexAddress;
  desc.Triangles.VertexBuffer.StrideInBytes = vertexStride;
  if(indexFormat != DXGI_FORMAT_UNKNOWN)
  {
    desc.Triangles.IndexFormat = indexFormat;
    desc.Triangles.IndexCount  = indexCount;
    desc.Triangles.IndexBuffer = indexAddress;
  }
  return desc;
}

D3D12_RAYTRACING_GEOMETRY_OMM_LINKAGE_DESC makeOmmLinkage(D3D12_GPU_VIRTUAL_ADDRESS ommArrayAddress,
                                                          D3D12_GPU_VIRTUAL_ADDRESS ommIndexAddress,
                                                          uint32_t                  ommIndexStride,
                                                          DXGI_FORMAT               indexFormat,
                                                          uint32_t                  baseLocation)
{
  D3D12_RAYTRACING_GEOMETRY_OMM_LINKAGE_DESC desc{};
  desc.OpacityMicromapIndexBuffer.StartAddress  = ommIndexAddress;
  desc.OpacityMicromapIndexBuffer.StrideInBytes = ommIndexStride;
  desc.OpacityMicromapIndexFormat               = indexFormat;
  desc.OpacityMicromapBaseLocation              = baseLocation;
  desc.OpacityMicromapArray                     = ommArrayAddress;
  return desc;
}

void toTransformMatrix(const glm::mat4& matrix, FLOAT out[3][4])
{
  // The instance transform is row-major 3x4, glm::mat4 column-major:
  // transpose and keep the first three rows.
  const glm::mat4 transposed = glm::transpose(matrix);
  std::memcpy(out, &transposed, 12 * sizeof(FLOAT));
}

//--------------------------------------------------------------------------------------------------

void AccelerationStructureBuildData::addGeometry(const D3D12_RAYTRACING_GEOMETRY_DESC& geometry)
{
  // OMM geometries are re-pointed into the owned storage, which only addOmmGeometry fills.
  assert(geometry.Type != D3D12_RAYTRACING_GEOMETRY_TYPE_OMM_TRIANGLES && "use addOmmGeometry");
  asGeometry.push_back(geometry);
}

void AccelerationStructureBuildData::addOmmGeometry(const D3D12_RAYTRACING_GEOMETRY_TRIANGLES_DESC&   triangles,
                                                    const D3D12_RAYTRACING_GEOMETRY_OMM_LINKAGE_DESC& linkage,
                                                    D3D12_RAYTRACING_GEOMETRY_FLAGS                   flags)
{
  ommTriangles.push_back(triangles);
  ommLinkages.push_back(linkage);
  D3D12_RAYTRACING_GEOMETRY_DESC desc{};
  desc.Type  = D3D12_RAYTRACING_GEOMETRY_TYPE_OMM_TRIANGLES;
  desc.Flags = flags;
  asGeometry.push_back(desc);
  repointOmmGeometries(*this);
}

void AccelerationStructureBuildData::setInstances(D3D12_GPU_VIRTUAL_ADDRESS instanceAddress, uint32_t instanceCount)
{
  asType                  = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL;
  buildInfo.NumDescs      = instanceCount;
  buildInfo.InstanceDescs = instanceAddress;
}

D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO AccelerationStructureBuildData::finalizeGeometry(ID3D12Device5* device,
                                                                                                       D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAGS flags)
{
  buildInfo.Type        = asType;
  buildInfo.Flags       = flags;
  buildInfo.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
  if(asType == D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL)
  {
    assert(!asGeometry.empty() && "No geometry added to Build Structure");
    repointOmmGeometries(*this);
    buildInfo.NumDescs       = uint32_t(asGeometry.size());
    buildInfo.pGeometryDescs = asGeometry.data();
  }
  sizeInfo = {};
  device->GetRaytracingAccelerationStructurePrebuildInfo(&buildInfo, &sizeInfo);
  return sizeInfo;
}

AccelerationStructureCreateInfo AccelerationStructureBuildData::makeCreateInfo() const
{
  assert(sizeInfo.ResultDataMaxSizeInBytes > 0 && "Acceleration Structure Size not set, call finalizeGeometry");
  return {.type = asType, .size = sizeInfo.ResultDataMaxSizeInBytes};
}

void AccelerationStructureBuildData::cmdBuildAccelerationStructure(ID3D12GraphicsCommandList4* cmd,
                                                                   D3D12_GPU_VIRTUAL_ADDRESS   accelerationStructure,
                                                                   D3D12_GPU_VIRTUAL_ADDRESS   scratchAddress)
{
  assert(accelerationStructure != 0 && "Acceleration Structure not created, first call createAcceleration");
  if(asType == D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL)
  {
    // In case the structure was copied, refresh the pointers.
    repointOmmGeometries(*this);
    buildInfo.pGeometryDescs = asGeometry.data();
  }
  D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC desc{};
  desc.DestAccelerationStructureData    = accelerationStructure;
  desc.Inputs                           = buildInfo;
  desc.ScratchAccelerationStructureData = scratchAddress;
  cmd->BuildRaytracingAccelerationStructure(&desc, 0, nullptr);
}

void AccelerationStructureBuildData::cmdUpdateAccelerationStructure(ID3D12GraphicsCommandList4* cmd,
                                                                    D3D12_GPU_VIRTUAL_ADDRESS   accelerationStructure,
                                                                    D3D12_GPU_VIRTUAL_ADDRESS   scratchAddress)
{
  assert(accelerationStructure != 0 && "Acceleration Structure not created, first call createAcceleration");
  assert((buildInfo.Flags & D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_UPDATE) && "refit needs a build with ALLOW_UPDATE");
  if(asType == D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL)
  {
    repointOmmGeometries(*this);
    buildInfo.pGeometryDescs = asGeometry.data();
  }
  // In-place refit: the source is the current contents of the destination.
  D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC desc{};
  desc.DestAccelerationStructureData = accelerationStructure;
  desc.Inputs                        = buildInfo;
  desc.Inputs.Flags = buildInfo.Flags | D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PERFORM_UPDATE;
  desc.SourceAccelerationStructureData  = accelerationStructure;
  desc.ScratchAccelerationStructureData = scratchAddress;
  cmd->BuildRaytracingAccelerationStructure(&desc, 0, nullptr);
}

uint64_t getMaxScratchSize(std::span<const AccelerationStructureBuildData> buildData)
{
  uint64_t maxScratch = 0;
  for(size_t i = 0; i < buildData.size(); i++)
    maxScratch = std::max(maxScratch, buildData[i].sizeInfo.ScratchDataSizeInBytes);
  return maxScratch;
}

//--------------------------------------------------------------------------------------------------

void OpacityMicromapBuildData::setInputs(D3D12_GPU_VIRTUAL_ADDRESS inputData, D3D12_GPU_VIRTUAL_ADDRESS perOmmDescs, uint32_t perOmmDescStride)
{
  arrayDesc.InputBuffer               = inputData;
  arrayDesc.PerOmmDescs.StartAddress  = perOmmDescs;
  arrayDesc.PerOmmDescs.StrideInBytes = perOmmDescStride;
}

const D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO& OpacityMicromapBuildData::finalize(ID3D12Device5* device,
                                                                                                D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAGS flags)
{
  arrayDesc.NumOmmHistogramEntries    = uint32_t(histogram.size());
  arrayDesc.pOmmHistogram             = histogram.data();
  buildInfo.Type                      = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_OPACITY_MICROMAP_ARRAY;
  buildInfo.Flags                     = flags;
  buildInfo.NumDescs                  = 1;  // the array desc, which counts the OMMs in its histogram
  buildInfo.DescsLayout               = D3D12_ELEMENTS_LAYOUT_ARRAY;
  buildInfo.pOpacityMicromapArrayDesc = &arrayDesc;
  sizeInfo                            = {};
  device->GetRaytracingAccelerationStructurePrebuildInfo(&buildInfo, &sizeInfo);
  return sizeInfo;
}

void OpacityMicromapBuildData::cmdBuild(ID3D12GraphicsCommandList4* cmd, D3D12_GPU_VIRTUAL_ADDRESS destination, D3D12_GPU_VIRTUAL_ADDRESS scratch)
{
  // The histogram may have been moved since finalize; the desc points at the owned storage.
  arrayDesc.NumOmmHistogramEntries    = uint32_t(histogram.size());
  arrayDesc.pOmmHistogram             = histogram.data();
  buildInfo.pOpacityMicromapArrayDesc = &arrayDesc;
  D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC desc{};
  desc.DestAccelerationStructureData    = destination;
  desc.Inputs                           = buildInfo;
  desc.ScratchAccelerationStructureData = scratch;
  cmd->BuildRaytracingAccelerationStructure(&desc, 0, nullptr);
}

AccelerationStructureCreateInfo OpacityMicromapBuildData::makeCreateInfo() const
{
  assert(sizeInfo.ResultDataMaxSizeInBytes > 0 && "Opacity micromap array size not set, call finalize");
  return {.type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_OPACITY_MICROMAP_ARRAY, .size = sizeInfo.ResultDataMaxSizeInBytes};
}

//--------------------------------------------------------------------------------------------------

void cmdEmitCompactedSizes(ID3D12GraphicsCommandList4* cmd, std::span<const D3D12_GPU_VIRTUAL_ADDRESS> sources, D3D12_GPU_VIRTUAL_ADDRESS destination)
{
  if(sources.empty())
    return;
  // One D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_COMPACTED_SIZE_DESC
  // (a single UINT64) per source, written consecutively.
  D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_DESC desc{};
  desc.DestBuffer = destination;
  desc.InfoType   = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_COMPACTED_SIZE;
  cmd->EmitRaytracingAccelerationStructurePostbuildInfo(&desc, UINT(sources.size()), sources.data());
}

void cmdCopyCompacted(ID3D12GraphicsCommandList4* cmd, D3D12_GPU_VIRTUAL_ADDRESS destination, D3D12_GPU_VIRTUAL_ADDRESS source)
{
  cmd->CopyRaytracingAccelerationStructure(destination, source, D3D12_RAYTRACING_ACCELERATION_STRUCTURE_COPY_MODE_COMPACT);
}

void cmdAccelerationStructureBarrier(ID3D12GraphicsCommandList7* cmd, D3D12_BARRIER_SYNC dstSync)
{
  constexpr D3D12_BARRIER_SYNC srcSync = D3D12_BARRIER_SYNC_BUILD_RAYTRACING_ACCELERATION_STRUCTURE
                                         | D3D12_BARRIER_SYNC_COPY_RAYTRACING_ACCELERATION_STRUCTURE;
  // Builds and copies after the barrier read and may write the structures (nvvk: READ | WRITE).
  const bool                 rewrites    = (dstSync & srcSync) != 0;
  const D3D12_BARRIER_ACCESS dstAccess   = rewrites ? D3D12_BARRIER_ACCESS_RAYTRACING_ACCELERATION_STRUCTURE_READ
                                                        | D3D12_BARRIER_ACCESS_RAYTRACING_ACCELERATION_STRUCTURE_WRITE :
                                                      D3D12_BARRIER_ACCESS_RAYTRACING_ACCELERATION_STRUCTURE_READ;
  const D3D12_GLOBAL_BARRIER barriers[2] = {
      makeMemoryBarrier(srcSync, dstSync, D3D12_BARRIER_ACCESS_RAYTRACING_ACCELERATION_STRUCTURE_WRITE, dstAccess),
      // The scratch accesses of the builds are UAV accesses: a later build may reuse the scratch memory.
      makeMemoryBarrier(D3D12_BARRIER_SYNC_BUILD_RAYTRACING_ACCELERATION_STRUCTURE, D3D12_BARRIER_SYNC_BUILD_RAYTRACING_ACCELERATION_STRUCTURE,
                        D3D12_BARRIER_ACCESS_UNORDERED_ACCESS, D3D12_BARRIER_ACCESS_UNORDERED_ACCESS),
  };
  const bool toBuilds = (dstSync & D3D12_BARRIER_SYNC_BUILD_RAYTRACING_ACCELERATION_STRUCTURE) != 0;
  const D3D12_BARRIER_GROUP group{.Type = D3D12_BARRIER_TYPE_GLOBAL, .NumBarriers = toBuilds ? 2u : 1u, .pGlobalBarriers = barriers};
  cmd->Barrier(1, &group);
}

//--------------------------------------------------------------------------------------------------
// AccelerationStructureBuilder

namespace {

uint64_t alignScratch(uint64_t size, uint64_t alignment)
{
  return (size + alignment - 1) / alignment * alignment;
}

}  // namespace

AccelerationStructureBuilder& AccelerationStructureBuilder::operator=(AccelerationStructureBuilder&& other) noexcept
{
  // Swap rather than default-move, so that `other` does not keep the resources.
  if(this != &other)
  {
    assert(!m_queryPool.sizes);  // Did you forget to deinitialize before moving into this object?
    std::swap(m_alloc, other.m_alloc);
    std::swap(m_queryPool, other.m_queryPool);
    std::swap(m_currentBlasIdx, other.m_currentBlasIdx);
    std::swap(m_scratchAlignment, other.m_scratchAlignment);
    std::swap(m_batches, other.m_batches);
    std::swap(m_cleanupBlasAccel, other.m_cleanupBlasAccel);
    std::swap(m_stats, other.m_stats);
  }
  return *this;
}

void AccelerationStructureBuilder::init(ResourceAllocator* allocator)
{
  m_alloc            = allocator;
  m_currentBlasIdx   = 0;
  m_scratchAlignment = kScratchAlignment;
}

void AccelerationStructureBuilder::deinit()
{
  if(m_alloc != nullptr)
  {
    destroy();
  }
  *this = {};
}

HRESULT AccelerationStructureBuilder::cmdCreateBlas(ID3D12GraphicsCommandList7*                cmd,
                                                    std::span<AccelerationStructureBuildData>& blasBuildData,
                                                    std::span<AccelerationStructure>&          blasAccel,
                                                    D3D12_GPU_VIRTUAL_ADDRESS                  scratchAddress,
                                                    uint64_t                                   scratchSize,
                                                    uint64_t                                   hintMaxBudget)
{
  assert(blasAccel.size() >= blasBuildData.size());
  assert((scratchAddress % kScratchAlignment) == 0 && "scratch buffer must be kScratchAlignment aligned");
  NVDX12_FAIL_RETURN(initializeQueryPoolIfNeeded(blasBuildData));

  // Track the starting BLAS index for this batch; the compacted sizes are indexed like the BLAS.
  const uint32_t batchStartIdx = m_currentBlasIdx;

  const D3D12_GPU_VIRTUAL_ADDRESS scratchAddressEnd = scratchAddress + scratchSize;
  uint64_t                        budgetUsed        = 0;  // Budget used in this call

  // Process the BLAS while staying under the memory budget, one scratch sub-batch at a time.
  HRESULT result = S_OK;
  while(m_currentBlasIdx < blasBuildData.size() && budgetUsed < hintMaxBudget && SUCCEEDED(result))
  {
    result = cmdBuildAccelerationStructures(cmd, budgetUsed, blasBuildData, blasAccel, scratchAddress, scratchAddressEnd,
                                            hintMaxBudget, m_queryPool.sizes ? &m_queryPool : nullptr);
  }

  // Store the batch for compaction; the compacted sizes reach the readback
  // buffer by the end of `cmd`. Also after a failing sub-batch: the BLAS
  // built before it are recorded in `cmd` and stay compactable.
  if(m_currentBlasIdx > batchStartIdx && m_queryPool.sizes)
  {
    const uint64_t offset   = uint64_t(batchStartIdx) * sizeof(uint64_t);
    const uint64_t byteSize = uint64_t(m_currentBlasIdx - batchStartIdx) * sizeof(uint64_t);
    cmdMemoryBarrier(cmd, D3D12_BARRIER_SYNC_EMIT_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO,
                     D3D12_BARRIER_SYNC_COPY, D3D12_BARRIER_ACCESS_UNORDERED_ACCESS, D3D12_BARRIER_ACCESS_COPY_SOURCE);
    cmd->CopyBufferRegion(m_queryPool.readback.buffer.Get(), offset, m_queryPool.sizes.buffer.Get(), offset, byteSize);

    m_batches.push({.startIdx = batchStartIdx, .endIdx = m_currentBlasIdx});
  }
  if(FAILED(result))
  {
    return result;
  }

  // Check if all BLAS have been built.
  return (m_currentBlasIdx < blasBuildData.size()) ? S_FALSE : S_OK;
}

// Creates the compacted-size buffers if any remaining BLAS is compactable:
// one UINT64 per BLAS of `blasBuildData`, indexed like the BLAS.
HRESULT AccelerationStructureBuilder::initializeQueryPoolIfNeeded(const std::span<AccelerationStructureBuildData>& blasBuildData)
{
  bool needsCompaction = false;
  for(size_t i = m_currentBlasIdx; i < blasBuildData.size(); i++)
  {
    if(blasBuildData[i].hasCompactFlag())
    {
      needsCompaction = true;
      break;
    }
  }
  // The entries are indexed like the BLAS: a pool from an earlier, smaller span would overflow.
  assert((!m_queryPool.sizes || m_queryPool.count >= blasBuildData.size()) && "same BLAS span on every call; deinit between sets");
  if(!needsCompaction || m_queryPool.sizes)
  {
    // D3D12 has no query reset: each emission overwrites its entries.
    return S_OK;
  }

  const uint32_t count    = uint32_t(blasBuildData.size());
  const uint64_t byteSize = uint64_t(count) * sizeof(uint64_t);
  // The post-build info is written through unordered access.
  NVDX12_FAIL_RETURN(m_alloc->createBuffer(m_queryPool.sizes, byteSize, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS));
  NVDX12_DBG_NAME(m_queryPool.sizes.buffer);
  const HRESULT hr = m_alloc->createBuffer(m_queryPool.readback, byteSize, D3D12_RESOURCE_FLAG_NONE, MemoryUsage::eHostReadback);
  if(FAILED(hr))
  {
    m_alloc->destroyBuffer(m_queryPool.sizes);
    return NVDX12_FAIL_REPORT(hr);
  }
  NVDX12_DBG_NAME(m_queryPool.readback.buffer);
  m_queryPool.count = count;
  return S_OK;
}

// Builds the BLAS from m_currentBlasIdx on that fit in the scratch range and
// the budget, side by side in the scratch buffer, then records the barriers
// after them and the emission of their compacted sizes.
HRESULT AccelerationStructureBuilder::cmdBuildAccelerationStructures(ID3D12GraphicsCommandList7* cmd,
                                                                     uint64_t&                   budgetUsed,
                                                                     std::span<AccelerationStructureBuildData>& blasBuildData,
                                                                     std::span<AccelerationStructure>& blasAccel,
                                                                     D3D12_GPU_VIRTUAL_ADDRESS         scratchAddress,
                                                                     D3D12_GPU_VIRTUAL_ADDRESS scratchAddressEnd,
                                                                     uint64_t                  hintMaxBudget,
                                                                     QueryPool*                queryPool)
{
  const uint32_t firstIdx = m_currentBlasIdx;
  HRESULT        result   = S_OK;

  while(scratchAddress < scratchAddressEnd && budgetUsed < hintMaxBudget && m_currentBlasIdx < blasBuildData.size())
  {
    AccelerationStructureBuildData& data = blasBuildData[m_currentBlasIdx];

    if(scratchAddress + data.sizeInfo.ScratchDataSizeInBytes > scratchAddressEnd)
      break;

    // Create the acceleration structure and record its build. On failure the
    // BLAS recorded before it still get their barrier and size emission below.
    result = NVDX12_FAIL_REPORT(m_alloc->createAcceleration(blasAccel[m_currentBlasIdx], data.makeCreateInfo()));
    if(FAILED(result))
      break;
    NVDX12_DBG_NAME(blasAccel[m_currentBlasIdx].buffer.buffer);
    data.cmdBuildAccelerationStructure(cmd, blasAccel[m_currentBlasIdx].gpuAddress, scratchAddress);

    // Update the used budget and the scratch address, kept aligned.
    budgetUsed += data.sizeInfo.ResultDataMaxSizeInBytes;
    scratchAddress += alignScratch(data.sizeInfo.ScratchDataSizeInBytes, m_scratchAlignment);
    m_currentBlasIdx++;
  }

  if(m_currentBlasIdx == firstIdx)
  {
    if(FAILED(result))
      return result;
    // Not even one BLAS fits the scratch range: calling again would not progress.
    assert(!"Scratch buffer smaller than the scratch size of a BLAS, see getScratchSize");
    return NVDX12_FAIL_REPORT(E_INVALIDARG);
  }

  // The results are complete for later builds, copies, the size emission and
  // traces; the next sub-batch may reuse the scratch memory.
  cmdAccelerationStructureBarrier(cmd);

  // Emit the compacted sizes of the compactable BLAS of this sub-batch, each
  // at the entry of its BLAS index (non-compactable ones have no size).
  if(queryPool != nullptr)
  {
    for(uint32_t idx = firstIdx; idx < m_currentBlasIdx; idx++)
    {
      if(!blasBuildData[idx].hasCompactFlag())
        continue;
      const D3D12_GPU_VIRTUAL_ADDRESS source = blasAccel[idx].gpuAddress;
      cmdEmitCompactedSizes(cmd, std::span(&source, 1), queryPool->sizes.gpuAddressAt(uint64_t(idx) * sizeof(uint64_t)));
    }
  }

  return result;
}

HRESULT AccelerationStructureBuilder::cmdCompactBlas(ID3D12GraphicsCommandList7*                cmd,
                                                     std::span<AccelerationStructureBuildData>& blasBuildData,
                                                     std::span<AccelerationStructure>&          blasAccel)
{
  if(m_batches.empty())
  {
    return S_OK;
  }

  // Process the oldest batch: its build command list has completed, so the
  // readback buffer holds the compacted sizes.
  const CompactBatchInfo batch = m_batches.front();
  m_batches.pop();
  if(!m_queryPool.readback.mapping)
  {
    return m_batches.empty() ? S_OK : S_FALSE;
  }
  const uint64_t* compactSizes = reinterpret_cast<const uint64_t*>(m_queryPool.readback.mapping);

  bool    copied = false;
  HRESULT result = S_OK;
  for(uint32_t blasIdx = batch.startIdx; blasIdx < batch.endIdx && blasIdx < blasBuildData.size(); blasIdx++)
  {
    if(!blasBuildData[blasIdx].hasCompactFlag())
      continue;  // no size emitted

    const uint64_t compactSize = compactSizes[blasIdx];
    if(compactSize == 0)
      continue;

    // Create the compacted BLAS and copy the original into it; on failure the
    // original stays in place.
    AccelerationStructure compacted;
    result = m_alloc->createAcceleration(compacted, {.type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL,
                                                     .size = compactSize});
    if(FAILED(result))
      break;
    NVDX12_DBG_NAME(compacted.buffer.buffer);
    cmdCopyCompacted(cmd, compacted.gpuAddress, blasAccel[blasIdx].gpuAddress);
    copied = true;

    // Update statistical tracking of sizes before and after compaction.
    m_stats.totalCompactSize += compactSize;
    m_stats.totalOriginalSize += blasBuildData[blasIdx].sizeInfo.ResultDataMaxSizeInBytes;
    blasBuildData[blasIdx].sizeInfo.ResultDataMaxSizeInBytes = compactSize;
    m_cleanupBlasAccel.push_back(blasAccel[blasIdx]);  // Schedule the original BLAS for cleanup.
    blasAccel[blasIdx] = compacted;
  }

  // The compacted BLAS may be consumed in the same command list.
  if(copied)
    cmdAccelerationStructureBarrier(cmd);
  if(FAILED(result))
    return NVDX12_FAIL_REPORT(result);

  return m_batches.empty() ? S_OK : S_FALSE;
}

void AccelerationStructureBuilder::destroyNonCompactedBlas()
{
  for(size_t i = 0; i < m_cleanupBlasAccel.size(); i++)
  {
    m_alloc->destroyAcceleration(m_cleanupBlasAccel[i]);
  }
  m_cleanupBlasAccel.clear();
}

void AccelerationStructureBuilder::destroyQueryPool()
{
  if(m_queryPool.sizes)
    m_alloc->destroyBuffer(m_queryPool.sizes);
  if(m_queryPool.readback)
    m_alloc->destroyBuffer(m_queryPool.readback);
  m_queryPool = {};
  m_batches   = {};
}

void AccelerationStructureBuilder::destroy()
{
  destroyQueryPool();
  destroyNonCompactedBlas();
}

// Find a scratch size that is within the budget, or at least as big as the largest required scratch.
uint64_t AccelerationStructureBuilder::getScratchSize(uint64_t                                         hintMaxBudget,
                                                      const std::span<AccelerationStructureBuildData>& buildData) const
{
  uint64_t maxScratch   = 0;
  uint64_t totalScratch = 0;
  for(size_t i = 0; i < buildData.size(); i++)
  {
    const uint64_t alignedSize = alignScratch(buildData[i].sizeInfo.ScratchDataSizeInBytes, m_scratchAlignment);
    maxScratch                 = std::max(maxScratch, alignedSize);
    totalScratch += alignedSize;
  }

  if(totalScratch <= hintMaxBudget)
  {
    // can be smaller than hintMaxBudget if all fit
    return totalScratch;
  }
  // must at least be maximum required, otherwise exhaust budget
  return std::max(maxScratch, hintMaxBudget);
}

std::string AccelerationStructureBuilder::Stats::toString() const
{
  const uint64_t savedSize       = totalOriginalSize - totalCompactSize;
  const float    fractionSmaller = (totalOriginalSize == 0) ? 0.0f : savedSize / static_cast<float>(totalOriginalSize);
  return fmt::format("BLAS Compaction: {} bytes -> {} bytes ({} bytes saved, {:.2f}% smaller)", totalOriginalSize,
                     totalCompactSize, savedSize, fractionSmaller * 100.0f);
}

//--------------------------------------------------------------------------------------------------
// AccelerationStructureHelper

namespace {

// The TLAS build reads the uploaded instances as a shader resource.
void cmdInstanceUploadBarrier(ID3D12GraphicsCommandList7* cmd)
{
  cmdMemoryBarrier(cmd, D3D12_BARRIER_SYNC_COPY, D3D12_BARRIER_SYNC_BUILD_RAYTRACING_ACCELERATION_STRUCTURE,
                   D3D12_BARRIER_ACCESS_COPY_DEST, D3D12_BARRIER_ACCESS_SHADER_RESOURCE);
}

HRESULT createScratchBuffer(const ResourceAllocator* alloc, Buffer& buffer, uint64_t size)
{
  return alloc->createBuffer(buffer, size, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, MemoryUsage::eDevice,
                             D3D12MA::ALLOCATION_FLAG_NONE, kScratchAlignment);
}

uint64_t tlasScratchSize(const D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO& sizeInfo)
{
  // Builds and updates share the scratch buffer.
  return std::max(sizeInfo.ScratchDataSizeInBytes, sizeInfo.UpdateScratchDataSizeInBytes);
}

}  // namespace

void AccelerationStructureHelper::init(ResourceAllocator*    alloc,
                                       CmdUploaderInterface* uploader,
                                       QueueInfo             queueInfo,
                                       uint64_t              hintMaxAccelerationStructureSize /*= 512'000'000*/,
                                       uint64_t              hintMaxScratchStructureSize /*= 128'000'000*/)
{
  assert(!m_transientPool && "init() called multiple times");

  m_queueInfo                       = queueInfo;
  m_alloc                           = alloc;
  m_uploader                        = uploader;
  m_blasAccelerationStructureBudget = hintMaxAccelerationStructureSize;
  m_blasScratchBudget               = hintMaxScratchStructureSize;
  m_transientPool = createTransientCommandPool(alloc->getDevice(), D3D12_COMMAND_LIST_TYPE(queueInfo.familyIndex));
}

void AccelerationStructureHelper::deinit(void)
{
  m_transientPool = nullptr;
  m_alloc         = nullptr;
  m_queueInfo     = {};
  m_uploader      = nullptr;
}

void AccelerationStructureHelper::deinitAccelerationStructures(void)
{
  // BLAS related
  for(size_t i = 0; i < blasSet.size(); i++)
  {
    AccelerationStructure& b = blasSet[i];
    if(b)
      m_alloc->destroyAcceleration(b);
  }
  blasSet.clear();
  blasBuildData.clear();
  blasBuildStatistics = {};
  if(blasScratchBuffer)
    m_alloc->destroyBuffer(blasScratchBuffer);
  blasScratchBuffer = {};

  // TLAS related
  if(tlas)
    m_alloc->destroyAcceleration(tlas);
  if(tlasInstancesBuffer)
    m_alloc->destroyBuffer(tlasInstancesBuffer);
  if(tlasScratchBuffer)
    m_alloc->destroyBuffer(tlasScratchBuffer);

  tlas                = {};
  tlasInstancesBuffer = {};
  tlasScratchBuffer   = {};
  tlasBuildData       = {};
  tlasSize            = {};
}

void AccelerationStructureHelper::blasSubmitBuildAndWait(const std::vector<D3D12_RAYTRACING_GEOMETRY_DESC>& asGeoInfoSet,
                                                         D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAGS buildFlags)
{
  ID3D12Device10*               device = m_alloc->getDevice();
  const D3D12_COMMAND_LIST_TYPE type   = D3D12_COMMAND_LIST_TYPE(m_queueInfo.familyIndex);

  assert(blasSet.empty() && "we must not invoke build if already built. use deinitAccelerationStructures before.");
  if(asGeoInfoSet.empty())
  {
    return;  // nothing to build, and no zero-sized scratch buffer to create
  }

  // Prepare the BLAS build data
  blasBuildData.reserve(asGeoInfoSet.size());
  for(size_t i = 0; i < asGeoInfoSet.size(); i++)
  {
    AccelerationStructureBuildData buildData{D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL};
    buildData.addGeometry(asGeoInfoSet[i]);
    buildData.finalizeGeometry(device, buildFlags);
    blasBuildData.emplace_back(std::move(buildData));
  }

  // build the set of BLAS
  blasSet.resize(blasBuildData.size());

  // Find the most optimal size for our scratch buffer, to allow a maximum
  // of BLAS to be built in parallel, within the budget
  AccelerationStructureBuilder blasBuilder;
  blasBuilder.init(m_alloc);
  const uint64_t scratchSize = blasBuilder.getScratchSize(m_blasScratchBudget, blasBuildData);
  NVDX12_CHECK(createScratchBuffer(m_alloc, blasScratchBuffer, scratchSize));
  NVDX12_DBG_NAME(blasScratchBuffer.buffer);

  // Start the build and compaction of the BLAS
  bool                                      finished      = false;
  std::span<AccelerationStructureBuildData> buildDataSpan = blasBuildData;
  std::span<AccelerationStructure>          blasSpan      = blasSet;
  do
  {
    {
      ComPtr<ID3D12GraphicsCommandList7> cmd = createSingleTimeCommands(device, m_transientPool.Get(), type);
      const HRESULT result = blasBuilder.cmdCreateBlas(cmd.Get(), buildDataSpan, blasSpan, blasScratchBuffer.gpuAddress,
                                                       blasScratchBuffer.bufferSize, m_blasAccelerationStructureBudget);
      if(result == S_OK)
      {
        finished = true;
      }
      else if(result != S_FALSE)
      {
        // Any result other than S_OK or S_FALSE is an error
        NVDX12_CHECK(result);
      }
      NVDX12_CHECK(endSingleTimeCommands(cmd.Get(), device, m_transientPool.Get(), m_queueInfo.queue.Get()));
    }
    // compact BLAS if needed
    if(buildFlags & D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_COMPACTION)
    {
      // Compacting the BLAS, and destroy the previous ones
      ComPtr<ID3D12GraphicsCommandList7> cmd = createSingleTimeCommands(device, m_transientPool.Get(), type);
      NVDX12_CHECK(blasBuilder.cmdCompactBlas(cmd.Get(), buildDataSpan, blasSpan));
      NVDX12_CHECK(endSingleTimeCommands(cmd.Get(), device, m_transientPool.Get(), m_queueInfo.queue.Get()));
      blasBuilder.destroyNonCompactedBlas();
    }
  } while(!finished);

  blasBuildStatistics = blasBuilder.getStatistics();

  // Giving a name to the BLAS
  for(size_t i = 0; i < blasSet.size(); i++)
  {
    NVDX12_DBG_NAME(blasSet[i].buffer.buffer);
  }

  // Cleanup
  blasBuilder.deinit();
}

void AccelerationStructureHelper::tlasSubmitBuildAndWait(const std::vector<D3D12_RAYTRACING_INSTANCE_DESC>& tlasInstances,
                                                         D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAGS buildFlags)
{
  ID3D12Device10*               device = m_alloc->getDevice();
  const D3D12_COMMAND_LIST_TYPE type   = D3D12_COMMAND_LIST_TYPE(m_queueInfo.familyIndex);

  // we must not invoke build if already built. use update.
  assert(!tlasInstancesBuffer && "Do not invoke build if already built. Build with ALLOW_UPDATE, then use tlasSubmitUpdateAndWait");

  ComPtr<ID3D12GraphicsCommandList7> cmd = createSingleTimeCommands(device, m_transientPool.Get(), type);

  // Create the buffer of instances. D3D12 requires 16-byte aligned instance
  // descs, which every buffer placement satisfies.
  const std::span<const D3D12_RAYTRACING_INSTANCE_DESC> instances(tlasInstances);
  NVDX12_CHECK(m_alloc->createBuffer(tlasInstancesBuffer,
                                     std::max<uint64_t>(instances.size_bytes(), sizeof(D3D12_RAYTRACING_INSTANCE_DESC)),
                                     D3D12_RESOURCE_FLAG_NONE));
  NVDX12_DBG_NAME(tlasInstancesBuffer.buffer);
  NVDX12_CHECK(m_uploader->appendBuffer(tlasInstancesBuffer, 0, instances));
  m_uploader->cmdUploadAppended(cmd.Get());
  cmdInstanceUploadBarrier(cmd.Get());

  tlasBuildData = {};
  tlasBuildData.setInstances(tlasInstancesBuffer.gpuAddress, uint32_t(tlasInstances.size()));

  // Get the size of the TLAS
  const D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO sizeInfo = tlasBuildData.finalizeGeometry(device, buildFlags);

  // Create the scratch buffer
  NVDX12_CHECK(createScratchBuffer(m_alloc, tlasScratchBuffer, tlasScratchSize(sizeInfo)));
  NVDX12_DBG_NAME(tlasScratchBuffer.buffer);

  // Create the TLAS
  NVDX12_CHECK(m_alloc->createAcceleration(tlas, tlasBuildData.makeCreateInfo()));
  NVDX12_DBG_NAME(tlas.buffer.buffer);
  tlasBuildData.cmdBuildAccelerationStructure(cmd.Get(), tlas.gpuAddress, tlasScratchBuffer.gpuAddress);
  cmdAccelerationStructureBarrier(cmd.Get());

  tlasSize = tlasInstances.size();

  NVDX12_CHECK(endSingleTimeCommands(cmd.Get(), device, m_transientPool.Get(), m_queueInfo.queue.Get()));
}

void AccelerationStructureHelper::tlasSubmitUpdateAndWait(const std::vector<D3D12_RAYTRACING_INSTANCE_DESC>& tlasInstances)
{
  ID3D12Device10*               device = m_alloc->getDevice();
  const D3D12_COMMAND_LIST_TYPE type   = D3D12_COMMAND_LIST_TYPE(m_queueInfo.familyIndex);
  assert(tlas && "tlasSubmitBuildAndWait first");

  const bool                                            sizeChanged = (tlasInstances.size() != tlasSize);
  const std::span<const D3D12_RAYTRACING_INSTANCE_DESC> instances(tlasInstances);

  if(sizeChanged)
  {
    // D3D12 inputs carry the instance count: rebuild with the new count,
    // growing the resources when needed (previous submissions have completed).
    if(instances.size_bytes() > tlasInstancesBuffer.bufferSize)
    {
      m_alloc->destroyBuffer(tlasInstancesBuffer);
      NVDX12_CHECK(m_alloc->createBuffer(tlasInstancesBuffer, instances.size_bytes(), D3D12_RESOURCE_FLAG_NONE));
      NVDX12_DBG_NAME(tlasInstancesBuffer.buffer);
    }
    tlasBuildData.setInstances(tlasInstancesBuffer.gpuAddress, uint32_t(tlasInstances.size()));
    const D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO sizeInfo =
        tlasBuildData.finalizeGeometry(device, tlasBuildData.buildInfo.Flags);
    if(sizeInfo.ResultDataMaxSizeInBytes > tlas.size)
    {
      m_alloc->destroyAcceleration(tlas);
      NVDX12_CHECK(m_alloc->createAcceleration(tlas, tlasBuildData.makeCreateInfo()));
      NVDX12_DBG_NAME(tlas.buffer.buffer);
    }
    if(tlasScratchBuffer && tlasScratchSize(sizeInfo) > tlasScratchBuffer.bufferSize)
      m_alloc->destroyBuffer(tlasScratchBuffer);
  }

  if(!tlasScratchBuffer)
  {
    NVDX12_CHECK(createScratchBuffer(m_alloc, tlasScratchBuffer, tlasScratchSize(tlasBuildData.sizeInfo)));
    NVDX12_DBG_NAME(tlasScratchBuffer.buffer);
  }

  ComPtr<ID3D12GraphicsCommandList7> cmd = createSingleTimeCommands(device, m_transientPool.Get(), type);

  // Update the instance buffer
  NVDX12_CHECK(m_uploader->appendBuffer(tlasInstancesBuffer, 0, instances));
  m_uploader->cmdUploadAppended(cmd.Get());

  // Make sure the instances are copied before triggering the acceleration structure build
  cmdInstanceUploadBarrier(cmd.Get());

  // Building or updating the top-level acceleration structure
  if(sizeChanged)
  {
    tlasBuildData.cmdBuildAccelerationStructure(cmd.Get(), tlas.gpuAddress, tlasScratchBuffer.gpuAddress);
    tlasSize = tlasInstances.size();
  }
  else
  {
    tlasBuildData.cmdUpdateAccelerationStructure(cmd.Get(), tlas.gpuAddress, tlasScratchBuffer.gpuAddress);
  }
  cmdAccelerationStructureBarrier(cmd.Get());

  // Make sure to have the TLAS ready before using it
  NVDX12_CHECK(endSingleTimeCommands(cmd.Get(), device, m_transientPool.Get(), m_queueInfo.queue.Get()));
}

}  // namespace nvdx12

//--------------------------------------------------------------------------------------------------
// Usage example
//--------------------------------------------------------------------------------------------------
[[maybe_unused]] static void usage_AccelerationStructures()
{
  ID3D12Device5*              device = nullptr;
  ID3D12GraphicsCommandList7* cmd    = nullptr;
  nvdx12::ResourceAllocator   allocator;
  nvdx12::Buffer vertices, indices, instances;  // vertices: float3, indices: uint32, instances: D3D12_RAYTRACING_INSTANCE_DESC

  // BLAS: one triangle geometry, compactable.
  nvdx12::AccelerationStructureBuildData blasData;
  blasData.addGeometry(nvdx12::makeTriangleGeometry(vertices.gpuAddress, 3 * sizeof(float), 3, indices.gpuAddress,
                                                    DXGI_FORMAT_R32_UINT, 3, D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE));
  blasData.finalizeGeometry(device, D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE
                                        | D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_COMPACTION);

  // TLAS over `instances`, refittable.
  nvdx12::AccelerationStructureBuildData tlasData;
  tlasData.setInstances(instances.gpuAddress, 1);
  tlasData.finalizeGeometry(device, D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_BUILD
                                        | D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_UPDATE);

  // Storage: acceleration-structure memory for the results, one shared device buffer for scratch.
  nvdx12::AccelerationStructure blas, tlas;
  nvdx12::Buffer                scratch;
  NVDX12_CHECK(allocator.createAcceleration(blas, blasData.makeCreateInfo()));
  // Top level: an SRV slot for shaders
  NVDX12_CHECK(allocator.createAcceleration(tlas, tlasData.makeCreateInfo()));
  // The scratch serves the builds and the TLAS refit.
  const nvdx12::AccelerationStructureBuildData builds[] = {blasData, tlasData};
  NVDX12_CHECK(allocator.createBuffer(scratch, std::max(nvdx12::getMaxScratchSize(builds), tlasData.sizeInfo.UpdateScratchDataSizeInBytes)));

  // Build the BLAS, then the TLAS that references it. The builds record no
  // barrier themselves (unlike nvvk's): order them explicitly. The barrier
  // also orders the scratch accesses, so the next build may reuse the scratch.
  blasData.cmdBuildAccelerationStructure(cmd, blas.gpuAddress, scratch.gpuAddress);
  nvdx12::cmdAccelerationStructureBarrier(cmd);
  tlasData.cmdBuildAccelerationStructure(cmd, tlas.gpuAddress, scratch.gpuAddress);
  nvdx12::cmdAccelerationStructureBarrier(cmd);

  // Compaction: emit the size, read it back (not shown), copy into right-sized storage.
  nvdx12::Buffer                  compactedSizes;  // UINT64 per structure, device memory
  const D3D12_GPU_VIRTUAL_ADDRESS sources[] = {blas.gpuAddress};
  nvdx12::cmdEmitCompactedSizes(cmd, sources, compactedSizes.gpuAddress);
  nvdx12::AccelerationStructure compactedBlas;  // createAcceleration with the read-back size
  nvdx12::cmdCopyCompacted(cmd, compactedBlas.gpuAddress, blas.gpuAddress);

  // Later frames: refit the TLAS in place after the instance buffer changed.
  tlasData.cmdUpdateAccelerationStructure(cmd, tlas.gpuAddress, scratch.gpuAddress);
  nvdx12::cmdAccelerationStructureBarrier(cmd);

  allocator.destroyBuffer(scratch);
  allocator.destroyAcceleration(blas);
  allocator.destroyAcceleration(tlas);
}

[[maybe_unused]] static void usage_AccelerationStructureBuilder()
{
  nvdx12::ResourceAllocator              allocator;
  nvdx12::QueueInfo                      queueInfo;  // familyIndex = D3D12_COMMAND_LIST_TYPE_DIRECT
  nvdx12::ComPtr<ID3D12CommandAllocator> pool =
      nvdx12::createTransientCommandPool(allocator.getDevice(), D3D12_COMMAND_LIST_TYPE_DIRECT);
  std::vector<nvdx12::AccelerationStructureBuildData> blasBuildData;  // finalized with ALLOW_COMPACTION
  std::vector<nvdx12::AccelerationStructure>          blasAccel(blasBuildData.size());

  nvdx12::AccelerationStructureBuilder builder;
  builder.init(&allocator);

  // Scratch buffer within a 64 MB budget (at least the largest BLAS scratch).
  nvdx12::Buffer scratch;
  NVDX12_CHECK(allocator.createBuffer(scratch, builder.getScratchSize(64'000'000, blasBuildData),
                                      D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, nvdx12::MemoryUsage::eDevice,
                                      D3D12MA::ALLOCATION_FLAG_NONE, builder.getScratchAlignment()));

  std::span<nvdx12::AccelerationStructureBuildData> buildData = blasBuildData;
  std::span<nvdx12::AccelerationStructure>          accels    = blasAccel;
  HRESULT                                           result    = S_FALSE;
  while(result == S_FALSE)
  {
    nvdx12::ComPtr<ID3D12GraphicsCommandList7> cmd = nvdx12::createSingleTimeCommands(allocator.getDevice(), pool.Get());
    result = builder.cmdCreateBlas(cmd.Get(), buildData, accels, scratch.gpuAddress, scratch.bufferSize);
    NVDX12_CHECK(nvdx12::endSingleTimeCommands(cmd.Get(), allocator.getDevice(), pool.Get(), queueInfo.queue.Get()));

    // The batch completed: compact it, then free the originals once the copies completed.
    cmd = nvdx12::createSingleTimeCommands(allocator.getDevice(), pool.Get());
    builder.cmdCompactBlas(cmd.Get(), buildData, accels);
    NVDX12_CHECK(nvdx12::endSingleTimeCommands(cmd.Get(), allocator.getDevice(), pool.Get(), queueInfo.queue.Get()));
    builder.destroyNonCompactedBlas();
  }
  LOGI("%s\n", builder.getStatistics().toString().c_str());
  builder.deinit();
  allocator.destroyBuffer(scratch);

  // Or, with submit-and-wait calls: one BLAS per geometry, then a TLAS over
  // them, the instance array uploaded through a FrameUploader whose blocks hold it.
  nvdx12::FrameUploader                       uploader;
  nvdx12::AccelerationStructureHelper         helper;
  std::vector<D3D12_RAYTRACING_GEOMETRY_DESC> geometries;  // nvdx12::makeTriangleGeometry
  NVDX12_CHECK(uploader.init({.allocator = &allocator}));
  helper.init(&allocator, &uploader, queueInfo);
  helper.blasSubmitBuildAndWait(geometries, D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE
                                                | D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_COMPACTION);
  std::vector<D3D12_RAYTRACING_INSTANCE_DESC> instances(helper.blasSet.size());
  for(size_t i = 0; i < instances.size(); i++)
  {
    nvdx12::toTransformMatrix(glm::mat4(1.0f), instances[i].Transform);
    instances[i].InstanceMask          = 0xFF;
    instances[i].AccelerationStructure = helper.blasSet[i].gpuAddress;
  }
  helper.tlasSubmitBuildAndWait(instances, D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_UPDATE);
  // Shaders trace against helper.tlas.shaderHandle(). Later: move the instances and refit.
  helper.tlasSubmitUpdateAndWait(instances);
  // The helper waited for its submissions: the staging can go.
  uploader.releaseCompletedAllocations();
  helper.deinitAccelerationStructures();
  helper.deinit();
  uploader.deinit();
}
