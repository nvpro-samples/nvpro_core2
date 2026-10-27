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

#include "sbt_generator.hpp"

#include <cassert>
#include <cstring>

#include <nvutils/alignment.hpp>
#include <nvutils/logger.hpp>

#include "check_error.hpp"
#include "debug_util.hpp"
#include "resource_allocator.hpp"
#include "text_encoding.hpp"

namespace nvdx12 {

size_t SBTGenerator::calculateSBTBufferSize(ID3D12StateObject* rayPipeline)
{
  resetBuffer();
  m_pipeline = rayPipeline;
  // Each group's table starts on a D3D12_RAYTRACING_SHADER_TABLE_BYTE_ALIGNMENT boundary.
  uint64_t offset = 0;
  for(uint32_t type = eRaygen; type <= eCallable; type++)
  {
    m_groupOffsets[type] = offset;
    offset = nvutils::align_up(offset + getGroupSize(GroupType(type)), uint64_t(D3D12_RAYTRACING_SHADER_TABLE_BYTE_ALIGNMENT));
  }
  m_bufferSize = size_t(offset);
  return m_bufferSize;
}

HRESULT SBTGenerator::populateSBTBuffer(D3D12_GPU_VIRTUAL_ADDRESS bufferAddress, size_t bufferSize, void* bufferData)
{
  if(!m_pipeline || !bufferData || bufferSize < m_bufferSize || (bufferAddress % getBufferAlignment()) != 0)
  {
    LOGE("SBTGenerator: no pipeline (see calculateSBTBufferSize), or buffer smaller than %zu bytes or misaligned\n", m_bufferSize);
    return E_INVALIDARG;
  }
  if(m_exports[eRaygen].empty())
  {
    LOGE("SBTGenerator: no raygen record\n");
    return E_INVALIDARG;
  }

  ComPtr<ID3D12StateObjectProperties> properties;
  HRESULT                             hr = NVDX12_FAIL_REPORT(m_pipeline->QueryInterface(IID_PPV_ARGS(&properties)));
  if(FAILED(hr))
    return hr;

  uint8_t* data = static_cast<uint8_t*>(bufferData);
  std::memset(data, 0, m_bufferSize);
  for(uint32_t type = eRaygen; type <= eCallable; type++)
  {
    const std::vector<std::wstring>& exports = m_exports[type];
    const uint32_t                   stride  = getGroupStride(GroupType(type));
    for(size_t i = 0; i < exports.size(); i++)
    {
      const void* identifier = properties->GetShaderIdentifier(exports[i].c_str());
      if(!identifier)
      {
        LOGE("SBTGenerator: no shader identifier for export %s\n", utf8FromWtf16(exports[i].c_str()).c_str());
        hr = E_INVALIDARG;
        continue;
      }
      std::memcpy(data + m_groupOffsets[type] + i * stride, identifier, D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES);
    }
    m_bufferAddresses[type] = exports.empty() ? 0 : bufferAddress + m_groupOffsets[type];
  }
  if(FAILED(hr))
    resetBuffer();
  return hr;
}

D3D12_GPU_VIRTUAL_ADDRESS_RANGE_AND_STRIDE SBTGenerator::getSBTRegion(GroupType type, uint32_t indexOffset) const
{
  const uint32_t count = getGroupIndexCount(type);
  if(count == 0 || indexOffset >= count || m_bufferAddresses[type] == 0)
    return {};
  const uint32_t stride = getGroupStride(type);
  const uint64_t start  = m_bufferAddresses[type] + uint64_t(indexOffset) * stride;
  if(start % D3D12_RAYTRACING_SHADER_TABLE_BYTE_ALIGNMENT != 0)
  {
    assert(!"SBTGenerator::getSBTRegion: the region start is not 64-byte aligned");
    LOGE("SBTGenerator::getSBTRegion: record %u of group %d does not start on a %u-byte boundary\n", indexOffset,
         int(type), D3D12_RAYTRACING_SHADER_TABLE_BYTE_ALIGNMENT);
    return {};
  }
  // A raygen region holds exactly one record; the others run to the end of the group.
  const uint32_t records = type == eRaygen ? 1 : count - indexOffset;
  return D3D12_GPU_VIRTUAL_ADDRESS_RANGE_AND_STRIDE{start, uint64_t(records) * stride, stride};
}

SBTGenerator::Regions SBTGenerator::getSBTRegions(uint32_t rayGenIndexOffset) const
{
  const D3D12_GPU_VIRTUAL_ADDRESS_RANGE_AND_STRIDE raygen = getSBTRegion(eRaygen, rayGenIndexOffset);
  Regions                                          regions;
  regions.raygen   = {raygen.StartAddress, raygen.SizeInBytes};
  regions.miss     = getSBTRegion(eMiss);
  regions.hit      = getSBTRegion(eHit);
  regions.callable = getSBTRegion(eCallable);
  return regions;
}

D3D12_DISPATCH_RAYS_DESC SBTGenerator::getDispatchRaysDesc(uint32_t width, uint32_t height, uint32_t depth, uint32_t rayGenIndexOffset) const
{
  const Regions            regions = getSBTRegions(rayGenIndexOffset);
  D3D12_DISPATCH_RAYS_DESC desc{};
  desc.RayGenerationShaderRecord = regions.raygen;
  desc.MissShaderTable           = regions.miss;
  desc.HitGroupTable             = regions.hit;
  desc.CallableShaderTable       = regions.callable;
  desc.Width                     = width;
  desc.Height                    = height;
  desc.Depth                     = depth;
  return desc;
}

void SBTGenerator::reset()
{
  for(size_t i = 0; i < m_exports.size(); i++)
    m_exports[i].clear();
  resetBuffer();
}

void SBTGenerator::resetBuffer()
{
  m_groupOffsets    = {};
  m_bufferAddresses = {};
  m_bufferSize      = 0;
  m_pipeline        = nullptr;
}

}  // namespace nvdx12

//--------------------------------------------------------------------------------------------------
// Usage example
//--------------------------------------------------------------------------------------------------
[[maybe_unused]] static void usage_SBTGenerator()
{
  ID3D12GraphicsCommandList4* cmd = nullptr;
  nvdx12::ResourceAllocator   allocator;
  ID3D12StateObject*          pipeline = nullptr;  // see usage_RayTracingPipeline()

  // Records in table order: hit group i is InstanceContributionToHitGroupIndex i.
  nvdx12::SBTGenerator sbtGenerator;
  sbtGenerator.addRaygen(L"rayGenMain");
  sbtGenerator.addRaygen(L"rayGenDebug");
  sbtGenerator.addMiss(L"missMain");
  sbtGenerator.addMiss(L"shadowMissMain");
  sbtGenerator.addHitGroup(L"HitGroup");

  // Host-visible buffer of the computed size; the allocator places buffers on
  // BUFFER_PLACEMENT_ALIGNMENT (256) bytes, a multiple of getBufferAlignment().
  nvdx12::Buffer sbtBuffer;
  NVDX12_CHECK(allocator.createBuffer(sbtBuffer, sbtGenerator.calculateSBTBufferSize(pipeline),
                                      D3D12_RESOURCE_FLAG_NONE, nvdx12::MemoryUsage::eHostUpload));
  NVDX12_DBG_NAME(sbtBuffer.buffer);

  // Writes the identifiers of the state object; calculate and populate again after a recompile.
  NVDX12_CHECK(sbtGenerator.populateSBTBuffer(sbtBuffer.gpuAddress, size_t(sbtBuffer.bufferSize), sbtBuffer.mapping));

  // Trace
  cmd->SetPipelineState1(pipeline);
  const D3D12_DISPATCH_RAYS_DESC desc = sbtGenerator.getDispatchRaysDesc(1920, 1080);
  cmd->DispatchRays(&desc);

  // The regions are also available individually, e.g. to launch the second raygen record.
  const nvdx12::SBTGenerator::Regions regions = sbtGenerator.getSBTRegions(1);
  (void)regions;

  allocator.destroyBuffer(sbtBuffer);
}
