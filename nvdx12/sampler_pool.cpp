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

#include "sampler_pool.hpp"

#include <algorithm>
#include <cassert>

#include "check_error.hpp"

namespace nvdx12 {

D3D12_SAMPLER_DESC makeSamplerDesc(D3D12_FILTER filter, D3D12_TEXTURE_ADDRESS_MODE address, float maxLod, float maxAnisotropy)
{
  D3D12_SAMPLER_DESC desc = DEFAULT_D3D12_SAMPLER_DESC;
  desc.Filter             = filter;
  desc.AddressU           = address;
  desc.AddressV           = address;
  desc.AddressW           = address;
  // D3D12_REQ_MAXANISOTROPY (16) is the largest value the runtime accepts.
  desc.MaxAnisotropy = static_cast<UINT>(std::clamp(maxAnisotropy, 1.0f, float(D3D12_REQ_MAXANISOTROPY)));
  desc.MaxLOD        = maxLod;
  return desc;
}

HRESULT SamplerPool::acquireSampler(uint32_t& slot, const D3D12_SAMPLER_DESC& desc)
{
  assert(m_heap && "Initialization was missing");
  slot = m_heap->acquireSamplerDescriptor(desc);
  return slot != kInvalidSlot ? S_OK : E_OUTOFMEMORY;
}

void SamplerPool::releaseSampler(uint32_t slot)
{
  assert(m_heap && "Initialization was missing");
  m_heap->releaseSamplerDescriptor(slot);
}

}  // namespace nvdx12

//--------------------------------------------------------------------------------------------------
// Usage example
//--------------------------------------------------------------------------------------------------
[[maybe_unused]] static void usage_SamplerPool()
{
  nvdx12::DescriptorHeap* heap = nullptr;  // EX: the application's descriptor heap
  nvdx12::SamplerPool     samplerPool;
  samplerPool.init(heap);

  // Trilinear / wrap without a description; any D3D12_SAMPLER_DESC otherwise.
  uint32_t linearSlot = nvdx12::kInvalidSlot;
  NVDX12_CHECK(samplerPool.acquireSampler(linearSlot));
  uint32_t clampSlot = nvdx12::kInvalidSlot;
  NVDX12_CHECK(samplerPool.acquireSampler(clampSlot, nvdx12::makeSamplerDesc(D3D12_FILTER_MIN_MAG_MIP_POINT,
                                                                             D3D12_TEXTURE_ADDRESS_MODE_CLAMP)));

  // Shaders sample with SamplerDescriptorHeap[linearSlot]
  // ...

  samplerPool.releaseSampler(clampSlot);
  samplerPool.releaseSampler(linearSlot);
  samplerPool.deinit();
}
