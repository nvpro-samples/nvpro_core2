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

#include "default_structs.hpp"
#include "descriptor_heap.hpp"

namespace nvdx12 {

/*-------------------------------------------------------------------------------------------------
# class nvdx12::SamplerPool

> Deduplicating sampler creation, the D3D12 counterpart of nvvk::SamplerPool.

In D3D12 a sampler *is* a slot of the shader-visible sampler heap, so the
pool is a facade over nvdx12::DescriptorHeap that keeps the nvvk call shape
(`acquireSampler(slot, desc)` / `releaseSampler(slot)`): equal descriptions
share one slot, which is freed when its last user releases it. The slot is
what shaders index SamplerDescriptorHeap[] with.

Differences with nvvk that follow from D3D12:
- `init` takes the DescriptorHeap whose sampler heap holds the samplers
  (nvvk takes the VkDevice), and samplers are uint32_t slots, not VkSampler.
- The default sampler is DEFAULT_D3D12_SAMPLER_DESC: trilinear, wrap, over
  the full mip chain. nvvk's default VkSamplerCreateInfo is bilinear with
  mipmapMode NEAREST and maxLod 0, so it samples mip 0 only.
- D3D12 samplers have no pNext chain (reduction modes are D3D12_FILTER
  values; there are no YCbCr conversions).

See usage_SamplerPool() in sampler_pool.cpp.
-------------------------------------------------------------------------------------------------*/

// Sampler description from the filter, the address mode of all three axes
// and the limits most applications vary; `maxAnisotropy` is clamped to [1, 16].
D3D12_SAMPLER_DESC makeSamplerDesc(D3D12_FILTER               filter,
                                   D3D12_TEXTURE_ADDRESS_MODE address,
                                   float                      maxLod        = D3D12_FLOAT32_MAX,
                                   float                      maxAnisotropy = 1.0f);

class SamplerPool
{
public:
  SamplerPool() = default;
  ~SamplerPool() { assert(m_heap == nullptr && "Missing deinit()"); }

  SamplerPool(const SamplerPool&)            = delete;
  SamplerPool& operator=(const SamplerPool&) = delete;

  SamplerPool(SamplerPool&& other) noexcept
      : m_heap(other.m_heap)
  {
    other.m_heap = nullptr;
  }
  SamplerPool& operator=(SamplerPool&& other) noexcept
  {
    if(this != &other)
    {
      assert(m_heap == nullptr && "Missing deinit()");
      m_heap       = other.m_heap;
      other.m_heap = nullptr;
    }
    return *this;
  }

  // The pool does not own the heap.
  void init(DescriptorHeap* heap) { m_heap = heap; }
  // The samplers stay in the heap until their last releaseSampler.
  void deinit() { m_heap = nullptr; }

  // Slot of the sampler described by `desc`, shared with every other
  // acquisition of the same description; E_OUTOFMEMORY when the sampler
  // heap is full.
  HRESULT acquireSampler(uint32_t& slot, const D3D12_SAMPLER_DESC& desc = DEFAULT_D3D12_SAMPLER_DESC);
  void    releaseSampler(uint32_t slot);

private:
  DescriptorHeap* m_heap{nullptr};  // not owned
};

}  // namespace nvdx12
