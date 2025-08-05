/*
 * Copyright (c) 2023-2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include <cstring>
#include <unordered_map>
#include <mutex>

#include <vulkan/vulkan_core.h>

#include <nvutils/hash_operations.hpp>

namespace nvvk {

//-----------------------------------------------------------------
// Samplers are limited in Vulkan.
// This class is used to create and store samplers, and to avoid creating the same sampler multiple times.
//
// Usage:
//      see usage_SamplerPool in sampler_pool.cpp
//-----------------------------------------------------------------
class SamplerPool
{
public:
  SamplerPool() = default;
  ~SamplerPool();

  // Delete copy constructor and copy assignment operator
  SamplerPool(const SamplerPool&)            = delete;
  SamplerPool& operator=(const SamplerPool&) = delete;

  // Allow move constructor and move assignment operator
  SamplerPool(SamplerPool&& other) noexcept;
  SamplerPool& operator=(SamplerPool&& other) noexcept;

  // Initialize the sampler pool with the device reference, then we can later acquire samplers
  void init(VkDevice device);
  // Destroy internal resources and reset its initial state
  void deinit();
  // Get or create VkSampler based on VkSamplerCreateInfo
  // The pNext chain may contain VkSamplerReductionModeCreateInfo as well as VkSamplerYcbcrConversionCreateInfo, but no other
  // structs are supported.
  VkResult acquireSampler(VkSampler&                 sampler,
                          const VkSamplerCreateInfo& createInfo = {.sType     = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
                                                                   .magFilter = VK_FILTER_LINEAR,
                                                                   .minFilter = VK_FILTER_LINEAR});

  void releaseSampler(VkSampler sampler);

private:
  VkDevice m_device{};

  struct SamplerState
  {
    VkSamplerCreateInfo                createInfo{};
    VkSamplerReductionModeCreateInfo   reduction{};
    VkSamplerYcbcrConversionCreateInfo ycbr{};

    bool operator==(const SamplerState& other) const
    {
      return other.createInfo.flags == createInfo.flags && other.createInfo.magFilter == createInfo.magFilter
             && other.createInfo.minFilter == createInfo.minFilter && other.createInfo.mipmapMode == createInfo.mipmapMode
             && other.createInfo.addressModeU == createInfo.addressModeU && other.createInfo.addressModeV == createInfo.addressModeV
             && other.createInfo.addressModeW == createInfo.addressModeW && other.createInfo.mipLodBias == createInfo.mipLodBias
             && other.createInfo.anisotropyEnable == createInfo.anisotropyEnable
             && other.createInfo.maxAnisotropy == createInfo.maxAnisotropy
             && other.createInfo.compareEnable == createInfo.compareEnable
             && other.createInfo.compareOp == createInfo.compareOp && other.createInfo.minLod == createInfo.minLod
             && other.createInfo.maxLod == createInfo.maxLod && other.createInfo.borderColor == createInfo.borderColor
             && other.createInfo.unnormalizedCoordinates == createInfo.unnormalizedCoordinates
             && other.reduction.reductionMode == reduction.reductionMode && other.ycbr.format == ycbr.format
             && other.ycbr.ycbcrModel == ycbr.ycbcrModel && other.ycbr.ycbcrRange == ycbr.ycbcrRange
             && other.ycbr.components.r == ycbr.components.r && other.ycbr.components.g == ycbr.components.g
             && other.ycbr.components.b == ycbr.components.b && other.ycbr.components.a == ycbr.components.a
             && other.ycbr.xChromaOffset == ycbr.xChromaOffset && other.ycbr.yChromaOffset == ycbr.yChromaOffset
             && other.ycbr.chromaFilter == ycbr.chromaFilter
             && other.ycbr.forceExplicitReconstruction == ycbr.forceExplicitReconstruction;
    }
  };

  struct SamplerStateHashFn
  {
    std::size_t operator()(const SamplerState& s) const
    {
      return nvutils::hashVal(s.createInfo.flags, s.createInfo.magFilter, s.createInfo.minFilter, s.createInfo.mipmapMode,
                              s.createInfo.addressModeU, s.createInfo.addressModeV, s.createInfo.addressModeW,
                              s.createInfo.mipLodBias, s.createInfo.anisotropyEnable, s.createInfo.maxAnisotropy,
                              s.createInfo.compareEnable, s.createInfo.compareOp, s.createInfo.minLod, s.createInfo.maxLod,
                              s.createInfo.borderColor, s.createInfo.unnormalizedCoordinates, s.reduction.reductionMode,
                              s.ycbr.format, s.ycbr.ycbcrModel, s.ycbr.ycbcrRange, s.ycbr.components.r,
                              s.ycbr.components.g, s.ycbr.components.b, s.ycbr.components.a, s.ycbr.xChromaOffset,
                              s.ycbr.yChromaOffset, s.ycbr.chromaFilter, s.ycbr.forceExplicitReconstruction);
    }
  };

  struct SamplerEntry
  {
    VkSampler sampler;
    uint32_t  refCount;
  };

  // Stores unique samplers with their corresponding VkSamplerCreateInfo and reference counts
  std::unordered_map<SamplerState, SamplerEntry, SamplerStateHashFn> m_samplerMap{};

  // Reverse lookup map for O(1) sampler release - must stay in sync with m_samplerMap
  std::unordered_map<VkSampler, SamplerState> m_samplerToState{};

  // Mutex for thread-safe access to both maps
  mutable std::mutex m_mutex;
};


}  // namespace nvvk
