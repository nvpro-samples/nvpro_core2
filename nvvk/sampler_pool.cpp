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

#include <volk.h>

#include "sampler_pool.hpp"
#include "check_error.hpp"

#include <cassert>
#include <mutex>

nvvk::SamplerPool::SamplerPool(SamplerPool&& other) noexcept
    : m_device(other.m_device)
    , m_samplerMap(std::move(other.m_samplerMap))
    , m_samplerToState(std::move(other.m_samplerToState))
{
  // Reset the moved-from object to a valid state
  other.m_device = VK_NULL_HANDLE;
}

nvvk::SamplerPool& nvvk::SamplerPool::operator=(SamplerPool&& other) noexcept
{
  if(this != &other)
  {
    m_device         = std::move(other.m_device);
    m_samplerMap     = std::move(other.m_samplerMap);
    m_samplerToState = std::move(other.m_samplerToState);
  }
  return *this;
}

nvvk::SamplerPool::~SamplerPool()
{
  assert(m_device == VK_NULL_HANDLE && "Missing deinit()");
}

void nvvk::SamplerPool::init(VkDevice device)
{
  m_device = device;
}

void nvvk::SamplerPool::deinit()
{
  std::lock_guard<std::mutex> lock(m_mutex);
  for(const auto& entry : m_samplerMap)
  {
    vkDestroySampler(m_device, entry.second.sampler, nullptr);
  }
  m_samplerMap.clear();
  m_samplerToState.clear();
  m_device = VK_NULL_HANDLE;
}

VkResult nvvk::SamplerPool::acquireSampler(VkSampler& sampler, const VkSamplerCreateInfo& createInfo)
{
  SamplerState samplerState;
  samplerState.createInfo = createInfo;

  // add supported extensions
  const VkBaseInStructure* ext = (const VkBaseInStructure*)createInfo.pNext;
  while(ext)
  {
    switch(ext->sType)
    {
      case VK_STRUCTURE_TYPE_SAMPLER_REDUCTION_MODE_CREATE_INFO:
        samplerState.reduction = *(const VkSamplerReductionModeCreateInfo*)ext;
        break;
      case VK_STRUCTURE_TYPE_SAMPLER_YCBCR_CONVERSION_CREATE_INFO:
        samplerState.ycbr = *(const VkSamplerYcbcrConversionCreateInfo*)ext;
        break;
      default:
        assert(0 && "unsupported sampler extension");
    }
    ext = ext->pNext;
  }
  // always remove pointers for comparison lookup
  samplerState.createInfo.pNext = nullptr;
  samplerState.reduction.pNext  = nullptr;
  samplerState.ycbr.pNext       = nullptr;

  assert(m_device && "Initialization was missing");

  std::lock_guard<std::mutex> lock(m_mutex);
  if(auto it = m_samplerMap.find(samplerState); it != m_samplerMap.end())
  {
    // If found, increment reference count and return existing sampler
    it->second.refCount++;
    sampler = it->second.sampler;
    return VK_SUCCESS;
  }

  // Otherwise, create a new sampler
  NVVK_FAIL_RETURN(vkCreateSampler(m_device, &createInfo, nullptr, &sampler));
  m_samplerMap[samplerState] = {sampler, 1};
  m_samplerToState[sampler]  = samplerState;
  return VK_SUCCESS;
}

void nvvk::SamplerPool::releaseSampler(VkSampler sampler)
{
  std::lock_guard<std::mutex> lock(m_mutex);
  if(sampler == VK_NULL_HANDLE)
    return;

  auto stateIt = m_samplerToState.find(sampler);
  if(stateIt == m_samplerToState.end())
  {
    // Sampler not found - this shouldn't happen in correct usage
    assert(false && "Attempting to release unknown sampler");
    return;
  }

  auto samplerIt = m_samplerMap.find(stateIt->second);
  assert(samplerIt != m_samplerMap.end() && "Inconsistent sampler pool state");

  samplerIt->second.refCount--;
  if(samplerIt->second.refCount == 0)
  {
    vkDestroySampler(m_device, sampler, nullptr);
    m_samplerMap.erase(samplerIt);
    m_samplerToState.erase(stateIt);
  }
}


//--------------------------------------------------------------------------------------------------
// Usage example
//--------------------------------------------------------------------------------------------------
[[maybe_unused]] static void usage_SamplerPool()
{
  VkDevice          device = nullptr;  // EX: get the device from the app (m_app->getDevice())
  nvvk::SamplerPool samplerPool;
  samplerPool.init(device);

  VkSamplerCreateInfo createInfo = {};  // EX: create a sampler create info or use the default one (DEFAULT_VkSamplerCreateInfo)
  VkSampler sampler;
  samplerPool.acquireSampler(sampler, createInfo);

  // Use the sampler
  // ...

  samplerPool.releaseSampler(sampler);
}
