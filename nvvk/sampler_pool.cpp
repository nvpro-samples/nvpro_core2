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

nvvk::SamplerPool::SamplerPool(SamplerPool&& other) noexcept
    : m_device(other.m_device)
    , m_samplerMap(std::move(other.m_samplerMap))
{
  // Reset the moved-from object to a valid state
  other.m_device = VK_NULL_HANDLE;
}

nvvk::SamplerPool& nvvk::SamplerPool::operator=(SamplerPool&& other) noexcept
{
  if(this != &other)
  {
    m_device     = std::move(other.m_device);
    m_samplerMap = std::move(other.m_samplerMap);
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
  for(const auto& entry : m_samplerMap)
  {
    vkDestroySampler(m_device, entry.second, nullptr);
  }
  m_samplerMap.clear();
  *this = {};
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
  if(auto it = m_samplerMap.find(samplerState); it != m_samplerMap.end())
  {
    // If found, return existing sampler
    sampler = it->second;
    return VK_SUCCESS;
  }

  // Otherwise, create a new sampler
  NVVK_FAIL_RETURN(vkCreateSampler(m_device, &createInfo, nullptr, &sampler));
  m_samplerMap[samplerState] = sampler;
  return VK_SUCCESS;
}

void nvvk::SamplerPool::releaseSampler(VkSampler sampler)
{
  for(auto it = m_samplerMap.begin(); it != m_samplerMap.end();)
  {
    if(it->second == sampler)
    {
      vkDestroySampler(m_device, it->second, nullptr);
      it = m_samplerMap.erase(it);
    }
    else
    {
      ++it;
    }
  }
}


//--------------------------------------------------------------------------------------------------
// Usage example
//--------------------------------------------------------------------------------------------------
static void usage_SamplerPool()
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
