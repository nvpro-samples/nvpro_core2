/*
 * Copyright (c) 2022-2025, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */


#pragma once
//////////////////////////////////////////////////////////////////////////


#include <array>
#include <vector>
#include <assert.h>
#include <filesystem>

#include <vulkan/vulkan_core.h>

#include "descriptors.hpp"
#include "resource_allocator.hpp"
#include "sampler_pool.hpp"
#include "staging.hpp"


namespace nvvk {
class Context;


/*-------------------------------------------------------------------------------------------------
# class nvvkhl::HdrEnv

High-Dynamic-Range (HDR) environment map used for Image-Based Lighting (IBL).

>  Load an environment image (HDR) and create an acceleration structure for important light sampling.
  
-------------------------------------------------------------------------------------------------*/
class HdrIbl
{
public:
  HdrIbl() = default;
  ~HdrIbl() { assert(m_device == VK_NULL_HANDLE); }

  void init(nvvk::ResourceAllocator* allocator, nvvk::SamplerPool* samplerPool);
  void deinit();

  void loadEnvironment(VkCommandBuffer cmd, nvvk::StagingUploader& staging, const std::filesystem::path& hdrImage, bool enableMipmaps = false);
  void destroyEnvironment();

  float              getIntegral() const { return m_integral; }
  float              getAverage() const { return m_average; }
  bool               isValid() const { return m_valid; }
  const nvvk::Buffer getEnvAccel() const { return m_accelImpSmpl; }

  // HDR + importance sampling
  inline VkDescriptorSetLayout getDescriptorSetLayout() const { return m_descPack.getLayout(); }
  inline VkDescriptorSet       getDescriptorSet() const { return m_descPack.getSet(0); }
  const nvvk::Image&           getHdrImage() { return m_texHdr; }  // The loaded HDR texture
  VkExtent2D                   getHdrImageSize() const { return m_hdrImageSize; }

private:
  VkDevice                 m_device{VK_NULL_HANDLE};
  nvvk::ResourceAllocator* m_alloc{nullptr};
  nvvk::SamplerPool*       m_samplerPool{};

  float      m_integral{1.F};
  float      m_average{1.F};
  bool       m_valid{false};
  VkExtent2D m_hdrImageSize{1, 1};

  // Resources
  nvvk::Image          m_texHdr;
  nvvk::Buffer         m_accelImpSmpl;
  nvvk::DescriptorPack m_descPack;


  void createDescriptorSetLayout();
};

}  // namespace nvvk
