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

#include <glm/glm.hpp>
#include "vulkan/vulkan_core.h"
#include "nvvk/descriptors.hpp"
#include "nvvk/resource_allocator.hpp"
#include "nvvk/sampler_pool.hpp"


namespace nvshaders {
class Context;


/*-------------------------------------------------------------------------------------------------
# class nvvkhl::HdrEnvDome

>  Use an environment image (HDR) and create the cubic textures for glossy reflection and diffuse illumination. It also has the ability to render the HDR environment, in the background of an image.

 Using 4 compute shaders
 - hdr_dome: to make the HDR as background
 - hdr_integrate_brdf     : generate the BRDF lookup table
 - hdr_prefilter_diffuse  : integrate the diffuse contribution in a cubemap
 - hdr_prefilter_glossy   : integrate the glossy reflection in a cubemap

-------------------------------------------------------------------------------------------------*/
class HdrEnvDome
{
public:
  HdrEnvDome() = default;
  ~HdrEnvDome() { assert(m_device == VK_NULL_HANDLE); }  // Missing deinit() call

  void init(nvvk::ResourceAllocator* allocator, nvvk::SamplerPool* samplerPool, const nvvk::QueueInfo& queueInfo);
  void deinit();


  void create(VkDescriptorSet                  dstSet,
              VkDescriptorSetLayout            dstSetLayout,
              const std::span<const uint32_t>& spirvPrefilterDiffuse,
              const std::span<const uint32_t>& spirvPrefilterGlossy,
              const std::span<const uint32_t>& spirvIntegrateBrdf,
              const std::span<const uint32_t>& spirvDrawDome);

  void setOutImage(const VkDescriptorImageInfo& outimage);
  void draw(const VkCommandBuffer& cmd,
            const glm::mat4&       view,
            const glm::mat4&       proj,
            const VkExtent2D&      size,
            const glm::vec4&       color    = {1.f, 1.f, 1.f, 1.f},  // color multiplier (intensity)
            float                  rotation = 0.F,
            float                  blur     = 0.F);
  void destroy();

  inline VkDescriptorSetLayout getDescLayout() const { return m_hdrPack.getLayout(); }
  inline VkDescriptorSet       getDescSet() const { return m_hdrPack.getSet(0); }

  const std::vector<nvvk::Image> getTextures() const
  {
    return {m_textures.diffuse, m_textures.glossy, m_textures.lutBrdf};
  }

private:
  // Resources
  VkDevice                 m_device{VK_NULL_HANDLE};
  nvvk::ResourceAllocator* m_alloc{nullptr};
  nvvk::SamplerPool*       m_samplerPool{nullptr};

  // From HdrEnv
  VkDescriptorSet       m_hdrEnvSet{VK_NULL_HANDLE};
  VkDescriptorSetLayout m_hdrEnvLayout{VK_NULL_HANDLE};

  // To draw the HDR in image
  nvvk::DescriptorPack m_domePack;
  VkPipeline           m_domePipeline{VK_NULL_HANDLE};
  VkPipelineLayout     m_domePipelineLayout{VK_NULL_HANDLE};

  nvvk::DescriptorPack m_hdrPack;

  VkCommandPool   m_transientCmdPool{};
  nvvk::QueueInfo m_queueInfo;

  struct Textures
  {
    nvvk::Image diffuse;
    nvvk::Image glossy;
    nvvk::Image lutBrdf;
  } m_textures;

  void createDescriptorSetLayout();
  void createDrawPipeline(const std::span<const uint32_t>& spirvDrawDome);
  void integrateBrdf(uint32_t dimension, nvvk::Image& target, const std::span<const uint32_t>& spirvIntegrateBrdf);
  void prefilterHdr(uint32_t dim, nvvk::Image& target, const std::span<const uint32_t>& spirvCode, bool doMipmap);
  void renderToCube(const VkCommandBuffer& cmd, nvvk::Image& target, nvvk::Image& scratch, VkPipelineLayout pipelineLayout, uint32_t dim, uint32_t numMips);
};

}  // namespace nvshaders
