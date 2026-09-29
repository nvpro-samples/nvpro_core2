/*
 * Copyright (c) 2022-2026, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */


#pragma once
//////////////////////////////////////////////////////////////////////////

#include <array>
#include <vector>

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
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

  // Re-run only the diffuse + glossy prefilter dispatches against the existing cubes.
  // Intended for environment-image producers (nvvk::HdrIbl or an equivalent) that have just
  // updated the HDR content that `hdrEnvSet` samples from; the caller owns `cmd` and is
  // responsible for the surrounding begin / end / submit. Skips re-creating the cube images
  // and the BRDF LUT (both allocated once by create()), avoiding the multi-hundred-millisecond
  // teardown-and-realloc that a full create() would incur.
  //
  // `hdrEnvSet` may be the same descriptor set passed to create() (contents rewritten by the
  // producer) or a new one; either way it must be bound-compatible with the layout passed to
  // create().
  // Which prefiltered cubes updateEnvironment() re-dispatches.
  //
  // The two are not the same price: the glossy cube is 512x512x6 with a mip chain and costs
  // roughly twice the 128x128x6 diffuse one. A caller that is refreshing the environment every
  // frame -- while a slider is being dragged -- can take the diffuse cube alone and let specular
  // reflections lag by the length of the drag, which is far less noticeable than the frame rate
  // that refreshing both costs.
  enum class PrefilterSet
  {
    eAll,          // Diffuse + glossy. Correct result; the price of a parameter commit.
    eDiffuseOnly,  // Diffuse only. For per-frame refreshes during an interaction.
  };

  void updateEnvironment(VkCommandBuffer cmd, VkDescriptorSet hdrEnvSet, PrefilterSet which = PrefilterSet::eAll);

  void setOutImage(const VkDescriptorImageInfo& outimage);
  // `rotation` takes the HDR's frame to world: any unit quaternion, or an angle about +Y in radians.
  void draw(const VkCommandBuffer& cmd,
            const glm::mat4&       view,
            const glm::mat4&       proj,
            const VkExtent2D&      size,
            const glm::vec4&       color,  // color multiplier (intensity)
            const glm::quat&       rotation,
            float                  blur = 0.F);
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

  // Persistent per-target state for the diffuse / glossy prefilter passes. Held between
  // create() and destroy() so updateEnvironment() can re-dispatch against the same scratch
  // image, pipeline, and descriptor pack without reallocating -- the expensive part of a
  // create() is the image allocation, not the compute dispatches themselves.
  struct PrefilterCtx
  {
    nvvk::Image          scratch;
    VkPipeline           pipeline{VK_NULL_HANDLE};
    VkPipelineLayout     pipelineLayout{VK_NULL_HANDLE};
    nvvk::DescriptorPack descPack;
    uint32_t             dim{0};
    uint32_t             numMipmaps{0};
  } m_diffuseCtx, m_glossyCtx;

  void createDescriptorSetLayout();
  void createDrawPipeline(const std::span<const uint32_t>& spirvDrawDome);
  void integrateBrdf(uint32_t dimension, nvvk::Image& target, const std::span<const uint32_t>& spirvIntegrateBrdf);
  // `lowestMipLevel` (only relevant when doMipmap=true) caps the mip chain at
  // `floor(log2(dim)) + 1 - lowestMipLevel`, mirroring the Khronos glTF-Sample-Renderer
  // convention -- the smallest baked mip is `dim >> (mipCount - 1)`. 0 = no cap (full chain).
  // 4 matches Khronos for 256-pix cubes (= 5 mips, smallest = 16x16).
  //
  // prefilterHdrAllocate creates the target cube, the scratch image, the pipeline, and the
  // descriptor pack; prefilterHdrRun binds and dispatches against a caller-owned command buffer.
  // Splitting them lets create() do a one-shot init (via the prefilterHdr wrapper) while
  // updateEnvironment() re-runs just the dispatch half using an externally supplied cmd.
  void prefilterHdrAllocate(uint32_t                         dim,
                            nvvk::Image&                     target,
                            PrefilterCtx&                    ctx,
                            const std::span<const uint32_t>& spirvCode,
                            bool                             doMipmap,
                            uint32_t                         lowestMipLevel = 0);
  void prefilterHdrRun(VkCommandBuffer cmd, nvvk::Image& target, PrefilterCtx& ctx);
  // Thin allocate + one-shot dispatch wrapper used only by create() -- preserves the original
  // one-submit-per-target behaviour so existing callers see zero change.
  void prefilterHdr(uint32_t                         dim,
                    nvvk::Image&                     target,
                    PrefilterCtx&                    ctx,
                    const std::span<const uint32_t>& spirvCode,
                    bool                             doMipmap,
                    uint32_t                         lowestMipLevel = 0);
  void renderToCube(const VkCommandBuffer& cmd, nvvk::Image& target, nvvk::Image& scratch, VkPipelineLayout pipelineLayout, uint32_t dim, uint32_t numMips);
};

}  // namespace nvshaders
