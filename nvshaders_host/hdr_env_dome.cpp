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


#define _USE_MATH_DEFINES
#include <array>
#include <cmath>
#include <iostream>
#include <numeric>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "nvshaders/slang_types.h"
#include "nvshaders/hdr_io.h.slang"


#include "hdr_env_dome.hpp"
#include "nvvk/debug_util.hpp"
#include "nvvk/compute_pipeline.hpp"
#include "nvvk/descriptors.hpp"
#include "nvutils/timers.hpp"
#include "nvvk/check_error.hpp"
#include "nvvk/default_structs.hpp"
#include "nvvk/commands.hpp"
#include "nvvk/command_pools.hpp"
#include "nvvk/shaders.hpp"


namespace nvshaders {


//--------------------------------------------------------------------------------------------------
//
//
void HdrEnvDome::init(nvvk::ResourceAllocator* allocator, nvvk::SamplerPool* samplerPool, const nvvk::QueueInfo& queueInfo)
{
  m_device      = allocator->getDevice();
  m_alloc       = allocator;
  m_samplerPool = samplerPool;
  m_queueInfo   = queueInfo;
}


void HdrEnvDome::deinit()
{
  destroy();
  m_device = {};
}

//--------------------------------------------------------------------------------------------------
// The descriptor set and layout are from the HdrIbl class
// - it consists of the HDR image and the acceleration structure
// - those will be used to create the diffuse and glossy image
// - Also use to 'clear' the image with the background image
//
void HdrEnvDome::create(VkDescriptorSet                  dstSet,
                        VkDescriptorSetLayout            dstSetLayout,
                        const std::span<const uint32_t>& spirvPrefilterDiffuse,
                        const std::span<const uint32_t>& spirvPrefilterGlossy,
                        const std::span<const uint32_t>& spirvIntegrateBrdf,
                        const std::span<const uint32_t>& spirvDrawDome)
{
  destroy();
  m_hdrEnvSet    = dstSet;
  m_hdrEnvLayout = dstSetLayout;


  const VkCommandPoolCreateInfo commandPoolCreateInfo{
      .sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .flags            = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,  // Hint that commands will be short-lived
      .queueFamilyIndex = m_queueInfo.familyIndex,
  };
  NVVK_CHECK(vkCreateCommandPool(m_device, &commandPoolCreateInfo, nullptr, &m_transientCmdPool));
  NVVK_DBG_NAME(m_transientCmdPool);

  createDrawPipeline(spirvDrawDome);
  integrateBrdf(512, m_textures.lutBrdf, spirvIntegrateBrdf);
  prefilterHdr(128, m_textures.diffuse, spirvPrefilterDiffuse, false);
  prefilterHdr(512, m_textures.glossy, spirvPrefilterGlossy, true);
  createDescriptorSetLayout();

  NVVK_DBG_NAME(m_textures.lutBrdf.image);
  NVVK_DBG_NAME(m_textures.diffuse.image);
  NVVK_DBG_NAME(m_textures.glossy.image);

  vkDestroyCommandPool(m_device, m_transientCmdPool, nullptr);
}

//--------------------------------------------------------------------------------------------------
// This is the image the HDR will be write to, a framebuffer image or an off-screen image
//
void HdrEnvDome::setOutImage(const VkDescriptorImageInfo& outimage)
{
  VkWriteDescriptorSet wds{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
  wds.dstSet          = m_domePack.getSet(0);
  wds.dstBinding      = shaderio::EnvDomeDraw::eHdrImage;
  wds.descriptorCount = 1;
  wds.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
  wds.pImageInfo      = &outimage;
  vkUpdateDescriptorSets(m_device, 1, &wds, 0, nullptr);
}


//--------------------------------------------------------------------------------------------------
// Compute Pipeline to "Clear" the image with the HDR as background
//
void HdrEnvDome::createDrawPipeline(const std::span<const uint32_t>& spirvDrawDome)
{
  nvvk::DescriptorBindings bindings;
  // Descriptor: the output image
  bindings.addBinding(shaderio::EnvDomeDraw::eHdrImage, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT);
  NVVK_CHECK(m_domePack.init(bindings, m_device, 1));
  NVVK_DBG_NAME(m_domePack.getLayout());
  NVVK_DBG_NAME(m_domePack.getPool());
  NVVK_DBG_NAME(m_domePack.getSet(0));

  // Creating the pipeline layout
  const VkPushConstantRange pushConstantRange{
      .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT, .offset = 0, .size = sizeof(shaderio::HdrDomePushConstant)};
  NVVK_CHECK(nvvk::createPipelineLayout(m_device, &m_domePipelineLayout, {m_domePack.getLayout(), m_hdrEnvLayout}, {pushConstantRange}));
  NVVK_DBG_NAME(m_domePipelineLayout);

  VkShaderModuleCreateInfo moduleInfo = {
      .sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = spirvDrawDome.size_bytes(),
      .pCode    = spirvDrawDome.data(),
  };

  // HDR Dome compute shader
  VkPipelineShaderStageCreateInfo stageInfo{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
  stageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  stageInfo.pName = "main";
  stageInfo.pNext = &moduleInfo;

  VkComputePipelineCreateInfo compInfo{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
  compInfo.layout = m_domePipelineLayout;
  compInfo.stage  = stageInfo;

  vkCreateComputePipelines(m_device, {}, 1, &compInfo, nullptr, &m_domePipeline);
  NVVK_DBG_NAME(m_domePipeline);
}


//--------------------------------------------------------------------------------------------------
// Draw the HDR to the image (setOutImage)
// - view and projection matrix should come from the camera
// - size is the image output size (framebuffer size)
// - color is the color multiplier of the HDR (intensity)
//
void HdrEnvDome::draw(const VkCommandBuffer& cmd,
                      const glm::mat4&       view,
                      const glm::mat4&       proj,
                      const VkExtent2D&      size,
                      const glm::vec4&       color,  // color multiplier (intensity)
                      float                  rotation /*= 0.F*/,
                      float                  blur /*= 0.F*/)
{
  NVVK_DBG_SCOPE(cmd);

  // Information to the compute shader
  shaderio::HdrDomePushConstant pushConst{};
  glm::mat4                     noTranslate = view;
  noTranslate[3]                            = glm::vec4(0, 0, 0, 1);  // Remove translation
  pushConst.mvp = glm::inverse(noTranslate) * glm::inverse(proj);  // This will be to have a world direction vector pointing to the pixel
  pushConst.multColor = color;
  pushConst.rotation  = rotation;
  pushConst.blur      = blur;

  // Execution
  std::vector<VkDescriptorSet> dst_sets{m_domePack.getSet(0), m_hdrEnvSet};
  vkCmdPushConstants(cmd, m_domePipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(shaderio::HdrDomePushConstant), &pushConst);
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_domePipelineLayout, 0,
                          static_cast<uint32_t>(dst_sets.size()), dst_sets.data(), 0, nullptr);
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_domePipeline);
  VkExtent2D group_counts = nvvk::getGroupCounts(size, HDR_WORKGROUP_SIZE);
  vkCmdDispatch(cmd, group_counts.width, group_counts.height, 1);
}


//--------------------------------------------------------------------------------------------------
//
//
void HdrEnvDome::destroy()
{
  m_samplerPool->releaseSampler(m_textures.diffuse.descriptor.sampler);
  m_samplerPool->releaseSampler(m_textures.lutBrdf.descriptor.sampler);
  m_samplerPool->releaseSampler(m_textures.glossy.descriptor.sampler);
  m_alloc->destroyImage(m_textures.diffuse);
  m_alloc->destroyImage(m_textures.lutBrdf);
  m_alloc->destroyImage(m_textures.glossy);

  vkDestroyPipeline(m_device, m_domePipeline, nullptr);
  vkDestroyPipelineLayout(m_device, m_domePipelineLayout, nullptr);
  m_domePack.deinit();
  m_hdrPack.deinit();
}


//--------------------------------------------------------------------------------------------------
// Descriptors of the HDR and the acceleration structure
//
void HdrEnvDome::createDescriptorSetLayout()
{
  nvvk::DescriptorBindings bindings;
  bindings.addBinding(shaderio::EnvDomeBindings::eHdrBrdf, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_ALL);  // HDR image
  bindings.addBinding(shaderio::EnvDomeBindings::eHdrDiffuse, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_ALL);  // HDR image
  bindings.addBinding(shaderio::EnvDomeBindings::eHdrSpecular, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_ALL);  // HDR image

  NVVK_CHECK(m_hdrPack.init(bindings, m_device, 1));
  NVVK_DBG_NAME(m_hdrPack.getLayout());
  NVVK_DBG_NAME(m_hdrPack.getPool());
  NVVK_DBG_NAME(m_hdrPack.getSet(0));

  nvvk::WriteSetContainer writeContainer;
  writeContainer.append(m_hdrPack.makeWrite(shaderio::EnvDomeBindings::eHdrBrdf), m_textures.lutBrdf);
  writeContainer.append(m_hdrPack.makeWrite(shaderio::EnvDomeBindings::eHdrDiffuse), m_textures.diffuse);
  writeContainer.append(m_hdrPack.makeWrite(shaderio::EnvDomeBindings::eHdrSpecular), m_textures.glossy);
  vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(writeContainer.size()), writeContainer.data(), 0, nullptr);
}


//--------------------------------------------------------------------------------------------------
// Pre-integrate glossy BRDF, see
// http://blog.selfshadow.com/publications/s2013-shading-course/karis/s2013_pbs_epic_notes_v2.pdf
void HdrEnvDome::integrateBrdf(uint32_t dimension, nvvk::Image& target, const std::span<const uint32_t>& spirvIntegrateBrdf)
{
  nvutils::ScopedTimer st(__FUNCTION__);

  // Create an image RG16 to store the BRDF
  VkImageCreateInfo imageInfo = DEFAULT_VkImageCreateInfo;
  imageInfo.extent            = {dimension, dimension, 1};
  imageInfo.format            = VK_FORMAT_R16G16_SFLOAT;
  imageInfo.usage             = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;

  NVVK_CHECK(m_alloc->createImage(target, imageInfo, DEFAULT_VkImageViewCreateInfo));
  NVVK_DBG_NAME(target.image);
  NVVK_DBG_NAME(target.descriptor.imageView);
  m_samplerPool->acquireSampler(target.descriptor.sampler);
  NVVK_DBG_NAME(target.descriptor.sampler);

  target.descriptor.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

  // Compute shader
  nvvk::DescriptorPack descPack;
  VkPipeline           pipeline{VK_NULL_HANDLE};
  VkPipelineLayout     pipelineLayout{VK_NULL_HANDLE};

  VkCommandBuffer cmd{};
  NVVK_CHECK(nvvk::beginSingleTimeCommands(cmd, m_device, m_transientCmdPool));
  {
    NVVK_DBG_SCOPE(cmd);

    // Change image layout to general
    nvvk::cmdImageMemoryBarrier(cmd, {target.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL});

    // The output image is the one we have just created
    nvvk::DescriptorBindings bindings;
    bindings.addBinding(shaderio::EnvDomeDraw::eHdrImage, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT);
    NVVK_CHECK(descPack.init(bindings, m_device, 1));
    NVVK_DBG_NAME(descPack.getLayout());
    NVVK_DBG_NAME(descPack.getPool());
    NVVK_DBG_NAME(descPack.getSet(0));

    // Writing the output image
    nvvk::WriteSetContainer writeContainer;
    writeContainer.append(descPack.makeWrite(shaderio::EnvDomeDraw::eHdrImage), target);
    vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(writeContainer.size()), writeContainer.data(), 0, nullptr);

    // Creating the pipeline
    NVVK_CHECK(nvvk::createPipelineLayout(m_device, &pipelineLayout, {descPack.getLayout()}));
    NVVK_DBG_NAME(pipelineLayout);

    VkShaderModuleCreateInfo moduleInfo = {
        .sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = spirvIntegrateBrdf.size_bytes(),
        .pCode    = spirvIntegrateBrdf.data(),
    };

    VkPipelineShaderStageCreateInfo stage_info{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stage_info.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stage_info.pName = "main";
    stage_info.pNext = &moduleInfo;

    VkComputePipelineCreateInfo comp_info{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    comp_info.layout = pipelineLayout;
    comp_info.stage  = stage_info;

    vkCreateComputePipelines(m_device, {}, 1, &comp_info, nullptr, &pipeline);

    // Run shader
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, descPack.getSetPtr(), 0, nullptr);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);

    VkExtent2D group_counts = nvvk::getGroupCounts({dimension, dimension}, HDR_WORKGROUP_SIZE);
    vkCmdDispatch(cmd, group_counts.width, group_counts.height, 1);
  }
  nvvk::endSingleTimeCommands(cmd, m_device, m_transientCmdPool, m_queueInfo.queue);

  // Clean up
  vkDestroyPipeline(m_device, pipeline, nullptr);
  vkDestroyPipelineLayout(m_device, pipelineLayout, nullptr);
  descPack.deinit();
}


//--------------------------------------------------------------------------------------------------
//
//
void HdrEnvDome::prefilterHdr(uint32_t dim, nvvk::Image& target, const std::span<const uint32_t>& spirvData, bool doMipmap)
{
  const VkExtent2D size{dim, dim};
  VkFormat         format     = VK_FORMAT_R16G16B16A16_SFLOAT;
  const uint32_t   numMipmaps = doMipmap ? static_cast<uint32_t>(floor(::log2(dim))) + 1 : 1;

  nvutils::ScopedTimer st("%s: %u", __FUNCTION__, numMipmaps);

  VkSamplerCreateInfo samplerCreateInfo = DEFAULT_VkSamplerCreateInfo;
  samplerCreateInfo.maxLod              = static_cast<float>(numMipmaps);


  {  // Target - cube
    VkImageCreateInfo imageInfo = DEFAULT_VkImageCreateInfo;
    imageInfo.flags             = VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
    imageInfo.extent            = {dim, dim, 1};
    imageInfo.imageType         = VK_IMAGE_TYPE_2D;
    imageInfo.format            = format;
    imageInfo.mipLevels         = numMipmaps;
    imageInfo.arrayLayers       = 6;  // Cube
    imageInfo.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT
                      | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    VkImageViewCreateInfo imageView = DEFAULT_VkImageViewCreateInfo;
    imageView.viewType              = VK_IMAGE_VIEW_TYPE_CUBE;


    NVVK_CHECK(m_alloc->createImage(target, imageInfo, imageView));
    NVVK_DBG_NAME(target.image);
    NVVK_DBG_NAME(target.descriptor.imageView);
    target.descriptor.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    m_samplerPool->acquireSampler(target.descriptor.sampler, samplerCreateInfo);
  }

  nvvk::Image scratchTexture;
  {  // Scratch texture
    VkImageCreateInfo imageInfo = DEFAULT_VkImageCreateInfo;
    imageInfo.extent            = {dim, dim, 1};
    imageInfo.format            = format;
    imageInfo.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;

    NVVK_CHECK(m_alloc->createImage(scratchTexture, imageInfo, DEFAULT_VkImageViewCreateInfo));
    NVVK_DBG_NAME(scratchTexture.image);
    NVVK_DBG_NAME(scratchTexture.descriptor.imageView);
    scratchTexture.descriptor.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    m_samplerPool->acquireSampler(scratchTexture.descriptor.sampler, samplerCreateInfo);
  }


  // Compute shader
  VkPipeline       pipeline{VK_NULL_HANDLE};
  VkPipelineLayout pipelineLayout{VK_NULL_HANDLE};

  // Descriptors
  nvvk::DescriptorPack     descPack;
  nvvk::DescriptorBindings bindings;
  bindings.addBinding(shaderio::EnvDomeDraw::eHdrImage, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT);
  NVVK_CHECK(descPack.init(bindings, m_device, 1));
  NVVK_DBG_NAME(descPack.getLayout());
  NVVK_DBG_NAME(descPack.getPool());
  NVVK_DBG_NAME(descPack.getSet(0));

  nvvk::WriteSetContainer writeContainer;
  writeContainer.append(descPack.makeWrite(shaderio::EnvDomeDraw::eHdrImage), scratchTexture);
  vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(writeContainer.size()), writeContainer.data(), 0, nullptr);

  // Creating the pipeline
  const VkPushConstantRange pushConstantRange{.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT, .offset = 0, .size = sizeof(shaderio::HdrPushBlock)};
  NVVK_CHECK(nvvk::createPipelineLayout(m_device, &pipelineLayout, {descPack.getLayout(), m_hdrEnvLayout}, {pushConstantRange}));

  VkShaderModuleCreateInfo moduleInfo = {
      .sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = spirvData.size_bytes(),
      .pCode    = spirvData.data(),
  };

  VkPipelineShaderStageCreateInfo stageInfo{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
  stageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  stageInfo.pNext = &moduleInfo;
  stageInfo.pName = "main";

  VkComputePipelineCreateInfo comp_info{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
  comp_info.layout = pipelineLayout;
  comp_info.stage  = stageInfo;

  vkCreateComputePipelines(m_device, {}, 1, &comp_info, nullptr, &pipeline);

  {
    VkCommandBuffer cmd{};
    NVVK_CHECK(nvvk::beginSingleTimeCommands(cmd, m_device, m_transientCmdPool));

    // Change scratch to general
    nvvk::cmdImageMemoryBarrier(cmd, {scratchTexture.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL});
    // Change target to destination
    VkImageSubresourceRange subresourceRange{VK_IMAGE_ASPECT_COLOR_BIT, 0, numMipmaps, 0, 6};
    nvvk::cmdImageMemoryBarrier(cmd, {target.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, subresourceRange});

    std::array<VkDescriptorSet, 2> dstSets{descPack.getSet(0), m_hdrEnvSet};
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0,
                            static_cast<uint32_t>(dstSets.size()), dstSets.data(), 0, nullptr);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);

    renderToCube(cmd, target, scratchTexture, pipelineLayout, dim, numMipmaps);

    nvvk::endSingleTimeCommands(cmd, m_device, m_transientCmdPool, m_queueInfo.queue);
  }

  // Clean up
  vkDestroyPipeline(m_device, pipeline, nullptr);
  vkDestroyPipelineLayout(m_device, pipelineLayout, nullptr);
  descPack.deinit();

  m_alloc->destroyImage(scratchTexture);
}


//--------------------------------------------------------------------------------------------------
//
//
void HdrEnvDome::renderToCube(const VkCommandBuffer& cmd,
                              nvvk::Image&           target,
                              nvvk::Image&           scratch,
                              VkPipelineLayout       pipelineLayout,
                              uint32_t               dim,
                              uint32_t               numMips)
{
  NVVK_DBG_SCOPE(cmd);

  glm::mat4 mat_pers = glm::perspectiveRH_ZO(glm::radians(90.0F), 1.0F, 0.1F, 10.0F);
  mat_pers[1][1] *= -1.0F;
  mat_pers = glm::inverse(mat_pers);

  std::array<glm::mat4, 6> mv;
  const glm::vec3          pos(0.0F, 0.0F, 0.0F);
  mv[0] = glm::lookAt(pos, glm::vec3(1.0F, 0.0F, 0.0F), glm::vec3(0.0F, -1.0F, 0.0F));   // Positive X
  mv[1] = glm::lookAt(pos, glm::vec3(-1.0F, 0.0F, 0.0F), glm::vec3(0.0F, -1.0F, 0.0F));  // Negative X
  mv[2] = glm::lookAt(pos, glm::vec3(0.0F, -1.0F, 0.0F), glm::vec3(0.0F, 0.0F, -1.0F));  // Positive Y
  mv[3] = glm::lookAt(pos, glm::vec3(0.0F, 1.0F, 0.0F), glm::vec3(0.0F, 0.0F, 1.0F));    // Negative Y
  mv[4] = glm::lookAt(pos, glm::vec3(0.0F, 0.0F, 1.0F), glm::vec3(0.0F, -1.0F, 0.0F));   // Positive Z
  mv[5] = glm::lookAt(pos, glm::vec3(0.0F, 0.0F, -1.0F), glm::vec3(0.0F, -1.0F, 0.0F));  // Negative Z
  for(auto& m : mv)
    m = glm::inverse(m);

  // Change image layout for all cubemap faces to transfer destination
  VkImageSubresourceRange subresourceRange{VK_IMAGE_ASPECT_COLOR_BIT, 0, numMips, 0, 6};
  nvvk::cmdImageMemoryBarrier(cmd, {target.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, subresourceRange});


  // Image barrier for compute stage
  auto barrier = [&](VkImageLayout oldLayout, VkImageLayout newLayout, VkAccessFlags srcAccess, VkAccessFlags dstAccess,
                     VkPipelineStageFlags srcStage, VkPipelineStageFlags dstStage) {
    VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    VkImageMemoryBarrier    imageMemoryBarrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    imageMemoryBarrier.oldLayout           = oldLayout;
    imageMemoryBarrier.newLayout           = newLayout;
    imageMemoryBarrier.image               = scratch.image;
    imageMemoryBarrier.subresourceRange    = range;
    imageMemoryBarrier.srcAccessMask       = srcAccess;
    imageMemoryBarrier.dstAccessMask       = dstAccess;
    imageMemoryBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    imageMemoryBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    vkCmdPipelineBarrier(cmd, srcStage, dstStage, 0, 0, nullptr, 0, nullptr, 1, &imageMemoryBarrier);
  };


  VkExtent3D             extent{dim, dim, 1};
  shaderio::HdrPushBlock push_block{};

  for(uint32_t mip = 0; mip < numMips; mip++)
  {
    for(uint32_t f = 0; f < 6; f++)
    {
      // Update shader push constant block
      float roughness       = static_cast<float>(mip) / static_cast<float>(numMips - 1);
      push_block.roughness  = roughness;
      push_block.mvp        = mv[f] * mat_pers;
      push_block.size       = glm::vec2(glm::uvec2(extent.width, extent.height));
      push_block.numSamples = 1024 / (mip + 1);
      vkCmdPushConstants(cmd, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(shaderio::HdrPushBlock), &push_block);

      // Execute compute shader
      VkExtent2D group_counts = nvvk::getGroupCounts({extent.width, extent.height}, HDR_WORKGROUP_SIZE);
      vkCmdDispatch(cmd, group_counts.width, group_counts.height, 1);

      // Wait for compute to finish before copying
      barrier(VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_ACCESS_MEMORY_WRITE_BIT,
              VK_ACCESS_TRANSFER_READ_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);

      // Copy region for transfer from framebuffer to cube face
      VkImageCopy copy_region{};
      copy_region.srcSubresource.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
      copy_region.srcSubresource.baseArrayLayer = 0;
      copy_region.srcSubresource.mipLevel       = 0;
      copy_region.srcSubresource.layerCount     = 1;
      copy_region.dstSubresource.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
      copy_region.dstSubresource.baseArrayLayer = f;
      copy_region.dstSubresource.mipLevel       = mip;
      copy_region.dstSubresource.layerCount     = 1;
      copy_region.extent                        = extent;

      vkCmdCopyImage(cmd, scratch.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, target.image,
                     VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy_region);

      // Transform scratch texture back to general
      // After copy
      barrier(VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_TRANSFER_READ_BIT,
              VK_ACCESS_SHADER_WRITE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    }

    // Next mipmap level
    if(extent.width > 1)
      extent.width /= 2;
    if(extent.height > 1)
      extent.height /= 2;
  }


  nvvk::cmdImageMemoryBarrier(cmd, {target.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL, subresourceRange});
}

}  // namespace nvshaders
