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


#include <array>

#include "tonemapper.hpp"
#include "nvvk/check_error.hpp"
#include "nvvk/debug_util.hpp"


void nvshaders::Tonemapper::init(nvvk::ResourceAllocator* alloc, std::span<const uint32_t> spirv)
{
  m_device = alloc->getDevice();

  // Binding layout
  const auto layoutBindings = std::to_array<VkDescriptorSetLayoutBinding>({
      {
          .binding         = shaderio::TonemapBinding::eInput,
          .descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
          .descriptorCount = 1,
          .stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT,
      },
      {
          .binding         = shaderio::TonemapBinding::eOutput,
          .descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
          .descriptorCount = 1,
          .stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT,
      },
  });

  // Descriptor set layout
  const VkDescriptorSetLayoutCreateInfo descriptorSetLayoutInfo{
      .sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .flags        = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT,
      .bindingCount = uint32_t(layoutBindings.size()),
      .pBindings    = layoutBindings.data(),
  };
  NVVK_CHECK(vkCreateDescriptorSetLayout(m_device, &descriptorSetLayoutInfo, nullptr, &m_descriptorSetLayout));
  NVVK_DBG_NAME(m_descriptorSetLayout);

  // Push constant
  const VkPushConstantRange pushConstantRange{
      .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
      .offset     = 0,
      .size       = sizeof(shaderio::TonemapperData),
  };

  // Pipeline layout
  const VkPipelineLayoutCreateInfo pipelineLayoutInfo{
      .sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .setLayoutCount         = 1,
      .pSetLayouts            = &m_descriptorSetLayout,
      .pushConstantRangeCount = 1,
      .pPushConstantRanges    = &pushConstantRange,
  };
  NVVK_CHECK(vkCreatePipelineLayout(m_device, &pipelineLayoutInfo, nullptr, &m_pipelineLayout));
  NVVK_DBG_NAME(m_pipelineLayout);

  // Shader
  VkShaderCreateInfoEXT shaderInfo{
      .sType                  = VK_STRUCTURE_TYPE_SHADER_CREATE_INFO_EXT,
      .stage                  = VK_SHADER_STAGE_COMPUTE_BIT,
      .codeType               = VK_SHADER_CODE_TYPE_SPIRV_EXT,
      .codeSize               = spirv.size_bytes(),
      .pCode                  = spirv.data(),
      .pName                  = "main",
      .setLayoutCount         = 1,
      .pSetLayouts            = &m_descriptorSetLayout,
      .pushConstantRangeCount = 1,
      .pPushConstantRanges    = &pushConstantRange,
  };
  vkCreateShadersEXT(m_device, 1U, &shaderInfo, nullptr, &m_shader);
  NVVK_DBG_NAME(m_shader);
}

void nvshaders::Tonemapper::deinit()
{
  vkDestroyShaderEXT(m_device, m_shader, nullptr);
  vkDestroyDescriptorSetLayout(m_device, m_descriptorSetLayout, nullptr);
  vkDestroyPipelineLayout(m_device, m_pipelineLayout, nullptr);
  m_pipelineLayout      = VK_NULL_HANDLE;
  m_shader              = VK_NULL_HANDLE;
  m_descriptorSetLayout = VK_NULL_HANDLE;
}

void nvshaders::Tonemapper::runCompute(VkCommandBuffer                 cmd,
                                       const VkExtent2D&               size,
                                       const shaderio::TonemapperData& tonemapper,
                                       const VkDescriptorImageInfo&    inImage,
                                       const VkDescriptorImageInfo&    outImage)
{
  NVVK_DBG_SCOPE(cmd);  // <-- Helps to debug in NSight

  // Bind shader
  const VkShaderStageFlagBits stages[1] = {VK_SHADER_STAGE_COMPUTE_BIT};
  vkCmdBindShadersEXT(cmd, 1, stages, &m_shader);

  // Push constant
  vkCmdPushConstants(cmd, m_pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(shaderio::TonemapperData), &tonemapper);

  // Update descriptor sets
  VkWriteDescriptorSet writeDescriptorSet[2]{
      {
          .sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
          .dstSet          = 0,
          .dstBinding      = shaderio::TonemapBinding::eInput,
          .descriptorCount = 1,
          .descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
          .pImageInfo      = &inImage,
      },
      {
          .sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
          .dstSet          = 0,
          .dstBinding      = shaderio::TonemapBinding::eOutput,
          .descriptorCount = 1,
          .descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
          .pImageInfo      = &outImage,
      },
  };
  vkCmdPushDescriptorSetKHR(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineLayout, 0, 2, writeDescriptorSet);

  // Dispatching the compute job
  vkCmdDispatch(cmd, (size.width + 15) / 16, (size.height + 15) / 16, 1);
}
