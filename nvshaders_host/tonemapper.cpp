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
#include <nvvk/barriers.hpp>
#include <nvvk/check_error.hpp>
#include <nvvk/commands.hpp>
#include <nvvk/compute_pipeline.hpp>
#include <nvvk/debug_util.hpp>
#include <nvvk/default_structs.hpp>

#include "nvshaders/tonemap_functions.h.slang"

VkResult nvshaders::Tonemapper::init(nvvk::ResourceAllocator* alloc, std::span<const uint32_t> spirv)
{
  assert(!m_device);
  m_alloc  = alloc;
  m_device = alloc->getDevice();

  // Create buffers
  alloc->createBuffer(m_exposureBuffer, sizeof(float),
                      VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT,
                      VMA_MEMORY_USAGE_AUTO);
  NVVK_DBG_NAME(m_exposureBuffer.buffer);
  alloc->createBuffer(m_histogramBuffer, sizeof(uint32_t) * EXPOSURE_HISTOGRAM_SIZE,
                      VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT,
                      VMA_MEMORY_USAGE_AUTO);


  // Shader descriptor set layout
  nvvk::DescriptorBindings bindings;
  bindings.addBinding(shaderio::TonemapBinding::eImageInput, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT);
  bindings.addBinding(shaderio::TonemapBinding::eImageOutput, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT);
  bindings.addBinding(shaderio::TonemapBinding::eHistogramInputOutput, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT);
  bindings.addBinding(shaderio::TonemapBinding::eLuminanceInputOutput, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT);

  NVVK_CHECK(m_descriptorPack.init(bindings, m_device, 0, VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR));
  NVVK_DBG_NAME(m_descriptorPack.getLayout());

  // Push constant
  VkPushConstantRange pushConstantRange{.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT, .size = sizeof(shaderio::TonemapperData)};

  // Pipeline layout
  const VkPipelineLayoutCreateInfo pipelineLayoutInfo{
      .sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .setLayoutCount         = 1,
      .pSetLayouts            = m_descriptorPack.getLayoutPtr(),
      .pushConstantRangeCount = 1,
      .pPushConstantRanges    = &pushConstantRange,
  };
  NVVK_FAIL_RETURN(vkCreatePipelineLayout(m_device, &pipelineLayoutInfo, nullptr, &m_pipelineLayout));
  NVVK_DBG_NAME(m_pipelineLayout);

  // Compute Pipeline
  VkComputePipelineCreateInfo compInfo   = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
  VkShaderModuleCreateInfo    shaderInfo = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
  compInfo.stage                         = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
  compInfo.stage.stage                   = VK_SHADER_STAGE_COMPUTE_BIT;
  compInfo.stage.pNext                   = &shaderInfo;
  compInfo.layout                        = m_pipelineLayout;

  shaderInfo.codeSize = uint32_t(spirv.size_bytes());  // All shaders are in the same spirv
  shaderInfo.pCode    = spirv.data();

  // Tonemap Pipelines
  compInfo.stage.pName = "Tonemap";
  NVVK_FAIL_RETURN(vkCreateComputePipelines(m_device, nullptr, 1, &compInfo, nullptr, &m_tonemapPipeline));
  NVVK_DBG_NAME(m_tonemapPipeline);

  // Auto-Exposure Pipelines
  compInfo.stage.pName = "Histogram";
  NVVK_FAIL_RETURN(vkCreateComputePipelines(m_device, nullptr, 1, &compInfo, nullptr, &m_histogramPipeline));
  NVVK_DBG_NAME(m_histogramPipeline);

  compInfo.stage.pName = "AutoExposure";
  NVVK_FAIL_RETURN(vkCreateComputePipelines(m_device, nullptr, 1, &compInfo, nullptr, &m_exposurePipeline));
  NVVK_DBG_NAME(m_exposurePipeline);

  return VK_SUCCESS;
}

void nvshaders::Tonemapper::deinit()
{
  if(!m_device)
    return;

  m_alloc->destroyBuffer(m_exposureBuffer);
  m_alloc->destroyBuffer(m_histogramBuffer);

  vkDestroyPipeline(m_device, m_tonemapPipeline, nullptr);
  vkDestroyPipeline(m_device, m_histogramPipeline, nullptr);
  vkDestroyPipeline(m_device, m_exposurePipeline, nullptr);
  vkDestroyPipelineLayout(m_device, m_pipelineLayout, nullptr);
  m_descriptorPack.deinit();

  m_pipelineLayout  = VK_NULL_HANDLE;
  m_tonemapPipeline = VK_NULL_HANDLE;
  m_device          = VK_NULL_HANDLE;
}


//----------------------------------
// Run the tonemapper compute shader
//
void nvshaders::Tonemapper::runCompute(VkCommandBuffer                 cmd,
                                       const VkExtent2D&               size,
                                       const shaderio::TonemapperData& tonemapper,
                                       const VkDescriptorImageInfo&    inImage,
                                       const VkDescriptorImageInfo&    outImage)
{
  NVVK_DBG_SCOPE(cmd);  // <-- Helps to debug in NSight

  // Push constant
  shaderio::TonemapperData tonemapperData = tonemapper;
  tonemapperData.autoExposureSpeed *= float(m_timer.getSeconds());
  tonemapperData.inputMatrix =
      shaderio::getColorCorrectionMatrix(tonemapperData.exposure, tonemapper.temperature, tonemapper.tint);
  vkCmdPushConstants(cmd, m_pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(shaderio::TonemapperData), &tonemapperData);
  m_timer.reset();

  // Push information to the descriptor set
  nvvk::WriteSetContainer writeSetContainer;
  writeSetContainer.append(m_descriptorPack.makeWrite(shaderio::TonemapBinding::eImageInput), inImage);
  writeSetContainer.append(m_descriptorPack.makeWrite(shaderio::TonemapBinding::eImageOutput), outImage);
  writeSetContainer.append(m_descriptorPack.makeWrite(shaderio::TonemapBinding::eHistogramInputOutput), m_histogramBuffer);
  writeSetContainer.append(m_descriptorPack.makeWrite(shaderio::TonemapBinding::eLuminanceInputOutput), m_exposureBuffer);
  vkCmdPushDescriptorSetKHR(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineLayout, 0, writeSetContainer.size(),
                            writeSetContainer.data());

  // Run auto-exposure histogram/exposure if enabled
  if(tonemapper.isActive && tonemapper.autoExposure)
  {
    static bool firstRun = true;
    if(firstRun)
    {
      clearHistogram(cmd);
      firstRun = false;
    }

    runAutoExposureHistogram(cmd, size, inImage);
    runAutoExposure(cmd);
  }

  // Run tonemapper compute shader
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_tonemapPipeline);
  VkExtent2D groupSize = nvvk::getGroupCounts(size, VkExtent2D{TONEMAP_WORKGROUP_SIZE, TONEMAP_WORKGROUP_SIZE});
  vkCmdDispatch(cmd, groupSize.width, groupSize.height, 1);
}

void nvshaders::Tonemapper::runAutoExposureHistogram(VkCommandBuffer cmd, const VkExtent2D& size, const VkDescriptorImageInfo& inImage)
{
  NVVK_DBG_SCOPE(cmd);  // <-- Helps to debug in NSight

  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_histogramPipeline);
  VkExtent2D groupSize = nvvk::getGroupCounts(size, TONEMAP_WORKGROUP_SIZE);
  vkCmdDispatch(cmd, groupSize.width, groupSize.height, 1);
  nvvk::cmdBufferMemoryBarrier(cmd, {.buffer        = m_histogramBuffer.buffer,
                                     .srcStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                     .dstStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                     .srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                                     .dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT});
}

void nvshaders::Tonemapper::runAutoExposure(VkCommandBuffer cmd)
{
  NVVK_DBG_SCOPE(cmd);  // <-- Helps to debug in NSight
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_exposurePipeline);
  vkCmdDispatch(cmd, 1, 1, 1);
  nvvk::cmdBufferMemoryBarrier(cmd, {.buffer        = m_exposureBuffer.buffer,
                                     .srcStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                     .dstStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                     .srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                                     .dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT});
}

void nvshaders::Tonemapper::clearHistogram(VkCommandBuffer cmd)
{
  std::array<uint32_t, EXPOSURE_HISTOGRAM_SIZE> histogramData{0};
  vkCmdUpdateBuffer(cmd, m_histogramBuffer.buffer, 0, sizeof(uint32_t) * EXPOSURE_HISTOGRAM_SIZE, histogramData.data());

  // Add barrier to ensure update buffer completes before compute shader writes to the buffer
  nvvk::cmdBufferMemoryBarrier(cmd, {.buffer        = m_histogramBuffer.buffer,
                                     .srcStageMask  = VK_PIPELINE_STAGE_2_CLEAR_BIT,
                                     .dstStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                     .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
                                     .dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT});
}