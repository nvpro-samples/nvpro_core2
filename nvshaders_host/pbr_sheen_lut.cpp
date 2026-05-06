/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "pbr_sheen_lut.hpp"

#include "nvvk/barriers.hpp"
#include "nvvk/check_error.hpp"
#include "nvvk/commands.hpp"
#include "nvvk/debug_util.hpp"
#include "nvvk/default_structs.hpp"
#include "nvvk/descriptors.hpp"


namespace nvshaders {

nvvk::Image bakeCharlieSheenLut(nvvk::ResourceAllocator&         allocator,
                                nvvk::SamplerPool&               samplerPool,
                                const nvvk::QueueInfo&           queueInfo,
                                const std::span<const uint32_t>& spirvCode,
                                VkExtent2D                       dim)
{
  VkDevice device = allocator.getDevice();

  // --- Image + linear sampler ---
  VkImageCreateInfo imageInfo = DEFAULT_VkImageCreateInfo;
  imageInfo.extent            = {dim.width, dim.height, 1};
  imageInfo.format            = VK_FORMAT_R16G16_SFLOAT;
  imageInfo.usage             = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;

  nvvk::Image lut{};
  NVVK_CHECK(allocator.createImage(lut, imageInfo, DEFAULT_VkImageViewCreateInfo));
  NVVK_DBG_NAME(lut.image);
  NVVK_DBG_NAME(lut.descriptor.imageView);

  NVVK_CHECK(samplerPool.acquireSampler(lut.descriptor.sampler));
  NVVK_DBG_NAME(lut.descriptor.sampler);

  // nvvk::WriteSetContainer::append reads descriptor.imageLayout, so set it to GENERAL for the
  // storage-image write below.
  lut.descriptor.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

  // --- Descriptor layout (binding 0 = storage image) ---
  nvvk::DescriptorBindings bindings;
  bindings.addBinding(0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT);
  nvvk::DescriptorPack descPack;
  NVVK_CHECK(descPack.init(bindings, device, 1));
  NVVK_DBG_NAME(descPack.getLayout());

  nvvk::WriteSetContainer writes;
  writes.append(descPack.makeWrite(0), lut);

  // --- Pipeline layout with uint2 lutSize push constant ---
  VkPushConstantRange pcRange{
      .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
      .offset     = 0,
      .size       = sizeof(uint32_t) * 2,
  };
  VkPipelineLayout pipelineLayout{VK_NULL_HANDLE};
  NVVK_CHECK(nvvk::createPipelineLayout(device, &pipelineLayout, {descPack.getLayout()}, {pcRange}));
  NVVK_DBG_NAME(pipelineLayout);

  // --- Compute pipeline ---
  VkShaderModuleCreateInfo moduleInfo{
      .sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = spirvCode.size_bytes(),
      .pCode    = spirvCode.data(),
  };
  VkPipelineShaderStageCreateInfo stageInfo{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
      .pNext = &moduleInfo,
      .stage = VK_SHADER_STAGE_COMPUTE_BIT,
      .pName = "main",
  };
  VkComputePipelineCreateInfo cpi{
      .sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
      .stage  = stageInfo,
      .layout = pipelineLayout,
  };
  VkPipeline pipeline{VK_NULL_HANDLE};
  NVVK_CHECK(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cpi, nullptr, &pipeline));
  NVVK_DBG_NAME(pipeline);

  // --- Transient command pool + single-time dispatch ---
  VkCommandPool                 transientPool{VK_NULL_HANDLE};
  const VkCommandPoolCreateInfo poolInfo{
      .sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .flags            = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,
      .queueFamilyIndex = queueInfo.familyIndex,
  };
  NVVK_CHECK(vkCreateCommandPool(device, &poolInfo, nullptr, &transientPool));

  VkCommandBuffer cmd{};
  NVVK_CHECK(nvvk::beginSingleTimeCommands(cmd, device, transientPool));
  {
    NVVK_DBG_SCOPE(cmd);

    nvvk::cmdImageMemoryBarrier(cmd, {lut.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL});
    vkUpdateDescriptorSets(device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, descPack.getSetPtr(), 0, nullptr);

    uint32_t lutSize[2] = {dim.width, dim.height};
    vkCmdPushConstants(cmd, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(lutSize), lutSize);

    // Workgroup size is 8x8 (matches `hdr_charlie_brdf_lut.slang`).
    vkCmdDispatch(cmd, (dim.width + 7) / 8, (dim.height + 7) / 8, 1);

    nvvk::cmdImageMemoryBarrier(cmd, {lut.image, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL});
  }
  NVVK_CHECK(nvvk::endSingleTimeCommands(cmd, device, transientPool, queueInfo.queue));

  lut.descriptor.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

  // --- Cleanup bake-only resources ---
  vkDestroyPipeline(device, pipeline, nullptr);
  vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
  descPack.deinit();
  vkDestroyCommandPool(device, transientPool, nullptr);

  return lut;
}

}  // namespace nvshaders
