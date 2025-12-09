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

#include <glm/glm.hpp>

#include "check_error.hpp"
#include "debug_util.hpp"
#include "descriptors.hpp"
#include "helpers.hpp"
#include "ray_picker.hpp"
#include "resource_allocator.hpp"


void nvvk::RayPicker::init(nvvk::ResourceAllocator* allocator)
{
  m_alloc = allocator;

  createOutputResult();
  createDescriptorSet();
  createPipeline();
}

void nvvk::RayPicker::deinit()
{
  if(m_alloc == nullptr)
  {
    return;
  }
  VkDevice device = m_alloc->getDevice();

  m_alloc->destroyBuffer(m_pickResult);
  m_alloc->destroyBuffer(m_sbtBuffer);
  vkDestroyDescriptorSetLayout(device, m_descriptorSetLayout, nullptr);
  vkDestroyPipeline(device, m_pipeline, nullptr);
  vkDestroyPipelineLayout(device, m_pipelineLayout, nullptr);

  m_pickResult          = {};
  m_descriptorSetLayout = {};
  m_pipelineLayout      = {};
  m_pipeline            = {};

  m_alloc = nullptr;
}

bool nvvk::RayPicker::isValid() const
{
  return m_pipeline != VK_NULL_HANDLE;
}

void nvvk::RayPicker::run(VkCommandBuffer cmd, const PickInfo& pickInfo)
{
  nvvk::WriteSetContainer writeContainer;
  writeContainer.append(m_bindings.getWriteSet(0), pickInfo.tlas);
  writeContainer.append(m_bindings.getWriteSet(1), m_pickResult);

  VkPushDescriptorSetInfo pushDescriptorSetInfo{
      .sType                = VK_STRUCTURE_TYPE_PUSH_DESCRIPTOR_SET_INFO_KHR,
      .stageFlags           = VK_SHADER_STAGE_COMPUTE_BIT,
      .layout               = m_pipelineLayout,
      .descriptorWriteCount = writeContainer.size(),
      .pDescriptorWrites    = writeContainer.data(),
  };
  vkCmdPushDescriptorSet2(cmd, &pushDescriptorSetInfo);

  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipeline);
  //vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineLayout, 0, 1, &m_descriptorSet, 0, nullptr);
  vkCmdPushConstants(cmd, m_pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PickInfo), &pickInfo);
  vkCmdDispatch(cmd, 1, 1, 1);  // one pixel

  // Wait for result
  VkBufferMemoryBarrier bmb{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
  bmb.srcAccessMask       = VK_ACCESS_MEMORY_WRITE_BIT;
  bmb.dstAccessMask       = VK_ACCESS_MEMORY_READ_BIT;
  bmb.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  bmb.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  bmb.buffer              = m_pickResult.buffer;
  bmb.size                = VK_WHOLE_SIZE;
  vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                       VK_DEPENDENCY_DEVICE_GROUP_BIT, 0, nullptr, 1, &bmb, 0, nullptr);
}

nvvk::RayPicker::PickResult nvvk::RayPicker::getResult() const
{
  PickResult pr{};
  memcpy(&pr, m_pickResult.mapping, sizeof(PickResult));
  return pr;
}

void nvvk::RayPicker::createOutputResult()
{
  PickResult presult{};
  NVVK_CHECK(m_alloc->createBuffer(m_pickResult, sizeof(PickResult),
                                   VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_AUTO,
                                   VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT));
  NVVK_DBG_NAME(m_pickResult.buffer);
}

void nvvk::RayPicker::createDescriptorSet()
{
  VkDevice device = m_alloc->getDevice();

  vkDestroyDescriptorSetLayout(device, m_descriptorSetLayout, nullptr);

  m_bindings.clear();
  m_bindings.addBinding(0, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1, VK_SHADER_STAGE_COMPUTE_BIT);
  m_bindings.addBinding(1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT);

  NVVK_CHECK(m_bindings.createDescriptorSetLayout(device, VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR, &m_descriptorSetLayout));
  NVVK_DBG_NAME(m_descriptorSetLayout);
}

void nvvk::RayPicker::createPipeline()
{
  VkDevice device = m_alloc->getDevice();

  vkDestroyPipeline(device, m_pipeline, nullptr);
  vkDestroyPipelineLayout(device, m_pipelineLayout, nullptr);

  const VkPushConstantRange pushConstant{.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT, .offset = 0, .size = sizeof(PickInfo)};
  NVVK_CHECK(nvvk::createPipelineLayout(device, &m_pipelineLayout, {m_descriptorSetLayout}, {pushConstant}));
  NVVK_DBG_NAME(m_pipelineLayout);

  std::span<const uint32_t> spirvData  = getSpirV();
  VkShaderModuleCreateInfo  moduleInfo = {
       .sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
       .codeSize = spirvData.size_bytes(),
       .pCode    = spirvData.data(),
  };

  VkPipelineShaderStageCreateInfo stageInfo{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
  stageInfo.pNext = &moduleInfo;
  stageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  stageInfo.pName = "main";

  VkComputePipelineCreateInfo computePipelineCreateInfo{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
  computePipelineCreateInfo.layout = m_pipelineLayout;
  computePipelineCreateInfo.stage  = stageInfo;
  vkCreateComputePipelines(device, {}, 1, &computePipelineCreateInfo, nullptr, &m_pipeline);
  NVVK_DBG_NAME(m_pipeline);
}


const std::span<const uint32_t> nvvk::RayPicker::getSpirV()
{
  // glslangValidator.exe --target-env vulkan1.2 --variable-name pick
  //const uint32_t pick[] =
  static std::vector<uint32_t> spirvData = {
      0x07230203, 0x00010500, 0x0008000a, 0x00000089, 0x00000000, 0x00020011, 0x00000001, 0x00020011, 0x00001178,
      0x0006000a, 0x5f565053, 0x5f52484b, 0x5f796172, 0x72657571, 0x00000079, 0x0006000b, 0x00000001, 0x4c534c47,
      0x6474732e, 0x3035342e, 0x00000000, 0x0003000e, 0x00000000, 0x00000001, 0x0008000f, 0x00000005, 0x00000004,
      0x6e69616d, 0x00000000, 0x0000000e, 0x00000047, 0x0000005f, 0x00060010, 0x00000004, 0x00000011, 0x00000001,
      0x00000001, 0x00000001, 0x00030003, 0x00000002, 0x000001cc, 0x00060004, 0x455f4c47, 0x725f5458, 0x715f7961,
      0x79726575, 0x00000000, 0x00040005, 0x00000004, 0x6e69616d, 0x00000000, 0x00050005, 0x00000009, 0x65786970,
      0x6e65436c, 0x00726574, 0x00050005, 0x0000000c, 0x736e6f43, 0x746e6174, 0x00000073, 0x00070006, 0x0000000c,
      0x00000000, 0x65646f6d, 0x6569566c, 0x766e4977, 0x00000000, 0x00070006, 0x0000000c, 0x00000001, 0x73726570,
      0x74636570, 0x49657669, 0x0000766e, 0x00050006, 0x0000000c, 0x00000002, 0x6b636970, 0x00000058, 0x00050006,
      0x0000000c, 0x00000003, 0x6b636970, 0x00000059, 0x00030005, 0x0000000e, 0x00000000, 0x00030005, 0x00000018,
      0x00000064, 0x00040005, 0x00000020, 0x6769726f, 0x00006e69, 0x00040005, 0x00000028, 0x67726174, 0x00007465,
      0x00050005, 0x00000036, 0x65726964, 0x6f697463, 0x0000006e, 0x00050005, 0x00000044, 0x51796172, 0x79726575,
      0x00000000, 0x00050005, 0x00000047, 0x4c706f74, 0x6c657665, 0x00005341, 0x00030005, 0x00000058, 0x00746968,
      0x00050005, 0x0000005c, 0x6b636950, 0x75736552, 0x0000746c, 0x00070006, 0x0000005c, 0x00000000, 0x6c726f77,
      0x79615264, 0x6769724f, 0x00006e69, 0x00080006, 0x0000005c, 0x00000001, 0x6c726f77, 0x79615264, 0x65726944,
      0x6f697463, 0x0000006e, 0x00050006, 0x0000005c, 0x00000002, 0x54746968, 0x00000000, 0x00060006, 0x0000005c,
      0x00000003, 0x6d697270, 0x76697469, 0x00444965, 0x00060006, 0x0000005c, 0x00000004, 0x74736e69, 0x65636e61,
      0x00004449, 0x00080006, 0x0000005c, 0x00000005, 0x74736e69, 0x65636e61, 0x74737543, 0x6e496d6f, 0x00786564,
      0x00060006, 0x0000005c, 0x00000006, 0x79726162, 0x726f6f43, 0x00000064, 0x00050005, 0x0000005d, 0x7365725f,
      0x50746c75, 0x006b6369, 0x00060006, 0x0000005d, 0x00000000, 0x75736572, 0x6950746c, 0x00006b63, 0x00030005,
      0x0000005f, 0x00000000, 0x00040005, 0x00000079, 0x79726162, 0x00000000, 0x00040048, 0x0000000c, 0x00000000,
      0x00000005, 0x00050048, 0x0000000c, 0x00000000, 0x00000023, 0x00000000, 0x00050048, 0x0000000c, 0x00000000,
      0x00000007, 0x00000010, 0x00040048, 0x0000000c, 0x00000001, 0x00000005, 0x00050048, 0x0000000c, 0x00000001,
      0x00000023, 0x00000040, 0x00050048, 0x0000000c, 0x00000001, 0x00000007, 0x00000010, 0x00050048, 0x0000000c,
      0x00000002, 0x00000023, 0x00000080, 0x00050048, 0x0000000c, 0x00000003, 0x00000023, 0x00000084, 0x00030047,
      0x0000000c, 0x00000002, 0x00040047, 0x00000047, 0x00000022, 0x00000000, 0x00040047, 0x00000047, 0x00000021,
      0x00000000, 0x00050048, 0x0000005c, 0x00000000, 0x00000023, 0x00000000, 0x00050048, 0x0000005c, 0x00000001,
      0x00000023, 0x00000010, 0x00050048, 0x0000005c, 0x00000002, 0x00000023, 0x00000020, 0x00050048, 0x0000005c,
      0x00000003, 0x00000023, 0x00000024, 0x00050048, 0x0000005c, 0x00000004, 0x00000023, 0x00000028, 0x00050048,
      0x0000005c, 0x00000005, 0x00000023, 0x0000002c, 0x00050048, 0x0000005c, 0x00000006, 0x00000023, 0x00000030,
      0x00050048, 0x0000005d, 0x00000000, 0x00000023, 0x00000000, 0x00030047, 0x0000005d, 0x00000002, 0x00040047,
      0x0000005f, 0x00000022, 0x00000000, 0x00040047, 0x0000005f, 0x00000021, 0x00000001, 0x00020013, 0x00000002,
      0x00030021, 0x00000003, 0x00000002, 0x00030016, 0x00000006, 0x00000020, 0x00040017, 0x00000007, 0x00000006,
      0x00000002, 0x00040020, 0x00000008, 0x00000007, 0x00000007, 0x00040017, 0x0000000a, 0x00000006, 0x00000004,
      0x00040018, 0x0000000b, 0x0000000a, 0x00000004, 0x0006001e, 0x0000000c, 0x0000000b, 0x0000000b, 0x00000006,
      0x00000006, 0x00040020, 0x0000000d, 0x00000009, 0x0000000c, 0x0004003b, 0x0000000d, 0x0000000e, 0x00000009,
      0x00040015, 0x0000000f, 0x00000020, 0x00000001, 0x0004002b, 0x0000000f, 0x00000010, 0x00000002, 0x00040020,
      0x00000011, 0x00000009, 0x00000006, 0x0004002b, 0x0000000f, 0x00000014, 0x00000003, 0x0004002b, 0x00000006,
      0x0000001a, 0x40000000, 0x0004002b, 0x00000006, 0x0000001c, 0x3f800000, 0x00040020, 0x0000001f, 0x00000007,
      0x0000000a, 0x0004002b, 0x0000000f, 0x00000021, 0x00000000, 0x00040020, 0x00000022, 0x00000009, 0x0000000b,
      0x0004002b, 0x00000006, 0x00000025, 0x00000000, 0x0007002c, 0x0000000a, 0x00000026, 0x00000025, 0x00000025,
      0x00000025, 0x0000001c, 0x0004002b, 0x0000000f, 0x00000029, 0x00000001, 0x00040015, 0x0000002c, 0x00000020,
      0x00000000, 0x0004002b, 0x0000002c, 0x0000002d, 0x00000000, 0x00040020, 0x0000002e, 0x00000007, 0x00000006,
      0x0004002b, 0x0000002c, 0x00000031, 0x00000001, 0x00040017, 0x00000039, 0x00000006, 0x00000003, 0x00021178,
      0x00000042, 0x00040020, 0x00000043, 0x00000007, 0x00000042, 0x000214dd, 0x00000045, 0x00040020, 0x00000046,
      0x00000000, 0x00000045, 0x0004003b, 0x00000046, 0x00000047, 0x00000000, 0x0004002b, 0x0000002c, 0x00000049,
      0x000000ff, 0x0004002b, 0x00000006, 0x0000004c, 0x3727c5ac, 0x0004002b, 0x00000006, 0x0000004f, 0x749dc5ae,
      0x00020014, 0x00000055, 0x00040020, 0x00000057, 0x00000007, 0x00000055, 0x00030029, 0x00000055, 0x00000059,
      0x0009001e, 0x0000005c, 0x0000000a, 0x0000000a, 0x00000006, 0x0000000f, 0x0000000f, 0x0000000f, 0x00000039,
      0x0003001e, 0x0000005d, 0x0000005c, 0x00040020, 0x0000005e, 0x0000000c, 0x0000005d, 0x0004003b, 0x0000005e,
      0x0000005f, 0x0000000c, 0x00040020, 0x00000061, 0x0000000c, 0x0000000a, 0x00040020, 0x00000066, 0x0000000c,
      0x00000006, 0x00040020, 0x00000069, 0x0000000c, 0x0000000f, 0x0004002b, 0x0000000f, 0x0000006b, 0x00000004,
      0x00040020, 0x0000006d, 0x00000007, 0x0000000f, 0x0004002b, 0x0000000f, 0x00000073, 0xffffffff, 0x0004002b,
      0x0000000f, 0x00000076, 0x00000005, 0x0004002b, 0x0000000f, 0x0000007b, 0x00000006, 0x00040020, 0x00000087,
      0x0000000c, 0x00000039, 0x00050036, 0x00000002, 0x00000004, 0x00000000, 0x00000003, 0x000200f8, 0x00000005,
      0x0004003b, 0x00000008, 0x00000009, 0x00000007, 0x0004003b, 0x00000008, 0x00000018, 0x00000007, 0x0004003b,
      0x0000001f, 0x00000020, 0x00000007, 0x0004003b, 0x0000001f, 0x00000028, 0x00000007, 0x0004003b, 0x0000001f,
      0x00000036, 0x00000007, 0x0004003b, 0x00000043, 0x00000044, 0x00000007, 0x0004003b, 0x00000057, 0x00000058,
      0x00000007, 0x0004003b, 0x0000006d, 0x0000006e, 0x00000007, 0x0004003b, 0x00000008, 0x00000079, 0x00000007,
      0x00050041, 0x00000011, 0x00000012, 0x0000000e, 0x00000010, 0x0004003d, 0x00000006, 0x00000013, 0x00000012,
      0x00050041, 0x00000011, 0x00000015, 0x0000000e, 0x00000014, 0x0004003d, 0x00000006, 0x00000016, 0x00000015,
      0x00050050, 0x00000007, 0x00000017, 0x00000013, 0x00000016, 0x0003003e, 0x00000009, 0x00000017, 0x0004003d,
      0x00000007, 0x00000019, 0x00000009, 0x0005008e, 0x00000007, 0x0000001b, 0x00000019, 0x0000001a, 0x00050050,
      0x00000007, 0x0000001d, 0x0000001c, 0x0000001c, 0x00050083, 0x00000007, 0x0000001e, 0x0000001b, 0x0000001d,
      0x0003003e, 0x00000018, 0x0000001e, 0x00050041, 0x00000022, 0x00000023, 0x0000000e, 0x00000021, 0x0004003d,
      0x0000000b, 0x00000024, 0x00000023, 0x00050091, 0x0000000a, 0x00000027, 0x00000024, 0x00000026, 0x0003003e,
      0x00000020, 0x00000027, 0x00050041, 0x00000022, 0x0000002a, 0x0000000e, 0x00000029, 0x0004003d, 0x0000000b,
      0x0000002b, 0x0000002a, 0x00050041, 0x0000002e, 0x0000002f, 0x00000018, 0x0000002d, 0x0004003d, 0x00000006,
      0x00000030, 0x0000002f, 0x00050041, 0x0000002e, 0x00000032, 0x00000018, 0x00000031, 0x0004003d, 0x00000006,
      0x00000033, 0x00000032, 0x00070050, 0x0000000a, 0x00000034, 0x00000030, 0x00000033, 0x0000001c, 0x0000001c,
      0x00050091, 0x0000000a, 0x00000035, 0x0000002b, 0x00000034, 0x0003003e, 0x00000028, 0x00000035, 0x00050041,
      0x00000022, 0x00000037, 0x0000000e, 0x00000021, 0x0004003d, 0x0000000b, 0x00000038, 0x00000037, 0x0004003d,
      0x0000000a, 0x0000003a, 0x00000028, 0x0008004f, 0x00000039, 0x0000003b, 0x0000003a, 0x0000003a, 0x00000000,
      0x00000001, 0x00000002, 0x0006000c, 0x00000039, 0x0000003c, 0x00000001, 0x00000045, 0x0000003b, 0x00050051,
      0x00000006, 0x0000003d, 0x0000003c, 0x00000000, 0x00050051, 0x00000006, 0x0000003e, 0x0000003c, 0x00000001,
      0x00050051, 0x00000006, 0x0000003f, 0x0000003c, 0x00000002, 0x00070050, 0x0000000a, 0x00000040, 0x0000003d,
      0x0000003e, 0x0000003f, 0x00000025, 0x00050091, 0x0000000a, 0x00000041, 0x00000038, 0x00000040, 0x0003003e,
      0x00000036, 0x00000041, 0x0004003d, 0x00000045, 0x00000048, 0x00000047, 0x0004003d, 0x0000000a, 0x0000004a,
      0x00000020, 0x0008004f, 0x00000039, 0x0000004b, 0x0000004a, 0x0000004a, 0x00000000, 0x00000001, 0x00000002,
      0x0004003d, 0x0000000a, 0x0000004d, 0x00000036, 0x0008004f, 0x00000039, 0x0000004e, 0x0000004d, 0x0000004d,
      0x00000000, 0x00000001, 0x00000002, 0x00091179, 0x00000044, 0x00000048, 0x0000002d, 0x00000049, 0x0000004b,
      0x0000004c, 0x0000004e, 0x0000004f, 0x000200f9, 0x00000050, 0x000200f8, 0x00000050, 0x000400f6, 0x00000052,
      0x00000053, 0x00000000, 0x000200f9, 0x00000054, 0x000200f8, 0x00000054, 0x0004117d, 0x00000055, 0x00000056,
      0x00000044, 0x000400fa, 0x00000056, 0x00000051, 0x00000052, 0x000200f8, 0x00000051, 0x0002117c, 0x00000044,
      0x000200f9, 0x00000053, 0x000200f8, 0x00000053, 0x000200f9, 0x00000050, 0x000200f8, 0x00000052, 0x0005117f,
      0x0000002c, 0x0000005a, 0x00000044, 0x00000029, 0x000500ab, 0x00000055, 0x0000005b, 0x0000005a, 0x0000002d,
      0x0003003e, 0x00000058, 0x0000005b, 0x0004003d, 0x0000000a, 0x00000060, 0x00000020, 0x00060041, 0x00000061,
      0x00000062, 0x0000005f, 0x00000021, 0x00000021, 0x0003003e, 0x00000062, 0x00000060, 0x0004003d, 0x0000000a,
      0x00000063, 0x00000036, 0x00060041, 0x00000061, 0x00000064, 0x0000005f, 0x00000021, 0x00000029, 0x0003003e,
      0x00000064, 0x00000063, 0x00051782, 0x00000006, 0x00000065, 0x00000044, 0x00000029, 0x00060041, 0x00000066,
      0x00000067, 0x0000005f, 0x00000021, 0x00000010, 0x0003003e, 0x00000067, 0x00000065, 0x00051787, 0x0000000f,
      0x00000068, 0x00000044, 0x00000029, 0x00060041, 0x00000069, 0x0000006a, 0x0000005f, 0x00000021, 0x00000014,
      0x0003003e, 0x0000006a, 0x00000068, 0x0004003d, 0x00000055, 0x0000006c, 0x00000058, 0x000300f7, 0x00000070,
      0x00000000, 0x000400fa, 0x0000006c, 0x0000006f, 0x00000072, 0x000200f8, 0x0000006f, 0x00051784, 0x0000000f,
      0x00000071, 0x00000044, 0x00000029, 0x0003003e, 0x0000006e, 0x00000071, 0x000200f9, 0x00000070, 0x000200f8,
      0x00000072, 0x0003003e, 0x0000006e, 0x00000073, 0x000200f9, 0x00000070, 0x000200f8, 0x00000070, 0x0004003d,
      0x0000000f, 0x00000074, 0x0000006e, 0x00060041, 0x00000069, 0x00000075, 0x0000005f, 0x00000021, 0x0000006b,
      0x0003003e, 0x00000075, 0x00000074, 0x00051783, 0x0000000f, 0x00000077, 0x00000044, 0x00000029, 0x00060041,
      0x00000069, 0x00000078, 0x0000005f, 0x00000021, 0x00000076, 0x0003003e, 0x00000078, 0x00000077, 0x00051788,
      0x00000007, 0x0000007a, 0x00000044, 0x00000029, 0x0003003e, 0x00000079, 0x0000007a, 0x00050041, 0x0000002e,
      0x0000007c, 0x00000079, 0x0000002d, 0x0004003d, 0x00000006, 0x0000007d, 0x0000007c, 0x00050083, 0x00000006,
      0x0000007e, 0x0000001c, 0x0000007d, 0x00050041, 0x0000002e, 0x0000007f, 0x00000079, 0x00000031, 0x0004003d,
      0x00000006, 0x00000080, 0x0000007f, 0x00050083, 0x00000006, 0x00000081, 0x0000007e, 0x00000080, 0x00050041,
      0x0000002e, 0x00000082, 0x00000079, 0x0000002d, 0x0004003d, 0x00000006, 0x00000083, 0x00000082, 0x00050041,
      0x0000002e, 0x00000084, 0x00000079, 0x00000031, 0x0004003d, 0x00000006, 0x00000085, 0x00000084, 0x00060050,
      0x00000039, 0x00000086, 0x00000081, 0x00000083, 0x00000085, 0x00060041, 0x00000087, 0x00000088, 0x0000005f,
      0x00000021, 0x0000007b, 0x0003003e, 0x00000088, 0x00000086, 0x000100fd, 0x00010038};

  return spirvData;
}

std::string nvvk::RayPicker::getGlsl()
{
  return R"(
	#version 460
	#extension GL_EXT_ray_query : require

	// clang-format off
	struct PickResult
	{
	  vec4  worldRayOrigin;
	  vec4  worldRayDirection;
	  float hitT;
	  int   primitiveID;
	  int   instanceID;
	  int   instanceCustomIndex;
	  vec3  baryCoord;
	};

	layout(set = 0, binding = 0) uniform accelerationStructureEXT topLevelAS;
	layout(set = 0, binding = 1) buffer _resultPick { PickResult resultPick; };
	layout(push_constant) uniform Constants
	{
	  mat4  modelViewInv;
	  mat4  perspectiveInv;
	  float pickX;  // normalized
	  float pickY;
	};

	void main()
	{
	  const vec2 pixelCenter = vec2(pickX, pickY);
	  vec2       d           = pixelCenter * 2.0 - 1.0;
	  vec4 origin            = modelViewInv * vec4(0, 0, 0, 1);
	  vec4 target            = perspectiveInv * vec4(d.x, d.y, 1, 1);
	  vec4 direction         = modelViewInv * vec4(normalize(target.xyz), 0);

	  rayQueryEXT rayQuery;
	  rayQueryInitializeEXT(rayQuery, topLevelAS, 0, 0xff, origin.xyz, 0.00001, direction.xyz, 1e32);
	  while(rayQueryProceedEXT(rayQuery)) {rayQueryConfirmIntersectionEXT(rayQuery); }

	  bool hit = (rayQueryGetIntersectionTypeEXT(rayQuery, true) != gl_RayQueryCommittedIntersectionNoneEXT);
	  resultPick.worldRayOrigin      = origin;
	  resultPick.worldRayDirection   = direction;
	  resultPick.hitT                = rayQueryGetIntersectionTEXT(rayQuery, true);
	  resultPick.primitiveID         = rayQueryGetIntersectionPrimitiveIndexEXT(rayQuery, true);
	  resultPick.instanceID          = hit ? rayQueryGetIntersectionInstanceIdEXT(rayQuery, true) : ~0;
	  resultPick.instanceCustomIndex = rayQueryGetIntersectionInstanceCustomIndexEXT(rayQuery, true);
	  vec2 bary                      = rayQueryGetIntersectionBarycentricsEXT(rayQuery, true);
	  resultPick.baryCoord           = vec3(1.0 - bary.x - bary.y, bary.x, bary.y);
	}
	// clang-format on
	)";
}


[[maybe_unused]] static void usage_RayPicker()
{
  nvvk::RayPicker rayPicker;
  rayPicker.init(nullptr);  // Pass a valid ResourceAllocator instance

  VkCommandBuffer cmd{};  //= m_app->createTempCmdBuffer();

  // Convert screen coordinates to normalized viewport coordinates [0,1] : #define IMGUI_DEFINE_MATH_OPERATORS
  // ImVec2 localMousePos = (ImGui::GetMousePos() - ImGui::GetCursorScreenPos()) / ImGui::GetContentRegionAvail();
  glm::vec2 localMousePos = {0.5f, 0.5f};  // Example: center of the screen

  rayPicker.run(cmd, {
                         .modelViewInv   = {},  //glm::inverse(g_cameraManip->getViewMatrix()),
                         .perspectiveInv = {},  //glm::inverse(g_cameraManip->getPerspectiveMatrix()),
                         .pickPos        = localMousePos,
                         .tlas           = {},  //m_sceneRtx.tlas()
                     });
  // m_app->submitAndWaitTempCmdBuffer(cmd);
  nvvk::RayPicker ::PickResult pickResult = rayPicker.getResult();
  if(pickResult.instanceID > -1)  // Hit something
  {
    // Set the camera CENTER to the hit position
    glm::vec3 worldPos = pickResult.worldRayOrigin + pickResult.worldRayDirection * pickResult.hitT;
    // glm::vec3 eye, center, up;
    // g_cameraManip->getLookat(eye, center, up);
    // g_cameraManip->setLookat(eye, worldPos, up, false);  // Nice with CameraManip.updateAnim();
  }

  rayPicker.deinit();
}
