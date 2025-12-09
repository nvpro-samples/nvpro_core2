/*
* Copyright (c) 2021-2025, NVIDIA CORPORATION.  All rights reserved.
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
* SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
* SPDX-License-Identifier: Apache-2.0
*/

#include <algorithm>

#include <nvutils/alignment.hpp>

#include "debug_util.hpp"
#include "check_error.hpp"
#include "helpers.hpp"
#include "sbt_generator.hpp"
#include "staging.hpp"


namespace nvvk {
//--------------------------------------------------------------------------------------------------
// Default setup
//
void SBTGenerator::init(VkDevice device, const VkPhysicalDeviceRayTracingPipelinePropertiesKHR& rayProperties)
{
  assert(m_device == nullptr);

  m_device                   = device;
  m_handleSize               = rayProperties.shaderGroupHandleSize;       // Size of a program identifier
  m_handleAlignment          = rayProperties.shaderGroupHandleAlignment;  // Alignment in bytes for each SBT entry
  m_shaderGroupBaseAlignment = rayProperties.shaderGroupBaseAlignment;
}

//--------------------------------------------------------------------------------------------------
// Destroying the allocated buffers and clearing all vectors
//
void SBTGenerator::deinit()
{
  reset();
  m_device = {};
}

void SBTGenerator::reset()
{
  m_data               = {};
  m_shaderGroupIndices = {};
  m_stride             = {};
  m_bufferAddresses    = {};
  m_dataSize           = 0;
  m_totalGroupCount    = 0;
  m_pipeline           = 0;
}

void SBTGenerator::resetBuffer()
{
  m_bufferAddresses = {};
}

uint32_t SBTGenerator::getBufferAlignment() const
{
  return std::max(m_shaderGroupBaseAlignment, m_handleAlignment);
}

//--------------------------------------------------------------------------------------------------
// Finding the handle index position of each group type in the pipeline creation info.
// If the pipeline was created like: raygen, miss, hit, miss, hit, hit
// The result will be: raygen[0], miss[1, 3], hit[2, 4, 5], callable[]
//
void SBTGenerator::addIndices(VkRayTracingPipelineCreateInfoKHR                         rayPipelineInfo,
                              const std::span<const VkRayTracingPipelineCreateInfoKHR>& libraries)
{
  // Clear all shader group indices before adding new ones
  for(std::vector<uint32_t>& groupIndices : m_shaderGroupIndices)
    groupIndices = {};

  // Libraries contain stages referencing their internal groups. When those groups
  // are used in the final pipeline we need to offset them to ensure each group has
  // a unique index
  uint32_t groupOffset = 0;

  for(size_t pipelineIndex = 0; pipelineIndex < libraries.size() + 1; pipelineIndex++)
  {
    // When using libraries, their groups and stages are appended after the groups and
    // stages defined in the main VkRayTracingPipelineCreateInfoKHR
    const VkRayTracingPipelineCreateInfoKHR& info = (pipelineIndex == 0) ? rayPipelineInfo : libraries[pipelineIndex - 1];

    // Finding the handle position of each group, splitting by raygen, miss and hit group
    for(uint32_t g = 0; g < info.groupCount; g++)
    {
      // Check if the group is a general shader group (raygen, miss, or callable)
      if(info.pGroups[g].type == VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR)
      {
        uint32_t genShader = info.pGroups[g].generalShader;
        assert(genShader < info.stageCount);
        // Classify the group by shader stage type
        if(info.pStages[genShader].stage == VK_SHADER_STAGE_RAYGEN_BIT_KHR)
        {
          m_shaderGroupIndices[eRaygen].push_back(g + groupOffset);
        }
        else if(info.pStages[genShader].stage == VK_SHADER_STAGE_MISS_BIT_KHR)
        {
          m_shaderGroupIndices[eMiss].push_back(g + groupOffset);
        }
        else if(info.pStages[genShader].stage == VK_SHADER_STAGE_CALLABLE_BIT_KHR)
        {
          m_shaderGroupIndices[eCallable].push_back(g + groupOffset);
        }
      }
      else
      {
        // Otherwise, it's a hit group
        m_shaderGroupIndices[eHit].push_back(g + groupOffset);
      }
    }

    groupOffset += info.groupCount;  // Offset for next library's groups
  }
}

//--------------------------------------------------------------------------------------------------
// Calculates the required size for the Shader Binding Table (SBT) buffer for a ray tracing pipeline.
// Analyzes the pipeline and libraries to determine group counts, strides, and alignment.
// Updates internal state for SBT layout and returns the total buffer size needed.
// Call after pipeline creation; use result to allocate the SBT buffer.
//
size_t SBTGenerator::calculateSBTBufferSize(VkPipeline                                         rayPipeline,
                                            VkRayTracingPipelineCreateInfoKHR                  rayPipelineInfo,
                                            std::span<const VkRayTracingPipelineCreateInfoKHR> librariesInfo)
{
  // Reset group count and store the pipeline handle
  m_totalGroupCount = 0;
  m_pipeline        = rayPipeline;

  if(rayPipelineInfo.sType == VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR)
  {
    // Populate group indices for all shader group types (raygen, miss, hit, callable)
    addIndices(rayPipelineInfo, librariesInfo);
    // Count all groups in the main pipeline and libraries
    m_totalGroupCount += rayPipelineInfo.groupCount;
    for(const VkRayTracingPipelineCreateInfoKHR& lib : librariesInfo)
    {
      m_totalGroupCount += lib.groupCount;
    }
  }
  else
  {
    // If not using pipeline create info, determine group count from manually added indices
    // Find the largest index in all group types and add 1 (since indices are zero-based)
    for(const std::vector<uint32_t>& groupIndices : m_shaderGroupIndices)
    {
      if(!groupIndices.empty())
        m_totalGroupCount = std::max(m_totalGroupCount, *std::max_element(std::begin(groupIndices), std::end(groupIndices)));
    }
    m_totalGroupCount++;
  }

  // Lambda to compute the stride for each group type
  // Stride is the maximum of (handle size + data size), aligned to handle alignment
  auto findStride = [&](shaderRecordMap& entries, uint32_t& stride) {
    stride = nvutils::align_up(m_handleSize, m_handleAlignment);  // minimum stride is handle size aligned
    for(const std::pair<uint32_t, std::vector<uint8_t>>& entry : entries)
    {
      // Compute size for handle + any attached data, aligned
      uint32_t dataHandleSize =
          nvutils::align_up(static_cast<uint32_t>(m_handleSize + entry.second.size() * sizeof(uint8_t)), m_handleAlignment);
      stride = std::max(stride, dataHandleSize);  // Use the largest stride needed
    }
  };
  // Calculate stride for each group type
  findStride(m_data[eRaygen], m_stride[eRaygen]);
  findStride(m_data[eMiss], m_stride[eMiss]);
  findStride(m_data[eHit], m_stride[eHit]);
  findStride(m_data[eCallable], m_stride[eCallable]);

  // Special case: Raygen group stride must be aligned to shaderGroupBaseAlignment
  m_stride[eRaygen] = nvutils::align_up(m_stride[eRaygen], m_shaderGroupBaseAlignment);

  size_t   totalSize       = 0;
  uint32_t bufferAlignment = getBufferAlignment();

  // Compute buffer offsets for each group type and accumulate total buffer size
  // Each group section is aligned to bufferAlignment
  m_bufferAddresses[eRaygen] = totalSize;
  totalSize += nvutils::align_up(m_stride[eRaygen] * getGroupIndexCount(eRaygen), bufferAlignment);
  m_bufferAddresses[eMiss] = totalSize;
  totalSize += nvutils::align_up(m_stride[eMiss] * getGroupIndexCount(eMiss), bufferAlignment);
  m_bufferAddresses[eHit] = totalSize;
  totalSize += nvutils::align_up(m_stride[eHit] * getGroupIndexCount(eHit), bufferAlignment);
  m_bufferAddresses[eCallable] = totalSize;
  totalSize += nvutils::align_up(m_stride[eCallable] * getGroupIndexCount(eCallable), bufferAlignment);

  // Store the final computed buffer size
  m_dataSize = totalSize;

  // Return the total size required for the SBT buffer
  return totalSize;
}

//--------------------------------------------------------------------------------------------------
// Fills the SBT buffer with shader group handles and any attached data for the ray tracing pipeline.
// Requires a buffer of the correct size and alignment, as calculated by calculateSBTBufferSize().
// The buffer must be created with appropriate usage flags for SBT and device address access.
//
VkResult SBTGenerator::populateSBTBuffer(VkDeviceAddress bufferAddress, size_t bufferSize, void* bufferData)
{
  assert(m_pipeline && "Missing updatePipeline()");
  assert(bufferSize == m_dataSize);
  assert(m_bufferAddresses[eRaygen] == 0 && "must not call updateBuffer multiple times");
  assert(bufferAddress % getBufferAlignment() == 0);

  uint8_t* dataBytes = static_cast<uint8_t*>(bufferData);

  // Fetch all the shader handles used in the pipeline, so that they can be written in the SBT
  uint32_t             sbtSize = m_totalGroupCount * m_handleSize;
  std::vector<uint8_t> shaderHandleStorage(sbtSize);

  // Get the shader handles for all groups in the pipeline
  NVVK_FAIL_RETURN(vkGetRayTracingShaderGroupHandlesKHR(m_device, m_pipeline, 0, m_totalGroupCount, sbtSize,
                                                        shaderHandleStorage.data()));

  // Write the handles in the SBT buffer + data info (if any)
  auto copyHandles = [&](VkDeviceAddress offset, std::vector<uint32_t>& indices, uint32_t stride, shaderRecordMap& data) {
    uint8_t* pBuffer = dataBytes + offset;
    for(uint32_t index = 0; index < static_cast<uint32_t>(indices.size()); index++)
    {
      uint8_t* pStart = pBuffer;
      // Copy the handle for this group
      memcpy(pBuffer, shaderHandleStorage.data() + (indices[index] * m_handleSize), m_handleSize);
      // If there is data for this group index, copy it too
      auto recordIt = data.find(index);
      if(recordIt != data.end())
      {
        pBuffer += m_handleSize;  // Move pointer past handle
        memcpy(pBuffer, recordIt->second.data(), recordIt->second.size() * sizeof(uint8_t));
      }
      pBuffer = pStart + stride;  // Jumping to next group
    }
  };

  // Copy the handles/data to each staging buffer for each group type
  copyHandles(m_bufferAddresses[eRaygen], m_shaderGroupIndices[eRaygen], m_stride[eRaygen], m_data[eRaygen]);
  copyHandles(m_bufferAddresses[eMiss], m_shaderGroupIndices[eMiss], m_stride[eMiss], m_data[eMiss]);
  copyHandles(m_bufferAddresses[eHit], m_shaderGroupIndices[eHit], m_stride[eHit], m_data[eHit]);
  copyHandles(m_bufferAddresses[eCallable], m_shaderGroupIndices[eCallable], m_stride[eCallable], m_data[eCallable]);

  // update the addresses from offsets to full address
  m_bufferAddresses[eRaygen] += bufferAddress;
  m_bufferAddresses[eMiss] += bufferAddress;
  m_bufferAddresses[eHit] += bufferAddress;
  m_bufferAddresses[eCallable] += bufferAddress;

  return VK_SUCCESS;
}

VkDeviceAddress SBTGenerator::getGroupAddress(GroupType t) const
{
  assert(m_bufferAddresses[t]);
  return m_bufferAddresses[t];
}

const VkStridedDeviceAddressRegionKHR SBTGenerator::getSBTRegion(GroupType t, uint32_t indexOffset) const
{
  return VkStridedDeviceAddressRegionKHR{getGroupAddress(t) + indexOffset * getGroupStride(t), getGroupStride(t), getSize(t)};
}

const SBTGenerator::Regions SBTGenerator::getSBTRegions(uint32_t rayGenIndexOffset) const
{
  Regions regions{.raygen   = getSBTRegion(eRaygen, rayGenIndexOffset),
                  .miss     = getSBTRegion(eMiss),
                  .hit      = getSBTRegion(eHit),
                  .callable = getSBTRegion(eCallable)};
  return regions;
}

//--------------------------------------------------------------------------------------------------
// Usage of SBTGenerator
// This is a simple example of how to use the SBTGenerator class.
[[maybe_unused]] static void usage_SBTGenerator()
{
  VkPhysicalDevice        physicalDevice = VK_NULL_HANDLE;
  VkDevice                device         = VK_NULL_HANDLE;
  nvvk::ResourceAllocator allocator;

  VkPhysicalDeviceRayTracingPipelinePropertiesKHR rtProp{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR};
  VkPhysicalDeviceProperties2 prop2{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, .pNext = &rtProp};
  vkGetPhysicalDeviceProperties2(physicalDevice, &prop2);

  // Create the raytracing pipeline
  VkPipeline                        rtPipeline = VK_NULL_HANDLE;
  VkRayTracingPipelineCreateInfoKHR rayPipelineInfo{.sType = VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR};
  NVVK_CHECK(vkCreateRayTracingPipelinesKHR(device, {}, {}, 1, &rayPipelineInfo, nullptr, &rtPipeline));


  // Shader Binding Table (SBT) setup
  nvvk::Buffer       sbtBuffer;
  nvvk::SBTGenerator sbtGenerator;
  sbtGenerator.init(device, rtProp);

  // Prepare SBT data from ray pipeline
  size_t bufferSize = sbtGenerator.calculateSBTBufferSize(rtPipeline, rayPipelineInfo);

  // Create SBT buffer using the size from above
  NVVK_CHECK(allocator.createBuffer(sbtBuffer, bufferSize, VK_BUFFER_USAGE_2_SHADER_BINDING_TABLE_BIT_KHR, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
                                    VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT,
                                    sbtGenerator.getBufferAlignment()));

  // Pass the manual mapped pointer to fill the SBT data
  NVVK_CHECK(sbtGenerator.populateSBTBuffer(sbtBuffer.address, bufferSize, sbtBuffer.mapping));

  // Retrieve the regions, which are using addresses based on the m_sbtBuffer.address
  const SBTGenerator::Regions sbtRegions = sbtGenerator.getSBTRegions();

  // Clean up
  sbtGenerator.deinit();


  // Raytrace
  VkCommandBuffer   cmd  = {};
  const VkExtent2D& size = {};
  vkCmdTraceRaysKHR(cmd, &sbtRegions.raygen, &sbtRegions.miss, &sbtRegions.hit, &sbtRegions.callable, size.width, size.height, 1);

  //------------------------------------------------------------------------------------------------------------------
  // Extra: Adding data to the SBT
  // The SBT can have data attached to each group. This is done by calling addData() for each group.

  struct HitRecordBuffer
  {
    std::array<float, 4> color;
  };
  std::vector<HitRecordBuffer> m_hitShaderRecord = {{{0.0f, 1.0f, 0.0f, 0.0f}}, {{1.0f, 0.0f, 0.0f, 0.0f}}};

  sbtGenerator.addData(SBTGenerator::eHit, 1, m_hitShaderRecord[0]);
  sbtGenerator.addData(SBTGenerator::eHit, 2, m_hitShaderRecord[1]);


  // ##Special case
  // The SBT can be created with only a few groups, but the pipeline can have many more groups.
  // It is also possible to create a pipeline with only a few groups but having a SBT representing many more groups

  // The following example shows a more complex setup.There are : 1 x raygen,
  // 2 x miss, 2 x hit.BUT the SBT will have 3 hit by duplicating the second hit in its table.So,
  // the same hit shader defined in the pipeline,
  // can be called with different data.
  //
  // In this case,
  // the use must provide manually the information to the SBT.All extra group must be explicitly added.
  //
  // The following show how to get handle indices provided in the pipeline,
  // and we are adding another hit group, re - using the 4th pipeline entry.Note:
  // we are not providing the pipelineCreateInfo,
  // because we are manually defining it.

  // Manually defining group indices
  sbtGenerator.addIndices(rayPipelineInfo);  // Add raygen(0), miss(1), miss(2), hit(3), hit(4) from the pipeline info
  sbtGenerator.addIndex(SBTGenerator::eHit, 4);  // Adding a 3rd hit, duplicate from the hit:1, which make hit:2 available.
  sbtGenerator.addData(SBTGenerator::eHit, 2, m_hitShaderRecord[1]);  // Adding data to this hit shader
  sbtGenerator.calculateSBTBufferSize(rtPipeline);
}


}  // namespace nvvk
