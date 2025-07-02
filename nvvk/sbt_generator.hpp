/*
* Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
* SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
* SPDX-License-Identifier: Apache-2.0
*/

#pragma once

#include <array>
#include <unordered_map>

#include "resource_allocator.hpp"

namespace nvvk {

/*------------------------------------------------------------------------------------
#class nvvk::SBTGenerator

nvvk::SBTGenerator is a generic SBT builder from the ray tracing pipeline

The builder will iterate through the pipeline create info `VkRayTracingPipelineCreateInfoKHR`
to find the number of raygen, miss, hit and callable shader groups were created. 
The handles for those group will be retrieved from the pipeline and written in the right order in
separated buffer.

Convenient functions exist to retrieve all information to be used in TraceRayKHR.

## Usage
- Setup the builder (`init()`)
- After the pipeline creation, call `calculateSBTBufferSize()` which returns the size of the buffer to create
- Create the buffer then call `populateSBTBuffer()` to fill the buffer with the handles and data
- Use `getSBTRegions()` to get all the vk::StridedDeviceAddressRegionKHR needed by TraceRayKHR()


See under usage_SBTGenerator()
------------------------------------------------------------------------------------*/


class SBTGenerator
{
public:
  enum GroupType
  {
    eRaygen,
    eMiss,
    eHit,
    eCallable
  };

  // Return address regions of all groups.
  struct Regions
  {
    VkStridedDeviceAddressRegionKHR raygen{};
    VkStridedDeviceAddressRegionKHR miss{};
    VkStridedDeviceAddressRegionKHR hit{};
    VkStridedDeviceAddressRegionKHR callable{};
  };

  SBTGenerator() = default;
  ~SBTGenerator() { assert(m_device == VK_NULL_HANDLE); }  // To ensure deinit() was called

  void init(VkDevice device, const VkPhysicalDeviceRayTracingPipelinePropertiesKHR& rayProperties);
  void deinit();

  // Analyzes the ray tracing pipeline to determine the required SBT buffer size
  // Returns the size needed for the SBT buffer
  size_t calculateSBTBufferSize(VkPipeline                                         rayPipeline,
                                VkRayTracingPipelineCreateInfoKHR                  rayPipelineInfo = {},
                                std::span<const VkRayTracingPipelineCreateInfoKHR> librariesInfo   = {});

  // Populates the SBT buffer with shader handles and data
  // The bufferAddress must be aligned to getBufferAlignment();
  // The buffer should be created with VK_BUFFER_USAGE_2_SHADER_BINDING_TABLE_BIT_KHR | VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT
  VkResult populateSBTBuffer(VkDeviceAddress bufferAddress, size_t bufferSize, void* bufferData);


  // After updateBuffer one can retrieve the regions
  // Return the address region of a group. indexOffset allow to offset the starting shader of the group.
  const VkStridedDeviceAddressRegionKHR getSBTRegion(GroupType t, uint32_t indexOffset = 0) const;

  // Return the address regions of all groups. The offset allows to select which RayGen to use.
  const Regions getSBTRegions(uint32_t rayGenIndexOffset = 0) const;


  // Manually adds shader group indices from pipeline create info
  // Used when you want to specify indices directly instead of using calculateSBTBufferSize
  // The rayPipelineInfo parameter defines the pipeline, while librariesInfo describes potential input pipeline libraries
  void addIndices(VkRayTracingPipelineCreateInfoKHR                         rayPipelineInfo,
                  const std::span<const VkRayTracingPipelineCreateInfoKHR>& libraries = {});

  // Pushing back a GroupType and the handle pipeline index to use
  // i.e addIndex(eHit, 3) is pushing a Hit shader group using the 3rd entry in the pipeline
  void addIndex(GroupType t, uint32_t index) { m_shaderGroupIndices[t].push_back(index); }

  // Adding 'Shader Record' data to the group index.
  // i.e. addData(eHit, 0, myValue) is adding 'myValue' to the HIT group 0.
  template <typename T>
  void addData(GroupType t, uint32_t groupIndex, T& data)
  {
    addData(t, groupIndex, (uint8_t*)&data, sizeof(T));
  }

  void addData(GroupType t, uint32_t groupIndex, uint8_t* data, size_t dataSize)
  {
    m_data[t][groupIndex].assign(data, data + dataSize);
  }

  // Get buffer alignment
  uint32_t getBufferAlignment() const;

  // Resets internals, can start adding things freshly again
  void reset();

  // Reset state prior buffer updates only
  void resetBuffer();

private:
  // Getters
  uint32_t getGroupIndexCount(GroupType t) const { return static_cast<uint32_t>(m_shaderGroupIndices[t].size()); }
  uint32_t getGroupStride(GroupType t) const { return m_stride[t]; }
  VkDeviceAddress getGroupAddress(GroupType t) const;

  // returns the entire size of a group. Raygen Stride and Size must be equal, even if the buffer contains many of them.
  uint32_t getSize(GroupType t) const
  {
    return t == eRaygen ? getGroupStride(eRaygen) : getGroupStride(t) * getGroupIndexCount(t);
  }

  using shaderRecordMap = std::unordered_map<uint32_t, std::vector<uint8_t>>;

  std::array<std::vector<uint32_t>, 4> m_shaderGroupIndices;  // For each group type, stores the pipeline indices of shader groups
  std::array<VkDeviceAddress, 4> m_bufferAddresses{};   // The addresses of each group
  std::array<uint32_t, 4>        m_stride{0, 0, 0, 0};  // Stride of each group
  std::array<shaderRecordMap, 4> m_data;                // Local data to groups (Shader Record)

  uint32_t m_handleSize{0};
  uint32_t m_handleAlignment{0};
  uint32_t m_shaderGroupBaseAlignment{0};

  uint32_t   m_totalGroupCount{0};
  size_t     m_dataSize{0};
  VkPipeline m_pipeline{VK_NULL_HANDLE};

  VkDevice m_device{VK_NULL_HANDLE};
};
}  // namespace nvvk
