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


#include <nvvk/acceleration_structures.hpp>

#include "scene_vk.hpp"
/*-------------------------------------------------------------------------------------------------
# class nvvkgltf::SceneRtx

>  This class is responsible for the ray tracing acceleration structure. 

It is using the `nvvkgltf::Scene` and `nvvkgltf::SceneVk` information to create the acceleration structure.

 -------------------------------------------------------------------------------------------------*/
namespace nvvkgltf {

class SceneRtx
{
public:
  SceneRtx() = default;
  ~SceneRtx() { assert(m_blasAccel.empty()); }  // Missing deinit call

  void init(nvvk::ResourceAllocator* alloc);
  void deinit();


  // Create both bottom and top level acceleration structures (cannot compact)
  void create(VkCommandBuffer cmd, nvvk::StagingUploader& staging, const nvvkgltf::Scene& scn, const SceneVk& scnVk, VkBuildAccelerationStructureFlagsKHR flags);
  // Create the bottom level acceleration structure
  void createBottomLevelAccelerationStructure(const nvvkgltf::Scene& scene, const SceneVk& sceneVk, VkBuildAccelerationStructureFlagsKHR flags);
  // Build the bottom level acceleration structure
  bool cmdBuildBottomLevelAccelerationStructure(VkCommandBuffer cmd, VkDeviceSize hintMaxBudget = 512'000'000);

  // Create the top level acceleration structure
  void cmdCreateBuildTopLevelAccelerationStructure(VkCommandBuffer cmd, nvvk::StagingUploader& staging, const nvvkgltf::Scene& scene);
  // Compact the bottom level acceleration structure
  VkResult cmdCompactBlas(VkCommandBuffer cmd);
  // Destroy the original acceleration structures that was compacted
  void destroyNonCompactedBlas();
  // Update the instance buffer and build the TLAS (animation)
  void updateTopLevelAS(VkCommandBuffer cmd, nvvk::StagingUploader& staging, const nvvkgltf::Scene& scene);
  // Update the bottom level acceleration structure
  void updateBottomLevelAS(VkCommandBuffer cmd, const nvvkgltf::Scene& scene);

  // Return the constructed acceleration structure
  VkAccelerationStructureKHR tlas();

  // Destroy all acceleration structures
  void destroy();
  void destroyScratchBuffers();

protected:
  VkPhysicalDeviceRayTracingPipelinePropertiesKHR m_rtProperties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR};
  VkPhysicalDeviceAccelerationStructurePropertiesKHR m_rtASProperties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_PROPERTIES_KHR};

  nvvk::AccelerationStructureGeometryInfo renderPrimitiveToAsGeometry(const nvvkgltf::RenderPrimitive& prim,
                                                                      VkDeviceAddress                  vertexAddress,
                                                                      VkDeviceAddress                  indexAddress);

  VkDevice         m_device         = VK_NULL_HANDLE;
  VkPhysicalDevice m_physicalDevice = VK_NULL_HANDLE;

  nvvk::ResourceAllocator* m_alloc = nullptr;

  std::unique_ptr<nvvk::AccelerationStructureBuilder> m_blasBuilder;
  std::vector<nvvk::AccelerationStructureBuildData>   m_blasBuildData;
  std::vector<nvvk::AccelerationStructure>            m_blasAccel;

  nvvk::AccelerationStructureBuildData            m_tlasBuildData;
  nvvk::AccelerationStructure                     m_tlasAccel;
  std::vector<VkAccelerationStructureInstanceKHR> m_tlasInstances;

  nvvk::Buffer m_blasScratchBuffer;
  nvvk::Buffer m_tlasScratchBuffer;
  nvvk::Buffer m_instancesBuffer;

  uint32_t m_numVisibleElement = 0;
};

}  // namespace nvvkgltf
