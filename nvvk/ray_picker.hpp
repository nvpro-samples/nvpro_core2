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


#pragma once


/**
  # class nvvk::RayPicker

  nvvk::RayPicker is a utility to get hit information under a screen coordinate. 

  The information returned is: 
    - origin and direction in world space
    - hitT, the distance of the hit along the ray direction
    - primitiveID, instanceID and instanceCustomIndex
    - the barycentric coordinates in the triangle
  
  Getting results, for example, on mouse down:
  - fill the PickInfo structure
  - call run()
  - call getResult() to get all the information above

  See usage_RayPicker() for a complete example.

    ```
*/

#include <glm/glm.hpp>
#include "resource_allocator.hpp"
#include "descriptors.hpp"

namespace nvvk {

struct RayPicker
{
public:
  struct PickInfo
  {
    glm::mat4                  modelViewInv{1};    // inverse model view matrix
    glm::mat4                  perspectiveInv{1};  // inverse perspective matrix
    glm::vec2                  pickPos{0};         // normalized position
    VkAccelerationStructureKHR tlas{};             // top level acceleration structure
  };

  struct PickResult
  {
    glm::vec4 worldRayOrigin{0.f, 0.f, 0.f, 0.f};
    glm::vec4 worldRayDirection{0.f, 0.f, 0.f, 0.f};
    float     hitT{0.f};
    int       primitiveID{0};
    int       instanceID{-1};
    int       instanceCustomIndex{0};
    glm::vec3 baryCoord{0.f, 0.f, 0.f};
  };

  RayPicker() = default;
  ~RayPicker() { assert(!isValid()); }  // Missing deinit()

  void init(nvvk::ResourceAllocator* allocator);
  void deinit();

  // Set the top level acceleration structure
  void run(VkCommandBuffer cmd, const PickInfo& pickInfo);

  PickResult getResult() const;
  bool       isValid() const;

private:
  void                            createOutputResult();
  void                            createDescriptorSet();
  void                            createPipeline();
  const std::span<const uint32_t> getSpirV();
  std::string                     getGlsl();

  nvvk::Buffer             m_pickResult;
  nvvk::Buffer             m_sbtBuffer;
  nvvk::DescriptorBindings m_bindings;
  nvvk::ResourceAllocator* m_alloc{};

  VkDescriptorSetLayout m_descriptorSetLayout{};
  VkPipelineLayout      m_pipelineLayout{};
  VkPipeline            m_pipeline{};
};


}  // namespace nvvk
