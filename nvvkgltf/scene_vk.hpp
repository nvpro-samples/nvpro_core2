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

#include <filesystem>
#include <set>

#include <nvvk/resource_allocator.hpp>

#include "scene.hpp"
#include "nvvk/sampler_pool.hpp"
#include "nvvk/staging.hpp"
#include "gpu_memory_tracker.hpp"


/*-------------------------------------------------------------------------------------------------
# class nvvkgltf::SceneVk

>  This class is responsible for the Vulkan version of the scene. 

It is using `nvvkgltf::Scene` to create the Vulkan buffers and images.

-------------------------------------------------------------------------------------------------*/

namespace nvvkgltf {
// Create the Vulkan version of the Scene
// Allocate the buffers, etc.
class SceneVk
{
public:
  // Those are potential buffers that can be created for vertices
  struct VertexBuffers
  {
    nvvk::Buffer position;
    nvvk::Buffer normal;
    nvvk::Buffer tangent;
    nvvk::Buffer texCoord0;
    nvvk::Buffer texCoord1;
    nvvk::Buffer color;
  };

  SceneVk() = default;
  virtual ~SceneVk() { assert(!m_alloc); }  // Missing deinit call

  void init(nvvk::ResourceAllocator* alloc, nvvk::SamplerPool* samplerPool);
  void deinit();

  virtual void create(VkCommandBuffer        cmd,
                      nvvk::StagingUploader& staging,
                      const nvvkgltf::Scene& scn,
                      bool                   generateMipmaps  = true,
                      bool                   enableRayTracing = true);

  void update(VkCommandBuffer cmd, nvvk::StagingUploader& staging, const nvvkgltf::Scene& scn);
  void updateRenderNodesBuffer(VkCommandBuffer cmd, nvvk::StagingUploader& staging, const nvvkgltf::Scene& scn);
  void updateRenderPrimitivesBuffer(VkCommandBuffer cmd, nvvk::StagingUploader& staging, const nvvkgltf::Scene& scn);
  void updateRenderLightsBuffer(VkCommandBuffer cmd, nvvk::StagingUploader& staging, const nvvkgltf::Scene& scn);
  void updateMaterialBuffer(VkCommandBuffer cmd, nvvk::StagingUploader& staging, const nvvkgltf::Scene& scn);
  void updateVertexBuffers(VkCommandBuffer cmd, nvvk::StagingUploader& staging, const nvvkgltf::Scene& scene);
  virtual void destroy();

  // Getters
  const nvvk::Buffer&               material() const { return m_bMaterial; }
  const nvvk::Buffer&               primInfo() const { return m_bRenderPrim; }
  const nvvk::Buffer&               instances() const { return m_bRenderNode; }
  const nvvk::Buffer&               sceneDesc() const { return m_bSceneDesc; }
  const std::vector<VertexBuffers>& vertexBuffers() const { return m_vertexBuffers; }
  const std::vector<nvvk::Buffer>&  indices() const { return m_bIndices; }
  const std::vector<nvvk::Image>&   textures() const { return m_textures; }
  uint32_t                          nbTextures() const { return static_cast<uint32_t>(m_textures.size()); }
  const GpuMemoryTracker&           getMemoryTracker() const { return m_memoryTracker; }
  GpuMemoryTracker&                 getMemoryTracker() { return m_memoryTracker; }

protected:
  struct SceneImage  // Image to be loaded and created
  {
    nvvk::Image imageTexture{};

    // Loading information
    bool                           srgb{false};
    std::string                    imgName{};
    VkExtent2D                     size{0, 0};
    VkFormat                       format{VK_FORMAT_UNDEFINED};
    std::vector<std::vector<char>> mipData{};
  };

  VkBufferUsageFlags2 getBufferUsageFlags() const;
  virtual void createVertexBuffers(VkCommandBuffer cmd, nvvk::StagingUploader& staging, const nvvkgltf::Scene& scn);
  template <typename T>
  bool         updateAttributeBuffer(VkCommandBuffer            cmd,
                                     const std::string&         attributeName,
                                     const tinygltf::Model&     model,
                                     const tinygltf::Primitive& primitive,
                                     nvvk::ResourceAllocator*   alloc,
                                     nvvk::StagingUploader*     staging,
                                     nvvk::Buffer&              attributeBuffer);
  virtual void createTextureImages(VkCommandBuffer              cmd,
                                   nvvk::StagingUploader&       staging,
                                   const tinygltf::Model&       model,
                                   const std::filesystem::path& basedir);

  void findSrgbImages(const tinygltf::Model& model);

  virtual void loadImage(const std::filesystem::path& basedir, const tinygltf::Image& gltfImage, int imageID);
  virtual bool createImage(const VkCommandBuffer& cmd, nvvk::StagingUploader& staging, SceneImage& image);

  //--
  VkDevice         m_device{VK_NULL_HANDLE};
  VkPhysicalDevice m_physicalDevice{VK_NULL_HANDLE};

  nvvk::ResourceAllocator* m_alloc       = nullptr;
  nvvk::SamplerPool*       m_samplerPool = nullptr;

  nvvk::Buffer               m_bMaterial;
  nvvk::Buffer               m_bTextureInfos;
  nvvk::Buffer               m_bLights;
  nvvk::Buffer               m_bRenderPrim;
  nvvk::Buffer               m_bRenderNode;
  nvvk::Buffer               m_bSceneDesc;
  std::vector<nvvk::Buffer>  m_bIndices;
  std::vector<VertexBuffers> m_vertexBuffers;
  std::vector<SceneImage>    m_images;
  std::vector<nvvk::Image>   m_textures;  // Vector of all textures of the scene

  std::set<int> m_sRgbImages;  // All images that are in sRGB (typically, only the one used by baseColorTexture)

  bool m_generateMipmaps   = {};
  bool m_rayTracingEnabled = {};

  GpuMemoryTracker m_memoryTracker;  // GPU memory tracking
};

}  // namespace nvvkgltf
