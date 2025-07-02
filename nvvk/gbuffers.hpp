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
#include <vector>

#include "resource_allocator.hpp"

namespace nvvk {

//--- GBuffer ------------------------------------------------------------------------------------------------------------

// Usage:
//   see usage_GBuffer in gbuffers.cpp
//

/*--
 * GBuffer creation info
-*/
struct GBufferInitInfo
{
  nvvk::ResourceAllocator* allocator{};     // Allocator for the images
  std::vector<VkFormat>    colorFormats{};  // Array of formats for each color attachment (as many GBuffers as formats)
  VkFormat                 depthFormat{VK_FORMAT_UNDEFINED};    // Depth buffer (VK_FORMAT_UNDEFINED for no depth)
  VkSampleCountFlagBits    sampleCount{VK_SAMPLE_COUNT_1_BIT};  // MSAA sample count (default: no MSAA)
  VkSampler                imageSampler{};                      // Linear sampler for displaying the images (ImGui)
  VkDescriptorPool         descriptorPool{};                    // Pool for the ImGui descriptors
};

/*--
 * GBuffer - Multiple render targets with depth management
 * 
 * This class manages multiple color buffers and a depth buffer for deferred rendering or 
 * other multi-target rendering techniques. It supports:
 * - Multiple color attachments with configurable formats
 * - Optional depth buffer
 * - MSAA
 * - ImGui integration for debug visualization
 * - Automatic resource cleanup
 *
 * The GBuffer images can be used as:
 * - Color/Depth attachments (write)
 * - Texture sampling (read)
 * - Storage images (read/write)
 * - Transfer operations
-*/
class GBuffer
{
public:
  GBuffer()                          = default;
  GBuffer(const GBuffer&)            = delete;   // Prevent copying
  GBuffer& operator=(const GBuffer&) = delete;   // Prevent assignment
  GBuffer(GBuffer&& other) noexcept;             // Allow moving
  GBuffer& operator=(GBuffer&& other) noexcept;  // Move assignment
  ~GBuffer();

  // Initialize the GBuffer with the specified configuration.
  void init(const GBufferInitInfo& createInfo);

  // Destroy internal resources and reset its initial state
  void deinit();

  // Set or reset the size of the G-Buffers
  VkResult update(VkCommandBuffer cmd, VkExtent2D newSize);


  //--- Getters for the GBuffer resources -------------------------
  VkDescriptorSet              getDescriptorSet(uint32_t i = 0) const;  // Can be use as ImTextureID for ImGui
  VkExtent2D                   getSize() const;
  VkImage                      getColorImage(uint32_t i = 0) const;
  VkImage                      getDepthImage() const;
  VkImageView                  getColorImageView(uint32_t i = 0) const;
  const VkDescriptorImageInfo& getDescriptorImageInfo(uint32_t i = 0) const;
  VkImageView                  getDepthImageView() const;
  VkFormat                     getColorFormat(uint32_t i = 0) const;
  VkFormat                     getDepthFormat() const;
  VkSampleCountFlagBits        getSampleCount() const;
  float                        getAspectRatio() const;

private:
  /*--
   * Create the GBuffer with the specified configuration
   *
   * Each color buffer is created with:
   * - Color attachment usage     : For rendering
   * - Sampled bit                : For sampling in shaders
   * - Storage bit                : For compute shader access
   * - Transfer dst bit           : For clearing/copying
   * 
   * The depth buffer is created with:
   * - Depth/Stencil attachment   : For depth testing
   * - Sampled bit                : For sampling in shaders
   *
   * All images are transitioned to GENERAL layout and cleared to black.
   * ImGui descriptors are created for debug visualization.
  -*/
  VkResult initResources(VkCommandBuffer cmd);

  /*--
   * Clean up all Vulkan resources
   * - Images and image views
   * - Samplers
   * - ImGui descriptors
   * 
   * This must be called before destroying the GBuffer or when
   * recreating with different parameters
  -*/
  void deinitResources();


  // Resources holds all Vulkan objects for the GBuffer
  // This separation makes it easier to cleanup and recreate resources
  struct Resources
  {
    std::vector<nvvk::Image>     gBufferColor{};      // Color attachments
    nvvk::Image                  gBufferDepth{};      // Optional depth attachment
    std::vector<VkImageView>     uiImageViews{};      // Special views for ImGui (alpha=1)
    std::vector<VkDescriptorSet> uiDescriptorSets{};  // ImGui descriptor sets
  } m_res;                                            // All Vulkan resources

  VkExtent2D m_size{};  // Width and height of the buffers

  GBufferInitInfo       m_info{};        // Configuration
  VkDescriptorSetLayout m_descLayout{};  // Layout for the ImGui descriptors
};


}  // namespace nvvk
