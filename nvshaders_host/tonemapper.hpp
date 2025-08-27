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
#include <glm/glm.hpp>

#include "vulkan/vulkan_core.h"
#include "nvvk/resource_allocator.hpp"

#include <nvshaders/tonemap_io.h.slang>
#include <nvutils/timers.hpp>
#include <nvvk/descriptors.hpp>


namespace nvshaders {

class Tonemapper
{
public:
  Tonemapper() {};
  ~Tonemapper() { assert(m_device == VK_NULL_HANDLE); }  //  "Missing to call deinit"

  VkResult init(nvvk::ResourceAllocator* alloc, std::span<const uint32_t> spirv);
  void     deinit();

  void runCompute(VkCommandBuffer                 cmd,
                  const VkExtent2D&               size,
                  const shaderio::TonemapperData& tonemapper,
                  const VkDescriptorImageInfo&    inImage,
                  const VkDescriptorImageInfo&    outImage);

private:
  // Add new methods for histogram-based auto-exposure
  void runAutoExposureHistogram(VkCommandBuffer cmd, const VkExtent2D& size, const VkDescriptorImageInfo& inImage);
  void runAutoExposure(VkCommandBuffer cmd);
  void clearHistogram(VkCommandBuffer cmd);

  nvvk::ResourceAllocator* m_alloc{};

  VkDevice             m_device{};
  nvvk::DescriptorPack m_descriptorPack;
  VkPipelineLayout     m_pipelineLayout{};
  VkPipeline           m_tonemapPipeline{};
  VkPipeline           m_histogramPipeline{};
  VkPipeline           m_exposurePipeline{};

  nvutils::PerformanceTimer m_timer;  // Timer for performance measurement

  // Auto-Exposure
  nvvk::Buffer m_exposureBuffer;
  nvvk::Buffer m_histogramBuffer;
};


}  // namespace nvshaders