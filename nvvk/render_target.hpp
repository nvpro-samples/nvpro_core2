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

#pragma once

/***************************************************************
 ***************************************************************
 *****************     RENDER TARGET      ******************
 ***************************************************************
 ***************************************************************/

#include <string>
#include <vector>

#include <vulkan/vulkan_core.h>
#include "nvvk/resource_allocator.hpp"

namespace nvvk {

//
// RenderTargetState - transient dynamic-rendering state filled from a RenderTarget.
//
struct RenderTargetState
{
  VkRect2D              renderArea{};
  uint32_t              layerCount{1};
  VkSampleCountFlagBits sampleCount{VK_SAMPLE_COUNT_1_BIT};

  uint32_t                               colorCount{};
  std::vector<VkFormat>                  colorFormats;
  VkFormat                               depthFormat{VK_FORMAT_UNDEFINED};
  VkFormat                               stencilFormat{VK_FORMAT_UNDEFINED};
  std::vector<VkRenderingAttachmentInfo> colorAttachments;
  VkRenderingAttachmentInfo              depthAttachment{};
  VkRenderingAttachmentInfo              stencilAttachment{};

  RenderTargetState();

  void clear();

  void fillPipelineRenderingCreateInfo(VkPipelineRenderingCreateInfo& renderingInfo) const;

  struct AttachmentOps
  {
    VkAttachmentLoadOp  colorLoad    = VK_ATTACHMENT_LOAD_OP_CLEAR;
    VkAttachmentStoreOp colorStore   = VK_ATTACHMENT_STORE_OP_STORE;
    VkAttachmentLoadOp  depthLoad    = VK_ATTACHMENT_LOAD_OP_CLEAR;
    VkAttachmentStoreOp depthStore   = VK_ATTACHMENT_STORE_OP_STORE;
    VkAttachmentLoadOp  stencilLoad  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    VkAttachmentStoreOp stencilStore = VK_ATTACHMENT_STORE_OP_DONT_CARE;
  };
  void cmdBeginRendering(VkCommandBuffer cmd, const AttachmentOps& ops);
};

/***************************************************************
 ***************************************************************
 ********************    RenderTarget      ********************
 ***************************************************************
 ***************************************************************/

//  RenderTarget - Offscreen color (one or more) + optional depth/stencil.
//
//  Common configurations:
//  * Default (vk_mini_samples): layouts=GENERAL, default usages, LOAD_OP_CLEAR via RenderTargetState::cmdBeginRendering().
//  * Raster legacy: layouts.color/depth = *_ATTACHMENT_OPTIMAL, no STORAGE; caller transitions for sampling.
//  * Compute MRT: layouts=GENERAL, multiple colorFormats, storage via getColorStorageImageInfo().
//
//  Lifecycle: init() -> update(cmd, extent) on resize -> fillState() + cmdBeginRendering().
//  update() does not clear images; use cmdClear() or LOAD_OP_CLEAR in AttachmentOps.
//  Display: getUiImageView() (alpha=1 for ImGui). Shader read: getSampleView() / getColorSampleDescriptorImageInfo().
//  MSAA: getColorImage() is the multisampled attachment; getColorImageResolved() is the resolve target (null if
//  samples==1); getSampleImage() / getSampleView() pick resolve when it exists, else the attachment.
class RenderTarget
{
public:
  struct LayoutConfig
  {
    VkImageLayout color{VK_IMAGE_LAYOUT_GENERAL};
    VkImageLayout depth{VK_IMAGE_LAYOUT_GENERAL};
    // VK_IMAGE_LAYOUT_UNDEFINED: GENERAL when color is GENERAL, else SHADER_READ_ONLY_OPTIMAL.
    VkImageLayout colorSample{VK_IMAGE_LAYOUT_UNDEFINED};
  };

  struct MsaaConfig
  {
    // All resolve modes default to VK_RESOLVE_MODE_NONE (opt-in per aspect).
    VkSampleCountFlagBits samples{VK_SAMPLE_COUNT_1_BIT};
    VkResolveModeFlagBits colorResolve{};    // required when samples > 1 (e.g. VK_RESOLVE_MODE_AVERAGE_BIT)
    VkResolveModeFlagBits depthResolve{};    // e.g. VK_RESOLVE_MODE_SAMPLE_ZERO_BIT
    VkResolveModeFlagBits stencilResolve{};  // must not be AVERAGE (integer); typically VK_RESOLVE_MODE_SAMPLE_ZERO_BIT
  };

  struct CreateInfo
  {
    VkDevice                 device{};
    nvvk::ResourceAllocator* alloc{};
    std::vector<VkFormat>    colorFormats;
    VkFormat                 depthFormat{VK_FORMAT_UNDEFINED};
    VkFormat                 stencilFormat{VK_FORMAT_UNDEFINED};
    uint32_t                 layerCount{1};
    VkImageUsageFlags        colorUsage{0};  // 0 = default (+STORAGE per format when layouts are GENERAL)
    VkImageUsageFlags        depthUsage{0};
    MsaaConfig               msaa{};
    LayoutConfig             layouts{};
    std::string              debugName{};
  };

  RenderTarget() = default;
  ~RenderTarget();

  RenderTarget(const RenderTarget&)            = delete;
  RenderTarget& operator=(const RenderTarget&) = delete;
  RenderTarget(RenderTarget&&)                 = delete;
  RenderTarget& operator=(RenderTarget&&)      = delete;

  VkResult init(const CreateInfo& info);
  void     deinit();

  // Takes effect on the next update(). Prefer setMsaa() when changing sample count and resolve together.
  VkResult setMsaa(const MsaaConfig& msaa);
  VkResult setSampleCount(VkSampleCountFlagBits sampleCount,
                          VkResolveModeFlagBits colorResolveMode = VK_RESOLVE_MODE_NONE,
                          VkResolveModeFlagBits depthResolveMode = VK_RESOLVE_MODE_NONE);

  VkResult update(VkCommandBuffer cmd, VkExtent2D extent);  // no-op when extent and config unchanged; recreates on resize or setMsaa()

  void cmdClear(VkCommandBuffer          cmd,
                VkClearColorValue        colorClear = {{0.F, 0.F, 0.F, 0.F}},
                VkClearDepthStencilValue depthClear = {1.F, 0U});

  // fillState() options:
  //   depthOnly=false, addResolve=true  — full color pass with MSAA resolve (default).
  //   depthOnly=true                    — depth/stencil only (e.g. shadow pre-pass).
  //   addResolve=false                  — intermediate MSAA pass; resolve on the final pass only.
  void fillState(RenderTargetState& state, bool depthOnly = false, bool addResolve = true) const;

  VkExtent2D getSize() const;

  // --- Color: render attachment (multisampled when MSAA) vs resolve / readback ---
  VkImage     getColorImage(uint32_t i = 0) const;  // render target image (MSAA surface when msaa.samples > 1)
  VkImageView getColorAttachmentView(uint32_t i = 0) const;

  VkImage     getColorImageResolved(uint32_t i = 0) const;  // VK_NULL_HANDLE when no resolve image (samples==1, etc.)
  VkImageView getResolveImageView(uint32_t i = 0) const;

  // Single-sample result for sampling / save / copy: resolve image if MSAA, else same as getColorImage().
  VkImage     getSampleImage(uint32_t i = 0) const;
  VkImageView getSampleView(uint32_t i = 0) const;

  // Alpha=1 swizzled view for nvapp::ImTexture / ImGui::Image.
  VkImageView getUiImageView(uint32_t i = 0) const;

  // Storage write (compute / RT output) vs shader sample read (tonemap, blit, etc.).
  VkDescriptorImageInfo getColorStorageImageInfo(uint32_t i = 0) const;
  VkDescriptorImageInfo getColorSampleDescriptorImageInfo(uint32_t i = 0, VkSampler sampler = VK_NULL_HANDLE) const;

  VkImageLayout getColorLayout() const;
  VkImageLayout getDepthLayout() const;
  VkImageLayout getColorSampleLayout() const;

  // --- Depth ---
  VkImage     getDepthImage() const;
  VkImageView getDepthImageView() const;
  VkImage     getDepthImageResolved() const;
  VkImageView getDepthResolveView() const;

  VkFormat              getColorFormat(uint32_t i = 0) const;
  VkFormat              getDepthFormat() const;
  VkSampleCountFlagBits getSampleCount() const;
  uint32_t              getLayerCount() const;
  float                 getAspectRatio() const;

  // VMA allocations for memory tracking (GpuMemoryTracker, etc.).
  VmaAllocation getColorAllocation(uint32_t i = 0) const;
  VmaAllocation getColorAllocationResolved(uint32_t i = 0) const;
  VmaAllocation getDepthAllocation() const;
  VmaAllocation getDepthAllocationResolved() const;

private:
  struct ColorUsageResult
  {
    VkImageUsageFlags renderUsage{};
    VkImageUsageFlags resolveUsage{};
  };

  static VkResult          validateCreateInfo(const CreateInfo& info);
  static VkImageLayout     effectiveColorSampleLayout(const CreateInfo& info);
  static VkImageUsageFlags defaultColorUsage();
  static VkImageUsageFlags defaultDepthUsage();
  static VkImageViewType   attachmentViewType(uint32_t layerCount);
  static uint32_t          attachmentViewLayers(uint32_t layerCount);
  static VkImageSubresourceRange colorLayersRange(uint32_t layerCount, uint32_t baseArrayLayer = 0, uint32_t layers = 0);
  static VkImageSubresourceRange depthLayersRange(VkImageAspectFlags aspect,
                                                  uint32_t           layerCount,
                                                  uint32_t           baseArrayLayer = 0,
                                                  uint32_t           layers         = 0);

  static VkResult resolveColorUsage(const CreateInfo& info, VkFormat format, VkPhysicalDevice physicalDevice, ColorUsageResult& out);

  // Single source of truth for the depth/stencil aspect mask and the MSAA resolve decisions.
  VkImageAspectFlags depthStencilAspect() const;
  bool               isMsaa() const;
  bool               wantsColorResolve() const;
  bool               wantsDepthResolve() const;
  bool               wantsStencilResolve() const;
  bool wantsDepthStencilResolve() const;  // true if either depth or stencil resolves (shared resolve image)

  VkResult createColorAttachment(uint32_t i, VkExtent2D extent, bool resolveColor, const std::string& namePrefix, VkPhysicalDevice physicalDevice);
  VkResult createDepthAttachment(VkExtent2D extent, bool resolveDepth, const std::string& namePrefix, VkPhysicalDevice physicalDevice);
  VkResult transitionToWorkingLayouts(VkCommandBuffer cmd);

  VkResult createResources(VkCommandBuffer cmd, VkExtent2D extent);
  void     destroyResources();

  CreateInfo m_createInfo{};
  VkExtent2D m_extent{};
  bool       m_resourcesDirty{true};  // set by init/setMsaa; cleared after successful createResources()

  std::vector<nvvk::Image> m_colorImages{};
  nvvk::Image              m_depthImage{};
  std::vector<nvvk::Image> m_colorImagesResolved{};
  nvvk::Image              m_depthImageResolved{};

  std::vector<VkImageView> m_imageViews;
  std::vector<VkImageView> m_uiImageViews;
  std::vector<VkImageView> m_resolveImageViews;
  VkImageView              m_depthView{};
  VkImageView              m_depthResolveView{};
};

}  // namespace nvvk
