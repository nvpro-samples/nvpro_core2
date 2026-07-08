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


#include "render_target.hpp"

#include "barriers.hpp"
#include "check_error.hpp"
#include "debug_util.hpp"
#include "volk/volk.h"

#include <nvutils/logger.hpp>

namespace {

void cmdSubmitBarriers(VkCommandBuffer cmd, std::vector<VkImageMemoryBarrier2>& barriers)
{
  if(barriers.empty())
    return;
  const VkDependencyInfo depInfo{.sType                   = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
                                 .imageMemoryBarrierCount = uint32_t(barriers.size()),
                                 .pImageMemoryBarriers    = barriers.data()};
  vkCmdPipelineBarrier2(cmd, &depInfo);
  barriers.clear();
}

void cmdClearColorImage(VkCommandBuffer                     cmd,
                        VkImage                             image,
                        VkImageLayout                       layout,
                        const VkImageSubresourceRange&      range,
                        const VkClearColorValue&            clearValue,
                        std::vector<VkImageMemoryBarrier2>& barriers)
{
  if(image == VK_NULL_HANDLE)
    return;
  if(layout == VK_IMAGE_LAYOUT_GENERAL)
  {
    vkCmdClearColorImage(cmd, image, VK_IMAGE_LAYOUT_GENERAL, &clearValue, 1, &range);
    return;
  }

  barriers.push_back(nvvk::makeImageMemoryBarrier(
      {.image = image, .oldLayout = layout, .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, .subresourceRange = range}));
  cmdSubmitBarriers(cmd, barriers);
  vkCmdClearColorImage(cmd, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &clearValue, 1, &range);
  barriers.push_back(nvvk::makeImageMemoryBarrier(
      {.image = image, .oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, .newLayout = layout, .subresourceRange = range}));
  cmdSubmitBarriers(cmd, barriers);
}

void cmdClearDepthStencilImage(VkCommandBuffer                     cmd,
                               VkImage                             image,
                               VkImageLayout                       layout,
                               const VkImageSubresourceRange&      range,
                               const VkClearDepthStencilValue&     clearValue,
                               std::vector<VkImageMemoryBarrier2>& barriers)
{
  if(image == VK_NULL_HANDLE)
    return;
  if(layout == VK_IMAGE_LAYOUT_GENERAL)
  {
    vkCmdClearDepthStencilImage(cmd, image, VK_IMAGE_LAYOUT_GENERAL, &clearValue, 1, &range);
    return;
  }

  barriers.push_back(nvvk::makeImageMemoryBarrier(
      {.image = image, .oldLayout = layout, .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, .subresourceRange = range}));
  cmdSubmitBarriers(cmd, barriers);
  vkCmdClearDepthStencilImage(cmd, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &clearValue, 1, &range);
  barriers.push_back(nvvk::makeImageMemoryBarrier(
      {.image = image, .oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, .newLayout = layout, .subresourceRange = range}));
  cmdSubmitBarriers(cmd, barriers);
}

}  // namespace

////////////////////////////////////////////////////////////
// RENDER TARGET STATE
////////////////////////////////////////////////////////////

nvvk::RenderTargetState::RenderTargetState()
{
  clear();
}

void nvvk::RenderTargetState::clear()
{
  colorCount    = 0;
  depthFormat   = VK_FORMAT_UNDEFINED;
  stencilFormat = VK_FORMAT_UNDEFINED;
  layerCount    = 1;
  sampleCount   = VK_SAMPLE_COUNT_1_BIT;
  colorFormats.clear();
  colorAttachments.clear();
  depthAttachment   = {.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
  stencilAttachment = {.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
}

void nvvk::RenderTargetState::fillPipelineRenderingCreateInfo(VkPipelineRenderingCreateInfo& renderingInfo) const
{
  assert((colorCount == 0 || !colorFormats.empty()) && "Missing color format");
  renderingInfo = {
      .sType                   = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
      .colorAttachmentCount    = colorCount,
      .pColorAttachmentFormats = colorFormats.data(),
      .depthAttachmentFormat   = depthFormat,
      .stencilAttachmentFormat = stencilFormat,
  };
}

void nvvk::RenderTargetState::cmdBeginRendering(VkCommandBuffer cmd, const AttachmentOps& ops)
{
  assert((colorCount <= colorAttachments.size()) && "colorAttachments out of sync with colorCount");
  for(uint32_t i = 0; i < colorCount; ++i)
  {
    colorAttachments[i].loadOp  = ops.colorLoad;
    colorAttachments[i].storeOp = ops.colorStore;
  }
  depthAttachment.loadOp    = ops.depthLoad;
  depthAttachment.storeOp   = ops.depthStore;
  stencilAttachment.loadOp  = ops.stencilLoad;
  stencilAttachment.storeOp = ops.stencilStore;

  const VkRenderingInfo renderingInfo{
      .sType                = VK_STRUCTURE_TYPE_RENDERING_INFO,
      .renderArea           = renderArea,
      .layerCount           = layerCount,
      .colorAttachmentCount = colorCount,
      .pColorAttachments    = colorCount ? colorAttachments.data() : nullptr,
      .pDepthAttachment     = depthFormat != VK_FORMAT_UNDEFINED ? &depthAttachment : nullptr,
      .pStencilAttachment   = stencilFormat != VK_FORMAT_UNDEFINED ? &stencilAttachment : nullptr,
  };
  vkCmdBeginRendering(cmd, &renderingInfo);
}

////////////////////////////////////////////////////////////
// RENDER TARGET
////////////////////////////////////////////////////////////

nvvk::RenderTarget::~RenderTarget()
{
  assert((m_createInfo.alloc == nullptr) && "Missing deinit()");
}

VkResult nvvk::RenderTarget::init(const CreateInfo& info)
{
  if(m_createInfo.alloc != nullptr)
  {
    LOGE("nvvk::RenderTarget: double init()\n");
    return VK_ERROR_INITIALIZATION_FAILED;
  }

  NVVK_FAIL_RETURN(validateCreateInfo(info));
  m_createInfo     = info;
  m_resourcesDirty = true;
  if(m_createInfo.device == VK_NULL_HANDLE && m_createInfo.alloc != nullptr)
    m_createInfo.device = m_createInfo.alloc->getDevice();
  return VK_SUCCESS;
}

void nvvk::RenderTarget::deinit()
{
  if(m_createInfo.alloc == nullptr)
    return;
  destroyResources();
  m_createInfo = {};
}

VkResult nvvk::RenderTarget::setMsaa(const MsaaConfig& msaa)
{
  CreateInfo info = m_createInfo;
  info.msaa       = msaa;
  NVVK_FAIL_RETURN(validateCreateInfo(info));
  m_createInfo.msaa = msaa;
  m_resourcesDirty  = true;
  return VK_SUCCESS;
}

VkResult nvvk::RenderTarget::setSampleCount(VkSampleCountFlagBits sampleCount, VkResolveModeFlagBits colorResolveMode, VkResolveModeFlagBits depthResolveMode)
{
  return setMsaa({.samples = sampleCount, .colorResolve = colorResolveMode, .depthResolve = depthResolveMode});
}

VkResult nvvk::RenderTarget::update(VkCommandBuffer cmd, VkExtent2D extent)
{
  if(m_createInfo.alloc == nullptr)
  {
    LOGE("nvvk::RenderTarget: call init() before update()\n");
    return VK_ERROR_INITIALIZATION_FAILED;
  }
  if(extent.width == 0 || extent.height == 0)
  {
    LOGE("nvvk::RenderTarget: update() extent must be non-zero\n");
    return VK_ERROR_INITIALIZATION_FAILED;
  }
  if(extent.width == m_extent.width && extent.height == m_extent.height && !m_resourcesDirty)
    return VK_SUCCESS;

  destroyResources();
  const VkResult result = createResources(cmd, extent);
  if(result < VK_SUCCESS)
    destroyResources();
  return result;
}

void nvvk::RenderTarget::fillState(RenderTargetState& state, bool depthOnly, bool addResolve) const
{
  assert((!depthOnly || m_createInfo.depthFormat != VK_FORMAT_UNDEFINED || m_createInfo.stencilFormat != VK_FORMAT_UNDEFINED)
         && "depthOnly requires a depth or stencil format");

  state.clear();

  const uint32_t colorCount = depthOnly ? 0 : static_cast<uint32_t>(m_createInfo.colorFormats.size());
  state.colorCount          = colorCount;
  state.depthFormat         = m_createInfo.depthFormat;
  state.stencilFormat       = m_createInfo.stencilFormat;
  state.sampleCount         = m_createInfo.msaa.samples;
  state.layerCount          = m_createInfo.layerCount;
  state.colorFormats        = m_createInfo.colorFormats;
  state.colorAttachments.resize(colorCount, {.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO});
  state.renderArea = {.offset{}, .extent = m_extent};

  const bool resolveColor   = addResolve && wantsColorResolve();
  const bool resolveDepth   = addResolve && wantsDepthResolve();
  const bool resolveStencil = addResolve && wantsStencilResolve();

  for(uint32_t i = 0; i < colorCount; ++i)
  {
    if(m_createInfo.colorFormats[i] == VK_FORMAT_UNDEFINED || i >= m_colorImages.size() || m_colorImages[i].image == VK_NULL_HANDLE)
      continue;

    state.colorAttachments[i]             = {.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
    state.colorAttachments[i].imageView   = m_imageViews[i];
    state.colorAttachments[i].imageLayout = m_createInfo.layouts.color;

    if(resolveColor && i < m_colorImagesResolved.size() && m_colorImagesResolved[i].image != VK_NULL_HANDLE)
    {
      state.colorAttachments[i].resolveMode        = m_createInfo.msaa.colorResolve;
      state.colorAttachments[i].resolveImageView   = m_resolveImageViews[i];
      state.colorAttachments[i].resolveImageLayout = m_createInfo.layouts.color;
    }
  }

  if(m_createInfo.depthFormat != VK_FORMAT_UNDEFINED && m_depthImage.image != VK_NULL_HANDLE)
  {
    state.depthAttachment             = {.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
    state.depthAttachment.imageView   = m_depthView;
    state.depthAttachment.imageLayout = m_createInfo.layouts.depth;

    if(resolveDepth && m_depthImageResolved.image != VK_NULL_HANDLE)
    {
      state.depthAttachment.resolveMode        = m_createInfo.msaa.depthResolve;
      state.depthAttachment.resolveImageView   = m_depthResolveView;
      state.depthAttachment.resolveImageLayout = m_createInfo.layouts.depth;
    }
  }

  if(m_createInfo.stencilFormat != VK_FORMAT_UNDEFINED && m_depthImage.image != VK_NULL_HANDLE)
  {
    state.stencilAttachment             = {.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
    state.stencilAttachment.imageView   = m_depthView;
    state.stencilAttachment.imageLayout = m_createInfo.layouts.depth;

    if(resolveStencil && m_depthImageResolved.image != VK_NULL_HANDLE)
    {
      state.stencilAttachment.resolveMode        = m_createInfo.msaa.stencilResolve;
      state.stencilAttachment.resolveImageView   = m_depthResolveView;
      state.stencilAttachment.resolveImageLayout = m_createInfo.layouts.depth;
    }
  }
}

VkExtent2D nvvk::RenderTarget::getSize() const
{
  return m_extent;
}

VkImage nvvk::RenderTarget::getColorImage(uint32_t i) const
{
  assert(i < m_colorImages.size() && m_colorImages[i].image != VK_NULL_HANDLE);
  return m_colorImages[i].image;
}

VkImageView nvvk::RenderTarget::getColorAttachmentView(uint32_t i) const
{
  assert(i < m_imageViews.size() && m_imageViews[i] != VK_NULL_HANDLE);
  return m_imageViews[i];
}

VkImageView nvvk::RenderTarget::getSampleView(uint32_t i) const
{
  const VkImageView resolved = getResolveImageView(i);
  return resolved != VK_NULL_HANDLE ? resolved : getColorAttachmentView(i);
}

VkImageView nvvk::RenderTarget::getUiImageView(uint32_t i) const
{
  assert(i < m_uiImageViews.size() && m_uiImageViews[i] != VK_NULL_HANDLE);
  return m_uiImageViews[i];
}

VkImage nvvk::RenderTarget::getSampleImage(uint32_t i) const
{
  const VkImage resolved = getColorImageResolved(i);
  return resolved != VK_NULL_HANDLE ? resolved : getColorImage(i);
}

VkDescriptorImageInfo nvvk::RenderTarget::getColorStorageImageInfo(uint32_t i) const
{
  const VkImageView resolveView = getResolveImageView(i);
  if(resolveView != VK_NULL_HANDLE)
    return {.sampler = VK_NULL_HANDLE, .imageView = resolveView, .imageLayout = m_createInfo.layouts.color};
  return {.sampler = VK_NULL_HANDLE, .imageView = getColorAttachmentView(i), .imageLayout = m_createInfo.layouts.color};
}

VkDescriptorImageInfo nvvk::RenderTarget::getColorSampleDescriptorImageInfo(uint32_t i, VkSampler sampler) const
{
  return {.sampler = sampler, .imageView = getSampleView(i), .imageLayout = effectiveColorSampleLayout(m_createInfo)};
}

VkImageLayout nvvk::RenderTarget::getColorLayout() const
{
  return m_createInfo.layouts.color;
}

VkImageLayout nvvk::RenderTarget::getDepthLayout() const
{
  return m_createInfo.layouts.depth;
}

VkImageLayout nvvk::RenderTarget::getColorSampleLayout() const
{
  return effectiveColorSampleLayout(m_createInfo);
}

VkImage nvvk::RenderTarget::getColorImageResolved(uint32_t i) const
{
  return i < m_colorImagesResolved.size() ? m_colorImagesResolved[i].image : VK_NULL_HANDLE;
}

VkImageView nvvk::RenderTarget::getResolveImageView(uint32_t i) const
{
  return i < m_resolveImageViews.size() ? m_resolveImageViews[i] : VK_NULL_HANDLE;
}

VkImage nvvk::RenderTarget::getDepthImage() const
{
  return m_depthImage.image;
}

VkImageView nvvk::RenderTarget::getDepthImageView() const
{
  return m_depthView;
}

VkImage nvvk::RenderTarget::getDepthImageResolved() const
{
  return m_depthImageResolved.image;
}

VkImageView nvvk::RenderTarget::getDepthResolveView() const
{
  return m_depthResolveView;
}

VkFormat nvvk::RenderTarget::getColorFormat(uint32_t i) const
{
  return m_createInfo.colorFormats[i];
}

VkFormat nvvk::RenderTarget::getDepthFormat() const
{
  return m_createInfo.depthFormat;
}

VkSampleCountFlagBits nvvk::RenderTarget::getSampleCount() const
{
  return m_createInfo.msaa.samples;
}

uint32_t nvvk::RenderTarget::getLayerCount() const
{
  return m_createInfo.layerCount;
}

float nvvk::RenderTarget::getAspectRatio() const
{
  return m_extent.height > 0 ? float(m_extent.width) / float(m_extent.height) : 0.f;
}

VmaAllocation nvvk::RenderTarget::getColorAllocation(uint32_t i) const
{
  return i < m_colorImages.size() ? m_colorImages[i].allocation : nullptr;
}

VmaAllocation nvvk::RenderTarget::getColorAllocationResolved(uint32_t i) const
{
  return i < m_colorImagesResolved.size() ? m_colorImagesResolved[i].allocation : nullptr;
}

VmaAllocation nvvk::RenderTarget::getDepthAllocation() const
{
  return m_depthImage.image != VK_NULL_HANDLE ? m_depthImage.allocation : nullptr;
}

VmaAllocation nvvk::RenderTarget::getDepthAllocationResolved() const
{
  return m_depthImageResolved.image != VK_NULL_HANDLE ? m_depthImageResolved.allocation : nullptr;
}

void nvvk::RenderTarget::cmdClear(VkCommandBuffer cmd, VkClearColorValue colorClear, VkClearDepthStencilValue depthClear)
{
  assert((m_createInfo.alloc != nullptr) && "Call init() and update() before cmdClear()");

  std::vector<VkImageMemoryBarrier2> barriers;
  const VkImageSubresourceRange      colorRange = colorLayersRange(m_createInfo.layerCount);

  for(const nvvk::Image& color : m_colorImages)
    cmdClearColorImage(cmd, color.image, m_createInfo.layouts.color, colorRange, colorClear, barriers);
  for(const nvvk::Image& resolved : m_colorImagesResolved)
    cmdClearColorImage(cmd, resolved.image, m_createInfo.layouts.color, colorRange, colorClear, barriers);

  if(m_depthImage.image != VK_NULL_HANDLE)
  {
    const VkImageAspectFlags      depthAspect = depthStencilAspect();
    const VkImageSubresourceRange depthRange  = {depthAspect, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS};
    cmdClearDepthStencilImage(cmd, m_depthImage.image, m_createInfo.layouts.depth, depthRange, depthClear, barriers);
    if(m_depthImageResolved.image != VK_NULL_HANDLE)
      cmdClearDepthStencilImage(cmd, m_depthImageResolved.image, m_createInfo.layouts.depth, depthRange, depthClear, barriers);
  }
}

////////////////////////////////////////////////////////////
// INTERNAL
////////////////////////////////////////////////////////////

VkResult nvvk::RenderTarget::validateCreateInfo(const CreateInfo& info)
{
  if(info.alloc == nullptr)
  {
    LOGE("nvvk::RenderTarget: CreateInfo::alloc is required\n");
    return VK_ERROR_INITIALIZATION_FAILED;
  }

  if(info.colorFormats.empty() && info.depthFormat == VK_FORMAT_UNDEFINED && info.stencilFormat == VK_FORMAT_UNDEFINED)
  {
    LOGE("nvvk::RenderTarget: at least one color, depth, or stencil format is required\n");
    return VK_ERROR_INITIALIZATION_FAILED;
  }

  if(info.layerCount < 1)
  {
    LOGE("nvvk::RenderTarget: layerCount must be at least 1\n");
    return VK_ERROR_INITIALIZATION_FAILED;
  }

  if(info.depthFormat != VK_FORMAT_UNDEFINED && info.stencilFormat != VK_FORMAT_UNDEFINED && info.depthFormat != info.stencilFormat)
  {
    LOGE("nvvk::RenderTarget: depthFormat and stencilFormat must match when both are set (use a combined depth/stencil format)\n");
    return VK_ERROR_INITIALIZATION_FAILED;
  }

  if(info.msaa.samples != VK_SAMPLE_COUNT_1_BIT && !info.colorFormats.empty() && info.msaa.colorResolve == VK_RESOLVE_MODE_NONE)
  {
    LOGE("nvvk::RenderTarget: MSAA color targets require msaa.colorResolve (e.g. VK_RESOLVE_MODE_AVERAGE_BIT)\n");
    return VK_ERROR_INITIALIZATION_FAILED;
  }

  if(info.msaa.stencilResolve == VK_RESOLVE_MODE_AVERAGE_BIT)
  {
    LOGE("nvvk::RenderTarget: msaa.stencilResolve must not be VK_RESOLVE_MODE_AVERAGE_BIT (invalid for integer stencil)\n");
    return VK_ERROR_INITIALIZATION_FAILED;
  }

  // Validate depth/stencil resolve modes against device capabilities (VkPhysicalDeviceDepthStencilResolveProperties).
  if(info.msaa.samples != VK_SAMPLE_COUNT_1_BIT && (info.depthFormat != VK_FORMAT_UNDEFINED || info.stencilFormat != VK_FORMAT_UNDEFINED))
  {
    VkPhysicalDeviceDepthStencilResolveProperties resolveProps{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DEPTH_STENCIL_RESOLVE_PROPERTIES};
    VkPhysicalDeviceProperties2 props2{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, .pNext = &resolveProps};
    vkGetPhysicalDeviceProperties2(info.alloc->getPhysicalDevice(), &props2);

    const bool                  hasDepth    = info.depthFormat != VK_FORMAT_UNDEFINED;
    const bool                  hasStencil  = info.stencilFormat != VK_FORMAT_UNDEFINED;
    const VkResolveModeFlagBits depthMode   = hasDepth ? info.msaa.depthResolve : VK_RESOLVE_MODE_NONE;
    const VkResolveModeFlagBits stencilMode = hasStencil ? info.msaa.stencilResolve : VK_RESOLVE_MODE_NONE;

    if(depthMode != VK_RESOLVE_MODE_NONE && (resolveProps.supportedDepthResolveModes & depthMode) == 0)
    {
      LOGE("nvvk::RenderTarget: msaa.depthResolve (0x%x) not in supportedDepthResolveModes (0x%x)\n", depthMode,
           resolveProps.supportedDepthResolveModes);
      return VK_ERROR_INITIALIZATION_FAILED;
    }
    if(stencilMode != VK_RESOLVE_MODE_NONE && (resolveProps.supportedStencilResolveModes & stencilMode) == 0)
    {
      LOGE("nvvk::RenderTarget: msaa.stencilResolve (0x%x) not in supportedStencilResolveModes (0x%x)\n", stencilMode,
           resolveProps.supportedStencilResolveModes);
      return VK_ERROR_INITIALIZATION_FAILED;
    }
    // Combined depth/stencil: both modes non-NONE and differing requires independentResolve.
    if(hasDepth && hasStencil && depthMode != VK_RESOLVE_MODE_NONE && stencilMode != VK_RESOLVE_MODE_NONE
       && depthMode != stencilMode && resolveProps.independentResolve == VK_FALSE)
    {
      LOGE("nvvk::RenderTarget: differing depth/stencil resolve modes require independentResolve (device lacks it)\n");
      return VK_ERROR_INITIALIZATION_FAILED;
    }
    // Combined depth/stencil: exactly one mode NONE requires independentResolveNone.
    if(hasDepth && hasStencil && (depthMode == VK_RESOLVE_MODE_NONE) != (stencilMode == VK_RESOLVE_MODE_NONE)
       && resolveProps.independentResolveNone == VK_FALSE)
    {
      LOGE("nvvk::RenderTarget: resolving only one of depth/stencil requires independentResolveNone (device lacks it)\n");
      return VK_ERROR_INITIALIZATION_FAILED;
    }
  }

  const bool userDepthUsage = (info.depthUsage != 0);
  if(info.layouts.depth != VK_IMAGE_LAYOUT_GENERAL && userDepthUsage && (info.depthUsage & VK_IMAGE_USAGE_STORAGE_BIT) != 0)
  {
    LOGE("nvvk::RenderTarget: STORAGE in depthUsage requires GENERAL layouts.depth\n");
    return VK_ERROR_INITIALIZATION_FAILED;
  }

  const auto isColorRenderLayout = [](VkImageLayout layout) {
    return layout == VK_IMAGE_LAYOUT_GENERAL || layout == VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
  };
  const auto isDepthLayout = [](VkImageLayout layout) {
    return layout == VK_IMAGE_LAYOUT_GENERAL || layout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
  };
  const auto isColorSampleLayout = [](VkImageLayout layout) {
    return layout == VK_IMAGE_LAYOUT_UNDEFINED || layout == VK_IMAGE_LAYOUT_GENERAL || layout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
  };

  if(!isColorRenderLayout(info.layouts.color))
  {
    LOGE("nvvk::RenderTarget: unsupported layouts.color\n");
    return VK_ERROR_INITIALIZATION_FAILED;
  }
  if(!isDepthLayout(info.layouts.depth))
  {
    LOGE("nvvk::RenderTarget: unsupported layouts.depth\n");
    return VK_ERROR_INITIALIZATION_FAILED;
  }
  if(!isColorSampleLayout(info.layouts.colorSample))
  {
    LOGE("nvvk::RenderTarget: unsupported layouts.colorSample\n");
    return VK_ERROR_INITIALIZATION_FAILED;
  }

  const VkImageLayout sampleLayout = effectiveColorSampleLayout(info);
  if(sampleLayout != VK_IMAGE_LAYOUT_GENERAL && sampleLayout != VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
  {
    LOGE("nvvk::RenderTarget: effective color sample layout must be GENERAL or SHADER_READ_ONLY_OPTIMAL\n");
    return VK_ERROR_INITIALIZATION_FAILED;
  }
  if(info.layouts.color == VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL && sampleLayout == VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL)
  {
    LOGE("nvvk::RenderTarget: layouts.colorSample cannot be COLOR_ATTACHMENT_OPTIMAL for sampling\n");
    return VK_ERROR_INITIALIZATION_FAILED;
  }

  const bool              userColorUsage = (info.colorUsage != 0);
  const VkImageUsageFlags baseColorUsage = userColorUsage ? info.colorUsage : defaultColorUsage();
  if((info.layouts.color != VK_IMAGE_LAYOUT_GENERAL || sampleLayout != VK_IMAGE_LAYOUT_GENERAL) && userColorUsage
     && (baseColorUsage & VK_IMAGE_USAGE_STORAGE_BIT) != 0)
  {
    LOGE("nvvk::RenderTarget: STORAGE in colorUsage requires GENERAL layouts\n");
    return VK_ERROR_INITIALIZATION_FAILED;
  }

  return VK_SUCCESS;
}

VkImageLayout nvvk::RenderTarget::effectiveColorSampleLayout(const CreateInfo& info)
{
  if(info.layouts.colorSample != VK_IMAGE_LAYOUT_UNDEFINED)
    return info.layouts.colorSample;
  if(info.layouts.color == VK_IMAGE_LAYOUT_GENERAL)
    return VK_IMAGE_LAYOUT_GENERAL;
  return VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
}

VkImageUsageFlags nvvk::RenderTarget::defaultColorUsage()
{
  return VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT
         | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
}

VkImageUsageFlags nvvk::RenderTarget::defaultDepthUsage()
{
  return VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT
         | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
}

VkImageViewType nvvk::RenderTarget::attachmentViewType(uint32_t layerCount)
{
  return layerCount > 1 ? VK_IMAGE_VIEW_TYPE_2D_ARRAY : VK_IMAGE_VIEW_TYPE_2D;
}

uint32_t nvvk::RenderTarget::attachmentViewLayers(uint32_t layerCount)
{
  return layerCount > 1 ? layerCount : 1;
}

VkImageSubresourceRange nvvk::RenderTarget::colorLayersRange(uint32_t layerCount, uint32_t baseArrayLayer, uint32_t layers)
{
  return {
      .aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT,
      .baseMipLevel   = 0,
      .levelCount     = 1,
      .baseArrayLayer = baseArrayLayer,
      .layerCount     = layers != 0 ? layers : attachmentViewLayers(layerCount),
  };
}

VkImageSubresourceRange nvvk::RenderTarget::depthLayersRange(VkImageAspectFlags aspect, uint32_t layerCount, uint32_t baseArrayLayer, uint32_t layers)
{
  return {
      .aspectMask     = aspect,
      .baseMipLevel   = 0,
      .levelCount     = 1,
      .baseArrayLayer = baseArrayLayer,
      .layerCount     = layers != 0 ? layers : attachmentViewLayers(layerCount),
  };
}

VkResult nvvk::RenderTarget::resolveColorUsage(const CreateInfo& info, VkFormat format, VkPhysicalDevice physicalDevice, ColorUsageResult& out)
{
  const bool              userColorUsage = (info.colorUsage != 0);
  const VkImageUsageFlags baseColorUsage = userColorUsage ? info.colorUsage : defaultColorUsage();
  const VkImageLayout     sampleLayout   = effectiveColorSampleLayout(info);

  VkFormatProperties props{};
  vkGetPhysicalDeviceFormatProperties(physicalDevice, format, &props);
  const bool formatSupportsStorage = (props.optimalTilingFeatures & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT) != 0;

  VkImageUsageFlags formatColorUsage = baseColorUsage;
  if((formatColorUsage & VK_IMAGE_USAGE_STORAGE_BIT) != 0 && !formatSupportsStorage)
  {
    if(userColorUsage)
    {
      LOGE("nvvk::RenderTarget: color format %d does not support STORAGE requested via colorUsage\n", int(format));
      return VK_ERROR_INITIALIZATION_FAILED;
    }
    formatColorUsage &= ~VK_IMAGE_USAGE_STORAGE_BIT;
  }
  if(!userColorUsage && formatSupportsStorage && info.layouts.color == VK_IMAGE_LAYOUT_GENERAL && sampleLayout == VK_IMAGE_LAYOUT_GENERAL)
  {
    formatColorUsage |= VK_IMAGE_USAGE_STORAGE_BIT;
  }

  out.resolveUsage = formatColorUsage;
  out.renderUsage  = formatColorUsage;
  if(info.msaa.samples != VK_SAMPLE_COUNT_1_BIT && (out.renderUsage & VK_IMAGE_USAGE_STORAGE_BIT) != 0)
  {
    if(userColorUsage)
    {
      LOGE("nvvk::RenderTarget: STORAGE in colorUsage on MSAA render image requires shaderStorageImageMultisample\n");
      return VK_ERROR_INITIALIZATION_FAILED;
    }
    out.renderUsage &= ~VK_IMAGE_USAGE_STORAGE_BIT;
  }
  return VK_SUCCESS;
}

VkImageAspectFlags nvvk::RenderTarget::depthStencilAspect() const
{
  return (m_createInfo.depthFormat != VK_FORMAT_UNDEFINED ? VK_IMAGE_ASPECT_DEPTH_BIT : 0)
         | (m_createInfo.stencilFormat != VK_FORMAT_UNDEFINED ? VK_IMAGE_ASPECT_STENCIL_BIT : 0);
}

bool nvvk::RenderTarget::isMsaa() const
{
  return m_createInfo.msaa.samples != VK_SAMPLE_COUNT_1_BIT;
}

bool nvvk::RenderTarget::wantsColorResolve() const
{
  return isMsaa() && m_createInfo.msaa.colorResolve != VK_RESOLVE_MODE_NONE;
}

bool nvvk::RenderTarget::wantsDepthResolve() const
{
  return isMsaa() && m_createInfo.depthFormat != VK_FORMAT_UNDEFINED && m_createInfo.msaa.depthResolve != VK_RESOLVE_MODE_NONE;
}

bool nvvk::RenderTarget::wantsStencilResolve() const
{
  return isMsaa() && m_createInfo.stencilFormat != VK_FORMAT_UNDEFINED && m_createInfo.msaa.stencilResolve != VK_RESOLVE_MODE_NONE;
}

bool nvvk::RenderTarget::wantsDepthStencilResolve() const
{
  return wantsDepthResolve() || wantsStencilResolve();
}

void nvvk::RenderTarget::destroyResources()
{
  if(m_createInfo.alloc == nullptr)
    return;

  nvvk::ResourceAllocator* alloc  = m_createInfo.alloc;
  VkDevice                 device = m_createInfo.device;

  if(m_depthResolveView != VK_NULL_HANDLE)
    vkDestroyImageView(device, m_depthResolveView, nullptr);
  m_depthResolveView = VK_NULL_HANDLE;

  if(m_depthView != VK_NULL_HANDLE)
    vkDestroyImageView(device, m_depthView, nullptr);
  m_depthView = VK_NULL_HANDLE;

  for(VkImageView& view : m_resolveImageViews)
  {
    if(view != VK_NULL_HANDLE)
      vkDestroyImageView(device, view, nullptr);
  }
  m_resolveImageViews.clear();

  for(const VkImageView& view : m_imageViews)
    vkDestroyImageView(device, view, nullptr);
  m_imageViews.clear();

  for(const VkImageView& view : m_uiImageViews)
    vkDestroyImageView(device, view, nullptr);
  m_uiImageViews.clear();

  if(m_depthImageResolved.image != VK_NULL_HANDLE)
    alloc->destroyImage(m_depthImageResolved);
  m_depthImageResolved = {};

  if(m_depthImage.image != VK_NULL_HANDLE)
    alloc->destroyImage(m_depthImage);
  m_depthImage = {};

  for(nvvk::Image& image : m_colorImagesResolved)
  {
    if(image.image != VK_NULL_HANDLE)
      alloc->destroyImage(image);
    image = {};
  }
  m_colorImagesResolved.clear();

  for(nvvk::Image& image : m_colorImages)
  {
    if(image.image != VK_NULL_HANDLE)
      alloc->destroyImage(image);
    image = {};
  }
  m_colorImages.clear();

  m_extent         = {};
  m_resourcesDirty = true;
}

VkResult nvvk::RenderTarget::createColorAttachment(uint32_t i, VkExtent2D extent, bool resolveColor, const std::string& namePrefix, VkPhysicalDevice physicalDevice)
{
  nvvk::ResourceAllocator* alloc = m_createInfo.alloc;
  nvvk::DebugUtil&         dutil = nvvk::DebugUtil::getInstance();

  ColorUsageResult usages{};
  NVVK_FAIL_RETURN(resolveColorUsage(m_createInfo, m_createInfo.colorFormats[i], physicalDevice, usages));

  const VkImageCreateInfo imageInfo = {
      .sType       = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .imageType   = VK_IMAGE_TYPE_2D,
      .format      = m_createInfo.colorFormats[i],
      .extent      = {extent.width, extent.height, 1},
      .mipLevels   = 1,
      .arrayLayers = m_createInfo.layerCount,
      .samples     = m_createInfo.msaa.samples,
      .usage       = usages.renderUsage,
  };
  NVVK_FAIL_RETURN(alloc->createImage(m_colorImages[i], imageInfo));
  dutil.setObjectName(m_colorImages[i].image, namePrefix + ":colorImage" + std::to_string(i));

  VkImageViewCreateInfo viewInfo = {
      .sType            = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
      .image            = m_colorImages[i].image,
      .viewType         = attachmentViewType(m_createInfo.layerCount),
      .format           = m_createInfo.colorFormats[i],
      .subresourceRange = colorLayersRange(m_createInfo.layerCount),
  };
  NVVK_FAIL_RETURN(vkCreateImageView(m_createInfo.device, &viewInfo, nullptr, &m_imageViews[i]));
  dutil.setObjectName(m_imageViews[i], namePrefix + ":colorView" + std::to_string(i));

  auto makeUiColorView = [&](VkImage image, VkImageView& dst) -> VkResult {
    VkImageViewCreateInfo uiViewInfo = viewInfo;
    uiViewInfo.image                 = image;
    uiViewInfo.viewType              = VK_IMAGE_VIEW_TYPE_2D;
    uiViewInfo.subresourceRange      = colorLayersRange(m_createInfo.layerCount, 0, 1);
    uiViewInfo.components            = {.a = VK_COMPONENT_SWIZZLE_ONE};
    NVVK_FAIL_RETURN(vkCreateImageView(m_createInfo.device, &uiViewInfo, nullptr, &dst));
    return VK_SUCCESS;
  };

  if(resolveColor)
  {
    VkImageCreateInfo resolveInfo = imageInfo;
    resolveInfo.samples           = VK_SAMPLE_COUNT_1_BIT;
    resolveInfo.usage             = usages.resolveUsage;
    NVVK_FAIL_RETURN(alloc->createImage(m_colorImagesResolved[i], resolveInfo));
    dutil.setObjectName(m_colorImagesResolved[i].image, namePrefix + ":colorImageResolved" + std::to_string(i));

    viewInfo.image            = m_colorImagesResolved[i].image;
    viewInfo.components       = {.a = VK_COMPONENT_SWIZZLE_IDENTITY};
    viewInfo.viewType         = attachmentViewType(m_createInfo.layerCount);
    viewInfo.subresourceRange = colorLayersRange(m_createInfo.layerCount);
    NVVK_FAIL_RETURN(vkCreateImageView(m_createInfo.device, &viewInfo, nullptr, &m_resolveImageViews[i]));
    dutil.setObjectName(m_resolveImageViews[i], namePrefix + ":colorViewResolved" + std::to_string(i));

    NVVK_FAIL_RETURN(makeUiColorView(m_colorImagesResolved[i].image, m_uiImageViews[i]));
    dutil.setObjectName(m_uiImageViews[i], namePrefix + ":uiColorView" + std::to_string(i));
  }
  else
  {
    NVVK_FAIL_RETURN(makeUiColorView(m_colorImages[i].image, m_uiImageViews[i]));
    dutil.setObjectName(m_uiImageViews[i], namePrefix + ":uiColorView" + std::to_string(i));
  }

  return VK_SUCCESS;
}

VkResult nvvk::RenderTarget::createDepthAttachment(VkExtent2D extent, bool resolveDepth, const std::string& namePrefix, VkPhysicalDevice physicalDevice)
{
  nvvk::ResourceAllocator* alloc = m_createInfo.alloc;
  nvvk::DebugUtil&         dutil = nvvk::DebugUtil::getInstance();

  const VkFormat depthStencilFormat =
      m_createInfo.depthFormat != VK_FORMAT_UNDEFINED ? m_createInfo.depthFormat : m_createInfo.stencilFormat;

  const VkImageAspectFlags depthAspect = depthStencilAspect();

  const bool        userDepthUsage = (m_createInfo.depthUsage != 0);
  VkImageUsageFlags depthUsage     = userDepthUsage ? m_createInfo.depthUsage : defaultDepthUsage();
  {
    VkFormatProperties props{};
    vkGetPhysicalDeviceFormatProperties(physicalDevice, depthStencilFormat, &props);
    const bool depthSupportsStorage = (props.optimalTilingFeatures & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT) != 0;
    if(userDepthUsage && (depthUsage & VK_IMAGE_USAGE_STORAGE_BIT) != 0 && !depthSupportsStorage)
    {
      LOGE("nvvk::RenderTarget: depth format %d does not support STORAGE requested via depthUsage\n", int(depthStencilFormat));
      return VK_ERROR_INITIALIZATION_FAILED;
    }
    if(!userDepthUsage && depthSupportsStorage && m_createInfo.layouts.depth == VK_IMAGE_LAYOUT_GENERAL)
      depthUsage |= VK_IMAGE_USAGE_STORAGE_BIT;
  }

  VkImageUsageFlags formatDepthUsage = depthUsage;
  VkImageUsageFlags depthImageUsage  = formatDepthUsage;
  if(m_createInfo.msaa.samples != VK_SAMPLE_COUNT_1_BIT && (depthImageUsage & VK_IMAGE_USAGE_STORAGE_BIT) != 0)
  {
    if(userDepthUsage)
    {
      LOGE("nvvk::RenderTarget: STORAGE in depthUsage on MSAA depth image requires shaderStorageImageMultisample\n");
      return VK_ERROR_INITIALIZATION_FAILED;
    }
    depthImageUsage &= ~VK_IMAGE_USAGE_STORAGE_BIT;
  }
  if(m_createInfo.layouts.depth != VK_IMAGE_LAYOUT_GENERAL && (formatDepthUsage & VK_IMAGE_USAGE_STORAGE_BIT) != 0)
  {
    LOGE("nvvk::RenderTarget: STORAGE usage requires GENERAL depthLayout\n");
    return VK_ERROR_INITIALIZATION_FAILED;
  }

  const VkImageCreateInfo imageInfo = {
      .sType       = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .imageType   = VK_IMAGE_TYPE_2D,
      .format      = depthStencilFormat,
      .extent      = {extent.width, extent.height, 1},
      .mipLevels   = 1,
      .arrayLayers = m_createInfo.layerCount,
      .samples     = m_createInfo.msaa.samples,
      .usage       = depthImageUsage,
  };
  NVVK_FAIL_RETURN(alloc->createImage(m_depthImage, imageInfo));
  dutil.setObjectName(m_depthImage.image, namePrefix + ":depthImage");

  const VkImageViewCreateInfo viewInfo = {
      .sType            = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
      .image            = m_depthImage.image,
      .viewType         = attachmentViewType(m_createInfo.layerCount),
      .format           = depthStencilFormat,
      .subresourceRange = depthLayersRange(depthAspect, m_createInfo.layerCount),
  };
  NVVK_FAIL_RETURN(vkCreateImageView(m_createInfo.device, &viewInfo, nullptr, &m_depthView));
  dutil.setObjectName(m_depthView, namePrefix + ":depthView");

  if(resolveDepth)
  {
    VkImageCreateInfo resolveInfo = imageInfo;
    resolveInfo.samples           = VK_SAMPLE_COUNT_1_BIT;
    resolveInfo.usage             = formatDepthUsage;
    NVVK_FAIL_RETURN(alloc->createImage(m_depthImageResolved, resolveInfo));
    dutil.setObjectName(m_depthImageResolved.image, namePrefix + ":depthImageResolved");

    const VkImageViewCreateInfo resolveViewInfo = {
        .sType            = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .image            = m_depthImageResolved.image,
        .viewType         = attachmentViewType(m_createInfo.layerCount),
        .format           = depthStencilFormat,
        .subresourceRange = depthLayersRange(depthAspect, m_createInfo.layerCount),
    };
    NVVK_FAIL_RETURN(vkCreateImageView(m_createInfo.device, &resolveViewInfo, nullptr, &m_depthResolveView));
    dutil.setObjectName(m_depthResolveView, namePrefix + ":depthViewResolved");
  }

  return VK_SUCCESS;
}

VkResult nvvk::RenderTarget::transitionToWorkingLayouts(VkCommandBuffer cmd)
{
  std::vector<VkImageMemoryBarrier2> barriers;
  const VkImageLayout                colorLayout = m_createInfo.layouts.color;
  const VkImageLayout                depthLayout = m_createInfo.layouts.depth;

  auto addTransition = [&](VkImage image, VkImageLayout layout, VkImageAspectFlags aspect) {
    if(image == VK_NULL_HANDLE)
      return;
    barriers.push_back(nvvk::makeImageMemoryBarrier(
        {.image            = image,
         .oldLayout        = VK_IMAGE_LAYOUT_UNDEFINED,
         .newLayout        = layout,
         .subresourceRange = {aspect, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS}}));
  };

  for(const nvvk::Image& color : m_colorImages)
    addTransition(color.image, colorLayout, VK_IMAGE_ASPECT_COLOR_BIT);
  for(const nvvk::Image& resolved : m_colorImagesResolved)
    addTransition(resolved.image, colorLayout, VK_IMAGE_ASPECT_COLOR_BIT);

  if(m_depthImage.image != VK_NULL_HANDLE)
  {
    const VkImageAspectFlags depthAspect = depthStencilAspect();
    addTransition(m_depthImage.image, depthLayout, depthAspect);
    addTransition(m_depthImageResolved.image, depthLayout, depthAspect);
  }

  cmdSubmitBarriers(cmd, barriers);
  return VK_SUCCESS;
}

VkResult nvvk::RenderTarget::createResources(VkCommandBuffer cmd, VkExtent2D extent)
{
  const std::string namePrefix   = m_createInfo.debugName.empty() ? "RT" : "RenderTarget:" + m_createInfo.debugName;
  nvvk::ResourceAllocator* alloc = m_createInfo.alloc;
  const VkPhysicalDevice   physicalDevice = alloc->getPhysicalDevice();

  const bool resolveColor = wantsColorResolve();
  const bool resolveDepth = wantsDepthStencilResolve();  // one resolve image backs both aspects

  const uint32_t colorCount = static_cast<uint32_t>(m_createInfo.colorFormats.size());
  m_colorImages.resize(colorCount);
  m_colorImagesResolved.resize(colorCount);
  m_imageViews.resize(colorCount);
  m_uiImageViews.resize(colorCount);
  m_resolveImageViews.resize(colorCount, VK_NULL_HANDLE);

  for(uint32_t i = 0; i < colorCount; ++i)
  {
    if(m_createInfo.colorFormats[i] == VK_FORMAT_UNDEFINED)
      continue;
    NVVK_FAIL_RETURN(createColorAttachment(i, extent, resolveColor, namePrefix, physicalDevice));
  }

  if(m_createInfo.depthFormat != VK_FORMAT_UNDEFINED || m_createInfo.stencilFormat != VK_FORMAT_UNDEFINED)
    NVVK_FAIL_RETURN(createDepthAttachment(extent, resolveDepth, namePrefix, physicalDevice));

  NVVK_FAIL_RETURN(transitionToWorkingLayouts(cmd));
  m_extent         = extent;
  m_resourcesDirty = false;
  return VK_SUCCESS;
}

//--------------------------------------------------------------------------------------------------
// Usage example (matches vk_mini_samples raster flow; MSAA optional)
//--------------------------------------------------------------------------------------------------
[[maybe_unused]] static void usage_RenderTarget()
{
  VkDevice                device{};
  nvvk::ResourceAllocator allocator;

  nvvk::RenderTarget renderTarget;
  NVVK_CHECK(renderTarget.init({
      .device       = device,
      .alloc        = &allocator,
      .colorFormats = {VK_FORMAT_R16G16B16A16_SFLOAT},
      .depthFormat  = VK_FORMAT_D32_SFLOAT,
      .debugName    = "MyOffscreen",
  }));

  // Optional MSAA; takes effect on the next update().
  NVVK_CHECK(renderTarget.setMsaa({.samples = VK_SAMPLE_COUNT_4_BIT, .colorResolve = VK_RESOLVE_MODE_AVERAGE_BIT}));

  VkCommandBuffer cmd = VK_NULL_HANDLE;
  NVVK_CHECK(renderTarget.update(cmd, VkExtent2D{1920, 1080}));

  // Per frame: fill attachment views, set clear values, begin dynamic rendering.
  nvvk::RenderTargetState state;
  renderTarget.fillState(state);
  state.colorAttachments[0].clearValue = {.color = {{0.1F, 0.1F, 0.1F, 1.0F}}};
  state.depthAttachment.clearValue     = {.depthStencil = {1.0F, 0}};
  state.cmdBeginRendering(cmd, {});  // default AttachmentOps: LOAD_OP_CLEAR

  vkCmdEndRendering(cmd);

  // ImGui: viewportImage.update(renderTarget.getUiImageView());
  // Save:  saveImageToFile(renderTarget.getSampleImage(), renderTarget.getSize(), ...);

  renderTarget.deinit();
}
