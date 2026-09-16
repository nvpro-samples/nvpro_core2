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

#include <algorithm>
#include <cfloat>
#include <cstring>
#include <span>

#include <imgui/backends/imgui_impl_vulkan.h>

#include <nvgui/property_editor.hpp>
#include <nvvk/check_error.hpp>
#include <nvvk/debug_util.hpp>
#include <nvvk/descriptors.hpp>
#include <nvvk/graphics_pipeline.hpp>

#include "imgui_texture_visualizer.hpp"
#include "imgui_texture_visualizer.slang.h"

namespace nvapp {

namespace PE = nvgui::PropertyEditor;


VkResult ImTextureVisualizer::init(const InitInfo& info)
{
  assert(m_device == VK_NULL_HANDLE && "Missing deinit()");
  assert(info.device != VK_NULL_HANDLE);
  assert((info.renderPass != VK_NULL_HANDLE || !info.colorFormats.empty()) && "Needs either a render pass or color formats");

  m_device = info.device;

  const VkResult result = createResources(info);
  if(result != VK_SUCCESS)
  {
    deinit();
  }
  return result;
}

VkResult ImTextureVisualizer::createResources(const InitInfo& info)
{
  // The lod is always provided explicitly by the shader, the mipmap mode is linear
  // so that fractional lod values blend between two levels.
  VkSamplerCreateInfo samplerInfo{
      .sType        = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
      .magFilter    = VK_FILTER_LINEAR,
      .minFilter    = VK_FILTER_LINEAR,
      .mipmapMode   = VK_SAMPLER_MIPMAP_MODE_LINEAR,
      .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
      .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
      .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
      .minLod       = 0.0f,
      .maxLod       = VK_LOD_CLAMP_NONE,
  };
  NVVK_FAIL_RETURN(vkCreateSampler(m_device, &samplerInfo, nullptr, &m_samplerLinear));
  NVVK_DBG_NAME(m_samplerLinear);

  samplerInfo.magFilter = VK_FILTER_NEAREST;
  samplerInfo.minFilter = VK_FILTER_NEAREST;
  NVVK_FAIL_RETURN(vkCreateSampler(m_device, &samplerInfo, nullptr, &m_samplerNearest));
  NVVK_DBG_NAME(m_samplerNearest);

  // The texture is provided as push descriptor from within the ImGui draw callback,
  // this avoids having to manage descriptor sets per displayed image.
  nvvk::DescriptorBindings bindings;
  bindings.addBinding(0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT);
  NVVK_FAIL_RETURN(bindings.createDescriptorSetLayout(m_device, VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT,
                                                      &m_descriptorSetLayout));
  NVVK_DBG_NAME(m_descriptorSetLayout);

  const VkPushConstantRange pushConstantRange{.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                                              .offset     = 0,
                                              .size       = sizeof(PushConstants)};
  NVVK_FAIL_RETURN(nvvk::createPipelineLayout(m_device, &m_pipelineLayout, {m_descriptorSetLayout}, {pushConstantRange}));
  NVVK_DBG_NAME(m_pipelineLayout);

  // Matches what the ImGui Vulkan backend uses, so that the quad composites like any other ImGui draw
  const size_t attachmentCount = (info.renderPass != VK_NULL_HANDLE) ? 1 : info.colorFormats.size();

  nvvk::GraphicsPipelineState graphicsState;
  graphicsState.inputAssemblyState.topology           = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP;
  graphicsState.rasterizationState.cullMode           = VK_CULL_MODE_NONE;
  graphicsState.depthStencilState.depthTestEnable     = VK_FALSE;
  graphicsState.depthStencilState.depthWriteEnable    = VK_FALSE;
  graphicsState.multisampleState.rasterizationSamples = (info.samples != 0) ? info.samples : VK_SAMPLE_COUNT_1_BIT;
  graphicsState.colorBlendEnables.assign(attachmentCount, VK_TRUE);
  graphicsState.colorWriteMasks.assign(attachmentCount, VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT
                                                            | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT);
  graphicsState.colorBlendEquations.assign(attachmentCount, VkColorBlendEquationEXT{
                                                                .srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA,
                                                                .dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
                                                                .colorBlendOp        = VK_BLEND_OP_ADD,
                                                                .srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE,
                                                                .dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
                                                                .alphaBlendOp = VK_BLEND_OP_ADD,
                                                            });

  nvvk::GraphicsPipelineCreator creator;
  creator.pipelineInfo.layout                    = m_pipelineLayout;
  creator.pipelineInfo.renderPass                = info.renderPass;
  creator.pipelineInfo.subpass                   = info.subpass;
  creator.colorFormats                           = info.colorFormats;
  creator.renderingState.depthAttachmentFormat   = VK_FORMAT_UNDEFINED;
  creator.renderingState.stencilAttachmentFormat = VK_FORMAT_UNDEFINED;
  // ImGui uses the non "with count" variants, stay identical so that the states
  // set here and by the backend refer to the same thing.
  creator.dynamicStateValues          = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
  creator.viewportState.viewportCount = 1;
  creator.viewportState.scissorCount  = 1;
  // All entry points live in the same SPIR-V module, they are selected by name
  const std::span<const uint32_t> spirv(imgui_texture_visualizer_slang);

  creator.addShader(VK_SHADER_STAGE_VERTEX_BIT, "vertexMain", spirv);
  creator.addShader(VK_SHADER_STAGE_FRAGMENT_BIT, "fragmentMain", spirv);
  NVVK_FAIL_RETURN(creator.createGraphicsPipeline(m_device, info.pipelineCache, graphicsState, &m_pipeline));
  NVVK_DBG_NAME(m_pipeline);

  // Same state, but with the fragment entry point that samples a layer of a 2D array view
  creator.clearShaders();
  creator.addShader(VK_SHADER_STAGE_VERTEX_BIT, "vertexMain", spirv);
  creator.addShader(VK_SHADER_STAGE_FRAGMENT_BIT, "fragmentArrayMain", spirv);
  NVVK_FAIL_RETURN(creator.createGraphicsPipeline(m_device, info.pipelineCache, graphicsState, &m_pipelineArray));
  NVVK_DBG_NAME(m_pipelineArray);

  // And the one that samples a slice of a 3D view
  creator.clearShaders();
  creator.addShader(VK_SHADER_STAGE_VERTEX_BIT, "vertexMain", spirv);
  creator.addShader(VK_SHADER_STAGE_FRAGMENT_BIT, "fragment3DMain", spirv);
  NVVK_FAIL_RETURN(creator.createGraphicsPipeline(m_device, info.pipelineCache, graphicsState, &m_pipeline3D));
  NVVK_DBG_NAME(m_pipeline3D);

  return VK_SUCCESS;
}

void ImTextureVisualizer::deinit()
{
  if(m_device == VK_NULL_HANDLE)
  {
    return;
  }

  vkDestroyPipeline(m_device, m_pipeline, nullptr);
  vkDestroyPipeline(m_device, m_pipelineArray, nullptr);
  vkDestroyPipeline(m_device, m_pipeline3D, nullptr);
  vkDestroyPipelineLayout(m_device, m_pipelineLayout, nullptr);
  vkDestroyDescriptorSetLayout(m_device, m_descriptorSetLayout, nullptr);
  vkDestroySampler(m_device, m_samplerLinear, nullptr);
  vkDestroySampler(m_device, m_samplerNearest, nullptr);

  m_pipeline            = VK_NULL_HANDLE;
  m_pipelineArray       = VK_NULL_HANDLE;
  m_pipeline3D          = VK_NULL_HANDLE;
  m_pipelineLayout      = VK_NULL_HANDLE;
  m_descriptorSetLayout = VK_NULL_HANDLE;
  m_samplerLinear       = VK_NULL_HANDLE;
  m_samplerNearest      = VK_NULL_HANDLE;
  m_device              = VK_NULL_HANDLE;
}

void ImTextureVisualizer::image(VkImageView view, const ImVec2& size, const Settings& settings, VkImageLayout imageLayout, VkImageViewType viewType)
{
  // An unsupported view type must not fall through to the 2D pipeline, it would
  // sample with the wrong sampler type, hence the check in release builds too.
  const bool supportedViewType =
      (viewType == VK_IMAGE_VIEW_TYPE_2D || viewType == VK_IMAGE_VIEW_TYPE_2D_ARRAY || viewType == VK_IMAGE_VIEW_TYPE_3D);

  assert(isValid() && "Missing init()");
  assert(view != VK_NULL_HANDLE && "ImTextureVisualizer::image requires a valid VkImageView");
  assert(supportedViewType && "ImTextureVisualizer::image only supports 2D, 2D array and 3D views");

  // Reserve the space first, this also gives the item its id and clipping
  const ImVec2 topLeft = ImGui::GetCursorScreenPos();
  ImGui::Dummy(size);
  if(!isValid() || view == VK_NULL_HANDLE || !supportedViewType || !ImGui::IsItemVisible())
  {
    return;
  }
  const ImVec2 bottomRight(topLeft.x + size.x, topLeft.y + size.y);

  // The quad is drawn unclipped and the scissor takes care of the clipping, this
  // keeps the texture coordinates of the visible part correct.
  ImDrawList*  drawList = ImGui::GetWindowDrawList();
  const ImVec2 clipMin  = drawList->GetClipRectMin();
  const ImVec2 clipMax  = drawList->GetClipRectMax();
  const ImVec2 visibleMin(std::max(topLeft.x, clipMin.x), std::max(topLeft.y, clipMin.y));
  const ImVec2 visibleMax(std::min(bottomRight.x, clipMax.x), std::min(bottomRight.y, clipMax.y));
  if(visibleMax.x <= visibleMin.x || visibleMax.y <= visibleMin.y)
  {
    return;
  }

  // Same transform ImGui_ImplVulkan_SetupRenderState applies to the ImGui vertices.
  // Taken from the window's viewport so that this also works with multi viewports.
  const ImGuiViewport* imViewport  = ImGui::GetWindowViewport();
  const ImVec2         displayPos  = imViewport->Pos;
  const ImVec2         displaySize = imViewport->Size;
  const ImVec2 fbScale = (imViewport->FramebufferScale.x != 0.0f) ? imViewport->FramebufferScale : ImGui::GetIO().DisplayFramebufferScale;
  if(displaySize.x <= 0.0f || displaySize.y <= 0.0f)
  {
    return;
  }
  const float fbWidth  = displaySize.x * fbScale.x;
  const float fbHeight = displaySize.y * fbScale.y;

  auto toClipSpace = [&](const ImVec2& p) {
    return ImVec2((p.x - displayPos.x) * 2.0f / displaySize.x - 1.0f, (p.y - displayPos.y) * 2.0f / displaySize.y - 1.0f);
  };
  const ImVec2 clipTopLeft     = toClipSpace(topLeft);
  const ImVec2 clipBottomRight = toClipSpace(bottomRight);

  // Scissor in framebuffer space
  const float scissorMinX = std::clamp((visibleMin.x - displayPos.x) * fbScale.x, 0.0f, fbWidth);
  const float scissorMinY = std::clamp((visibleMin.y - displayPos.y) * fbScale.y, 0.0f, fbHeight);
  const float scissorMaxX = std::clamp((visibleMax.x - displayPos.x) * fbScale.x, 0.0f, fbWidth);
  const float scissorMaxY = std::clamp((visibleMax.y - displayPos.y) * fbScale.y, 0.0f, fbHeight);
  if(scissorMaxX <= scissorMinX || scissorMaxY <= scissorMinY)
  {
    return;
  }

  DrawCallbackData data{};
  data.pipeline       = (viewType == VK_IMAGE_VIEW_TYPE_2D_ARRAY) ? m_pipelineArray :
                        (viewType == VK_IMAGE_VIEW_TYPE_3D)       ? m_pipeline3D :
                                                                    m_pipeline;
  data.pipelineLayout = m_pipelineLayout;
  data.imageView      = view;
  data.sampler        = settings.nearestFilter ? m_samplerNearest : m_samplerLinear;
  data.imageLayout    = imageLayout;
  data.viewport = VkViewport{.x = 0.0f, .y = 0.0f, .width = fbWidth, .height = fbHeight, .minDepth = 0.0f, .maxDepth = 1.0f};
  data.scissor = VkRect2D{.offset = {int32_t(scissorMinX), int32_t(scissorMinY)},
                          .extent = {uint32_t(scissorMaxX - scissorMinX), uint32_t(scissorMaxY - scissorMinY)}};

  PushConstants& pushConstants = data.pushConstants;
  pushConstants.posRect        = glm::vec4(clipTopLeft.x, clipTopLeft.y, clipBottomRight.x, clipBottomRight.y);
  pushConstants.uvRect         = settings.uvRect;
  pushConstants.powExp         = settings.pow;
  pushConstants.scale          = settings.scale;
  pushConstants.add            = settings.add;
  pushConstants.lod            = settings.lod;
  pushConstants.layer          = settings.layer;
  pushConstants.zCoord         = settings.zCoord;
  pushConstants.swizzle        = (uint32_t(settings.swizzle.x) << 0) | (uint32_t(settings.swizzle.y) << 8)
                          | (uint32_t(settings.swizzle.z) << 16) | (uint32_t(settings.swizzle.w) << 24);
  pushConstants.invert = (settings.invertChannel.x ? 1u : 0u) | (settings.invertChannel.y ? 2u : 0u)
                         | (settings.invertChannel.z ? 4u : 0u) | (settings.invertChannel.w ? 8u : 0u);

  // ImGui keeps a copy of the data, no need to keep it alive here.
  drawList->AddCallback(drawCallback, &data, sizeof(data));
  // Restores the pipeline, vertex buffers, push constants and descriptors of the backend
  drawList->AddCallback(ImDrawCallback_ResetRenderState, nullptr);
}

void ImTextureVisualizer::drawCallback(const ImDrawList* /*drawList*/, const ImDrawCmd* drawCmd)
{
  const ImGui_ImplVulkan_RenderState* renderState =
      static_cast<const ImGui_ImplVulkan_RenderState*>(ImGui::GetPlatformIO().Renderer_RenderState);
  assert(renderState != nullptr && "The ImGui renderer backend must be the Vulkan one");
  assert(drawCmd->UserCallbackDataSize == int(sizeof(DrawCallbackData)));
  if(renderState == nullptr || drawCmd->UserCallbackDataSize != int(sizeof(DrawCallbackData)))
  {
    return;
  }

  // The ImGui side buffer holding the copy has no alignment guarantees
  DrawCallbackData data;
  memcpy(&data, drawCmd->UserCallbackData, sizeof(data));

  const VkCommandBuffer cmd = renderState->CommandBuffer;

  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, data.pipeline);
  vkCmdSetViewport(cmd, 0, 1, &data.viewport);
  vkCmdSetScissor(cmd, 0, 1, &data.scissor);

  const VkDescriptorImageInfo imageInfo{.sampler = data.sampler, .imageView = data.imageView, .imageLayout = data.imageLayout};
  const VkWriteDescriptorSet writeSet{
      .sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
      .dstBinding      = 0,
      .descriptorCount = 1,
      .descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
      .pImageInfo      = &imageInfo,
  };
  vkCmdPushDescriptorSet(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, data.pipelineLayout, 0, 1, &writeSet);
  vkCmdPushConstants(cmd, data.pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                     sizeof(PushConstants), &data.pushConstants);
  vkCmdDraw(cmd, 4, 1, 0, 0);
}

bool ImTextureVisualizer::settingsWidget(Settings& settings, uint32_t mipLevels, uint32_t layerCount, VkImageViewType viewType)
{
  static const char* channelNames[] = {"R", "G", "B", "A", "0", "1"};

  bool changed = false;

  ImGui::PushID(&settings);
  if(PE::begin())
  {
    changed |= PE::entry(
        "Swizzle",
        [&]() {
          bool        modified = false;
          const float width    = (ImGui::GetContentRegionAvail().x - ImGui::GetStyle().ItemSpacing.x * 3.0f) * 0.25f;
          for(int channel = 0; channel < 4; channel++)
          {
            if(channel != 0)
            {
              ImGui::SameLine();
            }
            ImGui::PushID(channel);
            ImGui::SetNextItemWidth(width);
            int current = int(settings.swizzle[channel]);
            if(ImGui::Combo("##swizzle", &current, channelNames, IM_ARRAYSIZE(channelNames)))
            {
              settings.swizzle[channel] = uint8_t(current);
              modified                  = true;
            }
            ImGui::PopID();
          }
          return modified;
        },
        "Source of the R, G, B and A output channels: a texture channel, or the constant 0 or 1");

    changed |= PE::entry(
        "Invert",
        [&]() {
          bool modified = false;
          for(int channel = 0; channel < 4; channel++)
          {
            if(channel != 0)
            {
              ImGui::SameLine();
            }
            modified |= ImGui::Checkbox(channelNames[channel], &settings.invertChannel[channel]);
          }
          return modified;
        },
        "Replaces the R, G, B, A output channel by 1 - value. Applied last, after the exponent");

    changed |= PE::DragFloat4("Scale", &settings.scale.x, 0.01f, -FLT_MAX, FLT_MAX, "%.3f", 0,
                              "Per channel multiplier, applied after the swizzle");
    changed |= PE::DragFloat4("Add", &settings.add.x, 0.01f, -FLT_MAX, FLT_MAX, "%.3f", 0, "Per channel offset, applied after the scale");
    changed |= PE::DragFloat4("Exponent", &settings.pow.x, 0.01f, 0.0f, 100.0f, "%.3f", ImGuiSliderFlags_AlwaysClamp,
                              "Per channel exponent, use 1/2.2 = 0.4545 to gamma correct linear data");
    changed |= PE::SliderFloat("Mip Level", &settings.lod, 0.0f, float(std::max(mipLevels, 1u) - 1), "%.2f",
                               ImGuiSliderFlags_AlwaysClamp, "Mip level to sample, fractional levels are interpolated");
    if(viewType == VK_IMAGE_VIEW_TYPE_2D_ARRAY && layerCount > 1)
    {
      int layer = int(settings.layer);
      if(PE::SliderInt("Array Layer", &layer, 0, int(layerCount) - 1, "%d", ImGuiSliderFlags_AlwaysClamp,
                       "Layer of the 2D array texture to display"))
      {
        settings.layer = uint32_t(layer);
        changed        = true;
      }
    }
    if(viewType == VK_IMAGE_VIEW_TYPE_3D)
    {
      changed |= PE::SliderFloat("Z Coord", &settings.zCoord, 0.0f, 1.0f, "%.3f", ImGuiSliderFlags_AlwaysClamp,
                                 "Normalized z of the 3D texture slice to display, slices are interpolated");
    }
    changed |= PE::Checkbox("Nearest Filter", &settings.nearestFilter, "Sample with nearest instead of linear filtering");

    changed |= PE::DragFloat4("UV Rect", &settings.uvRect.x, 0.002f, -FLT_MAX, FLT_MAX, "%.3f", 0,
                              "Sub-rectangle of the texture to display (u0, v0, u1, v1), "
                              "0,0,1,1 is the whole texture, a smaller rectangle zooms in");
    changed |= PE::entry(
        "UV Zoom",
        [&]() {
          // Zoom in/out around the center of the current rectangle
          const glm::vec2 center = glm::vec2(settings.uvRect.x + settings.uvRect.z, settings.uvRect.y + settings.uvRect.w) * 0.5f;
          const glm::vec2 half = glm::vec2(settings.uvRect.z - settings.uvRect.x, settings.uvRect.w - settings.uvRect.y) * 0.5f;

          bool  modified = false;
          float factor   = 0.0f;
          if(ImGui::SmallButton("-"))
          {
            factor = 2.0f;
          }
          ImGui::SameLine();
          if(ImGui::SmallButton("+"))
          {
            factor = 0.5f;
          }
          ImGui::SameLine();
          if(ImGui::SmallButton("fit"))
          {
            settings.uvRect = glm::vec4(0.0f, 0.0f, 1.0f, 1.0f);
            modified        = true;
          }
          if(factor != 0.0f)
          {
            settings.uvRect = glm::vec4(center - half * factor, center + half * factor);
            modified        = true;
          }
          return modified;
        },
        "Zoom the sub-rectangle out, in, or back to the whole texture");

    changed |= PE::entry(
        "Presets",
        [&]() {
          struct Preset
          {
            const char* name;
            glm::u8vec4 swizzle;
          };
          static const Preset presets[] = {
              {"RGBA", {eChannelR, eChannelG, eChannelB, eChannelA}},
              {"RGB", {eChannelR, eChannelG, eChannelB, eChannelOne}},
              {"R", {eChannelR, eChannelR, eChannelR, eChannelOne}},
              {"G", {eChannelG, eChannelG, eChannelG, eChannelOne}},
              {"B", {eChannelB, eChannelB, eChannelB, eChannelOne}},
              {"A", {eChannelA, eChannelA, eChannelA, eChannelOne}},
          };

          bool modified = false;
          for(size_t i = 0; i < IM_ARRAYSIZE(presets); i++)
          {
            if(i != 0)
            {
              ImGui::SameLine();
            }
            if(ImGui::SmallButton(presets[i].name))
            {
              settings.swizzle = presets[i].swizzle;
              modified         = true;
            }
          }
          ImGui::SameLine();
          if(ImGui::SmallButton("reset"))
          {
            settings = Settings{};
            modified = true;
          }
          return modified;
        },
        "Channel mapping shortcuts, 'reset' restores all default values");

    PE::end();
  }
  ImGui::PopID();

  return changed;
}

}  // namespace nvapp
