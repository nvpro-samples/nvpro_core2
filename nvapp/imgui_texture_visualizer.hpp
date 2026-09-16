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

#include <cassert>
#include <vector>

#include <glm/glm.hpp>
#include <glm/ext/vector_uint4_sized.hpp>

#include <imgui/imgui.h>
#include <volk.h>

namespace nvapp {

//---------------------------------------------------------------------------
// nvapp::ImTextureVisualizer
//
// Displays a VkImageView inside an ImGui window with debug visualization
// controls: per output channel source selection (swizzle), per channel
// inversion, scale/bias, per channel exponent (gamma), mip level, array layer
// and 3D slice selection, and a texture sub-rectangle to zoom into.
//
// It works by inserting an `ImDrawList::AddCallback` into the ImGui draw list.
// During `ImGui_ImplVulkan_RenderDrawData` that callback binds a small private
// graphics pipeline and draws the quad itself, none of this goes through the
// textured pipeline of the ImGui backend. As a consequence:
//   * The image does NOT need to be registered with `ImGui_ImplVulkan_AddTexture`
//     (no `nvapp::ImTexture` needed); the VkImageView is bound through a push
//     descriptor at draw time.
//   * The sampler is ours, therefore explicit mip level sampling is possible.
//   * Three pipelines are created, sampling a 2D view, a layer of a 2D array
//     view, or a slice of a 3D view. The sampler type is part of the shader, so
//     this can be neither a push nor a specialization constant; `image()` picks
//     the pipeline from `viewType`. All share one SPIR-V module, the fragment
//     entry point selects the sampler.
//   * The ImGui render state is restored afterwards by appending an
//     `ImDrawCallback_ResetRenderState` callback.
//
// Requirements:
//   * Vulkan 1.4 (push descriptors are used, `vkCmdPushDescriptorSet`).
//   * The image view must be a VK_IMAGE_VIEW_TYPE_2D, _2D_ARRAY or _3D view of
//     a color format that can be sampled as float (UNORM/SNORM/SFLOAT/sRGB ...),
//     integer views are not supported.
//   * The image must be in `imageLayout` and readable by the fragment stage
//     while ImGui is rendered.
//
// Lifecycle (matches nvpro_core2 convention: explicit init/deinit, no RAII):
//   * `init()` once the format ImGui renders into is known, `deinit()` before
//     destruction. If that format changes, call `deinit()` and `init()` again.
//
// Usage:
//   nvapp::ImTextureVisualizer visualizer;
//   visualizer.init({.device = app.getDevice(), .colorFormats = {app.getSwapchainFormat()}});
//   ...
//   // onUIRender()
//   nvapp::ImTextureVisualizer::settingsWidget(m_settings, texture.mipLevels);
//   visualizer.image(texture.view, ImGui::GetContentRegionAvail(), m_settings);
//   ...
//   visualizer.deinit();
//
// The shader lives in `imgui_texture_visualizer.slang`, its SPIR-V is embedded
// in `imgui_texture_visualizer.slang.h` (see the regeneration command line in
// the .slang file).
//---------------------------------------------------------------------------
class ImTextureVisualizer
{
public:
  // Source of an output channel, see `Settings::swizzle`
  enum Channel : uint8_t
  {
    eChannelR    = 0,
    eChannelG    = 1,
    eChannelB    = 2,
    eChannelA    = 3,
    eChannelZero = 4,
    eChannelOne  = 5,
  };

  // Builds a `Settings::swizzle` value, one `Channel` per output channel
  static glm::u8vec4 makeSwizzle(Channel r, Channel g, Channel b, Channel a)
  {
    return glm::u8vec4(uint8_t(r), uint8_t(g), uint8_t(b), uint8_t(a));
  }

  // Which part of the texture is shown and how a sampled texel is turned into
  // the displayed color. The color operations are applied in this order:
  //   1. swizzle: color[i] = texel[swizzle[i]]  (or the constant 0 / 1)
  //   2. scale and bias: color = color * scale + add
  //   3. exponent: color = pow(max(color, 0), pow)   (e.g. 1/2.2 for gamma)
  //   4. invert: color[i] = 1 - color[i]  where invertChannel[i]
  // Inversion comes last on purpose, inverting before the exponent can already
  // be expressed with scale = -1 and add = 1.
  struct Settings
  {
    glm::bvec4  invertChannel{false, false, false, false};            // per output channel: 1 - value
    glm::u8vec4 swizzle{eChannelR, eChannelG, eChannelB, eChannelA};  // per output channel: `Channel` source
    glm::vec4   pow{1.0f, 1.0f, 1.0f, 1.0f};                          // per channel exponent
    glm::vec4   scale{1.0f, 1.0f, 1.0f, 1.0f};                        // per channel multiplier
    glm::vec4   add{0.0f, 0.0f, 0.0f, 0.0f};                          // per channel offset
    // Sub-rectangle of the texture that is stretched over the item: xy is the
    // texture coordinate of the top left corner, zw the one of the bottom right
    // corner. The default shows the whole texture, a smaller rectangle zooms in
    // and a reversed one mirrors.
    glm::vec4 uvRect{0.0f, 0.0f, 1.0f, 1.0f};
    float     lod{0.0f};             // mip level, fractional levels are interpolated
    uint32_t  layer{0};              // array layer, only used for VK_IMAGE_VIEW_TYPE_2D_ARRAY views
    float     zCoord{0.0f};          // normalized z, only used for VK_IMAGE_VIEW_TYPE_3D views
    bool      nearestFilter{false};  // magnify with nearest instead of linear
  };

  struct InitInfo
  {
    VkDevice device{VK_NULL_HANDLE};
    // Color attachment format(s) ImGui renders into, unused if `renderPass` is set.
    // With nvapp::Application this is `app.getSwapchainFormat()`.
    std::vector<VkFormat> colorFormats{};
    // Optional, when ImGui is rendered within a render pass rather than with dynamic rendering
    VkRenderPass          renderPass{VK_NULL_HANDLE};
    uint32_t              subpass{0};
    VkSampleCountFlagBits samples{VK_SAMPLE_COUNT_1_BIT};
    VkPipelineCache       pipelineCache{VK_NULL_HANDLE};
  };

  ImTextureVisualizer() = default;
  ~ImTextureVisualizer() { assert(m_device == VK_NULL_HANDLE && "Missing deinit()"); }

  ImTextureVisualizer(const ImTextureVisualizer&)            = delete;
  ImTextureVisualizer& operator=(const ImTextureVisualizer&) = delete;
  ImTextureVisualizer(ImTextureVisualizer&&)                 = delete;
  ImTextureVisualizer& operator=(ImTextureVisualizer&&)      = delete;

  VkResult init(const InitInfo& info);
  void     deinit();
  bool     isValid() const { return m_pipeline != VK_NULL_HANDLE; }

  // Adds an item of `size` pixels at the current cursor position (like
  // `ImGui::Image`) that displays the `Settings::uvRect` part of `view`.
  // `viewType` must be the type `view` was created with; a 2D array view shows
  // `Settings::layer` and a 3D view shows the `Settings::zCoord` slice.
  void image(VkImageView     view,
             const ImVec2&   size,
             const Settings& settings,
             VkImageLayout   imageLayout = VK_IMAGE_LAYOUT_GENERAL,
             VkImageViewType viewType    = VK_IMAGE_VIEW_TYPE_2D);

  // ImGui controls for `settings`, returns true when something was modified.
  // `mipLevels` and `layerCount` describe the displayed image and bound the lod
  // and layer sliders. `viewType` decides which of the slice controls is shown:
  // the array layer slider for _2D_ARRAY, the z coordinate slider for _3D.
  static bool settingsWidget(Settings& settings, uint32_t mipLevels = 1, uint32_t layerCount = 1, VkImageViewType viewType = VK_IMAGE_VIEW_TYPE_2D);

  // Push constants of the embedded shader, must match `imgui_texture_visualizer.slang`
  struct PushConstants
  {
    glm::vec4 posRect{0.0f, 0.0f, 0.0f, 0.0f};  // clip space rectangle, xy = min corner, zw = max corner
    glm::vec4 uvRect{0.0f, 0.0f, 1.0f, 1.0f};   // Settings::uvRect, at posRect.xy and at posRect.zw
    glm::vec4 powExp{1.0f, 1.0f, 1.0f, 1.0f};   // Settings::pow
    glm::vec4 scale{1.0f, 1.0f, 1.0f, 1.0f};    // Settings::scale
    glm::vec4 add{0.0f, 0.0f, 0.0f, 0.0f};      // Settings::add
    uint32_t  swizzle{0};                       // Settings::swizzle, one byte per output channel
    uint32_t  invert{0};                        // Settings::invertChannel, one bit per output channel
    float     lod{0.0f};                        // Settings::lod
    uint32_t  layer{0};                         // Settings::layer, only read by the 2D array variant
    float     zCoord{0.0f};                     // Settings::zCoord, only read by the 3D variant
  };
  static_assert(sizeof(PushConstants) == 100, "PushConstants must match the layout of the shader");

private:
  // Everything the draw callback needs, a copy of this is stored in the ImGui draw list
  struct DrawCallbackData
  {
    PushConstants    pushConstants{};
    VkPipeline       pipeline{VK_NULL_HANDLE};
    VkPipelineLayout pipelineLayout{VK_NULL_HANDLE};
    VkImageView      imageView{VK_NULL_HANDLE};
    VkSampler        sampler{VK_NULL_HANDLE};
    VkImageLayout    imageLayout{VK_IMAGE_LAYOUT_GENERAL};
    VkViewport       viewport{};
    VkRect2D         scissor{};
  };

  VkResult createResources(const InitInfo& info);

  static void drawCallback(const ImDrawList* drawList, const ImDrawCmd* drawCmd);

  VkDevice              m_device{VK_NULL_HANDLE};
  VkDescriptorSetLayout m_descriptorSetLayout{VK_NULL_HANDLE};
  VkPipelineLayout      m_pipelineLayout{VK_NULL_HANDLE};
  VkPipeline            m_pipeline{VK_NULL_HANDLE};       // samples a 2D view
  VkPipeline            m_pipelineArray{VK_NULL_HANDLE};  // samples a layer of a 2D array view
  VkPipeline            m_pipeline3D{VK_NULL_HANDLE};     // samples a slice of a 3D view
  VkSampler             m_samplerLinear{VK_NULL_HANDLE};
  VkSampler             m_samplerNearest{VK_NULL_HANDLE};
};

}  // namespace nvapp
