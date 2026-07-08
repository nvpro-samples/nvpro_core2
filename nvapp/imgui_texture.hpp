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

#include <vulkan/vulkan_core.h>

#include <imgui/imgui.h>
#include <imgui/backends/imgui_impl_vulkan.h>

namespace nvapp {

//---------------------------------------------------------------------------
// Wrapper around an ImGui (Vulkan backend) texture descriptor set.
//
// Turns any VkImageView into something that can be displayed with ImGui::Image
// (or ImGui::ImageButton). Internally calls ImGui_ImplVulkan_AddTexture /
// RemoveTexture, allocating from whatever descriptor pool the ImGui Vulkan
// backend was initialised with (e.g. nvapp::Application's UI descriptor pool).
//
// This lives in nvapp (not nvgui) because it depends on Vulkan and the ImGui
// Vulkan backend; nvapp is the layer that already bridges Vulkan + ImGui, while
// nvgui stays graphics-API agnostic.
//
// Lifecycle (matches nvpro_core2 convention: explicit init/deinit, no RAII):
//   * Default-construct freely (no Vulkan calls). Safe before ImGui init.
//   * init(view, layout)  -- allocate the descriptor. Requires the helper to
//                            be empty (asserts) and view to be non-null
//                            (asserts). Must be called AFTER
//                            ImGui_ImplVulkan_Init has run.
//   * deinit()            -- free the descriptor if any (no-op when empty).
//                            Must be called before the object's destructor.
//   * update(view, layout)-- convenience shortcut equivalent to
//                            deinit(); init(view, layout);
//                            Use on resize.
//   * The destructor asserts that the object has already been deinit'd.
//   * Non-copyable and non-movable (ownership transfers are easy to lose track
//     of when debugging GPU lifetimes).
//   * Single-thread (same constraint as ImGui_ImplVulkan_AddTexture).
//
// Usage:
//   nvapp::ImTexture image;
//   image.init(renderTarget.getUiImageView());          // once ImGui is up
//   ...
//   ImGui::Image(image, ImGui::GetContentRegionAvail()); // implicit ImTextureRef
//   ...
//   image.update(renderTarget.getUiImageView());        // on resize
//   ...
//   image.deinit();                                      // before destruction
//---------------------------------------------------------------------------
class ImTexture
{
public:
  ImTexture() = default;
  ~ImTexture() { assert(m_set == VK_NULL_HANDLE && "Missing deinit()"); }

  ImTexture(const ImTexture&)            = delete;
  ImTexture& operator=(const ImTexture&) = delete;
  ImTexture(ImTexture&&)                 = delete;
  ImTexture& operator=(ImTexture&&)      = delete;

  // Allocate a descriptor set for the given view. Requires empty state and a non-null view.
  void init(VkImageView view, VkImageLayout layout = VK_IMAGE_LAYOUT_GENERAL)
  {
    assert(m_set == VK_NULL_HANDLE && "Missing deinit()");
    assert(view != VK_NULL_HANDLE && "ImTexture::init requires a valid VkImageView");
    m_set = ImGui_ImplVulkan_AddTexture(view, layout);
  }

  // Free the descriptor set if any. No-op when empty.
  void deinit()
  {
    if(m_set != VK_NULL_HANDLE)
    {
      ImGui_ImplVulkan_RemoveTexture(m_set);
      m_set = VK_NULL_HANDLE;
    }
  }

  // Convenience: deinit() + init(view, layout). Use on resize.
  void update(VkImageView view, VkImageLayout layout = VK_IMAGE_LAYOUT_GENERAL)
  {
    deinit();
    init(view, layout);
  }

  operator ImTextureRef() const { return ImTextureRef(reinterpret_cast<ImTextureID>(m_set)); }

private:
  VkDescriptorSet m_set{VK_NULL_HANDLE};
};

}  // namespace nvapp
