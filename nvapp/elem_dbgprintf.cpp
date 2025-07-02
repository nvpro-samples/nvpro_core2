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

#include <volk.h>

#include <string_view>

#include <imgui_internal.h>
#include <nvvk/check_error.hpp>
#include <nvutils/logger.hpp>

#include "elem_dbgprintf.hpp"


glm::vec2 nvapp::ElementDbgPrintf::getMouseCoord()
{
  // Pick the mouse coordinate if the mouse is down
  if(ImGui::IsMouseClicked(ImGuiMouseButton_Left, true))
  {
    ImGuiWindow* window = ImGui::FindWindowByName("Viewport");
    if(window == nullptr)
      return {-1, -1};
    const glm::vec2 mouse_pos = {ImGui::GetMousePos().x, ImGui::GetMousePos().y};  // Current mouse pos in window
    const glm::vec2 corner    = {window->Pos.x, window->Pos.y};                    // Corner of the viewport
    return (mouse_pos - corner);
  }
  return {-1, -1};
}

void nvapp::ElementDbgPrintf::onAttach(Application* app)
{
  m_instance = app->getInstance();
  // Vulkan message callback - for receiving the printf in the shader
  // Note: there is already a callback in nvvk::Context, but by defaut it is not printing INFO severity
  //       this callback will catch the message and will make it clean for display.
  auto dbg_messenger_callback = [](VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity, VkDebugUtilsMessageTypeFlagsEXT messageType,
                                   const VkDebugUtilsMessengerCallbackDataEXT* callbackData, void* userData) -> VkBool32 {
    // Get rid of all the extra message we don't need
    std::string_view clean_msg = callbackData->pMessage;

    // Special ID for the printf message
    // see: https://github.com/KhronosGroup/Vulkan-ValidationLayers/blob/main/docs/debug_printf.md#debug-printf-output
    if(callbackData->messageIdNumber != 0x4fe1fef9)
      return VK_FALSE;

    // Define the prefix to remove
    const std::string_view prefix = "Printf:\n";
    // Check if the message starts with the prefix and remove it
    const size_t prefix_location = clean_msg.rfind(prefix);
    if(prefix_location != std::string::npos)
    {
      clean_msg = clean_msg.substr(prefix_location + prefix.size());
    }

    const std::string_view delimiter = " | ";
    const size_t           pos       = clean_msg.rfind(delimiter);  // Remove everything before the last " | "
    if(pos != std::string::npos)
      clean_msg = clean_msg.substr(pos + delimiter.size());

    LOGI("%s", clean_msg.data());  // <- This will end up in the Logger; still null-terminated because we never remove suffixes
    return VK_FALSE;               // to continue
  };

  // Creating the callback
  VkDebugUtilsMessengerCreateInfoEXT dbg_messenger_create_info{VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT};
  dbg_messenger_create_info.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT;
  dbg_messenger_create_info.messageType     = VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT;
  dbg_messenger_create_info.pfnUserCallback = dbg_messenger_callback;
  NVVK_CHECK(vkCreateDebugUtilsMessengerEXT(m_instance, &dbg_messenger_create_info, nullptr, &m_dbgMessenger));
}

void nvapp::ElementDbgPrintf::onDetach()
{
  vkDestroyDebugUtilsMessengerEXT(m_instance, m_dbgMessenger, nullptr);
}
