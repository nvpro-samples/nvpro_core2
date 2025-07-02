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

#include <nvutils/logger.hpp>
#include <nvgui/window.hpp>

#include "elem_camera.hpp"

void nvapp::ElementCamera::updateCamera(std::shared_ptr<nvutils::CameraManipulator> m_cameraManip, ImGuiWindow* viewportWindow)
{
  nvutils::CameraManipulator::Inputs inputs;  // Mouse and keyboard inputs

  m_cameraManip->updateAnim();  // This makes the camera to transition smoothly to the new position

  // Check if the mouse cursor is over the "Viewport", check for all inputs that can manipulate the camera.
  if(!nvgui::isWindowHovered(viewportWindow))
    return;

  // measure one frame at a time
  float keyFactor = ImGui::GetIO().DeltaTime * 50.0F;

  inputs.lmb      = ImGui::IsMouseDown(ImGuiMouseButton_Left);
  inputs.rmb      = ImGui::IsMouseDown(ImGuiMouseButton_Right);
  inputs.mmb      = ImGui::IsMouseDown(ImGuiMouseButton_Middle);
  inputs.ctrl     = ImGui::IsKeyDown(ImGuiKey_LeftCtrl) || ImGui::IsKeyDown(ImGuiKey_RightCtrl);
  inputs.shift    = ImGui::IsKeyDown(ImGuiKey_LeftShift) || ImGui::IsKeyDown(ImGuiKey_RightShift);
  inputs.alt      = ImGui::IsKeyDown(ImGuiKey_LeftAlt) || ImGui::IsKeyDown(ImGuiKey_RightAlt);
  ImVec2 mousePos = ImGui::GetMousePos();

  // For all pressed keys - apply the action
  m_cameraManip->keyMotion(0, 0, nvutils::CameraManipulator::NoAction);

  // None of the modifiers should be pressed for the single key: WASD and arrows
  if(!(inputs.ctrl || inputs.alt || inputs.shift))
  {
    if(ImGui::IsKeyDown(ImGuiKey_W))
    {
      m_cameraManip->keyMotion(keyFactor, 0, nvutils::CameraManipulator::Dolly);
    }

    if(ImGui::IsKeyDown(ImGuiKey_S))
    {
      m_cameraManip->keyMotion(-keyFactor, 0, nvutils::CameraManipulator::Dolly);
    }

    if(ImGui::IsKeyDown(ImGuiKey_D) || ImGui::IsKeyDown(ImGuiKey_RightArrow))
    {
      m_cameraManip->keyMotion(keyFactor, 0, nvutils::CameraManipulator::Pan);
    }

    if(ImGui::IsKeyDown(ImGuiKey_A) || ImGui::IsKeyDown(ImGuiKey_LeftArrow))
    {
      m_cameraManip->keyMotion(-keyFactor, 0, nvutils::CameraManipulator::Pan);
    }

    if(ImGui::IsKeyDown(ImGuiKey_UpArrow))
    {
      m_cameraManip->keyMotion(0, keyFactor, nvutils::CameraManipulator::Pan);
    }

    if(ImGui::IsKeyDown(ImGuiKey_DownArrow))
    {
      m_cameraManip->keyMotion(0, -keyFactor, nvutils::CameraManipulator::Pan);
    }
  }

  if(ImGui::IsMouseClicked(ImGuiMouseButton_Left) || ImGui::IsMouseClicked(ImGuiMouseButton_Middle)
     || ImGui::IsMouseClicked(ImGuiMouseButton_Right))
  {
    m_cameraManip->setMousePosition(static_cast<int>(mousePos.x), static_cast<int>(mousePos.y));
  }

  if(ImGui::IsMouseDragging(ImGuiMouseButton_Left, 1.0F) || ImGui::IsMouseDragging(ImGuiMouseButton_Middle, 1.0F)
     || ImGui::IsMouseDragging(ImGuiMouseButton_Right, 1.0F))
  {
    m_cameraManip->mouseMove(static_cast<int>(mousePos.x), static_cast<int>(mousePos.y), inputs);
  }

  // Mouse Wheel
  if(ImGui::GetIO().MouseWheel != 0.0F)
  {
    m_cameraManip->wheel(static_cast<int>(ImGui::GetIO().MouseWheel * 3), inputs);
  }
}

void nvapp::ElementCamera::onAttach(nvapp::Application* app)
{
  LOGI("Adding Camera Manipulator\n");
}

void nvapp::ElementCamera::onUIRender()
{
  assert(m_cameraManip && "Missing setCamera");
  updateCamera(m_cameraManip, ImGui::FindWindowByName("Viewport"));
}

void nvapp::ElementCamera::onResize(VkCommandBuffer cmd, const VkExtent2D& size)
{
  assert(m_cameraManip && "Missing setCamera");
  m_cameraManip->setWindowSize({size.width, size.height});
}
