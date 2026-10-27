/*
 * Copyright (c) 2022-2026, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.  All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <imgui/imgui.h>
#include <imgui/imgui_internal.h>
#include <nvutils/camera_manipulator.hpp>

#include "application.hpp"

/*-------------------------------------------------------------------------------------------------
# class nvdx12app::ElementCamera

This class is an element of the application that is responsible for the camera manipulation. 
It is using the `nvutils::CameraManipulator` to handle the camera movement and interaction.
Mirrors nvapp::ElementCamera, for nvdx12app::Application.

To use this class, you need to add it to the `nvdx12app::Application` using the `addElement` method.

-------------------------------------------------------------------------------------------------*/


namespace nvdx12app {

struct ElementCamera : public nvdx12app::IAppElement
{
  using nvdx12app::IAppElement::onResize;

  ElementCamera(std::shared_ptr<nvutils::CameraManipulator> camera = nullptr) { m_cameraManip = camera; }

  void setCameraManipulator(std::shared_ptr<nvutils::CameraManipulator>& pCamera) { m_cameraManip = pCamera; }
  void onAttach(nvdx12app::Application* app) override;
  void onUIRender() override;
  void onResize(const glm::uvec2& size) override;

  std::shared_ptr<nvutils::CameraManipulator> getCameraManipulator() const { return m_cameraManip; }

  // Can be called independently
  static void updateCamera(std::shared_ptr<nvutils::CameraManipulator> m_cameraManip, ImGuiWindow* viewportWindow);

  static double wheelSpeed;

private:
  std::shared_ptr<nvutils::CameraManipulator> m_cameraManip{};
};

}  // namespace nvdx12app
