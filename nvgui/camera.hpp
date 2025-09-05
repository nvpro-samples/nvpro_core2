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

// ImGui camera controls and camera preset management.

#pragma once
#include <filesystem>
#include <memory>

#include <nvutils/camera_manipulator.hpp>

namespace nvgui {

// Bitset enum for controlling which camera widget sections are open by default
enum CameraWidgetSections : uint32_t
{
  CameraSection_Position   = 1 << 0,  // Position section (eye, center, up)
  CameraSection_Projection = 1 << 1,  // Projection section (FOV, clip planes)
  CameraSection_Other      = 1 << 2,  // Other section (up vector, transition)

  // Convenience combinations
  CameraSection_None    = 0,
  CameraSection_All     = CameraSection_Position | CameraSection_Projection | CameraSection_Other,
  CameraSection_Default = CameraSection_Projection  // Current behavior - only projection open
};

// Shows GUI for nvutils::CameraManipulator.
// If `embed` is true, it will have text before it and appear in ImGui::BeginChild.
// `openSections` controls which sections are open by default.
// Returns whether camera parameters changed.
bool CameraWidget(std::shared_ptr<nvutils::CameraManipulator> cameraManip,
                  bool                                        embed        = false,
                  CameraWidgetSections                        openSections = CameraSection_Default);

// Sets the name (without .json) of the setting file. It will load and replace all cameras and settings
void SetCameraJsonFile(const std::filesystem::path& filename);
// Sets the home camera - replacing the one on load
void SetHomeCamera(const nvutils::CameraManipulator::Camera& camera);
// Adds a camera to the list of cameras
void AddCamera(const nvutils::CameraManipulator::Camera& camera);

}  // namespace nvgui
