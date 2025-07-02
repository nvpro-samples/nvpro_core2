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
#include <filesystem>
#include <memory>

#include <glm/glm.hpp>
#include <nvutils/camera_manipulator.hpp>

namespace nvgui {

/*  @DOC_START -------------------------------------------------------
 
  # functions in ImGuiH
  
  - CameraWidget : CameraWidget is a Camera widget for the the Camera Manipulator
  - SetCameraJsonFile : set the name (without .json) of the setting file. It will load and replace all camera and settings
  - SetHomeCamera : set the home camera - replace the one on load
  - AddCamera : adding a camera to the list of cameras

 --- @DOC_END ------------------------------------------------------- */

bool CameraWidget(std::shared_ptr<nvutils::CameraManipulator> cameraManip);
void SetCameraJsonFile(const std::filesystem::path& filename);
void SetHomeCamera(const nvutils::CameraManipulator::Camera& camera);
void AddCamera(const nvutils::CameraManipulator::Camera& camera);

}  // namespace nvgui
