/*
 * Copyright (c) 2022-2025, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

// Various Application utilities
// - Display a menu with File/Quit
// - Display basic information in the window title

#pragma once

#include <filesystem>
#include "nvgui/camera.hpp"
#include "nvvkgltf/scene.hpp"


namespace nvvkgltf {

// This function adds the camera to the camera manipulator
// It also sets the camera to the first camera in the list
// If there is no camera, it fits the camera to the scene
inline void addSceneCamerasToWidget(std::shared_ptr<nvutils::CameraManipulator> cameraManip,
                                    const std::filesystem::path&                filename,
                                    const std::vector<nvvkgltf::RenderCamera>&  cameras,
                                    const nvutils::Bbox&                        sceneBbox)
{
  nvgui::SetCameraJsonFile(filename.stem());
  if(!cameras.empty())
  {
    const auto& camera = cameras[0];
    cameraManip->setCamera(
        {camera.eye, camera.center, camera.up, static_cast<float>(glm::degrees(camera.yfov)), {camera.znear, camera.zfar}});
    nvgui::SetHomeCamera({camera.eye, camera.center, camera.up, static_cast<float>(glm::degrees(camera.yfov))});

    for(const auto& cam : cameras)
    {
      nvgui::AddCamera({cam.eye, cam.center, cam.up, static_cast<float>(glm::degrees(cam.yfov)), {camera.znear, camera.zfar}});
    }
  }
  else
  {
    // Re-adjusting camera to fit the new scene
    cameraManip->fit(sceneBbox.min(), sceneBbox.max(), true);
    cameraManip->setClipPlanes(glm::vec2(0.001F * sceneBbox.radius(), 100.0F * sceneBbox.radius()));
    nvgui::SetHomeCamera(cameraManip->getCamera());
  }
}


}  // namespace nvvkgltf
