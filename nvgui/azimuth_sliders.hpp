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

#include <glm/glm.hpp>
#include "nvgui/property_editor.hpp"


namespace nvgui {

inline bool azimuthElevationSliders(glm::vec3& direction, bool negative, bool yIsUp /*=true*/)
{
  glm::vec3 normalized_dir = normalize(direction);
  if(negative)
  {
    normalized_dir = -normalized_dir;
  }

  double       azimuth;
  double       elevation;
  const double min_azimuth   = -180.0;
  const double max_azimuth   = 180.0;
  const double min_elevation = -90.0;
  const double max_elevation = 90.0;

  if(yIsUp)
  {
    azimuth   = glm::degrees(atan2(normalized_dir.z, normalized_dir.x));
    elevation = glm::degrees(asin(normalized_dir.y));
  }
  else
  {
    azimuth   = glm::degrees(atan2(normalized_dir.y, normalized_dir.x));
    elevation = glm::degrees(asin(normalized_dir.z));
  }

  namespace PE = nvgui::PropertyEditor;
  bool changed = false;
  changed |= PE::SliderScalar("Azimuth", ImGuiDataType_Double, &azimuth, &min_azimuth, &max_azimuth, "%.1f deg",
                              ImGuiSliderFlags_NoRoundToFormat);
  changed |= PE::SliderScalar("Elevation", ImGuiDataType_Double, &elevation, &min_elevation, &max_elevation, "%.1f deg",
                              ImGuiSliderFlags_NoRoundToFormat);

  if(changed)
  {
    azimuth              = glm::radians(azimuth);
    elevation            = glm::radians(elevation);
    double cos_elevation = cos(elevation);

    if(yIsUp)
    {
      direction.y = static_cast<float>(sin(elevation));
      direction.x = static_cast<float>(cos(azimuth) * cos_elevation);
      direction.z = static_cast<float>(sin(azimuth) * cos_elevation);
    }
    else
    {
      direction.z = static_cast<float>(sin(elevation));
      direction.x = static_cast<float>(cos(azimuth) * cos_elevation);
      direction.y = static_cast<float>(sin(azimuth) * cos_elevation);
    }

    if(negative)
    {
      direction = -direction;
    }
  }

  return changed;
}


}  // namespace nvgui