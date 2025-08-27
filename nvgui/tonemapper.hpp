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

#include <nvshaders/tonemap_io.h.slang>

#include <nvgui/property_editor.hpp>


namespace nvgui {
namespace PE = nvgui::PropertyEditor;

inline bool tonemapperWidget(shaderio::TonemapperData& tonemapper)
{
  bool changed{false};

  const char* items[] = {"Filmic", "Uncharted 2", "Clip", "ACES", "AgX", "Khronos PBR"};

  if(PE::begin())
  {
    changed |= PE::Combo("Method", &tonemapper.method, items, IM_ARRAYSIZE(items));
    changed |= PE::Checkbox("Active", reinterpret_cast<bool*>(&tonemapper.isActive));
    ImGui::BeginDisabled(!tonemapper.isActive);

    changed |= PE::SliderFloat("Exposure", &tonemapper.exposure, 0.1F, 200.0F, "%.3f", ImGuiSliderFlags_Logarithmic);
    changed |= PE::SliderFloat("Brightness", &tonemapper.brightness, 0.0F, 2.0F);
    changed |= PE::SliderFloat("Contrast", &tonemapper.contrast, 0.0F, 2.0F);
    changed |= PE::SliderFloat("Saturation", &tonemapper.saturation, 0.0F, 2.0F);
    changed |= PE::SliderFloat("Vignette", &tonemapper.vignette, 0.0F, 1.0F);
    changed |= PE::Checkbox("Auto Exposure", reinterpret_cast<bool*>(&tonemapper.autoExposure));
    if(tonemapper.autoExposure)
    {
      ImGui::Indent();
      PE::Combo("Average Mode", (int*)&tonemapper.averageMode, "Mean\0Median");

      PE::DragFloat("Adaptation Speed", &tonemapper.autoExposureSpeed, 0.001f, 0.f, 100.f, "%.3f", ImGuiSliderFlags_AlwaysClamp);
      PE::DragFloat("Min (EV100)", &tonemapper.evMinValue, 0.01f, -24.f, 24.f);
      PE::DragFloat("Max (EV100)", &tonemapper.evMaxValue, 0.01f, -24.f, 24.f);

      PE::Checkbox("Center Weighted Metering", (bool*)&tonemapper.enableCenterMetering);
      ImGui::BeginDisabled(!tonemapper.enableCenterMetering);
      PE::DragFloat("Center Metering Size", &tonemapper.centerMeteringSize, 0.01f, 0.01f, 1.0f);
      ImGui::EndDisabled();
    }

    ImGui::EndDisabled();
    if(ImGui::SmallButton("reset"))
    {
      tonemapper = {};
      changed    = true;
    }
    PE::end();
  }
  return changed;
}


}  // namespace nvgui