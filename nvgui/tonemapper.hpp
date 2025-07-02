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

namespace shaderio {
using namespace glm;
#include <nvshaders/tonemap_io.h.slang>
}  // namespace shaderio

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
    changed |=
        PE::entry("Active", [&]() { return ImGui::Checkbox("Active", reinterpret_cast<bool*>(&tonemapper.isActive)); });
    changed |= PE::SliderFloat("Exposure", &tonemapper.exposure, 0.1F, 15.0F, "%.3f", ImGuiSliderFlags_Logarithmic);
    changed |= PE::SliderFloat("Brightness", &tonemapper.brightness, 0.0F, 2.0F);
    changed |= PE::SliderFloat("Contrast", &tonemapper.contrast, 0.0F, 2.0F);
    changed |= PE::SliderFloat("Saturation", &tonemapper.saturation, 0.0F, 2.0F);
    changed |= PE::SliderFloat("Vignette", &tonemapper.vignette, 0.0F, 1.0F);

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