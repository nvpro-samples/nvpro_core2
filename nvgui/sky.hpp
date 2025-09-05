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

#include "nvshaders/sky_io.h.slang"

#include "nvgui/property_editor.hpp"
#include "azimuth_sliders.hpp"


namespace nvgui {

inline bool skySimpleParametersUI(shaderio::SkySimpleParameters& params,
                                  const char*                    label = "PE::Table",
                                  ImGuiTableFlags flag = ImGuiTableFlags_BordersOuter | ImGuiTableFlags_Resizable)
{
  namespace PE = nvgui::PropertyEditor;

  bool changed{false};
  if(PE::begin(label, flag))
  {

    changed |= nvgui::azimuthElevationSliders(params.sunDirection, false, params.directionUp.y >= params.directionUp.z);
    changed |= PE::ColorEdit3("Color", &params.sunColor.x, ImGuiColorEditFlags_Float);
    changed |= PE::SliderFloat("Irradiance", &params.sunIntensity, 0.F, 100.F, "%.2f", ImGuiSliderFlags_Logarithmic);
    changed |= PE::SliderAngle("Angular Size", &params.angularSizeOfLight, 0.1F, 20.F);
    params.angularSizeOfLight = glm::clamp(params.angularSizeOfLight, glm::radians(0.1F), glm::radians(90.F));

    auto  square           = [](auto a) { return a * a; };
    float lightAngularSize = glm::clamp(params.angularSizeOfLight, glm::radians(0.1F), glm::radians(90.F));
    float lightSolidAngle  = 4.0F * glm::pi<float>() * square(sinf(lightAngularSize * 0.5F));
    float lightRadiance    = params.sunIntensity / lightSolidAngle;
    params.lightRadiance   = params.sunColor * lightRadiance;

    if(PE::treeNode("Extra"))
    {
      changed |= PE::SliderFloat("Brightness", &params.brightness, 0.F, 1.F);
      changed |= PE::SliderAngle("Glow Size", &params.glowSize, 0.F, 20.F);
      changed |= PE::SliderFloat("Glow Sharpness", &params.glowSharpness, 1.F, 10.F);
      changed |= PE::SliderFloat("Glow Intensity", &params.glowIntensity, 0.F, 1.F);
      changed |= PE::SliderAngle("Horizon Size", &params.horizonSize, 0.F, 90.F);
      changed |= PE::ColorEdit3("Sky Color", &params.skyColor.x, ImGuiColorEditFlags_Float);
      changed |= PE::ColorEdit3("Horizon Color", &params.horizonColor.x, ImGuiColorEditFlags_Float);
      changed |= PE::ColorEdit3("Ground Color", &params.groundColor.x, ImGuiColorEditFlags_Float);
      PE::treePop();
    }
    PE::end();
  }
  return changed;
}

inline bool skyPhysicalParameterUI(shaderio::SkyPhysicalParameters& params)
{
  namespace PE = nvgui::PropertyEditor;
  bool changed{false};
  if(PE::begin())
  {
    if(PE::entry("", [&] { return ImGui::SmallButton("reset"); }, "Default values"))
    {
      params  = shaderio::SkyPhysicalParameters();
      changed = true;
    }
    changed |= nvgui::azimuthElevationSliders(params.sunDirection, false, params.yIsUp == 1);
    changed |= PE::SliderFloat("Sun Disk Scale", &params.sunDiskScale, 0.F, 10.F);
    changed |= PE::SliderFloat("Sun Disk Intensity", &params.sunDiskIntensity, 0.F, 5.F);
    changed |= PE::SliderFloat("Sun Glow Intensity", &params.sunGlowIntensity, 0.F, 5.F);

    if(PE::treeNode("Extra"))
    {
      changed |= PE::SliderFloat("Haze", &params.haze, 0.F, 15.F);
      changed |= PE::SliderFloat("Red Blue Shift", &params.redblueshift, -1.F, 1.F);
      changed |= PE::SliderFloat("Saturation", &params.saturation, 0.F, 1.F);
      changed |= PE::SliderFloat("Horizon Height", &params.horizonHeight, -1.F, 1.F);
      changed |= PE::ColorEdit3("Ground Color", &params.groundColor.x, ImGuiColorEditFlags_Float);
      changed |= PE::SliderFloat("Horizon Blur", &params.horizonBlur, 0.F, 5.F);
      changed |= PE::ColorEdit3("Night Color", &params.nightColor.x, ImGuiColorEditFlags_Float);
      PE::treePop();
    }
    PE::end();
  }
  return changed;
}

}  // namespace nvgui