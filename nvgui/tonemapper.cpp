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

#include "tonemapper.hpp"

#include <fmt/format.h>
#include <nvgui/IconsMaterialSymbols.h>
#include <nvgui/property_editor.hpp>
#include <nvshaders/tonemap_functions.h.slang>

namespace nvgui {
namespace PE = nvgui::PropertyEditor;

bool tonemapperWidget(shaderio::TonemapperData& tonemapper)
{
  bool changed{false};

  const char* items[] = {"Filmic", "Uncharted 2", "Clip", "ACES", "AgX", "Khronos PBR"};

  if(PE::begin())
  {
    changed |= PE::Combo("Method", &tonemapper.method, items, IM_ARRAYSIZE(items), 0,
                         "Tone mapping algorithm to compress high dynamic range (HDR) to standard dynamic range (SDR)");
    changed |= PE::Checkbox("Active", reinterpret_cast<bool*>(&tonemapper.isActive), "Enable/disable tone mapping post-processing");
    ImGui::BeginDisabled(!tonemapper.isActive);

    changed |= PE::SliderFloat("Exposure", &tonemapper.exposure, 0.1F, 200.0F, "%.3f", ImGuiSliderFlags_Logarithmic,
                               "Multiplier for input colors (0.1 = very dark, 1 = neutral, 200 = very bright)");

    // The temperature/tint sliders have nonzero defaults, so we have special
    // reset buttons for them alone.
    const float itemSpacing = 4.F;
    const float resetButtonWidth = ImGui::CalcTextSize(ICON_MS_RESET_WHITE_BALANCE).x + ImGui::GetStyle().FramePadding.x * 2.F;
    const float whiteBalanceSliderWidth = ImGui::GetContentRegionAvail().x - resetButtonWidth - itemSpacing;
    PE::entry(
        "Temperature",
        [&]() {
          ImGui::SetNextItemWidth(whiteBalanceSliderWidth);
          changed |= ImGui::SliderFloat("##Temperature", &tonemapper.temperature, 2000.0F, 15000.0F, "%.0f K");
          ImGui::SameLine(0, itemSpacing);
          ImGui::SetNextItemWidth(-FLT_MIN);
          if(ImGui::Button(ICON_MS_RESET_WHITE_BALANCE))
          {
            tonemapper.temperature = shaderio::TonemapperData().temperature;
            changed                = true;
          }
          return changed;
        },
        "Scene lighting temperature to correct for in degrees Kelvin "
        "(6506K = D65 neutral, higher values make the image more orange because they're correcting for cooler lighting)");

    PE::entry(
        "Tint",
        [&]() {
          ImGui::SetNextItemWidth(whiteBalanceSliderWidth);
          changed |= ImGui::SliderFloat("##Tint", &tonemapper.tint, -.03F, .03F, "%.5f");
          ImGui::SameLine(0, itemSpacing);
          ImGui::SetNextItemWidth(-FLT_MIN);
          if(ImGui::Button(ICON_MS_RESET_WHITE_BALANCE))
          {
            tonemapper.tint = shaderio::TonemapperData().tint;
            changed         = true;
          }
          return changed;
        },
        "Green/magenta lighting tint to correct for in ANSI C78.377-2008 Duv units "
        "(-.03 = very green, 0 = blackbody, .00326 = D65 neutral, .03 = very magenta)");

    changed |= PE::SliderFloat("Contrast", &tonemapper.contrast, 0.0F, 2.0F, "%.2f", 0,
                               "Scales colors away from gray (0 = no contrast, 1 = neutral, 2 = high contrast)");
    changed |= PE::SliderFloat("Brightness", &tonemapper.brightness, 0.0F, 2.0F, "%.2f", 0,
                               "Gamma curve for output colors (1 = neutral, higher values make midtones brighter)");
    changed |= PE::SliderFloat("Saturation", &tonemapper.saturation, 0.0F, 2.0F, "%.2f", 0,
                               "Controls color intensity (0 = grayscale, 1 = neutral, 2 = high saturation)");
    changed |= PE::SliderFloat("Vignette", &tonemapper.vignette, -1.0F, 1.0F, "%.2f", 0,
                               "Darkens image edges (-1 = very bright, 0 = none, 1 = very dark)");

    changed |= PE::Checkbox("Auto Exposure", reinterpret_cast<bool*>(&tonemapper.autoExposure),
                            "Automatically adjust exposure based on scene brightness");
    if(tonemapper.autoExposure)
    {
      ImGui::Indent();
      changed |= PE::Combo("Average Mode", (int*)&tonemapper.averageMode, "Mean\0Median", 0,
                           "Method for calculating scene brightness (Mean = average, Median = value where 50% of pixels are darker and 50% of pixels are brighter)");

      changed |= PE::DragFloat("Adaptation Speed", &tonemapper.autoExposureSpeed, 0.001f, 0.f, 100.f, "%.3f", ImGuiSliderFlags_AlwaysClamp,
                               "How quickly auto exposure adapts to lighting changes (higher = faster adaptation)");
      changed |= PE::DragFloat("Min (EV100)", &tonemapper.evMinValue, 0.01f, -24.f, 24.f, "%.2f", 0,
                               "Minimum histogram luminance in logarithmic stops (-24 = very dark, +24 = very bright)");
      changed |= PE::DragFloat("Max (EV100)", &tonemapper.evMaxValue, 0.01f, -24.f, 24.f, "%.2f", 0,
                               "Maximum histogram luminance in logarithmic stops (-24 = very dark, +24 = very bright)");

      changed |= PE::Checkbox("Center Weighted Metering", (bool*)&tonemapper.enableCenterMetering,
                              "Use center area for exposure calculation instead of full frame");
      ImGui::BeginDisabled(!tonemapper.enableCenterMetering);
      changed |= PE::DragFloat("Center Metering Size", &tonemapper.centerMeteringSize, 0.01f, 0.01f, 1.0f, "%.2f", 0,
                               "Size of center area for exposure calculation (0.01 = small spot, 1.0 = full frame)");
      ImGui::EndDisabled();
    }
    changed |= PE::Checkbox("Dither", reinterpret_cast<bool*>(&tonemapper.dither));

    ImGui::EndDisabled();
    if(ImGui::SmallButton("reset"))
    {
      tonemapper = {};
      changed    = true;
    }
    if(ImGui::IsItemHovered())
    {
      ImGui::SetTooltip("Reset all tonemapper settings to default values");
    }
    PE::end();
  }
  return changed;
}

}  // namespace nvgui
