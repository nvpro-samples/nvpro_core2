/*
 * Copyright (c) 2023-2026, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */


#pragma once

#include <algorithm>

#include <imgui/imgui.h>

namespace nvgui {

// Mouse-wheel scrolling on the last submitted ImGui item, useful e.g.
// to cycle through a combo without opening the popup:
//   ImGui::Combo("Value", &comboItem, items, itemsCount);
//   nvgui::hoverScrolling(comboItem, 0, itemsCount - 1);
template <typename T>
bool hoverScrolling(T& data, T minVal, T maxVal, float inc = 1.0F)
{
  const float wheel = ImGui::GetIO().MouseWheel;
  
  assert(minVal <= maxVal);

  // No scrolling if not hovered or no wheel delta
  if(!ImGui::IsItemHovered() || wheel == 0.0F)
    return false;

  // Claim the wheel event so the parent scrollable region doesn't consume it
  ImGui::SetItemKeyOwner(ImGuiKey_MouseWheelY);
  
  // Scroll the data
  const T oldData = data;
  data = std::clamp(data + T(wheel * inc), minVal, maxVal);
  
  // Return true if the data was changed
  return data != oldData;
}

}  // namespace nvgui
