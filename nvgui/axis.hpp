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

#pragma once

#include <imgui/imgui.h>
#include <glm/glm.hpp>

/*  @DOC_START -------------------------------------------------------
 
Function `Axis(ImVec2 pos, const glm::mat4& modelView, float size = 20.f)`
which display right-handed axis in a ImGui window.

Example

```cpp
{  // Display orientation axis at the bottom left corner of the window
  float  axisSize = 25.F;
  ImVec2 pos      = ImGui::GetWindowPos();
  pos.y += ImGui::GetWindowSize().y;
  pos += ImVec2(axisSize * 1.1F, -axisSize * 1.1F) * ImGui::GetWindowDpiScale();  // Offset
  ImGuiH::Axis(pos, CameraManip.getMatrix(), axisSize);
}
```      

--- @DOC_END ------------------------------------------------------- */

// The API
namespace nvgui {

// This utility is adding the 3D axis at `pos`, using the matrix `modelView`
void Axis(ImVec2 pos, const glm::mat4& modelView, float size = 20.f);

// Place the axis at the bottom left corner of the window
inline void Axis(const glm::mat4& modelView, float size = 20.f)
{
  ImVec2 windowPos  = ImGui::GetWindowPos();
  ImVec2 windowSize = ImGui::GetWindowSize();
  ImVec2 offset     = ImVec2(size * 1.1F * ImGui::GetWindowDpiScale(), -size * 1.1F * ImGui::GetWindowDpiScale());
  ImVec2 pos        = ImVec2(windowPos.x + offset.x, windowPos.y + windowSize.y + offset.y);
  Axis(pos, modelView, size);
}


};  // namespace nvgui
