/*
* Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
* SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
* SPDX-License-Identifier: Apache-2.0
*/

#include "window.hpp"

namespace nvgui {

bool isWindowHovered(ImGuiWindow* refWindow)
{
  if(!refWindow)
    return false;
  ImGuiContext& g = *ImGui::GetCurrentContext();
  if(g.HoveredWindow != refWindow)
    return false;
  if(!ImGui::IsWindowContentHoverable(refWindow, ImGuiFocusedFlags_RootWindow))
    return false;
  if(g.ActiveId != 0 && !g.ActiveIdAllowOverlap && g.ActiveId != refWindow->MoveId)
    return false;

  // Cancel if over the title bar
  {
    if(g.IO.ConfigWindowsMoveFromTitleBarOnly)
      if(!(refWindow->Flags & ImGuiWindowFlags_NoTitleBar) || refWindow->DockIsActive)
        if(refWindow->TitleBarRect().Contains(g.IO.MousePos))
          return false;
  }

  return true;
}
}  // namespace nvgui