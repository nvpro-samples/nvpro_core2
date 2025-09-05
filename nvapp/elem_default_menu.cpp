/*
 * Copyright (c) 2024-2025, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */


#include <imgui/imgui.h>
#include <implot/implot.h>
#include <nvutils/logger.hpp>
#include <nvgui/fonts.hpp>

#include "elem_default_menu.hpp"

// Uncomment to show ImGui Demo
// #define SHOW_IMGUI_DEMO 1

void nvapp::ElementDefaultMenu::onAttach(nvapp::Application* app)
{
  LOGI("Adding Default Menu\n");
  m_app = app;
}

void nvapp::ElementDefaultMenu::onUIMenu()
{
  static bool close_app{false};
  bool        v_sync = m_app->isVsync();
#ifdef SHOW_IMGUI_DEMO
  static bool s_showDemo{false};
  static bool s_showDemoPlot{false};
#endif
  if(ImGui::BeginMenu("File"))
  {
    if(ImGui::MenuItem(ICON_MS_POWER_SETTINGS_NEW " Exit", "Ctrl+Q"))
    {
      close_app = true;
    }
    ImGui::EndMenu();
  }
  if(ImGui::BeginMenu("View"))
  {
    ImGui::MenuItem(ICON_MS_BOTTOM_PANEL_OPEN " V-Sync", "Ctrl+Shift+V", &v_sync);
    ImGui::EndMenu();
  }
#ifdef SHOW_IMGUI_DEMO
  if(ImGui::BeginMenu("ImGui-Debug"))
  {
    ImGui::MenuItem("Show ImGui Demo", nullptr, &s_showDemo);
    ImGui::MenuItem("Show ImPlot Demo", nullptr, &s_showDemoPlot);
    ImGui::EndMenu();
  }
#endif  // SHOW_IMGUI_DEMO

  // Shortcuts
  if(ImGui::IsKeyPressed(ImGuiKey_Q) && ImGui::IsKeyDown(ImGuiKey_LeftCtrl))
  {
    close_app = true;
  }

  if(ImGui::IsKeyPressed(ImGuiKey_V) && ImGui::IsKeyDown(ImGuiKey_LeftCtrl) && ImGui::IsKeyDown(ImGuiKey_LeftShift))
  {
    v_sync = !v_sync;
  }

  if(close_app)
  {
    m_app->close();
  }
#ifdef SHOW_IMGUI_DEMO
  if(s_showDemo)
  {
    ImGui::ShowDemoWindow(&s_showDemo);
  }
  if(s_showDemoPlot)
  {
    ImPlot::ShowDemoWindow(&s_showDemoPlot);
  }
#endif  // SHOW_IMGUI_DEMO

  if(m_app->isVsync() != v_sync)
  {
    m_app->setVsync(v_sync);
  }
}
