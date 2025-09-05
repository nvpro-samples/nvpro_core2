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


#include "elem_logger.hpp"
#include <nvgui/fonts.hpp>


nvapp::ElementLogger::ElementLogger(bool show /*= false*/)
    : m_showLog(show)
{
}

void nvapp::ElementLogger::onAttach(Application* /*app*/)
{
  LOGI("Adding Logger UI\n");

  m_settingsHandler.setHandlerName("ElementLogger");
  m_settingsHandler.setSetting("ShowLog", &m_showLog);
  m_settingsHandler.setSetting("LogLevel", &m_levelFilter);
  m_settingsHandler.addImGuiHandler();
}

void nvapp::ElementLogger::onUIRender()
{
  if(ImGui::IsKeyChordPressed(ImGuiMod_Ctrl | ImGuiKey_ModShift | ImGuiKey_L))
  {
    m_showLog = !m_showLog;
  }

  if(!m_showLog)
  {
    return;
  }

  ImGui::SetNextWindowCollapsed(false, ImGuiCond_Appearing);
  ImGui::SetNextWindowSize({400, 200}, ImGuiCond_Appearing);
  ImGui::SetNextWindowBgAlpha(0.7F);
  draw("Log", &m_showLog);
}

void nvapp::ElementLogger::onUIMenu()
{
  if(ImGui::BeginMenu("View"))
  {
    ImGui::MenuItem(ICON_MS_TEXT_AD " Log Window", "Ctrl+Shift+L", &m_showLog);
    ImGui::EndMenu();
  }
}


void nvapp::ElementLogger::setLevelFilter(uint32_t levelFilter)
{
  std::lock_guard<std::mutex> lock(m_modificationMutex);
  m_levelFilter = levelFilter;
}

void nvapp::ElementLogger::clear()
{
  m_buf.clear();
  m_lineOffsets.clear();
  m_lineOffsets.push_back(0);
  m_lineLevels.clear();
}

void nvapp::ElementLogger::addLog(uint32_t level, const char* fmt, ...)
{
  std::lock_guard<std::mutex> lock(m_modificationMutex);

  if((m_levelFilter & (1 << level)) == 0)
    return;

  int     old_size = m_buf.size();
  va_list args     = {};
  va_start(args, fmt);
  m_buf.appendfv(fmt, args);
  va_end(args);
  for(int new_size = m_buf.size(); old_size < new_size; old_size++)
  {
    if(m_buf[old_size] == '\n')
    {
      m_lineOffsets.push_back(old_size + 1);
      m_lineLevels.push_back(level);
    }
  }
}

void nvapp::ElementLogger::initColors()
{
  ImGuiContext& g = *GImGui;
  m_colors.resize(8);
  m_colors[nvutils::Logger::LogLevel::eINFO]    = g.Style.Colors[ImGuiCol_Text];  // Default text color
  m_colors[nvutils::Logger::LogLevel::eWARNING] = ImVec4(1.0, 0.5, 0.0, 1.0);     // Orange
  m_colors[nvutils::Logger::LogLevel::eERROR]   = ImVec4(1.0, 0.0, 0.0, 1.0);     // Bright Red
  m_colors[nvutils::Logger::LogLevel::eDEBUG]   = ImVec4(0.5, 0.5, 1.0, 1.0);     // Light Blue
  m_colors[nvutils::Logger::LogLevel::eSTATS]   = ImVec4(0.0, 0.75, 0.0, 1.0);    // Light Green
  m_colors[nvutils::Logger::LogLevel::eOK]      = ImVec4(0.0, 1.0, 0.0, 1.0);     // Green
}

void nvapp::ElementLogger::draw(const char* title, bool* p_open /*= nullptr*/)
{
  if(ImGui::GetCurrentContext() == nullptr)
    return;
  if(m_colors.empty())  // Initialize colors late, as we need the ImGui context
    initColors();

  if(!ImGui::Begin(title, p_open))
  {
    ImGui::End();
    return;
  }

  // Options menu
  if(ImGui::BeginPopup("Options"))
  {
    ImGui::Checkbox("Auto-scroll", &m_autoScroll);
    ImGui::EndPopup();
  }

  // Main window
  if(ImGui::Button("Options"))
  {
    ImGui::OpenPopup("Options");
  }
  ImGui::SameLine();
  bool do_clear = ImGui::Button("Clear");
  ImGui::SameLine();
  bool copy = ImGui::Button("Copy");
  ImGui::SameLine();
  ImGui::CheckboxFlags("All", &m_levelFilter, eBitAll);
  ImGui::SameLine();
  ImGui::CheckboxFlags("Stats", &m_levelFilter, eBitSTATS);
  ImGui::SameLine();
  ImGui::CheckboxFlags("Debug", &m_levelFilter, eBitDEBUG);
  ImGui::SameLine();
  ImGui::CheckboxFlags("Info", &m_levelFilter, eBitINFO);
  ImGui::SameLine();
  ImGui::CheckboxFlags("Warnings", &m_levelFilter, eBitWARNING);
  ImGui::SameLine();
  ImGui::CheckboxFlags("Errors", &m_levelFilter, eBitERROR);
  ImGui::SameLine();
  ImGui::Text("Filter");
  ImGui::SameLine();
  m_filter.Draw("##Filter", -100.0F);
  ImGui::SameLine();
  bool clear_filter = ImGui::SmallButton("X");

  ImGui::Separator();
  ImGui::BeginChild("scrolling", ImVec2(0, 0), false, ImGuiWindowFlags_HorizontalScrollbar);

  if(do_clear)
  {
    clear();
  }
  if(copy)
  {
    ImGui::SetClipboardText(m_buf.c_str());
  }
  if(clear_filter)
  {
    m_filter.Clear();
  }


  ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(0, 0));
  const char* buf     = m_buf.begin();
  const char* buf_end = m_buf.end();
  if(m_filter.IsActive())
  {
    // In this example we don't use the clipper when Filter is enabled.
    // This is because we don't have a random access on the result on our filter.
    // A real application processing logs with ten of thousands of entries may want to store the result of
    // search/filter.. especially if the filtering function is not trivial (e.g. reg-exp).
    for(int line_no = 0; line_no < m_lineOffsets.Size; line_no++)
    {
      const char*   line_start = buf + m_lineOffsets[line_no];
      const char*   line_end   = (line_no + 1 < m_lineOffsets.Size) ? (buf + m_lineOffsets[line_no + 1] - 1) : buf_end;
      const int32_t level      = line_no < m_lineLevels.Size ? m_lineLevels[line_no] : 0;
      if(m_filter.PassFilter(line_start, line_end))
      {
        // Setting the color of the line
        ImGuiContext& g               = *GImGui;
        ImVec4        backupTextColor = g.Style.Colors[ImGuiCol_Text];
        g.Style.Colors[ImGuiCol_Text] = m_colors[level];

        ImGui::TextUnformatted(line_start, line_end);

        g.Style.Colors[ImGuiCol_Text] = backupTextColor;  // restore color
      }
    }
  }
  else
  {
    // This is from ImGui::demo, check for details in the ImGui::demo
    ImGuiListClipper clipper;
    clipper.Begin(m_lineOffsets.Size);
    while(clipper.Step())
    {
      for(int line_no = clipper.DisplayStart; line_no < clipper.DisplayEnd; line_no++)
      {
        const char*   line_start = buf + m_lineOffsets[line_no];
        const char*   line_end = (line_no + 1 < m_lineOffsets.Size) ? (buf + m_lineOffsets[line_no + 1] - 1) : buf_end;
        const int32_t level    = line_no < m_lineLevels.Size ? m_lineLevels[line_no] : 0;

        // Setting the color of the line
        ImGuiContext& g               = *GImGui;
        ImVec4        backupTextColor = g.Style.Colors[ImGuiCol_Text];
        g.Style.Colors[ImGuiCol_Text] = m_colors[level];

        ImGui::TextUnformatted(line_start, line_end);

        g.Style.Colors[ImGuiCol_Text] = backupTextColor;  // restore color
      }
    }
    clipper.End();
  }
  ImGui::PopStyleVar();

  if(m_autoScroll && ImGui::GetScrollY() >= ImGui::GetScrollMaxY())
  {
    ImGui::SetScrollHereY(1.0F);
  }

  ImGui::EndChild();
  ImGui::End();
}
