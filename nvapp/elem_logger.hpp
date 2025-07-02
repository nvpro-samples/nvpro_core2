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

#include <sstream>

#include <imgui/imgui.h>
#include <imgui/imgui_internal.h>
#include <nvutils/logger.hpp>
#include "nvgui/settings_handler.hpp"

#include "application.hpp"

namespace nvapp {

/*-------------------------------------------------------------------------------------------------
# class nvapp::ElementLogger

>  This class is an element of the application that can redirect all logs to a ImGui window in the application

To use this class, you need to add it to the `nvapp::Application` using the `addElement` method.
  
Create the element such that it will be available to the target application

Example:
  ```cpp
  static nvapp::SampleAppLog g_logger;
  nvprintSetCallback([](int level, const char* fmt)
  {
    g_logger.addLog("%s", fmt);
  });
  
  app->addElement(std::make_unique<nvapp::ElementLogger>(&g_logger, true));  // Add logger window
  ```
  
-------------------------------------------------------------------------------------------------*/


// Helper class to show the log in the application
class ElementLogger : public nvapp::IAppElement
{
public:
  enum LogLevelBit
  {
    eBitDEBUG   = 1 << nvutils::Logger::LogLevel::eDEBUG,
    eBitINFO    = 1 << nvutils::Logger::LogLevel::eINFO,
    eBitWARNING = 1 << nvutils::Logger::LogLevel::eWARNING,
    eBitERROR   = 1 << nvutils::Logger::LogLevel::eERROR,
    eBitSTATS   = 1 << nvutils::Logger::LogLevel::eSTATS,
    eBitOK      = 1 << nvutils::Logger::LogLevel::eOK,
    eBitAll     = eBitDEBUG | eBitINFO | eBitWARNING | eBitERROR | eBitSTATS | eBitOK
  };


  explicit ElementLogger(bool show = false);
  virtual ~ElementLogger() = default;

  void onAttach(Application* app) override;
  void onUIRender() override;  // Called for anything related to UI
  void onUIMenu() override;    // This is the menubar to create

  void setLevelFilter(uint32_t levelFilter);
  void addLog(uint32_t level, const char* fmt, ...);

private:
  nvgui::SettingsHandler m_settingsHandler;

  void clear();
  void initColors();
  void draw(const char* title, bool* p_open = nullptr);

  uint32_t         m_levelFilter = eBitERROR | eBitWARNING | eBitINFO;
  ImGuiTextBuffer  m_buf{};
  ImGuiTextFilter  m_filter{};
  ImVector<int>    m_lineOffsets;       // Index to lines offset. We maintain this with AddLog() calls.
  ImVector<int>    m_lineLevels;        // Log level per line.
  ImVector<ImVec4> m_colors;            // Line color based on log level
  bool             m_autoScroll{true};  // Keep scrolling if already at the bottom.
  bool             m_showLog{false};
  std::mutex       m_modificationMutex;  // To protect from concurrent access
};

}  // namespace nvapp
