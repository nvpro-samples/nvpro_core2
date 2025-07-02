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


#include <GLFW/glfw3.h>
#undef APIENTRY

#include <nvutils/file_operations.hpp>
#include <nvutils/logger.hpp>
#include <fmt/format.h>


#include "elem_default_title.hpp"

nvapp::ElementDefaultWindowTitle::ElementDefaultWindowTitle(const std::string& prefix /*= ""*/, const std::string& suffix /*= ""*/)
    : m_prefix(prefix)
    , m_suffix(suffix)
{
}

void nvapp::ElementDefaultWindowTitle::onAttach(nvapp::Application* app)
{
  LOGI("Adding DefaultWindowTitle\n");
  m_app = app;
}

void nvapp::ElementDefaultWindowTitle::onUIRender()
{
  GLFWwindow* window = m_app->getWindowHandle();
  if(window == nullptr)  // This can happen in headless mode
  {
    return;
  }

  // Window Title
  m_dirtyTimer += ImGui::GetIO().DeltaTime;
  if(m_dirtyTimer > 1.0F)  // Refresh every second
  {
    const auto& size = m_app->getViewportSize();
    std::string title;
    if(!m_prefix.empty())
    {
      title += fmt::format("{} | ", m_prefix.c_str());
    }
    const std::string exeName = nvutils::utf8FromPath(nvutils::getExecutablePath().stem());
    title += fmt::format("{} | {}x{} | {:.0f} FPS / {:.3f}ms", exeName, size.width, size.height,
                         ImGui::GetIO().Framerate, 1000.F / ImGui::GetIO().Framerate);
    if(!m_suffix.empty())
    {
      title += fmt::format(" | {}", m_suffix.c_str());
    }
    glfwSetWindowTitle(m_app->getWindowHandle(), title.c_str());
    m_dirtyTimer = 0;
  }
}

void nvapp::ElementDefaultWindowTitle::setPrefix(const std::string& str)
{
  m_prefix = str;
}

void nvapp::ElementDefaultWindowTitle::setSuffix(const std::string& str)
{
  m_suffix = str;
}
