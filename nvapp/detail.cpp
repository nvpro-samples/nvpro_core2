/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "detail.hpp"

#include <imgui_internal.h>
#include <implot/implot.h>

#include <nvgui/fonts.hpp>
#include <nvgui/settings_handler.hpp>
#include <nvgui/style.hpp>
#include <nvutils/file_operations.hpp>
#include <nvutils/logger.hpp>

namespace nvapp::detail {

void initContexts()
{
  glfwInit();
  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImPlot::CreateContext();
}

std::string defaultIniFilename()
{
  return nvutils::utf8FromPath(nvutils::getExecutablePath().replace_extension(".ini"));
}

void initImGuiSettings(nvgui::SettingsHandler& handler, glm::uvec2& winSize, glm::ivec2& winPos, bool persistSettings, const std::string& iniFilename)
{
  nvgui::setStyle(false);

  handler.setHandlerName("Application");
  handler.setSetting("Size", &winSize);
  handler.setSetting("Pos", &winPos);
  handler.addImGuiHandler();

  // Only sessions that persist restore the window size, position and layout:
  // a headless, scripted or benchmark result must not depend on how the last
  // session was left (a restored window size changes the viewport, and with it
  // the image and the timings).
  if(persistSettings)
  {
    ImGui::LoadIniSettingsFromDisk(iniFilename.c_str());
  }

  // Likewise ImGui auto-saves persisting sessions only, so an automated run
  // does not overwrite the layout of interactive sessions.
  ImGuiIO& io    = ImGui::GetIO();
  io.IniFilename = persistSettings ? iniFilename.c_str() : nullptr;

  nvgui::addDefaultFont();
  io.FontDefault = nvgui::getDefaultFont();
  nvgui::addMonospaceFont();
}

void centerOnPrimaryMonitor(const glm::ivec2& winSize, glm::ivec2& winPos)
{
  GLFWmonitor* primary = glfwGetPrimaryMonitor();
  if(primary)
  {
    int monX, monY;
    glfwGetMonitorPos(primary, &monX, &monY);
    const GLFWvidmode* mode = glfwGetVideoMode(primary);
    winPos.x                = monX + (mode->width - winSize.x) / 2;
    winPos.y                = monY + (mode->height - winSize.y) / 2;
  }
  else
  {
    winPos.x = 0;
    winPos.y = 0;
  }
}

bool isWindowPosValid(const glm::ivec2& winPos)
{
  int           monitorCount;
  GLFWmonitor** monitors = glfwGetMonitors(&monitorCount);
  for(int i = 0; i < monitorCount; i++)
  {
    GLFWmonitor*       monitor = monitors[i];
    const GLFWvidmode* mode    = glfwGetVideoMode(monitor);
    int                monX, monY;
    glfwGetMonitorPos(monitor, &monX, &monY);
    if(winPos.x >= monX && winPos.x < monX + mode->width && winPos.y >= monY && winPos.y < monY + mode->height)
    {
      return true;
    }
  }
  return false;
}

void chooseWindowSizeAndPos(const glm::uvec2& requested, bool headless, glm::uvec2& winSize, glm::ivec2& winPos)
{
  bool centerWindow = false;
  if(requested.x != 0 && requested.y != 0)
  {
    winSize      = requested;
    centerWindow = true;  // a requested size is centered
  }

  // Still (0,0) unless the settings file provided a size.
  if(winSize.x == 0 && winSize.y == 0)
  {
    if(headless)
    {
      winSize = {800, 600};
    }
    else
    {
      // 80% of the primary monitor; none when there is no display (SSH, headless-like).
      GLFWmonitor* primary = glfwGetPrimaryMonitor();
      if(primary)
      {
        const GLFWvidmode* mode = glfwGetVideoMode(primary);
        winSize.x               = static_cast<uint32_t>(mode->width * 0.8f);
        winSize.y               = static_cast<uint32_t>(mode->height * 0.8f);
      }
      else
      {
        winSize = {800, 600};
      }
    }
    if(!headless)
    {
      centerOnPrimaryMonitor(winSize, winPos);
    }
  }
  else if(!headless)
  {
    if(!isWindowPosValid(winPos) || centerWindow)
    {
      centerOnPrimaryMonitor(winSize, winPos);
    }
  }
}

GLFWwindow* createWindow(const char* title, const glm::uvec2& size, const glm::ivec2& pos, void* userPointer, GLFWdropfun dropCallback)
{
  glfwInit();
  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
  glfwWindowHint(GLFW_SCALE_TO_MONITOR, GLFW_TRUE);  // Aware of DPI scaling
  GLFWwindow* window = glfwCreateWindow(int(size.x), int(size.y), title, nullptr, nullptr);
  if(!window)
  {
    LOGE("glfwCreateWindow failed (no display? set DISPLAY env for X11)\n");
    return nullptr;
  }
  glfwSetWindowSize(window, int(size.x), int(size.y));  // Sets the size of the window using the DPI scaling
  glfwSetWindowPos(window, pos.x, pos.y);
  glfwSetWindowUserPointer(window, userPointer);
  glfwSetDropCallback(window, dropCallback);
  return window;
}

void setupImguiDock(const std::function<void(ImGuiID)>& dockSetup)
{
  const ImGuiDockNodeFlags dockFlags = ImGuiDockNodeFlags_PassthruCentralNode | ImGuiDockNodeFlags_NoDockingInCentralNode;
  ImGuiID dockID = ImGui::DockSpaceOverViewport(0, ImGui::GetMainViewport(), dockFlags);
  // The layout is built only when none exists yet.
  if(!ImGui::DockBuilderGetNode(dockID)->IsSplitNode() && !ImGui::FindWindowByName("Viewport"))
  {
    ImGui::DockBuilderDockWindow("Viewport", dockID);  // Dock "Viewport" to the central node
    ImGui::DockBuilderGetCentralNode(dockID)->LocalFlags |= ImGuiDockNodeFlags_NoTabBar;  // no tab on the central node
    if(dockSetup)
    {
      dockSetup(dockID);
    }
    else
    {
      ImGuiID leftID = ImGui::DockBuilderSplitNode(dockID, ImGuiDir_Left, 0.2f, nullptr, &dockID);
      ImGui::DockBuilderDockWindow("Settings", leftID);
    }
  }
}

float updateDpiScale(GLFWwindow* window, float previousScale)
{
  float xscale, yscale;
  glfwGetWindowContentScale(window, &xscale, &yscale);
  ImGui::GetIO().FontGlobalScale *= xscale / previousScale;
  return xscale;
}

bool framebufferResized(GLFWwindow* window, glm::uvec2& size)
{
  int width, height;
  glfwGetFramebufferSize(window, &width, &height);
  const glm::uvec2 newSize = {static_cast<uint32_t>(width), static_cast<uint32_t>(height)};
  if(newSize.x > 0 && newSize.y > 0 && newSize != size)
  {
    size = newSize;
    return true;
  }
  return false;
}

glm::uvec2 viewportWindowSize(const glm::uvec2& windowSize)
{
  const ImGuiWindow* viewport = ImGui::FindWindowByName("Viewport");
  if(!viewport)
  {
    return windowSize;
  }
  const glm::uvec2 size{uint32_t(viewport->Size.x), uint32_t(viewport->Size.y)};
  ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
  ImGui::Begin("Viewport");
  ImGui::End();
  ImGui::PopStyleVar();
  return size;
}

void freeQueueSubmit(FreeQueue& queue, uint32_t frame, std::function<void()>&& func)
{
  if(frame < queue.size())
  {
    queue[frame].emplace_back(std::move(func));
  }
  else
  {
    func();
  }
}

void freeQueueReset(FreeQueue& queue, uint32_t size)
{
  for(size_t i = 0; i < queue.size(); i++)
  {
    auto& slot = queue[i];
    for(size_t j = 0; j < slot.size(); j++)
    {
      slot[j]();
    }
    slot.clear();
  }
  queue.clear();
  queue.resize(size);
}

void freeQueueRun(FreeQueue& queue, uint32_t frame)
{
  for(size_t i = 0; i < queue[frame].size(); i++)
  {
    queue[frame][i]();
  }
  queue[frame].clear();
}

}  // namespace nvapp::detail


//--------------------------------------------------------------------------------------------------
// Usage example
//--------------------------------------------------------------------------------------------------
[[maybe_unused]] static void usage_detail()
{
  // The steps an application class strings together; nvapp::Application and
  // nvapp::ApplicationBase keep the state in their own members.
  nvapp::detail::initContexts();  // GLFW, ImGui and ImPlot

  // Window size and position: saved by ImGui's .ini, else 80% of the primary monitor
  nvgui::SettingsHandler settings;
  glm::uvec2             winSize{};
  glm::ivec2             winPos{};
  const std::string      iniFilename = nvapp::detail::defaultIniFilename();
  nvapp::detail::initImGuiSettings(settings, winSize, winPos, true, iniFilename);
  nvapp::detail::chooseWindowSizeAndPos({0, 0}, false, winSize, winPos);

  GLFWwindow* window = nvapp::detail::createWindow("Usage", winSize, winPos, nullptr, nullptr);
  if(window == nullptr)
  {
    return;
  }
  nvapp::detail::framebufferResized(window, winSize);  // the framebuffer size under DPI scaling

  // One list of deferred releases per frame in flight
  const uint32_t           framesInFlight = 2;
  nvapp::detail::FreeQueue freeQueue;
  nvapp::detail::freeQueueReset(freeQueue, framesInFlight);

  uint32_t frame    = 0;
  float    dpiScale = 1.f;
  while(!glfwWindowShouldClose(window))
  {
    glfwPollEvents();
    if(nvapp::detail::framebufferResized(window, winSize))
    {
      // EX: rebuild the swapchain at winSize
    }
    dpiScale = nvapp::detail::updateDpiScale(window, dpiScale);

    // EX: ImGui_ImplXXX_NewFrame(); ImGui_ImplGlfw_NewFrame(); ImGui::NewFrame();
    nvapp::detail::setupImguiDock(nullptr);  // "Settings" left of "Viewport"
    [[maybe_unused]] const glm::uvec2 viewportSize = nvapp::detail::viewportWindowSize(winSize);  // EX: resize the render targets

    nvapp::detail::freeQueueRun(freeQueue, frame);  // the slot's previous GPU work completed
    nvapp::detail::freeQueueSubmit(freeQueue, frame, [] { /* EX: destroy a resource this frame stops using */ });
    // EX: record, submit and present
    frame = (frame + 1) % framesInFlight;
  }
  nvapp::detail::freeQueueReset(freeQueue, 0);  // runs every pending release
  glfwDestroyWindow(window);
}
