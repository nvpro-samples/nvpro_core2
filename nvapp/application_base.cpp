/*
 * Copyright (c) 2014-2026, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2014-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <filesystem>

#include <GLFW/glfw3.h>
#undef APIENTRY

#include <backends/imgui_impl_glfw.h>
#include <implot/implot.h>
#include <nvutils/logger.hpp>
#include <nvutils/file_operations.hpp>
#include <nvutils/timers.hpp>

#include "application_base.hpp"
#include "detail.hpp"

// GLFW Callback for file drop
static void dropCb(GLFWwindow* window, int count, const char** paths)
{
  auto* app = static_cast<nvapp::ApplicationBase*>(glfwGetWindowUserPointer(window));
  for(int i = 0; i < count; i++)
  {
    app->onFileDrop(nvutils::pathFromUtf8(paths[i]));
  }
}

nvapp::ApplicationBase::ApplicationBase(void)
{
  detail::initContexts();
}

bool nvapp::ApplicationBase::init(ApplicationBaseCreateInfo& info)
{
  m_vsyncWanted        = info.vSync;
  m_useMenubar         = info.useMenu;
  m_dockSetup          = info.dockSetup;
  m_headless           = info.headless;
  m_persistSettings    = info.persistSettings && !info.headless;  // headless has no window or layout to keep
  m_headlessFrameCount = info.headlessFrameCount;
  m_viewportSize       = {};  // Will be set by the first viewport size

  if(info.hasUndockableViewport == true)
  {
    info.imguiConfigFlags |= ImGuiConfigFlags_ViewportsEnable;
  }

  m_iniFilename = detail::defaultIniFilename();

  // Initialize ImGui Context and settings handler, and load window size/pos from ini
  initializeImGuiContextAndSettings();

  // Set the default size and position of the window
  testAndSetWindowSizeAndPos({info.windowSize.x, info.windowSize.y});

  // Initialize GLFW and create the window only if not headless
  if(!m_headless)
  {
    if(!initGlfw(info))
    {
      return false;
    }
  }

  return true;
}

bool nvapp::ApplicationBase::initGlfw(ApplicationBaseCreateInfo& info)
{
  m_windowHandle = detail::createWindow(info.name.c_str(), m_windowSize, m_winPos, this, &dropCb);
  if(m_windowHandle == nullptr)
  {
    return false;
  }
  // The back end renders at the framebuffer size, which differs from the
  // requested size under DPI scaling.
  detail::framebufferResized(m_windowHandle, m_windowSize);
  return true;
}

//-----------------------------------------------------------------------
// Shutdown the application
// This will destroy all resources and clean up the application.
void nvapp::ApplicationBase::deinit()
{
  // Query the size/pos of the window, such that it get persisted
  if(!m_headless && m_windowHandle != nullptr)
  {
    glm::ivec2 winSize{};
    glfwGetWindowSize(m_windowHandle, &winSize.x, &winSize.y);
    m_winSize = {uint32_t(winSize.x), uint32_t(winSize.y)};
    glfwGetWindowPos(m_windowHandle, &m_winPos.x, &m_winPos.y);
  }

  // Deferred frees can capture allocator/tracker state owned by application elements. Drain them
  // while every element is still attached; otherwise an element may destroy its allocator in
  // onDetach() before resetFreeQueue() invokes one of its pending callbacks.
  resetFreeQueue(0);

  // This will call the onDetach of the elements
  for(size_t i = 0; i < m_elements.size(); i++)
  {
    m_elements[i]->onDetach();
  }

  // This avoids ImGui to access destroyed elements (Handler for example)
  //
  // A session that doesn't persist (ApplicationBaseCreateInfo::persistSettings off, or headless) must not
  // save: a scripted or benchmark run arranges the window for its own purposes, and saving that
  // arrangement leaves the user's next interactive session with a layout nobody chose. An
  // application may also clear io.IniFilename temporarily to skip a save; honor that too, since
  // this call names the file directly rather than going through io.IniFilename.
  if(m_persistSettings && ImGui::GetIO().IniFilename != nullptr)
  {
    ImGui::SaveIniSettingsToDisk(m_iniFilename.c_str());
  }
  ImGui::GetIO().IniFilename = nullptr;  // Don't save the ini file again

  // Destroy the elements
  m_elements.clear();

  // A callback submitted during onDetach executes immediately while the queue has size zero. Keep a
  // final drain as a defensive cleanup before destroying application-owned API state.
  resetFreeQueue(0);

  deinitApi();

  if(!m_headless)
  {
    ImGui_ImplGlfw_Shutdown();
  }

  ImGui::DestroyContext();

  if(ImPlot::GetCurrentContext() != nullptr)
  {
    ImPlot::DestroyContext();
  }

  if(!m_headless)
  {
    // Glfw cleanup
    glfwDestroyWindow(m_windowHandle);
    glfwTerminate();
  }
}

void nvapp::ApplicationBase::onFileDrop(const std::filesystem::path& filename)
{
  for(size_t i = 0; i < m_elements.size(); i++)
  {
    m_elements[i]->onFileDrop(filename);
  }
}

void nvapp::ApplicationBase::close()
{
  if(m_headless)
  {
    m_headlessClose = true;
  }
  else
  {
    glfwSetWindowShouldClose(m_windowHandle, GLFW_TRUE);
  }
}

//-----------------------------------------------------------------------
// Main loop of the application
// It will run until the window is closed.
// Call all onUIRender() and onRender() for each element.
//
void nvapp::ApplicationBase::run()
{
  LOGI("Running application\n");

  if(!m_headless && m_windowHandle == nullptr)
  {
    LOGE("Application::run - m_windowHandle is null, aborting\n");
    return;
  }

  // Re-load ImGui settings from disk, as there might be application elements with settings to restore.
  // A session that doesn't persist doesn't restore either: its result would otherwise depend on
  // whatever the last interactive session saved. An application that wants specific settings can
  // call ImGui::LoadIniSettingsFromDisk() itself.
  if(m_persistSettings)
  {
    ImGui::LoadIniSettingsFromDisk(m_iniFilename.c_str());
  }

  // Handle headless mode
  if(m_headless)
  {
    headlessRun();
    return;
  }

  // Main rendering loop
  while(!glfwWindowShouldClose(m_windowHandle))
  {
    // Window System Events.
    // We add a delay before polling to reduce latency: one frame per refresh
    // instead of queueing into the swapchain; the back ends present as is.
    if(m_vsyncWanted)
    {
      m_framePacer.pace();
    }
    glfwPollEvents();

    // Window resize; a minimized window keeps the last valid size.
    if(detail::framebufferResized(m_windowHandle, m_windowSize))
    {
      apiSwapChainRebuild();
    }

    // Run non-render application work before the minimized early-out so services
    // waiting on the application thread can complete without forcing a frame.
    for(size_t i = 0; i < m_elements.size(); i++)
    {
      m_elements[i]->onUpdate();
    }

    // Skip rendering when minimized (unless we need to keep rendering for external presenters)
    if(glfwGetWindowAttrib(m_windowHandle, GLFW_ICONIFIED) == GLFW_TRUE && !m_renderWhileMinimized)
    {
      ImGui_ImplGlfw_Sleep(10);
      continue;
    }

    // Begin New Frame for ImGui
    apiImguiBeginFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    // Setup ImGui Docking and UI
    setupImguiDock();
    if(m_useMenubar && ImGui::BeginMainMenuBar())
    {
      for(size_t i = 0; i < m_elements.size(); i++)
      {
        m_elements[i]->onUIMenu();
      }
      ImGui::EndMainMenuBar();
    }

    // The "Viewport" window, or the whole window without one
    const glm::uvec2 viewportSize = detail::viewportWindowSize(m_windowSize);
    if(m_viewportSize != viewportSize)
    {
      onViewportSizeChange(viewportSize);
    }

    // Frame resources, element callbacks, ImGui draw, submit and present
    apiNormalRun();

    // End ImGui frame
    ImGui::EndFrame();

    // Handle Additional ImGui Windows
    if((ImGui::GetIO().ConfigFlags & ImGuiConfigFlags_ViewportsEnable) != 0)
    {
      ImGui::UpdatePlatformWindows();
      ImGui::RenderPlatformWindowsDefault();
    }
  }
}

//-----------------------------------------------------------------------
// IMGUI Docking
// Create a dockspace and dock the viewport and settings window.
// The central node is named "Viewport", which can be used later with Begin("Viewport")  to render the final image.
void nvapp::ApplicationBase::setupImguiDock()
{
  detail::setupImguiDock(m_dockSetup);
}

//-----------------------------------------------------------------------
// Call this function if the viewport size changes
// This happens when the window is resized, or when the ImGui viewport window is resized.
//
void nvapp::ApplicationBase::onViewportSizeChange(const glm::uvec2& size)
{
  m_dpiScale = detail::updateDpiScale(m_windowHandle, m_dpiScale);

  m_viewportSize = size;

  apiViewportSizeChange();
}

//-----------------------------------------------------------------------
// This is the headless version of the run loop.
// It will render the scene for the number of frames specified in the headlessFrameCount.
// It will call onUIRender() and onRender() for each element.
//
void nvapp::ApplicationBase::headlessRun()
{
  nvutils::ScopedTimer st(__FUNCTION__);
  m_viewportSize = m_windowSize;

  // Set the display for Imgui
  ImGuiIO& io      = ImGui::GetIO();
  io.DisplaySize.x = static_cast<float>(m_viewportSize.x);
  io.DisplaySize.y = static_cast<float>(m_viewportSize.y);

  // Make the size has been communicated everywhere
  apiViewportSizeChange();

  // Need to render the UI twice: the first pass sets up the internal state and layout,
  // and the second pass finalizes the rendering with the updated state.
  {
    apiImguiBeginFrame();
    ImGui::NewFrame();
    setupImguiDock();

    // Call UI rendering for each element
    for(size_t i = 0; i < m_elements.size(); i++)
    {
      m_elements[i]->onUIRender();
    }
    ImGui::EndFrame();
  }

  // Rendering n-times the scene. A zero count keeps a headless application
  // alive for an external controller until close() is requested.
  for(uint32_t frameID = 0; (m_headlessFrameCount == 0 || frameID < m_headlessFrameCount) && !m_headlessClose; frameID++)
  {
    for(size_t i = 0; i < m_elements.size(); i++)
    {
      m_elements[i]->onUpdate();
    }

    apiImguiBeginFrame();
    ImGui::NewFrame();  // Even if isn't directly used, helps advancing time if query

    apiHeadlessRun();

    ImGui::EndFrame();
  }
  ImGui::Render();  // This is creating the data to draw the UI (not on GPU yet)

  // At this point, everything has been rendered. Let it finish.
  apiWaitIdle();

  // Call back the application, such that it can do something with the rendered image
  for(size_t i = 0; i < m_elements.size(); i++)
  {
    m_elements[i]->onLastHeadlessFrame();
  }
}

//-----------------------------------------------------------------------
// Initializes the ImGui context, sets up the settings handler for window
// size and position, loads the ImGui .ini file, and sets up fonts and style.
// This function should be called before any logic that depends on the ImGui
// context or the loaded window settings.
//
void nvapp::ApplicationBase::initializeImGuiContextAndSettings()
{
  detail::initImGuiSettings(m_settingsHandler, m_winSize, m_winPos, m_persistSettings, m_iniFilename);
}


void nvapp::ApplicationBase::submitResourceFree(std::function<void()>&& func)
{
  detail::freeQueueSubmit(m_resourceFreeQueue, m_frameRingCurrent, std::move(func));
}

void nvapp::ApplicationBase::resetFreeQueue(uint32_t size)
{
  apiWaitIdle();
  detail::freeQueueReset(m_resourceFreeQueue, size);
}

// This is called to free all resources that are not used anymore
// By using the frameRingCurrent we can free resources that are not used anymore
void nvapp::ApplicationBase::freeResourcesQueue()
{
  detail::freeQueueRun(m_resourceFreeQueue, m_frameRingCurrent);
}


void nvapp::ApplicationBase::testAndSetWindowSizeAndPos(const glm::uvec2& winSize)
{
  detail::chooseWindowSizeAndPos(winSize, m_headless, m_winSize, m_winPos);
  m_windowSize = m_winSize;
}


//--------------------------------------------------------------------------------------------------
// Usage example
//--------------------------------------------------------------------------------------------------
namespace {
// A minimal back end: the API-specific part of the create info is left out,
// the api* hooks mark where the swapchain, command lists and ImGui renderer go.
class UsageApplication : public nvapp::ApplicationBase
{
public:
  bool init(nvapp::ApplicationBaseCreateInfo& info)
  {
    if(!nvapp::ApplicationBase::init(info))  // ImGui context and settings, window placement, GLFW window
    {
      return false;
    }
    m_frameRingSize = 2;  // EX: one command list per frame in flight
    resetFreeQueue(m_frameRingSize);
    // EX: create the swapchain on getWindowHandle() at m_windowSize, the frame resources and the ImGui renderer
    return true;
  }
  using nvapp::ApplicationBase::deinit;  // free queue, onDetach, .ini save, deinitApi, ImGui and GLFW shutdown
  using nvapp::ApplicationBase::run;

  void addElement(const std::shared_ptr<nvapp::IAppElementBase>& element) override
  {
    m_elements.emplace_back(element);
    element->onAttach(this);
  }
  void setVsync(bool v) override { m_vsyncWanted = v; }  // EX: or request a swapchain rebuild
  void requestScreenShot(const std::filesystem::path& /*filename*/, int /*quality*/) override {}  // EX: saved a few frames later

protected:
  void apiImguiBeginFrame() override {}   // EX: ImGui_ImplXXX_NewFrame()
  void apiSwapChainRebuild() override {}  // EX: resize the swapchain to m_windowSize at the next frame
  void apiViewportSizeChange() override   // the "Viewport" window changed size
  {
    for(size_t i = 0; i < m_elements.size(); i++)
    {
      m_elements[i]->onResize(m_viewportSize);
    }
  }
  void apiNormalRun() override
  {
    // EX: wait for the frame slot, freeResourcesQueue(), onUIRender, ImGui::Render(), onPreRender,
    // onRender, ImGui draw into the back buffer, submit, onPostRender, present
    m_frameRingCurrent = (m_frameRingCurrent + 1) % m_frameRingSize;
  }
  void apiHeadlessRun() override { apiNormalRun(); }  // EX: the same without the present
  void apiWaitIdle() override {}                      // EX: block until the GPU is idle
  void deinitApi() override {}  // EX: destroy the swapchain, the frame resources and the ImGui renderer
};
}  // namespace

[[maybe_unused]] static void usage_ApplicationBase()
{
  nvapp::ApplicationBaseCreateInfo info{.name = "Usage", .windowSize = {1280, 720}, .vSync = false};
  UsageApplication                 app;
  if(app.init(info))
  {
    // EX: app.addElement(std::make_shared<MyElement>());
    app.run();  // until close() or the window closes
  }
  app.deinit();
}
