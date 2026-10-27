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

#pragma once

#include <cassert>
#include <filesystem>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include <glm/vec2.hpp>
#include <imgui/imgui.h>

#include <nvgui/settings_handler.hpp>

#include "frame_pacer.hpp"


/*-------------------------------------------------------------------------------------------------
# class nvapp::ApplicationBase

> The API-agnostic part of an application: the GLFW window, the ImGui context,
> dockspace and menu bar, the element list, the main and headless loops, the
> persisted window placement and the deferred resource free queue.

Back ends other than Vulkan derive from it and implement the `api*` hooks;
`nvdx12app::Application` is one. The Vulkan `nvapp::Application` is a separate
class with the same element callbacks and create-info fields. A back end
exposes its own `init` / `deinit`, forwards the API-agnostic part of its
create info as an ApplicationBaseCreateInfo, and makes `run()` public.

Example:
````cpp
    class MyApplication : public nvapp::ApplicationBase
    {
    public:
      bool init(MyCreateInfo& info)
      {
        nvapp::ApplicationBaseCreateInfo base{.name = info.name, .windowSize = info.windowSize};
        if(!nvapp::ApplicationBase::init(base))  // ImGui context and settings, window placement, GLFW window
          return false;
        m_frameRingSize = info.framesInFlight;   // one slot of the free queue per frame in flight
        resetFreeQueue(m_frameRingSize);
        // ... swapchain on getWindowHandle(), per-frame command lists, ImGui renderer
        return true;
      }
      using nvapp::ApplicationBase::run;
      using nvapp::ApplicationBase::deinit;
      // ... addElement, setVsync, requestScreenShot and the api* hooks
    };
````

Main loop, per frame:
* frame pacing (vsync on), glfwPollEvents, framebuffer resize -> apiSwapChainRebuild
* onUpdate for each element; a minimized window skips the rest
* apiImguiBeginFrame, ImGui::NewFrame, dockspace, onUIMenu for each element
* "Viewport" window size change -> apiViewportSizeChange
* apiNormalRun: the back end runs onUIRender, onPreRender, onRender, onPostRender,
  draws ImGui and presents
* ImGui::EndFrame and the platform windows

Headless: apiViewportSizeChange once, one UI pass for the dock layout, then
headlessFrameCount times onUpdate and apiHeadlessRun, apiWaitIdle, and
onLastHeadlessFrame for each element. A zero count runs until close().

`init`, `deinit`, `run` and `headlessRun` parallel those of the Vulkan
`nvapp::Application`: both call the shared steps in `nvapp::detail`
(detail.hpp) but keep their own sequence, so a change to one must be made to
the other. Both keep the same invariants: onUpdate runs before
the minimized-window skip, onPostRender runs before present, the window
placement and ImGui layout are only saved when persistSettings is on (never
headless), and headless runs headlessFrameCount frames, or until close() when
it is zero.

Elements that record into the back end's command list derive from its
interface (`nvdx12app::IAppElement`); API-agnostic elements derive from
IAppElementBase. Order of elements can be important if one depends on the other.

## Docking

The layout can be customized by providing a function to the create info:
````cpp
    appInfo.dockSetup = [](ImGuiID viewportID) {
      ImGuiID settingID = ImGui::DockBuilderSplitNode(viewportID, ImGuiDir_Right, 0.2F, nullptr, &viewportID);
      ImGui::DockBuilderDockWindow("Settings", settingID);
    };
````
-------------------------------------------------------------------------------------------------*/

// Forward declarations
struct GLFWwindow;

namespace nvapp {
// Forward declarations
class ApplicationBase;

//-------------------------------------------------------------------------------------------------
// Interface for API-agnostic application elements
struct IAppElementBase
{
  // Interface
  virtual void onAttach(ApplicationBase* app) {}  // Called once at start
  virtual void onDetach() {}                      // Called before destroying the application
  // Keeps non-render application work responsive before an application loop can skip drawing.
  virtual void onUpdate() {}                        // Called once per application loop, before any render early-out
  virtual void onUIRender() {}                      // Called for anything related to UI
  virtual void onUIMenu() {}                        // This is the menubar to create
  virtual void onResize(const glm::uvec2& size) {}  // Called when the viewport size is changing
  virtual void onPreRender() {}  // called post onUIRender and prior onRender (looped over all elements)
  // Called after submission and before presentation takes ownership. Consumers
  // that read pixels on the CPU must wait for the submission to complete.
  virtual void onPostRender() {}
  virtual void onFileDrop(const std::filesystem::path& filename) {}  // For when a file is dragged on top of the window
  virtual void onLastHeadlessFrame() {};  // Called at the end of the last frame in headless mode

  virtual ~IAppElementBase() = default;
};

// Application creation info: the API-agnostic fields of a back end's create info
struct ApplicationBaseCreateInfo
{
  // General
  std::string name;  // Application name

  // GLFW
  glm::uvec2 windowSize{0, 0};  // Window size (width, height) or Viewport size (headless)
  bool       vSync{true};       // Enable V-Sync by default

  // UI
  bool                         useMenu{true};                 // Include a menubar
  bool                         hasUndockableViewport{false};  // Allow floating windows
  std::function<void(ImGuiID)> dockSetup;                     // Dock layout setup
  ImGuiConfigFlags imguiConfigFlags{ImGuiConfigFlags_NavEnableKeyboard | ImGuiConfigFlags_NavEnableGamepad | ImGuiConfigFlags_DockingEnable};
  bool persistSettings{true};  // Restore and save <exe>.ini (window, layout, element settings). Turn off for
                               // scripted/benchmark runs so they don't depend on the last session. Always off headless.

  // Headless
  bool     headless{false};        // Run without a window
  uint32_t headlessFrameCount{1};  // Frames to render in headless mode; 0 runs until close()
};

class ApplicationBase
{
public:
  ApplicationBase();
  virtual ~ApplicationBase() { assert(m_elements.empty()); }  // Forgot to call deinit

  // Application control
  void close();  // Stopping the application

  // Adding engines
  virtual void addElement(const std::shared_ptr<IAppElementBase>& layer) = 0;

  // Safely freeing up resources
  void submitResourceFree(std::function<void()>&& func);

  // Utilities
  bool         isVsync() const { return m_vsyncWanted; }                 // Return true if V-Sync is on
  virtual void setVsync(bool v) = 0;                                     // Set V-Sync on or off
  bool         isHeadless() const { return m_headless; }                 // Return true if headless
  bool         getPersistSettings() const { return m_persistSettings; }  // Return true if <exe>.ini is restored/saved
  void         setRenderWhileMinimized(bool v) { m_renderWhileMinimized = v; }  // Continue rendering when minimized
  uint32_t     getHeadlessFrameCount() const { return m_headlessFrameCount; }  // Number of frames to render in headless
  bool         getUseMenubar() const { return m_useMenubar; }  // Return true if the main menubar is drawn
  void         setUseMenubar(bool v) { m_useMenubar = v; }     // Show/hide the menubar (skips onUIMenu)

  // Getters
  inline const glm::uvec2& getViewportSize() const { return m_viewportSize; }
  inline const glm::uvec2& getWindowSize() const { return m_windowSize; }
  inline GLFWwindow*       getWindowHandle() const { return m_windowHandle; }
  inline uint32_t          getFrameCycleIndex() const { return m_frameRingCurrent; }
  inline uint32_t          getFrameCycleSize() const { return m_frameRingSize; }

  // Simulate the Drag&Drop of a file
  void onFileDrop(const std::filesystem::path& filename);

  // Record that a screenshot is requested; the back end saves the presented
  // image a few frames later (so that ImGui has time to clear the menu).
  virtual void requestScreenShot(const std::filesystem::path& filename, int quality = 90) = 0;


protected:
  // Back-end hooks
  virtual void apiImguiBeginFrame()    = 0;  // ImGui_ImplXXX_NewFrame()
  virtual void apiSwapChainRebuild()   = 0;  // The framebuffer size changed; m_windowSize holds the new size
  virtual void apiViewportSizeChange() = 0;  // m_viewportSize changed; calls onResize on the elements
  virtual void apiNormalRun()          = 0;  // One windowed frame: elements, ImGui draw, submit, present
  virtual void apiHeadlessRun()        = 0;  // One headless frame: elements and submit
  virtual void apiWaitIdle()           = 0;  // Blocks until the GPU is idle
  virtual void deinitApi() = 0;  // Destroys the API objects; the elements are gone, ImGui and GLFW are still up

  bool init(ApplicationBaseCreateInfo& info);
  bool initGlfw(ApplicationBaseCreateInfo& info);
  void deinit();
  void run();  // Run indefinitely until close is requested
  void headlessRun();

  void onViewportSizeChange(const glm::uvec2& size);
  void testAndSetWindowSizeAndPos(const glm::uvec2& winSize);
  void initializeImGuiContextAndSettings();
  void setupImguiDock();
  void resetFreeQueue(uint32_t size);
  void freeResourcesQueue();


  std::vector<std::shared_ptr<IAppElementBase>> m_elements;  // List of application elements to be called

  bool        m_useMenubar{true};   // Will use a menubar
  bool        m_vsyncWanted{true};  // Wanting swapchain with vsync
  std::string m_iniFilename;        // Holds an .ini name as UTF-8 since ImGui uses this encoding

  FramePacer m_framePacer;  // Low-latency system

  uint32_t m_frameRingCurrent{0};  // Current frame index in the ring buffer (cycles through available frames) : static for resource free queue
  uint32_t m_frameRingSize{0};  // Frames in flight; the back end sets it before resetFreeQueue()

  GLFWwindow* m_windowHandle{nullptr};  // GLFW Window
  glm::uvec2  m_viewportSize{0, 0};     // Size of the viewport
  glm::uvec2  m_windowSize{0, 0};       // Size of the window (framebuffer size)
  float       m_dpiScale{1.f};          // Current scaling due to DPI.

  //--
  std::vector<std::vector<std::function<void()>>> m_resourceFreeQueue;  // Queue of functions to free resources

  //--
  std::function<void(ImGuiID)> m_dockSetup;  // Function to setup the docking

  bool     m_headless{false};
  bool     m_persistSettings{true};  // ApplicationBaseCreateInfo::persistSettings, forced off in headless
  bool     m_headlessClose{false};
  uint32_t m_headlessFrameCount{1};
  bool     m_renderWhileMinimized{false};  // Continue rendering when window minimized (for external presenters)

  // Use for persist the data
  nvgui::SettingsHandler m_settingsHandler;
  glm::ivec2             m_winPos{};
  glm::uvec2             m_winSize{};
};


}  // namespace nvapp
