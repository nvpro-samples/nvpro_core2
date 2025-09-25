/*
 * Copyright (c) 2014-2025, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2014-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include <vulkan/vulkan_core.h>

#include <functional>
#include <filesystem>
#include <memory>
#include <string>

#include <glm/vec2.hpp>
#include <imgui/imgui.h>

#include <nvgui/settings_handler.hpp>
#include <nvvk/resources.hpp>
#include <nvvk/swapchain.hpp>
#include "frame_pacer.hpp"

/*-------------------------------------------------------------------------------------------------
# class nvapp::Application

To use the application, 
* Fill the ApplicationCreateInfo with all the information, 

Example:
````cpp
    nvapp::ApplicationCreateInfo appInfo;
    appInfo.name           = "Minimal Test";
    appInfo.width          = 800;
    appInfo.height         = 600;
    appInfo.vSync          = false;
    appInfo.instance       = vkContext.getInstance();
    appInfo.physicalDevice = vkContext.getPhysicalDevice();
    appInfo.device         = vkContext.getDevice();
    appInfo.queues         = vkContext.getQueueInfos();
 ```

* Attach elements to the application, the main rendering, and others like, camera, etc.
* Call run() to start the application.
*
* The application will create the window, and the ImGui context.

Worth notice
* ::init() : will create the GLFW window, call nvvk::context for the creation of the 
              Vulkan context, initialize ImGui , create the surface and window (::setupVulkanWindow)  
* ::shutdown() : the opposite of init
* ::run() : while running, render the frame and present the frame. Check for resize, minimize window 
              and other changes. In the loop, it will call some functions for each 'element' that is connected.
              onUIRender, onUIMenu, onRender. See IApplication for details.
* The Application is a singleton, and the main loop is inside the run() function.
* The Application is the owner of the elements, and it will call the onRender, onUIRender, onUIMenu
    for each element that is connected to it.
* The Application is the owner of the Vulkan context, and it will create the surface and window.
* The Application is the owner of the ImGui context, and it will create the dockspace and the main menu.
* The Application is the owner of the GLFW window, and it will create the window and handle the events.


The application itself does not render per se. It contains control buffers for the images in flight,
it calls ImGui rendering for Vulkan, but that's it. Note that none of the samples render
directly into the swapchain. Instead, they render into an image, and the image is displayed in the ImGui window
window called "Viewport".

Application elements must be created to render scenes or add "elements" to the application.  Several elements 
can be added to an application, and each of them will be called during the frame. This allows the application 
to be divided into smaller parts, or to reuse elements in various samples. For example, there is an element 
that adds a default menu (File/Tools), another that changes the window title with FPS, the resolution, and there
is also an element for our automatic tests.

Each added element will be called in a frame, see the IAppElement interface for information on virtual functions.
Basically there is a call to create and destroy, a call to render the user interface and a call to render the 
frame with the command buffer.

Note: order of Elements can be important if one depends on the other. For example, the ElementCamera should
      be added before the rendering sample, such that its matrices are updated before pulled by the renderer.


## Docking

The layout can be customized by providing a function to the ApplicationCreateInfo. This function will be called

Example:
````
    // Setting up the layout of the application
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
class Application;

//-------------------------------------------------------------------------------------------------
// Interface for application elements
struct IAppElement
{
  // Interface
  virtual void onAttach(Application* app) {}                             // Called once at start
  virtual void onDetach() {}                                             // Called before destroying the application
  virtual void onResize(VkCommandBuffer cmd, const VkExtent2D& size) {}  // Called when the viewport size is changing
  virtual void onUIRender() {}                                           // Called for anything related to UI
  virtual void onUIMenu() {}                                             // This is the menubar to create
  virtual void onPreRender() {}                  // called post onUIRender and prior onRender (looped over all elements)
  virtual void onRender(VkCommandBuffer cmd) {}  // For anything to render within a frame
  virtual void onFileDrop(const std::filesystem::path& filename) {}  // For when a file is dragged on top of the window
  virtual void onLastHeadlessFrame() {};  // Called at the end of the last frame in headless mode


  virtual ~IAppElement() = default;
};

// Application creation info
struct ApplicationCreateInfo
{
  // General
  std::string name{"Vulkan_App"};  // Application name

  // Vulkan
  VkInstance                   instance{VK_NULL_HANDLE};        // Vulkan instance
  VkDevice                     device{VK_NULL_HANDLE};          // Logical device
  VkPhysicalDevice             physicalDevice{VK_NULL_HANDLE};  // Physical device
  std::vector<nvvk::QueueInfo> queues;                          // Queue family and properties (0: Graphics)
  uint32_t                     texturePoolSize = 128U;          // Maximum number of textures in the descriptor pool

  // GLFW
  glm::uvec2 windowSize{0, 0};  // Window size (width, height) or Viewport size (headless)
  bool       vSync{true};       // Enable V-Sync by default

  // UI
  bool                         useMenu{true};                 // Include a menubar
  bool                         hasUndockableViewport{false};  // Allow floating windows
  std::function<void(ImGuiID)> dockSetup;                     // Dock layout setup
  ImGuiConfigFlags             imguiConfigFlags{ImGuiConfigFlags_NavEnableKeyboard | ImGuiConfigFlags_DockingEnable};

  // Headless
  bool     headless{false};        // Run without a window
  uint32_t headlessFrameCount{1};  // Frames to render in headless mode

  // Swapchain
  // VK_PRESENT_MODE_MAX_ENUM_KHR means no preference
  VkPresentModeKHR preferredVsyncOffMode = VK_PRESENT_MODE_MAX_ENUM_KHR;
  VkPresentModeKHR preferredVsyncOnMode  = VK_PRESENT_MODE_MAX_ENUM_KHR;
};


class Application
{
public:
  Application();
  virtual ~Application() { assert(m_elements.empty()); }  // Forgot to call deinit

  // Initialization and shutdown
  void init(ApplicationCreateInfo& info);
  void deinit();

  // Application control
  void run();    // Run indefinitely until close is requested
  void close();  // Stopping the application

  // Adding engines
  void addElement(const std::shared_ptr<IAppElement>& layer);

  // Safely freeing up resources
  void submitResourceFree(std::function<void()>&& func);

  // Utilities
  bool isVsync() const { return m_vsyncWanted; }  // Return true if V-Sync is on
  void setVsync(bool v);                          // Set V-Sync on or off
  bool isHeadless() const { return m_headless; }  // Return true if headless

  // Following three functions affect the preparation of the current frame's submit info.
  // Content is appended to vectors that are reset every frame
  void                addWaitSemaphore(const VkSemaphoreSubmitInfo& wait);
  void                addSignalSemaphore(const VkSemaphoreSubmitInfo& signal);
  nvvk::SemaphoreInfo getFrameSignalSemaphore() const;  // Return the current frame's signal semaphore

  // these command buffers are enqueued before the command buffer that is provided `onRender(cmd)`
  void prependCommandBuffer(const VkCommandBufferSubmitInfo& cmd);

  // Utility to create a temporary command buffer
  VkCommandBuffer createTempCmdBuffer() const;
  void            submitAndWaitTempCmdBuffer(VkCommandBuffer cmd);

  // Getters
  inline VkInstance             getInstance() const { return m_instance; }
  inline VkPhysicalDevice       getPhysicalDevice() const { return m_physicalDevice; }
  inline VkDevice               getDevice() const { return m_device; }
  inline const nvvk::QueueInfo& getQueue(uint32_t index) const { return m_queues[index]; }
  inline VkCommandPool          getCommandPool() const { return m_transientCmdPool; }
  inline VkDescriptorPool       getTextureDescriptorPool() const { return m_descriptorPool; }
  inline const VkExtent2D&      getViewportSize() const { return m_viewportSize; }
  inline const VkExtent2D&      getWindowSize() const { return m_windowSize; }
  inline GLFWwindow*            getWindowHandle() const { return m_windowHandle; }
  inline uint32_t               getFrameCycleIndex() const { return m_frameRingCurrent; }
  inline uint32_t               getFrameCycleSize() const { return uint32_t(m_frameData.size()); }

  // Simulate the Drag&Drop of a file
  void onFileDrop(const std::filesystem::path& filename);

  // Record that a screenshot is requested, and will be saved after a full
  // frame cycle loop (so that ImGui has time to clear the menu).
  void screenShot(const std::filesystem::path& filename, int quality = 100);

  // Saves a VkImage to a file, blitting it to RGBA8 format along the way.
  void saveImageToFile(VkImage srcImage, VkExtent2D imageSize, const std::filesystem::path& filename, int quality = 100);


private:
  void            initGlfw(ApplicationCreateInfo& info);
  void            createTransientCommandPool();
  void            createFrameSubmission(uint32_t numFrames);
  void            createDescriptorPool();
  void            onViewportSizeChange(VkExtent2D size);
  void            headlessRun();
  VkCommandBuffer beginCommandRecording();
  void            addSwapchainSemaphores();
  void            drawFrame(VkCommandBuffer cmd);
  void            renderToSwapchain(VkCommandBuffer cmd);
  bool            prepareFrameResources();
  void            endFrame(VkCommandBuffer cmd, uint32_t frameInFlights);
  void            presentFrame();
  void            advanceFrame(uint32_t frameInFlights);
  void            waitForFrameCompletion() const;
  void            beginDynamicRenderingToSwapchain(VkCommandBuffer cmd) const;
  void            endDynamicRenderingToSwapchain(VkCommandBuffer cmd);
  void            saveScreenShot(const std::filesystem::path& filename, int quality);  // Immediately save the frame
  void            resetFreeQueue(uint32_t size);
  void            freeResourcesQueue();
  void            setupImguiDock();
  void            prepareFrameToSignal(int32_t numFramesInFlight);
  void            testAndSetWindowSizeAndPos(const glm::uvec2& winSize);
  bool            isWindowPosValid(const glm::ivec2& winPos);
  void            initializeImGuiContextAndSettings();
  void            setupImGuiVulkanBackend(ImGuiConfigFlags configFlags);


  std::vector<std::shared_ptr<IAppElement>> m_elements;  // List of application elements to be called

  bool        m_useMenubar{true};   // Will use a menubar
  bool        m_vsyncWanted{true};  // Wanting swapchain with vsync
  std::string m_iniFilename;        // Holds an .ini name as UTF-8 since ImGui uses this encoding

  // Vulkan resources
  VkInstance                   m_instance{VK_NULL_HANDLE};
  VkPhysicalDevice             m_physicalDevice{VK_NULL_HANDLE};
  VkDevice                     m_device{VK_NULL_HANDLE};
  std::vector<nvvk::QueueInfo> m_queues{};             // All queues, first one should be a graphics queue
  VkSurfaceKHR                 m_surface{};            // The window surface
  VkCommandPool                m_transientCmdPool{};   // The command pool
  VkDescriptorPool             m_descriptorPool{};     // Application descriptor pool
  uint32_t                     m_maxTexturePool{128};  // Maximum number of textures in the descriptor pool

  // Frame resources and synchronization (Swapchain, Command buffers, Semaphores, Fences)
  nvvk::Swapchain m_swapchain;
  struct FrameData
  {
    VkCommandPool   cmdPool{};      // Command pool for recording commands for this frame
    VkCommandBuffer cmdBuffer{};    // Command buffer containing the frame's rendering commands
    uint64_t        frameNumber{};  // Timeline value for synchronization (increases each frame)
  };
  std::vector<FrameData> m_frameData{};    // Collection of per-frame resources to support multiple frames in flight
  VkSemaphore m_frameTimelineSemaphore{};  // Timeline semaphore used to synchronize CPU submission with GPU completion
  uint32_t m_frameRingCurrent{0};  // Current frame index in the ring buffer (cycles through available frames) : static for resource free queue

  // Fine control over the frame submission
  std::vector<VkSemaphoreSubmitInfo>     m_waitSemaphores;    // Possible extra frame wait semaphores
  std::vector<VkSemaphoreSubmitInfo>     m_signalSemaphores;  // Possible extra frame signal semaphores
  std::vector<VkCommandBufferSubmitInfo> m_commandBuffers;    // Possible extra frame command buffers

  FramePacer m_framePacer;  // Low-latency system

  GLFWwindow* m_windowHandle{nullptr};  // GLFW Window
  VkExtent2D  m_viewportSize{0, 0};     // Size of the viewport
  VkExtent2D  m_windowSize{0, 0};       // Size of the window
  float       m_dpiScale{1.f};          // Current scaling due to DPI.

  //--
  std::vector<std::vector<std::function<void()>>> m_resourceFreeQueue;  // Queue of functions to free resources

  //--
  std::function<void(ImGuiID)> m_dockSetup;  // Function to setup the docking

  bool                  m_headless{false};
  bool                  m_headlessClose{false};
  uint32_t              m_headlessFrameCount{1};
  bool                  m_screenShotRequested = false;
  int                   m_screenShotFrame     = 0;
  std::filesystem::path m_screenShotFilename;

  // Use for persist the data
  nvgui::SettingsHandler m_settingsHandler;
  glm::ivec2             m_winPos{};
  glm::uvec2             m_winSize{};
};


}  // namespace nvapp
