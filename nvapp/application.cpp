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

#include <array>
#include <filesystem>

#include <volk/volk.h>

#include <GLFW/glfw3.h>
#undef APIENTRY

#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_vulkan.h>
#include <fmt/ranges.h>
#include <imgui_internal.h>
#include <implot/implot.h>
#define NVLOGGER_ENABLE_FMT
#include <nvutils/logger.hpp>
#include <nvutils/file_operations.hpp>
#include <nvutils/timers.hpp>
#include <nvvk/barriers.hpp>
#include <nvvk/check_error.hpp>
#include <nvvk/debug_util.hpp>
#include <nvvk/commands.hpp>
#include <nvvk/helpers.hpp>

#include <nvgui/fonts.hpp>
#include <nvgui/style.hpp>

#include "application.hpp"

// Default values
constexpr int32_t k_imageQuality = 90;

// GLFW Callback for file drop
static void dropCb(GLFWwindow* window, int count, const char** paths)
{
  auto* app = static_cast<nvapp::Application*>(glfwGetWindowUserPointer(window));
  for(int i = 0; i < count; i++)
  {
    app->onFileDrop(nvutils::pathFromUtf8(paths[i]));
  }
}

nvapp::Application::Application(void)
{
  glfwInit();
  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImPlot::CreateContext();
}

// Provides additional diagnostic information about which GPUs can be used with
// the given VkSurface. Only used when handling errors.
void reportSwapchainDiagnostics(VkInstance instance, nvvk::Swapchain::InitInfo& swapchainParams)
{
  LOGI("\nAvailable GPUs and presentation support for surface %p:\n", swapchainParams.surface);
  uint32_t                      gpuCount = 0;
  std::vector<VkPhysicalDevice> gpus;
  if(instance == nullptr || swapchainParams.surface == VK_NULL_HANDLE)
  {
    LOGI("  <instance or surface was nullptr>\n");
  }
  else if(VK_SUCCESS != vkEnumeratePhysicalDevices(instance, &gpuCount, nullptr))
  {
    LOGI("  <vkEnumeratePhysicalDevices failed>\n");
  }
  else if(0 == gpuCount)
  {
    LOGI("  <no devices>\n");
  }
  else
  {
    gpus.resize(gpuCount);
    vkEnumeratePhysicalDevices(instance, &gpuCount, gpus.data());
    for(uint32_t gpuIdx = 0; gpuIdx < gpuCount; gpuIdx++)
    {
      VkPhysicalDeviceProperties deviceProps{};
      vkGetPhysicalDeviceProperties(gpus[gpuIdx], &deviceProps);

      // Check which queue families on this GPU can present
      uint32_t queueFamilyCount = 0;
      vkGetPhysicalDeviceQueueFamilyProperties(gpus[gpuIdx], &queueFamilyCount, nullptr);
      bool                  anyCanPresent = false;
      std::vector<uint32_t> presentableQueueFamilies;
      for(uint32_t queueFamilyIdx = 0; queueFamilyIdx < queueFamilyCount; queueFamilyIdx++)
      {
        VkBool32 presentSupported = VK_FALSE;
        vkGetPhysicalDeviceSurfaceSupportKHR(gpus[gpuIdx], queueFamilyIdx, swapchainParams.surface, &presentSupported);
        if(VK_TRUE == presentSupported)
        {
          anyCanPresent = true;
          presentableQueueFamilies.push_back(queueFamilyIdx);
        }
      }

      if(anyCanPresent)
      {
        PRINTI("  GPU {} ({}): CAN present (using queue family indices {})\n", gpuIdx, deviceProps.deviceName, presentableQueueFamilies);
      }
      else
      {
        PRINTI("  GPU {} ({}): CANNOT present\n", gpuIdx, deviceProps.deviceName);
      }
    }
  }

  VkPhysicalDeviceProperties chosenDeviceProps{};
  vkGetPhysicalDeviceProperties(swapchainParams.physicalDevice, &chosenDeviceProps);
  LOGE(
      "Failed to create the swapchain for VkSurface %p with VkPhysicalDevice %p (%s).\n"
      "This might happen if you're on a multi-monitor Linux system with different GPUs plugged into different windowing system desktops, and GLFW chose a desktop not connected to the physical device that the sample or nvvk::Context chose.\n"
      "To fix this, set nvvk::ContextInfo in the sample to the index of a GPU with \"CAN Present\" listed next to it above.\n",
      swapchainParams.surface, swapchainParams.physicalDevice, chosenDeviceProps.deviceName);
  // Note that this is essentially a workaround for a bug that would require
  // changing the nvpro_core2 design; to fix this, we would either need to create
  // the window and surface before the context, or we would need to link NVVK
  // against GLFW and have nvvk::Context call glfwGetPhysicalDevicePresentationSupport.
}

void nvapp::Application::init(ApplicationCreateInfo& info)
{
  m_instance           = info.instance;
  m_device             = info.device;
  m_physicalDevice     = info.physicalDevice;
  m_queues             = info.queues;
  m_vsyncWanted        = info.vSync;
  m_useMenubar         = info.useMenu;
  m_dockSetup          = info.dockSetup;
  m_headless           = info.headless;
  m_headlessFrameCount = info.headlessFrameCount;
  m_viewportSize       = {};  // Will be set by the first viewport size
  m_maxTexturePool     = info.texturePoolSize;

  if(info.hasUndockableViewport == true)
  {
    info.imguiConfigFlags |= ImGuiConfigFlags_ViewportsEnable;
  }

  // Get the executable path and set the ini file name
  m_iniFilename = nvutils::utf8FromPath(nvutils::getExecutablePath().replace_extension(".ini"));

  // Initialize ImGui Context and settings handler, and load window size/pos from ini
  initializeImGuiContextAndSettings();

  // Set the default size and position of the window
  testAndSetWindowSizeAndPos({info.windowSize.x, info.windowSize.y});

  // Initialize GLFW and create the window only if not headless
  if(!m_headless)
  {
    initGlfw(info);
  }

  // Used for creating single-time command buffers
  createTransientCommandPool();

  // Create a descriptor pool for creating descriptor set in the application
  createDescriptorPool();

  // Create the swapchain
  if(!m_headless)
  {
    nvvk::Swapchain::InitInfo swapChainInit{
        .physicalDevice        = m_physicalDevice,
        .device                = m_device,
        .queue                 = m_queues[0],
        .surface               = m_surface,
        .cmdPool               = m_transientCmdPool,
        .preferredVsyncOffMode = info.preferredVsyncOffMode,
        .preferredVsyncOnMode  = info.preferredVsyncOnMode,
    };

    // We do some custom error-handling here to provide additional information
    // about the reason creating the swapchain failed.
    const VkResult result = m_swapchain.init(swapChainInit);
    if(VK_SUCCESS != result)
    {
      reportSwapchainDiagnostics(m_instance, swapChainInit);
      // So that this is treated the same way as other NVVK_CHECK errors:
      nvvk::CheckError::getInstance().check(result, "m_swapchain.init(swapChainInit)", __FILE__, __LINE__);
    }
    // Update the window size to the actual size of the surface
    NVVK_CHECK(m_swapchain.initResources(m_windowSize, m_vsyncWanted));

    // Create what is needed to submit the scene for each frame in-flight
    createFrameSubmission(m_swapchain.getMaxFramesInFlight());
  }
  else
  {
    // In headless mode, there's only 2 pipeline stages (CPU and GPU, no display),
    // so we double instead of triple-buffer.
    createFrameSubmission(2);
  }

  // Set up the resource free queue
  resetFreeQueue(getFrameCycleSize());

  // Initialize Dear ImGui
  setupImGuiVulkanBackend(info.imguiConfigFlags);
}

void nvapp::Application::initGlfw(ApplicationCreateInfo& info)
{
  glfwInit();

  // Create the GLFW Window
  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
  glfwWindowHint(GLFW_SCALE_TO_MONITOR, GLFW_TRUE);  // Aware of DPI scaling
  m_windowHandle = glfwCreateWindow(m_windowSize.width, m_windowSize.height, info.name.c_str(), nullptr, nullptr);
  glfwSetWindowSize(m_windowHandle, m_windowSize.width, m_windowSize.height);  // Sets the size of the window using the DPI scaling
  glfwSetWindowPos(m_windowHandle, m_winPos.x, m_winPos.y);

  // Create the window surface
  NVVK_CHECK(glfwCreateWindowSurface(m_instance, m_windowHandle, nullptr, reinterpret_cast<VkSurfaceKHR*>(&m_surface)));
  NVVK_DBG_NAME(m_surface);

  // Set the Drop callback
  glfwSetWindowUserPointer(m_windowHandle, this);
  glfwSetDropCallback(m_windowHandle, &dropCb);
}

//-----------------------------------------------------------------------
// Shutdown the application
// This will destroy all resources and clean up the application.
void nvapp::Application::deinit()
{
  // Query the size/pos of the window, such that it get persisted
  if(!m_headless)
  {
    glm::ivec2 winSize{};
    glfwGetWindowSize(m_windowHandle, &winSize.x, &winSize.y);
    m_winSize = {uint32_t(winSize.x), uint32_t(winSize.y)};
    glfwGetWindowPos(m_windowHandle, &m_winPos.x, &m_winPos.y);
  }

  // This will call the onDetach of the elements
  for(std::shared_ptr<IAppElement>& e : m_elements)
  {
    e->onDetach();
  }

  // This avoids ImGui to access destroyed elements (Handler for example)
  if(!m_headless)
  {
    ImGui::SaveIniSettingsToDisk(m_iniFilename.c_str());
  }
  ImGui::GetIO().IniFilename = nullptr;  // Don't save the ini file again

  // Destroy the elements
  m_elements.clear();

  NVVK_CHECK(vkDeviceWaitIdle(m_device));

  // Clean pending
  resetFreeQueue(0);

  // ImGui cleanup
  ImGui_ImplVulkan_Shutdown();
  if(!m_headless)
  {
    ImGui_ImplGlfw_Shutdown();
    m_swapchain.deinit();
  }

  // Frame info
  for(size_t i = 0; i < m_frameData.size(); i++)
  {
    vkFreeCommandBuffers(m_device, m_frameData[i].cmdPool, 1, &m_frameData[i].cmdBuffer);
    vkDestroyCommandPool(m_device, m_frameData[i].cmdPool, nullptr);
  }
  vkDestroySemaphore(m_device, m_frameTimelineSemaphore, nullptr);
  ImGui::DestroyContext();

  if(ImPlot::GetCurrentContext() != nullptr)
  {
    ImPlot::DestroyContext();
  }

  vkDestroyCommandPool(m_device, m_transientCmdPool, nullptr);
  vkDestroyDescriptorPool(m_device, m_descriptorPool, nullptr);

  if(!m_headless)
  {
    vkDestroySurfaceKHR(m_instance, m_surface, nullptr);

    // Glfw cleanup
    glfwDestroyWindow(m_windowHandle);
    glfwTerminate();
  }
}

void nvapp::Application::addElement(const std::shared_ptr<IAppElement>& layer)
{
  m_elements.emplace_back(layer);
  layer->onAttach(this);
}

void nvapp::Application::setVsync(bool v)
{
  m_vsyncWanted = v;
  m_swapchain.requestRebuild();
}

VkCommandBuffer nvapp::Application::createTempCmdBuffer() const
{
  VkCommandBuffer cmd{};
  NVVK_CHECK(nvvk::beginSingleTimeCommands(cmd, m_device, m_transientCmdPool));
  return cmd;
}

void nvapp::Application::submitAndWaitTempCmdBuffer(VkCommandBuffer cmd)
{
  NVVK_CHECK(nvvk::endSingleTimeCommands(cmd, m_device, m_transientCmdPool, m_queues[0].queue));
}

void nvapp::Application::onFileDrop(const std::filesystem::path& filename)
{
  for(std::shared_ptr<IAppElement>& e : m_elements)
  {
    e->onFileDrop(filename);
  }
}

void nvapp::Application::close()
{
  if(m_headless)
  {
    m_headlessClose = true;
  }
  else
  {
    glfwSetWindowShouldClose(m_windowHandle, true);
  }
}

//-----------------------------------------------------------------------
// Main loop of the application
// It will run until the window is closed.
// Call all onUIRender() and onRender() for each element.
//
void nvapp::Application::run()
{
  LOGI("Running application\n");
  // Re-load ImGui settings from disk, as there might be application elements with settings to restore.
  ImGui::LoadIniSettingsFromDisk(m_iniFilename.c_str());

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
    // We add a delay before polling to reduce latency.
    if(m_vsyncWanted)
    {
      m_framePacer.pace();
    }
    glfwPollEvents();

    // Skip rendering when minimized
    if(glfwGetWindowAttrib(m_windowHandle, GLFW_ICONIFIED) == GLFW_TRUE)
    {
      ImGui_ImplGlfw_Sleep(10);
      continue;
    }

    // Begin New Frame for ImGui
    ImGui_ImplVulkan_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    // Setup ImGui Docking and UI
    setupImguiDock();
    if(m_useMenubar && ImGui::BeginMainMenuBar())
    {
      for(std::shared_ptr<IAppElement>& e : m_elements)
      {
        e->onUIMenu();
      }
      ImGui::EndMainMenuBar();
    }

    // Handle Viewport Updates
    VkExtent2D         viewportSize = m_windowSize;
    const ImGuiWindow* viewport     = ImGui::FindWindowByName("Viewport");
    if(viewport)
    {
      viewportSize = {uint32_t(viewport->Size.x), uint32_t(viewport->Size.y)};
      ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
      ImGui::Begin("Viewport");
      ImGui::End();
      ImGui::PopStyleVar();
    }

    // Update viewport if size changed
    if(m_viewportSize.width != viewportSize.width || m_viewportSize.height != viewportSize.height)
    {
      onViewportSizeChange(viewportSize);
    }

    // Handle Screenshot Requests
    if(m_screenShotRequested && (m_frameRingCurrent == m_screenShotFrame))
    {
      saveScreenShot(m_screenShotFilename, k_imageQuality);
      m_screenShotRequested = false;
    }

    // Frame Resource Preparation
    if(prepareFrameResources())
    {
      // Free resources from previous frame
      freeResourcesQueue();

      // Prepare Frame Synchronization
      prepareFrameToSignal(m_swapchain.getMaxFramesInFlight());

      // Record Commands
      VkCommandBuffer cmd = beginCommandRecording();
      drawFrame(cmd);            // Call onUIRender() and onRender() for each element
      renderToSwapchain(cmd);    // Render ImGui to swapchain
      addSwapchainSemaphores();  // Setup synchronization
      endFrame(cmd, m_swapchain.getMaxFramesInFlight());

      // Present Frame
      presentFrame();  // This can also trigger swapchain rebuild

      // Advance Frame
      advanceFrame(m_swapchain.getMaxFramesInFlight());
    }

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
void nvapp::Application::setupImguiDock()
{
  const ImGuiDockNodeFlags dockFlags = ImGuiDockNodeFlags_PassthruCentralNode | ImGuiDockNodeFlags_NoDockingInCentralNode;
  ImGuiID dockID = ImGui::DockSpaceOverViewport(0, ImGui::GetMainViewport(), dockFlags);
  // Docking layout, must be done only if it doesn't exist
  if(!ImGui::DockBuilderGetNode(dockID)->IsSplitNode() && !ImGui::FindWindowByName("Viewport"))
  {
    ImGui::DockBuilderDockWindow("Viewport", dockID);  // Dock "Viewport" to  central node
    ImGui::DockBuilderGetCentralNode(dockID)->LocalFlags |= ImGuiDockNodeFlags_NoTabBar;  // Remove "Tab" from the central node
    if(m_dockSetup)
    {
      // This override allow to create the layout of windows by default.
      m_dockSetup(dockID);
    }
    else
    {
      ImGuiID leftID = ImGui::DockBuilderSplitNode(dockID, ImGuiDir_Left, 0.2f, nullptr, &dockID);  // Split the central node
      ImGui::DockBuilderDockWindow("Settings", leftID);  // Dock "Settings" to the left node
    }
  }
}

//-----------------------------------------------------------------------
// Call this function if the viewport size changes
// This happens when the window is resized, or when the ImGui viewport window is resized.
//
void nvapp::Application::onViewportSizeChange(VkExtent2D size)
{
  // Check for DPI scaling and adjust the font size
  float xscale, yscale;
  glfwGetWindowContentScale(m_windowHandle, &xscale, &yscale);
  ImGui::GetIO().FontGlobalScale *= xscale / m_dpiScale;
  m_dpiScale = xscale;

  m_viewportSize = size;
  // Recreate the G-Buffer to the size of the viewport
  NVVK_CHECK(vkQueueWaitIdle(m_queues[0].queue));
  {
    VkCommandBuffer cmd{};
    NVVK_CHECK(nvvk::beginSingleTimeCommands(cmd, m_device, m_transientCmdPool));
    // Call the implementation of the UI rendering
    for(std::shared_ptr<IAppElement>& e : m_elements)
    {
      e->onResize(cmd, m_viewportSize);
    }
    NVVK_CHECK(nvvk::endSingleTimeCommands(cmd, m_device, m_transientCmdPool, m_queues[0].queue));
  }
}

//-----------------------------------------------------------------------
// Main frame rendering function
// - Acquire the image to render into
// - Call onUIRender() for each element
// - Call onRender() for each element
// - Render the ImGui UI
// - Present the image to the screen
//
void nvapp::Application::drawFrame(VkCommandBuffer cmd)
{
  // Reset the extra semaphores and command buffers
  m_waitSemaphores.clear();
  m_signalSemaphores.clear();
  m_commandBuffers.clear();


  // Call UI rendering for each element
  for(std::shared_ptr<IAppElement>& e : m_elements)
  {
    e->onUIRender();
  }

  // This is creating the data to draw the UI (not on GPU yet)
  ImGui::Render();

  // Call onPreRender for each element with the command buffer of the frame
  for(std::shared_ptr<IAppElement>& e : m_elements)
  {
    e->onPreRender();
  }

  // Call onRender for each element with the command buffer of the frame
  for(std::shared_ptr<IAppElement>& e : m_elements)
  {
    e->onRender(cmd);
  }
}

void nvapp::Application::renderToSwapchain(VkCommandBuffer cmd)
{

  // Start rendering to the swapchain
  beginDynamicRenderingToSwapchain(cmd);
  {
    nvvk::DebugUtil::ScopedCmdLabel scopedCmdLabel(cmd, "ImGui");
    // The ImGui draw commands are recorded to the command buffer, which includes the display of our GBuffer image
    ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cmd);
  }
  endDynamicRenderingToSwapchain(cmd);
}

//-----------------------------------------------------------------------
// prepareFrameResources is the first step in the rendering process.
// It looks if the swapchain require rebuild, which happens when the window is resized.
// It acquires the image from the swapchain to render into.
///
bool nvapp::Application::prepareFrameResources()
{
  if(m_swapchain.needRebuilding())
  {
    NVVK_CHECK(m_swapchain.reinitResources(m_windowSize, m_vsyncWanted));
  }

  waitForFrameCompletion();  // Wait until GPU has finished processing

  VkResult result = m_swapchain.acquireNextImage(m_device);
  return (result == VK_SUCCESS || result == VK_SUBOPTIMAL_KHR);  // Continue only if we got a valid image
}

//-----------------------------------------------------------------------
// Begin the command buffer recording for the frame
// It resets the command pool to reuse the command buffer for recording new rendering commands for the current frame.
// and it returns the command buffer for the frame.
VkCommandBuffer nvapp::Application::beginCommandRecording()
{
  // Get the frame data for the current frame in the ring buffer
  FrameData& frame = m_frameData[m_frameRingCurrent];

  // Reset the command pool to reuse the command buffer for recording new rendering commands for the current frame.
  NVVK_CHECK(vkResetCommandPool(m_device, frame.cmdPool, 0));
  VkCommandBuffer cmd = frame.cmdBuffer;

  // Begin the command buffer recording for the frame
  const VkCommandBufferBeginInfo beginInfo{.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                                           .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
  NVVK_CHECK(vkBeginCommandBuffer(cmd, &beginInfo));

  return cmd;
}

//-----------------------------------------------------------------------
// Adds the semaphores for the swapchain to the list of semaphores to wait for and signal.
//
void nvapp::Application::addSwapchainSemaphores()
{
  // Prepare to submit the current frame for rendering
  // First add the swapchain semaphore to wait for the image to be available.
  m_waitSemaphores.push_back({
      .sType     = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
      .semaphore = m_swapchain.getImageAvailableSemaphore(),
      .stageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
  });
  m_signalSemaphores.push_back({
      .sType     = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
      .semaphore = m_swapchain.getRenderFinishedSemaphore(),
      .stageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,  // Ensure everything is done before presenting
  });
}

//-----------------------------------------------------------------------
// End the frame by submitting the command buffer to the GPU
// Adds binary semaphores to wait for the image to be available and signal when rendering is done.
// Adds the timeline semaphore to signal when the frame is completed.
// Moves to the next frame.
//
void nvapp::Application::endFrame(VkCommandBuffer cmd, uint32_t frameInFlights)
{
  // Ends recording of commands for the frame
  NVVK_CHECK(vkEndCommandBuffer(cmd));


  // Get the frame data for the current frame in the ring buffer
  FrameData& frame = m_frameData[m_frameRingCurrent];

  // Add timeline semaphore to signal when GPU completes this frame
  // The color attachment output stage is used since that's when the frame is fully rendered
  m_signalSemaphores.push_back({
      .sType     = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
      .semaphore = m_frameTimelineSemaphore,
      .value     = frame.frameNumber,
      .stageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,  // Wait that everything is completed
  });

  // Adding the command buffer of the frame to the list of command buffers to submit
  // Note: extra command buffers could have been added to the list from other parts of the application (elements)
  m_commandBuffers.push_back({.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO, .commandBuffer = cmd});

  // Populate the submit info to synchronize rendering and send the command buffer
  const VkSubmitInfo2 submitInfo{
      .sType                    = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
      .waitSemaphoreInfoCount   = uint32_t(m_waitSemaphores.size()),    //
      .pWaitSemaphoreInfos      = m_waitSemaphores.data(),              // Wait for the image to be available
      .commandBufferInfoCount   = uint32_t(m_commandBuffers.size()),    //
      .pCommandBufferInfos      = m_commandBuffers.data(),              // Command buffer to submit
      .signalSemaphoreInfoCount = uint32_t(m_signalSemaphores.size()),  //
      .pSignalSemaphoreInfos    = m_signalSemaphores.data(),            // Signal when rendering is finished
  };

  // Submit the command buffer to the GPU and signal when it's done
  NVVK_CHECK(vkQueueSubmit2(m_queues[0].queue, 1, &submitInfo, nullptr));
}

//-----------------------------------------------------------------------
// The presentFrame function is the last step in the rendering process.
// It presents the image to the screen and moves to the next frame.
//
void nvapp::Application::presentFrame()
{
  // Present the image
  m_swapchain.presentFrame(m_queues[0].queue);
}

//-----------------------------------------------------------------------
//
void nvapp::Application::advanceFrame(uint32_t frameInFlights)
{
  // Move to the next frame
  m_frameRingCurrent = (m_frameRingCurrent + 1) % frameInFlights;
}

//-----------------------------------------------------------------------
//
void nvapp::Application::waitForFrameCompletion() const
{
  // Wait until GPU has finished processing the frame that was using these resources previously (numFramesInFlight frames ago)
  const VkSemaphoreWaitInfo waitInfo = {
      .sType          = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
      .semaphoreCount = 1,
      .pSemaphores    = &m_frameTimelineSemaphore,
      .pValues        = &m_frameData[m_frameRingCurrent].frameNumber,
  };
  vkWaitSemaphores(m_device, &waitInfo, std::numeric_limits<uint64_t>::max());
}


//-----------------------------------------------------------------------
// We are using dynamic rendering, which is a more flexible way to render to the swapchain image.
//
void nvapp::Application::beginDynamicRenderingToSwapchain(VkCommandBuffer cmd) const
{
  // Image to render to
  const VkRenderingAttachmentInfo colorAttachment{
      .sType       = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
      .imageView   = m_swapchain.getImageView(),
      .imageLayout = VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL,
      .loadOp      = VK_ATTACHMENT_LOAD_OP_CLEAR,   // Clear the image (see clearValue)
      .storeOp     = VK_ATTACHMENT_STORE_OP_STORE,  // Store the image (keep the image)
      .clearValue  = {{{0.0f, 0.0f, 0.0f, 1.0f}}},
  };

  // Details of the dynamic rendering
  const VkRenderingInfo renderingInfo{
      .sType                = VK_STRUCTURE_TYPE_RENDERING_INFO,
      .renderArea           = {{0, 0}, m_windowSize},
      .layerCount           = 1,
      .colorAttachmentCount = 1,
      .pColorAttachments    = &colorAttachment,
  };

  // Transition the swapchain image to the color attachment layout, needed when using dynamic rendering
  nvvk::cmdImageMemoryBarrier(cmd, {m_swapchain.getImage(), VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL});

  vkCmdBeginRendering(cmd, &renderingInfo);
}

//-----------------------------------------------------------------------
// End of dynamic rendering.
// The image is transitioned back to the present layout, and the rendering is ended.
//
void nvapp::Application::endDynamicRenderingToSwapchain(VkCommandBuffer cmd)
{
  vkCmdEndRendering(cmd);

  // Transition the swapchain image back to the present layout
  nvvk::cmdImageMemoryBarrier(cmd, {m_swapchain.getImage(), VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR});
}

//-----------------------------------------------------------------------
// This is the headless version of the run loop.
// It will render the scene for the number of frames specified in the headlessFrameCount.
// It will call onUIRender() and onRender() for each element.
//
void nvapp::Application::headlessRun()
{
  nvutils::ScopedTimer st(__FUNCTION__);
  m_viewportSize = m_windowSize;

  // Set the display for Imgui
  ImGuiIO& io      = ImGui::GetIO();
  io.DisplaySize.x = float(m_viewportSize.width);
  io.DisplaySize.y = float(m_viewportSize.height);

  // Make the size has been communicated everywhere
  VkCommandBuffer cmd{};
  NVVK_CHECK(nvvk::beginSingleTimeCommands(cmd, m_device, m_transientCmdPool));
  for(std::shared_ptr<IAppElement>& e : m_elements)
  {
    e->onResize(cmd, m_viewportSize);
  }
  NVVK_CHECK(nvvk::endSingleTimeCommands(cmd, m_device, m_transientCmdPool, m_queues[0].queue));

  // Need to render the UI twice: the first pass sets up the internal state and layout,
  // and the second pass finalizes the rendering with the updated state.
  {
    ImGui_ImplVulkan_NewFrame();
    ImGui::NewFrame();
    setupImguiDock();

    // Call UI rendering for each element
    for(std::shared_ptr<IAppElement>& e : m_elements)
    {
      e->onUIRender();
    }
    ImGui::EndFrame();
  }

  // Rendering n-times the scene
  for(uint32_t frameID = 0; frameID < m_headlessFrameCount && !m_headlessClose; frameID++)
  {
    ImGui_ImplVulkan_NewFrame();
    ImGui::NewFrame();  // Even if isn't directly used, helps advancing time if query

    waitForFrameCompletion();

    prepareFrameToSignal(getFrameCycleSize());

    VkCommandBuffer cmd = beginCommandRecording();  // Start the command buffer
    drawFrame(cmd);                                 // Call onUIRender() and onRender() for each element
    endFrame(cmd, getFrameCycleSize());             // End the frame and submit it
    advanceFrame(getFrameCycleSize());              // Advance to the next frame in the ring buffer

    ImGui::EndFrame();
  }
  ImGui::Render();  // This is creating the data to draw the UI (not on GPU yet)

  // At this point, everything has been rendered. Let it finish.
  vkDeviceWaitIdle(m_device);

  // Call back the application, such that it can do something with the rendered image
  for(std::shared_ptr<IAppElement>& e : m_elements)
  {
    e->onLastHeadlessFrame();
  }
}

//-----------------------------------------------------------------------
// Create a command pool for short lived operations
// The command pool is used to allocate command buffers.
// In the case of this sample, we only need one command buffer, for temporary execution.
//
void nvapp::Application::createTransientCommandPool()
{
  const VkCommandPoolCreateInfo commandPoolCreateInfo{
      .sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .flags            = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,  // Hint that commands will be short-lived
      .queueFamilyIndex = m_queues[0].familyIndex,
  };
  NVVK_CHECK(vkCreateCommandPool(m_device, &commandPoolCreateInfo, nullptr, &m_transientCmdPool));
  NVVK_DBG_NAME(m_transientCmdPool);
}

//-----------------------------------------------------------------------
// Creates a command pool (long life) and buffer for each frame in flight. Unlike the temporary command pool,
// these pools persist between frames and don't use VK_COMMAND_POOL_CREATE_TRANSIENT_BIT.
// Each frame gets its own command buffer which records all rendering commands for that frame.
//
void nvapp::Application::createFrameSubmission(uint32_t numFrames)
{
  assert(numFrames >= 2);  // Must have at least 2 frames in flight
  VkDevice device = m_device;

  m_frameData.resize(numFrames);

  // Initialize timeline semaphore with (numFrames - 1) to allow concurrent frame submission. See details in README.md
  const uint64_t initialValue = (static_cast<uint64_t>(numFrames) - 1);

  VkSemaphoreTypeCreateInfo timelineCreateInfo = {
      .sType         = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,
      .pNext         = nullptr,
      .semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE,
      .initialValue  = initialValue,
  };

  // Create timeline semaphore for GPU-CPU synchronization
  // This ensures resources aren't overwritten while still in use by the GPU
  const VkSemaphoreCreateInfo semaphoreCreateInfo{.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, .pNext = &timelineCreateInfo};
  NVVK_CHECK(vkCreateSemaphore(device, &semaphoreCreateInfo, nullptr, &m_frameTimelineSemaphore));
  NVVK_DBG_NAME(m_frameTimelineSemaphore);

  //Create command pools and buffers for each frame
  //Each frame gets its own command pool to allow parallel command recording while previous frames may still be executing on the GPU
  const VkCommandPoolCreateInfo cmdPoolCreateInfo{
      .sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .queueFamilyIndex = m_queues[0].familyIndex,
  };

  for(uint32_t i = 0; i < numFrames; i++)
  {
    m_frameData[i].frameNumber = i;  // Track frame index for synchronization

    // Separate pools allow independent reset/recording of commands while other frames are still in-flight
    NVVK_CHECK(vkCreateCommandPool(device, &cmdPoolCreateInfo, nullptr, &m_frameData[i].cmdPool));
    NVVK_DBG_NAME(m_frameData[i].cmdPool);

    const VkCommandBufferAllocateInfo commandBufferAllocateInfo = {
        .sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool        = m_frameData[i].cmdPool,
        .level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1,
    };
    NVVK_CHECK(vkAllocateCommandBuffers(device, &commandBufferAllocateInfo, &m_frameData[i].cmdBuffer));
    NVVK_DBG_NAME(m_frameData[i].cmdBuffer);
  }
}

//-----------------------------------------------------------------------
// The Descriptor Pool is used to allocate descriptor sets.
// Currently, ImGui only requires combined image samplers.
// We allocate up to m_maxTexturePool of them so that we can display additional
// images using ImGui_ImplVulkan_AddTexture().
//
void nvapp::Application::createDescriptorPool()
{
  const std::array<VkDescriptorPoolSize, 1> poolSizes{
      {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, m_maxTexturePool},
  };

  const VkDescriptorPoolCreateInfo poolInfo = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
      .flags = VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT |  //  allows descriptor sets to be updated after they have been bound to a command buffer
               VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,  // individual descriptor sets can be freed from the descriptor pool
      .maxSets       = m_maxTexturePool,  // Allowing to create many sets (ImGui uses this for textures)
      .poolSizeCount = uint32_t(poolSizes.size()),
      .pPoolSizes    = poolSizes.data(),
  };
  NVVK_CHECK(vkCreateDescriptorPool(m_device, &poolInfo, nullptr, &m_descriptorPool));
  NVVK_DBG_NAME(m_descriptorPool);
}

//-----------------------------------------------------------------------
// Initializes the ImGui context, sets up the settings handler for window
// size and position, loads the ImGui .ini file, and sets up fonts and style.
// This function should be called before any logic that depends on the ImGui
// context or the loaded window settings.
//
void nvapp::Application::initializeImGuiContextAndSettings()
{

  nvgui::setStyle(false);

  m_settingsHandler.setHandlerName("Application");
  m_settingsHandler.setSetting("Size", &m_winSize);
  m_settingsHandler.setSetting("Pos", &m_winPos);
  m_settingsHandler.addImGuiHandler();

  // Load the settings from the ini file
  ImGui::LoadIniSettingsFromDisk(m_iniFilename.c_str());

  ImGuiIO& io = ImGui::GetIO();
  // Set the ini file name
  io.IniFilename = m_iniFilename.c_str();

  // Initialize fonts
  nvgui::addDefaultFont();
  io.FontDefault = nvgui::getDefaultFont();
  nvgui::addMonospaceFont();
}


//-----------------------------------------------------------------------
// Sets up the ImGui Vulkan and GLFW backends, and initializes ImPlot context.
// Assumes the ImGui context and fonts are already initialized.
//
void nvapp::Application::setupImGuiVulkanBackend(ImGuiConfigFlags configFlags)
{
  static VkFormat imageFormats = VK_FORMAT_B8G8R8A8_UNORM;  // Must be static for ImGui_ImplVulkan_InitInfo

  ImGuiIO& io    = ImGui::GetIO();
  io.ConfigFlags = configFlags;
  if(m_headless)
  {
    io.ConfigFlags &= ~ImGuiConfigFlags_ViewportsEnable;  // In headless mode, we don't allow other viewport
  }

  if(!m_headless)
  {
    ImGui_ImplGlfw_InitForVulkan(m_windowHandle, true);
    imageFormats = m_swapchain.getImageFormat();
  }

  // ImGui Initialization for Vulkan
  ImGui_ImplVulkan_InitInfo initInfo = {
      .ApiVersion                  = VK_API_VERSION_1_4,
      .Instance                    = m_instance,
      .PhysicalDevice              = m_physicalDevice,
      .Device                      = m_device,
      .QueueFamily                 = m_queues[0].familyIndex,
      .Queue                       = m_queues[0].queue,
      .DescriptorPool              = m_descriptorPool,
      .MinImageCount               = 2U,
      .ImageCount                  = std::max(m_swapchain.getMaxFramesInFlight(), 2U),
      .UseDynamicRendering         = true,
      .PipelineRenderingCreateInfo =  // Dynamic rendering
      {
          .sType                   = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
          .colorAttachmentCount    = 1,
          .pColorAttachmentFormats = &imageFormats,
      },
  };
  ImGui_ImplVulkan_Init(&initInfo);
}


void nvapp::Application::saveImageToFile(VkImage srcImage, VkExtent2D imageSize, const std::filesystem::path& filename, int quality)
{
  VkDevice         device         = m_device;
  VkPhysicalDevice physicalDevice = m_physicalDevice;
  VkImage          dstImage       = {};
  VkDeviceMemory   dstImageMemory = {};
  VkCommandBuffer  cmd            = createTempCmdBuffer();

  VkFormat format = VK_FORMAT_R8G8B8A8_UNORM;
  if(filename.extension() == ".hdr")
  {
    format = VK_FORMAT_R32G32B32A32_SFLOAT;
  }
  nvvk::imageToLinear(cmd, device, physicalDevice, srcImage, imageSize, dstImage, dstImageMemory, format);
  submitAndWaitTempCmdBuffer(cmd);

  nvvk::saveImageToFile(device, dstImage, dstImageMemory, imageSize, filename, quality);

  // Clean up resources
  vkUnmapMemory(device, dstImageMemory);
  vkFreeMemory(device, dstImageMemory, nullptr);
  vkDestroyImage(device, dstImage, nullptr);
}


// Record that a screenshot is requested, and will be saved after a full
// frame cycle loop (so that ImGui has time to clear the menu).
void nvapp::Application::screenShot(const std::filesystem::path& filename, int quality)
{
  m_screenShotRequested = true;
  m_screenShotFilename  = filename;
  // Making sure the screenshot is taken after the swapchain loop (remove the menu after click)
  m_screenShotFrame = (m_frameRingCurrent - 1 + m_swapchain.getMaxFramesInFlight()) % m_swapchain.getMaxFramesInFlight();
}

// Save the current swapchain image to a file
void nvapp::Application::saveScreenShot(const std::filesystem::path& filename, int quality)
{
  VkExtent2D     size     = m_windowSize;
  VkImage        srcImage = m_swapchain.getImage();
  VkImage        dstImage;
  VkDeviceMemory dstImageMemory;

  vkDeviceWaitIdle(m_device);
  VkCommandBuffer cmd = createTempCmdBuffer();
  nvvk::cmdImageMemoryBarrier(cmd, {srcImage, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, VK_IMAGE_LAYOUT_GENERAL});
  nvvk::imageToLinear(cmd, m_device, m_physicalDevice, srcImage, size, dstImage, dstImageMemory, VK_FORMAT_R8G8B8A8_UNORM);
  nvvk::cmdImageMemoryBarrier(cmd, {srcImage, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR});
  submitAndWaitTempCmdBuffer(cmd);

  nvvk::saveImageToFile(m_device, dstImage, dstImageMemory, size, filename, quality);

  // Clean up resources
  vkUnmapMemory(m_device, dstImageMemory);
  vkFreeMemory(m_device, dstImageMemory, nullptr);
  vkDestroyImage(m_device, dstImage, nullptr);
}


void nvapp::Application::submitResourceFree(std::function<void()>&& func)
{
  if(m_frameRingCurrent < m_resourceFreeQueue.size())
  {
    m_resourceFreeQueue[m_frameRingCurrent].emplace_back(func);
  }
  else
  {
    func();
  }
}

void nvapp::Application::resetFreeQueue(uint32_t size)
{
  vkDeviceWaitIdle(m_device);

  for(auto& queue : m_resourceFreeQueue)
  {
    // Free resources in queue
    for(auto& func : queue)
    {
      func();
    }
    queue.clear();
  }
  m_resourceFreeQueue.clear();
  m_resourceFreeQueue.resize(size);
}

// This is called to free all resources that are not used anymore
// By using the frameRingCurrent we can free resources that are not used anymore
void nvapp::Application::freeResourcesQueue()
{
  for(auto& func : m_resourceFreeQueue[m_frameRingCurrent])
  {
    func();  // Free resources in queue
  }
  m_resourceFreeQueue[m_frameRingCurrent].clear();
}

void nvapp::Application::addWaitSemaphore(const VkSemaphoreSubmitInfo& wait)
{
  m_waitSemaphores.push_back(wait);
}
void nvapp::Application::addSignalSemaphore(const VkSemaphoreSubmitInfo& signal)
{
  m_signalSemaphores.push_back(signal);
}

// Calculate the signal value for when this frame completes
// Signal value = current frame number + numFramesInFlight
// Example with 3 frames in flight:
//   Frame 0 signals value 3 (allowing Frame 3 to start when complete)
//   Frame 1 signals value 4 (allowing Frame 4 to start when complete)
void nvapp::Application::prepareFrameToSignal(int32_t numFramesInFlight)
{
  m_frameData[m_frameRingCurrent].frameNumber += numFramesInFlight;
}

//Returning the frame semaphore and the value that will be signaled when the frame completes
nvvk::SemaphoreInfo nvapp::Application::getFrameSignalSemaphore() const
{
  return {m_frameTimelineSemaphore, m_frameData[m_frameRingCurrent].frameNumber};
}

void nvapp::Application::prependCommandBuffer(const VkCommandBufferSubmitInfo& cmd)
{
  m_commandBuffers.push_back(cmd);
}


//-----------------------------------------------------------------------
// Helpers


void nvapp::Application::testAndSetWindowSizeAndPos(const glm::uvec2& winSize)
{
  bool centerWindow = false;
  // If winSize is provided, use it
  if(winSize.x != 0 && winSize.y != 0)
  {
    m_winSize    = winSize;
    centerWindow = true;  // When the window size is requested, it will be centered
  }

  // If m_winSize is still (0,0), set defaults
  // Could be not zero if the user set it in the settings (see loading of the ini file)
  if(m_winSize.x == 0 && m_winSize.y == 0)
  {
    if(m_headless)
    {
      m_winSize = {800, 600};
    }
    else
    {
      // Get 80% of primary monitor
      const GLFWvidmode* mode = glfwGetVideoMode(glfwGetPrimaryMonitor());
      m_winSize.x             = static_cast<int>(mode->width * 0.8f);
      m_winSize.y             = static_cast<int>(mode->height * 0.8f);
    }
    // Center the window
    if(!m_headless)
    {
      int monX, monY;
      glfwGetMonitorPos(glfwGetPrimaryMonitor(), &monX, &monY);
      const GLFWvidmode* mode = glfwGetVideoMode(glfwGetPrimaryMonitor());
      m_winPos.x              = monX + (mode->width - m_winSize.x) / 2;
      m_winPos.y              = monY + (mode->height - m_winSize.y) / 2;
    }
  }
  else if(!m_headless)
  {
    // If m_winPos was retrieved, check if it is valid
    if(!isWindowPosValid(m_winPos) || centerWindow)
    {
      // Center the window
      int monX, monY;
      glfwGetMonitorPos(glfwGetPrimaryMonitor(), &monX, &monY);
      const GLFWvidmode* mode = glfwGetVideoMode(glfwGetPrimaryMonitor());
      m_winPos.x              = monX + (mode->width - m_winSize.x) / 2;
      m_winPos.y              = monY + (mode->height - m_winSize.y) / 2;
    }
  }

  m_windowSize = {m_winSize.x, m_winSize.y};
}

// Check if window position is within visible monitor bounds
bool nvapp::Application::isWindowPosValid(const glm::ivec2& winPos)
{
  int           monitorCount;
  GLFWmonitor** monitors = glfwGetMonitors(&monitorCount);

  // For each connected monitor
  for(int i = 0; i < monitorCount; i++)
  {
    GLFWmonitor*       monitor = monitors[i];
    const GLFWvidmode* mode    = glfwGetVideoMode(monitor);

    int monX, monY;
    glfwGetMonitorPos(monitor, &monX, &monY);

    // Check if window position is within this monitor's bounds
    // Add some margin to account for window decorations
    if(winPos.x >= monX && winPos.x < monX + mode->width && winPos.y >= monY && winPos.y < monY + mode->height)
    {
      return true;
    }
  }

  return false;
}
