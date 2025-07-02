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

// This example demonstrates a minimal Vulkan application using the NVIDIA
// Vulkan utility libraries. It creates a window displaying a single colored
// pixel that animates through the HSV color space.

#define VMA_IMPLEMENTATION

#include <backends/imgui_impl_vulkan.h>
#include <nvapp/application.hpp>
#include <nvapp/elem_profiler.hpp>
#include <nvapp/elem_logger.hpp>
#include <nvapp/elem_default_menu.hpp>
#include <nvapp/elem_default_title.hpp>
#include <nvutils/logger.hpp>
#include <nvvk/check_error.hpp>
#include <nvvk/context.hpp>
#include <nvvk/debug_util.hpp>
#include <nvvk/default_structs.hpp>
#include <nvvk/resource_allocator.hpp>
#include <nvvk/sampler_pool.hpp>
#include <nvvk/staging.hpp>
#include <nvvk/profiler_vk.hpp>
#include <nvutils/parameter_parser.hpp>

class SampleElement : public nvapp::IAppElement
{
public:
  struct Info
  {
    nvutils::ProfilerManager*   profilerManager{};
    nvutils::ParameterRegistry* parameterRegistry{};
  };


  SampleElement(const Info& info)
      : m_info(info)
  {
    // let's add a command-line option to toggle animation
    m_info.parameterRegistry->add({"animate"}, &m_animate);
  }

  ~SampleElement() override = default;

  void onAttach(nvapp::Application* app) override
  {
    m_app                                = app;
    VmaAllocatorCreateInfo allocatorInfo = {
        .flags          = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT,
        .physicalDevice = app->getPhysicalDevice(),
        .device         = app->getDevice(),
        .instance       = app->getInstance(),
    };

    // Initialize core components
    NVVK_CHECK(m_alloc.init(allocatorInfo));
    m_samplerPool.init(app->getDevice());
    m_stagingUploader.init(&m_alloc, true);

#if 0
    // VMA might report memory leaks for example:
    // "UNFREED ALLOCATION; Offset: 1158736; Size: 16; UserData: 0000000000000000; Name: allocID: 45; Type: BUFFER; Usage: 131107"
    // Then look for the leak name: "allocID: 45" and feed that ID to the following function.
    m_alloc.setLeakID(45);
    // You should get a breakpoint at the creation of the resource that was leaked.
#endif

    // Create a 1x1 Vulkan texture
    VkCommandBuffer   cmd       = m_app->createTempCmdBuffer();
    VkImageCreateInfo imageInfo = DEFAULT_VkImageCreateInfo;
    imageInfo.extent            = {1, 1, 1};
    imageInfo.format            = VK_FORMAT_R32G32B32A32_SFLOAT;
    imageInfo.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;  // Added transfer dst bit
    std::array<float, 4> imageData = {0.46F, 0.72F, 0, 1};                           // NVIDIA Green

    VkImageViewCreateInfo viewInfo = DEFAULT_VkImageViewCreateInfo;
    viewInfo.components            = {.a = VK_COMPONENT_SWIZZLE_ONE};  // Force alpha to 1.0
    NVVK_CHECK(m_alloc.createImage(m_viewportImage, imageInfo, viewInfo));
    NVVK_CHECK(m_samplerPool.acquireSampler(m_viewportImage.descriptor.sampler));

    NVVK_DBG_NAME(m_viewportImage.image);
    NVVK_DBG_NAME(m_viewportImage.descriptor.imageView);
    NVVK_DBG_NAME(m_viewportImage.descriptor.sampler);

    // upload image
    NVVK_CHECK(m_stagingUploader.appendImage(m_viewportImage, std::span<float>(imageData.data(), imageData.size()),
                                             VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL));
    m_stagingUploader.cmdUploadAppended(cmd);
    m_app->submitAndWaitTempCmdBuffer(cmd);
    m_stagingUploader.releaseStaging();

    // Add image to ImGui, for display
    m_imguiImage = ImGui_ImplVulkan_AddTexture(m_viewportImage.descriptor.sampler, m_viewportImage.descriptor.imageView,
                                               VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

    // Init profiler with a single queue
    m_profilerTimeline = m_info.profilerManager->createTimeline({"graphics"});
    m_profilerGpuTimer.init(m_profilerTimeline, app->getDevice(), app->getPhysicalDevice(), app->getQueue(0).familyIndex, true);
  }

  void onDetach() override
  {
    NVVK_CHECK(vkDeviceWaitIdle(m_app->getDevice()));

    ImGui_ImplVulkan_RemoveTexture(m_imguiImage);
    m_alloc.destroyImage(m_viewportImage);
    m_stagingUploader.deinit();
    m_samplerPool.deinit();
    m_alloc.deinit();
    m_profilerGpuTimer.deinit();
    m_info.profilerManager->destroyTimeline(m_profilerTimeline);
  }

  void onUIRender() override
  {
    ImGui::Begin("Settings");
    ImGui::Checkbox("Animated Viewport", &m_animate);
    ImGui::TextDisabled("%d FPS / %.3fms", static_cast<int>(ImGui::GetIO().Framerate), 1000.F / ImGui::GetIO().Framerate);

    // Add window information
    const VkExtent2D& viewportSize = m_app->getViewportSize();
    ImGui::Text("Viewport Size: %d x %d", viewportSize.width, viewportSize.height);

    ImGui::End();

    // Rendered image displayed fully in 'Viewport' window
    ImGui::Begin("Viewport");
    ImGui::Image((ImTextureID)m_imguiImage, ImGui::GetContentRegionAvail());
    ImGui::End();
  }

  void onPreRender() override { m_profilerTimeline->frameAdvance(); }

  void onRender(VkCommandBuffer cmd)
  {
    if(m_animate)
    {
      auto timerSection = m_profilerGpuTimer.cmdFrameSection(cmd, "Animation");

      VkClearColorValue clearColor{};
      ImGui::ColorConvertHSVtoRGB((float)ImGui::GetTime() * 0.05f, 1, 1, clearColor.float32[0], clearColor.float32[1],
                                  clearColor.float32[2]);
      VkImageSubresourceRange range = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
      nvvk::cmdImageMemoryBarrier(cmd, {m_viewportImage.image, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL});
      vkCmdClearColorImage(cmd, m_viewportImage.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &clearColor, 1, &range);
      nvvk::cmdImageMemoryBarrier(cmd, {m_viewportImage.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL});
    }
  }

  // Called if showMenu is true
  void onUIMenu() override
  {
    bool vsync = m_app->isVsync();

    if(ImGui::BeginMenu("File"))
    {
      if(ImGui::MenuItem("Exit", "Ctrl+Q"))
        m_app->close();
      ImGui::EndMenu();
    }
    if(ImGui::BeginMenu("View"))
    {
      ImGui::MenuItem("V-Sync", "Ctrl+Shift+V", &vsync);
      ImGui::EndMenu();
    }

    if(ImGui::IsKeyPressed(ImGuiKey_Q) && ImGui::IsKeyDown(ImGuiKey_LeftCtrl))
    {
      m_app->close();
    }

    if(ImGui::IsKeyPressed(ImGuiKey_V) && ImGui::IsKeyDown(ImGuiKey_LeftCtrl) && ImGui::IsKeyDown(ImGuiKey_LeftShift))
    {
      vsync = !vsync;
    }

    if(vsync != m_app->isVsync())
    {
      m_app->setVsync(vsync);
    }
  }

private:
  Info m_info;
  bool m_animate = false;

  nvvk::ResourceAllocator m_alloc{};
  nvapp::Application*     m_app{};
  nvvk::SamplerPool       m_samplerPool{};
  nvvk::StagingUploader   m_stagingUploader{};

  nvutils::ProfilerTimeline* m_profilerTimeline{};
  nvvk::ProfilerGpuTimer     m_profilerGpuTimer;

  nvvk::Image m_viewportImage{};

  VkDescriptorSet m_imguiImage{};
};


int main(int argc, char** argv)
{
  nvutils::ProfilerManager   profilerManager;
  nvutils::ParameterRegistry parameterRegistry;
  nvutils::ParameterParser   parameterParser;

  // setup sample element
  SampleElement::Info sampleInfo = {
      .profilerManager   = &profilerManager,
      .parameterRegistry = &parameterRegistry,
  };
  std::shared_ptr<SampleElement> sampleElement = std::make_shared<SampleElement>(sampleInfo);

  // setup logger element, `true` means shown by default
  // we add it early so outputs are captured early on, you might want to defer this to a later timer.
  std::shared_ptr<nvapp::ElementLogger> elementLogger = std::make_shared<nvapp::ElementLogger>(true);
  nvutils::Logger::getInstance().setLogCallback([&](nvutils::Logger::LogLevel logLevel, const std::string& text) {
    elementLogger->addLog(logLevel, "%s", text.c_str());
  });

  nvvk::ContextInitInfo vkSetup{
      .instanceExtensions = {VK_EXT_DEBUG_UTILS_EXTENSION_NAME},
      .deviceExtensions   = {{VK_KHR_SWAPCHAIN_EXTENSION_NAME}},
  };

  // let's add a command-line option to enable/disable validation layers
  parameterRegistry.add({"validation"}, &vkSetup.enableValidationLayers);
  parameterRegistry.add({"verbose"}, &vkSetup.verbose);
  // as well as an option to force the vulkan device based on canonical index
  parameterRegistry.add({"forcedevice"}, &vkSetup.forceGPU);

  // add all parameters to the parser
  parameterParser.add(parameterRegistry);

  // and then parse command line
  parameterParser.parse(argc, argv);

  nvvk::addSurfaceExtensions(vkSetup.instanceExtensions);
  nvvk::Context vkContext;
  if(vkContext.init(vkSetup) != VK_SUCCESS)
  {
    LOGE("Error in Vulkan context creation\n");
    return 1;
  }

  nvapp::ApplicationCreateInfo appInfo;
  appInfo.name           = "The Empty Example";
  appInfo.useMenu        = true;
  appInfo.instance       = vkContext.getInstance();
  appInfo.device         = vkContext.getDevice();
  appInfo.physicalDevice = vkContext.getPhysicalDevice();
  appInfo.queues         = vkContext.getQueueInfos();
  appInfo.dockSetup      = [](ImGuiID viewportID) {
    // right side panel container
    ImGuiID settingID = ImGui::DockBuilderSplitNode(viewportID, ImGuiDir_Right, 0.25F, nullptr, &viewportID);
    ImGui::DockBuilderDockWindow("Settings", settingID);

    // bottom panel container
    ImGuiID loggerID = ImGui::DockBuilderSplitNode(viewportID, ImGuiDir_Down, 0.35F, nullptr, &viewportID);
    ImGui::DockBuilderDockWindow("Log", loggerID);
    ImGuiID profilerID = ImGui::DockBuilderSplitNode(loggerID, ImGuiDir_Right, 0.4F, nullptr, &loggerID);
    ImGui::DockBuilderDockWindow("Profiler", profilerID);
  };

  // Create the application
  nvapp::Application app;
  app.init(appInfo);

  // add the sample main element
  app.addElement(sampleElement);
  app.addElement(std::make_shared<nvapp::ElementDefaultWindowTitle>());
  // add profiler element
  app.addElement(std::make_shared<nvapp::ElementProfiler>(&profilerManager));
  // add logger element
  app.addElement(elementLogger);

  LOGI("%s", "Wohoo let's run this sample!\n");

  // enter the main loop
  app.run();

  // Cleanup in reverse order
  app.deinit();
  vkContext.deinit();

  return 0;
}
