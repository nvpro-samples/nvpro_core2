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

// The Direct3D 12 counterpart of main_vk.cpp: the same window, elements and
// animated viewport, on nvdx12app::Application instead of nvvkapp::Application.

#include <array>
#include <cassert>

#include <nvdx12app/application.hpp>
#include <nvdx12app/elem_default_menu.hpp>
#include <nvdx12app/elem_default_title.hpp>
#include <nvdx12app/elem_logger.hpp>
#include <nvdx12app/elem_profiler.hpp>
#include <nvdx12/check_error.hpp>
#include <nvdx12/context.hpp>
#include <nvdx12/descriptor_heap.hpp>
#include <nvdx12/profiler_dx12.hpp>
#include <nvdx12/queue.hpp>
#include <nvdx12/render_target.hpp>
#include <nvdx12/resource_allocator.hpp>
#include <nvutils/logger.hpp>
#include <nvutils/parameter_parser.hpp>

class SampleElement : public nvdx12app::IAppElement
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
    assert(m_info.parameterRegistry && "Info::parameterRegistry is required");
    assert(m_info.profilerManager && "Info::profilerManager is required");

    // let's add a command-line option to toggle animation
    m_info.parameterRegistry->add({"animate"}, &m_animate);
  }

  ~SampleElement() override = default;

  void onAttach(nvdx12app::Application* app) override
  {
    assert(app && "onAttach without an application");
    m_app = app;

    // Initialize core components
    NVDX12_CHECK(m_alloc.init({.pDevice = app->getDevice()}, app->getDescriptorHeap()));

    // The image the sample renders into and shows in the "Viewport" window.
    // It is cleared through its UAV, so it rests in UNORDERED_ACCESS.
    NVDX12_CHECK(m_viewportTarget.init({
        .alloc        = &m_alloc,
        .colorFormats = {DXGI_FORMAT_R16G16B16A16_FLOAT},
        .layouts      = {.color = D3D12_BARRIER_LAYOUT_UNORDERED_ACCESS},
        .debugName    = "Viewport",
    }));

    // Init profiler on the application's queue
    m_profilerTimeline = m_info.profilerManager->createTimeline({"graphics"});
    m_profilerGpuTimer.init(m_profilerTimeline, app->getDevice(), app->getQueue(), true);
  }

  void onDetach() override
  {
    m_viewportTarget.deinit();
    m_alloc.deinit();
    m_profilerGpuTimer.deinit();
    m_info.profilerManager->destroyTimeline(m_profilerTimeline);
  }

  void onResize(ID3D12GraphicsCommandList7* cmd, const glm::uvec2& size) override
  {
    NVDX12_CHECK(m_viewportTarget.update(cmd, {size.x, size.y}));
    // update() does not clear: paint it NVIDIA green, so a sample that is not
    // animating still shows something (the Vulkan sample uploads the same color).
    m_viewportTarget.cmdClear(cmd, kNvidiaGreen);
  }

  void onUIRender() override
  {
    ImGui::Begin("Settings");
    ImGui::Checkbox("Animated Viewport", &m_animate);
    ImGui::TextDisabled("%d FPS / %.3fms", static_cast<int>(ImGui::GetIO().Framerate), 1000.F / ImGui::GetIO().Framerate);

    // Add window information
    const glm::uvec2& viewportSize = m_app->getViewportSize();
    ImGui::Text("Viewport Size: %d x %d", viewportSize.x, viewportSize.y);

    ImGui::End();

    // Rendered image displayed fully in 'Viewport' window
    ImGui::Begin("Viewport");
    ImGui::Image(ImTextureID(m_viewportTarget.getUiImageView().ptr), ImGui::GetContentRegionAvail());
    ImGui::End();

    // Transitioned to SHADER_RESOURCE before the ImGui draw
    m_app->registerImguiImage(&m_viewportTarget.getColorImage());
  }

  void onPreRender() override { m_profilerTimeline->frameAdvance(); }

  void onRender(ID3D12GraphicsCommandList7* cmd) override
  {
    if(m_animate)
    {
      auto timerSection = m_profilerGpuTimer.cmdFrameSection(cmd, "Animation");

      std::array<float, 4> clearColor{0, 0, 0, 1};
      ImGui::ColorConvertHSVtoRGB((float)ImGui::GetTime() * 0.05f, 1, 1, clearColor[0], clearColor[1], clearColor[2]);
      m_viewportTarget.cmdClear(cmd, clearColor);
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
  static constexpr std::array<float, 4> kNvidiaGreen{0.46F, 0.72F, 0.F, 1.F};

  Info m_info{};
  bool m_animate = false;

  nvdx12::ResourceAllocator m_alloc{};
  nvdx12app::Application*   m_app{};
  nvdx12::RenderTarget      m_viewportTarget{};

  nvutils::ProfilerTimeline* m_profilerTimeline{};
  nvdx12::ProfilerGpuTimer   m_profilerGpuTimer{};
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
  std::shared_ptr<nvdx12app::ElementLogger> elementLogger = std::make_shared<nvdx12app::ElementLogger>(true);
  nvutils::Logger::getInstance().setLogCallback([elementLogger](nvutils::Logger::LogLevel logLevel, const std::string& text) {
    elementLogger->addLog(logLevel, "%s", text.c_str());
  });

  nvdx12::ContextInitInfo dxSetup{};

  // let's add a command-line option to enable/disable the debug layer
  parameterRegistry.add({"validation"}, &dxSetup.enableValidationLayers);
  parameterRegistry.add({"verbose"}, &dxSetup.verbose);
  // as well as an option to force the adapter based on canonical index
  parameterRegistry.add({"forcedevice"}, &dxSetup.forceGPU);

  // add all parameters to the parser
  parameterParser.add(parameterRegistry);

  // and then parse command line
  parameterParser.parse(argc, argv);

  nvdx12::Context dxContext;
  const HRESULT   contextResult = dxContext.init(dxSetup);
  if(FAILED(contextResult))
  {
    LOGE("Error in D3D12 context creation: %s\n", nvdx12::hresultToString(contextResult).c_str());
    return 1;
  }

  // The queue the frames are submitted to, and the heap ImGui's font and the
  // elements' images take slots of; both are shared with the application.
  nvdx12::QueueTimeline queue;
  NVDX12_CHECK(queue.init(dxContext.getDevice(), dxContext.getQueueInfo(0)));
  nvdx12::DescriptorHeap descriptorHeap;
  NVDX12_CHECK(descriptorHeap.init(dxContext.getDevice()));

  nvdx12app::ApplicationCreateInfo appInfo;
  appInfo.name           = "The Empty Example";
  appInfo.useMenu        = true;
  appInfo.context        = &dxContext;
  appInfo.queue          = &queue;
  appInfo.descriptorHeap = &descriptorHeap;
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
  nvdx12app::Application app;
  NVDX12_CHECK(app.init(appInfo));

  // add the sample main element
  app.addElement(sampleElement);
  app.addElement(std::make_shared<nvdx12app::ElementDefaultWindowTitle>());
  // add profiler element
  app.addElement(std::make_shared<nvdx12app::ElementProfiler>(&profilerManager));
  // add logger element
  app.addElement(elementLogger);

  LOGI("%s", "Wohoo let's run this sample!\n");

  // enter the main loop
  app.run();

  // Cleanup in reverse order
  app.deinit();
  descriptorHeap.deinit();
  queue.deinit();
  dxContext.deinit();

  return 0;
}
