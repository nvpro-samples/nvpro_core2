/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <filesystem>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include <glm/glm.hpp>
#include <imgui/imgui.h>

#include <nvapp/application_base.hpp>
#include <nvdx12/commands.hpp>
#include <nvdx12/context.hpp>
#include <nvdx12/descriptor_heap.hpp>
#include <nvdx12/queue.hpp>
#include <nvdx12/resources.hpp>
#include <nvdx12/upload_ring.hpp>

/*-------------------------------------------------------------------------------------------------
# class nvdx12app::Application

> The Direct3D 12 application, the counterpart of the Vulkan nvapp::Application:
> nvapp::ApplicationBase plus the DXGI swapchain, the per-frame command lists,
> ImGui's DX12 renderer and screenshots.

To use the application,
* Fill the ApplicationCreateInfo with all the information,

Example:
````cpp
    nvdx12app::ApplicationCreateInfo appInfo;
    appInfo.name           = "Minimal Test";
    appInfo.windowSize     = {800, 600};
    appInfo.vSync          = false;
    appInfo.context        = &dxContext;       // nvdx12::Context
    appInfo.queue          = &queueTimeline;   // nvdx12::QueueTimeline on the direct queue
    appInfo.descriptorHeap = &descriptorHeap;  // nvdx12::DescriptorHeap the elements' images live in
````

* Attach elements to the application, the main rendering, and others like, camera, etc.
* Call run() to start the application.

Worth notice
* ::init() : creates the GLFW window and sets up the ImGui context the constructor created
             (nvapp::ApplicationBase, nvapp::detail::initContexts), then an UploadRing on the
             frame queue (cmdPushData, getUploadRing, ApplicationCreateInfo::uploadRingBytes),
             one command context per frame in flight, the flip-model swapchain and the ImGui
             DX12 renderer on the application's descriptor heap; temporary command lists
             (createTempCmdList / submitAndWaitTempCmdList) are pooled on demand
* ::deinit() : the opposite of init
* ::run() : while running, render the frame and present the frame. Check for resize, minimize window
            and other changes. In the loop, it calls onUpdate, onUIMenu, onUIRender, onPreRender,
            onRender and onPostRender for each connected element. See IAppElement and
            nvapp::IAppElementBase for details.
* The Application is the owner of the elements and calls those hooks for each of them.
* The Application is the owner of the ImGui context, and it will create the dockspace and the main menu.
* The Application is the owner of the GLFW window, and it will create the window and handle the events.
* The device, the queue and the descriptor heap are created by main() before the application and
    shared with it: ImGui's font atlas and the elements' preview images take slots of that heap.

The application itself does not render per se. It holds a command context per frame in flight,
the upload ring and the temporary command lists, and it calls ImGui rendering for DX12, but
that's it. Note that none of the samples render directly into the swapchain. Instead, they render
into an image, and the image is displayed in the ImGui window called "Viewport".

Per frame the elements see: onUpdate -> onUIMenu -> onUIRender -> onPreRender -> onRender(cmd) ->
ImGui draw into the back buffer -> submit -> onPostRender -> present; a minimized window stops
after onUpdate. onResize(cmd, size) records into a temporary command list that is submitted and
waited for. Images ImGui samples this frame are registered with registerImguiImage() and
transitioned to SHADER_RESOURCE before the ImGui draw.

Note: order of Elements can be important if one depends on the other. For example, a camera element
      should be added before the rendering sample, such that its matrices are updated before pulled by
      the renderer.


## Docking

The layout can be customized by providing a function to the ApplicationCreateInfo. It is called with the
dockspace's central node when no layout exists yet (none restored from the .ini file).

Example:
````
    // Setting up the layout of the application
    appInfo.dockSetup = [](ImGuiID viewportID) {
      ImGuiID settingID = ImGui::DockBuilderSplitNode(viewportID, ImGuiDir_Right, 0.2F, nullptr, &viewportID);
      ImGui::DockBuilderDockWindow("Settings", settingID);
    };
````
-------------------------------------------------------------------------------------------------*/

namespace nvdx12app {
// Forward declarations
class Application;

//-------------------------------------------------------------------------------------------------
// Interface for application elements that record into the frame's command list.
// API-agnostic elements (nvapp::IAppElementBase) can be added to the application as well.
struct IAppElement : nvapp::IAppElementBase
{
  using nvapp::IAppElementBase::onAttach;
  using nvapp::IAppElementBase::onResize;

  // Interface
  virtual void onAttach(Application* app);  // Called once at start; calls onAttach(nvapp::ApplicationBase*) by default
  virtual void onResize(ID3D12GraphicsCommandList7* cmd, const glm::uvec2& size) {}  // Called when the viewport size is changing
  virtual void onRender(ID3D12GraphicsCommandList7* cmd) {}  // For anything to render within a frame
};

// Application creation info
struct ApplicationCreateInfo
{
  // General
  std::string name{"D3D12_App"};  // Application name

  // Direct3D 12: shared objects created by main() before the application
  nvdx12::Context*       context{nullptr};          // Device, adapter, DXGI factory and feature table
  nvdx12::QueueTimeline* queue{nullptr};            // The direct queue the frames are submitted to
  nvdx12::DescriptorHeap* descriptorHeap{nullptr};  // Shader-visible heap; ImGui's font and the elements' images take slots
  // DXGI factory the swapchain is created from; the context factory when
  // null. Interposers (Streamline) hand in their proxy factory here.
  nvdx12::ComPtr<IDXGIFactory6> swapchainFactory;
  uint64_t uploadRingBytes{16ull << 20};  // Upload ring of the frame queue: push data beyond the root constants, transient uploads

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

  // Swapchain
  DXGI_FORMAT swapchainFormat{DXGI_FORMAT_R8G8B8A8_UNORM};  // R10G10B10A2_UNORM or R16G16B16A16_FLOAT for HDR output
  DXGI_COLOR_SPACE_TYPE colorSpace{DXGI_COLOR_SPACE_RGB_FULL_G22_NONE_P709};  // Requested; see getSwapchainColorSpace()
  uint32_t              framesInFlight{2};  // CPU-side double buffering; also the swapchain buffer count (at least two)
};


class Application : public nvapp::ApplicationBase
{
public:
  Application()           = default;
  ~Application() override = default;

  // Initialization and shutdown
  HRESULT init(ApplicationCreateInfo& info);
  void    deinit();

  // Application control: run indefinitely until close() is requested or the window closes
  using nvapp::ApplicationBase::run;

  // Adding engines
  void addElement(const std::shared_ptr<nvapp::IAppElementBase>& element) override;

  // Utilities
  void setVsync(bool v) override;  // Set V-Sync on or off; applied at the next Present, no swapchain rebuild

  // Swapchain format control
  void setPreferredSwapchainFormat(DXGI_FORMAT format, DXGI_COLOR_SPACE_TYPE colorSpace);  // Set format and request rebuild
  DXGI_FORMAT getSwapchainFormat() const { return m_swapchainFormat; }
  // The color space the swapchain presents in: the requested one when the display
  // accepts it, otherwise the format's DXGI default.
  DXGI_COLOR_SPACE_TYPE getSwapchainColorSpace() const { return m_colorSpace; }
  void                  requestSwapchainRebuild() { m_swapchainRebuild = true; }  // Rebuilt at the next frame boundary
  HWND                  getHwnd() const { return m_hwnd; }  // The window the swapchain presents to

  // Following three functions affect the current frame's submission.
  // Content is appended to vectors that are reset every frame.
  // A D3D12 fence value is assigned at submit, so there is no getFrameSignalSemaphore().
  void addWaitSemaphore(const nvdx12::SemaphoreInfo& wait);  // The queue waits for the fence value before the frame's lists
  void addSignalSemaphore(const nvdx12::SemaphoreInfo& signal);  // The queue signals the fence value after the frame's lists
  void prependCommandList(ID3D12CommandList* cmd);  // Executed before the command list that is provided to `onRender(cmd)`

  // These callbacks will be invoked by the CPU with the current frame number before
  // and after the Application's queue submit and swapchain present call.
  // They're most useful for setting latency markers (Reflex / PCL) when timing
  // frames with DLSS Frame Generation.
  using FrameMarkerCallback = std::function<void(uint64_t frameNumber)>;
  void setQueueSubmitStartCallback(FrameMarkerCallback&& callback) { m_queueSubmitStartCallback = std::move(callback); }
  void setQueueSubmitEndCallback(FrameMarkerCallback&& callback) { m_queueSubmitEndCallback = std::move(callback); }
  void setPresentStartCallback(FrameMarkerCallback&& callback) { m_presentStartCallback = std::move(callback); }
  void setPresentEndCallback(FrameMarkerCallback&& callback) { m_presentEndCallback = std::move(callback); }
  // This callback will be invoked by the CPU on the back buffer after ImGui drew
  // into it, before presentation. It's intended so DLSS Frame Generation can
  // generate extrapolated frames. The back buffer is in the RENDER_TARGET
  // layout on entry and must be in it on exit; the application then
  // transitions it to PRESENT (== COMMON, the legacy PRESENT state) for the present.
  using FrameGenerationCallback =
      std::function<void(ID3D12GraphicsCommandList7*, ID3D12Resource* backBuffer, DXGI_FORMAT, const glm::uvec2& size, uint64_t frameNumber)>;
  void setFrameGenerationCallback(FrameGenerationCallback&& callback)
  {
    m_frameGenerationCallback = std::move(callback);
  }

  // Push data beyond the root constants (kPushConstantsMaxBytes): copied into
  // the application's upload ring and bound as the push-data root SRV for the
  // chosen bind point, the counterpart of vkCmdPushDataEXT; false when the
  // ring has no room (nothing bound, skip the dispatch). The ring itself
  // serves any other transient upload (UploadRing::allocate, cmdUpdateBuffer).
  bool cmdPushData(ID3D12GraphicsCommandList* cmd, const void* data, uint32_t size, nvdx12::BindPoint bindPoint);
  nvdx12::UploadRing& getUploadRing() { return m_uploadRing; }

  // Utility to create a temporary command list, bound to the descriptor heap
  // (null on failure). Each call opens a list of its own, so several may be
  // open at once (one from inside onResize while the application records the
  // resize list); submitAndWaitTempCmdList closes, executes and waits for it,
  // and returns the first failure of the close, the submission, the wait or a
  // lost device. Not thread-safe.
  // From onPreRender to the frame's submission the upload ring defers
  // (UploadRing::beginDeferral): a temporary submit leaves the frame's
  // allocations in flight, and those of the temporary lists live until the
  // frame completes.
  ID3D12GraphicsCommandList7* createTempCmdList();
  HRESULT                     submitAndWaitTempCmdList(ID3D12GraphicsCommandList7* cmd);

  // The error that closed the application: a failed frame submission or
  // resize list, a lost device or an unusable swapchain; S_OK otherwise.
  // nvapp::Application exits the process in these cases; here run() returns
  // (headless runs still call onLastHeadlessFrame) and the caller decides.
  HRESULT getFailure() const { return m_failure; }

  // Getters
  inline nvdx12::Context*        getContext() const { return m_context; }
  inline ID3D12Device*           getDevice() const { return m_device; }
  inline ID3D12CommandQueue*     getQueue() const { return m_queue->getQueue(); }
  inline nvdx12::QueueTimeline*  getQueueTimeline() const { return m_queue; }
  inline nvdx12::DescriptorHeap* getDescriptorHeap() const { return m_heap; }
  inline IDXGISwapChain4*        getSwapchain() const { return m_swapchain.Get(); }
  inline uint64_t                getFrameNumber() const { return m_frameNumber; }  // Frames submitted since init()
  ID3D12Resource*                getCurrentBackBuffer() const;                     // Null without a swapchain
  // The current back buffer as an image in its PRESENT (== COMMON) layout and
  // its render-target view, the swapchain image nvapp::Application exposes.
  // The application clears the back buffer and draws ImGui into it after
  // onRender(), transitioning it with enhanced barriers from the tracked
  // layout: a caller that transitions the image must return it to the layout
  // it found it in. Empty / null without a swapchain. Keep no copy across
  // frames: a copy holds the back buffer, which a swapchain resize releases.
  const nvdx12::Image&        getCurrentBackBufferImage() const;
  nvdx12::Image&              getCurrentBackBufferImage();
  D3D12_CPU_DESCRIPTOR_HANDLE getBackBufferRtv() const;

  // ImGui texture handle (ImTextureID) of an image's SRV slot on the descriptor heap
  ImTextureID imguiTexture(uint32_t srvSlot) const;
  // Images ImGui samples this frame: transitioned to SHADER_RESOURCE before
  // the ImGui draw. Register them every frame, from onUIRender().
  void registerImguiImage(nvdx12::Image* image);

  // Record that a screenshot is requested, and will be saved two frames
  // later (so that ImGui has time to clear the menu).
  void requestScreenShot(const std::filesystem::path& filename, int quality = 90) override;

  // Saves the current back buffer. Windowed only; this may only be called
  // after submission and before presentation. Use requestScreenShot() outside
  // that frame phase.
  HRESULT saveScreenShot(const std::filesystem::path& filename, int quality = 90);

  // Returns an RGBA8 PNG of the current back buffer in pngData. Windowed
  // only; the same frame-phase restrictions as saveScreenShot apply.
  HRESULT encodeScreenShotToPng(std::vector<uint8_t>& pngData);

  // Saves an image (RGBA8, BGRA8, RGBA16F, RGBA32F or RGB10A2) as RGBA8 (.png,
  // .jpg, .jpeg, .bmp; any other extension as .png) or RGBA32F (.hdr) through
  // a readback copy. Waits for the queue to idle first and restores the image's layout.
  // `forceOpaqueAlpha` writes alpha as 1 for images whose alpha channel
  // carries render data rather than coverage.
  HRESULT saveImageToFile(nvdx12::Image& image, const std::filesystem::path& filename, int quality = 100, bool forceOpaqueAlpha = false);

  // Encodes an image as RGBA8 PNG bytes in memory, with the same readback as saveImageToFile.
  HRESULT encodeImageToPng(nvdx12::Image& image, std::vector<uint8_t>& pngData);

protected:
  // nvapp::ApplicationBase hooks
  void apiImguiBeginFrame() override;
  void apiSwapChainRebuild() override;
  void apiViewportSizeChange() override;
  void apiNormalRun() override;
  void apiHeadlessRun() override;
  void apiWaitIdle() override;
  void deinitApi() override;

private:
  struct FrameData
  {
    nvdx12::CommandContext cmd;            // Allocator and list recording the frame's commands
    uint64_t               fenceValue{0};  // Queue timeline value signaled when the frame completed
  };

  HRESULT initSwapchain();
  HRESULT acquireBackBuffer(uint32_t index);
  HRESULT applySwapchainColorSpace();
  void    destroySwapchain();
  HRESULT rebuildSwapchainIfNeeded();
  void    setupImGuiPlatformBackend(ImGuiConfigFlags configFlags);
  bool    initImGuiDx12Renderer();
  void    shutdownImGuiDx12Renderer();
  bool    prepareFrameResources();
  bool    drawFrame(ID3D12GraphicsCommandList7* cmd);
  void    renderToSwapchain(ID3D12GraphicsCommandList7* cmd);
  bool    endFrame();
  void    stopAfterFailure(HRESULT hr);
  void    presentFrame();
  void    advanceFrame();
  HRESULT readbackImage(nvdx12::Image& image, bool hdr, glm::uvec2& size, std::vector<uint8_t>& pixels8, std::vector<float>& pixelsF);

  // Direct3D 12 objects shared with the application
  nvdx12::Context*        m_context{nullptr};  // ApplicationCreateInfo::context
  ID3D12Device*           m_device{nullptr};   // The context's device
  nvdx12::QueueTimeline*  m_queue{nullptr};    // ApplicationCreateInfo::queue, the frame queue
  nvdx12::DescriptorHeap* m_heap{nullptr};     // ApplicationCreateInfo::descriptorHeap
  HWND                    m_hwnd{nullptr};     // The GLFW window's handle; null headless
  bool     m_baseInitialized{false};       // nvapp::ApplicationBase::init() succeeded; deinit() is a no-op otherwise
  bool     m_imguiDx12Initialized{false};  // ImGui_ImplDX12_Init succeeded
  uint32_t m_imguiFallbackSlot{nvdx12::kInvalidSlot};  // shared by ImGui textures created while the heap is full

  // Elements with per-frame command list access, in m_elements order.
  std::vector<IAppElement*> m_frameElements;

  // Swapchain
  nvdx12::ComPtr<IDXGISwapChain4>          m_swapchain;                // Null headless
  std::vector<nvdx12::Image>               m_backBuffers;              // Layout tracked: PRESENT between frames
  nvdx12::ComPtr<ID3D12DescriptorHeap>     m_rtvHeap;                  // One RTV per back buffer
  std::vector<D3D12_CPU_DESCRIPTOR_HANDLE> m_rtvHandles;               // Into m_rtvHeap
  nvdx12::ComPtr<IDXGIFactory6>            m_swapchainFactory;         // The context's, or an interposer's
  bool                                     m_swapchainRebuild{false};  // Applied in apiImguiBeginFrame()
  bool                                     m_allowTearing{false};      // DXGI_PRESENT_ALLOW_TEARING with vsync off
  // Back buffer format, the color space last requested, and the one the swapchain presents in (getSwapchainColorSpace())
  DXGI_FORMAT           m_swapchainFormat{DXGI_FORMAT_R8G8B8A8_UNORM};
  DXGI_COLOR_SPACE_TYPE m_requestedColorSpace{DXGI_COLOR_SPACE_RGB_FULL_G22_NONE_P709};
  DXGI_COLOR_SPACE_TYPE m_colorSpace{DXGI_COLOR_SPACE_RGB_FULL_G22_NONE_P709};

  // Frame resources and synchronization
  std::vector<FrameData> m_frames;          // One per frame in flight
  uint64_t               m_frameNumber{0};  // Frames submitted since init()
  nvdx12::UploadRing     m_uploadRing;      // on the frame queue; cmdPushData / getUploadRing
  // createTempCmdList / submitAndWaitTempCmdList: allocator + list pairs, one
  // per open list, recycled once their recording has executed
  std::vector<nvdx12::CommandContext> m_tempCmdOpen;
  std::vector<nvdx12::CommandContext> m_tempCmdFree;

  // The frame's submission: the extra wait and signal fences and the prepended
  // lists of addWaitSemaphore / addSignalSemaphore / prependCommandList, then the frame list
  nvdx12::SubmitInfo          m_frameSubmit;
  std::vector<nvdx12::Image*> m_imguiImages;    // registerImguiImage() of this frame
  HRESULT                     m_failure{S_OK};  // getFailure()

  FrameGenerationCallback m_frameGenerationCallback;   // setFrameGenerationCallback()
  FrameMarkerCallback     m_queueSubmitStartCallback;  // setQueueSubmitStartCallback()
  FrameMarkerCallback     m_queueSubmitEndCallback;    // setQueueSubmitEndCallback()
  FrameMarkerCallback     m_presentStartCallback;      // setPresentStartCallback()
  FrameMarkerCallback     m_presentEndCallback;        // setPresentEndCallback()

  bool                  m_screenShotRequested{false};  // requestScreenShot() pending
  uint64_t              m_screenShotTargetFrame{0};    // Saved once m_frameNumber reaches it
  int                   m_screenShotQuality{90};       // JPEG quality of the pending screenshot
  std::filesystem::path m_screenShotFilename;          // File of the pending screenshot
};

}  // namespace nvdx12app
