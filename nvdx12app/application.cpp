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

#include "application.hpp"

#include <algorithm>
#include <cassert>

#include <GLFW/glfw3.h>
#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3native.h>
#undef APIENTRY

#include <d3dx12.h>

#include <backends/imgui_impl_dx12.h>
#include <backends/imgui_impl_glfw.h>
#include <glm/gtc/packing.hpp>
#include <imgui_internal.h>
#include <implot/implot.h>

// Makes stb_image_write take UTF-8 strings instead of the default code page as input on Windows
#define STBIW_WINDOWS_UTF8
#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_WRITE_STATIC
#pragma warning(push)
#pragma warning(disable : 4996)
#include <stb/stb_image_write.h>
#pragma warning(pop)

#include <nvdx12/barriers.hpp>
#include <nvdx12/check_error.hpp>
#include <nvdx12/debug_util.hpp>
#include <nvdx12/formats.hpp>
#include <nvutils/file_operations.hpp>
#include <nvutils/logger.hpp>

namespace {

// The color space DXGI presents a format in without SetColorSpace1: scRGB
// for float formats, sRGB for everything else, 10-bit UNORM included.
DXGI_COLOR_SPACE_TYPE defaultColorSpace(DXGI_FORMAT format)
{
  return format == DXGI_FORMAT_R16G16B16A16_FLOAT ? DXGI_COLOR_SPACE_RGB_FULL_G10_NONE_P709 : DXGI_COLOR_SPACE_RGB_FULL_G22_NONE_P709;
}

// Converts one row of the readback footprint into 8-bit RGBA (`dst8`) or
// float RGBA (`dstF`, .hdr output). Returns false for formats the writer
// does not handle.
bool convertRow(DXGI_FORMAT format, const uint8_t* src, uint32_t width, uint8_t* dst8, float* dstF)
{
  auto store = [&](uint32_t x, float r, float g, float b, float a) {
    if(dstF)
    {
      dstF[x * 4 + 0] = r;
      dstF[x * 4 + 1] = g;
      dstF[x * 4 + 2] = b;
      dstF[x * 4 + 3] = a;
    }
    else
    {
      dst8[x * 4 + 0] = uint8_t(std::clamp(r, 0.f, 1.f) * 255.f + 0.5f);
      dst8[x * 4 + 1] = uint8_t(std::clamp(g, 0.f, 1.f) * 255.f + 0.5f);
      dst8[x * 4 + 2] = uint8_t(std::clamp(b, 0.f, 1.f) * 255.f + 0.5f);
      dst8[x * 4 + 3] = uint8_t(std::clamp(a, 0.f, 1.f) * 255.f + 0.5f);
    }
  };

  switch(format)
  {
    case DXGI_FORMAT_R8G8B8A8_UNORM:
    case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB:
    case DXGI_FORMAT_B8G8R8A8_UNORM:
    case DXGI_FORMAT_B8G8R8A8_UNORM_SRGB: {
      const bool bgra = (format == DXGI_FORMAT_B8G8R8A8_UNORM || format == DXGI_FORMAT_B8G8R8A8_UNORM_SRGB);
      for(uint32_t x = 0; x < width; x++)
      {
        const uint8_t* p = src + x * 4;
        const float    r = p[bgra ? 2 : 0] / 255.f, g = p[1] / 255.f, b = p[bgra ? 0 : 2] / 255.f, a = p[3] / 255.f;
        store(x, r, g, b, a);
      }
      return true;
    }
    case DXGI_FORMAT_R16G16B16A16_FLOAT: {
      const uint16_t* p = reinterpret_cast<const uint16_t*>(src);
      for(uint32_t x = 0; x < width; x++)
      {
        store(x, glm::unpackHalf1x16(p[x * 4 + 0]), glm::unpackHalf1x16(p[x * 4 + 1]),
              glm::unpackHalf1x16(p[x * 4 + 2]), glm::unpackHalf1x16(p[x * 4 + 3]));
      }
      return true;
    }
    case DXGI_FORMAT_R32G32B32A32_FLOAT: {
      const float* p = reinterpret_cast<const float*>(src);
      for(uint32_t x = 0; x < width; x++)
      {
        store(x, p[x * 4 + 0], p[x * 4 + 1], p[x * 4 + 2], p[x * 4 + 3]);
      }
      return true;
    }
    case DXGI_FORMAT_R10G10B10A2_UNORM: {
      const uint32_t* p = reinterpret_cast<const uint32_t*>(src);
      for(uint32_t x = 0; x < width; x++)
      {
        const uint32_t v = p[x];
        store(x, float(v & 0x3FF) / 1023.f, float((v >> 10) & 0x3FF) / 1023.f, float((v >> 20) & 0x3FF) / 1023.f,
              float(v >> 30) / 3.f);
      }
      return true;
    }
    default:
      return false;
  }
}

}  // namespace

namespace nvdx12app {

void IAppElement::onAttach(Application* app)
{
  onAttach(static_cast<nvapp::ApplicationBase*>(app));
}

//-----------------------------------------------------------------------
// Initialization and shutdown

HRESULT Application::init(ApplicationCreateInfo& info)
{
  if(info.context == nullptr || info.queue == nullptr || info.descriptorHeap == nullptr)
  {
    LOGE("Application::init: a nvdx12::Context, a nvdx12::QueueTimeline and a nvdx12::DescriptorHeap are required\n");
    return E_INVALIDARG;
  }
  m_context             = info.context;
  m_queue               = info.queue;
  m_heap                = info.descriptorHeap;
  m_device              = m_context->getDevice();
  m_swapchainFactory    = info.swapchainFactory ? info.swapchainFactory.Get() : m_context->getDxgiFactory();
  m_swapchainFormat     = info.swapchainFormat;
  m_requestedColorSpace = info.colorSpace;
  m_colorSpace          = info.colorSpace;
  m_frameNumber         = 0;
  m_failure             = S_OK;

  // ImGui context and settings, window placement, GLFW window
  nvapp::ApplicationBaseCreateInfo base{.name                  = info.name,
                                        .windowSize            = info.windowSize,
                                        .vSync                 = info.vSync,
                                        .useMenu               = info.useMenu,
                                        .hasUndockableViewport = info.hasUndockableViewport,
                                        .dockSetup             = info.dockSetup,
                                        .imguiConfigFlags      = info.imguiConfigFlags,
                                        .persistSettings       = info.persistSettings,
                                        .headless              = info.headless,
                                        .headlessFrameCount    = info.headlessFrameCount};
  if(!nvapp::ApplicationBase::init(base))
  {
    return E_FAIL;
  }
  if(!m_headless)
  {
    m_hwnd = glfwGetWin32Window(m_windowHandle);
  }
  // ImGui's platform backend: from here on deinit() tears down what init() created.
  setupImGuiPlatformBackend(base.imguiConfigFlags);
  m_baseInitialized = true;

  // Per-frame command contexts and the upload ring; temporary command lists
  // are pooled on demand (createTempCmdList).
  m_frameRingSize    = std::max(1u, info.framesInFlight);
  m_frameRingCurrent = 0;
  NVDX12_FAIL_RETURN(m_uploadRing.init(*m_queue, info.uploadRingBytes));
  m_frames.resize(m_frameRingSize);
  for(size_t i = 0; i < m_frames.size(); i++)
  {
    const std::string name = "Application frame " + std::to_string(i);
    NVDX12_FAIL_RETURN(m_frames[i].cmd.init(m_device, name.c_str()));
  }

  // Set up the resource free queue
  resetFreeQueue(m_frameRingSize);

  // Create the swapchain
  if(!m_headless)
  {
    NVDX12_FAIL_RETURN(initSwapchain());
  }

  // ImGui's DX12 renderer
  if(!initImGuiDx12Renderer())
  {
    return E_FAIL;
  }

  return S_OK;
}

// Also after a failed init(): everything init() created is torn down once
// nvapp::ApplicationBase::init() succeeded; before that, only the GLFW, ImGui
// and ImPlot contexts the constructor created (nvapp::detail::initContexts).
void Application::deinit()
{
  if(!m_baseInitialized)
  {
    if(ImGui::GetCurrentContext() != nullptr)
    {
      ImGui::GetIO().IniFilename = nullptr;  // nothing to persist from a failed init
      ImGui::DestroyContext();
      if(ImPlot::GetCurrentContext() != nullptr)
      {
        ImPlot::DestroyContext();
      }
      glfwTerminate();
    }
    return;
  }
  nvapp::ApplicationBase::deinit();  // free queue, onDetach, .ini save, deinitApi, ImGui and GLFW shutdown
  m_baseInitialized = false;
}

// Called by the base shutdown after the elements are detached and destroyed,
// with the free queue drained and the queue idle.
void Application::deinitApi()
{
  if(m_context == nullptr)
  {
    return;
  }

  shutdownImGuiDx12Renderer();
  destroySwapchain();

  m_uploadRing.deinit();
  if(!m_tempCmdOpen.empty())
  {
    LOGW("%zu temporary command list(s) never submitted (submitAndWaitTempCmdList)\n", m_tempCmdOpen.size());
  }
  for(size_t i = 0; i < m_tempCmdOpen.size(); i++)
  {
    m_tempCmdOpen[i].deinit();
  }
  for(size_t i = 0; i < m_tempCmdFree.size(); i++)
  {
    m_tempCmdFree[i].deinit();
  }
  m_tempCmdOpen.clear();
  m_tempCmdFree.clear();
  for(size_t i = 0; i < m_frames.size(); i++)
  {
    m_frames[i].cmd.deinit();
  }
  m_frames.clear();
  m_frameElements.clear();
  m_imguiImages.clear();
  m_frameSubmit.clear();
  m_swapchainFactory.Reset();
  m_hwnd    = nullptr;
  m_device  = nullptr;
  m_queue   = nullptr;
  m_heap    = nullptr;
  m_context = nullptr;
}

//-----------------------------------------------------------------------
// Swapchain

// Flip-model swapchain on the window, with the requested color space when
// the display supports it, and one RTV per back buffer.
HRESULT Application::initSwapchain()
{
  m_allowTearing = m_context->getFeatures().tearing;

  DXGI_SWAP_CHAIN_DESC1 desc{};
  desc.Width            = m_windowSize.x;
  desc.Height           = m_windowSize.y;
  desc.Format           = m_swapchainFormat;
  desc.SampleDesc.Count = 1;
  desc.BufferUsage      = DXGI_USAGE_RENDER_TARGET_OUTPUT;
  desc.BufferCount      = std::max(2u, uint32_t(m_frames.size()));  // flip model needs two
  desc.Scaling          = DXGI_SCALING_STRETCH;
  desc.SwapEffect       = DXGI_SWAP_EFFECT_FLIP_DISCARD;
  desc.AlphaMode        = DXGI_ALPHA_MODE_IGNORE;
  desc.Flags            = m_allowTearing ? DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING : 0;

  nvdx12::ComPtr<IDXGISwapChain1> swapchain1;
  NVDX12_FAIL_RETURN(m_swapchainFactory->CreateSwapChainForHwnd(m_queue->getQueue(), m_hwnd, &desc, nullptr, nullptr, &swapchain1));
  NVDX12_FAIL_RETURN(swapchain1.As(&m_swapchain));
  m_colorSpace = defaultColorSpace(m_swapchainFormat);
  // Alt+Enter exclusive fullscreen is incompatible with tearing presents.
  NVDX12_FAIL_RETURN(m_swapchainFactory->MakeWindowAssociation(m_hwnd, DXGI_MWA_NO_ALT_ENTER));
  NVDX12_FAIL_RETURN(applySwapchainColorSpace());

  D3D12_DESCRIPTOR_HEAP_DESC rtvDesc{};
  rtvDesc.Type           = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
  rtvDesc.NumDescriptors = desc.BufferCount;
  NVDX12_FAIL_RETURN(m_device->CreateDescriptorHeap(&rtvDesc, IID_PPV_ARGS(&m_rtvHeap)));
  nvdx12::setDebugName(m_rtvHeap.Get(), "Swapchain RTVs");

  const UINT stride = m_device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
  m_backBuffers.resize(desc.BufferCount);
  m_rtvHandles.resize(desc.BufferCount);
  for(UINT i = 0; i < desc.BufferCount; i++)
  {
    m_rtvHandles[i] = m_rtvHeap->GetCPUDescriptorHandleForHeapStart();
    m_rtvHandles[i].ptr += SIZE_T(i) * stride;
    NVDX12_FAIL_RETURN(acquireBackBuffer(i));
  }
  m_swapchainRebuild = false;
  return S_OK;
}

// Back buffer `index` as a tracked image in the PRESENT layout, and its RTV.
HRESULT Application::acquireBackBuffer(uint32_t index)
{
  nvdx12::Image& image = m_backBuffers[index];
  image                = nvdx12::Image{};
  NVDX12_FAIL_RETURN(m_swapchain->GetBuffer(index, IID_PPV_ARGS(&image.image)));
  nvdx12::setDebugName(image.image.Get(), "Back buffer " + std::to_string(index));
  const D3D12_RESOURCE_DESC desc = image.image->GetDesc();
  image.format                   = desc.Format;
  image.extent                   = {uint32_t(desc.Width), desc.Height, 1};
  image.flags                    = desc.Flags;
  image.layout                   = D3D12_BARRIER_LAYOUT_PRESENT;
  m_device->CreateRenderTargetView(image.image.Get(), nullptr, m_rtvHandles[index]);
  return S_OK;
}

// Presents in the requested color space when the display accepts it,
// otherwise in the format's default. m_colorSpace holds the color space the
// swapchain presents in, which survives ResizeBuffers: only a change is set.
HRESULT Application::applySwapchainColorSpace()
{
  DXGI_COLOR_SPACE_TYPE colorSpace = m_requestedColorSpace;
  UINT                  support    = 0;
  if(FAILED(m_swapchain->CheckColorSpaceSupport(colorSpace, &support))
     || (support & DXGI_SWAP_CHAIN_COLOR_SPACE_SUPPORT_FLAG_PRESENT) == 0)
  {
    colorSpace = defaultColorSpace(m_swapchainFormat);
    LOGW("Swapchain color space %d is not supported for presentation; presenting in %d\n", int(m_requestedColorSpace),
         int(colorSpace));
  }
  if(colorSpace != m_colorSpace)
  {
    NVDX12_FAIL_RETURN(m_swapchain->SetColorSpace1(colorSpace));
    m_colorSpace = colorSpace;
  }
  return S_OK;
}

void Application::destroySwapchain()
{
  if(m_swapchain)
  {
    m_queue->waitIdle();
  }
  m_backBuffers.clear();
  m_rtvHandles.clear();
  m_rtvHeap.Reset();
  m_swapchain.Reset();
}

// Resizes the back buffers to the window, in the requested format, and
// applies the color space; a new format also rebuilds the ImGui renderer,
// whose pipeline targets it. Fails when the swapchain is no longer usable.
HRESULT Application::rebuildSwapchainIfNeeded()
{
  if(!m_swapchainRebuild || !m_swapchain)
  {
    return S_OK;
  }
  m_swapchainRebuild = false;

  DXGI_SWAP_CHAIN_DESC1 desc{};
  NVDX12_FAIL_RETURN(m_swapchain->GetDesc1(&desc));
  m_queue->waitIdle();

  for(size_t i = 0; i < m_backBuffers.size(); i++)
  {
    m_backBuffers[i] = nvdx12::Image{};
  }
  const HRESULT hr = m_swapchain->ResizeBuffers(desc.BufferCount, m_windowSize.x, m_windowSize.y, m_swapchainFormat, desc.Flags);
  if(hr == DXGI_ERROR_DEVICE_REMOVED || hr == DXGI_ERROR_DEVICE_RESET)
  {
    m_context->checkDeviceRemoved();
    return hr;
  }
  NVDX12_FAIL_RETURN(hr);
  for(UINT i = 0; i < desc.BufferCount; i++)
  {
    NVDX12_FAIL_RETURN(acquireBackBuffer(i));
  }
  NVDX12_FAIL_RETURN(applySwapchainColorSpace());

  // ResizeBuffers restarts the swapchain at back buffer 0. The frame ring
  // indexes both the per-frame resources and the fence the next frame waits
  // on, and prepareFrameResources() only waits for the right buffer while the
  // two stay in phase -- otherwise a frame records into a back buffer that is
  // still on screen. The queue is idle here, so re-phasing is safe.
  m_frameRingCurrent = m_swapchain->GetCurrentBackBufferIndex() % m_frameRingSize;

  if(desc.Format != m_swapchainFormat)
  {
    // The renderer's shutdown destroys the platform windows, the main
    // window's GLFW backend data included: the GLFW backend restarts too.
    shutdownImGuiDx12Renderer();
    ImGui_ImplGlfw_Shutdown();
    ImGui_ImplGlfw_InitForOther(m_windowHandle, true);
    if(!initImGuiDx12Renderer())
    {
      return E_FAIL;
    }
  }
  return S_OK;
}

void Application::setPreferredSwapchainFormat(DXGI_FORMAT format, DXGI_COLOR_SPACE_TYPE colorSpace)
{
  if(format != m_swapchainFormat || colorSpace != m_requestedColorSpace)
  {
    m_swapchainFormat     = format;
    m_requestedColorSpace = colorSpace;
    m_swapchainRebuild    = true;
  }
}

void Application::apiSwapChainRebuild()
{
  m_swapchainRebuild = true;  // applied at the next frame boundary
}

void Application::setVsync(bool v)
{
  m_vsyncWanted = v;
}

ID3D12Resource* Application::getCurrentBackBuffer() const
{
  if(!m_swapchain || m_backBuffers.empty())
  {
    return nullptr;
  }
  return m_backBuffers[m_swapchain->GetCurrentBackBufferIndex()].image.Get();
}

const nvdx12::Image& Application::getCurrentBackBufferImage() const
{
  static const nvdx12::Image empty{};
  if(!m_swapchain || m_backBuffers.empty())
  {
    return empty;
  }
  return m_backBuffers[m_swapchain->GetCurrentBackBufferIndex()];
}

nvdx12::Image& Application::getCurrentBackBufferImage()
{
  static nvdx12::Image empty{};
  if(!m_swapchain || m_backBuffers.empty())
  {
    empty = nvdx12::Image{};  // a caller may have written to it
    return empty;
  }
  return m_backBuffers[m_swapchain->GetCurrentBackBufferIndex()];
}

D3D12_CPU_DESCRIPTOR_HANDLE Application::getBackBufferRtv() const
{
  if(!m_swapchain || m_rtvHandles.empty())
  {
    return D3D12_CPU_DESCRIPTOR_HANDLE{};
  }
  return m_rtvHandles[m_swapchain->GetCurrentBackBufferIndex()];
}

//-----------------------------------------------------------------------
// ImGui

// Sets up the ImGui configuration and GLFW backend. Assumes the ImGui context
// and fonts are already initialized.
void Application::setupImGuiPlatformBackend(ImGuiConfigFlags configFlags)
{
  ImGuiIO& io    = ImGui::GetIO();
  io.ConfigFlags = configFlags;
  if(m_headless)
  {
    io.ConfigFlags &= ~ImGuiConfigFlags_ViewportsEnable;  // In headless mode, we don't allow other viewport
  }
  else
  {
    ImGui_ImplGlfw_InitForOther(m_windowHandle, true);
  }
}

// ImGui's DX12 renderer targets the swapchain format. Its SRV descriptors
// (font atlas, backend textures) come from single slots of the shared
// shader-visible heap, so ImGui::Image can take the heap handles of the
// elements' images directly.
bool Application::initImGuiDx12Renderer()
{
  // ImGui cannot take a failed descriptor allocation: textures created while
  // the heap is full share this slot.
  m_imguiFallbackSlot = m_heap->acquireSlots();
  if(m_imguiFallbackSlot == nvdx12::kInvalidSlot)
  {
    LOGE("ImGui DX12 renderer: the descriptor heap is full\n");
    return false;
  }

  ImGui_ImplDX12_InitInfo initInfo{};
  initInfo.Device               = m_device;
  initInfo.CommandQueue         = m_queue->getQueue();
  initInfo.NumFramesInFlight    = int(m_frames.size());
  initInfo.RTVFormat            = m_swapchainFormat;
  initInfo.DSVFormat            = DXGI_FORMAT_UNKNOWN;
  initInfo.UserData             = this;
  initInfo.SrvDescriptorHeap    = m_heap->resourceHeap();
  initInfo.SrvDescriptorAllocFn = [](ImGui_ImplDX12_InitInfo* ii, D3D12_CPU_DESCRIPTOR_HANDLE* outCpu,
                                     D3D12_GPU_DESCRIPTOR_HANDLE* outGpu) {
    auto*    app  = static_cast<Application*>(ii->UserData);
    uint32_t slot = app->m_heap->acquireSlots();
    if(slot == nvdx12::kInvalidSlot)
    {
      // The texture shows whatever the last texture written to the fallback slot holds.
      LOGE("ImGui DX12 renderer: the descriptor heap is full, an ImGui texture takes the fallback slot\n");
      slot = app->m_imguiFallbackSlot;
    }
    *outCpu = app->m_heap->cpuHandle(slot);
    *outGpu = app->m_heap->gpuHandle(slot);
  };
  initInfo.SrvDescriptorFreeFn = [](ImGui_ImplDX12_InitInfo* ii, D3D12_CPU_DESCRIPTOR_HANDLE, D3D12_GPU_DESCRIPTOR_HANDLE gpu) {
    auto*          app  = static_cast<Application*>(ii->UserData);
    const uint32_t slot = app->m_heap->slotFromGpuHandle(gpu);
    assert(slot != nvdx12::kInvalidSlot && "descriptor outside the shared heap");
    if(slot != nvdx12::kInvalidSlot && slot != app->m_imguiFallbackSlot)
    {
      app->m_heap->releaseSlots(slot);
    }
  };
  m_imguiDx12Initialized = ImGui_ImplDX12_Init(&initInfo);
  if(!m_imguiDx12Initialized)
  {
    LOGE("ImGui_ImplDX12_Init failed\n");
  }
  return m_imguiDx12Initialized;
}

void Application::shutdownImGuiDx12Renderer()
{
  if(m_imguiDx12Initialized)
  {
    ImGui_ImplDX12_Shutdown();
    m_imguiDx12Initialized = false;
  }
  if(m_imguiFallbackSlot != nvdx12::kInvalidSlot)
  {
    m_heap->releaseSlots(m_imguiFallbackSlot);
    m_imguiFallbackSlot = nvdx12::kInvalidSlot;
  }
}

// The swapchain, and the ImGui renderer with it, are rebuilt before ImGui's
// frame begins, so the frame records against the renderer that draws it.
void Application::apiImguiBeginFrame()
{
  const HRESULT rebuilt = rebuildSwapchainIfNeeded();
  if(FAILED(rebuilt) && SUCCEEDED(m_failure))
  {
    m_failure = rebuilt;  // prepareFrameResources() closes the application
  }
  if(m_imguiDx12Initialized)
  {
    ImGui_ImplDX12_NewFrame();
  }
}

ImTextureID Application::imguiTexture(uint32_t srvSlot) const
{
  return ImTextureID(m_heap->gpuHandle(srvSlot).ptr);
}

void Application::registerImguiImage(nvdx12::Image* image)
{
  if(image != nullptr && std::find(m_imguiImages.begin(), m_imguiImages.end(), image) == m_imguiImages.end())
  {
    m_imguiImages.push_back(image);
  }
}

//-----------------------------------------------------------------------
// Elements and control

void Application::addElement(const std::shared_ptr<nvapp::IAppElementBase>& element)
{
  m_elements.emplace_back(element);
  if(auto* frameElement = dynamic_cast<IAppElement*>(element.get()))
  {
    m_frameElements.push_back(frameElement);
    frameElement->onAttach(this);
  }
  else
  {
    element->onAttach(this);
  }
}

// Every element gets onResize(size); the ones recording commands also get
// onResize(cmd, size) with a temporary command list, submitted after the loop.
void Application::apiViewportSizeChange()
{
  if(m_viewportSize.x == 0 || m_viewportSize.y == 0)
  {
    return;
  }
  m_queue->waitIdle();
  for(size_t i = 0; i < m_elements.size(); i++)
  {
    m_elements[i]->onResize(m_viewportSize);
  }
  ID3D12GraphicsCommandList7* cmd = createTempCmdList();
  if(cmd == nullptr)
  {
    LOGE("Application: no command list for the viewport resize to %ux%u\n", m_viewportSize.x, m_viewportSize.y);
    return;
  }
  for(size_t i = 0; i < m_frameElements.size(); i++)
  {
    m_frameElements[i]->onResize(cmd, m_viewportSize);
  }
  // The elements expect their resize commands executed: a failed submission
  // stops the application, as a failed frame does.
  const HRESULT hr = submitAndWaitTempCmdList(cmd);
  if(FAILED(hr))
  {
    stopAfterFailure(hr);
  }
}

void Application::apiWaitIdle()
{
  if(m_context != nullptr)
  {
    m_queue->waitIdle();
  }
}

void Application::addWaitSemaphore(const nvdx12::SemaphoreInfo& wait)
{
  m_frameSubmit.waitSemaphores.push_back(wait);
}

void Application::addSignalSemaphore(const nvdx12::SemaphoreInfo& signal)
{
  m_frameSubmit.signalSemaphores.push_back(signal);
}

void Application::prependCommandList(ID3D12CommandList* cmd)
{
  m_frameSubmit.commandLists.push_back(cmd);
}

bool Application::cmdPushData(ID3D12GraphicsCommandList* cmd, const void* data, uint32_t size, nvdx12::BindPoint bindPoint)
{
  return m_uploadRing.cmdPushData(cmd, data, size, bindPoint);
}

ID3D12GraphicsCommandList7* Application::createTempCmdList()
{
  // A recycled pair, or a new one; the pool grows to the largest number of
  // lists open at the same time.
  if(m_tempCmdFree.empty())
  {
    nvdx12::CommandContext temp;
    const std::string      name = "Application temp " + std::to_string(m_tempCmdOpen.size());
    if(FAILED(NVDX12_FAIL_REPORT(temp.init(m_device, name.c_str()))))
    {
      return nullptr;
    }
    m_tempCmdFree.push_back(std::move(temp));
  }
  m_tempCmdOpen.push_back(std::move(m_tempCmdFree.back()));
  m_tempCmdFree.pop_back();

  nvdx12::CommandContext& temp = m_tempCmdOpen.back();
  if(FAILED(NVDX12_FAIL_REPORT(temp.begin())))
  {
    m_tempCmdFree.push_back(std::move(temp));
    m_tempCmdOpen.pop_back();
    return nullptr;
  }
  m_heap->cmdBind(temp.list.Get());
  return temp.list.Get();
}

HRESULT Application::submitAndWaitTempCmdList(ID3D12GraphicsCommandList7* cmd)
{
  auto it = std::find_if(m_tempCmdOpen.begin(), m_tempCmdOpen.end(),
                         [cmd](const nvdx12::CommandContext& temp) { return temp.list.Get() == cmd; });
  if(it == m_tempCmdOpen.end())
  {
    LOGE("submitAndWaitTempCmdList: the list did not come from createTempCmdList\n");
    return E_INVALIDARG;
  }
  const HRESULT closed = it->close();
  if(FAILED(closed))
  {
    // Never execute a list that failed to close; drop it instead of recycling it.
    it->deinit();
    m_tempCmdOpen.erase(it);
    return closed;
  }
  const uint64_t value = m_queue->submit(cmd);  // 0: the signal failed (already reported)
  HRESULT        hr    = value != 0 ? m_queue->waitForValue(value) : E_FAIL;
  if(SUCCEEDED(hr))
  {
    hr = m_device->GetDeviceRemovedReason();  // a removed device completes every wait
  }
  m_tempCmdFree.push_back(std::move(*it));
  m_tempCmdOpen.erase(it);
  return hr;
}

//-----------------------------------------------------------------------
// Frame loop

// One windowed frame; the base has polled events, begun the ImGui frame and
// drawn the menu bar and dockspace.
void Application::apiNormalRun()
{
  if(!prepareFrameResources())
  {
    return;  // close() was called, the loop ends
  }

  ID3D12GraphicsCommandList7* cmd = m_frames[m_frameRingCurrent].cmd.list.Get();
  if(!drawFrame(cmd))  // Call onUIRender(), onPreRender() and onRender() for each element
  {
    return;
  }
  renderToSwapchain(cmd);  // Render ImGui to the back buffer
  if(!endFrame())          // Submit the frame
  {
    return;
  }

  // This is the last application-owned point for the submitted image.
  // Consumers that read it on the CPU synchronize before accessing it.
  for(size_t i = 0; i < m_elements.size(); i++)
  {
    m_elements[i]->onPostRender();
  }

  // Handle screenshot requests here: capture the composited frame after the
  // submit and before the present, while it is in the current back buffer.
  if(m_screenShotRequested && m_frameNumber >= m_screenShotTargetFrame)
  {
    saveScreenShot(m_screenShotFilename, m_screenShotQuality);
    m_screenShotRequested = false;
  }

  presentFrame();  // This can also detect a lost device
  advanceFrame();
}

// One headless frame; the base has resized the elements, run the UI once
// for the dock layout, and calls onLastHeadlessFrame after the last frame.
void Application::apiHeadlessRun()
{
  if(!prepareFrameResources())
  {
    return;  // close() was called, the loop ends
  }

  ID3D12GraphicsCommandList7* cmd = m_frames[m_frameRingCurrent].cmd.list.Get();
  if(!drawFrame(cmd))
  {
    return;
  }
  renderToSwapchain(cmd);  // image transitions only
  if(!endFrame())
  {
    return;
  }

  for(size_t i = 0; i < m_elements.size(); i++)
  {
    m_elements[i]->onPostRender();
  }
  advanceFrame();
}

// prepareFrameResources is the first step in the rendering process.
// It waits for the frame slot's fence and runs the slot's free queue (the
// swapchain was rebuilt in apiImguiBeginFrame). False, after close(), when
// an earlier step failed, the wait fails or the device is lost.
bool Application::prepareFrameResources()
{
  if(FAILED(m_failure) || (!m_headless && !m_swapchain))
  {
    close();
    return false;
  }

  // Wait until GPU has finished processing the frame that was using these resources previously
  HRESULT hr = NVDX12_FAIL_REPORT(m_queue->waitForValue(m_frames[m_frameRingCurrent].fenceValue));
  if(SUCCEEDED(hr))
  {
    hr = m_device->GetDeviceRemovedReason();
  }
  if(FAILED(hr))
  {
    stopAfterFailure(hr);  // the frame's allocator may still be in use
    return false;
  }

  freeResourcesQueue();  // Free resources from previous frame
  return true;
}

// Main frame rendering function
// - Call onUIRender() for each element and build the ImGui draw data
// - Call onPreRender() for each element
// - Open the frame's command list and call onRender() for each element
bool Application::drawFrame(ID3D12GraphicsCommandList7* cmd)
{
  // Reset the extra fences, command lists and ImGui images
  m_frameSubmit.clear();
  m_imguiImages.clear();

  // Call UI rendering for each element
  for(size_t i = 0; i < m_elements.size(); i++)
  {
    m_elements[i]->onUIRender();
  }

  // This is creating the data to draw the UI (not on GPU yet)
  ImGui::Render();

  // The frame's upload-ring allocations stay in flight across temporary
  // submissions until the submission of the frame's list.
  m_uploadRing.beginDeferral(cmd);

  // Call onPreRender for each element
  for(size_t i = 0; i < m_elements.size(); i++)
  {
    m_elements[i]->onPreRender();
  }

  // Call onRender for each element with the command list of the frame
  const HRESULT begun = m_frames[m_frameRingCurrent].cmd.begin();
  if(FAILED(begun))
  {
    m_uploadRing.cancelDeferral();
    stopAfterFailure(begun);
    return false;
  }
  m_heap->cmdBind(cmd);
  for(size_t i = 0; i < m_frameElements.size(); i++)
  {
    m_frameElements[i]->onRender(cmd);
  }
  return true;
}

void Application::renderToSwapchain(ID3D12GraphicsCommandList7* cmd)
{
  // Images ImGui samples this frame become shader resources; the global
  // barrier extends the visibility to the pixel shader stage.
  for(size_t i = 0; i < m_imguiImages.size(); i++)
  {
    nvdx12::cmdImageBarrier(cmd, *m_imguiImages[i], D3D12_BARRIER_LAYOUT_SHADER_RESOURCE);
  }
  nvdx12::cmdMemoryBarrier(cmd);

  if(m_headless || !m_swapchain)
  {
    return;
  }

  const uint32_t backIndex  = m_swapchain->GetCurrentBackBufferIndex();
  nvdx12::Image& backBuffer = m_backBuffers[backIndex];

  // PRESENT -> RENDER_TARGET. The present is ordered by the swapchain; the
  // source scope inferred from the PRESENT layout (SYNC_ALL, ACCESS_COMMON)
  // orders any access by a caller that transitioned the back buffer earlier
  // in the submission and restored its layout.
  nvdx12::cmdImageMemoryBarrier(cmd, backBuffer, {.newLayout = D3D12_BARRIER_LAYOUT_RENDER_TARGET});
  const float clearColor[4] = {0.0f, 0.0f, 0.0f, 1.0f};
  cmd->OMSetRenderTargets(1, &m_rtvHandles[backIndex], FALSE, nullptr);
  cmd->ClearRenderTargetView(m_rtvHandles[backIndex], clearColor, 0, nullptr);
  {
    NVDX12_DBG_SCOPE_NAMED(cmd, "ImGui");
    // The ImGui draw commands are recorded to the command list, which includes the display of our GBuffer image
    ImGui_ImplDX12_RenderDrawData(ImGui::GetDrawData(), cmd);
  }
  m_heap->cmdBind(cmd);  // ImGui rebound the descriptor heaps

  if(m_frameGenerationCallback)
  {
    m_frameGenerationCallback(cmd, backBuffer.image.Get(), m_swapchainFormat, m_windowSize, m_frameNumber);
  }
  // RENDER_TARGET -> PRESENT; the present follows the submission, outside any command list.
  nvdx12::cmdImageMemoryBarrier(cmd, backBuffer,
                                {.newLayout     = D3D12_BARRIER_LAYOUT_PRESENT,
                                 .dstStageMask  = D3D12_BARRIER_SYNC_NONE,
                                 .dstAccessMask = uint32_t(D3D12_BARRIER_ACCESS_NO_ACCESS)});
}

// End the frame by submitting the command lists to the GPU
// Waits for and signals the extra fences, and records the queue timeline
// value that completes the frame in the frame slot.
bool Application::endFrame()
{
  FrameData&    frame  = m_frames[m_frameRingCurrent];
  const HRESULT closed = frame.cmd.close();
  if(FAILED(closed))
  {
    m_uploadRing.cancelDeferral();
    stopAfterFailure(closed);
    return false;
  }

  // Adding the command list of the frame to the list of command lists to submit
  // Note: extra command lists could have been added to the list from other parts of the application (elements)
  m_frameSubmit.commandLists.push_back(frame.cmd.list.Get());

  if(m_queueSubmitStartCallback)
  {
    m_queueSubmitStartCallback(m_frameNumber);
  }

  // Waits for the extra fences, executes the lists, signals the timeline and the extra fences.
  nvdx12::SemaphoreState submitState;
  const HRESULT          submitResult = NVDX12_FAIL_REPORT(m_queue->submit(m_frameSubmit, submitState));
  if(SUCCEEDED(submitResult))
  {
    frame.fenceValue = submitState.getTimelineValue();
  }
  else
  {
    m_uploadRing.cancelDeferral();  // no-op when the frame's list executed and ended the deferral
  }

  if(m_queueSubmitEndCallback)
  {
    m_queueSubmitEndCallback(m_frameNumber);
  }
  if(FAILED(submitResult))
  {
    stopAfterFailure(submitResult);
    return false;
  }
  return true;
}

// A frame or resize list failed to reset, close, submit or complete:
// reports a lost device, records the error (getFailure) and closes the
// application (nvapp exits on a failed vkQueueSubmit2).
void Application::stopAfterFailure(HRESULT hr)
{
  m_context->checkDeviceRemoved();
  if(SUCCEEDED(m_failure))
  {
    m_failure = hr;
  }
  close();
}

// The presentFrame function is the last step in the rendering process.
// It presents the image to the screen; a lost device closes the application.
void Application::presentFrame()
{
  if(m_presentStartCallback)
  {
    m_presentStartCallback(m_frameNumber);
  }

  const UINT    syncInterval = m_vsyncWanted ? 1 : 0;
  const UINT    flags        = (!m_vsyncWanted && m_allowTearing) ? DXGI_PRESENT_ALLOW_TEARING : 0;
  const HRESULT hr           = m_swapchain->Present(syncInterval, flags);
  if(hr == DXGI_ERROR_DEVICE_REMOVED || hr == DXGI_ERROR_DEVICE_RESET)
  {
    LOGE("Present failed: %s\n", nvdx12::hresultToString(hr).c_str());
    stopAfterFailure(hr);
  }
  else if(FAILED(hr))
  {
    NVDX12_FAIL_REPORT(hr);
  }

  if(m_presentEndCallback)
  {
    m_presentEndCallback(m_frameNumber);
  }
}

void Application::advanceFrame()
{
  // Move to the next frame
  m_frameRingCurrent = (m_frameRingCurrent + 1) % m_frameRingSize;
  m_frameNumber++;
}

//-----------------------------------------------------------------------
// Screenshots

void Application::requestScreenShot(const std::filesystem::path& filename, int quality)
{
  if(m_headless)
  {
    LOGE("Screenshot capture requires a windowed application\n");
    return;
  }
  m_screenShotRequested = true;
  m_screenShotFilename  = filename;
  // Making sure the screenshot is taken after the menu that triggered it has closed
  m_screenShotTargetFrame = m_frameNumber + 2;
  m_screenShotQuality     = quality;
}

HRESULT Application::saveScreenShot(const std::filesystem::path& filename, int quality)
{
  if(!m_swapchain || m_backBuffers.empty())
  {
    LOGE("Screenshot capture requires a windowed application\n");
    return DXGI_ERROR_UNSUPPORTED;
  }
  return saveImageToFile(m_backBuffers[m_swapchain->GetCurrentBackBufferIndex()], filename, quality);
}

HRESULT Application::encodeScreenShotToPng(std::vector<uint8_t>& pngData)
{
  pngData.clear();
  if(!m_swapchain || m_backBuffers.empty())
  {
    LOGE("Screenshot capture requires a windowed application\n");
    return DXGI_ERROR_UNSUPPORTED;
  }
  return encodeImageToPng(m_backBuffers[m_swapchain->GetCurrentBackBufferIndex()], pngData);
}

// Waits for the queue, copies mip 0 of `image` through a readback buffer
// and converts it to RGBA8, or RGBA32F when `hdr`. The image returns to
// its entry layout.
HRESULT Application::readbackImage(nvdx12::Image& image, bool hdr, glm::uvec2& size, std::vector<uint8_t>& pixels8, std::vector<float>& pixelsF)
{
  if(!image)
  {
    LOGE("readbackImage: null image\n");
    return E_INVALIDARG;
  }

  m_queue->waitIdle();

  const nvdx12::SubresourceFootprint footprint = nvdx12::imageSubresourceFootprint(image, 0);

  // Readback buffer created directly: the application owns no ResourceAllocator.
  nvdx12::Buffer readback{};
  {
    const CD3DX12_HEAP_PROPERTIES heapProps(D3D12_HEAP_TYPE_READBACK);
    const CD3DX12_RESOURCE_DESC   desc = CD3DX12_RESOURCE_DESC::Buffer(footprint.totalBytes);
    NVDX12_FAIL_RETURN(m_device->CreateCommittedResource(&heapProps, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COPY_DEST,
                                                         nullptr, IID_PPV_ARGS(&readback.buffer)));
    nvdx12::setDebugName(readback.buffer.Get(), "Screenshot readback");
    readback.bufferSize  = footprint.totalBytes;
    readback.memoryUsage = nvdx12::MemoryUsage::eHostReadback;
    readback.gpuAddress  = readback.buffer->GetGPUVirtualAddress();
  }

  // The copy transitions the image to COPY_SOURCE; it returns to its entry layout.
  const D3D12_BARRIER_LAYOUT entryLayout = image.layout;
  {
    ID3D12GraphicsCommandList7* cmd = createTempCmdList();
    if(cmd == nullptr)
    {
      LOGE("readbackImage: no command list for the copy\n");
      return E_FAIL;
    }
    nvdx12::cmdCopyImageToBuffer(cmd, image, readback, 0);
    if(entryLayout != D3D12_BARRIER_LAYOUT_UNDEFINED)
    {
      nvdx12::cmdImageBarrier(cmd, image, entryLayout);
    }
    NVDX12_FAIL_RETURN(submitAndWaitTempCmdList(cmd));
  }

  void* mappedPtr = nullptr;
  NVDX12_FAIL_RETURN(readback.buffer->Map(0, nullptr, &mappedPtr));

  const uint32_t width  = footprint.footprint.Footprint.Width;
  const uint32_t height = footprint.footprint.Footprint.Height;
  const uint64_t pitch  = footprint.footprint.Footprint.RowPitch;
  const uint8_t* mapped = static_cast<const uint8_t*>(mappedPtr) + footprint.footprint.Offset;

  if(hdr)
  {
    pixelsF.resize(size_t(width) * height * 4);
  }
  else
  {
    pixels8.resize(size_t(width) * height * 4);
  }
  bool ok = true;
  for(uint32_t y = 0; y < height && ok; y++)
  {
    ok = convertRow(image.format, mapped + y * pitch, width, hdr ? nullptr : pixels8.data() + size_t(y) * width * 4,
                    hdr ? pixelsF.data() + size_t(y) * width * 4 : nullptr);
  }
  const D3D12_RANGE noWrite{0, 0};
  readback.buffer->Unmap(0, &noWrite);

  if(!ok)
  {
    LOGE("readbackImage: unsupported image format %d\n", int(image.format));
    return DXGI_ERROR_UNSUPPORTED;
  }
  size = {width, height};
  return S_OK;
}

HRESULT Application::saveImageToFile(nvdx12::Image& image, const std::filesystem::path& filename, int quality, bool forceOpaqueAlpha)
{
  // Unknown extensions are saved as PNG, as nvvk::saveImageToFile does.
  const bool hdr = nvutils::extensionMatches(filename, ".hdr");
  const bool jpg = nvutils::extensionMatches(filename, ".jpg") || nvutils::extensionMatches(filename, ".jpeg");
  const bool bmp = nvutils::extensionMatches(filename, ".bmp");
  std::filesystem::path path = filename;
  if(!hdr && !jpg && !bmp && !nvutils::extensionMatches(filename, ".png"))
  {
    LOGW("saveImageToFile: unknown file extension, saving as PNG\n");
    path.replace_extension(".png");
  }

  glm::uvec2           size{};
  std::vector<uint8_t> pixels8;
  std::vector<float>   pixelsF;
  NVDX12_FAIL_RETURN(readbackImage(image, hdr, size, pixels8, pixelsF));

  if(forceOpaqueAlpha)
  {
    for(size_t i = 3; i < size_t(size.x) * size.y * 4; i += 4)
    {
      if(hdr)
        pixelsF[i] = 1.0f;
      else
        pixels8[i] = 255;
    }
  }

  const std::string utf8    = nvutils::utf8FromPath(path);
  int               written = 0;
  if(hdr)
  {
    written = stbi_write_hdr(utf8.c_str(), int(size.x), int(size.y), 4, pixelsF.data());
  }
  else if(jpg)
  {
    written = stbi_write_jpg(utf8.c_str(), int(size.x), int(size.y), 4, pixels8.data(), quality);
  }
  else if(bmp)
  {
    written = stbi_write_bmp(utf8.c_str(), int(size.x), int(size.y), 4, pixels8.data());
  }
  else
  {
    written = stbi_write_png(utf8.c_str(), int(size.x), int(size.y), 4, pixels8.data(), int(size.x) * 4);
  }

  if(written == 0)
  {
    LOGE("Failed to write image %s\n", utf8.c_str());
    return E_FAIL;
  }
  LOGI("Image saved to %s\n", utf8.c_str());
  return S_OK;
}

HRESULT Application::encodeImageToPng(nvdx12::Image& image, std::vector<uint8_t>& pngData)
{
  pngData.clear();

  glm::uvec2           size{};
  std::vector<uint8_t> pixels8;
  std::vector<float>   pixelsF;
  NVDX12_FAIL_RETURN(readbackImage(image, false, size, pixels8, pixelsF));

  struct WriteContext
  {
    std::vector<uint8_t>* bytes;
    bool                  failed{false};
  } context{&pngData};
  // stb writes encoded chunks through a C callback: an allocation failure must
  // not unwind through stb, so it is caught inside (as nvvk::encodeImageToPng).
  auto append = [](void* user, void* data, int size) {
    WriteContext& writeContext = *static_cast<WriteContext*>(user);
    if(writeContext.failed)
      return;
    try
    {
      const uint8_t* first = static_cast<const uint8_t*>(data);
      writeContext.bytes->insert(writeContext.bytes->end(), first, first + size);
    }
    catch(...)
    {
      writeContext.failed = true;
    }
  };
  const int written = stbi_write_png_to_func(append, &context, int(size.x), int(size.y), 4, pixels8.data(), int(size.x) * 4);
  if(context.failed || written == 0)
  {
    pngData.clear();
    LOGE("encodeImageToPng: PNG encoding failed\n");
    return context.failed ? E_OUTOFMEMORY : E_FAIL;
  }
  return S_OK;
}

}  // namespace nvdx12app


//--------------------------------------------------------------------------------------------------
// Usage example
//--------------------------------------------------------------------------------------------------
namespace {
// An element that renders into its own image and shows it in the "Viewport" window.
struct UsageElement : nvdx12app::IAppElement
{
  nvdx12app::Application* app{nullptr};
  nvdx12::Image           image;                          // EX: created by a nvdx12::ResourceAllocator in onAttach
  uint32_t                srvSlot{nvdx12::kInvalidSlot};  // EX: acquired on app->getDescriptorHeap()

  void onAttach(nvdx12app::Application* a) override { app = a; }
  void onResize(ID3D12GraphicsCommandList7* /*cmd*/, const glm::uvec2& /*size*/) override {}  // EX: recreate `image`
  void onUIRender() override
  {
    ImGui::Begin("Viewport");
    ImGui::Image(app->imguiTexture(srvSlot), ImGui::GetContentRegionAvail());
    ImGui::End();
    app->registerImguiImage(&image);  // transitioned to SHADER_RESOURCE before the ImGui draw
  }
  void onRender(ID3D12GraphicsCommandList7* cmd) override
  {
    nvdx12::cmdImageBarrier(cmd, image, D3D12_BARRIER_LAYOUT_UNORDERED_ACCESS);
    // EX: dispatch into `image`
  }
};
}  // namespace

[[maybe_unused]] static void usage_Application()
{
  nvdx12::Context        context;         // EX: context.init(contextInfo)
  nvdx12::QueueTimeline  queue;           // EX: queue.init(device, context.getQueueInfo(0))
  nvdx12::DescriptorHeap descriptorHeap;  // EX: descriptorHeap.init(device)

  nvdx12app::ApplicationCreateInfo appInfo{.name = "Usage", .windowSize = {1280, 720}, .vSync = false};
  appInfo.context        = &context;
  appInfo.queue          = &queue;
  appInfo.descriptorHeap = &descriptorHeap;
  appInfo.dockSetup      = [](ImGuiID viewportID) {
    ImGuiID settingID = ImGui::DockBuilderSplitNode(viewportID, ImGuiDir_Right, 0.25F, nullptr, &viewportID);
    ImGui::DockBuilderDockWindow("Settings", settingID);
  };

  nvdx12app::Application app;
  if(SUCCEEDED(app.init(appInfo)))
  {
    app.addElement(std::make_shared<UsageElement>());
    app.run();  // until close() or the window closes
  }
  app.deinit();
}
