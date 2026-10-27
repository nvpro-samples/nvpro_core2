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

#include "context.hpp"

#include <algorithm>
#include <iomanip>
#include <sstream>

#include <nvutils/logger.hpp>

#include "check_error.hpp"
#include "debug_util.hpp"
#include "text_encoding.hpp"

#if NVDX12_EXPORT_AGILITY_SDK
// The Agility SDK runtime reads these exports from the executable module.
// D3D12Core.dll has to live in a subdirectory of the executable:
// https://devblogs.microsoft.com/directx/gettingstarted-dx12agility/#subdirissue
extern "C" {
__declspec(dllexport) extern const unsigned int D3D12SDKVersion = NVDX12_D3D12SDKVERSION;
__declspec(dllexport) extern const char*        D3D12SDKPath    = ".\\D3D12\\";
}
#endif

namespace nvdx12 {

//--------------------------------------------------------------------------------------------------
// Catching D3D12 debug-layer messages
// https://devblogs.microsoft.com/directx/d3d12-debug-layer-message-callback/
static void CALLBACK debugMessenger(D3D12_MESSAGE_CATEGORY category,
                                    D3D12_MESSAGE_SEVERITY severity,
                                    D3D12_MESSAGE_ID       id,
                                    LPCSTR                 description,
                                    void*                  context)
{
  // Severities are ordered from CORRUPTION (0) to MESSAGE (4).
  const Context* owner = static_cast<const Context*>(context);
  if(owner && severity > owner->contextInfo.minMessageSeverity)
  {
    return;
  }
  nvutils::Logger::LogLevel level        = nvutils::Logger::LogLevel::eERROR;
  const char*               severityText = "<unknown>";
  switch(severity)
  {
    case D3D12_MESSAGE_SEVERITY_CORRUPTION:
      level        = nvutils::Logger::LogLevel::eERROR;
      severityText = "Corruption error";  // so that searching for "error" also shows this
      break;
    case D3D12_MESSAGE_SEVERITY_ERROR:
      level        = nvutils::Logger::LogLevel::eERROR;
      severityText = "Error";
      break;
    case D3D12_MESSAGE_SEVERITY_WARNING:
      level        = nvutils::Logger::LogLevel::eWARNING;
      severityText = "Warning";
      break;
    case D3D12_MESSAGE_SEVERITY_INFO:
      level        = nvutils::Logger::LogLevel::eINFO;
      severityText = "Info";
      break;
    case D3D12_MESSAGE_SEVERITY_MESSAGE:
      level        = nvutils::Logger::LogLevel::eINFO;
      severityText = "Message";
      break;
  }
  nvutils::Logger::getInstance().log(level, "D3D12 Validation %s | MessageID = %u\n%s\n", severityText,
                                     static_cast<unsigned>(id), description);
}

const char* getShaderModelName(D3D_SHADER_MODEL shaderModel)
{
  switch(int(shaderModel))
  {
    case 0x66:
      return "6.6";
    case 0x67:
      return "6.7";
    case 0x68:
      return "6.8";
    case 0x69:
      return "6.9";
    case 0x6A:
      return "6.10";
    default:
      return "< 6.6";
  }
}

// Highest shader model the runtime reports. CheckFeatureSupport fails with
// E_INVALIDARG when the requested model is unknown to the runtime, so the
// candidates are tried from the newest down.
static D3D_SHADER_MODEL queryShaderModel(ID3D12Device* device)
{
  const D3D_SHADER_MODEL candidates[] = {D3D_SHADER_MODEL(0x6A), D3D_SHADER_MODEL(0x69), D3D_SHADER_MODEL(0x68),
                                         D3D_SHADER_MODEL(0x67), D3D_SHADER_MODEL(0x66)};
  for(size_t i = 0; i < std::size(candidates); i++)
  {
    D3D12_FEATURE_DATA_SHADER_MODEL sm{candidates[i]};
    if(SUCCEEDED(device->CheckFeatureSupport(D3D12_FEATURE_SHADER_MODEL, &sm, sizeof(sm))))
    {
      return sm.HighestShaderModel;
    }
  }
  return D3D_SHADER_MODEL_6_0;
}

// Reads OPTIONS21::OpacityMicromapSupported on SDKs that declare it; the
// template keeps the member access out of translation on SDKs that do not.
template <class Options21>
static bool queryOpacityMicromapBit(ID3D12Device* device, bool fallback)
{
  if constexpr(requires(Options21 o) { o.OpacityMicromapSupported; })
  {
    Options21 options21{};
    if(SUCCEEDED(device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS21, &options21, sizeof(options21))))
    {
      return options21.OpacityMicromapSupported != 0;
    }
  }
  return fallback;
}

//--------------------------------------------------------------------------------------------------

// D3D12CreateDevice when the executable exports an Agility SDK version whose
// runtime is not in the exported D3D12SDKPath (winerror.h of recent SDKs:
// D3D12_ERROR_INVALID_HOST_EXE_SDK_VERSION).
constexpr HRESULT kInvalidHostExeSdkVersion = HRESULT(0x887E0003);

// Full path of a loaded module (null: the executable), not limited to MAX_PATH.
static std::wstring modulePath(HMODULE module)
{
  std::wstring path(MAX_PATH, L'\0');
  for(;;)
  {
    const DWORD length = GetModuleFileNameW(module, path.data(), DWORD(path.size()));
    if(length == 0)
      return {};
    if(length < path.size())
    {
      path.resize(length);
      return path;
    }
    path.resize(path.size() * 2);  // truncated
  }
}

// The D3D12Core.dll the device runs on: the Agility SDK's from the directory
// the executable exports as D3D12SDKPath, or the OS runtime when that one is
// at least as recent.
static void reportD3D12Core()
{
  const HMODULE core = GetModuleHandleW(L"D3D12Core.dll");
  if(core == nullptr)
  {
    return;
  }
  const std::wstring corePath   = modulePath(core);
  const std::wstring exePath    = modulePath(nullptr);
  const std::wstring exeDir     = exePath.substr(0, exePath.find_last_of(L'\\') + 1);
  const bool         fromExeDir = _wcsnicmp(corePath.c_str(), exeDir.c_str(), exeDir.size()) == 0;
  LOGI("D3D12Core: %s (%s)\n", utf8FromWtf16(corePath.c_str()).c_str(), fromExeDir ? "Agility SDK" : "OS runtime");
}

HRESULT Context::init(const ContextInitInfo& contextInitInfo)
{
  contextInfo = contextInitInfo;
  // Each step logs its own failure. deinit() releases whatever a failed step
  // left behind (device, info-queue callback), so a caller that returns on
  // failure does not need it.
  HRESULT hr = createInstance();
  if(SUCCEEDED(hr))
  {
    hr = selectPhysicalDevice();
  }
  if(SUCCEEDED(hr))
  {
    hr = createDevice();
  }
  if(FAILED(hr))
  {
    deinit();
    return hr;
  }
  queryFeatures();
  if(contextInfo.verbose)
  {
    reportD3D12Core();
    LOGI("D3D12 device: %s (adapter %u, %.1f GiB dedicated), RT tier %d, SM %s, OMM %s, SER %s, GPU upload heap %s\n",
         m_features.adapterName.c_str(), m_features.adapterIndex, double(m_features.dedicatedVideoMemory) / double(1ull << 30),
         int(m_features.raytracingTier), getShaderModelName(m_features.shaderModel), m_features.opacityMicromaps ? "yes" : "no",
         m_features.shaderExecutionReordering ? "yes" : "no", m_features.gpuUploadHeap ? "yes" : "no");
  }
  return S_OK;
}

void Context::deinit()
{
  if(m_messageCallbackInterface && m_messageCallbackCookie)
  {
    m_messageCallbackInterface->UnregisterMessageCallback(m_messageCallbackCookie);
    m_messageCallbackCookie = 0;
  }
  m_messageCallbackInterface = nullptr;
  m_queueInfos               = {};
  m_device                   = nullptr;
  m_adapter                  = nullptr;
  m_dxgiFactory              = nullptr;
  m_features                 = {};
}

HRESULT Context::createInstance()
{
  if(!contextInfo.experimentalFeatures.empty())
  {
    HRESULT hr = D3D12EnableExperimentalFeatures(UINT(contextInfo.experimentalFeatures.size()),
                                                 contextInfo.experimentalFeatures.data(), nullptr, nullptr);
    if(FAILED(hr) && contextInfo.experimentalFeatures.size() > 1)
    {
      // Preview features may be unavailable on this runtime; keep the first entry alone.
      hr = D3D12EnableExperimentalFeatures(1, contextInfo.experimentalFeatures.data(), nullptr, nullptr);
    }
    if(FAILED(hr))
    {
      LOGW("D3D12EnableExperimentalFeatures failed (%s)\n", hresultToString(hr).c_str());
    }
  }

  // Debug layer and DXGI debug factory, as in the DirectX-Graphics-Samples
  // HelloWindow sample and https://walbourn.github.io/dxgi-debug-device/
  UINT dxgiFactoryFlags = 0;
  if(contextInfo.enableValidationLayers)
  {
    // The D3D12 debug layer requires the Windows Graphics Tools optional
    // feature (Settings > System > Optional features) or d3d12SDKLayers.dll
    // next to D3D12Core.dll.
    ComPtr<ID3D12Debug> debugController;
    HRESULT             debugHr = D3D12GetDebugInterface(IID_PPV_ARGS(&debugController));
    if(SUCCEEDED(debugHr))
    {
      debugController->EnableDebugLayer();
      dxgiFactoryFlags |= DXGI_CREATE_FACTORY_DEBUG;
      if(contextInfo.enableGpuBasedValidation)
      {
        ComPtr<ID3D12Debug1> debugController1;
        if(SUCCEEDED(debugController->QueryInterface(IID_PPV_ARGS(&debugController1))))
        {
          debugController1->SetEnableGPUBasedValidation(TRUE);
        }
      }
      LOGI("D3D12 debug layer enabled%s\n", contextInfo.enableGpuBasedValidation ? " with GPU-based validation" : "");
    }
    else
    {
      LOGW("D3D12 debug layer requested but unavailable: %s\n", hresultToString(debugHr).c_str());
    }
  }
  if(contextInfo.enableDred)
  {
    // Independent of the debug layer: DRED is what makes a Release device
    // removal diagnosable (checkDeviceRemoved reports breadcrumbs and page faults).
    ComPtr<ID3D12DeviceRemovedExtendedDataSettings1> dred;
    if(SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&dred))))
    {
      dred->SetAutoBreadcrumbsEnablement(D3D12_DRED_ENABLEMENT_FORCED_ON);
      dred->SetPageFaultEnablement(D3D12_DRED_ENABLEMENT_FORCED_ON);
      dred->SetBreadcrumbContextEnablement(D3D12_DRED_ENABLEMENT_FORCED_ON);
    }
  }

  if(contextInfo.dxgiFactory)
  {
    m_dxgiFactory = contextInfo.dxgiFactory;
  }
  else
  {
    HRESULT hr = CreateDXGIFactory2(dxgiFactoryFlags, IID_PPV_ARGS(&m_dxgiFactory));
    if(FAILED(hr) && dxgiFactoryFlags != 0)
    {
      // The DXGI debug flag needs the Graphics Tools optional feature.
      LOGW("DXGI debug layer unavailable (%s); creating the factory without it\n", hresultToString(hr).c_str());
      hr = CreateDXGIFactory2(0, IID_PPV_ARGS(&m_dxgiFactory));
    }
    NVDX12_FAIL_RETURN(hr);
  }
  return S_OK;
}

// Fills the adapter part of the feature table.
static void recordAdapter(DeviceFeatures& features, IDXGIAdapter4* adapter, uint32_t adapterIndex)
{
  DXGI_ADAPTER_DESC1 desc{};
  adapter->GetDesc1(&desc);
  features.adapterIndex         = adapterIndex;
  features.adapterName          = utf8FromWtf16(desc.Description);
  features.adapterLuid          = desc.AdapterLuid;
  features.dedicatedVideoMemory = desc.DedicatedVideoMemory;
  LARGE_INTEGER umd{};
  if(SUCCEEDED(adapter->CheckInterfaceSupport(__uuidof(IDXGIDevice), &umd)))
  {
    const uint32_t subVersion = HIWORD(umd.LowPart);
    const uint32_t build      = LOWORD(umd.LowPart);
    // NVIDIA encodes the public version in the last five digits (x.y.1z.wwww -> zww.ww).
    features.driverVersionMajor = desc.VendorId == 0x10DE ? (subVersion % 10) * 100 + build / 100 : subVersion;
  }
}

HRESULT Context::selectPhysicalDevice()
{
  if(!m_dxgiFactory)
  {
    LOGE("%s: no DXGI factory; call createInstance() first.\n", __FUNCTION__);
    return E_FAIL;
  }
  m_adapter = nullptr;
  m_device  = nullptr;

  // Every adapter in DXGI high-performance order; the WARP adapter is among them.
  std::vector<ComPtr<IDXGIAdapter4>> adapters;
  {
    ComPtr<IDXGIAdapter4> adapter;
    for(UINT i = 0;
        SUCCEEDED(m_dxgiFactory->EnumAdapterByGpuPreference(i, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE, IID_PPV_ARGS(&adapter))); ++i)
    {
      adapters.push_back(std::move(adapter));
    }
  }

  // The checks of a created device: the selection callback, then enhanced
  // barriers, which every nvdx12 helper requires. Releases the device and
  // returns the reason when it is rejected.
  auto rejectDevice = [this](IDXGIAdapter4* adapter) -> const char* {
    D3D12_FEATURE_DATA_D3D12_OPTIONS12 options12 = {};
    const char*                        reason    = nullptr;
    if(contextInfo.preSelectPhysicalDeviceCallback
       && !(*contextInfo.preSelectPhysicalDeviceCallback)(m_dxgiFactory.Get(), adapter, m_device.Get()))
      reason = "rejected by device selection callback";
    else if(FAILED(m_device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS12, &options12, sizeof(options12))))
      reason = "could not check feature support";
    else if(!options12.EnhancedBarriersSupported)
      reason = "does not support enhanced barriers";
    if(reason != nullptr)
      m_device = nullptr;
    return reason;
  };

  if(contextInfo.useWarpDevice)
  {
    ComPtr<IDXGIAdapter4> warp;
    NVDX12_FAIL_RETURN(m_dxgiFactory->EnumWarpAdapter(IID_PPV_ARGS(&warp)));
    NVDX12_FAIL_RETURN(D3D12CreateDevice(warp.Get(), contextInfo.featureLevel, IID_PPV_ARGS(&m_device)));
    if(const char* reason = rejectDevice(warp.Get()))
    {
      LOGE("WARP adapter: %s\n", reason);
      return E_FAIL;
    }
    // Its index in the enumeration, as for a hardware adapter.
    DXGI_ADAPTER_DESC1 warpDesc{};
    warp->GetDesc1(&warpDesc);
    uint32_t warpIndex = 0;
    for(uint32_t i = 0; i < uint32_t(adapters.size()); i++)
    {
      DXGI_ADAPTER_DESC1 desc{};
      adapters[i]->GetDesc1(&desc);
      if(desc.AdapterLuid.LowPart == warpDesc.AdapterLuid.LowPart && desc.AdapterLuid.HighPart == warpDesc.AdapterLuid.HighPart)
        warpIndex = i;
    }
    m_adapter = warp;
    recordAdapter(m_features, warp.Get(), warpIndex);
    return S_OK;
  }
  const bool forced = contextInfo.forceGPU >= 0 && size_t(contextInfo.forceGPU) < adapters.size();
  if(contextInfo.forceGPU != -1 && !forced)
  {
    LOGW("forceGPU %d is not an adapter index (%zu adapters); selecting automatically\n", contextInfo.forceGPU, adapters.size());
  }

  // First adapter, in DXGI high-performance order, that supports the
  // requested feature level and passes the checks: hardware only, unless forced.
  std::stringstream textBlock;
  HRESULT           lastCreateResult = E_FAIL;
  for(uint32_t adapterIndex = 0; adapterIndex < uint32_t(adapters.size()); ++adapterIndex)
  {
    IDXGIAdapter4*     adapter = adapters[adapterIndex].Get();
    DXGI_ADAPTER_DESC1 desc{};
    adapter->GetDesc1(&desc);
    textBlock << " - " << adapterIndex << ") " << utf8FromWtf16(desc.Description);
    if(forced && adapterIndex != uint32_t(contextInfo.forceGPU))
    {
      textBlock << " (skipped: adapter " << contextInfo.forceGPU << " requested)";
    }
    else if(!forced && (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE))
    {
      textBlock << " (software)";
    }
    else if(!m_device)
    {
      // a failure is reported with the adapter list, as another adapter may still be selected
      lastCreateResult = D3D12CreateDevice(adapter, contextInfo.featureLevel, IID_PPV_ARGS(&m_device));
      if(FAILED(lastCreateResult))
      {
        textBlock << " (creation failed with HRESULT 0x" << std::hex << std::setw(8)
                  << static_cast<uint32_t>(lastCreateResult) << std::dec << ")";
      }
      else if(const char* reason = rejectDevice(adapter))
      {
        textBlock << " (" << reason << ")";
      }
      else
      {
        m_adapter = adapter;
        recordAdapter(m_features, adapter, adapterIndex);
      }
    }
    textBlock << "\n";
  }
  if(contextInfo.verbose)
  {
    LOGI("\n_________________________________________________\nAvailable adapters: %zu\n%s", adapters.size(),
         textBlock.str().c_str());
  }
  if(!m_device)
  {
    LOGE("No compatible GPU\n%s", contextInfo.verbose ? "" : textBlock.str().c_str());
    if(lastCreateResult == kInvalidHostExeSdkVersion)
    {
      LOGE(
          "The executable exports an Agility SDK version (D3D12SDKVersion) but the runtime of that version is not "
          "in the exported D3D12SDKPath, the D3D12 folder next to the executable. Copy D3D12Core.dll and "
          "d3d12SDKLayers.dll there: copy_to_runtime_and_install(... FILES ${DX12AgilitySdk_DLLS} SUBDIRECTORY \"D3D12\")\n");
    }
    return FAILED(lastCreateResult) ? lastCreateResult : E_FAIL;
  }
  if(contextInfo.verbose)
  {
    LOGI("Using adapter %u.\n", m_features.adapterIndex);
  }
  return S_OK;
}

HRESULT Context::createDevice()
{
  if(!m_adapter)
  {
    LOGE("%s: no adapter; call selectPhysicalDevice() first.\n", __FUNCTION__);
    return E_FAIL;
  }
  if(!m_device)
  {
    NVDX12_FAIL_RETURN(D3D12CreateDevice(m_adapter.Get(), contextInfo.featureLevel, IID_PPV_ARGS(&m_device)));
  }
  NVDX12_DBG_NAME(m_device);

  // The info queue is a property of the device, not of the factory.
  if(contextInfo.enableValidationLayers)
  {
    if(SUCCEEDED(m_device->QueryInterface(IID_PPV_ARGS(&m_messageCallbackInterface))))
    {
      if(FAILED(m_messageCallbackInterface->RegisterMessageCallback(debugMessenger, D3D12_MESSAGE_CALLBACK_FLAG_NONE,
                                                                    this, &m_messageCallbackCookie)))
      {
        m_messageCallbackCookie = 0;
      }
      if(contextInfo.breakOnError && IsDebuggerPresent())
      {
        m_messageCallbackInterface->SetBreakOnSeverity(D3D12_MESSAGE_SEVERITY_CORRUPTION, TRUE);
        m_messageCallbackInterface->SetBreakOnSeverity(D3D12_MESSAGE_SEVERITY_ERROR, TRUE);
      }
    }
    // An ID3D12InfoQueue without ID3D12InfoQueue1: the debug layer runs on an
    // older runtime. Without the debug layer there are no messages at all.
    else if(ComPtr<ID3D12InfoQueue> infoQueue; SUCCEEDED(m_device.As(&infoQueue)))
    {
      LOGW("ID3D12InfoQueue1 unavailable; debug messages are pulled by drainDebugMessages()\n");
    }
  }

  // One queue per request; the type stands for the queue family, and the
  // queue index counts the earlier requests of the same type.
  const size_t numRequests = contextInfo.queues.size();
  m_queueInfos.assign(numRequests, {});
  for(size_t i = 0; i < numRequests; i++)
  {
    const D3D12_COMMAND_QUEUE_DESC& desc = contextInfo.queues[i];
    const auto sameType   = [&desc](const D3D12_COMMAND_QUEUE_DESC& other) { return other.Type == desc.Type; };
    QueueInfo& queueInfo  = m_queueInfos[i];
    queueInfo.familyIndex = uint32_t(desc.Type);
    queueInfo.queueIndex = uint32_t(std::count_if(contextInfo.queues.begin(), contextInfo.queues.begin() + i, sameType));
    NVDX12_FAIL_RETURN(m_device->CreateCommandQueue(&desc, IID_PPV_ARGS(&queueInfo.queue)));
    setDebugName(queueInfo.queue.Get(), "nvdx12::Context queue " + std::to_string(i));
  }
  return S_OK;
}

void Context::queryFeatures()
{
  D3D12_FEATURE_DATA_D3D12_OPTIONS options{};
  m_device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS, &options, sizeof(options));
  m_features.typedUavLoadAdditionalFormats = options.TypedUAVLoadAdditionalFormats != 0;

  D3D12_FEATURE_DATA_D3D12_OPTIONS1 options1{};
  m_device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS1, &options1, sizeof(options1));
  m_features.waveLaneCountMin = options1.WaveLaneCountMin;
  m_features.waveLaneCountMax = options1.WaveLaneCountMax;

  D3D12_FEATURE_DATA_D3D12_OPTIONS5 options5{};
  m_device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS5, &options5, sizeof(options5));
  m_features.raytracingTier = options5.RaytracingTier;

  D3D12_FEATURE_DATA_D3D12_OPTIONS12 options12{};
  m_device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS12, &options12, sizeof(options12));
  m_features.enhancedBarriers = options12.EnhancedBarriersSupported != 0;

  D3D12_FEATURE_DATA_D3D12_OPTIONS16 options16{};
  if(SUCCEEDED(m_device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS16, &options16, sizeof(options16))))
  {
    m_features.gpuUploadHeap = options16.GPUUploadHeapSupported != 0;
  }

  m_features.shaderModel = queryShaderModel(m_device.Get());

  // Ray tracing tier 1.2 (DXR 1.2) mandates opacity micromaps and shader
  // execution reordering; SER additionally needs the SM 6.9 intrinsics.
  m_features.opacityMicromaps          = m_features.raytracingTier >= D3D12_RAYTRACING_TIER(12);
  m_features.shaderExecutionReordering = m_features.opacityMicromaps && int(m_features.shaderModel) >= 0x69;
  // SDKs that expose the dedicated OMM bit on OPTIONS21 refine the answer.
  m_features.opacityMicromaps =
      queryOpacityMicromapBit<D3D12_FEATURE_DATA_D3D12_OPTIONS21>(m_device.Get(), m_features.opacityMicromaps);

  BOOL tearing = FALSE;
  if(SUCCEEDED(m_dxgiFactory->CheckFeatureSupport(DXGI_FEATURE_PRESENT_ALLOW_TEARING, &tearing, sizeof(tearing))))
  {
    m_features.tearing = tearing != FALSE;
  }
}

bool Context::checkDeviceRemoved() const
{
  if(!m_device)
  {
    return false;
  }
  const HRESULT reason = m_device->GetDeviceRemovedReason();
  if(SUCCEEDED(reason))
  {
    return false;
  }
  LOGE("D3D12 device removed: %s\n", hresultToString(reason).c_str());

  ComPtr<ID3D12DeviceRemovedExtendedData1> dred;
  if(SUCCEEDED(m_device.As(&dred)))
  {
    D3D12_DRED_AUTO_BREADCRUMBS_OUTPUT1 breadcrumbs{};
    if(SUCCEEDED(dred->GetAutoBreadcrumbsOutput1(&breadcrumbs)))
    {
      for(const D3D12_AUTO_BREADCRUMB_NODE1* node = breadcrumbs.pHeadAutoBreadcrumbNode; node; node = node->pNext)
      {
        const UINT last = node->pLastBreadcrumbValue ? *node->pLastBreadcrumbValue : 0;
        if(last == node->BreadcrumbCount)
        {
          continue;  // this list completed
        }
        LOGE("  DRED list '%s' on queue '%s': %u of %u operations completed\n",
             node->pCommandListDebugNameA ? node->pCommandListDebugNameA : "?",
             node->pCommandQueueDebugNameA ? node->pCommandQueueDebugNameA : "?", last, node->BreadcrumbCount);
        const UINT first = last > 8 ? last - 8 : 0;
        for(UINT i = first; i < std::min(node->BreadcrumbCount, last + 2); i++)
        {
          LOGE("    op %u: breadcrumb %d%s\n", i, int(node->pCommandHistory[i]), i == last ? "  <- faulting" : "");
        }
        for(UINT c = 0; c < node->BreadcrumbContextsCount; c++)
        {
          const D3D12_DRED_BREADCRUMB_CONTEXT& context = node->pBreadcrumbContexts[c];
          if(context.BreadcrumbIndex + 2 >= first && context.BreadcrumbIndex <= last + 2)
          {
            LOGE("    marker at op %u: %s\n", context.BreadcrumbIndex, utf8FromWtf16(context.pContextString).c_str());
          }
        }
      }
    }
    D3D12_DRED_PAGE_FAULT_OUTPUT1 pageFault{};
    if(SUCCEEDED(dred->GetPageFaultAllocationOutput1(&pageFault)) && pageFault.PageFaultVA)
    {
      LOGE("  DRED page fault at GPU VA 0x%016llX\n", static_cast<unsigned long long>(pageFault.PageFaultVA));
      for(const D3D12_DRED_ALLOCATION_NODE1* node = pageFault.pHeadExistingAllocationNode; node; node = node->pNext)
      {
        LOGE("    live allocation: '%s' (type %d)\n", node->ObjectNameA ? node->ObjectNameA : "?", int(node->AllocationType));
      }
      for(const D3D12_DRED_ALLOCATION_NODE1* node = pageFault.pHeadRecentFreedAllocationNode; node; node = node->pNext)
      {
        LOGE("    recently freed: '%s' (type %d)\n", node->ObjectNameA ? node->ObjectNameA : "?", int(node->AllocationType));
      }
    }
  }
  return true;
}

void Context::drainDebugMessages()
{
  if(!m_device || m_messageCallbackCookie != 0)
  {
    return;  // the callback already forwards every message
  }
  ComPtr<ID3D12InfoQueue> queue;
  if(FAILED(m_device.As(&queue)))
  {
    return;
  }
  const UINT64      count = queue->GetNumStoredMessages();
  std::vector<char> storage;
  for(UINT64 i = 0; i < count; i++)
  {
    SIZE_T length = 0;
    if(FAILED(queue->GetMessage(i, nullptr, &length)) || length == 0)
    {
      continue;
    }
    storage.resize(length);
    D3D12_MESSAGE* message = reinterpret_cast<D3D12_MESSAGE*>(storage.data());
    if(SUCCEEDED(queue->GetMessage(i, message, &length)))
    {
      debugMessenger(message->Category, message->Severity, message->ID, message->pDescription, this);
    }
  }
  queue->ClearStoredMessages();
}

}  // namespace nvdx12

//--------------------------------------------------------------------------------------------------
// Usage example
//--------------------------------------------------------------------------------------------------
[[maybe_unused]] static void usage_Context()
{
  // Configure the D3D12 context: feature level, queues, debug layer, preview features.
  nvdx12::ContextInitInfo setup{};
  setup.featureLevel = D3D_FEATURE_LEVEL_12_1;
  setup.queues       = {{D3D12_COMMAND_LIST_TYPE_DIRECT}, {D3D12_COMMAND_LIST_TYPE_COPY}};
  setup.experimentalFeatures.push_back(D3D12ExperimentalShaderModels);

  // Example preSelectPhysicalDeviceCallback, we look for a device with ray tracing tier 1.1.
  // Providing this callback is optional and can be left out.
  setup.preSelectPhysicalDeviceCallback = [](IDXGIFactory6* /*factory*/, IDXGIAdapter4* /*adapter*/, ID3D12Device* device) {
    D3D12_FEATURE_DATA_D3D12_OPTIONS5 options5{};
    device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS5, &options5, sizeof(options5));
    return options5.RaytracingTier >= D3D12_RAYTRACING_TIER_1_1;
  };
  // To pick an adapter by its index in DXGI high-performance order instead:
  // setup.forceGPU = 1;

  // Create and initialize the D3D12 context
  nvdx12::Context context;
  if(FAILED(context.init(setup)))
  {
    LOGE("Error in D3D12 context creation\n");
    return;
  }

  // The device, the queues in the order of `setup.queues`, and the feature table.
  [[maybe_unused]] ID3D12Device* device    = context.getDevice();
  const nvdx12::QueueInfo&       copyQueue = context.getQueueInfo(1);  // familyIndex == D3D12_COMMAND_LIST_TYPE_COPY
  const nvdx12::DeviceFeatures&  features  = context.getFeatures();
  LOGI("Shader model %s, copy queue %p\n", nvdx12::getShaderModelName(features.shaderModel), copyQueue.queue.Get());

  // After a failed Present or wait: logs the reason and the DRED report when the device was lost.
  if(context.checkDeviceRemoved())
  {
    return;
  }

  context.deinit();
}
