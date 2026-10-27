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

#include <cassert>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <vector>

#include "basic_types.hpp"  // before any Windows header: NOMINMAX and the Agility SDK's d3d12.h
#include <dxgi1_6.h>

#include "resources.hpp"

namespace nvdx12 {

/*-------------------------------------------------------------------------------------------------
# class nvdx12::Context

> Factory, adapter, device and command queues, the D3D12 counterpart of
> nvvk::Context.

`init` runs the steps of nvvk: `createInstance` (experimental features,
debug layer, DRED settings, DXGI factory), `selectPhysicalDevice` (the
adapter) and `createDevice` (device, info queue, queues), then
`queryFeatures`. Adapters are tried in DXGI high-performance order; the
first hardware adapter that creates a device of the requested feature
level, supports enhanced barriers and passes
`preSelectPhysicalDeviceCallback` wins. With the debug layer on, D3D12
messages of at least `minMessageSeverity` are forwarded to the logger. DRED
breadcrumbs and page-fault reports (`enableDred`) feed `checkDeviceRemoved`
in every build.

nvdx12 exports D3D12SDKVersion and D3D12SDKPath ".\D3D12\"
(NVDX12_EXPORT_AGILITY_SDK); with `verbose`, `init` logs which D3D12Core.dll
the device runs on, and a D3D12_ERROR_INVALID_HOST_EXE_SDK_VERSION failure
comes with the copy step that fixes it.

Differences with nvvk that follow from D3D12:
- getInstance / getPhysicalDevice are `getDxgiFactory` / `getAdapter`
  (IDXGIFactory6 / IDXGIAdapter4). Getters return raw pointers the context
  keeps alive, as nvvk returns handles.
- `featureLevel` stands for `apiVersion`. D3D12 has no extensions to
  request (instanceExtensions, deviceExtensions, instanceCreateInfoExt,
  enableAllFeatures): features are on when the device supports them, and
  `DeviceFeatures` (getFeatures) replaces getPhysicalDeviceFeatures*() and
  hasExtensionEnabled().
- `queues` are D3D12_COMMAND_QUEUE_DESCs, one queue each; QueueInfo's
  `familyIndex` holds the D3D12_COMMAND_LIST_TYPE (no findQueueFamilies).
- `preSelectPhysicalDeviceCallback` also receives a device created on the
  adapter, since D3D12 reports features through a device only. The
  selection keeps that device for `createDevice` (D3D12CreateDevice returns
  the same device per adapter anyway).
- D3D12 knobs standing for nvvk's validation settings:
  `enableGpuBasedValidation` (gpuav_enable), `breakOnError`
  (debug_action break), `minMessageSeverity` (the debug messenger's
  severity mask). `enableDred`, `useWarpDevice`, `experimentalFeatures` and
  `dxgiFactory` have no Vulkan equivalent.
- Left out: `alloc` (no allocation callbacks), `applicationName`,
  `internallySynchronizedQueues` (D3D12 queues are free-threaded),
  `postSelectPhysicalDeviceCallback` (call the steps one by one instead),
  the print* statics (the verbose selection log lists every adapter with
  the reason it was skipped) and addSurfaceExtensions (no surfaces).

```cpp
nvdx12::ContextInitInfo info;
info.featureLevel         = D3D_FEATURE_LEVEL_12_1;
info.experimentalFeatures = {D3D12ExperimentalShaderModels};
nvdx12::Context context;
NVDX12_CHECK(context.init(info));
const nvdx12::DeviceFeatures& features = context.getFeatures();
```

See usage_Context() in context.cpp.
-------------------------------------------------------------------------------------------------*/

struct ContextInitInfo
{
  // Minimum required feature level (nvvk: apiVersion).
  D3D_FEATURE_LEVEL featureLevel = D3D_FEATURE_LEVEL_11_0;

  // All desired queues, one queue per entry.
  std::vector<D3D12_COMMAND_QUEUE_DESC> queues = {{D3D12_COMMAND_LIST_TYPE_DIRECT}};

  // If != -1, the adapter to use, by index in DXGI high-performance order
  // (EnumAdapterByGpuPreference, which lists the WARP adapter too, so it can
  // be forced). The adapter still has to pass the checks; an index past the
  // last adapter falls back to the automatic selection, as in nvvk.
  int32_t forceGPU = -1;

#if NDEBUG
  bool enableValidationLayers = false;  // Disable the debug layer in release
  bool verbose                = false;  // Whether to print out additional system information
#else
  bool enableValidationLayers = true;
  bool verbose                = true;
#endif
  // GPU-based validation; slows execution down. Requires the debug layer.
  bool enableGpuBasedValidation = false;
  // Break into an attached debugger on ERROR / CORRUPTION messages (debug layer on).
  bool breakOnError = false;
  // Debug-layer messages below this severity are dropped before they reach
  // the logger; the default forwards warnings and errors, as nvvk does.
  D3D12_MESSAGE_SEVERITY minMessageSeverity = D3D12_MESSAGE_SEVERITY_WARNING;
  // Device Removed Extended Data (breadcrumbs, page faults) for checkDeviceRemoved(), debug layer or not.
  bool enableDred = true;

  // Whether to use the WARP adapter rather than a GPU; `forceGPU` is ignored.
  // The WARP device goes through the same checks (enhanced barriers,
  // preSelectPhysicalDeviceCallback) as a GPU.
  bool useWarpDevice = false;

  // Features to enable with D3D12EnableExperimentalFeatures before the device
  // is created (D3D12ExperimentalShaderModels, preview GUIDs). When the set
  // fails, the first entry alone is retried and a failure is only a warning.
  std::vector<GUID> experimentalFeatures;

  // [optional] Pre-created DXGI factory (an interposer such as Streamline's);
  // the context creates its own when null.
  ComPtr<IDXGIFactory6> dxgiFactory;

  // [optional] Callback called during the adapter selection, with a device
  // created on the adapter to query its features.
  // Return true to allow this adapter to be selected, false to reject it.
  std::optional<std::function<bool(IDXGIFactory6*, IDXGIAdapter4*, ID3D12Device*)>> preSelectPhysicalDeviceCallback;
};

// What the selected adapter and device support: the counterpart of nvvk's
// physical-device properties and VkPhysicalDeviceFeatures2 chain.
struct DeviceFeatures
{
  D3D12_RAYTRACING_TIER raytracingTier{D3D12_RAYTRACING_TIER_NOT_SUPPORTED};
  D3D_SHADER_MODEL      shaderModel{D3D_SHADER_MODEL_6_0};
  bool                  enhancedBarriers{false};
  bool                  opacityMicromaps{false};           // ray tracing tier 1.2
  bool                  shaderExecutionReordering{false};  // tier 1.2 with the Shader Model 6.9 intrinsics
  bool                  typedUavLoadAdditionalFormats{false};
  bool                  gpuUploadHeap{false};  // D3D12_HEAP_TYPE_GPU_UPLOAD (resizable BAR)
  bool                  tearing{false};        // DXGI_FEATURE_PRESENT_ALLOW_TEARING
  uint32_t              waveLaneCountMin{0};
  uint32_t              waveLaneCountMax{0};
  uint64_t              dedicatedVideoMemory{0};
  std::string           adapterName;
  uint32_t              adapterIndex{0};  // in DXGI high-performance order, as ContextInitInfo::forceGPU
  LUID                  adapterLuid{};
  uint32_t              driverVersionMajor{0};  // from the UMD version, informational
};

// "6.6" ... "6.10" for the models the helpers know, "< 6.6" otherwise.
const char* getShaderModelName(D3D_SHADER_MODEL shaderModel);

class Context
{
public:
  Context() = default;
  ~Context() { assert(!m_device && "Missing deinit()?"); }
  // The debug-layer callback holds `this`.
  Context(const Context&)            = delete;
  Context& operator=(const Context&) = delete;

  // All-in-one factory, adapter, device and queue creation.
  // Returns S_OK on success.
  [[nodiscard]] HRESULT init(const ContextInitInfo& contextInitInfo);
  void                  deinit();

  IDXGIFactory6*                getDxgiFactory() const { return m_dxgiFactory.Get(); }  // nvvk: getInstance
  ID3D12Device*                 getDevice() const { return m_device.Get(); }
  IDXGIAdapter4*                getAdapter() const { return m_adapter.Get(); }  // nvvk: getPhysicalDevice
  const QueueInfo&              getQueueInfo(uint32_t index) const { return m_queueInfos[index]; }
  const std::vector<QueueInfo>& getQueueInfos() const { return m_queueInfos; }
  const DeviceFeatures&         getFeatures() const { return m_features; }
  // Null when the debug layer is off or the runtime has no message callback.
  ID3D12InfoQueue1* getInfoQueue() const { return m_messageCallbackInterface.Get(); }

  // Logs the removal reason, and the DRED breadcrumbs and page faults when
  // enabled, if the device has been lost. Returns true when lost. Fence
  // waits and tests report completion once the device is lost (queue.hpp,
  // semaphore.hpp) so that polling loops end; their callers tell a loss
  // from a completion with this function.
  bool checkDeviceRemoved() const;

  // Pulls stored debug-layer messages and forwards them to the logger, for
  // runtimes without the message callback.
  void drainDebugMessages();

public:
  // Those functions are used internally to create the D3D12 context, but could be used externally if needed.
  // Experimental features, debug layer, DRED settings and the DXGI factory.
  [[nodiscard]] HRESULT createInstance();
  // Picks the adapter; keeps the device created on it to check it.
  [[nodiscard]] HRESULT selectPhysicalDevice();
  // The device on the selected adapter, its info queue, and the queues.
  [[nodiscard]] HRESULT createDevice();
  // Fills the DeviceFeatures of the device.
  void            queryFeatures();
  ContextInitInfo contextInfo{};  // What was used to create the context

private:
  ComPtr<IDXGIFactory6>    m_dxgiFactory;
  ComPtr<IDXGIAdapter4>    m_adapter;
  ComPtr<ID3D12Device>     m_device;
  std::vector<QueueInfo>   m_queueInfos;
  ComPtr<ID3D12InfoQueue1> m_messageCallbackInterface;  // debug-layer messages to the logger
  DWORD                    m_messageCallbackCookie = 0;
  DeviceFeatures           m_features{};
};

}  // namespace nvdx12
