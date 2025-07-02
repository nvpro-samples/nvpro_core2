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

#pragma once
#include <span>
#include <vector>

#include <vulkan/vulkan_core.h>

#ifdef AFTERMATH_AVAILABLE
#include <map>
#include <memory>
#include "NsightAftermathGpuCrashTracker.h"
#endif

/*-----
// Usage:
// 1. Call AftermathCrashTracker::getInstance().initialize() at the beginning of the application
// 2. Add the Aftermath extension to the device extensions
// 3. Add a callback to the CheckError to catch the device lost.
// 4. Add the shader binaries to the AftermathCrashTracker::getInstance().addShaderBinary(data) when compiling shaders
// 5. Add to CMake the Aftermath library: nvpro2::nvaftermath 


Example:
  see usage_AftermathCrashTracker in aftermath.cpp


------*/


class AftermathCrashTracker
{
public:
  static AftermathCrashTracker& getInstance();
  void                          initialize();
#ifdef AFTERMATH_AVAILABLE
  GpuCrashTracker& getGpuCrashTracker();
#endif
  // Track a shader compiled with -g
  void addShaderBinary(const std::span<const uint32_t>& data);

  void errorCallback(VkResult result);

  template <typename T>
  void addExtensions(std::vector<T>& extensions);

private:
  AftermathCrashTracker();
  void initAftermath();

#ifdef AFTERMATH_AVAILABLE
  ::GpuCrashTracker::MarkerMap                                      m_marker;
  std::unique_ptr<::GpuCrashTracker>                                m_tracker;
  std::map<GFSDK_Aftermath_ShaderBinaryHash, std::vector<uint32_t>> m_shaderBinaries;
#endif

  VkPhysicalDeviceDiagnosticsConfigFeaturesNV m_diagnosticsConfigFeatures = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DIAGNOSTICS_CONFIG_FEATURES_NV};
  VkDeviceDiagnosticsConfigCreateInfoNV m_aftermath_info{.sType = VK_STRUCTURE_TYPE_DEVICE_DIAGNOSTICS_CONFIG_CREATE_INFO_NV,
                                                         .flags = VK_DEVICE_DIAGNOSTICS_CONFIG_ENABLE_SHADER_DEBUG_INFO_BIT_NV
                                                                  | VK_DEVICE_DIAGNOSTICS_CONFIG_ENABLE_RESOURCE_TRACKING_BIT_NV
                                                                  | VK_DEVICE_DIAGNOSTICS_CONFIG_ENABLE_AUTOMATIC_CHECKPOINTS_BIT_NV};
};

template <typename T>
void AftermathCrashTracker::addExtensions(std::vector<T>& extensions)
{
  extensions.emplace_back(VK_NV_DEVICE_DIAGNOSTIC_CHECKPOINTS_EXTENSION_NAME);
  extensions.emplace_back(VK_NV_DEVICE_DIAGNOSTICS_CONFIG_EXTENSION_NAME, &m_diagnosticsConfigFeatures);
  extensions.emplace_back(VK_NV_DEVICE_DIAGNOSTICS_CONFIG_EXTENSION_NAME, &m_aftermath_info);
}
