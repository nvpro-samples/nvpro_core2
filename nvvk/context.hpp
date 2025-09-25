/*
 * Copyright (c) 2024-2025, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <functional>
#include <optional>
#include <vector>

#include <vulkan/vulkan.h>

#include "resources.hpp"

static_assert(VK_HEADER_VERSION >= 304,
              "nvvk/context.hpp requires at least Vulkan SDK 1.4.304. "
              "Please install a newer Vulkan SDK or make sure CMake's finding "
              "the correct Vulkan SDK.");

//--------------------------------------------------------------------------------------------------
// Context
//
// Usage:
//   see usage_Context in context.cpp
//--------------------------------------------------------------------------------------------------


namespace nvvk {

// Struct to hold an extension and its corresponding feature
struct ExtensionInfo
{
  const char* extensionName    = nullptr;  // Name of the extension ex. VK_KHR_SWAPCHAIN_EXTENSION_NAME
  void*       feature          = nullptr;  // [optional] Pointer to the feature structure for the extension
  bool        required         = true;     // [optional] If the extension is required
  uint32_t    specVersion      = 0;        // [optional] Spec version of the extension, this version or higher
  bool        exactSpecVersion = false;    // [optional] If true, the spec version must match exactly
};


// Specific to the creation of Vulkan context
// instanceExtensions   : Instance extensions: VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME
// deviceExtensions     : Device extensions: {{VK_KHR_SWAPCHAIN_EXTENSION_NAME}, {VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME, &accelFeature}, {OTHER}}
// queues               : All desired queues
// instanceCreateInfoExt: Instance create info extension (ex: VkLayerSettingsCreateInfoEXT)
// applicationName      : Application name
// apiVersion           : Vulkan API version
// alloc                : Allocation callbacks
// enableAllFeatures    : If true, pull all capability of `features` from the physical device
// forceGPU             : If != -1, use GPU index, useful to select a specific GPU
struct ContextInitInfo
{
  std::vector<const char*>         instanceExtensions    = {};
  std::vector<nvvk::ExtensionInfo> deviceExtensions      = {};
  std::vector<VkQueueFlags>        queues                = {VK_QUEUE_GRAPHICS_BIT};
  void*                            instanceCreateInfoExt = nullptr;
  const char*                      applicationName       = "No Engine";
  uint32_t                         apiVersion            = VK_API_VERSION_1_4;
  VkAllocationCallbacks*           alloc                 = nullptr;
  bool                             enableAllFeatures     = true;
  int32_t                          forceGPU              = -1;
#if NDEBUG
  bool enableValidationLayers = false;  // Disable validation layers in release
  bool verbose                = false;
#else
  bool enableValidationLayers = true;  // Enable validation layers
  bool verbose                = true;
#endif
  // [optional] Callback called during physical device selection process.
  // Return true to allow this physical device to be selected, false to reject it.
  std::optional<std::function<bool(VkInstance, VkPhysicalDevice)>> preSelectPhysicalDeviceCallback;
  // [optional] Callback called after device selection but before device creation.
  // Can modify ContextInitInfo to add/remove extensions, change queue config, etc.
  // Return true to proceed with device creation, false to abort.
  std::optional<std::function<bool(VkInstance, VkPhysicalDevice, ContextInitInfo&)>> postSelectPhysicalDeviceCallback;
};


//--------------------------------------------------------------------------------------------------
// Simple class to handle the Vulkan context creation
class Context
{
public:
  Context() = default;
  ~Context() { assert(m_instance == VK_NULL_HANDLE); }  // Forgot to call deinit ?

  // All-in-one instance and device creation.
  // Returns `true` on success.
  [[nodiscard]] VkResult init(const ContextInitInfo& contextInitInfo);
  void                   deinit();

  VkInstance                          getInstance() const { return m_instance; }
  VkDevice                            getDevice() const { return m_device; }
  VkPhysicalDevice                    getPhysicalDevice() const { return m_physicalDevice; }
  const nvvk::QueueInfo&              getQueueInfo(uint32_t index) const { return m_queueInfos[index]; }
  const std::vector<nvvk::QueueInfo>& getQueueInfos() const { return m_queueInfos; }
  bool                                hasExtensionEnabled(const char* name) const;

  const VkPhysicalDeviceFeatures&         getPhysicalDeviceFeatures() const { return m_deviceFeatures.features; }
  const VkPhysicalDeviceVulkan11Features& getPhysicalDeviceFeatures11() const { return m_deviceFeatures11; }
  const VkPhysicalDeviceVulkan12Features& getPhysicalDeviceFeatures12() const { return m_deviceFeatures12; }
  const VkPhysicalDeviceVulkan13Features& getPhysicalDeviceFeatures13() const { return m_deviceFeatures13; }
  const VkPhysicalDeviceVulkan14Features& getPhysicalDeviceFeatures14() const { return m_deviceFeatures14; }


public:
  // Those functions are used internally to create the Vulkan context, but could be used externally if needed.
  [[nodiscard]] VkResult createInstance();
  [[nodiscard]] VkResult selectPhysicalDevice();
  [[nodiscard]] VkResult createDevice();
  [[nodiscard]] bool     findQueueFamilies();
  ContextInitInfo        contextInfo{};  // What was used to create the information

  // Static functions to print Vulkan information
  static VkResult printVulkanVersion();
  static VkResult printInstanceLayers();
  static VkResult printInstanceExtensions(const std::vector<const char*> ext);
  static VkResult printDeviceExtensions(VkPhysicalDevice physicalDevice, const std::vector<nvvk::ExtensionInfo> ext);
  static VkResult printGpus(VkInstance instance, VkPhysicalDevice usedGpu);
  static VkResult getDeviceExtensions(VkPhysicalDevice physicalDevice, std::vector<VkExtensionProperties>& extensionProperties);

private:
  VkInstance       m_instance{};
  VkDevice         m_device{};
  VkPhysicalDevice m_physicalDevice{};

  // For device creation
  VkPhysicalDeviceFeatures2        m_deviceFeatures{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
  VkPhysicalDeviceVulkan11Features m_deviceFeatures11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES};
  VkPhysicalDeviceVulkan12Features m_deviceFeatures12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
  VkPhysicalDeviceVulkan13Features m_deviceFeatures13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
  VkPhysicalDeviceVulkan14Features m_deviceFeatures14{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_4_FEATURES};

  // For Queue creation
  std::vector<VkQueueFlags>            m_desiredQueues{};
  std::vector<VkDeviceQueueCreateInfo> m_queueCreateInfos{};
  std::vector<nvvk::QueueInfo>         m_queueInfos{};
  std::vector<std::vector<float>>      m_queuePriorities{};  // Store priorities here

  // Callback for debug messages
  VkDebugUtilsMessengerEXT m_dbgMessenger = VK_NULL_HANDLE;


  // Filters available Vulkan extensions based on desired extensions and their specifications.
  bool filterAvailableExtensions(const std::vector<VkExtensionProperties>& availableExtensions,
                                 const std::vector<ExtensionInfo>&         desiredExtensions,
                                 std::vector<ExtensionInfo>&               filteredExtensions);
};


//--------------------------------------------------------------------------
// This function adds the surface extensions needed for the platform.
// If `deviceExtensions` is provided, then it also adds the
// swapchain device extensions.
void addSurfaceExtensions(std::vector<const char*>& instanceExtensions, std::vector<nvvk::ExtensionInfo>* deviceExtensions = nullptr);

}  // namespace nvvk
