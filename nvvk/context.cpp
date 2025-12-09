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


#include <csignal>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <unordered_set>

#include <volk.h>
#include <vulkan/vk_enum_string_helper.h>
#include <nvutils/logger.hpp>
#include <nvutils/timers.hpp>

#include "check_error.hpp"
#include "debug_util.hpp"
#include "context.hpp"

//--------------------------------------------------------------------------------------------------
// CATCHING VULKAN ERRORS
//--------------------------------------------------------------------------------------------------
static VKAPI_ATTR VkBool32 VKAPI_CALL VkContextDebugReport(VkDebugUtilsMessageSeverityFlagBitsEXT      messageSeverity,
                                                           VkDebugUtilsMessageTypeFlagsEXT             messageType,
                                                           const VkDebugUtilsMessengerCallbackDataEXT* callbackData,
                                                           void*                                       userData)
{
  const nvutils::Logger::LogLevel level =
      (messageSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) != 0   ? nvutils::Logger::LogLevel::eERROR :
      (messageSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) != 0 ? nvutils::Logger::LogLevel::eWARNING :
                                                                                 nvutils::Logger::LogLevel::eINFO;

  const char* levelString = (messageSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) != 0   ? "Error" :
                            (messageSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) != 0 ? "Warning" :
                                                                                                       "Info";

  nvutils::Logger::getInstance().log(level, "Validation %s: [ %s ] | MessageID = 0x%x\n%s\n", levelString,
                                     callbackData->pMessageIdName, callbackData->messageIdNumber, callbackData->pMessage);
  return VK_FALSE;
}


//////////////////////////////////////////////////////////////////////////
///
//////////////////////////////////////////////////////////////////////////

VkResult nvvk::Context::init(const ContextInitInfo& contextInitInfo)
{
  // Initialize the context information
  contextInfo = contextInitInfo;

  // Initialize the Vulkan loader
  NVVK_FAIL_RETURN(volkInitialize());

  {
    nvutils::ScopedTimer st("Creating Vulkan Context");

    NVVK_FAIL_RETURN(createInstance())
    NVVK_FAIL_RETURN(selectPhysicalDevice())
    NVVK_FAIL_RETURN(createDevice())

    nvvk::DebugUtil::getInstance().init(m_device);  // Initialize the debug utility

    NVVK_DBG_NAME(m_instance);
    NVVK_DBG_NAME(m_device);
    NVVK_DBG_NAME(m_physicalDevice);
    for(auto& q : m_queueInfos)
    {
      NVVK_DBG_NAME(q.queue);
    }
  }
  if(contextInfo.verbose)
  {
    NVVK_FAIL_RETURN(printVulkanVersion());
    NVVK_FAIL_RETURN(printInstanceLayers());
    NVVK_FAIL_RETURN(printInstanceExtensions(contextInfo.instanceExtensions));
    NVVK_FAIL_RETURN(printDeviceExtensions(m_physicalDevice, contextInfo.deviceExtensions));
    NVVK_FAIL_RETURN(printGpus(m_instance, m_physicalDevice));
    LOGI("_________________________________________________\n");
  }
  return VK_SUCCESS;
}

void nvvk::Context::deinit()
{
  if(m_device)
  {
    vkDestroyDevice(m_device, contextInfo.alloc);
  }

  if(m_instance)
  {
    if(m_dbgMessenger && vkDestroyDebugUtilsMessengerEXT)
    {
      vkDestroyDebugUtilsMessengerEXT(m_instance, m_dbgMessenger, contextInfo.alloc);
      m_dbgMessenger = VK_NULL_HANDLE;
    }
    vkDestroyInstance(m_instance, contextInfo.alloc);
  }
  m_device   = VK_NULL_HANDLE;
  m_instance = VK_NULL_HANDLE;
}

VkResult nvvk::Context::createInstance()
{
  nvutils::ScopedTimer st(__FUNCTION__);

  VkApplicationInfo appInfo{
      .sType              = VK_STRUCTURE_TYPE_APPLICATION_INFO,
      .pApplicationName   = contextInfo.applicationName,
      .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
      .pEngineName        = "My Engine",
      .engineVersion      = VK_MAKE_VERSION(1, 0, 0),
      .apiVersion         = contextInfo.apiVersion,
  };

  std::vector<const char*> layers;
  if(contextInfo.enableValidationLayers)
  {
    layers.push_back("VK_LAYER_KHRONOS_validation");
  }

  VkInstanceCreateInfo createInfo{
      .sType                   = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
      .pNext                   = contextInfo.instanceCreateInfoExt,
      .pApplicationInfo        = &appInfo,
      .enabledLayerCount       = uint32_t(layers.size()),
      .ppEnabledLayerNames     = layers.data(),
      .enabledExtensionCount   = uint32_t(contextInfo.instanceExtensions.size()),
      .ppEnabledExtensionNames = contextInfo.instanceExtensions.data(),
  };


  VkResult result = vkCreateInstance(&createInfo, contextInfo.alloc, &m_instance);
  if(result != VK_SUCCESS)
  {
    // Since the debug utils aren't available yet and this is usually the first
    // place an app can fail, we should print some additional help here.
    LOGE(
        "vkCreateInstance failed with error %s!\n"
        "You may need to install a newer Vulkan SDK, or check that it is properly installed.\n",
        string_VkResult(result));
    return result;
  }
  // Loading Vulkan functions
  volkLoadInstance(m_instance);

  if(contextInfo.enableValidationLayers)
  {
    if(vkCreateDebugUtilsMessengerEXT)
    {
      VkDebugUtilsMessengerCreateInfoEXT dbg_messenger_create_info{VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT};
      dbg_messenger_create_info.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT   // GPU info, bug
                                                  | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;  // Invalid usage
      dbg_messenger_create_info.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT        // Violation of spec
                                              | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;    // Non-optimal use
      //      dbg_messenger_create_info.messageSeverity |=
      //          VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT;
      //      dbg_messenger_create_info.messageType |=
      //          VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_DEVICE_ADDRESS_BINDING_BIT_EXT;
      dbg_messenger_create_info.pfnUserCallback = VkContextDebugReport;
      NVVK_FAIL_RETURN(vkCreateDebugUtilsMessengerEXT(m_instance, &dbg_messenger_create_info, nullptr, &m_dbgMessenger));
    }
    else
    {
      LOGW("\nMissing VK_EXT_DEBUG_UTILS extension, cannot use vkCreateDebugUtilsMessengerEXT for validation layers.\n");
    }
  }
  return VK_SUCCESS;
}

// Returns true if and only if Vulkan versionA >= Vulkan versionB, ignoring the
// variant part of the version.
static bool vkVersionAtLeast(uint32_t versionA, uint32_t versionB)
{
  const uint32_t aWithoutVariant = versionA - VK_MAKE_API_VERSION(VK_API_VERSION_VARIANT(versionA), 0, 0, 0);
  const uint32_t bWithoutVariant = versionB - VK_MAKE_API_VERSION(VK_API_VERSION_VARIANT(versionB), 0, 0, 0);
  return aWithoutVariant >= bWithoutVariant;
}

VkResult nvvk::Context::selectPhysicalDevice()
{
  if(m_instance == VK_NULL_HANDLE)
  {
    LOGE("%s: m_instance was null; call createInstance() first.", __FUNCTION__);
    return VK_ERROR_INITIALIZATION_FAILED;
  }

  // nvutils::ScopedTimer st(std::string(__FUNCTION__) + "\n");
  uint32_t deviceCount = 0;
  NVVK_FAIL_RETURN(vkEnumeratePhysicalDevices(m_instance, &deviceCount, nullptr));
  if(deviceCount == 0)
  {
    LOGE("%s: Failed to find any GPUs with Vulkan support!", __FUNCTION__);
    return VK_ERROR_INITIALIZATION_FAILED;
  }
  std::vector<VkPhysicalDevice> gpus(deviceCount);
  NVVK_FAIL_RETURN(vkEnumeratePhysicalDevices(m_instance, &deviceCount, gpus.data()));

  if((contextInfo.forceGPU == -1) || (contextInfo.forceGPU >= int(deviceCount)))
  {
    // Find the discrete GPU if one is present. If not, use the first one available.
    m_physicalDevice = gpus[0];
    for(VkPhysicalDevice& device : gpus)
    {
      if(contextInfo.preSelectPhysicalDeviceCallback)
      {
        if(!(*contextInfo.preSelectPhysicalDeviceCallback)(m_instance, device))
        {
          continue;
        }
      }
      VkPhysicalDeviceProperties properties;
      vkGetPhysicalDeviceProperties(device, &properties);
      if(properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)
      {
        m_physicalDevice = device;
        break;
      }
    }
  }
  else
  {
    // Using specified GPU
    m_physicalDevice = gpus[contextInfo.forceGPU];
  }

  {  // Check for available Vulkan version
    VkPhysicalDeviceProperties properties;
    vkGetPhysicalDeviceProperties(m_physicalDevice, &properties);
    uint32_t apiVersion = properties.apiVersion;
    if(!vkVersionAtLeast(apiVersion, contextInfo.apiVersion))
    {
      LOGW("Requested Vulkan version (%d.%d) is higher than available version (%d.%d).\n", VK_VERSION_MAJOR(contextInfo.apiVersion),
           VK_VERSION_MINOR(contextInfo.apiVersion), VK_VERSION_MAJOR(apiVersion), VK_VERSION_MINOR(apiVersion));
      m_physicalDevice = {};
      return VK_ERROR_INITIALIZATION_FAILED;
    }
  }

  // Query the physical device features
  m_deviceFeatures.pNext = &m_deviceFeatures11;
  if(vkVersionAtLeast(contextInfo.apiVersion, VK_API_VERSION_1_2))
    m_deviceFeatures11.pNext = &m_deviceFeatures12;
  if(vkVersionAtLeast(contextInfo.apiVersion, VK_API_VERSION_1_3))
    m_deviceFeatures12.pNext = &m_deviceFeatures13;
  if(vkVersionAtLeast(contextInfo.apiVersion, VK_API_VERSION_1_4))
    m_deviceFeatures13.pNext = &m_deviceFeatures14;
  vkGetPhysicalDeviceFeatures2(m_physicalDevice, &m_deviceFeatures);

  // Find the queues that we need
  m_desiredQueues = contextInfo.queues;
  if(!findQueueFamilies())
  {
    m_physicalDevice = {};
    return VK_ERROR_INITIALIZATION_FAILED;
  }


  return VK_SUCCESS;
}

static inline void pNextChainPushFront(void* mainStruct, void* newStruct)
{
  auto* newBaseStruct  = reinterpret_cast<VkBaseOutStructure*>(newStruct);
  auto* mainBaseStruct = reinterpret_cast<VkBaseOutStructure*>(mainStruct);

  newBaseStruct->pNext  = mainBaseStruct->pNext;
  mainBaseStruct->pNext = newBaseStruct;
}

VkResult nvvk::Context::createDevice()
{
  if(m_physicalDevice == VK_NULL_HANDLE)
  {
    LOGE("m_physicalDevice was null; call selectPhysicalDevice() first.");
    return VK_ERROR_INITIALIZATION_FAILED;
  }

  // Physical device has been chosen. Last chance to make changes to the contextInfo, like adding more extensions
  // (which might be dependent on the selected physical device)
  if(contextInfo.postSelectPhysicalDeviceCallback)
  {
    if(!(*contextInfo.postSelectPhysicalDeviceCallback)(m_instance, m_physicalDevice, contextInfo))
    {
      return VK_ERROR_INITIALIZATION_FAILED;
    }
  }

  // Filter the available extensions otherwise the device creation will fail
  std::vector<ExtensionInfo>         filteredExtensions;
  std::vector<VkExtensionProperties> extensionProperties;
  NVVK_FAIL_RETURN(getDeviceExtensions(m_physicalDevice, extensionProperties));
  bool allFound = filterAvailableExtensions(extensionProperties, contextInfo.deviceExtensions, filteredExtensions);
  if(!allFound)
  {
    m_physicalDevice = {};
    return VK_ERROR_INITIALIZATION_FAILED;
  }

  contextInfo.deviceExtensions = std::move(filteredExtensions);
  // nvutils::ScopedTimer st(__FUNCTION__);
  // Chain all custom features to the pNext chain of m_deviceFeatures
  for(const auto& extension : contextInfo.deviceExtensions)
  {
    if(extension.feature)
      pNextChainPushFront(&m_deviceFeatures, extension.feature);
  }
  // Activate features on request
  if(contextInfo.enableAllFeatures)
    vkGetPhysicalDeviceFeatures2(m_physicalDevice, &m_deviceFeatures);

  // List of extensions to enable
  std::vector<const char*> enabledExtensions;
  for(const auto& ext : contextInfo.deviceExtensions)
  {
    enabledExtensions.push_back(ext.extensionName);
  }

  VkDeviceCreateInfo createInfo{
      .sType                   = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
      .pNext                   = &m_deviceFeatures,
      .queueCreateInfoCount    = uint32_t(m_queueCreateInfos.size()),
      .pQueueCreateInfos       = m_queueCreateInfos.data(),
      .enabledExtensionCount   = static_cast<uint32_t>(enabledExtensions.size()),
      .ppEnabledExtensionNames = enabledExtensions.data(),
  };

  const VkResult result = vkCreateDevice(m_physicalDevice, &createInfo, contextInfo.alloc, &m_device);
  if(VK_SUCCESS != result)
  {
    LOGE("vkCreateDevice failed with error %s!", string_VkResult(result));
    return result;
  }
  volkLoadDevice(m_device);

  for(auto& queue : m_queueInfos)
    vkGetDeviceQueue(m_device, queue.familyIndex, queue.queueIndex, &queue.queue);

  return VK_SUCCESS;
}

bool nvvk::Context::findQueueFamilies()
{
  uint32_t queueFamilyCount;
  vkGetPhysicalDeviceQueueFamilyProperties(m_physicalDevice, &queueFamilyCount, nullptr);
  std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
  vkGetPhysicalDeviceQueueFamilyProperties(m_physicalDevice, &queueFamilyCount, queueFamilies.data());

  std::unordered_map<uint32_t, uint32_t> queueFamilyUsage;
  for(uint32_t i = 0; i < queueFamilyCount; ++i)
  {
    queueFamilyUsage[i] = 0;
  }

  for(size_t i = 0; i < m_desiredQueues.size(); ++i)
  {
    bool found = false;
    for(uint32_t j = 0; j < queueFamilyCount; ++j)
    {
      // Check for an exact match and unused queue family
      // Avoid queue family with VK_QUEUE_GRAPHICS_BIT if not needed
      if((queueFamilies[j].queueFlags & m_desiredQueues[i]) == m_desiredQueues[i] && queueFamilyUsage[j] == 0
         && ((m_desiredQueues[i] & VK_QUEUE_GRAPHICS_BIT) || !(queueFamilies[j].queueFlags & VK_QUEUE_GRAPHICS_BIT)))
      {
        m_queueInfos.push_back({j, queueFamilyUsage[j]});
        queueFamilyUsage[j]++;
        found = true;
        break;
      }
    }

    if(!found)
    {
      for(uint32_t j = 0; j < queueFamilyCount; ++j)
      {
        // Check for an exact match and allow reuse if queue count not exceeded
        // Avoid queue family with VK_QUEUE_GRAPHICS_BIT if not needed
        if((queueFamilies[j].queueFlags & m_desiredQueues[i]) == m_desiredQueues[i]
           && queueFamilyUsage[j] < queueFamilies[j].queueCount
           && ((m_desiredQueues[i] & VK_QUEUE_GRAPHICS_BIT) || !(queueFamilies[j].queueFlags & VK_QUEUE_GRAPHICS_BIT)))
        {
          m_queueInfos.push_back({j, queueFamilyUsage[j]});
          queueFamilyUsage[j]++;
          found = true;
          break;
        }
      }
    }

    if(!found)
    {
      for(uint32_t j = 0; j < queueFamilyCount; ++j)
      {
        // Check for a partial match and allow reuse if queue count not exceeded
        // Avoid queue family with VK_QUEUE_GRAPHICS_BIT if not needed
        if((queueFamilies[j].queueFlags & m_desiredQueues[i]) && queueFamilyUsage[j] < queueFamilies[j].queueCount
           && ((m_desiredQueues[i] & VK_QUEUE_GRAPHICS_BIT) || !(queueFamilies[j].queueFlags & VK_QUEUE_GRAPHICS_BIT)))
        {
          m_queueInfos.push_back({j, queueFamilyUsage[j]});
          queueFamilyUsage[j]++;
          found = true;
          break;
        }
      }
    }

    if(!found)
    {
      for(uint32_t j = 0; j < queueFamilyCount; ++j)
      {
        // Check for a partial match and allow reuse if queue count not exceeded
        if((queueFamilies[j].queueFlags & m_desiredQueues[i]) && queueFamilyUsage[j] < queueFamilies[j].queueCount)
        {
          m_queueInfos.push_back({j, queueFamilyUsage[j]});
          queueFamilyUsage[j]++;
          found = true;
          break;
        }
      }
    }

    if(!found)
    {
      // If no suitable queue family is found, assert a failure
      LOGE("Failed to find a suitable queue family!");
      return false;
    }
  }

  for(const auto& usage : queueFamilyUsage)
  {
    if(usage.second > 0)
    {
      m_queuePriorities.emplace_back(usage.second, 1.0f);  // Same priority for all queues in a family
      m_queueCreateInfos.push_back({VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, nullptr, 0, usage.first, usage.second,
                                    m_queuePriorities.back().data()});
    }
  }
  return true;
}

bool nvvk::Context::filterAvailableExtensions(const std::vector<VkExtensionProperties>& availableExtensions,
                                              const std::vector<ExtensionInfo>&         desiredExtensions,
                                              std::vector<ExtensionInfo>&               filteredExtensions)
{
  bool allFound = true;

  // Create a map for quick lookup of available extensions and their versions
  std::unordered_map<std::string, uint32_t> availableExtensionsMap;
  for(const auto& ext : availableExtensions)
  {
    availableExtensionsMap[ext.extensionName] = ext.specVersion;
  }

  // Iterate through all desired extensions
  for(const auto& desiredExtension : desiredExtensions)
  {
    auto it = availableExtensionsMap.find(desiredExtension.extensionName);

    bool     found        = it != availableExtensionsMap.end();
    uint32_t specVersion  = found ? it->second : 0;
    bool     validVersion = desiredExtension.exactSpecVersion ? desiredExtension.specVersion == specVersion :
                                                                desiredExtension.specVersion <= specVersion;
    if(found && validVersion)
    {
      filteredExtensions.push_back(desiredExtension);
    }
    else
    {
      std::string versionInfo;
      if(desiredExtension.specVersion != 0 || desiredExtension.exactSpecVersion)
      {
        versionInfo = " (v." + std::to_string(specVersion) + " " + (specVersion ? "== " : ">= ")
                      + std::to_string(desiredExtension.specVersion) + ")";
      }
      if(desiredExtension.required)
        allFound = false;
      nvutils::Logger::getInstance().log(
          desiredExtension.required ? nvutils::Logger::LogLevel::eERROR : nvutils::Logger::LogLevel::eWARNING,
          "Extension not available: %s %s\n", desiredExtension.extensionName, versionInfo.c_str());
    }
  }

  return allFound;
}

bool nvvk::Context::hasExtensionEnabled(const char* name) const
{
  for(auto& ext : contextInfo.deviceExtensions)
  {
    if(std::strcmp(ext.extensionName, name) == 0)  // Compare string content
    {
      return true;
    }
  }
  return false;
}


//--------------------------------------------------------------------------------------------------
// Static functions to print Vulkan information


static std::string getVendorName(uint32_t vendorID)
{
  static const std::unordered_map<uint32_t, std::string> vendorMap = {{0x1002, "AMD"},      {0x1010, "ImgTec"},
                                                                      {0x10DE, "NVIDIA"},   {0x13B5, "ARM"},
                                                                      {0x5143, "Qualcomm"}, {0x8086, "INTEL"}};

  auto it = vendorMap.find(vendorID);
  return it != vendorMap.end() ? it->second : "Unknown Vendor";
}

static std::string getDeviceType(uint32_t deviceType)
{
  static const std::unordered_map<uint32_t, std::string> deviceTypeMap = {{VK_PHYSICAL_DEVICE_TYPE_OTHER, "Other"},
                                                                          {VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU, "Integrated GPU"},
                                                                          {VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU, "Discrete GPU"},
                                                                          {VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU, "Virtual GPU"},
                                                                          {VK_PHYSICAL_DEVICE_TYPE_CPU, "CPU"}};

  auto it = deviceTypeMap.find(deviceType);
  return it != deviceTypeMap.end() ? it->second : "Unknown";
}


static std::string getVersionString(uint32_t version)
{
  return std::to_string(VK_VERSION_MAJOR(version)) + "."    //
         + std::to_string(VK_VERSION_MINOR(version)) + "."  //
         + std::to_string(VK_VERSION_PATCH(version));
}

static void printPhysicalDeviceProperties(const VkPhysicalDeviceProperties& properties)
{
  LOGI(" - Device Name    : %s\n", properties.deviceName);
  LOGI(" - Vendor         : %s\n", getVendorName(properties.vendorID).c_str());
  LOGI(" - Driver Version : %s\n", getVersionString(properties.driverVersion).c_str());
  LOGI(" - API Version    : %s\n", getVersionString(properties.apiVersion).c_str());
  LOGI(" - Device Type    : %s\n", getDeviceType(properties.deviceType).c_str());
}


VkResult nvvk::Context::printVulkanVersion()
{
  uint32_t version;
  NVVK_FAIL_RETURN(vkEnumerateInstanceVersion(&version));
  LOGI(
      "\n_________________________________________________\n"
      "Vulkan Version:  %d.%d.%d\n",
      VK_VERSION_MAJOR(version), VK_VERSION_MINOR(version), VK_VERSION_PATCH(version));
  return VK_SUCCESS;
}


VkResult nvvk::Context::printInstanceLayers()
{
  uint32_t                       count;
  std::vector<VkLayerProperties> layerProperties;
  NVVK_FAIL_RETURN(vkEnumerateInstanceLayerProperties(&count, nullptr));
  layerProperties.resize(count);
  NVVK_FAIL_RETURN(vkEnumerateInstanceLayerProperties(&count, layerProperties.data()));
  std::stringstream textBlock;
  for(auto& it : layerProperties)
  {
    textBlock << it.layerName << " (v. " << VK_VERSION_MAJOR(it.specVersion) << '.' << VK_VERSION_MINOR(it.specVersion) << '.'
              << VK_VERSION_PATCH(it.specVersion) << ' ' << it.implementationVersion << ") : " << it.description << '\n';
  }
  LOGI(
      "\n"
      "_________________________________________________\n"
      "Available Instance Layers :\n"
      "%s",
      textBlock.str().c_str());
  return VK_SUCCESS;
}

VkResult nvvk::Context::printInstanceExtensions(const std::vector<const char*> ext)
{
  std::unordered_set<std::string> exist;
  for(auto& e : ext)
    exist.insert(e);

  uint32_t                           count;
  std::vector<VkExtensionProperties> extensionProperties;
  NVVK_FAIL_RETURN(vkEnumerateInstanceExtensionProperties(nullptr, &count, nullptr));
  extensionProperties.resize(count);
  NVVK_FAIL_RETURN(vkEnumerateInstanceExtensionProperties(nullptr, &count, extensionProperties.data()));
  std::stringstream textBlock;
  for(const VkExtensionProperties& it : extensionProperties)
  {
    const char exists = ((exist.find(it.extensionName) != exist.end()) ? 'x' : ' ');
    textBlock << '[' << exists << "] " << it.extensionName << " (v. " << it.specVersion << ")\n";
  }
  LOGI(
      "\n"
      "_________________________________________________\n"
      "Available Instance Extensions :\n"
      "%s",
      textBlock.str().c_str());
  return VK_SUCCESS;
}

VkResult nvvk::Context::getDeviceExtensions(VkPhysicalDevice physicalDevice, std::vector<VkExtensionProperties>& extensionProperties)
{
  uint32_t count{};
  NVVK_FAIL_RETURN(vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, &count, nullptr));
  extensionProperties.resize(count);
  NVVK_FAIL_RETURN(vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, &count, extensionProperties.data()));
  extensionProperties.resize(std::min(extensionProperties.size(), size_t(count)));
  return VK_SUCCESS;
}

VkResult nvvk::Context::printDeviceExtensions(VkPhysicalDevice physicalDevice, const std::vector<nvvk::ExtensionInfo> ext)
{
  if(physicalDevice == VK_NULL_HANDLE)
    return VK_ERROR_INITIALIZATION_FAILED;

  std::unordered_set<std::string> exist;
  for(auto& e : ext)
    exist.insert(e.extensionName);

  uint32_t                           count;
  std::vector<VkExtensionProperties> extensionProperties;
  NVVK_FAIL_RETURN(vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, &count, nullptr));
  extensionProperties.resize(count);
  NVVK_FAIL_RETURN(vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, &count, extensionProperties.data()));
  std::stringstream textBlock;
  for(const VkExtensionProperties& it : extensionProperties)
  {
    const char exists = ((exist.find(it.extensionName) != exist.end()) ? 'x' : ' ');
    textBlock << '[' << exists << "] " << it.extensionName << " (v. " << it.specVersion << ")\n";
  }
  LOGI(
      "\n"
      "_________________________________________________\n"
      "Available Device Extensions :\n"
      "%s",
      textBlock.str().c_str());
  return VK_SUCCESS;
}

VkResult nvvk::Context::printGpus(VkInstance instance, VkPhysicalDevice usedGpu)
{
  uint32_t deviceCount = 0;
  NVVK_FAIL_RETURN(vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr));
  std::vector<VkPhysicalDevice> gpus(deviceCount);
  NVVK_FAIL_RETURN(vkEnumeratePhysicalDevices(instance, &deviceCount, gpus.data()));

  std::stringstream          textBlock;
  VkPhysicalDeviceProperties properties;
  uint32_t                   usedGpuIndex = 0;
  for(uint32_t d = 0; d < deviceCount; d++)
  {
    if(gpus[d] == usedGpu)
      usedGpuIndex = d;
    vkGetPhysicalDeviceProperties(gpus[d], &properties);
    textBlock << " - " << d << ") " << properties.deviceName << "\n";
  }
  LOGI(
      "\n"
      "_________________________________________________\n"
      "Available GPUs: %d\n"
      "%s",
      deviceCount, textBlock.str().c_str());
  if(usedGpu == VK_NULL_HANDLE)
  {
    LOGE("No compatible GPU\n");
    return VK_ERROR_INITIALIZATION_FAILED;
  }

  LOGI("Using GPU %u:\n", usedGpuIndex);
  vkGetPhysicalDeviceProperties(usedGpu, &properties);
  printPhysicalDeviceProperties(properties);
  return VK_SUCCESS;
}

void nvvk::addSurfaceExtensions(std::vector<const char*>& instanceExtensions, std::vector<nvvk::ExtensionInfo>* deviceExtensions)
{
  instanceExtensions.emplace_back(VK_KHR_SURFACE_EXTENSION_NAME);
  instanceExtensions.emplace_back(VK_KHR_GET_SURFACE_CAPABILITIES_2_EXTENSION_NAME);

#if defined(VK_USE_PLATFORM_WIN32_KHR)
  instanceExtensions.emplace_back(VK_KHR_WIN32_SURFACE_EXTENSION_NAME);
#endif
#if defined(VK_USE_PLATFORM_XCB_KHR)
  instanceExtensions.emplace_back(VK_KHR_XCB_SURFACE_EXTENSION_NAME);
#endif
#if defined(VK_USE_PLATFORM_XLIB_KHR)
  instanceExtensions.emplace_back(VK_KHR_XLIB_SURFACE_EXTENSION_NAME);
#endif
#if defined(VK_USE_PLATFORM_WAYLAND_KHR)
  instanceExtensions.emplace_back(VK_KHR_WAYLAND_SURFACE_EXTENSION_NAME);
#endif
#if defined(VK_USE_PLATFORM_ANDROID_KHR)
  instanceExtensions.emplace_back(VK_KHR_ANDROID_SURFACE_EXTENSION_NAME);
#endif
#if defined(VK_USE_PLATFORM_IOS_MVK)
  instanceExtensions.emplace_back(VK_MVK_IOS_SURFACE_EXTENSION_NAME);
#endif
#if defined(VK_USE_PLATFORM_MACOS_MVK)
  instanceExtensions.emplace_back(VK_MVK_MACOS_SURFACE_EXTENSION_NAME);
#endif

  if(deviceExtensions)
  {
    deviceExtensions->push_back({VK_KHR_SWAPCHAIN_EXTENSION_NAME});
  }
}

//--------------------------------------------------------------------------------------------------
// Usage example
//--------------------------------------------------------------------------------------------------
[[maybe_unused]] static void usage_Context()
{
  // Enable required features for ray tracing
  VkPhysicalDeviceAccelerationStructureFeaturesKHR accelFeature{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR};
  VkPhysicalDeviceRayTracingPipelineFeaturesKHR rtPipelineFeature{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR};

  // Configure Vulkan context initialization
  nvvk::ContextInitInfo vkSetup{.instanceExtensions = {VK_EXT_DEBUG_UTILS_EXTENSION_NAME},
                                .deviceExtensions   = {
                                    {VK_KHR_SWAPCHAIN_EXTENSION_NAME},
                                    {VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME, &accelFeature},
                                    {VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME, &rtPipelineFeature},
                                    {VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME},
                                }};
  nvvk::addSurfaceExtensions(vkSetup.instanceExtensions);

  // Example preSelectPhysicalDeviceCallback, we look for a device with large enough texture dimensions.
  // Providing this callback is optional and can be left out.
  vkSetup.preSelectPhysicalDeviceCallback = [](VkInstance instance, VkPhysicalDevice physicalDevice) {
    VkPhysicalDeviceProperties properties;
    vkGetPhysicalDeviceProperties(physicalDevice, &properties);
    return properties.limits.maxImageDimension2D >= 16384;
  };

  // Example for the postSelectPhysicalDeviceCallback
  // Providing this callback is optional and can be left out.
  vkSetup.postSelectPhysicalDeviceCallback = [](VkInstance instance, VkPhysicalDevice physicalDevice, nvvk::ContextInitInfo& info) {
    VkPhysicalDeviceProperties properties;
    vkGetPhysicalDeviceProperties(physicalDevice, &properties);
    if(properties.vendorID == 0x10DE)
    {
      // Require an additional extension, but only on NVIDIA devices
      info.deviceExtensions.push_back({.extensionName = VK_NV_EXTENDED_SPARSE_ADDRESS_SPACE_EXTENSION_NAME});
    }
    return true;
  };

  // Create and initialize Vulkan context
  nvvk::Context vkContext;
  if(vkContext.init(vkSetup) != VK_SUCCESS)
  {
    LOGE("Error in Vulkan context creation\n");
    return;
  }
}
