/*
* Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
* SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION.
* SPDX-License-Identifier: Apache-2.0
*/

#include <cassert>

#include <volk.h>

#include "physical_device.hpp"

namespace nvvk {

void PhysicalDeviceInfo::init(VkPhysicalDevice physicalDevice, uint32_t apiVersion)
{
  assert(apiVersion >= VK_API_VERSION_1_2);

  VkPhysicalDeviceProperties2 props = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
  props.pNext                       = &properties11;
  properties11.pNext                = &properties12;
  if(apiVersion >= VK_API_VERSION_1_3)
  {
    properties12.pNext = &properties13;
  }
  if(apiVersion >= VK_API_VERSION_1_4)
  {
    properties13.pNext = &properties14;
  }
  vkGetPhysicalDeviceProperties2(physicalDevice, &props);
  properties10 = props.properties;

  VkPhysicalDeviceFeatures2 features = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
  features.pNext                     = &features11;
  features11.pNext                   = &features12;
  if(apiVersion >= VK_API_VERSION_1_3)
  {
    features12.pNext = &features13;
  }
  if(apiVersion >= VK_API_VERSION_1_4)
  {
    features13.pNext = &features14;
  }
  vkGetPhysicalDeviceFeatures2(physicalDevice, &features);
  features10 = features.features;
}
}  // namespace nvvk