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
* SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
* SPDX-License-Identifier: Apache-2.0
*/

#pragma once

#include <vector>

#include <vulkan/vulkan_core.h>

namespace nvvk {
// A helper function to find a supported format from a list of candidates.
// For example, we can use this function to find a supported depth format.
VkFormat findSupportedFormat(VkPhysicalDevice             physicalDevice,
                             const std::vector<VkFormat>& candidates,
                             VkImageTiling                tiling,
                             VkFormatFeatureFlags2        features);

// A helper function to find the depth format that is supported by the physical device.
VkFormat findDepthFormat(VkPhysicalDevice physicalDevice);

// A helper function to find the depth stencil format that is supported by the physical device.
VkFormat findDepthStencilFormat(VkPhysicalDevice physicalDevice);

}  // namespace nvvk
