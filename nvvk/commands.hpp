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

#include <vulkan/vulkan_core.h>

#include "check_error.hpp"
#include "resources.hpp"

namespace nvvk {

// Helper to create a transient command pool
// Transient command pools are meant to be used for short-lived commands.
VkCommandPool createTransientCommandPool(VkDevice device, uint32_t queueFamilyIndex);

// Simple helper for the creation of a temporary command buffer, use to record the commands to upload data, or transition images.
// Submit the temporary command buffer, wait until the command is finished, and clean up.
// Allocates a single shot command buffer from the provided pool and begins it.
VkResult beginSingleTimeCommands(VkCommandBuffer& cmd, VkDevice device, VkCommandPool cmdPool);

inline VkCommandBuffer createSingleTimeCommands(VkDevice device, VkCommandPool cmdPool)
{
  VkCommandBuffer cmd{};
  NVVK_CHECK(beginSingleTimeCommands(cmd, device, cmdPool));
  return cmd;
}

// Ends command buffer, submits on the provided queue, then waits for completion, and frees the command buffer within the provided pool
VkResult endSingleTimeCommands(VkCommandBuffer cmd, VkDevice device, VkCommandPool cmdPool, VkQueue queue);

}  // namespace nvvk
