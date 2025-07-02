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

#include <array>

#include "commands.hpp"
#include "debug_util.hpp"


namespace nvvk {

VkCommandPool createTransientCommandPool(VkDevice device, uint32_t queueFamilyIndex)
{
  VkCommandPool                 cmdPool = VK_NULL_HANDLE;
  const VkCommandPoolCreateInfo commandPoolCreateInfo{
      .sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .flags            = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,  // Hint that commands will be short-lived
      .queueFamilyIndex = queueFamilyIndex,
  };
  NVVK_CHECK(vkCreateCommandPool(device, &commandPoolCreateInfo, nullptr, &cmdPool));
  NVVK_DBG_NAME(cmdPool);
  return cmdPool;
}


VkResult beginSingleTimeCommands(VkCommandBuffer& cmd, VkDevice device, VkCommandPool cmdPool)
{
  const VkCommandBufferAllocateInfo allocInfo{.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                                              .commandPool        = cmdPool,
                                              .level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                                              .commandBufferCount = 1};
  NVVK_FAIL_RETURN(vkAllocateCommandBuffers(device, &allocInfo, &cmd));
  const VkCommandBufferBeginInfo beginInfo{.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                                           .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
  NVVK_FAIL_RETURN(vkBeginCommandBuffer(cmd, &beginInfo));
  return VK_SUCCESS;
}

VkResult endSingleTimeCommands(VkCommandBuffer cmd, VkDevice device, VkCommandPool cmdPool, VkQueue queue)
{
  // Submit and clean up
  NVVK_FAIL_RETURN(vkEndCommandBuffer(cmd));

  // Create fence for synchronization
  const VkFenceCreateInfo fenceInfo{.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
  std::array<VkFence, 1>  fence{};
  NVVK_FAIL_RETURN(vkCreateFence(device, &fenceInfo, nullptr, fence.data()));

  const VkCommandBufferSubmitInfo cmdBufferInfo{.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO, .commandBuffer = cmd};
  const std::array<VkSubmitInfo2, 1> submitInfo{
      {{.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2, .commandBufferInfoCount = 1, .pCommandBufferInfos = &cmdBufferInfo}}};
  NVVK_FAIL_RETURN(vkQueueSubmit2(queue, uint32_t(submitInfo.size()), submitInfo.data(), fence[0]));
  NVVK_FAIL_RETURN(vkWaitForFences(device, uint32_t(fence.size()), fence.data(), VK_TRUE, UINT64_MAX));

  // Cleanup
  vkDestroyFence(device, fence[0], nullptr);
  vkFreeCommandBuffers(device, cmdPool, 1, &cmd);
  return VK_SUCCESS;
}

}  // namespace nvvk
