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

#include "semaphore.hpp"

namespace nvvk {

// This class is meant for single commit primary command buffers
// that each use a dedicated VkCommandPool. The command buffers
// and pools can be managed through two distinct modes:
//
// In `Mode::SEMAPHORE_STATE` they are recycled depending on the
// SemaphoreState which is safer
// In `Mode::EXPLICIT_INDEX` the user is responsible for tracking
// completion and provides and explicit pool index.

class ManagedCommandPools
{
public:
  enum class Mode
  {
    SEMAPHORE_STATE,
    EXPLICIT_INDEX,
  };

  ManagedCommandPools()                                      = default;
  ManagedCommandPools(const ManagedCommandPools&)            = delete;
  ManagedCommandPools& operator=(const ManagedCommandPools&) = delete;
  ManagedCommandPools(ManagedCommandPools&& other) noexcept;
  ManagedCommandPools& operator=(ManagedCommandPools&& other) noexcept;
  ~ManagedCommandPools();

  // initializes
  // `flags` must not contain VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT
  // all command buffers are meant for single submit. use `releaseCommandBuffer` otherwise.
  // In `Mode::EXPLICIT_INDEX` `maxPoolCount` many command pools are created immediately.
  // otherwise they are lazily created depending on how many are in-flight.
  VkResult init(VkDevice                 device,
                uint32_t                 queueFamilyIndex,
                Mode                     mode,
                VkCommandPoolCreateFlags flags        = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,
                uint32_t                 maxPoolCount = 8);

  // frees all command buffers and command pools independent of SemaphoreState
  void deinit();

  // The returned command buffer must be used only with a single submit that is provided through `submitSemaphoreState`.
  // Internally runs releaseCommandBuffer(cmd, 0) on the first completed command buffer it finds and then reuses its command pool.
  // If nothing can be recycled we will grow the pool up to `maxPoolCount` and if that isn't enough we will wait for
  // the completion of the oldest command buffer based on waitTimeOut.
  // Only legal for `Mode::SEMAPHORE_STATE`
  VkResult acquireCommandBuffer(const nvvk::SemaphoreState& submitSemaphoreState,
                                VkCommandBuffer&            cmd,
                                VkCommandBufferLevel        level       = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                                uint64_t                    waitTimeOut = ~0ULL);

  // The returned command buffer must be used only with a single submit.
  // Internally runs releaseIndexed(explicitIndex, 0) on the pool with the given `explicitIndex`.
  // The `explicitIndex` must be smaller than `maxPoolCount`.
  // Only legal for `Mode::EXPLICIT_INDEX`
  VkResult acquireCommandBuffer(uint32_t explicitIndex, VkCommandBuffer& cmd, VkCommandBufferLevel level = VK_COMMAND_BUFFER_LEVEL_PRIMARY);

  // User must ensure the command buffer originated here and was completed, or was never submitted.
  // Not required in typical use, mostly meant for aborting the use of a command buffer.
  // The command buffer is freed and its pool is reset using the provided flags.
  // Returns `VK_ERROR_UNKNOWN` if the command buffer wasn't found in the pool.
  VkResult releaseCommandBuffer(VkCommandBuffer cmd, VkCommandPoolResetFlags resetFlags = 0);

  // Can be called to free all completed command buffers & reset their command pools based on SemaphoreState
  // manually (typically not needed).
  // Only legal for `Mode::SEMAPHORE_STATE`
  VkResult releaseCompleted(VkCommandPoolResetFlags resetFlags = 0);

  // Free's the command buffer for this index (if it exists) and resets the corresponding command pool.
  // Only legal for `Mode::SEMAPHORE_STATE`
  VkResult releaseIndexed(uint32_t explicitIndex, VkCommandPoolResetFlags resetFlags = 0);

protected:
  struct ManagedCommandPool
  {
    VkCommandPool        commandPool{};
    nvvk::SemaphoreState semaphoreState{};
    VkCommandBuffer      cmd{};
    uint64_t             acquisitionIndex{};
  };

  VkDevice                        m_device{};
  uint32_t                        m_queueFamilyIndex{};
  uint32_t                        m_flags{};
  uint32_t                        m_maxPoolCount{};
  std::vector<ManagedCommandPool> m_managedPools;
  uint64_t                        m_acquisitionCounter{};
  Mode                            m_mode{};

  VkResult getManagedCommandBuffer(ManagedCommandPool&         entry,
                                   VkCommandBufferLevel        level,
                                   const nvvk::SemaphoreState& submitSemaphoreState,
                                   VkCommandBuffer&            cmd);
  VkResult reset(ManagedCommandPool& entry, VkCommandPoolResetFlags resetFlags);
};
}  // namespace nvvk
