/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cstdint>
#include <vector>

#include "basic_types.hpp"
#include "semaphore.hpp"

namespace nvdx12 {

/*-------------------------------------------------------------------------------------------------
# class nvdx12::ManagedCommandPools

> Single-submit command lists, each on a dedicated command allocator, the
> D3D12 counterpart of nvvk::ManagedCommandPools.

The command lists and their allocators are managed in one of two modes:

- `Mode::SEMAPHORE_STATE`: every acquisition takes the SemaphoreState of the
  submission the list goes to (a fence value of a QueueTimeline, fixed or
  dynamic). An allocator is reset and reused only once that state is
  signaled, which is the safer mode.
- `Mode::EXPLICIT_INDEX`: the caller tracks completion and names the
  allocator by index, as in a classic ring of frames in flight.

Differences with nvvk that follow from D3D12:
- A VkCommandPool is an ID3D12CommandAllocator and a VkCommandBuffer an
  ID3D12GraphicsCommandList7. `queueFamilyIndex` holds the
  D3D12_COMMAND_LIST_TYPE of the lists (QueueInfo::familyIndex); `flags` are
  the D3D12_COMMAND_LIST_FLAGS of ID3D12Device4::CreateCommandList1 (there
  are no allocator creation flags; Vulkan's TRANSIENT has no equivalent).
- The returned list is already recording (D3D12 has no begin call): record,
  `Close()` it and submit it. It stays owned by the pools.
- A D3D12 list cannot be freed back to its allocator: each managed allocator
  keeps its list object, and recycling resets the allocator and then the
  list onto it. A list being recycled must have been closed, which a
  submitted list always is. `releaseCommandBuffer` (the abort path) releases
  the list object itself, so a list that is still recording can be dropped;
  the next acquisition of that allocator creates a new one.
- No command buffer level: secondary command buffers are D3D12 bundles,
  which need bundle allocators and lists; use a separate instance with
  D3D12_COMMAND_LIST_TYPE_BUNDLE. No VkCommandPoolResetFlags either:
  ID3D12CommandAllocator::Reset takes none.
- Results are HRESULTs. In `acquireCommandBuffer`, a wait for the oldest
  allocator that times out returns WAIT_TIMEOUT (a success code, so test
  `!= S_OK`) and hands out no list; `waitTimeOut` is in nanoseconds as in
  SemaphoreState::wait. `releaseCommandBuffer` returns E_INVALIDARG for a
  list that does not come from here.

```cpp
nvdx12::ManagedCommandPools pools;
pools.init(device, D3D12_COMMAND_LIST_TYPE_DIRECT, nvdx12::ManagedCommandPools::Mode::SEMAPHORE_STATE);
nvdx12::SemaphoreState state = queue.createNextFixedSemaphoreState();
ID3D12GraphicsCommandList7* cmd{};
pools.acquireCommandBuffer(state, cmd);
... record, cmd->Close() ...
queue.submit(cmd, state);
...
pools.deinit();
```

See usage_ManagedCommandPools() in command_pools.cpp.
-------------------------------------------------------------------------------------------------*/

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
  // `queueFamilyIndex` is the D3D12_COMMAND_LIST_TYPE of the lists.
  // all command lists are meant for single submit. use `releaseCommandBuffer` otherwise.
  // In `Mode::EXPLICIT_INDEX` `maxPoolCount` many command allocators are created immediately.
  // otherwise they are lazily created depending on how many are in-flight.
  HRESULT init(ID3D12Device*            device,
               uint32_t                 queueFamilyIndex,
               Mode                     mode,
               D3D12_COMMAND_LIST_FLAGS flags        = D3D12_COMMAND_LIST_FLAG_NONE,
               uint32_t                 maxPoolCount = 8);

  // frees all command lists and command allocators independent of SemaphoreState;
  // none of them may still be executing.
  void deinit();

  // The returned command list is recording and must be used only with a single submit that is provided through
  // `submitSemaphoreState`. Internally recycles the allocator of the first completed list it finds.
  // If nothing can be recycled we will grow the pool up to `maxPoolCount` and if that isn't enough we will wait for
  // the completion of the oldest command list based on waitTimeOut (nanoseconds): WAIT_TIMEOUT is returned when
  // the wait timed out, and `cmd` is left untouched.
  // Only legal for `Mode::SEMAPHORE_STATE`
  HRESULT acquireCommandBuffer(const SemaphoreState& submitSemaphoreState, ID3D12GraphicsCommandList7*& cmd, uint64_t waitTimeOut = ~0ULL);

  // The returned command list is recording and must be used only with a single submit.
  // Internally runs releaseIndexed(explicitIndex) on the allocator with the given `explicitIndex`.
  // The `explicitIndex` must be smaller than `maxPoolCount`.
  // Only legal for `Mode::EXPLICIT_INDEX`
  HRESULT acquireCommandBuffer(uint32_t explicitIndex, ID3D12GraphicsCommandList7*& cmd);

  // User must ensure the command list originated here and was completed, or was never submitted.
  // Not required in typical use, mostly meant for aborting the use of a command list.
  // The command list is released (it may still be recording) and its allocator is reset.
  // Returns `E_INVALIDARG` if the command list wasn't found in the pool.
  HRESULT releaseCommandBuffer(ID3D12GraphicsCommandList7* cmd);

  // Can be called to reset the allocators of all completed command lists based on SemaphoreState
  // manually (typically not needed).
  // Only legal for `Mode::SEMAPHORE_STATE`
  HRESULT releaseCompleted();

  // Resets the allocator for this index if its list is in use; the list must be closed and completed.
  // Only legal for `Mode::EXPLICIT_INDEX`
  HRESULT releaseIndexed(uint32_t explicitIndex);

  // incremented each time we had to wait for completing an old command list that is in-flight
  // if this number increases too quickly consider increasing the `maxPoolCount`
  uint32_t getWaitCount() const { return m_waitCount; }

protected:
  struct ManagedCommandPool
  {
    ComPtr<ID3D12CommandAllocator> commandPool{};
    SemaphoreState                 semaphoreState{};
    // kept across recycles; null until first use or after releaseCommandBuffer
    ComPtr<ID3D12GraphicsCommandList7> cmd{};
    // the list is handed out and its allocator not yet reset (nvvk: cmd != null)
    bool     inUse{false};
    uint64_t acquisitionIndex{};
  };

  ComPtr<ID3D12Device4>           m_device{};
  uint32_t                        m_queueFamilyIndex{};
  D3D12_COMMAND_LIST_FLAGS        m_flags{};
  uint32_t                        m_maxPoolCount{};
  uint32_t                        m_waitCount{};
  std::vector<ManagedCommandPool> m_managedPools;
  uint64_t                        m_acquisitionCounter{};
  Mode                            m_mode{};

  HRESULT createCommandPool(ManagedCommandPool& entry);
  HRESULT getManagedCommandBuffer(ManagedCommandPool& entry, const SemaphoreState& submitSemaphoreState, ID3D12GraphicsCommandList7*& cmd);
  HRESULT reset(ManagedCommandPool& entry);
};

}  // namespace nvdx12
