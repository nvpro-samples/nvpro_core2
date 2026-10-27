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
#include <mutex>
#include <span>
#include <vector>

#include "basic_types.hpp"
#include "resources.hpp"
#include "semaphore.hpp"

namespace nvdx12 {

/*-------------------------------------------------------------------------------------------------
# class nvdx12::QueueTimeline

> A command queue with its monotonically increasing fence, the D3D12
> counterpart of nvvk::QueueTimeline.

`submit(SubmitInfo&, SemaphoreState&)` is nvvk's entry point: the
SubmitInfo carries the command lists, the fences to wait on before them and
to signal after them, and SemaphoreStates (CmdPreSubmitInfo) to wait on or
to complete with the submission; it returns an HRESULT and the state of the
submission. The span overloads below are the single-fence shortcuts the
D3D12 code mostly uses.

`signal` and the span overloads of `submit` return the fence value that
completes them, or 0 when the queue rejects the signal (a lost device);
`waitForValue` / `isComplete` consume those values and `waitIdle` drains
the queue. `createNextFixedSemaphoreState` hands out the value of the next
submission as a SemaphoreState that can be copied and tested later;
`createDynamicSemaphoreState` does the same when other submissions may come
first, and `submit(lists, state)` fills the value in. Listeners learn about
every signal (an upload ring recycling memory by fence value) and every
submission (deferred releases tied to a command list).

The queue is thread-safe: a mutex orders the fence values, the signals and
the listener calls of concurrent `signal` and `submit` calls. Listeners run
with that lock held, so they must not call `signal`, `submit`,
`lastSignaledValue` or `create*SemaphoreState` themselves. A removed
device reports every value as completed (`waitForValue`, `isComplete`), so
that waits end; nvdx12::Context::checkDeviceRemoved tells a loss apart.

Differences with nvvk that follow from D3D12:
- SubmitInfo holds ID3D12CommandList pointers instead of
  VkCommandBufferSubmitInfos, and SemaphoreInfos instead of
  VkSemaphoreSubmitInfos: `submit` lowers waits to ID3D12CommandQueue::Wait
  and signals to ID3D12CommandQueue::Signal around ExecuteCommandLists. No
  pNext / submit flags, device masks, stage masks or VkFence (wait on the
  returned state instead).
- `init` takes the debug name of the fence and returns an HRESULT;
  `getQueue` and `getFence` give the native objects.
- D3D12 extras: `signal`, `waitForValue`, `isComplete`, `completedValue`,
  `lastSignaledValue`, `waitIdle`, `submitAndWait`, the span overloads of
  `submit`, the timestamp frequency and the signal / submit listeners
  (UploadRing, deferred descriptor releases).
- No move constructor or assignment: the upload ring and the listeners
  hold pointers to the timeline, so it stays where it was initialized.

```cpp
nvdx12::QueueTimeline queue;
NVDX12_CHECK(queue.init(device, context.getQueueInfo(0)));
const uint64_t done = queue.submit(cmd);
...
queue.waitForValue(done);
```

See usage_QueueTimeline() in queue.cpp.
-------------------------------------------------------------------------------------------------*/

// Command lists handed around with what their submission must wait on and
// signal, the counterpart of nvvk::CmdPreSubmitInfo.
struct CmdPreSubmitInfo
{
  // The command list that a user would typically record to
  ID3D12CommandList* cmd{};

  // D3D12_COMMAND_LIST_TYPE of the queue the list goes to (QueueInfo::familyIndex)
  uint32_t queueFamilyIndex = 0;

  // contains the SemaphoreState that is triggered when this command list is submitted.
  // By making a copy of it, one can later test if the command list has been completed.
  SemaphoreState semaphoreState;

  // add SemaphoreStates here, that this command list must wait on when being submitted,
  // or are required to be signaled as well.
  std::vector<SemaphoreSubmitState> waitSemaphores;
  std::vector<SemaphoreSubmitState> signalSemaphores;
};

// What QueueTimeline::submit executes, waits on and signals, the
// counterpart of nvvk::SubmitInfo (VkSubmitInfo2). The waits happen before
// the command lists, the signals after them.
class SubmitInfo
{
public:
  std::vector<ID3D12CommandList*> commandLists;

  std::vector<SemaphoreInfo> waitSemaphores;
  std::vector<SemaphoreInfo> signalSemaphores;

  // any submits using SemaphoreStates should be put in these vectors
  // and not lowered into the above
  std::vector<SemaphoreSubmitState> waitSemaphoreStates;
  std::vector<SemaphoreSubmitState> signalSemaphoreStates;

  void clear()
  {
    commandLists.clear();
    waitSemaphores.clear();
    signalSemaphores.clear();
    waitSemaphoreStates.clear();
    signalSemaphoreStates.clear();
  }

  // An empty (default) semaphore state adds no signal.
  void append(const CmdPreSubmitInfo& preSubmit);
  void append(ID3D12CommandList* cmd, SemaphoreState signalSemaphoreState);
  void append(ID3D12CommandList* cmd);
};

class QueueTimeline
{
public:
  QueueTimeline() = default;
  ~QueueTimeline() { assert(!m_fence && "Missing deinit()?"); }
  QueueTimeline(const QueueTimeline&)            = delete;
  QueueTimeline& operator=(const QueueTimeline&) = delete;

  // `debugName` names the fence; the queue keeps the name it was created with.
  HRESULT init(ID3D12Device* device, QueueInfo queueInfo, const char* debugName = "nvdx12::QueueTimeline");
  void    deinit();

  QueueInfo           getQueueInfo() const { return m_queueInfo; }
  ID3D12CommandQueue* getQueue() const { return m_queueInfo.queue.Get(); }
  ID3D12Fence*        getFence() const { return m_fence.Get(); }
  // Ticks per second of the queue's timestamp queries.
  uint64_t getTimestampFrequency() const { return m_timestampFrequency; }

  // ---- Signals and waits ------------------------------------------------------
  // Enqueues a signal and returns its value, or 0 (the fence's initial value,
  // complete at once) when the queue rejects the signal, as a lost device does.
  uint64_t signal();
  // Blocks the CPU until `value` completed. Returns the wait's HRESULT; a
  // removed device reports every value as completed.
  HRESULT  waitForValue(uint64_t value) const;
  bool     isComplete(uint64_t value) const { return completedValue() >= value; }
  uint64_t completedValue() const { return m_fence->GetCompletedValue(); }
  uint64_t lastSignaledValue() const
  {
    std::lock_guard lock(m_mutex);
    return m_value;
  }
  // Signal + wait.
  void waitIdle() { waitForValue(signal()); }

  // ---- Submissions ----------------------------------------------------------
  // Waits on the wait semaphores and states, executes the command lists and
  // signals the signal semaphores and states; `submitState` receives the
  // fixed state of the submission on this queue's fence. A signal state of
  // this queue's fence (dynamic, or fixed to the next value) is completed
  // by the submission. A semaphore without fence or a state without timeline
  // value (a pending dynamic one) fails with E_INVALIDARG before anything is
  // enqueued, and a failing wait stops the call before the lists execute;
  // `submitState` is then empty. Otherwise returns the first failing HRESULT
  // of the signals; when this queue's own signal fails, `submitState` stays
  // empty and a dynamic signal state of this queue's fence gets no value.
  HRESULT submit(SubmitInfo& info, SemaphoreState& submitState);
  // Executes the lists and returns the fence value that completes them, or 0
  // when the signal fails (see `signal`).
  uint64_t submit(std::span<ID3D12CommandList* const> lists);
  uint64_t submit(ID3D12CommandList* list) { return submit(std::span<ID3D12CommandList* const>(&list, 1)); }
  // Same, and stores the value in `submitState`: a dynamic state receives
  // it, a fixed one must already hold it, an empty one becomes fixed; it is
  // left unchanged when the signal fails.
  uint64_t submit(std::span<ID3D12CommandList* const> lists, SemaphoreState& submitState);
  uint64_t submit(ID3D12CommandList* list, SemaphoreState& submitState)
  {
    return submit(std::span<ID3D12CommandList* const>(&list, 1), submitState);
  }
  // Executes and blocks until completion.
  void submitAndWait(ID3D12CommandList* list) { waitForValue(submit(list)); }

  // The fence value of the next `signal` or `submit` as a state that can be
  // stored and tested later; assumes nothing else signals in between.
  SemaphoreState createNextFixedSemaphoreState() const
  {
    std::lock_guard lock(m_mutex);
    return SemaphoreState::makeFixed(m_fence, m_value + 1);
  }

#ifndef NVDX12_DISABLE_DYNAMIC_SEMAPHORE_STATE
  // A state for a submission whose order is not known yet; copies can be
  // stored right away and `submit(lists, state)` sets the value.
  SemaphoreState createDynamicSemaphoreState() const { return SemaphoreState::makeDynamic(m_fence); }
#endif

  // ---- Listeners ------------------------------------------------------------
  // Called after every signal with the new fence value, and after every
  // submission with the lists and the value that completes them, under the
  // timeline's lock: a listener must not call back into the timeline. The
  // handle removes the listener again (an object that outlives its interest).
  using ListenerHandle = uint32_t;
  using SignalListener = std::function<void(uint64_t fenceValue)>;
  using SubmitListener = std::function<void(std::span<ID3D12CommandList* const> lists, uint64_t fenceValue)>;
  ListenerHandle addSignalListener(SignalListener listener);
  ListenerHandle addSubmitListener(SubmitListener listener);
  void           removeSignalListener(ListenerHandle handle);
  void           removeSubmitListener(ListenerHandle handle);

private:
  // The caller holds m_mutex.
  HRESULT signalLocked(uint64_t& value);
  HRESULT submitLocked(std::span<ID3D12CommandList* const> lists, uint64_t& value);

  QueueInfo                                              m_queueInfo;
  ComPtr<ID3D12Fence>                                    m_fence;
  uint64_t                                               m_value{0};  // last signaled value
  uint64_t                                               m_timestampFrequency{0};
  std::vector<std::pair<ListenerHandle, SignalListener>> m_signalListeners;
  std::vector<std::pair<ListenerHandle, SubmitListener>> m_submitListeners;
  ListenerHandle                                         m_nextListenerHandle{1};
  mutable std::mutex m_mutex;  // m_value, the listeners and the order of their calls
};

}  // namespace nvdx12
