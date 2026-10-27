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

#include "upload_ring.hpp"

#include <algorithm>
#include <cstring>

#include <nvutils/logger.hpp>

#include "check_error.hpp"
#include "debug_util.hpp"

namespace nvdx12 {

HRESULT UploadRing::init(QueueTimeline& queue, uint64_t bytes)
{
  assert(!m_resource && "init() called twice; missing deinit()?");
  if(queue.getQueue() == nullptr)
  {
    return E_INVALIDARG;
  }
  ComPtr<ID3D12Device10> device;
  NVDX12_FAIL_RETURN(queue.getQueue()->GetDevice(IID_PPV_ARGS(&device)));
  // A multiple of 64 KiB, so that offsets (monotonic positions modulo the
  // size) keep the alignment of the positions for every power-of-two
  // alignment up to that.
  constexpr uint64_t kSizeGranularity = D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT;
  const uint64_t     size = (std::max<uint64_t>(bytes, MIN_SIZE) + kSizeGranularity - 1) & ~(kSizeGranularity - 1);

  // built in locals, so that a failure leaves the ring uninitialized
  D3D12_HEAP_PROPERTIES heap{};
  heap.Type = D3D12_HEAP_TYPE_UPLOAD;
  D3D12_RESOURCE_DESC1 desc{};
  desc.Dimension        = D3D12_RESOURCE_DIMENSION_BUFFER;
  desc.Width            = size;
  desc.Height           = 1;
  desc.DepthOrArraySize = 1;
  desc.MipLevels        = 1;
  desc.Format           = DXGI_FORMAT_UNKNOWN;
  desc.SampleDesc.Count = 1;
  desc.Layout           = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
  ComPtr<ID3D12Resource> resource;
  NVDX12_FAIL_RETURN(device->CreateCommittedResource3(&heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_BARRIER_LAYOUT_UNDEFINED,
                                                      nullptr, nullptr, 0, nullptr, IID_PPV_ARGS(&resource)));
  setDebugName(resource.Get(), "nvdx12::UploadRing");
  D3D12_RANGE noRead{0, 0};
  uint8_t*    mapping = nullptr;
  NVDX12_FAIL_RETURN(resource->Map(0, &noRead, reinterpret_cast<void**>(&mapping)));
  m_resource = std::move(resource);
  m_mapping  = mapping;
  m_size     = size;
  m_queue    = &queue;
  m_head = m_tail = 0;
  m_deferredHead  = 0;
  m_deferredList  = nullptr;
  m_retires.clear();
  m_fullLogged     = false;
  m_signalListener = m_queue->addSignalListener([this](uint64_t fenceValue) { markSubmitted(fenceValue); });
  m_submitListener = m_queue->addSubmitListener(
      [this](std::span<ID3D12CommandList* const> lists, uint64_t fenceValue) { markListSubmitted(lists, fenceValue); });
  return S_OK;
}

void UploadRing::deinit()
{
  if(m_queue != nullptr && m_signalListener != 0)
  {
    m_queue->removeSignalListener(m_signalListener);
    m_queue->removeSubmitListener(m_submitListener);
  }
  m_queue          = nullptr;
  m_signalListener = 0;
  m_submitListener = 0;
  if(m_resource && m_mapping)
  {
    m_resource->Unmap(0, nullptr);
  }
  m_mapping = nullptr;
  m_resource.Reset();
  m_retires.clear();
  m_head = m_tail = 0;
  m_deferredHead  = 0;
  m_deferredList  = nullptr;
  m_size          = 0;
}

UploadSpan UploadRing::allocate(uint64_t size, uint64_t alignment)
{
  UploadSpan span{};
  if(size == 0)
  {
    return span;
  }
  // the ring size is a multiple of 64 KiB, so offsets keep alignments up to that
  assert((alignment & (alignment - 1)) == 0 && alignment <= D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT);
  alignment = std::max<uint64_t>(alignment, 4);
  size      = (size + alignment - 1) & ~(alignment - 1);
  if(size > m_size)
  {
    LOGE("UploadRing: request of %llu bytes exceeds the ring size (%llu)\n", static_cast<unsigned long long>(size),
         static_cast<unsigned long long>(m_size));
    return span;
  }

  for(;;)
  {
    uint64_t waitValue = 0;
    {
      std::lock_guard lock(m_mutex);
      // Positions are monotonic; the offset wraps modulo the size, and a
      // request that does not fit before the end skips to the start.
      uint64_t head   = (m_head + alignment - 1) & ~(alignment - 1);
      uint64_t offset = head % m_size;
      if(offset + size > m_size)
      {
        head += m_size - offset;
        offset = 0;
      }
      // With nothing in flight, the skipped bytes and the padding are free too.
      if(m_head == m_tail)
      {
        m_tail = head;
      }
      if(head + size - m_tail <= m_size)
      {
        m_head        = head + size;
        span.cpu      = m_mapping + offset;
        span.gpu      = m_resource->GetGPUVirtualAddress() + offset;
        span.resource = m_resource.Get();
        span.offset   = offset;
        span.size     = size;
        return span;
      }
      // Full: retire what the GPU already finished and retry.
      if(retireLocked(m_queue->completedValue()))
      {
        continue;
      }
      // Only signaled allocations can be waited for. The ones made since the
      // last signal (or deferred) belong to command lists still being
      // recorded: a signal now would retire them before their lists execute.
      if(m_retires.empty())
      {
        LOGE("UploadRing: %llu bytes do not fit; the allocations since the last signal fill the ring of %llu bytes\n",
             static_cast<unsigned long long>(size), static_cast<unsigned long long>(m_size));
        return span;
      }
      waitValue = m_retires.front().fenceValue;
    }
    if(!m_fullLogged.exchange(true))
    {
      LOGW("UploadRing of %llu bytes is full; waiting for the GPU (increase the ring size)\n",
           static_cast<unsigned long long>(m_size));
    }
    m_queue->waitForValue(waitValue);
  }
}

void UploadRing::markSubmitted(uint64_t fenceValue)
{
  // Called under the timeline's lock: the completed value is read from the
  // fence, never from the timeline.
  const uint64_t  completed = m_queue->completedValue();
  std::lock_guard lock(m_mutex);
  m_retires.push_back({fenceValue, m_deferredList ? m_deferredHead : m_head});
  retireLocked(completed);
}

void UploadRing::retire()
{
  const uint64_t  completed = m_queue->completedValue();
  std::lock_guard lock(m_mutex);
  retireLocked(completed);
}

bool UploadRing::retireLocked(uint64_t completedValue)
{
  bool progressed = false;
  while(!m_retires.empty() && m_retires.front().fenceValue <= completedValue)
  {
    // An empty ring may have moved the tail past older marks (allocate).
    m_tail = std::max(m_tail, m_retires.front().head);
    m_retires.pop_front();
    progressed = true;
  }
  return progressed;
}

void UploadRing::markListSubmitted(std::span<ID3D12CommandList* const> lists, uint64_t fenceValue)
{
  std::lock_guard lock(m_mutex);
  if(m_deferredList == nullptr || std::find(lists.begin(), lists.end(), m_deferredList) == lists.end())
  {
    return;
  }
  // markSubmitted marked up to m_deferredHead with this value already; the
  // second entry keeps the positions and fence values in order.
  m_retires.push_back({fenceValue, m_head});
  m_deferredList = nullptr;
}

void UploadRing::beginDeferral(ID3D12CommandList* list)
{
  std::lock_guard lock(m_mutex);
  assert(list && !m_deferredList && "beginDeferral() needs a list and does not nest");
  m_deferredList = list;
  m_deferredHead = m_head;
}

void UploadRing::cancelDeferral()
{
  std::lock_guard lock(m_mutex);
  m_deferredList = nullptr;
}

bool UploadRing::cmdPushData(ID3D12GraphicsCommandList* cmd, const void* data, uint32_t size, BindPoint bindPoint, uint32_t rootParameterIndex)
{
  UploadSpan span = allocate(size, D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT);
  if(!span)
  {
    return false;
  }
  memcpy(span.cpu, data, size);
  if(bindPoint != BindPoint::eGraphics)
  {
    cmd->SetComputeRootShaderResourceView(rootParameterIndex, span.gpu);
  }
  if(bindPoint != BindPoint::eCompute)
  {
    cmd->SetGraphicsRootShaderResourceView(rootParameterIndex, span.gpu);
  }
  return true;
}

}  // namespace nvdx12

//--------------------------------------------------------------------------------------------------
// Usage example
//--------------------------------------------------------------------------------------------------
[[maybe_unused]] static void usage_UploadRing()
{
  nvdx12::QueueTimeline* queue = nullptr;  // EX: the timeline of the queue the lists go to

  // The ring follows the queue's signals from here on: no fence bookkeeping.
  nvdx12::UploadRing ring;
  NVDX12_CHECK(ring.init(*queue, 64ull << 20));

  /* while(!glfwWindowShouldClose()) */
  {
    ID3D12GraphicsCommandList* cmd = nullptr;  // EX: this frame's command list

    // Push data beyond the root constants: lands in the ring, root parameter
    // kRootParamPushData points at it (PUSH_DATA(T, name) in the shader).
    // False when the ring has no room: nothing is bound, skip the dispatch.
    struct PushData
    {
      float    matrices[16 * 4];
      uint32_t width, height;
    } push{};
    if(ring.cmdPushData(cmd, &push, sizeof(push), nvdx12::BindPoint::eCompute))
    {
      // cmd->Dispatch(...)
    }

    // Any other transient upload: fill the span, then copy from span.resource.
    nvdx12::UploadSpan span = ring.allocate(4096);
    if(span)
    {
      memset(span.cpu, 0, span.size);
      // cmd->CopyBufferRegion(dst, dstOffset, span.resource, span.offset, span.size);
    }

    // queue->submit(cmd): the signal marks this frame's allocations and frees the completed ones
  }

  // A list recorded while other lists are submitted on the queue: its
  // allocations stay in flight until the submission that executes it.
  {
    ID3D12GraphicsCommandList* frameCmd = nullptr;  // EX: the frame's list
    ring.beginDeferral(frameCmd);
    // EX: record into frameCmd with ring.cmdPushData, submit and wait for a temporary list
    // queue->submit(frameCmd): marks the deferred allocations and ends the deferral;
    // ring.cancelDeferral() instead when frameCmd fails to close and is not submitted
  }

  ring.deinit();  // before the queue's deinit
}
