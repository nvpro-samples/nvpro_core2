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

#include <cstdint>
#include <functional>

#include "descriptor_heap.hpp"
#include "resources.hpp"
#include "upload_ring.hpp"

namespace nvdx12 {

/*-------------------------------------------------------------------------------------------------
# Command recording helpers

> The D3D12 counterparts of nvvk/commands.hpp and of the vkCmd* calls a
> compute and ray-tracing renderer records.

- Copies: `cmdCopyBuffer`, `cmdCopyImage`, `cmdCopyImageRegion`,
  `cmdCopyImageToBuffer`, `cmdCopyBufferToImage`. Image copies transition
  the images to the copy layouts and leave them there.
- `cmdDispatchIndirect` and `cmdUpdateBuffer` (inline update through an
  UploadRing).
- Single-time command lists as in nvvk: `createTransientCommandPool`,
  `beginSingleTimeCommands` / `createSingleTimeCommands` and
  `endSingleTimeCommands` on a caller-owned allocator and queue.
- D3D12 extras: `CommandContext`, an allocator + list pair, one per frame in
  flight or per subsystem recording outside the frame loop, and
  `ScopeCommandList`, a convenience
  without nvvk counterpart that records on its own allocator and submits
  through a callback when it goes out of scope.

Differences with nvvk that follow from D3D12:
- A VkCommandPool is an ID3D12CommandAllocator, and lists are created for
  a D3D12_COMMAND_LIST_TYPE (the queue family): the single-time helpers take
  the type where nvvk takes a queue family index or nothing.
- A D3D12 list cannot be freed back to its allocator: endSingleTimeCommands
  resets the allocator instead (after the wait), so the allocator must not
  back other lists still recording or executing. The list itself is
  released by the caller (or reused with Reset).

Usage:
  see usage_Commands() in commands.cpp
-------------------------------------------------------------------------------------------------*/

//-----------------------------------------------------------------------------
// Dispatch and buffers

// ExecuteIndirect of one D3D12_DISPATCH_ARGUMENTS at `byteOffset` in `args`
// (vkCmdDispatchIndirect). `signature` is a one-argument DISPATCH command
// signature (createDispatchIndirectSignature in compute_pipeline.hpp).
void cmdDispatchIndirect(ID3D12GraphicsCommandList* cmd, ID3D12CommandSignature* signature, const Buffer& args, uint64_t byteOffset);

// Inline update through the upload ring + CopyBufferRegion (vkCmdUpdateBuffer).
// Returns false, recording nothing, when the ring has no room or `dst` is null.
bool cmdUpdateBuffer(ID3D12GraphicsCommandList* cmd, UploadRing& ring, const Buffer& dst, uint64_t dstOffset, uint64_t size, const void* data);

// CopyBufferRegion of `size` bytes (vkCmdCopyBuffer).
void cmdCopyBuffer(ID3D12GraphicsCommandList* cmd, const Buffer& src, uint64_t srcOffset, const Buffer& dst, uint64_t dstOffset, uint64_t size);
// Whole source buffer to the start of `dst`.
inline void cmdCopyBuffer(ID3D12GraphicsCommandList* cmd, const Buffer& src, const Buffer& dst)
{
  cmdCopyBuffer(cmd, src, 0, dst, 0, src.bufferSize);
}

//-----------------------------------------------------------------------------
// Images

// Copies every subresource `src` and `dst` have in common; a whole-resource
// copy when their descriptions match. Depth-stencil and multisampled images
// copy whole subresources, so their destination levels must hold the source's
// with the same sample count (otherwise nothing is copied and an error is
// logged); without matching descriptions, a depth-stencil format copies its
// depth plane only.
void cmdCopyImage(ID3D12GraphicsCommandList7* cmd, Image& src, Image& dst);

// One mip level, sub-rectangle variant (vkCmdCopyImage with regions);
// `srcBox` null copies the whole level. Depth-stencil and multisampled images
// take a null `srcBox`, a zero destination offset and the same sample count on
// both sides (otherwise nothing is copied and an error is logged).
void cmdCopyImageRegion(ID3D12GraphicsCommandList7* cmd,
                        Image&                      src,
                        uint32_t                    srcMip,
                        Image&                      dst,
                        uint32_t                    dstMip,
                        const D3D12_BOX*            srcBox,
                        uint32_t                    dstX,
                        uint32_t                    dstY,
                        uint32_t                    dstZ);

// Copies one mip level into `dst` at `dstOffset` (512-byte aligned) with the
// 256-byte aligned row pitch of GetCopyableFootprints (vkCmdCopyImageToBuffer;
// imageSubresourceFootprint in formats.hpp reports the layout), and the
// inverse (vkCmdCopyBufferToImage).
void cmdCopyImageToBuffer(ID3D12GraphicsCommandList7* cmd,
                          Image&                      src,
                          const Buffer&               dst,
                          uint64_t                    dstOffset,
                          uint32_t                    mipLevel   = 0,
                          uint32_t                    arrayLayer = 0);
void cmdCopyBufferToImage(ID3D12GraphicsCommandList7* cmd,
                          const Buffer&               src,
                          uint64_t                    srcOffset,
                          Image&                      dst,
                          uint32_t                    mipLevel   = 0,
                          uint32_t                    arrayLayer = 0);

//-----------------------------------------------------------------------------
// Command contexts

// Allocator + list pair, one per frame in flight or per subsystem that
// records outside the frame loop. The list is created closed; `begin`
// resets both for a new recording.
struct CommandContext
{
  ComPtr<ID3D12CommandAllocator>     allocator;
  ComPtr<ID3D12GraphicsCommandList7> list;

  HRESULT init(ID3D12Device* device, const char* name, D3D12_COMMAND_LIST_TYPE type = D3D12_COMMAND_LIST_TYPE_DIRECT);
  void    deinit();
  // Resets the allocator and the list for a new recording; the previous
  // recording must have completed on the GPU.
  HRESULT begin(ID3D12PipelineState* initialState = nullptr);
  HRESULT close();
};

// Allocator for short-lived lists of `type`, the counterpart of
// nvvk::createTransientCommandPool. Null on failure (reported).
ComPtr<ID3D12CommandAllocator> createTransientCommandPool(ID3D12Device* device, D3D12_COMMAND_LIST_TYPE type);

// Opens a new command list on `allocator` (which must not be in use by a
// list that is still executing), the counterpart of
// nvvk::beginSingleTimeCommands.
HRESULT beginSingleTimeCommands(ComPtr<ID3D12GraphicsCommandList7>& cmd,
                                ID3D12Device*                       device,
                                ID3D12CommandAllocator*             allocator,
                                D3D12_COMMAND_LIST_TYPE             type = D3D12_COMMAND_LIST_TYPE_DIRECT);

// Same, returning the list and checking the result (NVDX12_CHECK), the
// counterpart of nvvk::createSingleTimeCommands.
ComPtr<ID3D12GraphicsCommandList7> createSingleTimeCommands(ID3D12Device*           device,
                                                            ID3D12CommandAllocator* allocator,
                                                            D3D12_COMMAND_LIST_TYPE type = D3D12_COMMAND_LIST_TYPE_DIRECT);

// Closes `cmd`, executes it on `queue`, blocks until it completed and resets
// `allocator`, the counterpart of nvvk::endSingleTimeCommands (which frees
// the command buffer back to its pool). The list is left for the caller to
// release.
HRESULT endSingleTimeCommands(ID3D12GraphicsCommandList* cmd, ID3D12Device* device, ID3D12CommandAllocator* allocator, ID3D12CommandQueue* queue);

// Single-time command list on its own allocator, a D3D12 convenience (nvvk
// has no counterpart): the destructor closes the list and hands it to
// `submitAndWait`, which executes it and blocks until it completed. A list
// that fails to close is reported and not submitted.
class ScopeCommandList
{
public:
  using SubmitAndWait = std::function<void(ID3D12CommandList*)>;

  // `heap`, when given, is bound at the top of the list.
  ScopeCommandList(ID3D12Device* device, SubmitAndWait submitAndWait, const DescriptorHeap* heap = nullptr);
  ~ScopeCommandList();
  ScopeCommandList(const ScopeCommandList&)            = delete;
  ScopeCommandList& operator=(const ScopeCommandList&) = delete;

  // Null when the list could not be created.
  operator ID3D12GraphicsCommandList7*() const { return m_cmd.Get(); }
  ID3D12GraphicsCommandList7* get() const { return m_cmd.Get(); }

private:
  SubmitAndWait                      m_submitAndWait;
  ComPtr<ID3D12CommandAllocator>     m_allocator;
  ComPtr<ID3D12GraphicsCommandList7> m_cmd;
};

}  // namespace nvdx12
