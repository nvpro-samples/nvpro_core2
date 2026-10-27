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
#include <vector>

#include "descriptor_heap.hpp"
#include "resources.hpp"
#include "upload_ring.hpp"

namespace nvdx12 {

class ShaderCompiler;

/*-------------------------------------------------------------------------------------------------
# class nvdx12::BuiltinKernels

> Compute kernels for the commands D3D12 lacks: vkCmdFillBuffer,
> vkCmdBlitImage and nvvk::cmdGenerateMipmaps.

The kernels are compiled at init from embedded HLSL against the bindless
root signature of pipeline.hpp (push data in the root SRV
`InitInfo::pushDataRootParameter`, heaps indexed directly) and run on the
application's descriptor heap and upload ring. Unlike the Vulkan commands
these are dispatches: each call rebinds the heaps (DescriptorHeap::cmdBind),
then `InitInfo::rootSignature`, and leaves the kernel's pipeline state
bound, so callers set their own root signature and PSO afterwards and order
the kernel against other accesses with a barrier. The blit and mipmap
kernels take transient descriptor slots from the heap. They go back to the
heap once the submission of their command list on the upload ring's queue
completed (returned by the next blit or mipmap call, or by `deinit`), or to
the owner through `setSlotReleaseCallback`. A kernel whose push data does
not fit in the upload ring is not dispatched (logged by the ring);
`cmdGenerateMipmaps` then builds no further level but still transitions
every level.

Differences with nvvk that follow from D3D12:
- `cmdGenerateMipmaps` is the member counterpart of nvvk's free function
  nvvk::cmdGenerateMipmaps (mipmaps.hpp): the kernel needs the heap and the
  upload ring held here. It takes the nvdx12::Image, whose extent, mip
  count, layer count and tracked layout replace nvvk's size, levelCount,
  layerCount and currentLayout, and always ends in SHADER_RESOURCE. It
  filters with an exact box (area-weighted) filter instead of a linear
  blit: the same on even sizes, and on odd sizes every source texel
  contributes (a linear blit skips parts of them). The level count of a
  full chain is nvdx12::mipLevels (formats.hpp).
- `cmdFillBuffer` and `cmdBlitImage` stand in for vkCmdFillBuffer and
  vkCmdBlitImage, which D3D12 does not have.

See usage_BuiltinKernels() in builtin_kernels.cpp.
-------------------------------------------------------------------------------------------------*/

enum class Filter
{
  eNearest,
  eLinear
};

// Pixel rectangle inside a mip level; a zero extent stands for the whole level.
struct Rect2D
{
  uint32_t x{0};
  uint32_t y{0};
  uint32_t width{0};
  uint32_t height{0};
};

class BuiltinKernels
{
public:
  struct InitInfo
  {
    ID3D12Device*        device{nullptr};
    ShaderCompiler*      compiler{nullptr};  // used during init only
    ID3D12RootSignature* rootSignature{nullptr};
    uint32_t             pushDataRootParameter{0};
    DescriptorHeap*      descriptorHeap{nullptr};  // may also be set later with setDescriptorHeap
    UploadRing*          uploadRing{nullptr};
    // Flags of the compute pipeline states.
    D3D12_PIPELINE_STATE_FLAGS pipelineFlags{D3D12_PIPELINE_STATE_FLAG_NONE};
  };

  BuiltinKernels()                                 = default;
  BuiltinKernels(const BuiltinKernels&)            = delete;
  BuiltinKernels& operator=(const BuiltinKernels&) = delete;
  ~BuiltinKernels() { assert(m_device == nullptr && "Missing deinit()"); }

  // Compiles the kernels; E_FAIL when one of them does not compile. The
  // upload ring must be initialized: its queue's submissions free the transient slots.
  HRESULT init(const InitInfo& info);
  // The GPU must have finished every command list the kernels recorded into.
  void deinit();

  // Called with the list the slots were recorded into; release them once it executed.
  using SlotReleaseCallback = std::function<void(ID3D12CommandList* cmd, uint32_t firstSlot, uint32_t count)>;
  void setSlotReleaseCallback(SlotReleaseCallback callback) { m_releaseSlots = std::move(callback); }
  // The heap the kernels bind and take transient slots from.
  void setDescriptorHeap(DescriptorHeap* heap) { m_heap = heap; }

  // Fills [offset, offset + size) with the 32-bit `value`; offset and size
  // are multiples of 4, size == ~0ull fills to the end of the buffer.
  void cmdFillBuffer(ID3D12GraphicsCommandList* cmd, const Buffer& dst, uint64_t offset, uint64_t size, uint32_t value);

  // Scaled copy of mip 0 of `src` into `dstRect` of mip 0 of `dst` (the
  // whole destination by default), sampled at the center of each
  // destination texel with clamp-to-edge, as vkCmdBlitImage; `dst` must
  // allow unordered access. Transitions src to SHADER_RESOURCE and dst to
  // UNORDERED_ACCESS and leaves them there; texels of `dst` outside the
  // rectangle keep their content (the destination is discarded only when
  // it leaves UNDEFINED and the blit covers all of it). Like other
  // dispatches, two blits into overlapping rectangles of one destination
  // need a barrier in between (cmdMemoryBarrier). Single-layer 2D
  // images; float, UNORM or SNORM formats (the destination's with typed UAV
  // stores, so not sRGB).
  void cmdBlitImage(ID3D12GraphicsCommandList7* cmd, Image& src, Image& dst, Filter filter, const Rect2D& dstRect = {});

  // Builds mips 1..N-1 of every layer, each from the previous level
  // (nvvk::cmdGenerateMipmaps). Each texel is the area-weighted average of
  // the previous level's texels under its footprint: 2 taps per axis on
  // even sizes (the 2x2 box), 3 taps with weights (n - x, n, x + 1) / (2n + 1)
  // on an odd size 2n + 1, so any size, power of two or not, is filtered
  // without dropping a row or column. 1D and 2D images and arrays, single
  // sample; float, UNORM or SNORM formats with typed UAV stores (sRGB
  // images cannot allow unordered access: build their chain on the CPU, or
  // use a UNORM image, which is filtered as stored). The image must allow
  // unordered access; every level is left in SHADER_RESOURCE with that
  // layout's whole scope (every shader stage) tracked.
  void cmdGenerateMipmaps(ID3D12GraphicsCommandList7* cmd, Image& image);

  ID3D12PipelineState* fillBufferPipeline() const { return m_fillBuffer.Get(); }
  ID3D12PipelineState* blitImagePipeline() const { return m_blitImage.Get(); }
  ID3D12PipelineState* mipmapPipeline() const { return m_mipmap.Get(); }      // 2D images
  ID3D12PipelineState* mipmap1DPipeline() const { return m_mipmap1D.Get(); }  // 1D images

private:
  void cmdBind(ID3D12GraphicsCommandList* cmd);
  // False when the upload ring has no room: the kernel must not be dispatched.
  bool cmdPushData(ID3D12GraphicsCommandList* cmd, const void* data, uint32_t size);
  void releaseSlots(ID3D12CommandList* cmd, uint32_t firstSlot, uint32_t count);
  // Returns the pending slots whose submission completed to the heap; all of them with `all`.
  void releaseCompletedSlots(bool all = false);

  // Transient slots recorded into `cmd`; its submission sets `fenceValue`
  // and clears `cmd`.
  struct PendingSlots
  {
    ID3D12CommandList* cmd{nullptr};
    uint64_t           fenceValue{0};
    uint32_t           firstSlot{0};
    uint32_t           count{0};
  };

  ID3D12Device*                 m_device{nullptr};         // not owned
  DescriptorHeap*               m_heap{nullptr};           // not owned
  UploadRing*                   m_ring{nullptr};           // not owned
  QueueTimeline*                m_queue{nullptr};          // not owned; the upload ring's
  ID3D12RootSignature*          m_rootSignature{nullptr};  // not owned; the kernels' root signature
  uint32_t                      m_pushDataRootParameter{0};
  ComPtr<ID3D12PipelineState>   m_fillBuffer;
  ComPtr<ID3D12PipelineState>   m_blitImage;
  ComPtr<ID3D12PipelineState>   m_mipmap;
  ComPtr<ID3D12PipelineState>   m_mipmap1D;
  SlotReleaseCallback           m_releaseSlots;  // empty: the slots wait in m_pendingSlots
  QueueTimeline::ListenerHandle m_submitListener{0};
  std::vector<PendingSlots>     m_pendingSlots;
  std::mutex                    m_pendingMutex;  // m_pendingSlots; the submit listener runs on any thread
};

}  // namespace nvdx12
