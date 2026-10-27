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
#include <mutex>
#include <unordered_map>
#include <vector>

#include "basic_types.hpp"
#include "resources.hpp"

namespace nvdx12 {

/*-------------------------------------------------------------------------------------------------
# class nvdx12::DescriptorHeap

> The process-wide shader-visible descriptor heaps, indexed directly from
> shaders with ResourceDescriptorHeap[] / SamplerDescriptorHeap[] (Shader
> Model 6.6). The D3D12 counterpart of nvvk::DescriptorHeap, with its
> descriptor-writing and sampler entry points.

Slot numbers travel to shaders in push constants, push data and shader I/O structs:

- Images take an ImageSlot, a pair: SRV over the whole mip chain at `srv`,
  UAV of one mip at `srv + 1` (`acquireImageSlot`, `writeImageSlot`).
- Everything else takes single slots (`acquireSlots`): buffers own one raw
  UAV slot per 4 GiB window, written by nvdx12::ResourceAllocator
  (`Buffer::address` is `uavSlot << 32`), acceleration structures one SRV
  slot (`writeAccelerationStructureSrv`), SRV-only images one SRV slot
  (`writeSampledImageDescriptor`).
- Samplers are deduplicated by description and reference counted
  (`acquireSamplerDescriptor`; nvdx12::SamplerPool wraps this with
  nvvk::SamplerPool's names).

Resource slot 0 is reserved as a null raw UAV so a shader address of 0 never
designates a live buffer; the write functions reject it (E_INVALIDARG).

Thread safety (an app's loader thread creates resources while the main
thread records frames):
- Thread-safe: slot acquisition and release (`acquireSlots`,
  `releaseSlots`, `acquireImageSlot`, `releaseImageSlot`), sampler
  acquisition and release, `usedResourceDescriptors`,
  `usedSamplerDescriptors`, and the binding state (`setRootSignature`,
  `cmdBind`): a `cmdBind` sees the root signature either before or after a
  concurrent update.
- Descriptor writes (`write*`) take no lock: writes to distinct slots may run
  concurrently, and the caller owns the slot it writes (acquired, not yet
  released). Handles and capacities are immutable between init and deinit.
- Not thread-safe: `init` and `deinit`, which no other call may overlap.

Release checks: releasing resource slots that are not currently allocated
(double release, never-acquired slots, slot 0, a range overlapping free
slots) and releasing a sampler slot that is not live are detected with
a cost proportional to the free list the release scans anyway. Debug
builds assert; release builds log an error and ignore the call, so the free
list stays consistent. A sampler shared by several acquisitions is
reference counted: an extra release by one owner of a shared sampler cannot
be told apart from another owner's release. `deinit` warns with the number
of resource slots and samplers still acquired (leaks), as
nvdx12::ResourceAllocator does for live bytes.

Differences with nvvk that follow from D3D12:
- The heaps are ID3D12DescriptorHeap objects the class creates in `init`
  (with their capacities), not user-owned buffers: there is no
  setupSamplerHeap / setupResourceHeap sizing, no heap base pointers in the
  write functions, no dirty-range tracking (descriptor writes are immediate)
  and no capacity queries beyond `maxResourceDescriptors` / `maxSamplerCount`.
- One resource index space shared by images, buffers and acceleration
  structures, as `ResourceDescriptorHeap[]` sees it: slot numbers are what
  shaders index, so there are no per-type local indices or shader index bases
  (imageShaderIndexBase, bufferShaderIndexBase, region byte offsets).
- Write functions take the D3D12 objects (an Image, a resource with a byte
  range) instead of image views, layouts and descriptor types; buffer
  descriptors are raw UAVs (ByteAddressBuffer) only.
- `cmdBind` replaces cmdBindHeaps and also sets the bindless root signature
  (D3D12 extra).
- D3D12 extras: slot allocation (`acquireSlots`, `acquireImageSlot`), the
  ImageSlot writer `writeImageSlot`, `writeDisplayImage`,
  `writeAccelerationStructureSrv`, and the CPU / GPU descriptor handles that
  ImGui and SDKs consume.

See usage_DescriptorHeap() in descriptor_heap.cpp.
-------------------------------------------------------------------------------------------------*/

class DescriptorHeap
{
public:
  DescriptorHeap() = default;
  ~DescriptorHeap() { assert(!isInitialized() && "Missing deinit()?"); }
  DescriptorHeap(const DescriptorHeap&)            = delete;
  DescriptorHeap& operator=(const DescriptorHeap&) = delete;

  // Creates the two shader-visible heaps: `maxResourceDescriptors` entries
  // shared by image pairs, buffer UAVs and acceleration-structure SRVs, and
  // `maxSamplers` sampler entries (D3D12 allows at most 2048).
  HRESULT init(ID3D12Device* device, uint32_t maxResourceDescriptors = 1u << 18, uint32_t maxSamplers = 2048);
  void    deinit();
  bool    isInitialized() const { return m_device != nullptr; }

  // Directly indexed heaps need resource binding tier 3 and Shader Model 6.6.
  static bool isSupported(ID3D12Device* device);

  ID3D12Device* getDevice() const { return m_device; }

  //-----------------------------------------------------------------------------
  // Image slots: the SRV / UAV pair of an image (ImageSlot in resources.hpp), D3D12 extra

  ImageSlot acquireImageSlot();
  void      releaseImageSlot(ImageSlot slot);

  // SRV over the whole mip chain at `slot.srv`, UAV of `uavMipLevel` at
  // `slot.uav()`, both in the image format or `viewFormat`. The image must
  // allow unordered access. For the SRV alone, use writeSampledImageDescriptor(slot.srv, ...).
  HRESULT writeImageSlot(ImageSlot slot, const Image& image, uint32_t uavMipLevel = 0, DXGI_FORMAT viewFormat = DXGI_FORMAT_UNKNOWN);

  //-----------------------------------------------------------------------------
  // Single slots: one descriptor each, `count` consecutive ones per call

  uint32_t acquireSlots(uint32_t count = 1);
  void     releaseSlots(uint32_t firstSlot, uint32_t count = 1);

  // SRV over the whole mip chain, in the image format (DXGI_FORMAT_UNKNOWN)
  // or an explicit view format (typeless images, sRGB views).
  HRESULT writeSampledImageDescriptor(uint32_t slot, const Image& image, DXGI_FORMAT viewFormat = DXGI_FORMAT_UNKNOWN);
  // UAV of one mip level, in the image format or an explicit view format
  // (typeless images, the UNORM twin of an sRGB image). The image must allow unordered access.
  HRESULT writeStorageImageDescriptor(uint32_t slot, const Image& image, uint32_t mipLevel = 0, DXGI_FORMAT viewFormat = DXGI_FORMAT_UNKNOWN);
  // Raw (R32_TYPELESS, RAW flag) UAV over [byteOffset, byteOffset + byteSize),
  // what shaders read as a (RW)ByteAddressBuffer; the offset is a multiple
  // of 16 (D3D12_RAW_UAV_SRV_BYTE_ALIGNMENT) and the window at most 4 GiB. A
  // window under 4 bytes gets a null view.
  HRESULT writeBufferDescriptor(uint32_t slot, ID3D12Resource* resource, uint64_t byteOffset, uint64_t byteSize);
  // Raw UAV over a whole buffer of at most 4 GiB.
  HRESULT writeBufferDescriptor(uint32_t slot, const Buffer& buffer)
  {
    return writeBufferDescriptor(slot, buffer.buffer.Get(), 0, buffer.bufferSize);
  }

  // SRV for on-screen display with alpha forced to one: ImGui blends
  // textures by their alpha, which rendered images may use for data. The
  // D3D12 counterpart of a VK_COMPONENT_SWIZZLE_ONE alpha swizzle (D3D12 extra).
  HRESULT writeDisplayImage(uint32_t slot, const Image& image);
  // Acceleration-structure SRV (GPU virtual address based, no resource), D3D12 extra.
  HRESULT writeAccelerationStructureSrv(uint32_t slot, D3D12_GPU_VIRTUAL_ADDRESS address);

  //-----------------------------------------------------------------------------
  // Samplers

  // Deduplicated by D3D12_SAMPLER_DESC and reference counted: identical
  // descriptions share a slot. kInvalidSlot when the sampler heap is full.
  uint32_t acquireSamplerDescriptor(const D3D12_SAMPLER_DESC& desc);
  void     releaseSamplerDescriptor(uint32_t index);

  // Explicit index (low-level): writes `desc` at `index`, outside the
  // deduplication. The caller owns `index`, which must not be a slot
  // acquireSamplerDescriptor handed out.
  HRESULT writeSamplerDescriptor(uint32_t index, const D3D12_SAMPLER_DESC& desc);

  //-----------------------------------------------------------------------------
  // Per command list (D3D12 extras)

  // SetDescriptorHeaps, plus the root signature when one was given: for
  // compute, and for graphics as well on direct lists; nothing on copy lists,
  // which take neither. Call at the top of every command list and again after
  // another library (ImGui, an SDK) changed the bound heaps or root signatures.
  void cmdBind(ID3D12GraphicsCommandList* cmd) const;
  // The bindless root signature of pipeline.hpp, shared by compute,
  // ray-tracing and graphics pipelines.
  void setRootSignature(ID3D12RootSignature* rootSignature);

  //-----------------------------------------------------------------------------
  // Handles (D3D12 extras) and capacities

  D3D12_GPU_DESCRIPTOR_HANDLE gpuHandle(uint32_t slot) const;  // ImGui ImTextureID, external SDKs
  D3D12_CPU_DESCRIPTOR_HANDLE cpuHandle(uint32_t slot) const;
  // Resource slot of a handle gpuHandle() returned (descriptor callbacks of
  // ImGui and SDKs); kInvalidSlot when the handle is outside the heap.
  uint32_t                    slotFromGpuHandle(D3D12_GPU_DESCRIPTOR_HANDLE handle) const;
  D3D12_GPU_DESCRIPTOR_HANDLE samplerGpuHandle(uint32_t slot) const;
  D3D12_CPU_DESCRIPTOR_HANDLE samplerCpuHandle(uint32_t slot) const;
  ID3D12DescriptorHeap*       resourceHeap() const { return m_resourceHeap.Get(); }
  ID3D12DescriptorHeap*       samplerHeap() const { return m_samplerHeap.Get(); }
  uint32_t                    resourceDescriptorSize() const { return m_resourceDescriptorSize; }
  uint32_t                    samplerDescriptorSize() const { return m_samplerDescriptorSize; }

  uint32_t maxResourceDescriptors() const { return m_maxResourceDescriptors; }
  uint32_t maxSamplerCount() const { return m_maxSamplers; }
  // Acquired slots, including the reserved slot 0 (1 when nothing is acquired).
  uint32_t usedResourceDescriptors() const;
  // Distinct live samplers (deduplicated descriptions), D3D12 extra.
  uint32_t usedSamplerDescriptors() const;

private:
  // First-fit free-list allocator over contiguous slot ranges.
  struct RangeAllocator
  {
    struct Range
    {
      uint32_t first;
      uint32_t count;
    };
    std::vector<Range> freeRanges;
    uint32_t           base{0};  // first allocatable slot (reserved slots below)
    uint32_t           next{0};  // first never-allocated slot
    uint32_t           capacity{0};

    uint32_t allocate(uint32_t count);  // kInvalidSlot when full
    // False, leaving the allocator untouched, when [first, first + count) is
    // not entirely allocated: outside [base, next) or overlapping a free range.
    bool     release(uint32_t first, uint32_t count);
    uint32_t usedCount() const;  // allocated slots in [base, next)
  };

  // A slot the write functions accept: inside the heap, and not the reserved null UAV of slot 0.
  bool isWritableSlot(uint32_t slot) const { return slot != 0 && slot < m_maxResourceDescriptors; }

  struct SamplerEntry
  {
    D3D12_SAMPLER_DESC desc;  // compared in full: equal hashes may still differ
    uint32_t           slot;
    uint32_t           refCount;
  };

  ID3D12Device*                m_device{nullptr};
  ID3D12RootSignature*         m_rootSignature{nullptr};  // guarded by m_bindMutex
  mutable std::mutex           m_bindMutex;               // root signature
  ComPtr<ID3D12DescriptorHeap> m_resourceHeap;
  ComPtr<ID3D12DescriptorHeap> m_samplerHeap;
  uint32_t                     m_resourceDescriptorSize{0};
  uint32_t                     m_samplerDescriptorSize{0};
  uint32_t                     m_maxResourceDescriptors{0};
  uint32_t                     m_maxSamplers{0};
  D3D12_CPU_DESCRIPTOR_HANDLE  m_resourceCpuStart{};
  D3D12_GPU_DESCRIPTOR_HANDLE  m_resourceGpuStart{};
  D3D12_CPU_DESCRIPTOR_HANDLE  m_samplerCpuStart{};
  D3D12_GPU_DESCRIPTOR_HANDLE  m_samplerGpuStart{};

  RangeAllocator                                          m_resourceSlots;
  RangeAllocator                                          m_samplerSlots;
  std::unordered_map<uint64_t, std::vector<SamplerEntry>> m_samplerByHash;  // hash(D3D12_SAMPLER_DESC) -> entries
  std::unordered_map<uint32_t, uint64_t> m_samplerHashBySlot;  // reverse lookup for releaseSamplerDescriptor
  mutable std::mutex                     m_mutex;              // guards the allocators and sampler maps
};

}  // namespace nvdx12
