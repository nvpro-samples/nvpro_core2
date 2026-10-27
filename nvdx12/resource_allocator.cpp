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

#include "resource_allocator.hpp"

#include <algorithm>
#include <string>
#include <utility>

#include <dxgi1_4.h>

#include <nvutils/logger.hpp>

#include "check_error.hpp"
#include "debug_util.hpp"
#include "default_structs.hpp"

namespace nvdx12 {

static constexpr uint64_t kUavWindowBytes = 1ull << 32;  // one raw UAV covers at most 4 GiB

ResourceAllocator::ResourceAllocator(ResourceAllocator&& other) noexcept
{
  *this = std::move(other);
}

ResourceAllocator& ResourceAllocator::operator=(ResourceAllocator&& other) noexcept
{
  if(this != &other)
  {
    assert(m_allocator == nullptr && "Missing deinit()");
    m_device                  = std::move(other.m_device);
    m_adapter                 = std::move(other.m_adapter);
    m_allocator               = std::exchange(other.m_allocator, nullptr);
    m_heap                    = std::exchange(other.m_heap, nullptr);
    m_maxMemoryAllocationSize = std::exchange(other.m_maxMemoryAllocationSize, 0);
    m_allocationCounter.store(other.m_allocationCounter.exchange(0));
    m_leakID = std::exchange(other.m_leakID, ~0u);
    m_bufferBytes.store(other.m_bufferBytes.exchange(0));
    m_imageBytes.store(other.m_imageBytes.exchange(0));
  }
  return *this;
}

ResourceAllocator::~ResourceAllocator()
{
  assert(m_allocator == nullptr && "Missing deinit()");
}

HRESULT ResourceAllocator::init(D3D12MA::ALLOCATOR_DESC allocatorDesc, DescriptorHeap* descriptorHeap)
{
  assert(m_allocator == nullptr && "Missing deinit()");
  if(allocatorDesc.pDevice == nullptr || descriptorHeap == nullptr)
  {
    return E_INVALIDARG;
  }
  // Locals until the allocator exists, so that a failure keeps no reference.
  ComPtr<ID3D12Device10> device;
  ComPtr<IDXGIAdapter>   adapter;
  NVDX12_FAIL_RETURN(allocatorDesc.pDevice->QueryInterface(IID_PPV_ARGS(&device)));  // CreateResource3

  // D3D12MA reads the memory budget from the device's adapter.
  if(allocatorDesc.pAdapter == nullptr)
  {
    ComPtr<IDXGIFactory4> factory;
    NVDX12_FAIL_RETURN(CreateDXGIFactory2(0, IID_PPV_ARGS(&factory)));
    NVDX12_FAIL_RETURN(factory->EnumAdapterByLuid(device->GetAdapterLuid(), IID_PPV_ARGS(&adapter)));
    allocatorDesc.pAdapter = adapter.Get();
  }
  else
  {
    adapter = allocatorDesc.pAdapter;
  }

  NVDX12_FAIL_RETURN(D3D12MA::CreateAllocator(&allocatorDesc, &m_allocator));
  LOGI("ResourceAllocator: tight alignment %s\n", m_allocator->IsTightAlignmentSupported() ? "supported" : "not supported");

  // D3D12 bounds a resource by its GPU virtual address range, not by an allocation size.
  D3D12_FEATURE_DATA_GPU_VIRTUAL_ADDRESS_SUPPORT vaSupport{};
  UINT                                           resourceVaBits = 40;
  if(SUCCEEDED(device->CheckFeatureSupport(D3D12_FEATURE_GPU_VIRTUAL_ADDRESS_SUPPORT, &vaSupport, sizeof(vaSupport)))
     && vaSupport.MaxGPUVirtualAddressBitsPerResource > 0 && vaSupport.MaxGPUVirtualAddressBitsPerResource < 64)
  {
    resourceVaBits = vaSupport.MaxGPUVirtualAddressBitsPerResource;
  }
  m_maxMemoryAllocationSize = 1ull << resourceVaBits;

  m_device  = std::move(device);
  m_adapter = std::move(adapter);
  m_heap    = descriptorHeap;
  m_allocationCounter.store(0);
  m_bufferBytes.store(0);
  m_imageBytes.store(0);
  return S_OK;
}

void ResourceAllocator::deinit()
{
  if(!m_allocator)
  {
    return;
  }
  if(m_bufferBytes.load() || m_imageBytes.load())
  {
    LOGW("ResourceAllocator::deinit with %llu buffer bytes and %llu image bytes still allocated\n",
         static_cast<unsigned long long>(m_bufferBytes.load()), static_cast<unsigned long long>(m_imageBytes.load()));
  }
  // Reports unfreed placed allocations by name ("nvdx12AllocID: <id>", see
  // setLeakID); unfreed committed ones only assert in debug builds.
  m_allocator->Release();
  m_allocator = nullptr;
  m_device.Reset();
  m_adapter.Reset();
  m_heap                    = nullptr;
  m_maxMemoryAllocationSize = 0;
  m_leakID                  = ~0u;
}

void ResourceAllocator::setLeakID(uint32_t id)
{
  m_leakID = id;
}

ID3D12Heap* ResourceAllocator::getHeap(D3D12MA::Allocation* allocation) const
{
  return allocation ? allocation->GetHeap() : nullptr;
}

void ResourceAllocator::addLeakDetection(D3D12MA::Allocation* allocation) const
{
  const uint32_t id = m_allocationCounter.fetch_add(1);
  if(m_leakID == id)
  {
    if(IsDebuggerPresent())
    {
      DebugBreak();
    }
  }
  allocation->SetName((L"nvdx12AllocID: " + std::to_wstring(id)).c_str());
}

//--------------------------------------------------------------------------------------------------
// Buffers

static D3D12_HEAP_TYPE heapType(MemoryUsage usage)
{
  switch(usage)
  {
    case MemoryUsage::eHostUpload:
      return D3D12_HEAP_TYPE_UPLOAD;
    case MemoryUsage::eHostReadback:
      return D3D12_HEAP_TYPE_READBACK;
    default:
      return D3D12_HEAP_TYPE_DEFAULT;
  }
}

static MemoryUsage toMemoryUsage(D3D12_HEAP_TYPE type)
{
  switch(type)
  {
    case D3D12_HEAP_TYPE_UPLOAD:
    case D3D12_HEAP_TYPE_GPU_UPLOAD:
      return MemoryUsage::eHostUpload;
    case D3D12_HEAP_TYPE_READBACK:
      return MemoryUsage::eHostReadback;
    default:
      return MemoryUsage::eDevice;
  }
}

HRESULT ResourceAllocator::createBuffer(Buffer&                   buffer,
                                        uint64_t                  size,
                                        D3D12_RESOURCE_FLAGS      usage,
                                        MemoryUsage               memoryUsage,
                                        D3D12MA::ALLOCATION_FLAGS flags,
                                        uint64_t                  minAlignment) const
{
  D3D12MA::ALLOCATION_DESC allocDesc{};
  allocDesc.Flags    = flags;
  allocDesc.HeapType = heapType(memoryUsage);

  // UPLOAD and READBACK heaps reject ALLOW_UNORDERED_ACCESS. Host-upload
  // buffers that shaders address go to the GPU_UPLOAD heap (resizable BAR);
  // without it, and for readback memory, the buffer gets no shader address.
  if(usage & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS)
  {
    if(memoryUsage == MemoryUsage::eHostUpload && m_allocator->IsGPUUploadHeapSupported())
    {
      allocDesc.HeapType = D3D12_HEAP_TYPE_GPU_UPLOAD;
    }
    else if(memoryUsage != MemoryUsage::eDevice)
    {
      static std::atomic<bool> warned = false;
      if(memoryUsage == MemoryUsage::eHostUpload && !warned.exchange(true))
      {
        LOGW("createBuffer: no GPU upload heap on this device; host-upload buffers have no shader address\n");
      }
      usage &= ~D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    }
  }

  D3D12_RESOURCE_DESC1 desc{};
  desc.Dimension        = D3D12_RESOURCE_DIMENSION_BUFFER;
  desc.Width            = size;
  desc.Height           = 1;
  desc.DepthOrArraySize = 1;
  desc.MipLevels        = 1;
  desc.Format           = DXGI_FORMAT_UNKNOWN;
  desc.SampleDesc.Count = 1;
  desc.Layout           = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
  desc.Flags            = usage;
  return createBuffer(buffer, desc, allocDesc, minAlignment);
}

HRESULT ResourceAllocator::createBuffer(Buffer&                         buffer,
                                        const D3D12_RESOURCE_DESC1&     bufferDesc,
                                        const D3D12MA::ALLOCATION_DESC& allocDesc,
                                        uint64_t                        minAlignment) const
{
  buffer = {};
  if(bufferDesc.Width == 0 || bufferDesc.Dimension != D3D12_RESOURCE_DIMENSION_BUFFER)
  {
    return E_INVALIDARG;
  }
  if(minAlignment > BUFFER_PLACEMENT_ALIGNMENT)
  {
    LOGE("createBuffer: minAlignment %llu exceeds the %llu-byte buffer placement\n",
         static_cast<unsigned long long>(minAlignment), static_cast<unsigned long long>(BUFFER_PLACEMENT_ALIGNMENT));
    return E_INVALIDARG;
  }

  HRESULT hr = m_allocator->CreateResource3(&allocDesc, &bufferDesc, D3D12_BARRIER_LAYOUT_UNDEFINED, nullptr, 0,
                                            nullptr, &buffer.allocation, IID_PPV_ARGS(&buffer.buffer));
  if(FAILED(hr))
  {
    LOGE("createBuffer (%llu bytes) failed: %s\n", static_cast<unsigned long long>(bufferDesc.Width),
         hresultToString(hr).c_str());
    buffer = {};
    return hr;
  }
  addLeakDetection(buffer.allocation);

  buffer.bufferSize  = bufferDesc.Width;
  buffer.memoryUsage = toMemoryUsage(allocDesc.HeapType);
  buffer.gpuAddress  = buffer.buffer->GetGPUVirtualAddress();

  if(buffer.memoryUsage != MemoryUsage::eDevice)
  {
    // Persistent mapping; the read range is irrelevant for upload memory
    // and the whole buffer for readback.
    D3D12_RANGE readRange{0, buffer.memoryUsage == MemoryUsage::eHostReadback ? SIZE_T(bufferDesc.Width) : 0};
    hr = buffer.buffer->Map(0, &readRange, reinterpret_cast<void**>(&buffer.mapping));
    if(FAILED(hr))
    {
      NVDX12_FAIL_REPORT(hr);
      buffer.buffer.Reset();
      buffer.allocation->Release();
      buffer = {};
      return hr;
    }
  }

  // Counted before the descriptors: a failure below goes through destroyBuffer, which subtracts it.
  m_bufferBytes.fetch_add(buffer.bufferSize);

  // Acceleration-structure storage is only ever addressed by the API and by
  // AS SRVs; a raw UAV on it is an invalid view, so it gets no shader address.
  if((bufferDesc.Flags & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS)
     && !(bufferDesc.Flags & D3D12_RESOURCE_FLAG_RAYTRACING_ACCELERATION_STRUCTURE))
  {
    hr = writeBufferDescriptors(buffer);
    if(FAILED(hr))
    {
      LOGE("createBuffer: the buffer descriptors could not be written (descriptor heap full?)\n");
      destroyBuffer(buffer);
      return hr;
    }
  }
  return S_OK;
}

HRESULT ResourceAllocator::writeBufferDescriptors(Buffer& buffer) const
{
  const uint32_t windows = static_cast<uint32_t>((buffer.bufferSize + kUavWindowBytes - 1) / kUavWindowBytes);
  const uint32_t first   = m_heap->acquireSlots(windows);
  if(first == kInvalidSlot)
  {
    return E_OUTOFMEMORY;
  }
  for(uint32_t w = 0; w < windows; w++)
  {
    const uint64_t offset = uint64_t(w) * kUavWindowBytes;
    const uint64_t bytes  = std::min(kUavWindowBytes, buffer.bufferSize - offset);
    // A raw view must end on a dword boundary; odd tails are not addressable.
    const HRESULT hr = m_heap->writeBufferDescriptor(first + w, buffer.buffer.Get(), offset, bytes & ~3ull);
    if(FAILED(hr))
    {
      m_heap->releaseSlots(first, windows);
      return hr;
    }
  }
  buffer.uavSlot      = first;
  buffer.uavSlotCount = windows;
  buffer.address      = uint64_t(first) << 32;
  return S_OK;
}

void ResourceAllocator::destroyBuffer(Buffer& buffer) const
{
  if(!buffer.buffer)
  {
    buffer = {};
    return;
  }
  if(buffer.uavSlot != kInvalidSlot)
  {
    m_heap->releaseSlots(buffer.uavSlot, buffer.uavSlotCount);
  }
  if(buffer.mapping)
  {
    buffer.buffer->Unmap(0, nullptr);
  }
  buffer.buffer.Reset();
  if(buffer.allocation)
  {
    buffer.allocation->Release();
  }
  const uint64_t bytes = buffer.bufferSize;
  uint64_t       live  = m_bufferBytes.load();
  while(!m_bufferBytes.compare_exchange_weak(live, live - std::min(live, bytes)))
  {
  }
  buffer = {};
}

//--------------------------------------------------------------------------------------------------
// Images

HRESULT ResourceAllocator::createImage(Image& image, const D3D12_RESOURCE_DESC1& imageDesc, D3D12_BARRIER_LAYOUT initialLayout) const
{
  // A placed render target or depth-stencil must be cleared or discarded
  // before its first use; committed ones need not, so they stay committed.
  D3D12MA::ALLOCATION_DESC allocDesc{};
  allocDesc.HeapType = D3D12_HEAP_TYPE_DEFAULT;
  if(imageDesc.Flags & (D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET | D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL))
  {
    allocDesc.Flags = D3D12MA::ALLOCATION_FLAG_COMMITTED;
  }
  return createImage(image, imageDesc, initialLayout, allocDesc);
}

HRESULT ResourceAllocator::createImage(Image&                          image,
                                       const D3D12_RESOURCE_DESC1&     imageDesc,
                                       D3D12_BARRIER_LAYOUT            initialLayout,
                                       const D3D12MA::ALLOCATION_DESC& allocDesc) const
{
  image = {};
  if(imageDesc.Width == 0 || imageDesc.Height == 0 || imageDesc.Format == DXGI_FORMAT_UNKNOWN
     || imageDesc.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER)
  {
    return E_INVALIDARG;
  }

  HRESULT hr = m_allocator->CreateResource3(&allocDesc, &imageDesc, initialLayout, nullptr, 0, nullptr,
                                            &image.allocation, IID_PPV_ARGS(&image.image));
  if(FAILED(hr))
  {
    LOGE("createImage (%llux%ux%u, format %d, %u mips) failed: %s\n", static_cast<unsigned long long>(imageDesc.Width),
         imageDesc.Height, imageDesc.DepthOrArraySize, int(imageDesc.Format), imageDesc.MipLevels, hresultToString(hr).c_str());
    image = {};
    return hr;
  }
  addLeakDetection(image.allocation);

  // Mip and layer counts as the runtime created them: MipLevels 0 requests
  // the full chain, and DepthOrArraySize is the depth of a 3D image.
  const D3D12_RESOURCE_DESC created = image.image->GetDesc();
  const bool                is3D    = imageDesc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D;
  image.dimension                   = imageDesc.Dimension;
  image.format                      = imageDesc.Format;
  image.extent = {static_cast<uint32_t>(imageDesc.Width), imageDesc.Height, is3D ? uint32_t(created.DepthOrArraySize) : 1u};
  image.mipLevels   = created.MipLevels;
  image.arrayLayers = is3D ? 1u : created.DepthOrArraySize;
  image.sampleCount = std::max(1u, imageDesc.SampleDesc.Count);
  image.flags       = imageDesc.Flags;
  image.layout      = initialLayout;

  m_imageBytes.fetch_add(image.allocation->GetSize());
  return S_OK;
}

void ResourceAllocator::destroyImage(Image& image) const
{
  image.image.Reset();
  if(image.allocation)
  {
    const uint64_t bytes = image.allocation->GetSize();
    uint64_t       live  = m_imageBytes.load();
    while(!m_imageBytes.compare_exchange_weak(live, live - std::min(live, bytes)))
    {
    }
    image.allocation->Release();
  }
  image = {};
}

//--------------------------------------------------------------------------------------------------
// Acceleration structures

HRESULT ResourceAllocator::createAcceleration(AccelerationStructure& accel, const AccelerationStructureCreateInfo& accInfo) const
{
  D3D12MA::ALLOCATION_DESC allocDesc{};
  allocDesc.HeapType = D3D12_HEAP_TYPE_DEFAULT;
  return createAcceleration(accel, accInfo, allocDesc);
}

HRESULT ResourceAllocator::createAcceleration(AccelerationStructure&                 accel,
                                              const AccelerationStructureCreateInfo& accInfo,
                                              const D3D12MA::ALLOCATION_DESC&        allocDesc) const
{
  accel = {};
  D3D12_RESOURCE_DESC1 desc{};
  desc.Dimension        = D3D12_RESOURCE_DIMENSION_BUFFER;
  desc.Width            = accInfo.size;
  desc.Height           = 1;
  desc.DepthOrArraySize = 1;
  desc.MipLevels        = 1;
  desc.Format           = DXGI_FORMAT_UNKNOWN;
  desc.SampleDesc.Count = 1;
  desc.Layout           = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
  desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS | D3D12_RESOURCE_FLAG_RAYTRACING_ACCELERATION_STRUCTURE;
  HRESULT hr = createBuffer(accel.buffer, desc, allocDesc);
  if(FAILED(hr))
  {
    return hr;
  }
  accel.gpuAddress = accel.buffer.gpuAddress;
  accel.size       = accInfo.size;
  if(accInfo.type == D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL)
  {
    accel.srvSlot = m_heap->acquireSlots();
    if(accel.srvSlot == kInvalidSlot)
    {
      LOGE("createAcceleration: out of descriptor-heap slots\n");
      destroyAcceleration(accel);
      return E_OUTOFMEMORY;
    }
    hr = m_heap->writeAccelerationStructureSrv(accel.srvSlot, accel.gpuAddress);
    if(FAILED(hr))
    {
      LOGE("createAcceleration: the TLAS SRV could not be written\n");
      destroyAcceleration(accel);
      return hr;
    }
  }
  return S_OK;
}

void ResourceAllocator::destroyAcceleration(AccelerationStructure& accel) const
{
  if(accel.srvSlot != kInvalidSlot)
  {
    m_heap->releaseSlots(accel.srvSlot);
  }
  destroyBuffer(accel.buffer);
  accel = {};
}

}  // namespace nvdx12

//--------------------------------------------------------------------------------------------------
// Usage example
//--------------------------------------------------------------------------------------------------
[[maybe_unused]] static void usage_ResourceAllocator()
{
  ID3D12Device*          device = nullptr;  // EX: get the device from the context
  nvdx12::DescriptorHeap descriptorHeap;    // EX: the application's shader-visible heap, initialized

  nvdx12::ResourceAllocator allocator;
  allocator.init({.pDevice = device}, &descriptorHeap);

  // Device buffer with a shader address: store `buffer.address` in shader I/O structs.
  nvdx12::Buffer buffer;
  allocator.createBuffer(buffer, 1024);
  NVDX12_DBG_NAME(buffer.buffer);

  // Host-readable buffer, persistently mapped; no shader address needed.
  nvdx12::Buffer readback;
  allocator.createBuffer(readback, 1024, D3D12_RESOURCE_FLAG_NONE, nvdx12::MemoryUsage::eHostReadback);

  D3D12_RESOURCE_DESC1 imageDesc = DEFAULT_D3D12_RESOURCE_DESC1;
  imageDesc.Width                = 256;
  imageDesc.Height               = 256;
  imageDesc.MipLevels            = 9;
  imageDesc.Format               = DXGI_FORMAT_R8G8B8A8_UNORM;
  nvdx12::Image image;
  allocator.createImage(image, imageDesc);

  // Traceable acceleration structure: `accel.shaderHandle()` goes to the shader,
  // `accel.gpuAddress` to the build.
  nvdx12::AccelerationStructure tlas;
  allocator.createAcceleration(tlas, {.type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL, .size = 64 * 1024});

  // destroy later, once the GPU no longer uses them
  allocator.destroyAcceleration(tlas);
  allocator.destroyImage(image);
  allocator.destroyBuffer(readback);
  allocator.destroyBuffer(buffer);
  allocator.deinit();
}
