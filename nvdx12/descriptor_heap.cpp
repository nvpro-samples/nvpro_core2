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

#include "descriptor_heap.hpp"

#include <algorithm>
#include <cstring>
#include <utility>

#include <nvutils/logger.hpp>

#include "check_error.hpp"
#include "debug_util.hpp"

namespace nvdx12 {

//--------------------------------------------------------------------------------------------------
// RangeAllocator

uint32_t DescriptorHeap::RangeAllocator::allocate(uint32_t count)
{
  for(size_t i = 0; i < freeRanges.size(); i++)
  {
    Range& range = freeRanges[i];
    if(range.count >= count)
    {
      const uint32_t first = range.first;
      range.first += count;
      range.count -= count;
      if(range.count == 0)
      {
        freeRanges[i] = freeRanges.back();
        freeRanges.pop_back();
      }
      return first;
    }
  }
  if(count <= capacity - next)  // next <= capacity; no overflow for huge counts
  {
    const uint32_t first = next;
    next += count;
    return first;
  }
  return kInvalidSlot;
}

bool DescriptorHeap::RangeAllocator::release(uint32_t first, uint32_t count)
{
  if(count == 0)
  {
    return true;
  }
  // Only allocated slots go back: inside [base, next) (no overflow: first < next)...
  if(first < base || first >= next || count > next - first)
  {
    return false;
  }
  // ... and not overlapping a free range. The absorb loop below scans the
  // free list anyway, so the check costs no extra pass in the common case.
  const uint32_t end = first + count;
  for(size_t i = 0; i < freeRanges.size(); i++)
  {
    const Range& range = freeRanges[i];
    if(first < range.first + range.count && range.first < end)
    {
      return false;
    }
  }
  // Absorb the free ranges on both sides, so that releases bridging two free
  // ranges leave one range and multi-slot requests still find room.
  for(size_t i = 0; i < freeRanges.size();)
  {
    const Range range = freeRanges[i];
    if(range.first + range.count == first)
    {
      first = range.first;
      count += range.count;
    }
    else if(first + count == range.first)
    {
      count += range.count;
    }
    else
    {
      i++;
      continue;
    }
    freeRanges[i] = freeRanges.back();
    freeRanges.pop_back();
  }
  // A range ending at the high-water mark shrinks it instead.
  if(first + count == next)
  {
    next = first;
    return true;
  }
  freeRanges.push_back({first, count});
  return true;
}

uint32_t DescriptorHeap::RangeAllocator::usedCount() const
{
  uint32_t freeCount = 0;
  for(size_t i = 0; i < freeRanges.size(); i++)
  {
    freeCount += freeRanges[i].count;
  }
  return next - base - freeCount;
}

//--------------------------------------------------------------------------------------------------
// Lifecycle

HRESULT DescriptorHeap::init(ID3D12Device* device, uint32_t maxResourceDescriptors, uint32_t maxSamplers)
{
  assert(!isInitialized() && "init() called twice");
  if(device == nullptr || maxResourceDescriptors < 2 || maxSamplers == 0 || maxSamplers > D3D12_MAX_SHADER_VISIBLE_SAMPLER_HEAP_SIZE)
  {
    return E_INVALIDARG;
  }
  // Created first, so that a failure leaves the object uninitialized.
  ComPtr<ID3D12DescriptorHeap> resourceHeap;
  ComPtr<ID3D12DescriptorHeap> samplerHeap;
  D3D12_DESCRIPTOR_HEAP_DESC   desc{};
  desc.Type           = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
  desc.NumDescriptors = maxResourceDescriptors;
  desc.Flags          = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
  NVDX12_FAIL_RETURN(device->CreateDescriptorHeap(&desc, IID_PPV_ARGS(&resourceHeap)));
  desc.Type           = D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER;
  desc.NumDescriptors = maxSamplers;
  NVDX12_FAIL_RETURN(device->CreateDescriptorHeap(&desc, IID_PPV_ARGS(&samplerHeap)));

  m_device                 = device;
  m_maxResourceDescriptors = maxResourceDescriptors;
  m_maxSamplers            = maxSamplers;
  m_resourceHeap           = std::move(resourceHeap);
  m_samplerHeap            = std::move(samplerHeap);
  setDebugName(m_resourceHeap.Get(), "nvdx12::DescriptorHeap resources");
  setDebugName(m_samplerHeap.Get(), "nvdx12::DescriptorHeap samplers");

  m_resourceDescriptorSize = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
  m_samplerDescriptorSize  = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER);
  m_resourceCpuStart       = m_resourceHeap->GetCPUDescriptorHandleForHeapStart();
  m_resourceGpuStart       = m_resourceHeap->GetGPUDescriptorHandleForHeapStart();
  m_samplerCpuStart        = m_samplerHeap->GetCPUDescriptorHandleForHeapStart();
  m_samplerGpuStart        = m_samplerHeap->GetGPUDescriptorHandleForHeapStart();

  m_resourceSlots          = {};
  m_resourceSlots.capacity = maxResourceDescriptors;
  // Slot 0 stays a null raw UAV so a shader address of 0 always means "no buffer".
  m_resourceSlots.base = 1;
  m_resourceSlots.next = 1;
  {
    D3D12_UNORDERED_ACCESS_VIEW_DESC nullUav{};
    nullUav.Format             = DXGI_FORMAT_R32_TYPELESS;
    nullUav.ViewDimension      = D3D12_UAV_DIMENSION_BUFFER;
    nullUav.Buffer.NumElements = 1;
    nullUav.Buffer.Flags       = D3D12_BUFFER_UAV_FLAG_RAW;
    device->CreateUnorderedAccessView(nullptr, nullptr, &nullUav, m_resourceCpuStart);
  }

  m_samplerSlots          = {};
  m_samplerSlots.capacity = maxSamplers;
  m_samplerByHash.clear();
  m_samplerHashBySlot.clear();
  return S_OK;
}

void DescriptorHeap::deinit()
{
  if(!isInitialized())
  {
    return;
  }
  std::lock_guard lock(m_mutex);
  // Leaks: slots and samplers acquired and never released.
  const uint32_t leakedSlots = m_resourceSlots.usedCount();
  if(leakedSlots != 0 || !m_samplerHashBySlot.empty())
  {
    LOGW("DescriptorHeap::deinit with %u resource slots and %zu samplers still acquired\n", leakedSlots,
         m_samplerHashBySlot.size());
  }
  m_resourceHeap.Reset();
  m_samplerHeap.Reset();
  m_resourceSlots = {};
  m_samplerSlots  = {};
  m_samplerByHash.clear();
  m_samplerHashBySlot.clear();
  {
    std::lock_guard bindLock(m_bindMutex);
    m_rootSignature = nullptr;
  }
  m_maxResourceDescriptors = 0;
  m_maxSamplers            = 0;
  m_device                 = nullptr;
}

//--------------------------------------------------------------------------------------------------
// Slots

ImageSlot DescriptorHeap::acquireImageSlot()
{
  return ImageSlot{acquireSlots(2)};
}

void DescriptorHeap::releaseImageSlot(ImageSlot slot)
{
  releaseSlots(slot.srv, 2);
}

uint32_t DescriptorHeap::acquireSlots(uint32_t count)
{
  if(count == 0)
  {
    return kInvalidSlot;
  }
  std::lock_guard lock(m_mutex);
  const uint32_t  slot = m_resourceSlots.allocate(count);
  if(slot == kInvalidSlot)
  {
    LOGE("DescriptorHeap: out of resource descriptors (%u)\n", m_maxResourceDescriptors);
  }
  return slot;
}

void DescriptorHeap::releaseSlots(uint32_t firstSlot, uint32_t count)
{
  if(firstSlot == kInvalidSlot || count == 0)
  {
    return;
  }
  bool released;
  {
    std::lock_guard lock(m_mutex);
    released = m_resourceSlots.release(firstSlot, count);
  }
  if(!released)
  {
    LOGE("DescriptorHeap::releaseSlots(%u, %u): slots not acquired (double release?), ignored\n", firstSlot, count);
    assert(false && "DescriptorHeap::releaseSlots: slots not acquired (double release?)");
  }
}

uint32_t DescriptorHeap::usedResourceDescriptors() const
{
  std::lock_guard lock(m_mutex);
  return m_resourceSlots.base + m_resourceSlots.usedCount();
}

uint32_t DescriptorHeap::usedSamplerDescriptors() const
{
  std::lock_guard lock(m_mutex);
  return static_cast<uint32_t>(m_samplerHashBySlot.size());
}

//--------------------------------------------------------------------------------------------------
// Image views

// Depth formats need their color-readable twin for SRVs.
static DXGI_FORMAT srvFormat(DXGI_FORMAT format)
{
  switch(format)
  {
    case DXGI_FORMAT_D32_FLOAT:
      return DXGI_FORMAT_R32_FLOAT;
    case DXGI_FORMAT_D16_UNORM:
      return DXGI_FORMAT_R16_UNORM;
    case DXGI_FORMAT_D24_UNORM_S8_UINT:
      return DXGI_FORMAT_R24_UNORM_X8_TYPELESS;
    case DXGI_FORMAT_D32_FLOAT_S8X24_UINT:
      return DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS;
    default:
      return format;
  }
}

static D3D12_SHADER_RESOURCE_VIEW_DESC makeImageSrv(const Image& image, DXGI_FORMAT viewFormat)
{
  D3D12_SHADER_RESOURCE_VIEW_DESC desc{};
  desc.Format                  = srvFormat(viewFormat);
  desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
  if(image.dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D)
  {
    desc.ViewDimension       = D3D12_SRV_DIMENSION_TEXTURE3D;
    desc.Texture3D.MipLevels = image.mipLevels;
  }
  else if(image.dimension == D3D12_RESOURCE_DIMENSION_TEXTURE1D)
  {
    if(image.arrayLayers > 1)
    {
      desc.ViewDimension            = D3D12_SRV_DIMENSION_TEXTURE1DARRAY;
      desc.Texture1DArray.MipLevels = image.mipLevels;
      desc.Texture1DArray.ArraySize = image.arrayLayers;
    }
    else
    {
      desc.ViewDimension       = D3D12_SRV_DIMENSION_TEXTURE1D;
      desc.Texture1D.MipLevels = image.mipLevels;
    }
  }
  else if(image.sampleCount > 1)
  {
    // Multisampled images have one mip level and no UAV.
    if(image.arrayLayers > 1)
    {
      desc.ViewDimension              = D3D12_SRV_DIMENSION_TEXTURE2DMSARRAY;
      desc.Texture2DMSArray.ArraySize = image.arrayLayers;
    }
    else
    {
      desc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2DMS;
    }
  }
  else if(image.arrayLayers > 1)
  {
    desc.ViewDimension            = D3D12_SRV_DIMENSION_TEXTURE2DARRAY;
    desc.Texture2DArray.MipLevels = image.mipLevels;
    desc.Texture2DArray.ArraySize = image.arrayLayers;
  }
  else
  {
    desc.ViewDimension       = D3D12_SRV_DIMENSION_TEXTURE2D;
    desc.Texture2D.MipLevels = image.mipLevels;
  }
  return desc;
}

static D3D12_UNORDERED_ACCESS_VIEW_DESC makeImageUav(const Image& image, uint32_t mipLevel, DXGI_FORMAT viewFormat)
{
  D3D12_UNORDERED_ACCESS_VIEW_DESC desc{};
  desc.Format = viewFormat;
  if(image.dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D)
  {
    desc.ViewDimension         = D3D12_UAV_DIMENSION_TEXTURE3D;
    desc.Texture3D.MipSlice    = mipLevel;
    desc.Texture3D.FirstWSlice = 0;
    desc.Texture3D.WSize       = ~0u;
  }
  else if(image.dimension == D3D12_RESOURCE_DIMENSION_TEXTURE1D)
  {
    if(image.arrayLayers > 1)
    {
      desc.ViewDimension            = D3D12_UAV_DIMENSION_TEXTURE1DARRAY;
      desc.Texture1DArray.MipSlice  = mipLevel;
      desc.Texture1DArray.ArraySize = image.arrayLayers;
    }
    else
    {
      desc.ViewDimension      = D3D12_UAV_DIMENSION_TEXTURE1D;
      desc.Texture1D.MipSlice = mipLevel;
    }
  }
  else if(image.arrayLayers > 1)
  {
    desc.ViewDimension            = D3D12_UAV_DIMENSION_TEXTURE2DARRAY;
    desc.Texture2DArray.MipSlice  = mipLevel;
    desc.Texture2DArray.ArraySize = image.arrayLayers;
  }
  else
  {
    desc.ViewDimension      = D3D12_UAV_DIMENSION_TEXTURE2D;
    desc.Texture2D.MipSlice = mipLevel;
  }
  return desc;
}

HRESULT DescriptorHeap::writeImageSlot(ImageSlot slot, const Image& image, uint32_t uavMipLevel, DXGI_FORMAT viewFormat)
{
  if(!slot.valid() || !image.image)
  {
    return E_INVALIDARG;
  }
  if(!image.allowsUav())
  {
    LOGE("writeImageSlot: image was created without D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS\n");
    return E_INVALIDARG;
  }
  NVDX12_FAIL_RETURN(writeSampledImageDescriptor(slot.srv, image, viewFormat));
  return writeStorageImageDescriptor(slot.uav(), image, uavMipLevel, viewFormat);
}

HRESULT DescriptorHeap::writeSampledImageDescriptor(uint32_t slot, const Image& image, DXGI_FORMAT viewFormat)
{
  if(!isWritableSlot(slot) || !image.image)
  {
    return E_INVALIDARG;
  }
  const D3D12_SHADER_RESOURCE_VIEW_DESC srv = makeImageSrv(image, viewFormat == DXGI_FORMAT_UNKNOWN ? image.format : viewFormat);
  m_device->CreateShaderResourceView(image.image.Get(), &srv, cpuHandle(slot));
  return S_OK;
}

HRESULT DescriptorHeap::writeStorageImageDescriptor(uint32_t slot, const Image& image, uint32_t mipLevel, DXGI_FORMAT viewFormat)
{
  if(!isWritableSlot(slot) || !image.image || mipLevel >= image.mipLevels)
  {
    return E_INVALIDARG;
  }
  if(!image.allowsUav() || image.sampleCount > 1)
  {
    LOGE("writeStorageImageDescriptor: image was created without D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS or is multisampled\n");
    return E_INVALIDARG;
  }
  const D3D12_UNORDERED_ACCESS_VIEW_DESC uav =
      makeImageUav(image, mipLevel, viewFormat == DXGI_FORMAT_UNKNOWN ? image.format : viewFormat);
  m_device->CreateUnorderedAccessView(image.image.Get(), nullptr, &uav, cpuHandle(slot));
  return S_OK;
}

HRESULT DescriptorHeap::writeDisplayImage(uint32_t slot, const Image& image)
{
  if(!isWritableSlot(slot) || !image.image)
  {
    return E_INVALIDARG;
  }
  D3D12_SHADER_RESOURCE_VIEW_DESC srv = makeImageSrv(image, image.format);
  srv.Shader4ComponentMapping         = D3D12_ENCODE_SHADER_4_COMPONENT_MAPPING(
      D3D12_SHADER_COMPONENT_MAPPING_FROM_MEMORY_COMPONENT_0, D3D12_SHADER_COMPONENT_MAPPING_FROM_MEMORY_COMPONENT_1,
      D3D12_SHADER_COMPONENT_MAPPING_FROM_MEMORY_COMPONENT_2, D3D12_SHADER_COMPONENT_MAPPING_FORCE_VALUE_1);
  m_device->CreateShaderResourceView(image.image.Get(), &srv, cpuHandle(slot));
  return S_OK;
}

//--------------------------------------------------------------------------------------------------
// Buffer views

HRESULT DescriptorHeap::writeBufferDescriptor(uint32_t slot, ID3D12Resource* resource, uint64_t byteOffset, uint64_t byteSize)
{
  if(!isWritableSlot(slot) || !resource)
  {
    return E_INVALIDARG;
  }
  if(byteOffset % D3D12_RAW_UAV_SRV_BYTE_ALIGNMENT != 0 || byteSize > (1ull << 32))
  {
    LOGE("writeBufferDescriptor: offset must be a multiple of 16 and the window at most 4 GiB\n");
    return E_INVALIDARG;
  }
  D3D12_UNORDERED_ACCESS_VIEW_DESC uav{};
  uav.Format              = DXGI_FORMAT_R32_TYPELESS;
  uav.ViewDimension       = D3D12_UAV_DIMENSION_BUFFER;
  uav.Buffer.FirstElement = byteOffset / 4;
  uav.Buffer.NumElements  = static_cast<UINT>(byteSize / 4);
  uav.Buffer.Flags        = D3D12_BUFFER_UAV_FLAG_RAW;
  if(uav.Buffer.NumElements == 0)
  {
    // Less than one dword: a view needs at least one element, so the slot
    // gets a null view (reads return 0, writes are dropped).
    uav.Buffer.FirstElement = 0;
    uav.Buffer.NumElements  = 1;
    resource                = nullptr;
  }
  m_device->CreateUnorderedAccessView(resource, nullptr, &uav, cpuHandle(slot));
  return S_OK;
}

HRESULT DescriptorHeap::writeAccelerationStructureSrv(uint32_t slot, D3D12_GPU_VIRTUAL_ADDRESS address)
{
  if(!isWritableSlot(slot) || address == 0)
  {
    return E_INVALIDARG;
  }
  D3D12_SHADER_RESOURCE_VIEW_DESC srv{};
  srv.Format                                   = DXGI_FORMAT_UNKNOWN;
  srv.ViewDimension                            = D3D12_SRV_DIMENSION_RAYTRACING_ACCELERATION_STRUCTURE;
  srv.Shader4ComponentMapping                  = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
  srv.RaytracingAccelerationStructure.Location = address;
  m_device->CreateShaderResourceView(nullptr, &srv, cpuHandle(slot));
  return S_OK;
}

//--------------------------------------------------------------------------------------------------
// Samplers

// FNV-1a over the struct bytes; D3D12_SAMPLER_DESC has no padding.
static uint64_t hashSamplerDesc(const D3D12_SAMPLER_DESC& desc)
{
  const uint8_t* bytes = reinterpret_cast<const uint8_t*>(&desc);
  uint64_t       hash  = 14695981039346656037ull;
  for(size_t i = 0; i < sizeof(desc); i++)
  {
    hash ^= bytes[i];
    hash *= 1099511628211ull;
  }
  return hash;
}

uint32_t DescriptorHeap::acquireSamplerDescriptor(const D3D12_SAMPLER_DESC& desc)
{
  const uint64_t  hash = hashSamplerDesc(desc);
  std::lock_guard lock(m_mutex);
  // A hash hit is only a candidate: entries with the same hash are told
  // apart by their full description.
  std::vector<SamplerEntry>& entries = m_samplerByHash[hash];
  for(size_t i = 0; i < entries.size(); i++)
  {
    SamplerEntry& entry = entries[i];
    if(memcmp(&entry.desc, &desc, sizeof(desc)) == 0)
    {
      entry.refCount++;
      return entry.slot;
    }
  }
  const uint32_t slot = m_samplerSlots.allocate(1);
  if(slot == kInvalidSlot)
  {
    if(entries.empty())
    {
      m_samplerByHash.erase(hash);
    }
    LOGE("DescriptorHeap: out of sampler descriptors (%u)\n", m_maxSamplers);
    return kInvalidSlot;
  }
  m_device->CreateSampler(&desc, samplerCpuHandle(slot));
  entries.push_back({desc, slot, 1});
  m_samplerHashBySlot[slot] = hash;
  return slot;
}

void DescriptorHeap::releaseSamplerDescriptor(uint32_t index)
{
  if(index == kInvalidSlot)
  {
    return;
  }
  std::unique_lock lock(m_mutex);
  auto             hashIt = m_samplerHashBySlot.find(index);
  if(hashIt == m_samplerHashBySlot.end())
  {
    lock.unlock();
    LOGE("DescriptorHeap::releaseSamplerDescriptor(%u): sampler not acquired (double release?), ignored\n", index);
    assert(false && "DescriptorHeap::releaseSamplerDescriptor: sampler not acquired (double release?)");
    return;
  }
  auto bucketIt = m_samplerByHash.find(hashIt->second);
  assert(bucketIt != m_samplerByHash.end() && "Inconsistent sampler state");
  if(bucketIt == m_samplerByHash.end())
  {
    return;
  }
  std::vector<SamplerEntry>& entries = bucketIt->second;
  auto entryIt = std::find_if(entries.begin(), entries.end(), [&](const SamplerEntry& e) { return e.slot == index; });
  assert(entryIt != entries.end() && "Inconsistent sampler state");
  if(entryIt != entries.end() && --entryIt->refCount == 0)
  {
    entries.erase(entryIt);
    if(entries.empty())
    {
      m_samplerByHash.erase(bucketIt);
    }
    m_samplerHashBySlot.erase(hashIt);
    [[maybe_unused]] const bool released = m_samplerSlots.release(index, 1);
    assert(released && "Inconsistent sampler state");
  }
}

HRESULT DescriptorHeap::writeSamplerDescriptor(uint32_t index, const D3D12_SAMPLER_DESC& desc)
{
  if(index >= m_maxSamplers)
  {
    return E_INVALIDARG;
  }
  m_device->CreateSampler(&desc, samplerCpuHandle(index));
  return S_OK;
}

bool DescriptorHeap::isSupported(ID3D12Device* device)
{
  if(device == nullptr)
  {
    return false;
  }
  D3D12_FEATURE_DATA_D3D12_OPTIONS options{};
  D3D12_FEATURE_DATA_SHADER_MODEL  shaderModel{D3D_SHADER_MODEL_6_6};
  return SUCCEEDED(device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS, &options, sizeof(options)))
         && options.ResourceBindingTier >= D3D12_RESOURCE_BINDING_TIER_3
         && SUCCEEDED(device->CheckFeatureSupport(D3D12_FEATURE_SHADER_MODEL, &shaderModel, sizeof(shaderModel)))
         && shaderModel.HighestShaderModel >= D3D_SHADER_MODEL_6_6;
}

//--------------------------------------------------------------------------------------------------
// Per command list

void DescriptorHeap::cmdBind(ID3D12GraphicsCommandList* cmd) const
{
  // Copy lists take neither descriptor heaps nor root signatures.
  if(cmd->GetType() == D3D12_COMMAND_LIST_TYPE_COPY)
  {
    return;
  }
  ID3D12DescriptorHeap* heaps[] = {m_resourceHeap.Get(), m_samplerHeap.Get()};
  cmd->SetDescriptorHeaps(2, heaps);
  // Recording commands under the lock is cheap (no GPU work) and keeps the
  // root signature consistent with a concurrent setRootSignature.
  std::lock_guard lock(m_bindMutex);
  if(m_rootSignature)
  {
    // A direct list takes the same root signature for graphics; pushes name their bind point (BindPoint::eBoth serves a
    // dispatch and a draw).
    const bool graphics = cmd->GetType() == D3D12_COMMAND_LIST_TYPE_DIRECT;
    cmd->SetComputeRootSignature(m_rootSignature);
    if(graphics)
    {
      cmd->SetGraphicsRootSignature(m_rootSignature);
    }
  }
}

void DescriptorHeap::setRootSignature(ID3D12RootSignature* rootSignature)
{
  std::lock_guard lock(m_bindMutex);
  m_rootSignature = rootSignature;
}

//--------------------------------------------------------------------------------------------------
// Handles

D3D12_GPU_DESCRIPTOR_HANDLE DescriptorHeap::gpuHandle(uint32_t slot) const
{
  return D3D12_GPU_DESCRIPTOR_HANDLE{m_resourceGpuStart.ptr + UINT64(slot) * m_resourceDescriptorSize};
}

D3D12_CPU_DESCRIPTOR_HANDLE DescriptorHeap::cpuHandle(uint32_t slot) const
{
  return D3D12_CPU_DESCRIPTOR_HANDLE{m_resourceCpuStart.ptr + SIZE_T(slot) * m_resourceDescriptorSize};
}

uint32_t DescriptorHeap::slotFromGpuHandle(D3D12_GPU_DESCRIPTOR_HANDLE handle) const
{
  if(m_resourceDescriptorSize == 0 || handle.ptr < m_resourceGpuStart.ptr)
  {
    return kInvalidSlot;
  }
  const UINT64 offset = handle.ptr - m_resourceGpuStart.ptr;
  const UINT64 slot   = offset / m_resourceDescriptorSize;
  if(slot >= m_maxResourceDescriptors || offset % m_resourceDescriptorSize != 0)
  {
    return kInvalidSlot;
  }
  return static_cast<uint32_t>(slot);
}

D3D12_GPU_DESCRIPTOR_HANDLE DescriptorHeap::samplerGpuHandle(uint32_t slot) const
{
  return D3D12_GPU_DESCRIPTOR_HANDLE{m_samplerGpuStart.ptr + UINT64(slot) * m_samplerDescriptorSize};
}

D3D12_CPU_DESCRIPTOR_HANDLE DescriptorHeap::samplerCpuHandle(uint32_t slot) const
{
  return D3D12_CPU_DESCRIPTOR_HANDLE{m_samplerCpuStart.ptr + SIZE_T(slot) * m_samplerDescriptorSize};
}

}  // namespace nvdx12

//--------------------------------------------------------------------------------------------------
// Usage example
//--------------------------------------------------------------------------------------------------
[[maybe_unused]] static void usage_DescriptorHeap()
{
  ID3D12Device*               device        = nullptr;  // EX: from nvdx12::Context::getDevice()
  ID3D12RootSignature*        rootSignature = nullptr;  // EX: from nvdx12::createBindlessRootSignature
  ID3D12GraphicsCommandList7* cmd           = nullptr;  // EX: the frame's command list
  nvdx12::Image               storageImage;             // EX: created with ALLOW_UNORDERED_ACCESS by the allocator
  nvdx12::Image               texture;                  // EX: a sampled texture

  // Directly indexed heaps: resource binding tier 3 and Shader Model 6.6.
  assert(nvdx12::DescriptorHeap::isSupported(device));
  nvdx12::DescriptorHeap heap;
  NVDX12_CHECK(heap.init(device));
  heap.setRootSignature(rootSignature);

  // Image pair: shaders receive `storageSlot.srv` and read HEAP_SRV(slot) as a
  // Texture2D and HEAP_UAV(slot) as an RWTexture2D of mip 0 (shaders/bindless.h).
  const nvdx12::ImageSlot storageSlot = heap.acquireImageSlot();
  NVDX12_CHECK(heap.writeImageSlot(storageSlot, storageImage));

  // SRV only, on a single slot; samplers are deduplicated, so equal
  // descriptions share a slot (HEAP_SAMPLER(slot)).
  const uint32_t textureSlot = heap.acquireSlots();
  NVDX12_CHECK(heap.writeSampledImageDescriptor(textureSlot, texture));
  D3D12_SAMPLER_DESC samplerDesc{};  // EX: fill in, or use nvdx12::makeSamplerDesc / DEFAULT_D3D12_SAMPLER_DESC
  const uint32_t     samplerSlot = heap.acquireSamplerDescriptor(samplerDesc);

  // Per command list: the heaps and the root signature (compute, and graphics on direct lists).
  heap.cmdBind(cmd);
  // ... SetPipelineState, push constants carrying the slots, Dispatch ...

  // ImGui draws the image through its GPU handle.
  // ImGui::Image(ImTextureID(heap.gpuHandle(textureSlot).ptr), size);
  // A descriptor-free callback that only gets the handle back maps it to
  // the slot it releases.
  const uint32_t imguiSlot = heap.acquireSlots();  // EX: ImGui's SrvDescriptorAllocFn
  assert(heap.slotFromGpuHandle(heap.gpuHandle(imguiSlot)) == imguiSlot);
  heap.releaseSlots(heap.slotFromGpuHandle(heap.gpuHandle(imguiSlot)));  // EX: SrvDescriptorFreeFn

  heap.releaseSamplerDescriptor(samplerSlot);
  heap.releaseSlots(textureSlot);
  heap.releaseImageSlot(storageSlot);
  heap.deinit();
}
