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

#include "builtin_kernels.hpp"

#include <algorithm>
#include <cassert>
#include <filesystem>
#include <string>
#include <string_view>

#include <nvutils/file_operations.hpp>
#include <nvutils/logger.hpp>

#include "barriers.hpp"
#include "check_error.hpp"
#include "debug_util.hpp"
#include "pipeline.hpp"
#include "shader_compiler.hpp"

namespace nvdx12 {

namespace {

// The C++ argument structs below are bit-identical to the HLSL ones in the
// kernel sources. Every kernel reads them with PUSH_DATA of the nvdx12 shader
// contract, from the root SRV the bindless root signature reserves for push
// data. The header is found through the search paths of the compiler, the
// nvpro_core2 root (NVDX12_SHADER_INCLUDE_DIR, set by nvdx12/CMakeLists.txt)
// or the executable folder, for installed layouts that ship nvdx12/shaders
// next to the executable.
constexpr std::string_view kKernelPrologue = R"hlsl(
#include <nvdx12/shaders/bindless.h>
)hlsl";

// Fill: each thread writes four consecutive dwords; the group grid is
// (groupsX, ceil(groups / groupsX)) so fills above 65535 groups fit.
struct FillBufferArgs
{
  uint32_t uavSlot;     // raw UAV of the 4 GiB window being filled
  uint32_t byteOffset;  // inside the window, multiple of 4
  uint32_t dwordCount;
  uint32_t value;
  uint32_t groupsX;
};
constexpr uint32_t kFillBufferThreads   = 256;
constexpr uint32_t kFillBufferPerThread = 4;

constexpr std::string_view kFillBufferSource = R"hlsl(
struct FillBufferArgs
{
  uint uavSlot;
  uint byteOffset;
  uint dwordCount;
  uint value;
  uint groupsX;
};
PUSH_DATA(FillBufferArgs, args);

[numthreads(256, 1, 1)]
void main(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
  uint thread = (gid.y * args.groupsX + gid.x) * 256u + gtid.x;
  uint dword  = thread * 4u;
  if(dword >= args.dwordCount)
    return;
  RWByteAddressBuffer buf = ResourceDescriptorHeap[args.uavSlot];
  uint address            = args.byteOffset + dword * 4u;
  if(dword + 4u <= args.dwordCount)
  {
    buf.Store4(address, uint4(args.value, args.value, args.value, args.value));
  }
  else
  {
    for(uint i = dword; i < args.dwordCount; i++)
    {
      buf.Store(address, args.value);
      address += 4u;
    }
  }
}
)hlsl";

// Blit: SRV of the source at `srcSlot` (whole mip chain; the kernel loads
// `srcMip`), UAV of the destination level at `dstSlot`. The blit writes the
// dstWidth x dstHeight rectangle at (dstX, dstY) and nothing else.
struct BlitImageArgs
{
  uint32_t srcSlot;
  uint32_t dstSlot;
  uint32_t srcMip;
  uint32_t filter;  // 0 nearest, 1 bilinear
  uint32_t srcWidth;
  uint32_t srcHeight;
  uint32_t dstWidth;
  uint32_t dstHeight;
  uint32_t dstX;
  uint32_t dstY;
};
constexpr uint32_t kBlitImageThreads = 8;  // 8x8 groups

// Nearest or bilinear fetch of the source level at a normalized coordinate,
// with clamped texel loads: vkCmdBlitImage's VK_FILTER_NEAREST / LINEAR
// with clamp-to-edge.
constexpr std::string_view kBlitCommon = R"hlsl(
struct BlitImageArgs
{
  uint srcSlot;
  uint dstSlot;
  uint srcMip;
  uint filter;
  uint srcWidth;
  uint srcHeight;
  uint dstWidth;
  uint dstHeight;
  uint dstX;
  uint dstY;
};
PUSH_DATA(BlitImageArgs, args);

float4 loadClamped(Texture2D<float4> src, int2 coord)
{
  coord = clamp(coord, int2(0, 0), int2(int(args.srcWidth) - 1, int(args.srcHeight) - 1));
  return src.Load(int3(coord, int(args.srcMip)));
}

float4 sampleBilinear(Texture2D<float4> src, float2 uv)
{
  float2 p  = uv * float2(args.srcWidth, args.srcHeight) - 0.5;
  float2 f  = frac(p);
  int2   c0 = int2(floor(p));
  float4 a  = loadClamped(src, c0);
  float4 b  = loadClamped(src, c0 + int2(1, 0));
  float4 c  = loadClamped(src, c0 + int2(0, 1));
  float4 d  = loadClamped(src, c0 + int2(1, 1));
  return lerp(lerp(a, b, f.x), lerp(c, d, f.x), f.y);
}
)hlsl";

// The whole source level maps onto the destination rectangle; dtid is the
// pixel inside the rectangle.
constexpr std::string_view kBlitBody = R"hlsl(
[numthreads(8, 8, 1)]
void main(uint3 dtid : SV_DispatchThreadID)
{
  if(dtid.x >= args.dstWidth || dtid.y >= args.dstHeight)
    return;
  Texture2D<float4>   src = ResourceDescriptorHeap[args.srcSlot];
  RWTexture2D<float4> dst = ResourceDescriptorHeap[args.dstSlot];
  float2 uv = (float2(dtid.xy) + 0.5) / float2(args.dstWidth, args.dstHeight);
  float4 value;
  if(args.filter == 0)
  {
    int2 coord = int2(floor(uv * float2(args.srcWidth, args.srcHeight)));
    value      = loadClamped(src, coord);
  }
  else
  {
    value = sampleBilinear(src, uv);
  }
  dst[dtid.xy + uint2(args.dstX, args.dstY)] = value;
}
)hlsl";

// Mipmap: one level of every layer from the previous one, SRV of level
// `mip - 1` at `srcSlot`, UAV of level `mip` at `dstSlot` (array views, so
// one kernel serves single images and arrays; MIP_1D selects the 1D views).
struct MipmapArgs
{
  uint32_t srcSlot;
  uint32_t dstSlot;
  uint32_t srcWidth;
  uint32_t srcHeight;
  uint32_t dstWidth;
  uint32_t dstHeight;
};

// Each destination texel is the average of the source texels its footprint
// covers, weighted by the covered area (a box filter that is exact for any
// size). Per axis the next level is floor(size / 2), so a texel covers
// either 1 source texel (size 1), 2 (even size), or, for an odd size
// 2n + 1, 2 + 1/n texels: texels 2x, 2x + 1 and 2x + 2 with weights
// (n - x), n and (x + 1), over 2n + 1. The 2x2 box of even sizes is the
// same as a bilinear sample at the texel center (nvvk's linear blit); on odd
// sizes no source row or column is dropped and the level's mean is kept.
constexpr std::string_view kMipmapSource = R"hlsl(
struct MipmapArgs
{
  uint srcSlot;
  uint dstSlot;
  uint srcWidth;
  uint srcHeight;
  uint dstWidth;
  uint dstHeight;
};
PUSH_DATA(MipmapArgs, args);

// Footprint of destination texel `d` along one axis: first source texel,
// tap count and weights.
uint footprint(uint d, uint srcSize, uint dstSize, out float w[3], out int first)
{
  if(srcSize == dstSize)  // 1 -> 1
  {
    first = int(d);
    w[0] = 1.0; w[1] = 0.0; w[2] = 0.0;
    return 1;
  }
  first = int(2 * d);
  if(srcSize == 2 * dstSize)
  {
    w[0] = 0.5; w[1] = 0.5; w[2] = 0.0;
    return 2;
  }
  // srcSize == 2 * dstSize + 1
  const float inv = 1.0 / float(srcSize);
  w[0] = float(dstSize - d) * inv;
  w[1] = float(dstSize) * inv;
  w[2] = float(d + 1) * inv;
  return 3;
}

#if MIP_1D
[numthreads(64, 1, 1)]
#else
[numthreads(8, 8, 1)]
#endif
void main(uint3 dtid : SV_DispatchThreadID)
{
  if(dtid.x >= args.dstWidth || dtid.y >= args.dstHeight)
    return;
#if MIP_1D
  Texture1DArray<float4>   src = ResourceDescriptorHeap[args.srcSlot];
  RWTexture1DArray<float4> dst = ResourceDescriptorHeap[args.dstSlot];
#else
  Texture2DArray<float4>   src = ResourceDescriptorHeap[args.srcSlot];
  RWTexture2DArray<float4> dst = ResourceDescriptorHeap[args.dstSlot];
#endif
  float wx[3], wy[3];
  int   x0, y0;
  const uint nx = footprint(dtid.x, args.srcWidth, args.dstWidth, wx, x0);
  const uint ny = footprint(dtid.y, args.srcHeight, args.dstHeight, wy, y0);
  float4 sum = float4(0.0, 0.0, 0.0, 0.0);
  for(uint j = 0; j < ny; j++)
  {
    float4 row = float4(0.0, 0.0, 0.0, 0.0);
    for(uint i = 0; i < nx; i++)
    {
#if MIP_1D
      row += wx[i] * src.Load(int3(x0 + int(i), int(dtid.z), 0));
#else
      row += wx[i] * src.Load(int4(x0 + int(i), y0 + int(j), int(dtid.z), 0));
#endif
    }
    sum += wy[j] * row;
  }
#if MIP_1D
  dst[uint2(dtid.x, dtid.z)] = sum;
#else
  dst[dtid] = sum;
#endif
}
)hlsl";

ComPtr<ID3D12PipelineState> compileKernel(const BuiltinKernels::InitInfo& info,
                                          std::string_view                source,
                                          const char*                     name,
                                          std::string_view                defines = {})
{
  const std::string  full          = std::string(defines) + std::string(kKernelPrologue) + std::string(source);
  const std::wstring includeArgs[] = {L"-I", std::filesystem::path(NVDX12_SHADER_INCLUDE_DIR).wstring(), L"-I",
                                      nvutils::getExecutablePath().parent_path().wstring()};
  ShaderBlob         blob          = info.compiler->compileSource(full, name, "cs_6_6", "main", {}, includeArgs);
  if(!blob.valid())
  {
    LOGE("Built-in kernel %s failed to compile\n", name);
    return nullptr;
  }
  ComPtr<ID3D12PipelineState> pso;
  if(FAILED(createComputePipeline(pso, info.device, info.rootSignature, blob, info.pipelineFlags)))
  {
    return nullptr;
  }
  setDebugName(pso.Get(), name);
  return pso;
}

}  // namespace

//--------------------------------------------------------------------------------------------------
// Lifecycle

HRESULT BuiltinKernels::init(const InitInfo& info)
{
  assert(m_device == nullptr && "init() called twice; missing deinit()?");
  if(!info.device || !info.compiler || !info.rootSignature || !info.uploadRing || !info.uploadRing->getQueueTimeline())
  {
    LOGE("BuiltinKernels::init: device, compiler, rootSignature and an initialized uploadRing are required\n");
    return E_INVALIDARG;
  }
  m_device                = info.device;
  m_heap                  = info.descriptorHeap;
  m_ring                  = info.uploadRing;
  m_queue                 = info.uploadRing->getQueueTimeline();
  m_rootSignature         = info.rootSignature;
  m_pushDataRootParameter = info.pushDataRootParameter;
  m_submitListener = m_queue->addSubmitListener([this](std::span<ID3D12CommandList* const> lists, uint64_t fenceValue) {
    std::lock_guard lock(m_pendingMutex);
    for(size_t i = 0; i < m_pendingSlots.size(); i++)
    {
      PendingSlots& pending = m_pendingSlots[i];
      if(pending.cmd != nullptr && std::find(lists.begin(), lists.end(), pending.cmd) != lists.end())
      {
        pending.cmd        = nullptr;
        pending.fenceValue = fenceValue;
      }
    }
  });

  m_fillBuffer = compileKernel(info, kFillBufferSource, "nvdx12 fill buffer");
  m_blitImage  = compileKernel(info, std::string(kBlitCommon) + std::string(kBlitBody), "nvdx12 blit image");
  m_mipmap     = compileKernel(info, kMipmapSource, "nvdx12 mipmap", "#define MIP_1D 0\n");
  m_mipmap1D   = compileKernel(info, kMipmapSource, "nvdx12 mipmap 1D", "#define MIP_1D 1\n");
  if(!m_fillBuffer || !m_blitImage || !m_mipmap || !m_mipmap1D)
  {
    deinit();  // releases the kernels that did compile
    return E_FAIL;
  }
  return S_OK;
}

void BuiltinKernels::deinit()
{
  if(m_queue != nullptr)
  {
    m_queue->removeSubmitListener(m_submitListener);
  }
  releaseCompletedSlots(true);
  m_queue          = nullptr;
  m_submitListener = 0;
  m_fillBuffer.Reset();
  m_blitImage.Reset();
  m_mipmap.Reset();
  m_mipmap1D.Reset();
  m_device        = nullptr;
  m_heap          = nullptr;
  m_ring          = nullptr;
  m_rootSignature = nullptr;
  m_releaseSlots  = {};
}

// The heap's binding may carry another root signature than the one the
// kernels were built against; setting the same one again keeps its bindings.
void BuiltinKernels::cmdBind(ID3D12GraphicsCommandList* cmd)
{
  m_heap->cmdBind(cmd);
  cmd->SetComputeRootSignature(m_rootSignature);
}

bool BuiltinKernels::cmdPushData(ID3D12GraphicsCommandList* cmd, const void* data, uint32_t size)
{
  return m_ring->cmdPushData(cmd, data, size, BindPoint::eCompute, m_pushDataRootParameter);
}

void BuiltinKernels::releaseSlots(ID3D12CommandList* cmd, uint32_t firstSlot, uint32_t count)
{
  if(m_releaseSlots)
  {
    m_releaseSlots(cmd, firstSlot, count);
    return;
  }
  // Held until the list's submission completed, so that no list recorded meanwhile reuses them.
  std::lock_guard lock(m_pendingMutex);
  m_pendingSlots.push_back({.cmd = cmd, .firstSlot = firstSlot, .count = count});
}

void BuiltinKernels::releaseCompletedSlots(bool all)
{
  const uint64_t  completed = (all || m_queue == nullptr) ? 0 : m_queue->completedValue();
  std::lock_guard lock(m_pendingMutex);
  for(size_t i = 0; i < m_pendingSlots.size();)
  {
    const PendingSlots& pending = m_pendingSlots[i];
    if(all || (pending.cmd == nullptr && pending.fenceValue <= completed))
    {
      if(m_heap != nullptr)
      {
        m_heap->releaseSlots(pending.firstSlot, pending.count);
      }
      m_pendingSlots[i] = m_pendingSlots.back();
      m_pendingSlots.pop_back();
    }
    else
    {
      i++;
    }
  }
}

//--------------------------------------------------------------------------------------------------
// Commands

static bool kernelImageSupported(const Image& image, const char* what)
{
  if(image.dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D || image.arrayLayers != 1 || image.sampleCount > 1)
  {
    LOGE("%s: only single-layer, single-sample 2D images are supported\n", what);
    return false;
  }
  return true;
}

// The kernels read float4 through an SRV and, for `store`, write float4
// through a typed UAV: float, UNORM and SNORM formats with typed UAV
// stores. sRGB formats have no typed UAV (D3D12 cannot create the image
// with ALLOW_UNORDERED_ACCESS), integer formats are not filterable.
static bool kernelFormatSupported(ID3D12Device* device, DXGI_FORMAT format, bool store, const char* what)
{
  D3D12_FEATURE_DATA_FORMAT_SUPPORT support{format, D3D12_FORMAT_SUPPORT1_NONE, D3D12_FORMAT_SUPPORT2_NONE};
  const bool ok = device && SUCCEEDED(device->CheckFeatureSupport(D3D12_FEATURE_FORMAT_SUPPORT, &support, sizeof(support)))
                  && (support.Support1 & D3D12_FORMAT_SUPPORT1_SHADER_SAMPLE) != 0
                  && (!store || (support.Support2 & D3D12_FORMAT_SUPPORT2_UAV_TYPED_STORE) != 0);
  if(!ok)
  {
    LOGE("%s: DXGI format %d is not supported (float, UNORM or SNORM formats only%s)\n", what, int(format),
         store ? ", with typed UAV stores: no sRGB" : "");
  }
  return ok;
}

void BuiltinKernels::cmdFillBuffer(ID3D12GraphicsCommandList* cmd, const Buffer& dst, uint64_t offset, uint64_t size, uint32_t value)
{
  if(!dst.buffer)
  {
    return;
  }
  if(size == ~0ull)
  {
    size = dst.bufferSize - std::min(offset, dst.bufferSize);
  }
  if(size == 0)
  {
    return;
  }
  if(dst.uavSlot == kInvalidSlot)
  {
    LOGE("cmdFillBuffer: the buffer has no raw UAV (created without D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS)\n");
    return;
  }
  assert((offset & 3) == 0 && (size & 3) == 0 && offset + size <= dst.bufferSize);
  assert(m_heap && "setDescriptorHeap was not called");

  cmdBind(cmd);
  cmd->SetPipelineState(m_fillBuffer.Get());
  constexpr uint64_t kWindow         = 1ull << 32;
  constexpr uint32_t kDwordsPerGroup = kFillBufferThreads * kFillBufferPerThread;
  const uint64_t     end             = offset + size;
  while(offset < end)
  {
    // One dispatch per 4 GiB window: the kernel addresses one raw UAV with 32-bit offsets.
    const uint32_t window    = static_cast<uint32_t>(offset / kWindow);
    const uint64_t windowEnd = std::min(end, uint64_t(window + 1) * kWindow);
    const uint32_t dwords    = static_cast<uint32_t>((windowEnd - offset) / 4);
    const uint32_t groups    = (dwords + kDwordsPerGroup - 1) / kDwordsPerGroup;
    const uint32_t groupsX   = std::min<uint32_t>(groups, D3D12_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION);
    const uint32_t groupsY   = (groups + groupsX - 1) / groupsX;
    FillBufferArgs args{};
    args.uavSlot    = dst.uavSlot + window;
    args.byteOffset = static_cast<uint32_t>(offset % kWindow);
    args.dwordCount = dwords;
    args.value      = value;
    args.groupsX    = groupsX;
    if(!cmdPushData(cmd, &args, sizeof(args)))
    {
      return;
    }
    cmd->Dispatch(groupsX, groupsY, 1);
    offset = windowEnd;
  }
}

void BuiltinKernels::cmdBlitImage(ID3D12GraphicsCommandList7* cmd, Image& src, Image& dst, Filter filter, const Rect2D& dstRect)
{
  if(!src.image || !dst.image)
  {
    return;
  }
  if(!kernelImageSupported(src, "cmdBlitImage") || !kernelImageSupported(dst, "cmdBlitImage"))
  {
    return;
  }
  if(!dst.allowsUav())
  {
    LOGE("cmdBlitImage: the destination must allow unordered access\n");
    return;
  }
  if(src.image == dst.image)
  {
    LOGE("cmdBlitImage: source and destination must be different images\n");
    return;
  }
  if(!kernelFormatSupported(m_device, src.format, false, "cmdBlitImage source")
     || !kernelFormatSupported(m_device, dst.format, true, "cmdBlitImage destination"))
  {
    return;
  }
  // A zero extent selects the whole destination level.
  Rect2D rect = dstRect;
  if(rect.width == 0 || rect.height == 0)
  {
    rect = {0, 0, dst.extent.width, dst.extent.height};
  }
  if(rect.x >= dst.extent.width || rect.y >= dst.extent.height || rect.width > dst.extent.width - rect.x
     || rect.height > dst.extent.height - rect.y)
  {
    LOGE("cmdBlitImage: destination rectangle (%u, %u, %u x %u) exceeds the %u x %u destination\n", rect.x, rect.y,
         rect.width, rect.height, dst.extent.width, dst.extent.height);
    return;
  }
  assert(m_heap && "setDescriptorHeap was not called");
  releaseCompletedSlots();
  // Taken before any barrier, so that a full heap records nothing.
  const uint32_t slots = m_heap->acquireSlots(2);
  if(slots == kInvalidSlot)
  {
    return;
  }
  cmdBind(cmd);
  cmdImageBarrier(cmd, src, D3D12_BARRIER_LAYOUT_SHADER_RESOURCE);
  // Texels outside the rectangle keep their content: the destination is only
  // discarded (cmdImageBarrier from UNDEFINED) when the blit overwrites all
  // of it, i.e. the whole of its only subresource.
  const bool wholeImage = rect.x == 0 && rect.y == 0 && rect.width == dst.extent.width
                          && rect.height == dst.extent.height && dst.mipLevels == 1;
  if(wholeImage)
  {
    cmdImageBarrier(cmd, dst, D3D12_BARRIER_LAYOUT_UNORDERED_ACCESS);
  }
  else if(dst.layout != D3D12_BARRIER_LAYOUT_UNORDERED_ACCESS)
  {
    cmdImageMemoryBarrier(cmd, dst, {.newLayout = D3D12_BARRIER_LAYOUT_UNORDERED_ACCESS});  // no DISCARD flag
  }

  if(FAILED(NVDX12_FAIL_REPORT(m_heap->writeSampledImageDescriptor(slots, src)))
     || FAILED(NVDX12_FAIL_REPORT(m_heap->writeStorageImageDescriptor(slots + 1, dst, 0))))
  {
    releaseSlots(cmd, slots, 2);
    return;
  }

  BlitImageArgs args{};
  args.srcSlot   = slots;
  args.dstSlot   = slots + 1;
  args.srcMip    = 0;
  args.filter    = filter == Filter::eLinear ? 1 : 0;
  args.srcWidth  = src.extent.width;
  args.srcHeight = src.extent.height;
  args.dstWidth  = rect.width;
  args.dstHeight = rect.height;
  args.dstX      = rect.x;
  args.dstY      = rect.y;
  cmd->SetPipelineState(m_blitImage.Get());
  if(cmdPushData(cmd, &args, sizeof(args)))
  {
    const uint32_t t = kBlitImageThreads;
    cmd->Dispatch((rect.width + t - 1) / t, (rect.height + t - 1) / t, 1);
  }
  releaseSlots(cmd, slots, 2);
}

void BuiltinKernels::cmdGenerateMipmaps(ID3D12GraphicsCommandList7* cmd, Image& image)
{
  if(!image.image)
  {
    return;
  }
  if(image.mipLevels <= 1)
  {
    cmdImageBarrier(cmd, image, D3D12_BARRIER_LAYOUT_SHADER_RESOURCE);
    return;
  }
  const bool is1D = image.dimension == D3D12_RESOURCE_DIMENSION_TEXTURE1D;
  if((!is1D && image.dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D) || image.sampleCount > 1)
  {
    LOGE("cmdGenerateMipmaps: only single-sample 1D and 2D images (and arrays) are supported\n");
    return;
  }
  if(!image.allowsUav())
  {
    LOGE("cmdGenerateMipmaps: the image must allow unordered access\n");
    return;
  }
  if(!kernelFormatSupported(m_device, image.format, true, "cmdGenerateMipmaps"))
  {
    return;
  }
  assert(m_heap && "setDescriptorHeap was not called");
  // Per generated level i: an SRV of level i - 1 and a UAV of level i, over
  // every layer. Taken before any barrier, so that a full heap records nothing.
  const uint32_t levels = image.mipLevels - 1;
  releaseCompletedSlots();
  const uint32_t slots = m_heap->acquireSlots(2 * levels);
  if(slots == kInvalidSlot)
  {
    return;
  }
  cmdBind(cmd);

  const uint32_t layers = image.arrayLayers;
  for(uint32_t mip = 1; mip < image.mipLevels; mip++)
  {
    D3D12_SHADER_RESOURCE_VIEW_DESC  srv{};
    D3D12_UNORDERED_ACCESS_VIEW_DESC uav{};
    srv.Format                  = image.format;
    srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    uav.Format                  = image.format;
    if(is1D)
    {
      srv.ViewDimension  = D3D12_SRV_DIMENSION_TEXTURE1DARRAY;
      srv.Texture1DArray = {.MostDetailedMip = mip - 1, .MipLevels = 1, .FirstArraySlice = 0, .ArraySize = layers};
      uav.ViewDimension  = D3D12_UAV_DIMENSION_TEXTURE1DARRAY;
      uav.Texture1DArray = {.MipSlice = mip, .FirstArraySlice = 0, .ArraySize = layers};
    }
    else
    {
      srv.ViewDimension  = D3D12_SRV_DIMENSION_TEXTURE2DARRAY;
      srv.Texture2DArray = {.MostDetailedMip = mip - 1, .MipLevels = 1, .FirstArraySlice = 0, .ArraySize = layers};
      uav.ViewDimension  = D3D12_UAV_DIMENSION_TEXTURE2DARRAY;
      uav.Texture2DArray = {.MipSlice = mip, .FirstArraySlice = 0, .ArraySize = layers};
    }
    m_device->CreateShaderResourceView(image.image.Get(), &srv, m_heap->cpuHandle(slots + 2 * (mip - 1)));
    m_device->CreateUnorderedAccessView(image.image.Get(), nullptr, &uav, m_heap->cpuHandle(slots + 2 * (mip - 1) + 1));
  }

  // Mip 0 is read, the others written; from here on layouts are per level.
  // Every level ends in SHADER_RESOURCE with that layout's whole scope
  // (every shader stage, as nvvk::cmdGenerateMipmaps ends in
  // inferPipelineStageAccessTuple(currentLayout)), so that draws, dispatches
  // and DispatchRays may read any level afterwards.
  const auto [readSync, readAccess]       = layoutDestinationScope(D3D12_BARRIER_LAYOUT_SHADER_RESOURCE);
  const auto [srcSync, srcAccess]         = imageSourceScope(image);
  constexpr D3D12_BARRIER_SYNC kWriteSync = D3D12_BARRIER_SYNC_COMPUTE_SHADING;
  {
    BarrierContainer barriers;
    barriers.imageBarriers.push_back(makeImageMemoryBarrier({.image            = image.image.Get(),
                                                             .oldLayout        = image.layout,
                                                             .newLayout        = D3D12_BARRIER_LAYOUT_SHADER_RESOURCE,
                                                             .subresourceRange = {0, 1, 0, layers, 0, 1},
                                                             .srcStageMask     = uint64_t(srcSync),
                                                             .dstStageMask     = uint64_t(readSync),
                                                             .srcAccessMask    = uint64_t(srcAccess),
                                                             .dstAccessMask    = uint64_t(readAccess)}));
    barriers.imageBarriers.push_back(makeImageMemoryBarrier({.image            = image.image.Get(),
                                                             .oldLayout        = image.layout,
                                                             .newLayout        = D3D12_BARRIER_LAYOUT_UNORDERED_ACCESS,
                                                             .subresourceRange = {1, levels, 0, layers, 0, 1},
                                                             .srcStageMask     = uint64_t(srcSync),
                                                             .dstStageMask     = kWriteSync,
                                                             .srcAccessMask    = uint64_t(srcAccess),
                                                             .dstAccessMask = D3D12_BARRIER_ACCESS_UNORDERED_ACCESS}));
    barriers.cmdPipelineBarrier(cmd);
  }

  cmd->SetPipelineState(is1D ? m_mipmap1D.Get() : m_mipmap.Get());
  const uint32_t groupX = is1D ? 64 : kBlitImageThreads;
  const uint32_t groupY = is1D ? 1 : kBlitImageThreads;
  bool           built  = true;  // each level is built from the previous one
  for(uint32_t mip = 1; mip < image.mipLevels; mip++)
  {
    MipmapArgs args{};
    args.srcSlot   = slots + 2 * (mip - 1);
    args.dstSlot   = slots + 2 * (mip - 1) + 1;
    args.srcWidth  = std::max(1u, image.extent.width >> (mip - 1));
    args.srcHeight = is1D ? 1u : std::max(1u, image.extent.height >> (mip - 1));
    args.dstWidth  = std::max(1u, image.extent.width >> mip);
    args.dstHeight = is1D ? 1u : std::max(1u, image.extent.height >> mip);
    // Without push data this level and the ones after it are not built; their
    // barriers still run so that every level ends in the tracked layout.
    if(built && !cmdPushData(cmd, &args, sizeof(args)))
    {
      LOGE("cmdGenerateMipmaps: no push data for mip %u; mips %u to %u are not built\n", mip, mip, levels);
      built = false;
    }
    if(built)
    {
      cmd->Dispatch((args.dstWidth + groupX - 1) / groupX, (args.dstHeight + groupY - 1) / groupY, layers);
    }
    // The level just written becomes the next source, and is readable by every stage.
    cmdImageMemoryBarrier(cmd, {.image            = image.image.Get(),
                                .oldLayout        = D3D12_BARRIER_LAYOUT_UNORDERED_ACCESS,
                                .newLayout        = D3D12_BARRIER_LAYOUT_SHADER_RESOURCE,
                                .subresourceRange = {mip, 1, 0, layers, 0, 1},
                                .srcStageMask     = kWriteSync,
                                .dstStageMask     = uint64_t(readSync),
                                .srcAccessMask    = D3D12_BARRIER_ACCESS_UNORDERED_ACCESS,
                                .dstAccessMask    = uint64_t(readAccess)});
  }
  releaseSlots(cmd, slots, 2 * levels);
  image.layout     = D3D12_BARRIER_LAYOUT_SHADER_RESOURCE;
  image.stageMask  = readSync;
  image.accessMask = readAccess;
}

}  // namespace nvdx12

//--------------------------------------------------------------------------------------------------
// Usage example
//--------------------------------------------------------------------------------------------------
[[maybe_unused]] static void usage_BuiltinKernels()
{
  ID3D12Device*        device        = nullptr;    // EX: from nvdx12::Context::getDevice()
  ID3D12RootSignature* rootSignature = nullptr;    // EX: nvdx12::createBindlessRootSignature(rootSignature, device)
  ID3D12GraphicsCommandList7* cmd    = nullptr;    // EX: the frame's command list
  nvdx12::ShaderCompiler      compiler;            // EX: initialized with dxcompiler.dll
  nvdx12::DescriptorHeap      heap;                // EX: initialized by the application
  nvdx12::UploadRing          ring;                // EX: initialized by the application
  nvdx12::Buffer              counters;            // EX: created by nvdx12::ResourceAllocator
  nvdx12::Image               texture, thumbnail;  // EX: 2D images created with ALLOW_UNORDERED_ACCESS

  nvdx12::BuiltinKernels kernels;
  NVDX12_CHECK(kernels.init({.device                = device,
                             .compiler              = &compiler,
                             .rootSignature         = rootSignature,
                             .pushDataRootParameter = nvdx12::kRootParamPushData,
                             .descriptorHeap        = &heap,
                             .uploadRing            = &ring}));
  // The transient slots of the blit and mipmap kernels return to the heap
  // once the submission of their list on the ring's queue completed;
  // setSlotReleaseCallback hands them to the owner instead.

  // Each call binds the heaps and leaves its own PSO bound.
  kernels.cmdFillBuffer(cmd, counters, 0, ~0ull, 0);
  nvdx12::cmdMemoryBarrier(cmd);
  kernels.cmdGenerateMipmaps(cmd, texture);
  kernels.cmdBlitImage(cmd, texture, thumbnail, nvdx12::Filter::eLinear);
  nvdx12::cmdMemoryBarrier(cmd);
  // ... cmd->SetPipelineState(ownPso) ...

  kernels.deinit();
}
