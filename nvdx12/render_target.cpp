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

#include "render_target.hpp"

#include <algorithm>
#include <cassert>

#include <nvutils/logger.hpp>

#include "barriers.hpp"
#include "check_error.hpp"
#include "debug_util.hpp"
#include "graphics_pipeline.hpp"

namespace {

// Typed UAV support of a format on the device (the VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT query of nvvk).
bool supportsTypedUav(ID3D12Device* device, DXGI_FORMAT format)
{
  D3D12_FEATURE_DATA_FORMAT_SUPPORT support{.Format = format};
  return SUCCEEDED(device->CheckFeatureSupport(D3D12_FEATURE_FORMAT_SUPPORT, &support, sizeof(support)))
         && (support.Support1 & D3D12_FORMAT_SUPPORT1_TYPED_UNORDERED_ACCESS_VIEW) != 0;
}

// UINT and SINT formats: cleared with integer values and never resolved with AVERAGE.
bool isIntegerFormat(DXGI_FORMAT format)
{
  switch(format)
  {
    case DXGI_FORMAT_R32G32B32A32_UINT:
    case DXGI_FORMAT_R32G32B32A32_SINT:
    case DXGI_FORMAT_R32G32B32_UINT:
    case DXGI_FORMAT_R32G32B32_SINT:
    case DXGI_FORMAT_R16G16B16A16_UINT:
    case DXGI_FORMAT_R16G16B16A16_SINT:
    case DXGI_FORMAT_R32G32_UINT:
    case DXGI_FORMAT_R32G32_SINT:
    case DXGI_FORMAT_R10G10B10A2_UINT:
    case DXGI_FORMAT_R8G8B8A8_UINT:
    case DXGI_FORMAT_R8G8B8A8_SINT:
    case DXGI_FORMAT_R16G16_UINT:
    case DXGI_FORMAT_R16G16_SINT:
    case DXGI_FORMAT_R32_UINT:
    case DXGI_FORMAT_R32_SINT:
    case DXGI_FORMAT_R8G8_UINT:
    case DXGI_FORMAT_R8G8_SINT:
    case DXGI_FORMAT_R16_UINT:
    case DXGI_FORMAT_R16_SINT:
    case DXGI_FORMAT_R8_UINT:
    case DXGI_FORMAT_R8_SINT:
      return true;
    default:
      return false;
  }
}

// Typeless family of a depth format: a resource read through both a DSV
// and an SRV (in the color twin of the format) must be created typeless.
DXGI_FORMAT depthTypelessFormat(DXGI_FORMAT format)
{
  switch(format)
  {
    case DXGI_FORMAT_D32_FLOAT:
      return DXGI_FORMAT_R32_TYPELESS;
    case DXGI_FORMAT_D24_UNORM_S8_UINT:
      return DXGI_FORMAT_R24G8_TYPELESS;
    case DXGI_FORMAT_D32_FLOAT_S8X24_UINT:
      return DXGI_FORMAT_R32G8X24_TYPELESS;
    case DXGI_FORMAT_D16_UNORM:
      return DXGI_FORMAT_R16_TYPELESS;
    default:
      return format;
  }
}

// Color formats of the depth and the stencil plane of a depth format, which
// render-pass resolves take.
DXGI_FORMAT depthPlaneFormat(DXGI_FORMAT format)
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

DXGI_FORMAT stencilPlaneFormat(DXGI_FORMAT format)
{
  switch(format)
  {
    case DXGI_FORMAT_D24_UNORM_S8_UINT:
      return DXGI_FORMAT_X24_TYPELESS_G8_UINT;
    case DXGI_FORMAT_D32_FLOAT_S8X24_UINT:
      return DXGI_FORMAT_X32_TYPELESS_G8X24_UINT;
    default:
      return format;
  }
}

bool isResolveMode(D3D12_RESOLVE_MODE mode)
{
  return mode == D3D12_RESOLVE_MODE_MIN || mode == D3D12_RESOLVE_MODE_MAX || mode == D3D12_RESOLVE_MODE_AVERAGE;
}

// CPU handle `index` of a non-shader-visible heap.
D3D12_CPU_DESCRIPTOR_HANDLE heapHandle(ID3D12Device* device, ID3D12DescriptorHeap* heap, D3D12_DESCRIPTOR_HEAP_TYPE type, uint32_t index)
{
  if(heap == nullptr)
  {
    return {};
  }
  D3D12_CPU_DESCRIPTOR_HANDLE handle = heap->GetCPUDescriptorHandleForHeapStart();
  handle.ptr += SIZE_T(index) * device->GetDescriptorHandleIncrementSize(type);
  return handle;
}

D3D12_RENDER_TARGET_VIEW_DESC makeRtvDesc(DXGI_FORMAT format, uint32_t samples, uint32_t layerCount)
{
  D3D12_RENDER_TARGET_VIEW_DESC desc{.Format = format};
  if(samples > 1)
  {
    desc.ViewDimension = layerCount > 1 ? D3D12_RTV_DIMENSION_TEXTURE2DMSARRAY : D3D12_RTV_DIMENSION_TEXTURE2DMS;
    desc.Texture2DMSArray.ArraySize = layerCount;
  }
  else if(layerCount > 1)
  {
    desc.ViewDimension            = D3D12_RTV_DIMENSION_TEXTURE2DARRAY;
    desc.Texture2DArray.ArraySize = layerCount;
  }
  else
  {
    desc.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2D;
  }
  return desc;
}

D3D12_DEPTH_STENCIL_VIEW_DESC makeDsvDesc(DXGI_FORMAT format, uint32_t samples, uint32_t layerCount)
{
  D3D12_DEPTH_STENCIL_VIEW_DESC desc{.Format = format};
  if(samples > 1)
  {
    desc.ViewDimension = layerCount > 1 ? D3D12_DSV_DIMENSION_TEXTURE2DMSARRAY : D3D12_DSV_DIMENSION_TEXTURE2DMS;
    desc.Texture2DMSArray.ArraySize = layerCount;
  }
  else if(layerCount > 1)
  {
    desc.ViewDimension            = D3D12_DSV_DIMENSION_TEXTURE2DARRAY;
    desc.Texture2DArray.ArraySize = layerCount;
  }
  else
  {
    desc.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2D;
  }
  return desc;
}

// Clears through the RTV, else through the UAV (`uavGpu` in the bound
// shader-visible heap, `uavCpu` in a CPU-only one), then returns the image to `layout`.
void cmdClearColorImage(ID3D12GraphicsCommandList7* cmd,
                        nvdx12::Image&              image,
                        D3D12_BARRIER_LAYOUT        layout,
                        D3D12_CPU_DESCRIPTOR_HANDLE rtv,
                        D3D12_GPU_DESCRIPTOR_HANDLE uavGpu,
                        D3D12_CPU_DESCRIPTOR_HANDLE uavCpu,
                        const std::array<float, 4>& clearValue)
{
  if(!image)
    return;
  if(rtv.ptr != 0)
  {
    nvdx12::cmdImageBarrier(cmd, image, D3D12_BARRIER_LAYOUT_RENDER_TARGET);
    cmd->ClearRenderTargetView(rtv, clearValue.data(), 0, nullptr);
  }
  else
  {
    nvdx12::cmdImageBarrier(cmd, image, D3D12_BARRIER_LAYOUT_UNORDERED_ACCESS);
    if(isIntegerFormat(image.format))
    {
      const UINT values[4] = {UINT(clearValue[0]), UINT(clearValue[1]), UINT(clearValue[2]), UINT(clearValue[3])};
      cmd->ClearUnorderedAccessViewUint(uavGpu, uavCpu, image.image.Get(), values, 0, nullptr);
    }
    else
    {
      cmd->ClearUnorderedAccessViewFloat(uavGpu, uavCpu, image.image.Get(), clearValue.data(), 0, nullptr);
    }
    // Later uses of the UNORDERED_ACCESS layout wait for the clear.
    nvdx12::cmdImageMemoryBarrier(cmd, image,
                                  {.newLayout     = D3D12_BARRIER_LAYOUT_UNORDERED_ACCESS,
                                   .srcStageMask  = D3D12_BARRIER_SYNC_CLEAR_UNORDERED_ACCESS_VIEW,
                                   .srcAccessMask = D3D12_BARRIER_ACCESS_UNORDERED_ACCESS});
  }
  nvdx12::cmdImageBarrier(cmd, image, layout);
}

void cmdClearDepthStencilImage(ID3D12GraphicsCommandList7*      cmd,
                               nvdx12::Image&                   image,
                               D3D12_BARRIER_LAYOUT             layout,
                               D3D12_CPU_DESCRIPTOR_HANDLE      dsv,
                               D3D12_CLEAR_FLAGS                clearFlags,
                               const D3D12_DEPTH_STENCIL_VALUE& clearValue)
{
  if(!image)
    return;
  nvdx12::cmdImageBarrier(cmd, image, D3D12_BARRIER_LAYOUT_DEPTH_STENCIL_WRITE);
  cmd->ClearDepthStencilView(dsv, clearFlags, clearValue.Depth, clearValue.Stencil, 0, nullptr);
  nvdx12::cmdImageBarrier(cmd, image, layout);
}

// Ending access of an attachment: a resolve keeps RESOLVE and the store op
// decides whether the multisampled source is preserved.
void applyEndingAccess(D3D12_RENDER_PASS_ENDING_ACCESS&                                                   access,
                       D3D12_RENDER_PASS_ENDING_ACCESS_TYPE                                               store,
                       const std::vector<D3D12_RENDER_PASS_ENDING_ACCESS_RESOLVE_SUBRESOURCE_PARAMETERS>& subresources,
                       size_t&                                                                            resolveOffset)
{
  if(access.Type != D3D12_RENDER_PASS_ENDING_ACCESS_TYPE_RESOLVE)
  {
    access.Type = store;
    return;
  }
  access.Resolve.PreserveResolveSource  = store == D3D12_RENDER_PASS_ENDING_ACCESS_TYPE_PRESERVE;
  access.Resolve.pSubresourceParameters = subresources.data() + resolveOffset;
  resolveOffset += access.Resolve.SubresourceCount;
}

}  // namespace

////////////////////////////////////////////////////////////
// RENDER TARGET STATE
////////////////////////////////////////////////////////////

nvdx12::RenderTargetState::RenderTargetState()
{
  clear();
}

void nvdx12::RenderTargetState::clear()
{
  colorCount    = 0;
  depthFormat   = DXGI_FORMAT_UNKNOWN;
  stencilFormat = DXGI_FORMAT_UNKNOWN;
  layerCount    = 1;
  sampleCount   = 1;
  renderArea    = {};
  colorFormats.clear();
  colorAttachments.clear();
  resolveSubresources.clear();
  depthStencilAttachment = {
      .DepthBeginningAccess   = {.Type = D3D12_RENDER_PASS_BEGINNING_ACCESS_TYPE_NO_ACCESS},
      .StencilBeginningAccess = {.Type = D3D12_RENDER_PASS_BEGINNING_ACCESS_TYPE_NO_ACCESS},
      .DepthEndingAccess      = {.Type = D3D12_RENDER_PASS_ENDING_ACCESS_TYPE_NO_ACCESS},
      .StencilEndingAccess    = {.Type = D3D12_RENDER_PASS_ENDING_ACCESS_TYPE_NO_ACCESS},
  };
}

void nvdx12::RenderTargetState::fillPipelineRenderingCreateInfo(GraphicsPipelineCreator& creator) const
{
  assert((colorCount == 0 || !colorFormats.empty()) && "Missing color format");
  creator.colorFormats.assign(colorFormats.begin(), colorFormats.begin() + colorCount);
  creator.renderingState.depthAttachmentFormat = depthFormat != DXGI_FORMAT_UNKNOWN ? depthFormat : stencilFormat;
}

void nvdx12::RenderTargetState::cmdBeginRendering(ID3D12GraphicsCommandList7* cmd, const AttachmentOps& ops)
{
  assert((colorCount <= colorAttachments.size()) && "colorAttachments out of sync with colorCount");
  size_t resolveOffset = 0;
  for(uint32_t i = 0; i < colorCount; ++i)
  {
    colorAttachments[i].BeginningAccess.Type = ops.colorLoad;
    applyEndingAccess(colorAttachments[i].EndingAccess, ops.colorStore, resolveSubresources, resolveOffset);
  }
  if(depthFormat != DXGI_FORMAT_UNKNOWN)
  {
    depthStencilAttachment.DepthBeginningAccess.Type = ops.depthLoad;
    applyEndingAccess(depthStencilAttachment.DepthEndingAccess, ops.depthStore, resolveSubresources, resolveOffset);
  }
  if(stencilFormat != DXGI_FORMAT_UNKNOWN)
  {
    depthStencilAttachment.StencilBeginningAccess.Type = ops.stencilLoad;
    applyEndingAccess(depthStencilAttachment.StencilEndingAccess, ops.stencilStore, resolveSubresources, resolveOffset);
  }
  assert(resolveOffset <= resolveSubresources.size() && "resolveSubresources out of sync with the attachments");

  const bool hasDepthStencil = depthFormat != DXGI_FORMAT_UNKNOWN || stencilFormat != DXGI_FORMAT_UNKNOWN;
  cmd->BeginRenderPass(colorCount, colorCount ? colorAttachments.data() : nullptr,
                       hasDepthStencil ? &depthStencilAttachment : nullptr, ops.flags);
}

////////////////////////////////////////////////////////////
// RENDER TARGET
////////////////////////////////////////////////////////////

nvdx12::RenderTarget::~RenderTarget()
{
  assert((m_createInfo.alloc == nullptr) && "Missing deinit()");
}

HRESULT nvdx12::RenderTarget::init(const CreateInfo& info)
{
  if(m_createInfo.alloc != nullptr)
  {
    LOGE("nvdx12::RenderTarget: double init()\n");
    return E_INVALIDARG;
  }

  NVDX12_FAIL_RETURN(validateCreateInfo(info));
  DescriptorHeap* heap = info.descriptorHeap ? info.descriptorHeap : info.alloc->getDescriptorHeap();
  if(heap == nullptr)
  {
    LOGE("nvdx12::RenderTarget: a descriptor heap is required\n");
    return E_INVALIDARG;
  }
  m_createInfo     = info;
  m_heap           = heap;
  m_resourcesDirty = true;

  // The slots are acquired once and keep their numbers across update().
  const size_t colorCount = m_createInfo.colorFormats.size();
  m_colorImages.resize(colorCount);
  m_colorImagesResolved.resize(colorCount);
  m_colorSlots.resize(colorCount);
  m_uiSlots.resize(colorCount, kInvalidSlot);
  bool slotsOk = true;
  for(size_t i = 0; i < colorCount; i++)
  {
    m_colorSlots[i] = m_heap->acquireImageSlot();
    slotsOk         = slotsOk && m_colorSlots[i].valid();
    if(m_createInfo.layerCount == 1)
    {
      m_uiSlots[i] = m_heap->acquireSlots();
      slotsOk      = slotsOk && m_uiSlots[i] != kInvalidSlot;
    }
  }
  if(sampleDepth())
  {
    m_depthSlot = m_heap->acquireSlots();
    slotsOk     = slotsOk && m_depthSlot != kInvalidSlot;
  }
  if(!slotsOk)
  {
    LOGE("nvdx12::RenderTarget: the descriptor heap is full\n");
    deinit();
    return E_OUTOFMEMORY;
  }
  return S_OK;
}

void nvdx12::RenderTarget::deinit()
{
  if(m_createInfo.alloc == nullptr)
    return;
  destroyResources();
  releaseSlots();
  m_colorImages         = {};
  m_colorImagesResolved = {};
  m_createInfo          = {};
  m_heap                = nullptr;
}

HRESULT nvdx12::RenderTarget::setMsaa(const MsaaConfig& msaa)
{
  CreateInfo info = m_createInfo;
  info.msaa       = msaa;
  NVDX12_FAIL_RETURN(validateCreateInfo(info));
  m_createInfo.msaa = msaa;
  m_resourcesDirty  = true;
  return S_OK;
}

HRESULT nvdx12::RenderTarget::setSampleCount(uint32_t                          sampleCount,
                                             std::optional<D3D12_RESOLVE_MODE> colorResolveMode,
                                             std::optional<D3D12_RESOLVE_MODE> depthResolveMode)
{
  return setMsaa({.samples = sampleCount, .colorResolve = colorResolveMode, .depthResolve = depthResolveMode});
}

HRESULT nvdx12::RenderTarget::update(ID3D12GraphicsCommandList7* cmd, Extent2D extent)
{
  if(m_createInfo.alloc == nullptr)
  {
    LOGE("nvdx12::RenderTarget: call init() before update()\n");
    return E_INVALIDARG;
  }
  if(extent.width == 0 || extent.height == 0)
  {
    LOGE("nvdx12::RenderTarget: update() extent must be non-zero\n");
    return E_INVALIDARG;
  }
  if(extent.width == m_extent.width && extent.height == m_extent.height && !m_resourcesDirty)
    return S_OK;

  destroyResources();
  const HRESULT result = createResources(cmd, extent);
  if(FAILED(result))
    destroyResources();
  return result;
}

void nvdx12::RenderTarget::fillState(RenderTargetState& state, bool depthOnly, bool addResolve) const
{
  assert((!depthOnly || hasDepthStencil()) && "depthOnly requires a depth or stencil format");

  state.clear();

  const uint32_t colorCount = depthOnly ? 0 : static_cast<uint32_t>(m_createInfo.colorFormats.size());
  state.colorCount          = colorCount;
  state.depthFormat         = m_createInfo.depthFormat;
  state.stencilFormat       = m_createInfo.stencilFormat;
  state.sampleCount         = m_createInfo.msaa.samples;
  state.layerCount          = m_createInfo.layerCount;
  state.colorFormats        = m_createInfo.colorFormats;
  state.colorAttachments.resize(colorCount);
  state.renderArea = DEFAULT_D3D12_RECT(m_extent.width, m_extent.height);

  const bool resolveColor   = addResolve && wantsColorResolve();
  const bool resolveDepth   = addResolve && wantsDepthResolve();
  const bool resolveStencil = addResolve && wantsStencilResolve();

  // One resolve region per layer; the stencil plane follows the depth plane's subresources.
  const uint32_t layerCount  = m_createInfo.layerCount;
  const auto     addResolves = [&](uint32_t firstSubresource) {
    for(uint32_t layer = 0; layer < layerCount; layer++)
    {
      state.resolveSubresources.push_back(
          {.SrcSubresource = firstSubresource + layer, .DstSubresource = firstSubresource + layer, .SrcRect = state.renderArea});
    }
  };
  const auto resolveAccess = [&](const Image& src, const Image& dst, DXGI_FORMAT format, D3D12_RESOLVE_MODE mode) {
    D3D12_RENDER_PASS_ENDING_ACCESS access{.Type = D3D12_RENDER_PASS_ENDING_ACCESS_TYPE_RESOLVE};
    access.Resolve = {.pSrcResource          = src.image.Get(),
                      .pDstResource          = dst.image.Get(),
                      .SubresourceCount      = layerCount,
                      .Format                = format,
                      .ResolveMode           = mode,
                      .PreserveResolveSource = TRUE};
    return access;
  };
  const auto clearAccess = [](DXGI_FORMAT format) {
    D3D12_RENDER_PASS_BEGINNING_ACCESS access{.Type = D3D12_RENDER_PASS_BEGINNING_ACCESS_TYPE_CLEAR};
    access.Clear.ClearValue.Format = format;
    return access;
  };

  for(uint32_t i = 0; i < colorCount; ++i)
  {
    if(i >= m_colorImages.size() || !m_colorImages[i])
      continue;

    D3D12_RENDER_PASS_RENDER_TARGET_DESC& attachment = state.colorAttachments[i];
    assert(getColorAttachmentView(i).ptr != 0 && "rendering needs ALLOW_RENDER_TARGET in colorFlags");
    attachment.cpuDescriptor     = getColorAttachmentView(i);
    attachment.BeginningAccess   = clearAccess(m_createInfo.colorFormats[i]);
    attachment.EndingAccess.Type = D3D12_RENDER_PASS_ENDING_ACCESS_TYPE_PRESERVE;

    if(resolveColor && i < m_colorImagesResolved.size() && m_colorImagesResolved[i])
    {
      attachment.EndingAccess = resolveAccess(m_colorImages[i], m_colorImagesResolved[i], m_createInfo.colorFormats[i],
                                              *m_createInfo.msaa.colorResolve);
      addResolves(0);
    }
  }

  if(!m_depthImage)
    return;
  D3D12_RENDER_PASS_DEPTH_STENCIL_DESC& attachment = state.depthStencilAttachment;
  attachment.cpuDescriptor                         = getDepthImageView();
  D3D12_RENDER_PASS_BEGINNING_ACCESS depthClear    = clearAccess(depthStencilFormat());
  depthClear.Clear.ClearValue.DepthStencil         = DEFAULT_D3D12_DEPTH_STENCIL_VALUE;
  // Resolves take the color format of each plane, also for an image created in the depth format.
  const DXGI_FORMAT depthResolveFormat   = depthPlaneFormat(depthStencilFormat());
  const DXGI_FORMAT stencilResolveFormat = stencilPlaneFormat(depthStencilFormat());

  if(m_createInfo.depthFormat != DXGI_FORMAT_UNKNOWN)
  {
    attachment.DepthBeginningAccess = depthClear;
    attachment.DepthEndingAccess    = {.Type = D3D12_RENDER_PASS_ENDING_ACCESS_TYPE_PRESERVE};
    if(resolveDepth && m_depthImageResolved)
    {
      attachment.DepthEndingAccess =
          resolveAccess(m_depthImage, m_depthImageResolved, depthResolveFormat, *m_createInfo.msaa.depthResolve);
      addResolves(0);
    }
  }

  if(m_createInfo.stencilFormat != DXGI_FORMAT_UNKNOWN)
  {
    attachment.StencilBeginningAccess = depthClear;
    attachment.StencilEndingAccess    = {.Type = D3D12_RENDER_PASS_ENDING_ACCESS_TYPE_PRESERVE};
    if(resolveStencil && m_depthImageResolved)
    {
      attachment.StencilEndingAccess =
          resolveAccess(m_depthImage, m_depthImageResolved, stencilResolveFormat, *m_createInfo.msaa.stencilResolve);
      addResolves(layerCount);
    }
  }
}

nvdx12::Extent2D nvdx12::RenderTarget::getSize() const
{
  return m_extent;
}

const nvdx12::Image& nvdx12::RenderTarget::getColorImage(uint32_t i) const
{
  assert(i < m_colorImages.size() && m_colorImages[i]);
  return m_colorImages[i];
}

nvdx12::Image& nvdx12::RenderTarget::getColorImage(uint32_t i)
{
  assert(i < m_colorImages.size() && m_colorImages[i]);
  return m_colorImages[i];
}

D3D12_CPU_DESCRIPTOR_HANDLE nvdx12::RenderTarget::getColorAttachmentView(uint32_t i) const
{
  assert(i < m_colorImages.size());
  if((m_colorImages[i].flags & D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET) == 0)
    return {};
  return rtvHandle(i);
}

const nvdx12::Image& nvdx12::RenderTarget::getColorImageResolved(uint32_t i) const
{
  assert(i < m_colorImagesResolved.size());
  return m_colorImagesResolved[i];
}

nvdx12::Image& nvdx12::RenderTarget::getColorImageResolved(uint32_t i)
{
  assert(i < m_colorImagesResolved.size());
  return m_colorImagesResolved[i];
}

const nvdx12::Image& nvdx12::RenderTarget::getSampleImage(uint32_t i) const
{
  const Image& resolved = getColorImageResolved(i);
  return resolved ? resolved : getColorImage(i);
}

nvdx12::Image& nvdx12::RenderTarget::getSampleImage(uint32_t i)
{
  Image& resolved = getColorImageResolved(i);
  return resolved ? resolved : getColorImage(i);
}

D3D12_GPU_DESCRIPTOR_HANDLE nvdx12::RenderTarget::getUiImageView(uint32_t i) const
{
  assert(i < m_uiSlots.size() && m_uiSlots[i] != kInvalidSlot && "no display SRV: missing init() or a layered target");
  return m_heap->gpuHandle(m_uiSlots[i]);
}

nvdx12::ImageSlot nvdx12::RenderTarget::getColorImageSlot(uint32_t i) const
{
  assert(i < m_colorSlots.size() && "Missing init()");
  return m_colorSlots[i];
}

D3D12_BARRIER_LAYOUT nvdx12::RenderTarget::getColorLayout() const
{
  return m_createInfo.layouts.color;
}

D3D12_BARRIER_LAYOUT nvdx12::RenderTarget::getDepthLayout() const
{
  return m_createInfo.layouts.depth;
}

D3D12_BARRIER_LAYOUT nvdx12::RenderTarget::getColorSampleLayout() const
{
  return effectiveColorSampleLayout(m_createInfo);
}

const nvdx12::Image& nvdx12::RenderTarget::getDepthImage() const
{
  return m_depthImage;
}

nvdx12::Image& nvdx12::RenderTarget::getDepthImage()
{
  return m_depthImage;
}

D3D12_CPU_DESCRIPTOR_HANDLE nvdx12::RenderTarget::getDepthImageView() const
{
  return m_depthImage ? dsvHandle(0) : D3D12_CPU_DESCRIPTOR_HANDLE{};
}

const nvdx12::Image& nvdx12::RenderTarget::getDepthImageResolved() const
{
  return m_depthImageResolved;
}

nvdx12::Image& nvdx12::RenderTarget::getDepthImageResolved()
{
  return m_depthImageResolved;
}

uint32_t nvdx12::RenderTarget::getDepthImageSlot() const
{
  return m_depthSlot;
}

DXGI_FORMAT nvdx12::RenderTarget::getColorFormat(uint32_t i) const
{
  return m_createInfo.colorFormats[i];
}

DXGI_FORMAT nvdx12::RenderTarget::getDepthFormat() const
{
  return m_createInfo.depthFormat;
}

uint32_t nvdx12::RenderTarget::getSampleCount() const
{
  return m_createInfo.msaa.samples;
}

uint32_t nvdx12::RenderTarget::getLayerCount() const
{
  return m_createInfo.layerCount;
}

float nvdx12::RenderTarget::getAspectRatio() const
{
  return m_extent.height > 0 ? float(m_extent.width) / float(m_extent.height) : 0.f;
}

D3D12MA::Allocation* nvdx12::RenderTarget::getColorAllocation(uint32_t i) const
{
  return i < m_colorImages.size() ? m_colorImages[i].allocation : nullptr;
}

D3D12MA::Allocation* nvdx12::RenderTarget::getColorAllocationResolved(uint32_t i) const
{
  return i < m_colorImagesResolved.size() ? m_colorImagesResolved[i].allocation : nullptr;
}

D3D12MA::Allocation* nvdx12::RenderTarget::getDepthAllocation() const
{
  return m_depthImage ? m_depthImage.allocation : nullptr;
}

D3D12MA::Allocation* nvdx12::RenderTarget::getDepthAllocationResolved() const
{
  return m_depthImageResolved ? m_depthImageResolved.allocation : nullptr;
}

void nvdx12::RenderTarget::cmdClear(ID3D12GraphicsCommandList7* cmd, const std::array<float, 4>& colorClear, D3D12_DEPTH_STENCIL_VALUE depthClear)
{
  assert((m_createInfo.alloc != nullptr) && "Call init() and update() before cmdClear()");

  const uint32_t colorCount = static_cast<uint32_t>(m_colorImages.size());
  bool           heapsBound = false;
  const auto     uavHandles = [&](uint32_t i, const Image& image) {
    std::pair<D3D12_GPU_DESCRIPTOR_HANDLE, D3D12_CPU_DESCRIPTOR_HANDLE> handles{};
    if(image.allowsUav() && (image.flags & D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET) == 0)
    {
      if(!heapsBound)
      {
        // A UAV clear takes the UAV in the bound shader-visible heap.
        ID3D12DescriptorHeap* heaps[] = {m_heap->resourceHeap(), m_heap->samplerHeap()};
        cmd->SetDescriptorHeaps(2, heaps);
        heapsBound = true;
      }
      handles = {m_heap->gpuHandle(m_colorSlots[i].uav()), uavClearHandle(i)};
    }
    return handles;
  };

  for(uint32_t i = 0; i < colorCount; i++)
  {
    Image& color          = m_colorImages[i];
    const auto [gpu, cpu] = uavHandles(i, color);
    cmdClearColorImage(cmd, color, m_createInfo.layouts.color, getColorAttachmentView(i), gpu, cpu, colorClear);
  }
  for(uint32_t i = 0; i < colorCount; i++)
  {
    Image& resolved = m_colorImagesResolved[i];
    if(!resolved)
      continue;
    const D3D12_CPU_DESCRIPTOR_HANDLE rtv = (resolved.flags & D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET) != 0 ?
                                                rtvHandle(colorCount + i) :
                                                D3D12_CPU_DESCRIPTOR_HANDLE{};
    const auto [gpu, cpu]                 = uavHandles(i, resolved);
    cmdClearColorImage(cmd, resolved, D3D12_BARRIER_LAYOUT_RESOLVE_DEST, rtv, gpu, cpu, colorClear);
  }

  if(m_depthImage)
  {
    const D3D12_CLEAR_FLAGS clearFlags =
        (m_createInfo.depthFormat != DXGI_FORMAT_UNKNOWN ? D3D12_CLEAR_FLAG_DEPTH : D3D12_CLEAR_FLAGS(0))
        | (m_createInfo.stencilFormat != DXGI_FORMAT_UNKNOWN ? D3D12_CLEAR_FLAG_STENCIL : D3D12_CLEAR_FLAGS(0));
    cmdClearDepthStencilImage(cmd, m_depthImage, m_createInfo.layouts.depth, dsvHandle(0), clearFlags, depthClear);
    cmdClearDepthStencilImage(cmd, m_depthImageResolved, D3D12_BARRIER_LAYOUT_RESOLVE_DEST, dsvHandle(1), clearFlags, depthClear);
  }
}

////////////////////////////////////////////////////////////
// INTERNAL
////////////////////////////////////////////////////////////

HRESULT nvdx12::RenderTarget::validateCreateInfo(const CreateInfo& info)
{
  if(info.alloc == nullptr)
  {
    LOGE("nvdx12::RenderTarget: CreateInfo::alloc is required\n");
    return E_INVALIDARG;
  }

  if(info.colorFormats.empty() && info.depthFormat == DXGI_FORMAT_UNKNOWN && info.stencilFormat == DXGI_FORMAT_UNKNOWN)
  {
    LOGE("nvdx12::RenderTarget: at least one color, depth, or stencil format is required\n");
    return E_INVALIDARG;
  }
  for(size_t i = 0; i < info.colorFormats.size(); i++)
  {
    if(info.colorFormats[i] == DXGI_FORMAT_UNKNOWN)
    {
      LOGE("nvdx12::RenderTarget: color format %zu is DXGI_FORMAT_UNKNOWN (render passes bind every target)\n", i);
      return E_INVALIDARG;
    }
  }

  if(info.layerCount < 1)
  {
    LOGE("nvdx12::RenderTarget: layerCount must be at least 1\n");
    return E_INVALIDARG;
  }

  if(info.depthFormat != DXGI_FORMAT_UNKNOWN && info.stencilFormat != DXGI_FORMAT_UNKNOWN && info.depthFormat != info.stencilFormat)
  {
    LOGE("nvdx12::RenderTarget: depthFormat and stencilFormat must match when both are set (use a combined depth/stencil format)\n");
    return E_INVALIDARG;
  }

  if(info.msaa.samples < 1)
  {
    LOGE("nvdx12::RenderTarget: msaa.samples must be at least 1\n");
    return E_INVALIDARG;
  }

  if(info.msaa.samples > 1 && !info.colorFormats.empty() && !info.msaa.colorResolve)
  {
    LOGE("nvdx12::RenderTarget: MSAA color targets require msaa.colorResolve (e.g. D3D12_RESOLVE_MODE_AVERAGE)\n");
    return E_INVALIDARG;
  }

  const std::array<std::optional<D3D12_RESOLVE_MODE>, 3> modes{info.msaa.colorResolve, info.msaa.depthResolve, info.msaa.stencilResolve};
  for(size_t i = 0; i < modes.size(); i++)
  {
    if(modes[i] && !isResolveMode(*modes[i]))
    {
      LOGE("nvdx12::RenderTarget: resolve modes are D3D12_RESOLVE_MODE_MIN, MAX or AVERAGE\n");
      return E_INVALIDARG;
    }
  }

  if(info.msaa.stencilResolve == D3D12_RESOLVE_MODE_AVERAGE)
  {
    LOGE("nvdx12::RenderTarget: msaa.stencilResolve must not be D3D12_RESOLVE_MODE_AVERAGE (invalid for integer stencil)\n");
    return E_INVALIDARG;
  }

  if(info.msaa.colorResolve == D3D12_RESOLVE_MODE_AVERAGE)
  {
    for(size_t i = 0; i < info.colorFormats.size(); i++)
    {
      if(isIntegerFormat(info.colorFormats[i]))
      {
        LOGE("nvdx12::RenderTarget: msaa.colorResolve must not be D3D12_RESOLVE_MODE_AVERAGE with integer color format %d\n",
             int(info.colorFormats[i]));
        return E_INVALIDARG;
      }
    }
  }

  if(info.layouts.color != D3D12_BARRIER_LAYOUT_RENDER_TARGET && info.layouts.color != D3D12_BARRIER_LAYOUT_UNORDERED_ACCESS)
  {
    LOGE("nvdx12::RenderTarget: unsupported layouts.color (RENDER_TARGET or UNORDERED_ACCESS)\n");
    return E_INVALIDARG;
  }
  if(info.layouts.depth != D3D12_BARRIER_LAYOUT_DEPTH_STENCIL_WRITE)
  {
    LOGE("nvdx12::RenderTarget: unsupported layouts.depth (DEPTH_STENCIL_WRITE)\n");
    return E_INVALIDARG;
  }
  if(info.layouts.colorSample != D3D12_BARRIER_LAYOUT_UNDEFINED && info.layouts.colorSample != D3D12_BARRIER_LAYOUT_SHADER_RESOURCE)
  {
    LOGE("nvdx12::RenderTarget: unsupported layouts.colorSample (UNDEFINED or SHADER_RESOURCE)\n");
    return E_INVALIDARG;
  }
  if(info.layouts.color == D3D12_BARRIER_LAYOUT_UNORDERED_ACCESS && info.msaa.samples > 1 && !info.colorFormats.empty())
  {
    LOGE("nvdx12::RenderTarget: MSAA color targets need layouts.color RENDER_TARGET (no unordered access)\n");
    return E_INVALIDARG;
  }

  const D3D12_RESOURCE_FLAGS colorTargetFlags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET | D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
  if(info.colorFlags != D3D12_RESOURCE_FLAG_NONE && (info.colorFlags & colorTargetFlags) == 0)
  {
    LOGE("nvdx12::RenderTarget: colorFlags need ALLOW_RENDER_TARGET or ALLOW_UNORDERED_ACCESS\n");
    return E_INVALIDARG;
  }
  if(info.colorFlags != D3D12_RESOURCE_FLAG_NONE && info.layouts.color == D3D12_BARRIER_LAYOUT_UNORDERED_ACCESS
     && (info.colorFlags & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS) == 0)
  {
    LOGE("nvdx12::RenderTarget: layouts.color UNORDERED_ACCESS needs ALLOW_UNORDERED_ACCESS in colorFlags\n");
    return E_INVALIDARG;
  }
  if(info.colorFlags != D3D12_RESOURCE_FLAG_NONE && info.msaa.samples > 1
     && (info.colorFlags & D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET) == 0)
  {
    LOGE("nvdx12::RenderTarget: MSAA color targets need ALLOW_RENDER_TARGET in colorFlags\n");
    return E_INVALIDARG;
  }

  if(info.depthFlags != D3D12_RESOURCE_FLAG_NONE
     && ((info.depthFlags & D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL) == 0
         || (info.depthFlags & (D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET | D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS)) != 0))
  {
    LOGE("nvdx12::RenderTarget: depthFlags need ALLOW_DEPTH_STENCIL, without ALLOW_RENDER_TARGET or ALLOW_UNORDERED_ACCESS\n");
    return E_INVALIDARG;
  }

  return S_OK;
}

D3D12_BARRIER_LAYOUT nvdx12::RenderTarget::effectiveColorSampleLayout(const CreateInfo& info)
{
  if(info.layouts.colorSample != D3D12_BARRIER_LAYOUT_UNDEFINED)
    return info.layouts.colorSample;
  return D3D12_BARRIER_LAYOUT_SHADER_RESOURCE;
}

HRESULT nvdx12::RenderTarget::resolveColorFlags(const CreateInfo& info, DXGI_FORMAT format, ID3D12Device* device, ColorFlagsResult& out)
{
  const bool           userColorFlags = (info.colorFlags != D3D12_RESOURCE_FLAG_NONE);
  D3D12_RESOURCE_FLAGS formatFlags    = userColorFlags ? info.colorFlags : D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;

  const bool formatSupportsUav = supportsTypedUav(device, format);
  if((formatFlags & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS) != 0 && !formatSupportsUav)
  {
    LOGE("nvdx12::RenderTarget: color format %d has no typed UAV, requested via colorFlags\n", int(format));
    return E_INVALIDARG;
  }
  if(!userColorFlags && formatSupportsUav && info.layouts.color == D3D12_BARRIER_LAYOUT_UNORDERED_ACCESS)
  {
    formatFlags |= D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
  }
  if(info.layouts.color == D3D12_BARRIER_LAYOUT_UNORDERED_ACCESS && (formatFlags & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS) == 0)
  {
    LOGE("nvdx12::RenderTarget: layouts.color UNORDERED_ACCESS needs a typed UAV, which color format %d does not have\n",
         int(format));
    return E_INVALIDARG;
  }

  out.resolveFlags = formatFlags;
  out.renderFlags  = formatFlags;
  if(info.msaa.samples > 1 && (out.renderFlags & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS) != 0)
  {
    if(userColorFlags)
    {
      LOGE("nvdx12::RenderTarget: ALLOW_UNORDERED_ACCESS in colorFlags on an MSAA render image (no multisampled UAVs)\n");
      return E_INVALIDARG;
    }
    out.renderFlags &= ~D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
  }
  return S_OK;
}

DXGI_FORMAT nvdx12::RenderTarget::depthStencilFormat() const
{
  return m_createInfo.depthFormat != DXGI_FORMAT_UNKNOWN ? m_createInfo.depthFormat : m_createInfo.stencilFormat;
}

bool nvdx12::RenderTarget::hasDepthStencil() const
{
  return depthStencilFormat() != DXGI_FORMAT_UNKNOWN;
}

bool nvdx12::RenderTarget::isMsaa() const
{
  return m_createInfo.msaa.samples > 1;
}

bool nvdx12::RenderTarget::wantsColorResolve() const
{
  return isMsaa() && m_createInfo.msaa.colorResolve.has_value();
}

bool nvdx12::RenderTarget::wantsDepthResolve() const
{
  return isMsaa() && m_createInfo.depthFormat != DXGI_FORMAT_UNKNOWN && m_createInfo.msaa.depthResolve.has_value();
}

bool nvdx12::RenderTarget::wantsStencilResolve() const
{
  return isMsaa() && m_createInfo.stencilFormat != DXGI_FORMAT_UNKNOWN && m_createInfo.msaa.stencilResolve.has_value();
}

bool nvdx12::RenderTarget::wantsDepthStencilResolve() const
{
  return wantsDepthResolve() || wantsStencilResolve();
}

bool nvdx12::RenderTarget::sampleDepth() const
{
  return hasDepthStencil() && (m_createInfo.depthFlags & D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE) == 0;
}

void nvdx12::RenderTarget::releaseSlots()
{
  if(m_heap == nullptr)
    return;
  for(size_t i = 0; i < m_colorSlots.size(); i++)
  {
    m_heap->releaseImageSlot(m_colorSlots[i]);
  }
  for(size_t i = 0; i < m_uiSlots.size(); i++)
  {
    m_heap->releaseSlots(m_uiSlots[i]);
  }
  m_heap->releaseSlots(m_depthSlot);
  m_colorSlots = {};
  m_uiSlots    = {};
  m_depthSlot  = kInvalidSlot;
}

void nvdx12::RenderTarget::destroyResources()
{
  if(m_createInfo.alloc == nullptr)
    return;

  ResourceAllocator* alloc = m_createInfo.alloc;
  if(m_depthImageResolved)
    alloc->destroyImage(m_depthImageResolved);
  m_depthImageResolved = {};

  if(m_depthImage)
    alloc->destroyImage(m_depthImage);
  m_depthImage = {};

  for(size_t i = 0; i < m_colorImagesResolved.size(); i++)
  {
    if(m_colorImagesResolved[i])
      alloc->destroyImage(m_colorImagesResolved[i]);
    m_colorImagesResolved[i] = {};
  }

  for(size_t i = 0; i < m_colorImages.size(); i++)
  {
    if(m_colorImages[i])
      alloc->destroyImage(m_colorImages[i]);
    m_colorImages[i] = {};
  }

  m_rtvHeap.Reset();
  m_dsvHeap.Reset();
  m_uavClearHeap.Reset();

  m_extent         = {};
  m_resourcesDirty = true;
}

D3D12_CPU_DESCRIPTOR_HANDLE nvdx12::RenderTarget::rtvHandle(uint32_t index) const
{
  return heapHandle(m_createInfo.alloc->getDevice(), m_rtvHeap.Get(), D3D12_DESCRIPTOR_HEAP_TYPE_RTV, index);
}

D3D12_CPU_DESCRIPTOR_HANDLE nvdx12::RenderTarget::dsvHandle(uint32_t index) const
{
  return heapHandle(m_createInfo.alloc->getDevice(), m_dsvHeap.Get(), D3D12_DESCRIPTOR_HEAP_TYPE_DSV, index);
}

D3D12_CPU_DESCRIPTOR_HANDLE nvdx12::RenderTarget::uavClearHandle(uint32_t index) const
{
  return heapHandle(m_createInfo.alloc->getDevice(), m_uavClearHeap.Get(), D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, index);
}

HRESULT nvdx12::RenderTarget::createColorAttachment(uint32_t i, Extent2D extent, bool resolveColor, const std::string& namePrefix)
{
  ResourceAllocator* alloc  = m_createInfo.alloc;
  ID3D12Device*      device = alloc->getDevice();
  const DXGI_FORMAT  format = m_createInfo.colorFormats[i];

  ColorFlagsResult flags{};
  NVDX12_FAIL_RETURN(resolveColorFlags(m_createInfo, format, device, flags));

  D3D12_RESOURCE_DESC1 imageDesc = DEFAULT_D3D12_RESOURCE_DESC1;
  imageDesc.Width                = extent.width;
  imageDesc.Height               = extent.height;
  imageDesc.DepthOrArraySize     = UINT16(m_createInfo.layerCount);
  imageDesc.Format               = format;
  imageDesc.SampleDesc.Count     = m_createInfo.msaa.samples;
  imageDesc.Flags                = flags.renderFlags;
  NVDX12_FAIL_RETURN(alloc->createImage(m_colorImages[i], imageDesc));
  setDebugName(m_colorImages[i].image.Get(), namePrefix + ":colorImage" + std::to_string(i));
  if((flags.renderFlags & D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET) != 0)
  {
    const D3D12_RENDER_TARGET_VIEW_DESC rtv = makeRtvDesc(format, m_createInfo.msaa.samples, m_createInfo.layerCount);
    device->CreateRenderTargetView(m_colorImages[i].image.Get(), &rtv, rtvHandle(i));
  }

  if(resolveColor)
  {
    D3D12_RESOURCE_DESC1 resolveDesc = imageDesc;
    resolveDesc.SampleDesc.Count     = 1;
    resolveDesc.Flags                = flags.resolveFlags;
    NVDX12_FAIL_RETURN(alloc->createImage(m_colorImagesResolved[i], resolveDesc));
    setDebugName(m_colorImagesResolved[i].image.Get(), namePrefix + ":colorImageResolved" + std::to_string(i));
    if((flags.resolveFlags & D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET) != 0)
    {
      const D3D12_RENDER_TARGET_VIEW_DESC rtv = makeRtvDesc(format, 1, m_createInfo.layerCount);
      device->CreateRenderTargetView(m_colorImagesResolved[i].image.Get(), &rtv, rtvHandle(uint32_t(m_colorImages.size()) + i));
    }
  }

  // Descriptors of the sample image: the slot pair, the display SRV, and the
  // CPU-only UAV a clear needs when the image has no RTV.
  const Image& sample = getSampleImage(i);
  if(sample.allowsUav())
  {
    NVDX12_FAIL_RETURN(m_heap->writeImageSlot(m_colorSlots[i], sample));
    if((sample.flags & D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET) == 0)
    {
      device->CreateUnorderedAccessView(sample.image.Get(), nullptr, nullptr, uavClearHandle(i));
    }
  }
  else
  {
    NVDX12_FAIL_RETURN(m_heap->writeSampledImageDescriptor(m_colorSlots[i].srv, sample));
  }
  if(m_uiSlots[i] != kInvalidSlot)
  {
    NVDX12_FAIL_RETURN(m_heap->writeDisplayImage(m_uiSlots[i], sample));
  }

  return S_OK;
}

HRESULT nvdx12::RenderTarget::createDepthAttachment(Extent2D extent, bool resolveDepth, const std::string& namePrefix)
{
  ResourceAllocator* alloc  = m_createInfo.alloc;
  ID3D12Device*      device = alloc->getDevice();
  const DXGI_FORMAT  format = depthStencilFormat();

  const bool userDepthFlags = (m_createInfo.depthFlags != D3D12_RESOURCE_FLAG_NONE);
  const D3D12_RESOURCE_FLAGS depthFlags = userDepthFlags ? m_createInfo.depthFlags : D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;

  // Sampled depth is created in the typeless family of its format: DSV in
  // the depth format, SRV in its color twin.
  D3D12_RESOURCE_DESC1 imageDesc = DEFAULT_D3D12_RESOURCE_DESC1;
  imageDesc.Width                = extent.width;
  imageDesc.Height               = extent.height;
  imageDesc.DepthOrArraySize     = UINT16(m_createInfo.layerCount);
  imageDesc.Format               = sampleDepth() ? depthTypelessFormat(format) : format;
  imageDesc.SampleDesc.Count     = m_createInfo.msaa.samples;
  imageDesc.Flags                = depthFlags;
  NVDX12_FAIL_RETURN(alloc->createImage(m_depthImage, imageDesc));
  setDebugName(m_depthImage.image.Get(), namePrefix + ":depthImage");
  const D3D12_DEPTH_STENCIL_VIEW_DESC dsv = makeDsvDesc(format, m_createInfo.msaa.samples, m_createInfo.layerCount);
  device->CreateDepthStencilView(m_depthImage.image.Get(), &dsv, dsvHandle(0));

  if(resolveDepth)
  {
    D3D12_RESOURCE_DESC1 resolveDesc = imageDesc;
    resolveDesc.SampleDesc.Count     = 1;
    NVDX12_FAIL_RETURN(alloc->createImage(m_depthImageResolved, resolveDesc));
    setDebugName(m_depthImageResolved.image.Get(), namePrefix + ":depthImageResolved");
    const D3D12_DEPTH_STENCIL_VIEW_DESC resolveDsv = makeDsvDesc(format, 1, m_createInfo.layerCount);
    device->CreateDepthStencilView(m_depthImageResolved.image.Get(), &resolveDsv, dsvHandle(1));
  }

  if(sampleDepth())
  {
    const Image& sample = m_depthImageResolved ? m_depthImageResolved : m_depthImage;
    NVDX12_FAIL_RETURN(m_heap->writeSampledImageDescriptor(m_depthSlot, sample, format));
  }
  return S_OK;
}

void nvdx12::RenderTarget::transitionToWorkingLayouts(ID3D12GraphicsCommandList7* cmd)
{
  // From UNDEFINED with DISCARD, which initializes the new images.
  BarrierContainer barriers;
  const auto       addTransition = [&](Image& image, D3D12_BARRIER_LAYOUT layout) {
    if(image)
      barriers.appendOptionalLayoutTransition(image, makeImageBarrier(image, layout));
  };

  for(size_t i = 0; i < m_colorImages.size(); i++)
    addTransition(m_colorImages[i], m_createInfo.layouts.color);
  for(size_t i = 0; i < m_colorImagesResolved.size(); i++)
    addTransition(m_colorImagesResolved[i], D3D12_BARRIER_LAYOUT_RESOLVE_DEST);
  addTransition(m_depthImage, m_createInfo.layouts.depth);
  addTransition(m_depthImageResolved, D3D12_BARRIER_LAYOUT_RESOLVE_DEST);

  if(!barriers.imageBarriers.empty())
    barriers.cmdPipelineBarrier(cmd);
}

HRESULT nvdx12::RenderTarget::createResources(ID3D12GraphicsCommandList7* cmd, Extent2D extent)
{
  const std::string namePrefix = m_createInfo.debugName.empty() ? "RT" : "RenderTarget:" + m_createInfo.debugName;
  ID3D12Device*     device     = m_createInfo.alloc->getDevice();

  const bool resolveColor = wantsColorResolve();
  const bool resolveDepth = wantsDepthStencilResolve();  // one resolve image backs both aspects

  // CPU-only heaps for the RTVs (each color image, then each resolve image),
  // the DSVs (depth, then its resolve image) and, for color images without
  // RTVs, the UAVs of clears.
  const uint32_t colorCount = static_cast<uint32_t>(m_createInfo.colorFormats.size());
  const bool     colorRtvs  = m_createInfo.colorFlags == D3D12_RESOURCE_FLAG_NONE
                         || (m_createInfo.colorFlags & D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET) != 0;
  if(colorCount > 0 && colorRtvs)
  {
    const D3D12_DESCRIPTOR_HEAP_DESC rtvDesc{.Type           = D3D12_DESCRIPTOR_HEAP_TYPE_RTV,
                                             .NumDescriptors = (resolveColor ? 2 : 1) * colorCount};
    NVDX12_FAIL_RETURN(device->CreateDescriptorHeap(&rtvDesc, IID_PPV_ARGS(&m_rtvHeap)));
    setDebugName(m_rtvHeap.Get(), namePrefix + ":rtvHeap");
  }
  if(colorCount > 0 && !colorRtvs)
  {
    const D3D12_DESCRIPTOR_HEAP_DESC uavDesc{.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, .NumDescriptors = colorCount};
    NVDX12_FAIL_RETURN(device->CreateDescriptorHeap(&uavDesc, IID_PPV_ARGS(&m_uavClearHeap)));
    setDebugName(m_uavClearHeap.Get(), namePrefix + ":uavClearHeap");
  }
  if(hasDepthStencil())
  {
    const D3D12_DESCRIPTOR_HEAP_DESC dsvDesc{.Type = D3D12_DESCRIPTOR_HEAP_TYPE_DSV, .NumDescriptors = 2};
    NVDX12_FAIL_RETURN(device->CreateDescriptorHeap(&dsvDesc, IID_PPV_ARGS(&m_dsvHeap)));
    setDebugName(m_dsvHeap.Get(), namePrefix + ":dsvHeap");
  }

  for(uint32_t i = 0; i < colorCount; ++i)
  {
    NVDX12_FAIL_RETURN(createColorAttachment(i, extent, resolveColor, namePrefix));
  }

  if(hasDepthStencil())
    NVDX12_FAIL_RETURN(createDepthAttachment(extent, resolveDepth, namePrefix));

  transitionToWorkingLayouts(cmd);
  m_extent         = extent;
  m_resourcesDirty = false;
  return S_OK;
}

//--------------------------------------------------------------------------------------------------
// Usage example (raster flow as nvvk's; MSAA optional)
//--------------------------------------------------------------------------------------------------
[[maybe_unused]] static void usage_RenderTarget()
{
  nvdx12::ResourceAllocator allocator;  // EX: initialized with the device and the descriptor heap

  nvdx12::RenderTarget renderTarget;
  NVDX12_CHECK(renderTarget.init({
      .alloc        = &allocator,
      .colorFormats = {DXGI_FORMAT_R16G16B16A16_FLOAT},
      .depthFormat  = DXGI_FORMAT_D32_FLOAT,
      .debugName    = "MyOffscreen",
  }));

  // Optional MSAA; takes effect on the next update().
  NVDX12_CHECK(renderTarget.setMsaa({.samples = 4, .colorResolve = D3D12_RESOLVE_MODE_AVERAGE}));

  ID3D12GraphicsCommandList7* cmd = nullptr;  // EX: a command list in the recording state
  NVDX12_CHECK(renderTarget.update(cmd, nvdx12::Extent2D{1920, 1080}));

  // Per frame: fill the attachments, set clear values, begin the render pass.
  nvdx12::RenderTargetState state;
  renderTarget.fillState(state);
  const float clearColor[4] = {0.1F, 0.1F, 0.1F, 1.0F};
  std::copy_n(clearColor, 4, state.colorAttachments[0].BeginningAccess.Clear.ClearValue.Color);
  state.depthStencilAttachment.DepthBeginningAccess.Clear.ClearValue.DepthStencil = {1.0F, 0};
  state.cmdBeginRendering(cmd, {});  // default AttachmentOps: CLEAR
  nvdx12::GraphicsPipelineState::cmdSetViewportAndScissor(cmd, renderTarget.getSize());

  cmd->EndRenderPass();

  // Sampling the resolved image: it rests in RESOLVE_DEST.
  nvdx12::cmdImageBarrier(cmd, renderTarget.getSampleImage(), renderTarget.getColorSampleLayout());
  // HEAP_SRV(renderTarget.getColorImageSlot().srv) in shaders (shaders/bindless.h).

  // ImGui: ImGui::Image(ImTextureID(renderTarget.getUiImageView().ptr), size), with the image in SHADER_RESOURCE.

  renderTarget.deinit();
}
