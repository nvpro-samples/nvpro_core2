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

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "basic_types.hpp"

#include "default_structs.hpp"
#include "descriptor_heap.hpp"
#include "resource_allocator.hpp"

namespace nvdx12 {

class GraphicsPipelineCreator;

/*-------------------------------------------------------------------------------------------------
# class nvdx12::RenderTarget / struct nvdx12::RenderTargetState

> Offscreen color targets (one or more) and an optional depth/stencil target
> at one size, the D3D12 counterpart of nvvk::RenderTarget, with the render
> pass state it fills (nvvk::RenderTargetState).

Common configurations:
- Raster (default): `layouts.color` RENDER_TARGET, default flags, CLEAR via
  RenderTargetState::cmdBeginRendering().
- Compute / ray tracing: `layouts.color` UNORDERED_ACCESS, storage through the
  UAV of getColorImageSlot(); add `colorFlags` ALLOW_UNORDERED_ACCESS alone
  for targets that are never rendered.

Lifecycle: init() -> update(cmd, extent) on resize -> fillState() + cmdBeginRendering().
update() does not clear the images; use cmdClear() or CLEAR in AttachmentOps.
Display: getUiImageView() (alpha one, ImGui). Shader access: getColorImageSlot().
MSAA: getColorImage() is the multisampled attachment, getColorImageResolved()
the resolve target (an empty Image without MSAA); getSampleImage() and the
descriptor slots pick the resolve image when it exists, else the attachment.

Each image is published through the descriptor heap, in slots acquired by
init() that keep their numbers across update() (only the descriptors are
rewritten, the GPU must be done with the previous images):
- Color: an ImageSlot of the sample image (SRV, and the UAV when the image
  allows unordered access) and a display SRV with alpha forced to one.
- Depth: an SRV of the depth sample image, in the color twin of the depth
  format (the depth images are created in its typeless family; none with
  DENY_SHADER_RESOURCE in `depthFlags`).

Differences with nvvk that follow from D3D12:
- Layouts are D3D12_BARRIER_LAYOUTs and there is no layout for every use, as
  VK_IMAGE_LAYOUT_GENERAL is: `layouts.color` is RENDER_TARGET (default, the
  layout render passes need) or UNORDERED_ACCESS (every color image then
  needs a UAV: no MSAA, formats with typed UAV support), `layouts.depth` is
  DEPTH_STENCIL_WRITE and the sample layout is SHADER_RESOURCE. Resolve
  images rest in RESOLVE_DEST, the layout a render-pass resolve writes. The
  images track their layout (nvdx12::Image), so the caller transitions them
  with cmdImageBarrier for other uses; cmdClear() returns them to the
  working layouts.
- Usages are D3D12_RESOURCE_FLAGS: `colorFlags` 0 means ALLOW_RENDER_TARGET,
  plus ALLOW_UNORDERED_ACCESS with an UNORDERED_ACCESS layout; `depthFlags` 0
  means ALLOW_DEPTH_STENCIL. Explicit color flags need ALLOW_RENDER_TARGET or
  ALLOW_UNORDERED_ACCESS, so that cmdClear() can clear the image. A color
  format cannot be DXGI_FORMAT_UNKNOWN: render passes bind every target.
- Views are descriptors: getColorAttachmentView() is the RTV and
  getDepthImageView() the DSV (CPU handles in heaps the target owns),
  getColorImageSlot() / getDepthImageSlot() are the shader-visible slots
  (what getSampleView, getColorStorageImageInfo and
  getColorSampleDescriptorImageInfo give in nvvk), and getUiImageView() is the
  GPU handle of the display SRV, whose `ptr` is the ImGui ImTextureID
  (nvapp::ImTexture has no D3D12 counterpart). There are no resolve views.
  Layered targets have no display SRV: a Texture2D view needs a single-layer
  resource.
- The getters return nvdx12::Image (nvvk's handles), D3D12MA::Allocation*
  (nvvk's VmaAllocation) and an Extent2D; results are HRESULTs.
- MSAA: `samples` is a count and the resolve modes are
  std::optional<D3D12_RESOLVE_MODE> (nullopt for VK_RESOLVE_MODE_NONE):
  MIN, MAX or AVERAGE (not for integer formats, so never for stencil).
  Resolves run at the end of the render pass (RESOLVE ending accesses, depth
  and stencil in the color format of their plane) without a device
  capability query. A multisampled render target needs the RENDER_TARGET layout.
- RenderTargetState holds D3D12_RENDER_PASS_RENDER_TARGET_DESCs and one
  D3D12_RENDER_PASS_DEPTH_STENCIL_DESC for depth and stencil;
  cmdBeginRendering() records BeginRenderPass (the caller records
  EndRenderPass) and needs the state to stay alive until then. Clear values
  go in BeginningAccess.Clear.ClearValue (fillState sets their formats);
  `renderArea` is not part of a D3D12 render pass, it is the full extent for
  RSSetScissorRects. AttachmentOps carries the render pass flags (D3D12
  extra, e.g. ALLOW_UAV_WRITES for pixel shaders writing UAVs).
- fillPipelineRenderingCreateInfo() fills the formats of a
  GraphicsPipelineCreator (colorFormats, renderingState).
- CreateInfo has no `device` (the allocator's) and takes the descriptor heap
  (the allocator's when null, D3D12 extra).
- cmdClear() clears through RTVs and DSVs, or through the UAV of an image
  without RTV (it then binds the descriptor heap's heaps to `cmd`), and takes
  float color values (converted for integer formats, as
  ClearRenderTargetView does) and a D3D12_DEPTH_STENCIL_VALUE.

See usage_RenderTarget() in render_target.cpp.
-------------------------------------------------------------------------------------------------*/

//
// RenderTargetState - transient render-pass state filled from a RenderTarget.
//
struct RenderTargetState
{
  D3D12_RECT renderArea{};
  uint32_t   layerCount{1};
  uint32_t   sampleCount{1};

  uint32_t                                          colorCount{};
  std::vector<DXGI_FORMAT>                          colorFormats;
  DXGI_FORMAT                                       depthFormat{DXGI_FORMAT_UNKNOWN};
  DXGI_FORMAT                                       stencilFormat{DXGI_FORMAT_UNKNOWN};
  std::vector<D3D12_RENDER_PASS_RENDER_TARGET_DESC> colorAttachments;
  D3D12_RENDER_PASS_DEPTH_STENCIL_DESC              depthStencilAttachment{};
  // Per-layer resolve regions the RESOLVE ending accesses point to: color
  // attachment i at [i * layerCount], then depth, then stencil.
  std::vector<D3D12_RENDER_PASS_ENDING_ACCESS_RESOLVE_SUBRESOURCE_PARAMETERS> resolveSubresources;

  RenderTargetState();

  void clear();

  void fillPipelineRenderingCreateInfo(GraphicsPipelineCreator& creator) const;

  struct AttachmentOps
  {
    D3D12_RENDER_PASS_BEGINNING_ACCESS_TYPE colorLoad    = D3D12_RENDER_PASS_BEGINNING_ACCESS_TYPE_CLEAR;
    D3D12_RENDER_PASS_ENDING_ACCESS_TYPE    colorStore   = D3D12_RENDER_PASS_ENDING_ACCESS_TYPE_PRESERVE;
    D3D12_RENDER_PASS_BEGINNING_ACCESS_TYPE depthLoad    = D3D12_RENDER_PASS_BEGINNING_ACCESS_TYPE_CLEAR;
    D3D12_RENDER_PASS_ENDING_ACCESS_TYPE    depthStore   = D3D12_RENDER_PASS_ENDING_ACCESS_TYPE_PRESERVE;
    D3D12_RENDER_PASS_BEGINNING_ACCESS_TYPE stencilLoad  = D3D12_RENDER_PASS_BEGINNING_ACCESS_TYPE_DISCARD;
    D3D12_RENDER_PASS_ENDING_ACCESS_TYPE    stencilStore = D3D12_RENDER_PASS_ENDING_ACCESS_TYPE_DISCARD;
    D3D12_RENDER_PASS_FLAGS                 flags        = D3D12_RENDER_PASS_FLAG_NONE;
  };
  // An attachment with a resolve ends in RESOLVE; its store op then tells
  // whether the multisampled source is preserved.
  void cmdBeginRendering(ID3D12GraphicsCommandList7* cmd, const AttachmentOps& ops);
};

/***************************************************************
 ***************************************************************
 ********************    RenderTarget      ********************
 ***************************************************************
 ***************************************************************/

class RenderTarget
{
public:
  struct LayoutConfig
  {
    D3D12_BARRIER_LAYOUT color{D3D12_BARRIER_LAYOUT_RENDER_TARGET};  // or UNORDERED_ACCESS
    D3D12_BARRIER_LAYOUT depth{D3D12_BARRIER_LAYOUT_DEPTH_STENCIL_WRITE};
    // D3D12_BARRIER_LAYOUT_UNDEFINED: SHADER_RESOURCE, the only sample layout.
    D3D12_BARRIER_LAYOUT colorSample{D3D12_BARRIER_LAYOUT_UNDEFINED};
  };

  struct MsaaConfig
  {
    // No resolve by default (opt-in per aspect).
    uint32_t                          samples{1};
    std::optional<D3D12_RESOLVE_MODE> colorResolve{};    // required when samples > 1 (e.g. D3D12_RESOLVE_MODE_AVERAGE)
    std::optional<D3D12_RESOLVE_MODE> depthResolve{};    // MIN, MAX or AVERAGE
    std::optional<D3D12_RESOLVE_MODE> stencilResolve{};  // MIN or MAX (integer)
  };

  struct CreateInfo
  {
    ResourceAllocator*       alloc{};
    DescriptorHeap*          descriptorHeap{};  // slots of the images; the allocator's heap when null
    std::vector<DXGI_FORMAT> colorFormats;
    DXGI_FORMAT              depthFormat{DXGI_FORMAT_UNKNOWN};
    DXGI_FORMAT              stencilFormat{DXGI_FORMAT_UNKNOWN};
    uint32_t                 layerCount{1};
    D3D12_RESOURCE_FLAGS colorFlags{};  // 0 = default (+ALLOW_UNORDERED_ACCESS per format when layouts.color is UNORDERED_ACCESS)
    D3D12_RESOURCE_FLAGS depthFlags{};
    MsaaConfig           msaa{};
    LayoutConfig         layouts{};
    std::string          debugName{};
  };

  RenderTarget() = default;
  ~RenderTarget();

  RenderTarget(const RenderTarget&)            = delete;
  RenderTarget& operator=(const RenderTarget&) = delete;
  RenderTarget(RenderTarget&&)                 = delete;
  RenderTarget& operator=(RenderTarget&&)      = delete;

  // Validates and stores the configuration and acquires the descriptor slots.
  HRESULT init(const CreateInfo& info);
  void    deinit();

  // Takes effect on the next update(). Prefer setMsaa() when changing sample count and resolve together.
  HRESULT setMsaa(const MsaaConfig& msaa);
  HRESULT setSampleCount(uint32_t                          sampleCount,
                         std::optional<D3D12_RESOLVE_MODE> colorResolveMode = std::nullopt,
                         std::optional<D3D12_RESOLVE_MODE> depthResolveMode = std::nullopt);

  // No-op when extent and config are unchanged; recreates the images on resize or setMsaa() and
  // leaves them in their working layouts. A zero extent fails.
  HRESULT update(ID3D12GraphicsCommandList7* cmd, Extent2D extent);

  void cmdClear(ID3D12GraphicsCommandList7* cmd,
                const std::array<float, 4>& colorClear = {0.F, 0.F, 0.F, 0.F},
                D3D12_DEPTH_STENCIL_VALUE   depthClear = DEFAULT_D3D12_DEPTH_STENCIL_VALUE);

  // fillState() options:
  //   depthOnly=false, addResolve=true  — full color pass with MSAA resolve (default).
  //   depthOnly=true                    — depth/stencil only (e.g. shadow pre-pass).
  //   addResolve=false                  — intermediate MSAA pass; resolve on the final pass only.
  void fillState(RenderTargetState& state, bool depthOnly = false, bool addResolve = true) const;

  Extent2D getSize() const;

  // --- Color: render attachment (multisampled when MSAA) vs resolve / readback ---
  const Image& getColorImage(uint32_t i = 0) const;  // render target image (MSAA surface when msaa.samples > 1)
  Image&       getColorImage(uint32_t i = 0);        // the barrier helpers track the layout in the image
  D3D12_CPU_DESCRIPTOR_HANDLE getColorAttachmentView(uint32_t i = 0) const;  // RTV; null without ALLOW_RENDER_TARGET

  const Image& getColorImageResolved(uint32_t i = 0) const;  // empty Image when there is no resolve image (samples == 1)
  Image& getColorImageResolved(uint32_t i = 0);

  // Single-sample result for sampling / save / copy: resolve image if MSAA, else same as getColorImage().
  const Image& getSampleImage(uint32_t i = 0) const;
  Image&       getSampleImage(uint32_t i = 0);

  // Display SRV with alpha one: `ptr` is the ImGui ImTextureID.
  D3D12_GPU_DESCRIPTOR_HANDLE getUiImageView(uint32_t i = 0) const;

  // SRV of the sample image, and its UAV when it allows unordered access (storage write vs sample read).
  ImageSlot getColorImageSlot(uint32_t i = 0) const;

  D3D12_BARRIER_LAYOUT getColorLayout() const;
  D3D12_BARRIER_LAYOUT getDepthLayout() const;
  D3D12_BARRIER_LAYOUT getColorSampleLayout() const;

  // --- Depth ---
  const Image&                getDepthImage() const;  // typeless unless depthFlags deny shader resources
  Image&                      getDepthImage();
  D3D12_CPU_DESCRIPTOR_HANDLE getDepthImageView() const;  // DSV in the depth format
  const Image&                getDepthImageResolved() const;
  Image&                      getDepthImageResolved();
  uint32_t                    getDepthImageSlot() const;  // SRV of the resolved depth if any, else of the depth image

  DXGI_FORMAT getColorFormat(uint32_t i = 0) const;
  DXGI_FORMAT getDepthFormat() const;
  uint32_t    getSampleCount() const;
  uint32_t    getLayerCount() const;
  float       getAspectRatio() const;

  // D3D12MA allocations for memory tracking.
  D3D12MA::Allocation* getColorAllocation(uint32_t i = 0) const;
  D3D12MA::Allocation* getColorAllocationResolved(uint32_t i = 0) const;
  D3D12MA::Allocation* getDepthAllocation() const;
  D3D12MA::Allocation* getDepthAllocationResolved() const;

private:
  struct ColorFlagsResult
  {
    D3D12_RESOURCE_FLAGS renderFlags{};
    D3D12_RESOURCE_FLAGS resolveFlags{};
  };

  static HRESULT              validateCreateInfo(const CreateInfo& info);
  static D3D12_BARRIER_LAYOUT effectiveColorSampleLayout(const CreateInfo& info);
  static HRESULT resolveColorFlags(const CreateInfo& info, DXGI_FORMAT format, ID3D12Device* device, ColorFlagsResult& out);

  // Single source of truth for the depth/stencil format and the MSAA resolve decisions.
  DXGI_FORMAT depthStencilFormat() const;
  bool        hasDepthStencil() const;
  bool        isMsaa() const;
  bool        wantsColorResolve() const;
  bool        wantsDepthResolve() const;
  bool        wantsStencilResolve() const;
  bool        wantsDepthStencilResolve() const;  // true if either depth or stencil resolves (shared resolve image)
  bool        sampleDepth() const;               // the depth images get an SRV

  HRESULT createColorAttachment(uint32_t i, Extent2D extent, bool resolveColor, const std::string& namePrefix);
  HRESULT createDepthAttachment(Extent2D extent, bool resolveDepth, const std::string& namePrefix);
  void    transitionToWorkingLayouts(ID3D12GraphicsCommandList7* cmd);

  HRESULT createResources(ID3D12GraphicsCommandList7* cmd, Extent2D extent);
  void    destroyResources();
  void    releaseSlots();

  D3D12_CPU_DESCRIPTOR_HANDLE rtvHandle(uint32_t index) const;  // color i at i, its resolve image at colorCount + i
  D3D12_CPU_DESCRIPTOR_HANDLE dsvHandle(uint32_t index) const;  // depth at 0, its resolve image at 1
  D3D12_CPU_DESCRIPTOR_HANDLE uavClearHandle(uint32_t index) const;

  CreateInfo      m_createInfo{};
  DescriptorHeap* m_heap{};  // m_createInfo.descriptorHeap or the allocator's
  Extent2D        m_extent{};
  bool            m_resourcesDirty{true};  // set by init/setMsaa; cleared after successful createResources()

  std::vector<Image> m_colorImages{};
  Image              m_depthImage{};
  std::vector<Image> m_colorImagesResolved{};
  Image              m_depthImageResolved{};

  ComPtr<ID3D12DescriptorHeap> m_rtvHeap;       // RTVs for rendering and clears
  ComPtr<ID3D12DescriptorHeap> m_dsvHeap;       // DSVs for rendering and clears
  ComPtr<ID3D12DescriptorHeap> m_uavClearHeap;  // CPU-only UAV of each color sample image, for clears without an RTV

  std::vector<ImageSlot> m_colorSlots;               // SRV / UAV of each color sample image
  std::vector<uint32_t>  m_uiSlots;                  // display SRV of each color sample image
  uint32_t               m_depthSlot{kInvalidSlot};  // SRV of the depth sample image
};

}  // namespace nvdx12
