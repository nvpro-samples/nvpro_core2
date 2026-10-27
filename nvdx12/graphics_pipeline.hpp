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
#include <span>
#include <vector>

#include "basic_types.hpp"
#include "resources.hpp"
#include "shader_compiler.hpp"

namespace nvdx12 {

/*-------------------------------------------------------------------------------------------------
# Graphics pipelines

> The rasterization state and the graphics PSO creation, the D3D12
> counterparts of nvvk::GraphicsPipelineState and nvvk::GraphicsPipelineCreator.

`GraphicsPipelineState` holds the fixed-function states under nvvk's member
names, as D3D12 structures, with nvvk's defaults: solid fill, back-face
culling with counter-clockwise front faces, triangle list, one sample, depth
test and write with LESS_EQUAL, no stencil, one color attachment without
blending. Its members are public and changed directly.

`GraphicsPipelineCreator` holds what belongs to the pipeline rather than to
the state: `pipelineInfo` (the root signature, as the pipeline layout of
nvvk's pipelineInfo, plus node mask, cached PSO and flags), the target
formats (`colorFormats`, `renderingState.depthAttachmentFormat`) and the
shaders, and creates the PSO. Vertices are usually pulled from buffers
through their shader addresses, so there is no input layout by default; an
input layout (`vertexAttributes`) needs a root signature created with
D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT, which the
bindless one of pipeline.hpp does not set.

Differences with nvvk that follow from D3D12:
- The states are D3D12 structures (D3D12_RASTERIZER_DESC,
  D3D12_DEPTH_STENCIL_DESC, ...) under nvvk's member names; the color blend
  enables / write masks / equations are kept per attachment as in nvvk and
  become D3D12_RENDER_TARGET_BLEND_DESC entries (attachments beyond the
  vectors get no blending and a full write mask).
- `vertexAttributes` are D3D12_INPUT_ELEMENT_DESC: an element carries its
  input slot and instance step rate, and the stride goes with the vertex
  buffer view, so there are no separate vertex bindings.
- The topology, blend constants and stencil reference are command-list
  state in D3D12: `cmdApplyAllStates` sets them (the PSO only takes the
  topology class); viewport and scissor are always command-list state
  (`cmdSetViewportAndScissor`), as nvvk's dynamic viewport and scissor.
- The default depth format is D32_FLOAT, the D3D12 depth buffer of choice,
  where nvvk defaults to X8_D24_UNORM. Depth testing is independent of the
  format as in nvvk; without a depth buffer (DXGI_FORMAT_UNKNOWN) the PSO
  is created with depth and stencil testing off, as Vulkan ignores them
  without a depth attachment.
- Shaders are DXIL blobs compiled for one entry point, so addShader takes
  no entry name and no specialization info (use ShaderDefine).

Left out: shader objects (cmdBindShaders, VK_EXT_shader_object), dynamic
states (cmdApplyDynamicState(s), dynamicStateValues: D3D12 PSOs have no
dynamic state beyond the command-list states above), pipeline caches (set
pipelineInfo.CachedPSO), line rasterization, tessellation domain origin and
fragment shading rate states, and task / mesh shaders, which need a pipeline
state stream (D3DX12_MESH_SHADER_PIPELINE_STATE_DESC).

See usage_GraphicsPipeline() in graphics_pipeline.cpp.
-------------------------------------------------------------------------------------------------*/

// Programmable stages of a graphics PSO, in the order of D3D12_GRAPHICS_PIPELINE_STATE_DESC.
enum class GraphicsShaderStage : uint32_t
{
  eVertex = 0,
  ePixel,
  eDomain,
  eHull,
  eGeometry,
  eCount
};

// Blend factors and operations of one color attachment (VkColorBlendEquationEXT).
// At namespace scope: GraphicsPipelineState uses its default member
// initializers in its own member initializers.
struct ColorBlendEquation
{
  D3D12_BLEND    srcColorBlendFactor{D3D12_BLEND_ZERO};
  D3D12_BLEND    dstColorBlendFactor{D3D12_BLEND_ZERO};
  D3D12_BLEND_OP colorBlendOp{D3D12_BLEND_OP_ADD};
  D3D12_BLEND    srcAlphaBlendFactor{D3D12_BLEND_ZERO};
  D3D12_BLEND    dstAlphaBlendFactor{D3D12_BLEND_ZERO};
  D3D12_BLEND_OP alphaBlendOp{D3D12_BLEND_OP_ADD};
};

class GraphicsPipelineState
{
public:
  // This object does not cover viewport and scissor states, which D3D12 always
  // sets on the command list; cmdSetViewportAndScissor sets one viewport and
  // scissor covering viewportSize.
  static void cmdSetViewportAndScissor(ID3D12GraphicsCommandList* cmd, const Extent2D& viewportSize, float minDepth = 0.0f, float maxDepth = 1.0f);

  using ColorBlendEquation = nvdx12::ColorBlendEquation;

  UINT sampleMask{~0U};

  struct InputAssemblyState
  {
    // Command-list topology (cmdApplyAllStates); the PSO takes its topology class.
    D3D12_PRIMITIVE_TOPOLOGY topology{D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST};
    // Primitive restart (nvvk: primitiveRestartEnable) for strip topologies.
    D3D12_INDEX_BUFFER_STRIP_CUT_VALUE stripCutValue{D3D12_INDEX_BUFFER_STRIP_CUT_VALUE_DISABLED};
  } inputAssemblyState;

  D3D12_RASTERIZER_DESC rasterizationState{
      .FillMode              = D3D12_FILL_MODE_SOLID,
      .CullMode              = D3D12_CULL_MODE_BACK,
      .FrontCounterClockwise = TRUE,
      .DepthBias             = D3D12_DEFAULT_DEPTH_BIAS,
      .DepthBiasClamp        = D3D12_DEFAULT_DEPTH_BIAS_CLAMP,
      .SlopeScaledDepthBias  = D3D12_DEFAULT_SLOPE_SCALED_DEPTH_BIAS,
      .DepthClipEnable       = TRUE,
      .MultisampleEnable     = FALSE,
      .AntialiasedLineEnable = FALSE,
      .ForcedSampleCount     = 0,
      .ConservativeRaster    = D3D12_CONSERVATIVE_RASTERIZATION_MODE_OFF,
  };

  struct MultisampleState
  {
    DXGI_SAMPLE_DESC rasterizationSamples{.Count = 1, .Quality = 0};
    BOOL             alphaToCoverageEnable{FALSE};
  } multisampleState;

  D3D12_DEPTH_STENCIL_DESC depthStencilState{
      .DepthEnable      = TRUE,
      .DepthWriteMask   = D3D12_DEPTH_WRITE_MASK_ALL,
      .DepthFunc        = D3D12_COMPARISON_FUNC_LESS_EQUAL,
      .StencilEnable    = FALSE,
      .StencilReadMask  = D3D12_DEFAULT_STENCIL_READ_MASK,
      .StencilWriteMask = D3D12_DEFAULT_STENCIL_WRITE_MASK,
      .FrontFace = {D3D12_STENCIL_OP_KEEP, D3D12_STENCIL_OP_KEEP, D3D12_STENCIL_OP_KEEP, D3D12_COMPARISON_FUNC_ALWAYS},
      .BackFace  = {D3D12_STENCIL_OP_KEEP, D3D12_STENCIL_OP_KEEP, D3D12_STENCIL_OP_KEEP, D3D12_COMPARISON_FUNC_ALWAYS},
  };
  // Command-list state in D3D12 (nvvk: depthStencilState.front/back.reference).
  uint32_t stencilReference{0};

  struct ColorBlendState
  {
    BOOL                 logicOpEnable{FALSE};
    D3D12_LOGIC_OP       logicOp{D3D12_LOGIC_OP_NOOP};  // D3D12's default (nvvk: CLEAR, unused while disabled)
    std::array<float, 4> blendConstants{1.0f, 1.0f, 1.0f, 1.0f};  // command-list state (cmdApplyAllStates)
  } colorBlendState;

  // By default one color attachment with blending disabled, per attachment as in nvvk.
  std::vector<BOOL>               colorBlendEnables{FALSE};
  std::vector<UINT8>              colorWriteMasks{D3D12_COLOR_WRITE_ENABLE_ALL};
  std::vector<ColorBlendEquation> colorBlendEquations{ColorBlendEquation{}};

  // Input layout; empty for vertex pulling through shader addresses.
  std::vector<D3D12_INPUT_ELEMENT_DESC> vertexAttributes{};

  // Sets the states D3D12 keeps on the command list: primitive topology,
  // blend constants and stencil reference.
  void cmdApplyAllStates(ID3D12GraphicsCommandList* cmd) const;
};

class GraphicsPipelineCreator
{
public:
  // pRootSignature (the pipeline layout), NodeMask, CachedPSO and Flags are
  // used as set; every other field is filled from the graphics state, the
  // formats and the shaders.
  D3D12_GRAPHICS_PIPELINE_STATE_DESC pipelineInfo{};

  // Added to pipelineInfo.Flags (nvvk: flags2).
  D3D12_PIPELINE_STATE_FLAGS flags{D3D12_PIPELINE_STATE_FLAG_NONE};

  struct RenderingState
  {
    // DSVFormat; DXGI_FORMAT_UNKNOWN for no depth buffer. A combined
    // depth-stencil format covers nvvk's stencilAttachmentFormat.
    DXGI_FORMAT depthAttachmentFormat{DXGI_FORMAT_D32_FLOAT};
  } renderingState;

  // RTVFormats, at most D3D12_SIMULTANEOUS_RENDER_TARGET_COUNT; empty for a depth-only pass.
  std::vector<DXGI_FORMAT> colorFormats{DXGI_FORMAT_R8G8B8A8_UNORM};

  // Shader bytecode is referenced, not copied: the blob or the span must
  // stay valid until createGraphicsPipeline returns. An invalid blob (a
  // failed compilation) makes createGraphicsPipeline fail.
  void clearShaders();
  void addShader(GraphicsShaderStage stage, const ShaderBlob& blob);
  void addShader(GraphicsShaderStage stage, std::span<const uint8_t> dxil);

  // None of the public members of `graphicsState` or this object are changed.
  HRESULT createGraphicsPipeline(ID3D12Device*                device,
                                 const GraphicsPipelineState& graphicsState,
                                 ComPtr<ID3D12PipelineState>& pipeline) const;

private:
  std::array<D3D12_SHADER_BYTECODE, uint32_t(GraphicsShaderStage::eCount)> m_shaders{};  // per stage; empty when unused
  bool m_invalidShader{false};                                                           // an added blob was not valid
};

}  // namespace nvdx12
