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

#include "graphics_pipeline.hpp"

#include <cassert>

#include <nvutils/logger.hpp>

#include "check_error.hpp"
#include "debug_util.hpp"
#include "descriptor_heap.hpp"  // the usage example
#include "pipeline.hpp"

namespace nvdx12 {

//--------------------------------------------------------------------------------------------------
// GraphicsPipelineState

void GraphicsPipelineState::cmdSetViewportAndScissor(ID3D12GraphicsCommandList* cmd, const Extent2D& viewportSize, float minDepth, float maxDepth)
{
  const D3D12_VIEWPORT viewport{0.0f, 0.0f, float(viewportSize.width), float(viewportSize.height), minDepth, maxDepth};
  const D3D12_RECT     scissor{0, 0, LONG(viewportSize.width), LONG(viewportSize.height)};
  cmd->RSSetViewports(1, &viewport);
  cmd->RSSetScissorRects(1, &scissor);
}

void GraphicsPipelineState::cmdApplyAllStates(ID3D12GraphicsCommandList* cmd) const
{
  cmd->IASetPrimitiveTopology(inputAssemblyState.topology);
  cmd->OMSetBlendFactor(colorBlendState.blendConstants.data());
  cmd->OMSetStencilRef(stencilReference);
}

//--------------------------------------------------------------------------------------------------
// GraphicsPipelineCreator

void GraphicsPipelineCreator::addShader(GraphicsShaderStage stage, const ShaderBlob& blob)
{
  if(!blob.valid())
  {
    LOGE("Graphics pipeline: shader blob is not valid\n%s\n", blob.errors.c_str());
    m_invalidShader = true;
    return;
  }
  addShader(stage, blob.dxil);
}

void GraphicsPipelineCreator::addShader(GraphicsShaderStage stage, std::span<const uint8_t> dxil)
{
  assert(stage < GraphicsShaderStage::eCount);
  m_shaders[uint32_t(stage)] = D3D12_SHADER_BYTECODE{dxil.data(), dxil.size()};
}

void GraphicsPipelineCreator::clearShaders()
{
  m_shaders       = {};
  m_invalidShader = false;
}

// The PSO topology class of a command-list topology.
static D3D12_PRIMITIVE_TOPOLOGY_TYPE topologyType(D3D12_PRIMITIVE_TOPOLOGY topology)
{
  switch(topology)
  {
    case D3D_PRIMITIVE_TOPOLOGY_POINTLIST:
      return D3D12_PRIMITIVE_TOPOLOGY_TYPE_POINT;
    case D3D_PRIMITIVE_TOPOLOGY_LINELIST:
    case D3D_PRIMITIVE_TOPOLOGY_LINESTRIP:
    case D3D_PRIMITIVE_TOPOLOGY_LINELIST_ADJ:
    case D3D_PRIMITIVE_TOPOLOGY_LINESTRIP_ADJ:
      return D3D12_PRIMITIVE_TOPOLOGY_TYPE_LINE;
    case D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST:
    case D3D_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP:
    case D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST_ADJ:
    case D3D_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP_ADJ:
      return D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    default:
      return topology >= D3D_PRIMITIVE_TOPOLOGY_1_CONTROL_POINT_PATCHLIST ? D3D12_PRIMITIVE_TOPOLOGY_TYPE_PATCH :
                                                                            D3D12_PRIMITIVE_TOPOLOGY_TYPE_UNDEFINED;
  }
}

HRESULT GraphicsPipelineCreator::createGraphicsPipeline(ID3D12Device*                device,
                                                        const GraphicsPipelineState& graphicsState,
                                                        ComPtr<ID3D12PipelineState>& pipeline) const
{
  pipeline.Reset();
  if(m_invalidShader)
  {
    LOGE("Graphics pipeline: a shader blob was not valid\n");
    return E_INVALIDARG;
  }
  if(m_shaders[uint32_t(GraphicsShaderStage::eVertex)].BytecodeLength == 0)
  {
    LOGE("Graphics pipeline: no vertex shader\n");
    return E_INVALIDARG;
  }
  if(colorFormats.size() > D3D12_SIMULTANEOUS_RENDER_TARGET_COUNT)
  {
    LOGE("Graphics pipeline: %zu color formats, at most %u\n", colorFormats.size(), D3D12_SIMULTANEOUS_RENDER_TARGET_COUNT);
    return E_INVALIDARG;
  }

  D3D12_GRAPHICS_PIPELINE_STATE_DESC desc = pipelineInfo;
  desc.VS                                 = m_shaders[uint32_t(GraphicsShaderStage::eVertex)];
  desc.PS                                 = m_shaders[uint32_t(GraphicsShaderStage::ePixel)];
  desc.DS                                 = m_shaders[uint32_t(GraphicsShaderStage::eDomain)];
  desc.HS                                 = m_shaders[uint32_t(GraphicsShaderStage::eHull)];
  desc.GS                                 = m_shaders[uint32_t(GraphicsShaderStage::eGeometry)];
  desc.StreamOutput                       = {};
  desc.SampleMask                         = graphicsState.sampleMask;
  desc.RasterizerState                    = graphicsState.rasterizationState;
  desc.DepthStencilState                  = graphicsState.depthStencilState;
  desc.InputLayout = {graphicsState.vertexAttributes.empty() ? nullptr : graphicsState.vertexAttributes.data(),
                      UINT(graphicsState.vertexAttributes.size())};
  desc.IBStripCutValue       = graphicsState.inputAssemblyState.stripCutValue;
  desc.PrimitiveTopologyType = topologyType(graphicsState.inputAssemblyState.topology);
  desc.SampleDesc            = graphicsState.multisampleState.rasterizationSamples;
  desc.Flags |= flags;

  // Blending is per attachment (IndependentBlendEnable), as nvvk's per-attachment
  // states; attachments without an entry get no blending and a full write mask.
  desc.BlendState.AlphaToCoverageEnable  = graphicsState.multisampleState.alphaToCoverageEnable;
  desc.BlendState.IndependentBlendEnable = TRUE;
  for(uint32_t i = 0; i < D3D12_SIMULTANEOUS_RENDER_TARGET_COUNT; i++)
  {
    D3D12_RENDER_TARGET_BLEND_DESC& target = desc.BlendState.RenderTarget[i];
    const ColorBlendEquation        equation =
        i < graphicsState.colorBlendEquations.size() ? graphicsState.colorBlendEquations[i] : ColorBlendEquation{};
    const UINT8 writeMask = i < graphicsState.colorWriteMasks.size() ? graphicsState.colorWriteMasks[i] :
                                                                       UINT8(D3D12_COLOR_WRITE_ENABLE_ALL);
    target.BlendEnable    = i < graphicsState.colorBlendEnables.size() ? graphicsState.colorBlendEnables[i] : FALSE;
    target.LogicOpEnable  = graphicsState.colorBlendState.logicOpEnable;
    target.SrcBlend       = equation.srcColorBlendFactor;
    target.DestBlend      = equation.dstColorBlendFactor;
    target.BlendOp        = equation.colorBlendOp;
    target.SrcBlendAlpha  = equation.srcAlphaBlendFactor;
    target.DestBlendAlpha = equation.dstAlphaBlendFactor;
    target.BlendOpAlpha   = equation.alphaBlendOp;
    target.LogicOp        = graphicsState.colorBlendState.logicOp;
    target.RenderTargetWriteMask = writeMask;
  }
  if(graphicsState.colorBlendState.logicOpEnable)
  {
    // D3D12 rejects logic ops combined with blending or with independent
    // blend states: RenderTarget[0] then describes every attachment, and
    // Vulkan ignores blending while the logic op is on.
    desc.BlendState.IndependentBlendEnable      = FALSE;
    desc.BlendState.RenderTarget[0].BlendEnable = FALSE;
  }

  desc.NumRenderTargets = UINT(colorFormats.size());
  for(uint32_t i = 0; i < D3D12_SIMULTANEOUS_RENDER_TARGET_COUNT; i++)
  {
    desc.RTVFormats[i] = i < colorFormats.size() ? colorFormats[i] : DXGI_FORMAT_UNKNOWN;
  }
  desc.DSVFormat = renderingState.depthAttachmentFormat;
  if(desc.DSVFormat == DXGI_FORMAT_UNKNOWN)
  {
    // No depth buffer: nothing to test against, as Vulkan without a depth attachment.
    desc.DepthStencilState.DepthEnable   = FALSE;
    desc.DepthStencilState.StencilEnable = FALSE;
  }
  return NVDX12_FAIL_REPORT(device->CreateGraphicsPipelineState(&desc, IID_PPV_ARGS(&pipeline)));
}

}  // namespace nvdx12

//--------------------------------------------------------------------------------------------------
// Usage example
//--------------------------------------------------------------------------------------------------
[[maybe_unused]] static void usage_GraphicsPipeline()
{
  ID3D12Device*        device        = nullptr;  // EX: from nvdx12::Context::getDevice()
  ID3D12RootSignature* rootSignature = nullptr;  // EX: nvdx12::createBindlessRootSignature(rootSignature, device)
  ID3D12GraphicsCommandList7* cmd    = nullptr;  // EX: the frame's command list, bound with DescriptorHeap::cmdBind
  nvdx12::ShaderCompiler      compiler;          // see usage_ShaderCompiler()
  nvdx12::DescriptorHeap*     heap = nullptr;    // EX: the application's heap

  // The state: nvvk's members as D3D12 structures, changed directly. Two
  // color attachments, the second one alpha-blended.
  nvdx12::GraphicsPipelineState graphicsState;
  graphicsState.depthStencilState.DepthFunc = D3D12_COMPARISON_FUNC_GREATER_EQUAL;
  graphicsState.colorBlendEnables           = {FALSE, TRUE};
  graphicsState.colorWriteMasks             = {D3D12_COLOR_WRITE_ENABLE_ALL, D3D12_COLOR_WRITE_ENABLE_ALL};
  graphicsState.colorBlendEquations         = {{},
                                               {.srcColorBlendFactor = D3D12_BLEND_SRC_ALPHA,
                                                .dstColorBlendFactor = D3D12_BLEND_INV_SRC_ALPHA,
                                                .colorBlendOp        = D3D12_BLEND_OP_ADD,
                                                .srcAlphaBlendFactor = D3D12_BLEND_ONE,
                                                .dstAlphaBlendFactor = D3D12_BLEND_INV_SRC_ALPHA,
                                                .alphaBlendOp        = D3D12_BLEND_OP_ADD}};

  // The PSO: shaders on the bindless root signature, vertices pulled from
  // buffers through their shader addresses.
  nvdx12::GraphicsPipelineCreator creator;
  creator.pipelineInfo.pRootSignature          = rootSignature;
  creator.colorFormats                         = {DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_FORMAT_R16G16B16A16_FLOAT};
  creator.renderingState.depthAttachmentFormat = DXGI_FORMAT_D32_FLOAT;  // EX: from a nvdx12::RenderTargetState
  nvdx12::ShaderBlob vertexBlob                = compiler.compileFile("raster.vert.hlsl", "vs_6_6");
  nvdx12::ShaderBlob pixelBlob                 = compiler.compileFile("raster.frag.hlsl", "ps_6_6");
  creator.addShader(nvdx12::GraphicsShaderStage::eVertex, vertexBlob);
  creator.addShader(nvdx12::GraphicsShaderStage::ePixel, pixelBlob);

  nvdx12::ComPtr<ID3D12PipelineState> pipeline;
  NVDX12_CHECK(creator.createGraphicsPipeline(device, graphicsState, pipeline));
  NVDX12_DBG_CUSTOM_NAME(pipeline, "raster");

  // Drawing: the heap bind set the graphics root signature too; a draw pushes
  // its constants to the graphics bind point.
  heap->cmdBind(cmd);
  // renderTargetState.cmdBeginRendering(cmd, {});  EX: nvdx12::RenderTarget::fillState, or OMSetRenderTargets
  nvdx12::GraphicsPipelineState::cmdSetViewportAndScissor(cmd, nvdx12::Extent2D{1920, 1080});
  graphicsState.cmdApplyAllStates(cmd);
  cmd->SetPipelineState(pipeline.Get());
  struct PushConstants
  {
    uint64_t vertexAddress;  // nvdx12::Buffer::address
  } push{};
  nvdx12::cmdPushConstants(cmd, push, nvdx12::BindPoint::eGraphics);
  cmd->DrawInstanced(3, 1, 0, 0);
}
