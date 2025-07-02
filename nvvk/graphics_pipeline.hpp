/*
* Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*
* SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
* SPDX-License-Identifier: Apache-2.0
*/

#pragma once
#include <vector>
#include <cstring>
#include <cassert>
#include <span>
#include <array>

#include <volk.h>

namespace nvvk {


//////////////////////////////////////////////////////////////////////////

class GraphicsPipelineState
{
public:
  // this object does not cover viewport and scissor states, we assume these to be done dynamically all the time
  // use this static function instead

  inline static void cmdSetViewportAndScissor(VkCommandBuffer cmd, const VkExtent2D& viewportSize, float minDepth = 0.0f, float maxDepth = 1.0f)
  {
    const VkViewport viewport{0.0F, 0.0F, float(viewportSize.width), float(viewportSize.height), minDepth, maxDepth};
    const VkRect2D   scissor{{0, 0}, viewportSize};
    vkCmdSetViewportWithCount(cmd, 1, &viewport);
    vkCmdSetScissorWithCount(cmd, 1, &scissor);
  }

  // static convenience function to bind shader objects

  struct BindableShaders
  {
    // warning: below functions depend on this ordering

    VkShaderEXT vertex      = VK_NULL_HANDLE;
    VkShaderEXT fragment    = VK_NULL_HANDLE;
    VkShaderEXT tessControl = VK_NULL_HANDLE;
    VkShaderEXT tessEval    = VK_NULL_HANDLE;
    VkShaderEXT geometry    = VK_NULL_HANDLE;
    VkShaderEXT task        = VK_NULL_HANDLE;
    VkShaderEXT mesh        = VK_NULL_HANDLE;
  };

  // Help binding all shaders the first time. In graphics unused stages must be bound with NULL if not used,
  // they cannot be left without binding.
  // Set `withMesh` to true if the mesh shader extension is active.
  inline static void cmdBindShaders(VkCommandBuffer cmd, const BindableShaders& shaders, bool withMesh = false)
  {
    std::array<VkShaderStageFlagBits, 7> shaderType = {
        VK_SHADER_STAGE_VERTEX_BIT,
        VK_SHADER_STAGE_FRAGMENT_BIT,
        VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT,
        VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT,
        VK_SHADER_STAGE_GEOMETRY_BIT,
        VK_SHADER_STAGE_TASK_BIT_EXT,
        VK_SHADER_STAGE_MESH_BIT_EXT,
    };
    vkCmdBindShadersEXT(cmd, withMesh ? 7 : 5, shaderType.data(), &shaders.vertex);
  }

  VkSampleMask sampleMask{~0U};

  VkPipelineInputAssemblyStateCreateInfo inputAssemblyState{
      .sType                  = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
      .flags                  = 0,
      .topology               = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
      .primitiveRestartEnable = VK_FALSE,
  };

  VkPipelineRasterizationStateCreateInfo rasterizationState{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
      // pNext will be set to `rasterizationLineState` when used in GraphicsPipelineCreator
      .flags                   = 0,
      .depthClampEnable        = VK_FALSE,
      .rasterizerDiscardEnable = VK_FALSE,
      .polygonMode             = VK_POLYGON_MODE_FILL,
      .cullMode                = VK_CULL_MODE_BACK_BIT,
      .frontFace               = VK_FRONT_FACE_COUNTER_CLOCKWISE,
      .depthBiasEnable         = VK_FALSE,
      .depthBiasConstantFactor = 0,
      .depthBiasClamp          = 0,
      .depthBiasSlopeFactor    = 0,
      .lineWidth               = 1,
  };

  VkPipelineRasterizationLineStateCreateInfo rasterizationLineState{
      .sType                 = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_LINE_STATE_CREATE_INFO,
      .lineRasterizationMode = VK_LINE_RASTERIZATION_MODE_DEFAULT,
      .stippledLineEnable    = VK_FALSE,
      .lineStippleFactor     = 1,
      .lineStipplePattern    = 0xAA,
  };

  VkPipelineMultisampleStateCreateInfo multisampleState{
      .sType                 = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
      .rasterizationSamples  = VK_SAMPLE_COUNT_1_BIT,
      .sampleShadingEnable   = VK_FALSE,
      .minSampleShading      = 0,
      .pSampleMask           = 0,  // do not use, implicitly set to &sampleMask in GraphicsPipelineCreator
      .alphaToCoverageEnable = VK_FALSE,
      .alphaToOneEnable      = VK_FALSE,
  };

  VkPipelineDepthStencilStateCreateInfo depthStencilState{
      .sType                 = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
      .depthTestEnable       = VK_TRUE,
      .depthWriteEnable      = VK_TRUE,
      .depthCompareOp        = VK_COMPARE_OP_LESS_OR_EQUAL,
      .depthBoundsTestEnable = VK_FALSE,
      .stencilTestEnable     = VK_FALSE,
      .front                 = {},
      .back                  = {},
      .minDepthBounds        = 0.0f,
      .maxDepthBounds        = 1.0f,
  };

  VkPipelineColorBlendStateCreateInfo colorBlendState{
      .sType           = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
      .logicOpEnable   = VK_FALSE,
      .logicOp         = VK_LOGIC_OP_CLEAR,
      .attachmentCount = 0,  // do not use, implicitly set GraphicsPipelineCreator
      .pAttachments    = 0,  // do not use, implicitly set GraphicsPipelineCreator
      .blendConstants  = {1.0f, 1.0f, 1.0f, 1.0f},
  };

  VkPipelineVertexInputStateCreateInfo vertexInputState{
      .sType                           = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
      .vertexBindingDescriptionCount   = 0,  // do not use, implicitly set in GraphicsPipelineCreator
      .pVertexBindingDescriptions      = 0,  // do not use, implicitly set in GraphicsPipelineCreator
      .vertexAttributeDescriptionCount = 0,  // do not use, implicitly set in GraphicsPipelineCreator
      .pVertexAttributeDescriptions    = 0,  // do not use, implicitly set in GraphicsPipelineCreator
  };

  VkPipelineTessellationStateCreateInfo tessellationState{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_TESSELLATION_STATE_CREATE_INFO,
      // pNext will be set to `tessellationDomainOriginState` when used in GraphicsPipelineCreator
      .patchControlPoints = 4,
  };

  VkPipelineTessellationDomainOriginStateCreateInfo tessellationDomainOriginState{
      .sType        = VK_STRUCTURE_TYPE_PIPELINE_TESSELLATION_DOMAIN_ORIGIN_STATE_CREATE_INFO,
      .domainOrigin = VK_TESSELLATION_DOMAIN_ORIGIN_UPPER_LEFT,
  };

  // by default we enable 1 color attachment with disabled blending

  std::vector<VkBool32>                colorBlendEnables{VK_FALSE};
  std::vector<VkColorComponentFlags>   colorWriteMasks{VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT
                                                     | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT};
  std::vector<VkColorBlendEquationEXT> colorBlendEquations{{
      .srcColorBlendFactor = VK_BLEND_FACTOR_ZERO,
      .dstColorBlendFactor = VK_BLEND_FACTOR_ZERO,
      .colorBlendOp        = VK_BLEND_OP_ADD,
      .srcAlphaBlendFactor = VK_BLEND_FACTOR_ZERO,
      .dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO,
      .alphaBlendOp        = VK_BLEND_OP_ADD,
  }};

  // GraphicsPipelineCreator will implicitly translate these to:
  // - VkVertexInputAttributeDescription
  // - VkVertexInputBindingDescription
  // - VkVertexInputBindingDivisorDescription
  std::vector<VkVertexInputBindingDescription2EXT>   vertexBindings{};
  std::vector<VkVertexInputAttributeDescription2EXT> vertexAttributes{};

  // only valid in 1.4 context and in combination with VK_EXT_shader_objects
  void cmdApplyAllStates(VkCommandBuffer cmd) const;

  // dynamic states not covered within, will be silently ignored
  void cmdApplyDynamicStates(VkCommandBuffer cmd, std::span<const VkDynamicState> dynamicStates) const;

  // returns true if not handled
  bool cmdApplyDynamicState(VkCommandBuffer cmd, VkDynamicState dynamicState) const;
};

class GraphicsPipelineCreator
{
public:
  VkGraphicsPipelineCreateInfo pipelineInfo{
      .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
      // if pStages is set manually, then clearShaders() must be used prior `createGraphicsPipeline`
      // all other pointers are automatically configured.
  };

  // if non-zero, is used instead of pipelineInfo's
  VkPipelineCreateFlags2 flags2 = 0;

  // used when pipelineInfo.renderPass is null
  VkPipelineRenderingCreateInfo renderingState{
      .sType                   = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
      .colorAttachmentCount    = 0,  // implicitly set by `colorFormats` vector
      .pColorAttachmentFormats = 0,  // implicitly set by `colorFormats` vector
      .depthAttachmentFormat   = VK_FORMAT_X8_D24_UNORM_PACK32,
      .stencilAttachmentFormat = VK_FORMAT_UNDEFINED,
  };

  std::vector<VkFormat> colorFormats{VK_FORMAT_R8G8B8A8_UNORM};

  VkPipelineViewportStateCreateInfo viewportState{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
  };

  std::vector<VkDynamicState> dynamicStateValues{VK_DYNAMIC_STATE_VIEWPORT_WITH_COUNT, VK_DYNAMIC_STATE_SCISSOR_WITH_COUNT};


  // either manually set pipelineInfo.pStages/pipelineInfo.stageCount
  // or use this wrapper, pointers must stay valid
  void clearShaders();
  void addShader(VkShaderStageFlagBits       stage,
                 const char*                 pEntryName,
                 size_t                      spirvSizeInBytes,
                 const uint32_t*             spirvData,
                 const VkSpecializationInfo* pSpecializationInfo  = nullptr,
                 uint32_t                    subgroupRequiredSize = 0);
  void addShader(VkShaderStageFlagBits       stage,
                 const char*                 pEntryName,
                 std::span<uint32_t const>   spirvData,
                 const VkSpecializationInfo* pSpecializationInfo  = nullptr,
                 uint32_t                    subgroupRequiredSize = 0);
  void addShader(VkShaderStageFlagBits       stage,
                 const char*                 pEntryName,
                 VkShaderModule              shaderModule,
                 const VkSpecializationInfo* pSpecializationInfo  = nullptr,
                 uint32_t                    subgroupRequiredSize = 0);

  // none of the public class members are changed during this process
  VkResult createGraphicsPipeline(VkDevice device, VkPipelineCache cache, const GraphicsPipelineState& graphicsState, VkPipeline* pPipeline);

protected:
  void buildPipelineCreateInfo(VkGraphicsPipelineCreateInfo& createInfoTemp, const GraphicsPipelineState& graphicsState);

  std::vector<VkPipelineShaderStageCreateInfo>                     m_shaderStages;
  std::vector<VkPipelineShaderStageRequiredSubgroupSizeCreateInfo> m_shaderStageSubgroupSizes;
  std::vector<VkShaderModuleCreateInfo>                            m_shaderStageModules;

  std::vector<VkVertexInputAttributeDescription>      m_staticVertexAttributes;
  std::vector<VkVertexInputBindingDescription>        m_staticVertexBindings;
  std::vector<VkVertexInputBindingDivisorDescription> m_staticVertexBindingDivisors;

  std::vector<VkPipelineColorBlendAttachmentState> m_staticAttachmentState;

  VkPipelineDynamicStateCreateInfo m_dynamicState{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
  };

  VkPipelineVertexInputDivisorStateCreateInfo m_vertexInputDivisorState{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_DIVISOR_STATE_CREATE_INFO,
  };

  VkPipelineCreateFlags2CreateInfo m_flags2Info{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_CREATE_FLAGS_2_CREATE_INFO,
  };

  VkPipelineVertexInputStateCreateInfo   m_vertexInputState{};
  VkPipelineMultisampleStateCreateInfo   m_multisampleState{};
  VkPipelineRasterizationStateCreateInfo m_rasterizationState{};
  VkPipelineColorBlendStateCreateInfo    m_colorBlendState{};
  VkPipelineTessellationStateCreateInfo  m_tessellationState{};
  VkPipelineRenderingCreateInfo          m_renderingState{};
};


//////////////////////////////////////////////////////////////////////////

inline bool GraphicsPipelineState::cmdApplyDynamicState(VkCommandBuffer cmd, VkDynamicState dynamicState) const
{
  switch(dynamicState)
  {
    case VK_DYNAMIC_STATE_LINE_WIDTH:
      vkCmdSetLineWidth(cmd, rasterizationState.lineWidth);
      return false;
    case VK_DYNAMIC_STATE_DEPTH_BIAS:
      vkCmdSetDepthBias(cmd, rasterizationState.depthBiasConstantFactor, rasterizationState.depthBiasClamp,
                        rasterizationState.depthBiasSlopeFactor);
      return false;
    case VK_DYNAMIC_STATE_BLEND_CONSTANTS:
      vkCmdSetBlendConstants(cmd, colorBlendState.blendConstants);
      return false;
    case VK_DYNAMIC_STATE_DEPTH_BOUNDS:
      vkCmdSetDepthBounds(cmd, depthStencilState.minDepthBounds, depthStencilState.maxDepthBounds);
      return false;
    case VK_DYNAMIC_STATE_STENCIL_COMPARE_MASK:
      vkCmdSetStencilCompareMask(cmd, VK_STENCIL_FACE_FRONT_BIT, depthStencilState.front.compareMask);
      vkCmdSetStencilCompareMask(cmd, VK_STENCIL_FACE_BACK_BIT, depthStencilState.back.compareMask);
      return false;
    case VK_DYNAMIC_STATE_STENCIL_WRITE_MASK:
      vkCmdSetStencilWriteMask(cmd, VK_STENCIL_FACE_FRONT_BIT, depthStencilState.front.writeMask);
      vkCmdSetStencilWriteMask(cmd, VK_STENCIL_FACE_BACK_BIT, depthStencilState.back.writeMask);
      return false;
    case VK_DYNAMIC_STATE_STENCIL_REFERENCE:
      vkCmdSetStencilReference(cmd, VK_STENCIL_FACE_FRONT_BIT, depthStencilState.front.reference);
      vkCmdSetStencilReference(cmd, VK_STENCIL_FACE_BACK_BIT, depthStencilState.back.reference);
      return false;
    case VK_DYNAMIC_STATE_CULL_MODE:
      vkCmdSetCullMode(cmd, rasterizationState.cullMode);
      return false;
    case VK_DYNAMIC_STATE_FRONT_FACE:
      vkCmdSetFrontFace(cmd, rasterizationState.frontFace);
      return false;
    case VK_DYNAMIC_STATE_PRIMITIVE_TOPOLOGY:
      vkCmdSetPrimitiveTopology(cmd, inputAssemblyState.topology);
      return false;
    case VK_DYNAMIC_STATE_VERTEX_INPUT_BINDING_STRIDE:
      return false;
    case VK_DYNAMIC_STATE_DEPTH_TEST_ENABLE:
      vkCmdSetDepthTestEnable(cmd, depthStencilState.depthTestEnable);
      return false;
    case VK_DYNAMIC_STATE_DEPTH_WRITE_ENABLE:
      vkCmdSetDepthWriteEnable(cmd, depthStencilState.depthWriteEnable);
      return false;
    case VK_DYNAMIC_STATE_DEPTH_COMPARE_OP:
      vkCmdSetDepthCompareOp(cmd, depthStencilState.depthCompareOp);
      return false;
    case VK_DYNAMIC_STATE_DEPTH_BOUNDS_TEST_ENABLE:
      vkCmdSetDepthBoundsTestEnable(cmd, depthStencilState.depthBoundsTestEnable);
      return false;
    case VK_DYNAMIC_STATE_STENCIL_TEST_ENABLE:
      vkCmdSetStencilTestEnable(cmd, depthStencilState.stencilTestEnable);
      return false;
    case VK_DYNAMIC_STATE_STENCIL_OP:
      vkCmdSetStencilOp(cmd, VK_STENCIL_FACE_FRONT_BIT, depthStencilState.front.failOp, depthStencilState.front.passOp,
                        depthStencilState.front.depthFailOp, depthStencilState.front.compareOp);
      vkCmdSetStencilOp(cmd, VK_STENCIL_FACE_BACK_BIT, depthStencilState.back.failOp, depthStencilState.back.passOp,
                        depthStencilState.back.depthFailOp, depthStencilState.back.compareOp);
      return false;
    case VK_DYNAMIC_STATE_RASTERIZER_DISCARD_ENABLE:
      vkCmdSetRasterizerDiscardEnable(cmd, rasterizationState.rasterizerDiscardEnable);
      return false;
    case VK_DYNAMIC_STATE_DEPTH_BIAS_ENABLE:
      vkCmdSetDepthBiasEnable(cmd, rasterizationState.depthBiasEnable);
      return false;
    case VK_DYNAMIC_STATE_PRIMITIVE_RESTART_ENABLE:
      vkCmdSetPrimitiveRestartEnable(cmd, inputAssemblyState.primitiveRestartEnable);
      return false;
    case VK_DYNAMIC_STATE_LINE_STIPPLE:
      vkCmdSetLineStipple(cmd, rasterizationLineState.lineStippleFactor, rasterizationLineState.lineStipplePattern);
      return false;
    case VK_DYNAMIC_STATE_LOGIC_OP_EXT:
      vkCmdSetLogicOpEXT(cmd, colorBlendState.logicOp);
      return false;
    case VK_DYNAMIC_STATE_COLOR_BLEND_ENABLE_EXT:
      vkCmdSetColorBlendEnableEXT(cmd, 0, static_cast<uint32_t>(colorBlendEnables.size()), colorBlendEnables.data());
      return false;
    case VK_DYNAMIC_STATE_COLOR_BLEND_EQUATION_EXT:
      vkCmdSetColorBlendEquationEXT(cmd, 0, static_cast<uint32_t>(colorBlendEquations.size()), colorBlendEquations.data());
      return false;
    case VK_DYNAMIC_STATE_COLOR_WRITE_MASK_EXT:
      vkCmdSetColorWriteMaskEXT(cmd, 0, static_cast<uint32_t>(colorWriteMasks.size()), colorWriteMasks.data());
      return false;
    case VK_DYNAMIC_STATE_TESSELLATION_DOMAIN_ORIGIN_EXT:
      vkCmdSetTessellationDomainOriginEXT(cmd, tessellationDomainOriginState.domainOrigin);
      return false;
    case VK_DYNAMIC_STATE_LINE_RASTERIZATION_MODE_EXT:
      vkCmdSetLineRasterizationModeEXT(cmd, rasterizationLineState.lineRasterizationMode);
      return false;
    case VK_DYNAMIC_STATE_LINE_STIPPLE_ENABLE_EXT:
      vkCmdSetLineStippleEnableEXT(cmd, rasterizationLineState.stippledLineEnable);
      return false;
    default:
      return true;
  }
}

}  // namespace nvvk
