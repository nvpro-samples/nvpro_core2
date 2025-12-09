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

#include "graphics_pipeline.hpp"

namespace nvvk {

void GraphicsPipelineState::cmdApplyAllStates(VkCommandBuffer cmd) const
{
  vkCmdSetLineWidth(cmd, rasterizationState.lineWidth);
  vkCmdSetLineStippleEnableEXT(cmd, rasterizationLineState.stippledLineEnable);
  vkCmdSetLineRasterizationModeEXT(cmd, rasterizationLineState.lineRasterizationMode);
  if(rasterizationLineState.stippledLineEnable)
  {
    vkCmdSetLineStipple(cmd, rasterizationLineState.lineStippleFactor, rasterizationLineState.lineStipplePattern);
  }

  vkCmdSetRasterizerDiscardEnable(cmd, rasterizationState.rasterizerDiscardEnable);
  vkCmdSetPolygonModeEXT(cmd, rasterizationState.polygonMode);
  vkCmdSetCullMode(cmd, rasterizationState.cullMode);
  vkCmdSetFrontFace(cmd, rasterizationState.frontFace);
  vkCmdSetDepthBiasEnable(cmd, rasterizationState.depthBiasEnable);
  if(rasterizationState.depthBiasEnable)
  {
    vkCmdSetDepthBias(cmd, rasterizationState.depthBiasConstantFactor, rasterizationState.depthBiasClamp,
                      rasterizationState.depthBiasSlopeFactor);
  }
  vkCmdSetDepthClampEnableEXT(cmd, rasterizationState.depthClampEnable);

  vkCmdSetDepthTestEnable(cmd, depthStencilState.depthTestEnable);
  if(depthStencilState.depthTestEnable)
  {
    vkCmdSetDepthBounds(cmd, depthStencilState.minDepthBounds, depthStencilState.maxDepthBounds);
    vkCmdSetDepthBoundsTestEnable(cmd, depthStencilState.depthBoundsTestEnable);
    vkCmdSetDepthCompareOp(cmd, depthStencilState.depthCompareOp);
    vkCmdSetDepthWriteEnable(cmd, depthStencilState.depthWriteEnable);
  }

  vkCmdSetStencilTestEnable(cmd, depthStencilState.stencilTestEnable);
  if(depthStencilState.stencilTestEnable)
  {
    vkCmdSetStencilCompareMask(cmd, VK_STENCIL_FACE_FRONT_BIT, depthStencilState.front.compareMask);
    vkCmdSetStencilCompareMask(cmd, VK_STENCIL_FACE_BACK_BIT, depthStencilState.back.compareMask);
    vkCmdSetStencilWriteMask(cmd, VK_STENCIL_FACE_FRONT_BIT, depthStencilState.front.writeMask);
    vkCmdSetStencilWriteMask(cmd, VK_STENCIL_FACE_BACK_BIT, depthStencilState.back.writeMask);
    vkCmdSetStencilReference(cmd, VK_STENCIL_FACE_FRONT_BIT, depthStencilState.front.reference);
    vkCmdSetStencilReference(cmd, VK_STENCIL_FACE_BACK_BIT, depthStencilState.back.reference);
    vkCmdSetStencilOp(cmd, VK_STENCIL_FACE_FRONT_BIT, depthStencilState.front.failOp, depthStencilState.front.passOp,
                      depthStencilState.front.depthFailOp, depthStencilState.front.compareOp);
    vkCmdSetStencilOp(cmd, VK_STENCIL_FACE_BACK_BIT, depthStencilState.back.failOp, depthStencilState.back.passOp,
                      depthStencilState.back.depthFailOp, depthStencilState.back.compareOp);
  }

  vkCmdSetPrimitiveRestartEnable(cmd, inputAssemblyState.primitiveRestartEnable);
  vkCmdSetPrimitiveTopology(cmd, inputAssemblyState.topology);

  vkCmdSetRasterizationSamplesEXT(cmd, multisampleState.rasterizationSamples);
  vkCmdSetSampleMaskEXT(cmd, multisampleState.rasterizationSamples, &sampleMask);
  vkCmdSetAlphaToCoverageEnableEXT(cmd, multisampleState.alphaToCoverageEnable);
  vkCmdSetAlphaToOneEnableEXT(cmd, multisampleState.alphaToOneEnable);

  if(vertexBindings.size() && vertexAttributes.size())
  {
    vkCmdSetVertexInputEXT(cmd, static_cast<uint32_t>(vertexBindings.size()), vertexBindings.data(),
                           static_cast<uint32_t>(vertexAttributes.size()), vertexAttributes.data());
  }

  assert(colorWriteMasks.size() == colorBlendEquations.size() && colorWriteMasks.size() == colorBlendEnables.size());

  uint32_t attachmentCount = static_cast<uint32_t>(colorWriteMasks.size());
  if(attachmentCount)
  {
    vkCmdSetColorBlendEquationEXT(cmd, 0, attachmentCount, colorBlendEquations.data());
    vkCmdSetColorBlendEnableEXT(cmd, 0, attachmentCount, colorBlendEnables.data());
    vkCmdSetColorWriteMaskEXT(cmd, 0, attachmentCount, colorWriteMasks.data());
  }

  vkCmdSetBlendConstants(cmd, colorBlendState.blendConstants);
  vkCmdSetLogicOpEnableEXT(cmd, colorBlendState.logicOpEnable);
}

void GraphicsPipelineState::cmdApplyDynamicStates(VkCommandBuffer cmd, std::span<const VkDynamicState> dynamicStates) const
{
  for(VkDynamicState state : dynamicStates)
  {
    cmdApplyDynamicState(cmd, state);
  }
}


//////////////////////////////////////////////////////////////////////////

void GraphicsPipelineCreator::clearShaders()
{
  m_shaderStages.clear();
  m_shaderStageModules.clear();
  m_shaderStageSubgroupSizes.clear();
}

void GraphicsPipelineCreator::addShader(VkShaderStageFlagBits       stage,
                                        const char*                 pEntryName,
                                        size_t                      spirvSizeInBytes,
                                        const uint32_t*             spirvData,
                                        const VkSpecializationInfo* pSpecializationInfo /*= nullptr*/,
                                        uint32_t                    subgroupRequiredSize /*= 0*/)
{
  VkPipelineShaderStageCreateInfo shaderInfo = {
      .sType               = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
      .stage               = stage,
      .pName               = pEntryName,
      .pSpecializationInfo = pSpecializationInfo,
  };

  VkShaderModuleCreateInfo moduleInfo = {
      .sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = spirvSizeInBytes,
      .pCode    = spirvData,
  };

  VkPipelineShaderStageRequiredSubgroupSizeCreateInfo subgroupInfo = {
      .sType                = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO,
      .requiredSubgroupSize = subgroupRequiredSize,
  };

  m_shaderStages.emplace_back(shaderInfo);
  m_shaderStageModules.emplace_back(moduleInfo);
  m_shaderStageSubgroupSizes.emplace_back(subgroupInfo);
}

void GraphicsPipelineCreator::addShader(VkShaderStageFlagBits       stage,
                                        const char*                 pEntryName,
                                        VkShaderModule              shaderModule,
                                        const VkSpecializationInfo* pSpecializationInfo /*= nullptr*/,
                                        uint32_t                    subgroupRequiredSize /*= 0*/)
{
  VkPipelineShaderStageCreateInfo shaderInfo = {
      .sType               = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
      .stage               = stage,
      .module              = shaderModule,
      .pName               = pEntryName,
      .pSpecializationInfo = pSpecializationInfo,
  };

  VkPipelineShaderStageRequiredSubgroupSizeCreateInfo subgroupInfo = {
      .sType                = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO,
      .requiredSubgroupSize = subgroupRequiredSize,
  };

  m_shaderStages.emplace_back(shaderInfo);
  m_shaderStageModules.push_back({});
  m_shaderStageSubgroupSizes.emplace_back(subgroupInfo);
}

void GraphicsPipelineCreator::addShader(VkShaderStageFlagBits       stage,
                                        const char*                 pEntryName,
                                        std::span<uint32_t const>   spirvData,
                                        const VkSpecializationInfo* pSpecializationInfo /*= nullptr*/,
                                        uint32_t                    subgroupRequiredSize /*= 0*/)
{
  addShader(stage, pEntryName, spirvData.size_bytes(), spirvData.data(), pSpecializationInfo, subgroupRequiredSize);
}

VkResult GraphicsPipelineCreator::createGraphicsPipeline(VkDevice                     device,
                                                         VkPipelineCache              cache,
                                                         const GraphicsPipelineState& graphicsState,
                                                         VkPipeline*                  pPipeline)
{
  VkGraphicsPipelineCreateInfo pipelineInfoTemp;

  buildPipelineCreateInfo(pipelineInfoTemp, graphicsState);

  VkResult result = vkCreateGraphicsPipelines(device, cache, 1, &pipelineInfoTemp, nullptr, pPipeline);

  return result;
}

void GraphicsPipelineCreator::buildPipelineCreateInfo(VkGraphicsPipelineCreateInfo& createTemp, const GraphicsPipelineState& graphicsState)
{
  // check unsupported input states
  assert(pipelineInfo.pColorBlendState == nullptr);
  assert(pipelineInfo.pDepthStencilState == nullptr);
  assert(pipelineInfo.pDynamicState == nullptr);
  assert(pipelineInfo.pInputAssemblyState == nullptr);
  assert(pipelineInfo.pMultisampleState == nullptr);
  assert(pipelineInfo.pRasterizationState == nullptr);
  assert(pipelineInfo.pTessellationState == nullptr);
  assert(pipelineInfo.pVertexInputState == nullptr);
  assert(pipelineInfo.pViewportState == nullptr);

  assert(graphicsState.rasterizationState.pNext == nullptr);
  assert(graphicsState.multisampleState.pSampleMask == nullptr);
  assert(graphicsState.tessellationState.pNext == nullptr);

  assert(graphicsState.vertexInputState.pVertexBindingDescriptions == nullptr);
  assert(graphicsState.vertexInputState.pVertexAttributeDescriptions == nullptr);
  assert(graphicsState.vertexInputState.vertexBindingDescriptionCount == 0);
  assert(graphicsState.vertexInputState.vertexAttributeDescriptionCount == 0);

  assert(graphicsState.colorBlendState.pAttachments == nullptr);
  assert(graphicsState.colorBlendState.attachmentCount == 0);

  assert(graphicsState.colorWriteMasks.size() == graphicsState.colorBlendEquations.size()
         && graphicsState.colorWriteMasks.size() == graphicsState.colorBlendEnables.size());

  // copy data that we end up modifying
  createTemp           = pipelineInfo;
  m_rasterizationState = graphicsState.rasterizationState;
  m_multisampleState   = graphicsState.multisampleState;
  m_tessellationState  = graphicsState.tessellationState;
  m_vertexInputState   = graphicsState.vertexInputState;
  m_renderingState     = renderingState;

  // setup various pointers
  if(pipelineInfo.renderPass == nullptr)
  {
    m_renderingState.colorAttachmentCount    = static_cast<uint32_t>(colorFormats.size());
    m_renderingState.pColorAttachmentFormats = colorFormats.data();

    m_renderingState.pNext = createTemp.pNext;
    createTemp.pNext       = &m_renderingState;
  }

  if(flags2 != 0)
  {
    // Only valid to enqueue if flags are non-zero
    m_flags2Info.flags = flags2;

    m_flags2Info.pNext = createTemp.pNext;
    createTemp.pNext   = &m_flags2Info;
  }

  createTemp.pColorBlendState    = &m_colorBlendState;
  createTemp.pDepthStencilState  = &graphicsState.depthStencilState;
  createTemp.pDynamicState       = &m_dynamicState;
  createTemp.pInputAssemblyState = &graphicsState.inputAssemblyState;
  createTemp.pMultisampleState   = &m_multisampleState;
  createTemp.pRasterizationState = &m_rasterizationState;
  createTemp.pTessellationState  = &m_tessellationState;
  createTemp.pVertexInputState   = &m_vertexInputState;
  createTemp.pViewportState      = &viewportState;

  m_rasterizationState.pNext     = &graphicsState.rasterizationLineState;
  m_multisampleState.pSampleMask = &graphicsState.sampleMask;
  m_tessellationState.pNext      = &graphicsState.tessellationDomainOriginState;

  // setup local arrays

  m_dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStateValues.size());
  m_dynamicState.pDynamicStates    = dynamicStateValues.data();

  m_staticVertexBindings.resize(graphicsState.vertexBindings.size());
  m_staticVertexBindingDivisors.resize(graphicsState.vertexBindings.size());
  m_staticVertexAttributes.resize(graphicsState.vertexAttributes.size());

  m_vertexInputState.pVertexBindingDescriptions    = m_staticVertexBindings.data();
  m_vertexInputState.pVertexAttributeDescriptions  = m_staticVertexAttributes.data();
  m_vertexInputDivisorState.pVertexBindingDivisors = m_staticVertexBindingDivisors.data();

  m_vertexInputState.vertexBindingDescriptionCount   = static_cast<uint32_t>(m_staticVertexBindings.size());
  m_vertexInputState.vertexAttributeDescriptionCount = static_cast<uint32_t>(m_staticVertexAttributes.size());

  // binding divisors are
  uint32_t divisorCount = 0;

  m_vertexInputDivisorState.vertexBindingDivisorCount = 0;
  m_vertexInputDivisorState.pNext                     = graphicsState.vertexInputState.pNext;

  for(size_t i = 0; i < graphicsState.vertexBindings.size(); i++)
  {
    m_staticVertexBindings[i].binding   = graphicsState.vertexBindings[i].binding;
    m_staticVertexBindings[i].inputRate = graphicsState.vertexBindings[i].inputRate;
    m_staticVertexBindings[i].stride    = graphicsState.vertexBindings[i].stride;
    if(m_staticVertexBindings[i].inputRate != VK_VERTEX_INPUT_RATE_VERTEX)
    {
      m_staticVertexBindingDivisors[divisorCount].binding = graphicsState.vertexBindings[i].binding;
      m_staticVertexBindingDivisors[divisorCount].divisor = graphicsState.vertexBindings[i].divisor;
      divisorCount++;
    }
  }

  if(divisorCount)
  {
    m_vertexInputDivisorState.vertexBindingDivisorCount = divisorCount;

    m_vertexInputDivisorState.pNext = graphicsState.vertexInputState.pNext;
    m_vertexInputState.pNext        = &m_vertexInputDivisorState;
  }

  for(size_t i = 0; i < graphicsState.vertexAttributes.size(); i++)
  {
    m_staticVertexAttributes[i].binding  = graphicsState.vertexAttributes[i].binding;
    m_staticVertexAttributes[i].format   = graphicsState.vertexAttributes[i].format;
    m_staticVertexAttributes[i].location = graphicsState.vertexAttributes[i].location;
    m_staticVertexAttributes[i].offset   = graphicsState.vertexAttributes[i].offset;
  }

  m_staticAttachmentState.resize(graphicsState.colorWriteMasks.size());

  m_colorBlendState.attachmentCount = static_cast<uint32_t>(m_staticAttachmentState.size());
  m_colorBlendState.pAttachments    = m_staticAttachmentState.data();

  for(size_t i = 0; i < graphicsState.colorWriteMasks.size(); i++)
  {
    m_staticAttachmentState[i].blendEnable         = graphicsState.colorBlendEnables[i];
    m_staticAttachmentState[i].colorWriteMask      = graphicsState.colorWriteMasks[i];
    m_staticAttachmentState[i].alphaBlendOp        = graphicsState.colorBlendEquations[i].alphaBlendOp;
    m_staticAttachmentState[i].colorBlendOp        = graphicsState.colorBlendEquations[i].colorBlendOp;
    m_staticAttachmentState[i].dstAlphaBlendFactor = graphicsState.colorBlendEquations[i].dstAlphaBlendFactor;
    m_staticAttachmentState[i].dstColorBlendFactor = graphicsState.colorBlendEquations[i].dstColorBlendFactor;
    m_staticAttachmentState[i].srcAlphaBlendFactor = graphicsState.colorBlendEquations[i].srcAlphaBlendFactor;
    m_staticAttachmentState[i].srcColorBlendFactor = graphicsState.colorBlendEquations[i].srcColorBlendFactor;
  }

  if(m_shaderStages.size())
  {
    // if we use locally provided shaders, then none must have been provided otherwise
    assert(createTemp.stageCount == 0 && createTemp.pStages == nullptr);

    createTemp.stageCount = static_cast<uint32_t>(m_shaderStages.size());
    createTemp.pStages    = m_shaderStages.data();

    for(uint32_t i = 0; i < createTemp.stageCount; i++)
    {
      if(m_shaderStages[i].module == nullptr)
      {
        m_shaderStages[i].pNext = &m_shaderStageModules[i];
      }
      if(m_shaderStageSubgroupSizes[i].requiredSubgroupSize)
      {
        m_shaderStageSubgroupSizes[i].pNext = const_cast<void*>(m_shaderStages[i].pNext);
        m_shaderStages[i].pNext             = &m_shaderStageSubgroupSizes[i];
      }
    }
  }
}

}  // namespace nvvk

//--------------------------------------------------------------------------------------------------
// Usage example
//--------------------------------------------------------------------------------------------------
[[maybe_unused]] static void usage_GraphicsPipeline()
{
  VkDevice device{};

  nvvk::GraphicsPipelineState graphicsState;

  // set some state
  // we are omitting most things to keep it short
  graphicsState.depthStencilState.depthTestEnable  = VK_TRUE;
  graphicsState.depthStencilState.depthWriteEnable = VK_TRUE;
  graphicsState.depthStencilState.depthCompareOp   = VK_COMPARE_OP_GREATER_OR_EQUAL;

  // example using traditional pipeline
  {
    std::vector<uint32_t> vertexCode;
    std::vector<uint32_t> fragmentCode;

    // we want to create a traditional pipeline
    nvvk::GraphicsPipelineCreator graphicsPipelineCreator;

    // manipulate the public members of the class directly to change the state used for creation
    graphicsPipelineCreator.flags2 = VK_PIPELINE_CREATE_2_CAPTURE_STATISTICS_BIT_KHR;

    graphicsPipelineCreator.dynamicStateValues.push_back(VK_DYNAMIC_STATE_DEPTH_WRITE_ENABLE);
    graphicsPipelineCreator.dynamicStateValues.push_back(VK_DYNAMIC_STATE_DEPTH_COMPARE_OP);

    graphicsPipelineCreator.addShader(VK_SHADER_STAGE_VERTEX_BIT, "main", vertexCode.size() * sizeof(uint32_t),
                                      vertexCode.data());
    graphicsPipelineCreator.addShader(VK_SHADER_STAGE_FRAGMENT_BIT, "main", fragmentCode.size() * sizeof(uint32_t),
                                      fragmentCode.data());

    // create the actual pipeline from a combination of state within `graphicsPipelineCreator` and `graphicsState`
    VkPipeline graphicsPipeline;
    VkResult result = graphicsPipelineCreator.createGraphicsPipeline(device, nullptr, graphicsState, &graphicsPipeline);


    VkCommandBuffer cmd{};
    VkExtent2D      viewportSize{};

    //we recommend (and set defaults) to always use dynamic state for viewport and scissor
    nvvk::GraphicsPipelineState::cmdSetViewportAndScissor(cmd, viewportSize);

    // bind pipeline
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);

    vkCmdSetDepthCompareOp(cmd, VK_COMPARE_OP_GREATER_OR_EQUAL);
    vkCmdSetDepthWriteEnable(cmd, VK_TRUE);

    vkCmdDraw(cmd, 1, 2, 3, 4);

    vkCmdSetDepthCompareOp(cmd, VK_COMPARE_OP_EQUAL);
    vkCmdSetDepthWriteEnable(cmd, VK_FALSE);

    vkCmdDraw(cmd, 1, 2, 3, 4);
  }

  // example in combination with shader objects
  {
    VkCommandBuffer cmd{};
    VkExtent2D      viewportSize{};

    VkShaderEXT vertexShader{};
    VkShaderEXT fragmentShader{};

    // note this is actually the static function as before, but for looks we used the graphicsState
    graphicsState.cmdSetViewportAndScissor(cmd, viewportSize);

    // bind default state via struct
    graphicsState.cmdApplyAllStates(cmd);

    // bind the shaders
    nvvk::GraphicsPipelineState::BindableShaders bindableShaders;
    bindableShaders.vertex   = vertexShader;
    bindableShaders.fragment = fragmentShader;

    bool supportsMeshShaders = true;

    // also a static function
    graphicsState.cmdBindShaders(cmd, bindableShaders, supportsMeshShaders);

    vkCmdSetDepthCompareOp(cmd, VK_COMPARE_OP_GREATER_OR_EQUAL);
    vkCmdSetDepthWriteEnable(cmd, VK_TRUE);

    vkCmdDraw(cmd, 1, 2, 3, 4);

    vkCmdSetDepthCompareOp(cmd, VK_COMPARE_OP_EQUAL);
    vkCmdSetDepthWriteEnable(cmd, VK_FALSE);

    vkCmdDraw(cmd, 1, 2, 3, 4);
  }
}
