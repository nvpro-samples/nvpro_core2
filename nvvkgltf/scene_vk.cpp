/*
 * Copyright (c) 2022-2025, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */


#include <cinttypes>
#include <mutex>
#include <sstream>
#include <span>


#include <glm/glm.hpp>

#include "nvshaders/gltf_scene_io.h.slang"  // Shared between host and device

#include "stb_image.h"
#include "nvimageformats/nv_dds.h"
#include "nvimageformats/nv_ktx.h"
#include "nvimageformats/texture_formats.h"
#include "nvutils/file_operations.hpp"
#include "nvutils/logger.hpp"
#include "nvutils/timers.hpp"
#include "nvvk/debug_util.hpp"
#include "nvvk/mipmaps.hpp"
#include "nvvk/check_error.hpp"
#include "nvvk/default_structs.hpp"
#include "nvvk/mipmaps.hpp"

#include "scene_vk.hpp"
#include "nvutils/parallel_work.hpp"
#include "nvvk/helpers.hpp"

// GPU memory category names for scene resources
namespace {
constexpr std::string_view kMemCategoryGeometry  = "Geometry";
constexpr std::string_view kMemCategorySceneData = "SceneData";
constexpr std::string_view kMemCategoryImages    = "Images";
}  // namespace

//--------------------------------------------------------------------------------------------------
// Forward declaration
std::vector<shaderio::GltfLight> getShaderLights(const std::vector<nvvkgltf::RenderLight>& rlights,
                                                 const std::vector<tinygltf::Light>&       gltfLights);

//-------------------------------------------------------------------------------------------------
//
//

void nvvkgltf::SceneVk::init(nvvk::ResourceAllocator* alloc, nvvk::SamplerPool* samplerPool)
{
  assert(!m_alloc);

  m_device         = alloc->getDevice();
  m_physicalDevice = alloc->getPhysicalDevice();
  m_alloc          = alloc;
  m_samplerPool    = samplerPool;
  m_memoryTracker.init(alloc);
}

void nvvkgltf::SceneVk::deinit()
{
  if(!m_alloc)
  {
    return;
  }

  destroy();

  m_alloc          = nullptr;
  m_samplerPool    = nullptr;
  m_physicalDevice = VK_NULL_HANDLE;
  m_device         = VK_NULL_HANDLE;
}

//--------------------------------------------------------------------------------------------------
// Create all Vulkan resources to hold a nvvkgltf::Scene
//
void nvvkgltf::SceneVk::create(VkCommandBuffer        cmd,
                               nvvk::StagingUploader& staging,
                               const nvvkgltf::Scene& scn,
                               bool                   generateMipmaps /*= true*/,
                               bool                   enableRayTracing /*= true*/)
{
  nvutils::ScopedTimer st(__FUNCTION__);
  destroy();  // Make sure not to leave allocated buffers

  m_generateMipmaps   = generateMipmaps;
  m_rayTracingEnabled = enableRayTracing;

  namespace fs     = std::filesystem;
  fs::path basedir = fs::path(scn.getFilename()).parent_path();
  updateMaterialBuffer(cmd, staging, scn);
  updateRenderNodesBuffer(cmd, staging, scn);
  createVertexBuffers(cmd, staging, scn);
  createTextureImages(cmd, staging, scn.getModel(), basedir);
  updateRenderLightsBuffer(cmd, staging, scn);

  // Update the buffers for morph and skinning
  updateRenderPrimitivesBuffer(cmd, staging, scn);

  // Buffer references
  shaderio::GltfScene scene_desc{};
  scene_desc.materials        = (shaderio::GltfShadeMaterial*)m_bMaterial.address;
  scene_desc.textureInfos     = (shaderio::GltfTextureInfo*)m_bTextureInfos.address;
  scene_desc.renderPrimitives = (shaderio::GltfRenderPrimitive*)m_bRenderPrim.address;
  scene_desc.renderNodes      = (shaderio::GltfRenderNode*)m_bRenderNode.address;
  scene_desc.lights           = (shaderio::GltfLight*)m_bLights.address;
  scene_desc.numLights        = static_cast<int>(scn.getRenderLights().size());

  NVVK_CHECK(m_alloc->createBuffer(m_bSceneDesc, std::span(&scene_desc, 1).size_bytes(),
                                   VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT));
  NVVK_CHECK(staging.appendBuffer(m_bSceneDesc, 0, std::span(&scene_desc, 1)));
  NVVK_DBG_NAME(m_bSceneDesc.buffer);
  m_memoryTracker.track(kMemCategorySceneData, m_bSceneDesc.allocation);
}

void nvvkgltf::SceneVk::update(VkCommandBuffer cmd, nvvk::StagingUploader& staging, const nvvkgltf::Scene& scn)
{
  updateMaterialBuffer(cmd, staging, scn);
  updateRenderNodesBuffer(cmd, staging, scn);
  updateRenderPrimitivesBuffer(cmd, staging, scn);
}

template <typename T>
inline shaderio::GltfTextureInfo getTextureInfo(const T& tinfo)
{
  const KHR_texture_transform& transform = tinygltf::utils::getTextureTransform(tinfo);
  const int                    texCoord  = std::min(tinfo.texCoord, 1);  // Only 2 texture coordinates

  // This is the texture info that will be used in the shader
  return {
      .uvTransform = shaderio::float3x2(transform.uvTransform[0][0], transform.uvTransform[1][0],   //
                                        transform.uvTransform[0][1], transform.uvTransform[1][1],   //
                                        transform.uvTransform[0][2], transform.uvTransform[1][2]),  //
      .index       = tinfo.index,
      .texCoord    = texCoord,
  };
}

// Helper to handle texture info and update textureInfos vector
template <typename T>
uint16_t addTextureInfo(const T& tinfo, std::vector<shaderio::GltfTextureInfo>& textureInfos)
{
  shaderio::GltfTextureInfo ti = getTextureInfo(tinfo);
  if(ti.index != -1)
  {
    uint16_t idx = static_cast<uint16_t>(textureInfos.size());
    textureInfos.push_back(ti);
    return idx;
  }
  return 0;  // No texture
}

static void getShaderMaterial(const tinygltf::Material&                 srcMat,
                              std::vector<shaderio::GltfShadeMaterial>& shadeMaterial,
                              std::vector<shaderio::GltfTextureInfo>&   textureInfos)
{
  int alphaMode = srcMat.alphaMode == "OPAQUE" ? 0 : (srcMat.alphaMode == "MASK" ? 1 : 2 /*BLEND*/);

  shaderio::GltfShadeMaterial dstMat = shaderio::defaultGltfMaterial();
  if(!srcMat.emissiveFactor.empty())
  {
    dstMat.emissiveFactor = glm::make_vec3<double>(srcMat.emissiveFactor.data());
  }

  dstMat.emissiveTexture     = addTextureInfo(srcMat.emissiveTexture, textureInfos);
  dstMat.normalTexture       = addTextureInfo(srcMat.normalTexture, textureInfos);
  dstMat.normalTextureScale  = static_cast<float>(srcMat.normalTexture.scale);
  dstMat.pbrBaseColorFactor  = glm::make_vec4<double>(srcMat.pbrMetallicRoughness.baseColorFactor.data());
  dstMat.pbrBaseColorTexture = addTextureInfo(srcMat.pbrMetallicRoughness.baseColorTexture, textureInfos);
  dstMat.pbrMetallicFactor   = static_cast<float>(srcMat.pbrMetallicRoughness.metallicFactor);
  dstMat.pbrMetallicRoughnessTexture = addTextureInfo(srcMat.pbrMetallicRoughness.metallicRoughnessTexture, textureInfos);
  dstMat.pbrRoughnessFactor = static_cast<float>(srcMat.pbrMetallicRoughness.roughnessFactor);
  dstMat.alphaMode          = alphaMode;
  dstMat.alphaCutoff        = static_cast<float>(srcMat.alphaCutoff);
  dstMat.occlusionStrength  = static_cast<float>(srcMat.occlusionTexture.strength);
  dstMat.occlusionTexture   = addTextureInfo(srcMat.occlusionTexture, textureInfos);
  dstMat.doubleSided        = srcMat.doubleSided ? 1 : 0;


  KHR_materials_transmission transmission = tinygltf::utils::getTransmission(srcMat);
  dstMat.transmissionFactor               = transmission.factor;
  dstMat.transmissionTexture              = addTextureInfo(transmission.texture, textureInfos);

  KHR_materials_ior ior = tinygltf::utils::getIor(srcMat);
  dstMat.ior            = ior.ior;

  KHR_materials_volume volume = tinygltf::utils::getVolume(srcMat);
  dstMat.attenuationColor     = volume.attenuationColor;
  dstMat.thicknessFactor      = volume.thicknessFactor;
  dstMat.thicknessTexture     = addTextureInfo(volume.thicknessTexture, textureInfos);
  dstMat.attenuationDistance  = volume.attenuationDistance;

  KHR_materials_clearcoat clearcoat = tinygltf::utils::getClearcoat(srcMat);
  dstMat.clearcoatFactor            = clearcoat.factor;
  dstMat.clearcoatRoughness         = clearcoat.roughnessFactor;
  dstMat.clearcoatRoughnessTexture  = addTextureInfo(clearcoat.roughnessTexture, textureInfos);
  dstMat.clearcoatTexture           = addTextureInfo(clearcoat.texture, textureInfos);
  dstMat.clearcoatNormalTexture     = addTextureInfo(clearcoat.normalTexture, textureInfos);

  KHR_materials_specular specular = tinygltf::utils::getSpecular(srcMat);
  dstMat.specularFactor           = specular.specularFactor;
  dstMat.specularTexture          = addTextureInfo(specular.specularTexture, textureInfos);
  dstMat.specularColorFactor      = specular.specularColorFactor;
  dstMat.specularColorTexture     = addTextureInfo(specular.specularColorTexture, textureInfos);

  KHR_materials_emissive_strength emissiveStrength = tinygltf::utils::getEmissiveStrength(srcMat);
  dstMat.emissiveFactor *= emissiveStrength.emissiveStrength;

  KHR_materials_unlit unlit = tinygltf::utils::getUnlit(srcMat);
  dstMat.unlit              = unlit.active ? 1 : 0;

  KHR_materials_iridescence iridescence = tinygltf::utils::getIridescence(srcMat);
  dstMat.iridescenceFactor              = iridescence.iridescenceFactor;
  dstMat.iridescenceTexture             = addTextureInfo(iridescence.iridescenceTexture, textureInfos);
  dstMat.iridescenceIor                 = iridescence.iridescenceIor;
  dstMat.iridescenceThicknessMaximum    = iridescence.iridescenceThicknessMaximum;
  dstMat.iridescenceThicknessMinimum    = iridescence.iridescenceThicknessMinimum;
  dstMat.iridescenceThicknessTexture    = addTextureInfo(iridescence.iridescenceThicknessTexture, textureInfos);

  KHR_materials_anisotropy anisotropy = tinygltf::utils::getAnisotropy(srcMat);
  dstMat.anisotropyRotation = glm::vec2(glm::sin(anisotropy.anisotropyRotation), glm::cos(anisotropy.anisotropyRotation));
  dstMat.anisotropyStrength = anisotropy.anisotropyStrength;
  dstMat.anisotropyTexture  = addTextureInfo(anisotropy.anisotropyTexture, textureInfos);

  KHR_materials_sheen sheen    = tinygltf::utils::getSheen(srcMat);
  dstMat.sheenColorFactor      = sheen.sheenColorFactor;
  dstMat.sheenColorTexture     = addTextureInfo(sheen.sheenColorTexture, textureInfos);
  dstMat.sheenRoughnessFactor  = sheen.sheenRoughnessFactor;
  dstMat.sheenRoughnessTexture = addTextureInfo(sheen.sheenRoughnessTexture, textureInfos);

  KHR_materials_dispersion dispersion = tinygltf::utils::getDispersion(srcMat);
  dstMat.dispersion                   = dispersion.dispersion;

  KHR_materials_pbrSpecularGlossiness pbr = tinygltf::utils::getPbrSpecularGlossiness(srcMat);
  dstMat.usePbrSpecularGlossiness =
      tinygltf::utils::hasElementName(srcMat.extensions, KHR_MATERIALS_PBR_SPECULAR_GLOSSINESS_EXTENSION_NAME);
  if(dstMat.usePbrSpecularGlossiness)
  {
    dstMat.pbrDiffuseFactor             = pbr.diffuseFactor;
    dstMat.pbrSpecularFactor            = pbr.specularFactor;
    dstMat.pbrGlossinessFactor          = pbr.glossinessFactor;
    dstMat.pbrDiffuseTexture            = addTextureInfo(pbr.diffuseTexture, textureInfos);
    dstMat.pbrSpecularGlossinessTexture = addTextureInfo(pbr.specularGlossinessTexture, textureInfos);
  }

  KHR_materials_diffuse_transmission diffuseTransmission = tinygltf::utils::getDiffuseTransmission(srcMat);
  dstMat.diffuseTransmissionFactor                       = diffuseTransmission.diffuseTransmissionFactor;
  dstMat.diffuseTransmissionTexture = addTextureInfo(diffuseTransmission.diffuseTransmissionTexture, textureInfos);
  dstMat.diffuseTransmissionColor   = diffuseTransmission.diffuseTransmissionColor;
  dstMat.diffuseTransmissionColorTexture = addTextureInfo(diffuseTransmission.diffuseTransmissionColorTexture, textureInfos);

  shadeMaterial.emplace_back(dstMat);
}

//--------------------------------------------------------------------------------------------------
// Create a buffer of all materials, with only the elements we need
//
void nvvkgltf::SceneVk::updateMaterialBuffer(VkCommandBuffer cmd, nvvk::StagingUploader& staging, const nvvkgltf::Scene& scn)
{
  nvutils::ScopedTimer st(__FUNCTION__);

  using namespace tinygltf;
  const std::vector<tinygltf::Material>& materials = scn.getModel().materials;

  std::vector<shaderio::GltfShadeMaterial> shadeMaterials;
  std::vector<shaderio::GltfTextureInfo>   textureInfos;
  textureInfos.push_back({});  // 0 is reserved for no texture
  shadeMaterials.reserve(materials.size());
  for(const auto& srcMat : materials)
  {
    getShaderMaterial(srcMat, shadeMaterials, textureInfos);
  }

  if(m_bMaterial.buffer == VK_NULL_HANDLE)
  {
    NVVK_CHECK(m_alloc->createBuffer(m_bMaterial, std::span(shadeMaterials).size_bytes(),
                                     VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT));
    NVVK_CHECK(staging.appendBuffer(m_bMaterial, 0, std::span(shadeMaterials)));
    NVVK_DBG_NAME(m_bMaterial.buffer);
    m_memoryTracker.track(kMemCategorySceneData, m_bMaterial.allocation);

    NVVK_CHECK(m_alloc->createBuffer(m_bTextureInfos, std::span(textureInfos).size_bytes(),
                                     VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT));
    NVVK_CHECK(staging.appendBuffer(m_bTextureInfos, 0, std::span(textureInfos)));
    NVVK_DBG_NAME(m_bTextureInfos.buffer);
    m_memoryTracker.track(kMemCategorySceneData, m_bTextureInfos.allocation);
  }
  else
  {
    staging.appendBuffer(m_bMaterial, 0, std::span(shadeMaterials));
    staging.appendBuffer(m_bTextureInfos, 0, std::span(textureInfos));
  }
}

// Function to blend positions of a primitive with morph targets
std::vector<glm::vec3> getBlendedPositions(const tinygltf::Accessor&  baseAccessor,
                                           const glm::vec3*           basePositionData,
                                           const tinygltf::Primitive& primitive,
                                           const tinygltf::Mesh&      mesh,
                                           const tinygltf::Model&     model)
{
  // Prepare for blending positions
  std::vector<glm::vec3> blendedPositions(baseAccessor.count);
  std::copy(basePositionData, basePositionData + baseAccessor.count, blendedPositions.begin());

  // Blend the positions with the morph targets
  for(size_t targetIndex = 0; targetIndex < primitive.targets.size(); ++targetIndex)
  {
    // Retrieve the weight for the current morph target
    float weight = float(mesh.weights[targetIndex]);
    if(weight == 0.0f)
      continue;  // Skip this morph target if its weight is zero

    // Get the morph target attribute (e.g., POSITION)
    const auto& findResult = primitive.targets[targetIndex].find("POSITION");
    if(findResult != primitive.targets[targetIndex].end())
    {
      const tinygltf::Accessor& morphAccessor = model.accessors[findResult->second];
      std::vector<glm::vec3>    tempStorage;
      const std::span<const glm::vec3> morphTargetData = tinygltf::utils::getAccessorData(model, morphAccessor, &tempStorage);

      // Apply the morph target offset in parallel, scaled by the corresponding weight
      nvutils::parallel_batches(blendedPositions.size(),
                                [&](uint64_t v) { blendedPositions[v] += weight * morphTargetData[v]; });
    }
  }

  return blendedPositions;
}

// Function to calculate skinned positions for a primitive
std::vector<glm::vec3> getSkinnedPositions(const std::span<const glm::vec3>&  basePositionData,
                                           const std::span<const glm::vec4>&  weights,
                                           const std::span<const glm::ivec4>& joints,
                                           const std::vector<glm::mat4>&      jointMatrices)
{
  size_t vertexCount = weights.size();

  // Prepare the output skinned positions
  std::vector<glm::vec3> skinnedPositions(vertexCount);

  // Apply skinning using multi-threading
  nvutils::parallel_batches<2048>(weights.size(), [&](uint64_t v) {
    glm::vec3 skinnedPosition(0.0f);

    // Skinning: blend the position based on joint weights and transforms
    for(int i = 0; i < 4; ++i)
    {
      const float& jointWeight = weights[v][i];
      if(jointWeight > 0.0f)
      {
        const int& jointIndex = joints[v][i];
        skinnedPosition += jointWeight * glm::vec3(jointMatrices[jointIndex] * glm::vec4(basePositionData[v], 1.0f));
      }
    }

    skinnedPositions[v] = skinnedPosition;
  });

  return skinnedPositions;
}

//--------------------------------------------------------------------------------------------------
// Array of instance information
// - Use by the vertex shader to retrieve the position of the instance
void nvvkgltf::SceneVk::updateRenderNodesBuffer(VkCommandBuffer cmd, nvvk::StagingUploader& staging, const nvvkgltf::Scene& scn)
{
  // nvutils::ScopedTimer st(__FUNCTION__);

  std::vector<shaderio::GltfRenderNode> instanceInfo;
  for(const nvvkgltf::RenderNode& renderNode : scn.getRenderNodes())
  {
    shaderio::GltfRenderNode info{};
    info.objectToWorld = renderNode.worldMatrix;
    info.worldToObject = glm::inverse(renderNode.worldMatrix);
    info.materialID    = renderNode.materialID;
    info.renderPrimID  = renderNode.renderPrimID;
    instanceInfo.emplace_back(info);
  }
  if(m_bRenderNode.buffer == VK_NULL_HANDLE)
  {
    NVVK_CHECK(m_alloc->createBuffer(m_bRenderNode, std::span(instanceInfo).size_bytes(),
                                     VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT));
    NVVK_CHECK(staging.appendBuffer(m_bRenderNode, 0, std::span(instanceInfo)));
    NVVK_DBG_NAME(m_bRenderNode.buffer);
    m_memoryTracker.track(kMemCategorySceneData, m_bRenderNode.allocation);
  }
  else
  {
    staging.appendBuffer(m_bRenderNode, 0, std::span(instanceInfo));
  }
}


//--------------------------------------------------------------------------------------------------
// Update the buffer of all lights
// - If the light data was changes, the buffer needs to be updated
void nvvkgltf::SceneVk::updateRenderLightsBuffer(VkCommandBuffer cmd, nvvk::StagingUploader& staging, const nvvkgltf::Scene& scn)
{
  const std::vector<nvvkgltf::RenderLight>& rlights = scn.getRenderLights();
  if(rlights.empty())
    return;

  std::vector<shaderio::GltfLight> shaderLights = getShaderLights(rlights, scn.getModel().lights);

  if(m_bLights.buffer == VK_NULL_HANDLE)
  {
    NVVK_CHECK(m_alloc->createBuffer(m_bLights, std::span(shaderLights).size_bytes(),
                                     VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT));
    NVVK_CHECK(staging.appendBuffer(m_bLights, 0, std::span(shaderLights)));
    NVVK_DBG_NAME(m_bLights.buffer);
    m_memoryTracker.track(kMemCategorySceneData, m_bLights.allocation);
  }
  else
  {
    staging.appendBuffer(m_bLights, 0, std::span(shaderLights));
  }
}

//--------------------------------------------------------------------------------------------------
// Update the buffer of all primitives that have morph targets
//
void nvvkgltf::SceneVk::updateRenderPrimitivesBuffer(VkCommandBuffer cmd, nvvk::StagingUploader& staging, const nvvkgltf::Scene& scn)
{
  const tinygltf::Model& model = scn.getModel();

  // ** Morph **
  for(uint32_t renderPrimID : scn.getMorphPrimitives())
  {
    const nvvkgltf::RenderPrimitive& renderPrimitive  = scn.getRenderPrimitive(renderPrimID);
    const tinygltf::Primitive&       primitive        = *renderPrimitive.pPrimitive;
    const tinygltf::Mesh&            mesh             = model.meshes[renderPrimitive.meshID];
    const tinygltf::Accessor&        positionAccessor = model.accessors[primitive.attributes.at("POSITION")];
    std::vector<glm::vec3>           tempStorage;
    const std::span<const glm::vec3> positionData = tinygltf::utils::getAccessorData(model, positionAccessor, &tempStorage);

    // Get blended position
    std::vector<glm::vec3> blendedPositions = getBlendedPositions(positionAccessor, positionData.data(), primitive, mesh, model);

    // Flush any pending buffer operations and add synchronization before updating morph/skinning buffers
    staging.cmdUploadAppended(cmd);
    nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_2_COPY_BIT,
                           VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);

    // Update buffer
    VertexBuffers& vertexBuffers = m_vertexBuffers[renderPrimID];
    staging.appendBuffer(vertexBuffers.position, 0, std::span(blendedPositions));
  }

  // ** Skin **
  const std::vector<nvvkgltf::RenderNode>& renderNodes = scn.getRenderNodes();
  for(uint32_t skinNodeID : scn.getSkinNodes())
  {
    const nvvkgltf::RenderNode& skinNode  = renderNodes[skinNodeID];
    const tinygltf::Skin&       skin      = model.skins[skinNode.skinID];
    const tinygltf::Primitive&  primitive = *scn.getRenderPrimitive(skinNode.renderPrimID).pPrimitive;

    int32_t                numJoints = int32_t(skin.joints.size());
    std::vector<glm::mat4> inverseBindMatrices(numJoints, glm::mat4(1));
    std::vector<glm::mat4> jointMatrices(numJoints, glm::mat4(1));

    if(skin.inverseBindMatrices > -1)
    {
      std::vector<glm::mat4> storage;
      std::span<const glm::mat4> ibm = tinygltf::utils::getAccessorData(model, model.accessors[skin.inverseBindMatrices], &storage);
      for(int i = 0; i < numJoints; i++)
      {
        inverseBindMatrices[i] = ibm[i];
      }
    }

    // Calculate joint matrices
    const std::vector<glm::mat4>& nodeMatrices = scn.getNodesWorldMatrices();
    glm::mat4 invNode = glm::inverse(nodeMatrices[skinNode.refNodeID]);  // Removing current node transform as it will be applied by the shaders
    for(int i = 0; i < numJoints; ++i)
    {
      int jointNodeID = skin.joints[i];
      jointMatrices[i] = invNode * nodeMatrices[jointNodeID] * inverseBindMatrices[i];  // World matrix of the joint's node
    }

    // Getting the weights of all positions/joint
    std::vector<glm::vec4> tempWeightStorage;
    std::span<const glm::vec4> weights = tinygltf::utils::getAttributeData3(model, primitive, "WEIGHTS_0", &tempWeightStorage);

    // Getting the joint that each position is using
    std::vector<glm::ivec4> tempJointStorage;
    std::span<const glm::ivec4> joints = tinygltf::utils::getAttributeData3(model, primitive, "JOINTS_0", &tempJointStorage);

    // Original vertex positions
    std::vector<glm::vec3>           tempPosStorage;
    const std::span<const glm::vec3> basePositionData =
        tinygltf::utils::getAttributeData3(model, primitive, "POSITION", &tempPosStorage);

    // Get skinned positions
    std::vector<glm::vec3> skinnedPositions = getSkinnedPositions(basePositionData, weights, joints, jointMatrices);

    // Flush any pending buffer operations and add synchronization before updating morph/skinning buffers
    staging.cmdUploadAppended(cmd);
    nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_2_COPY_BIT,
                           VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);

    // Update buffer
    VertexBuffers& vertexBuffers = m_vertexBuffers[skinNode.renderPrimID];
    staging.appendBuffer(vertexBuffers.position, 0, std::span(skinnedPositions));
  }
}

// Function to create attribute buffers in Vulkan only if the attribute is present
// Return true if a buffer was created, false if the buffer was updated
template <typename T>
bool nvvkgltf::SceneVk::updateAttributeBuffer(VkCommandBuffer cmd,               // Command buffer to record the copy
                                              const std::string& attributeName,  // Name of the attribute: POSITION, NORMAL, ...
                                              const tinygltf::Model&     model,      // GLTF model
                                              const tinygltf::Primitive& primitive,  // GLTF primitive
                                              nvvk::ResourceAllocator*   alloc,      // Allocator to create the buffer
                                              nvvk::StagingUploader*     staging,
                                              nvvk::Buffer&              attributeBuffer)  // Buffer to be created
{
  const auto& findResult = primitive.attributes.find(attributeName);
  if(findResult != primitive.attributes.end())
  {
    const tinygltf::Accessor& accessor = model.accessors[findResult->second];
    std::vector<T>            tempStorage;
    const std::span<const T>  data = tinygltf::utils::getAccessorData(model, accessor, &tempStorage);
    if(data.empty())
    {
      return false;  // The data was invalid
    }

    if(attributeBuffer.buffer == VK_NULL_HANDLE)
    {
      // We add VK_BUFFER_USAGE_2_VERTEX_BUFFER_BIT so it can be bound to
      // a vertex input binding:
      VkBufferUsageFlags2 bufferUsageFlag = getBufferUsageFlags() | VK_BUFFER_USAGE_2_VERTEX_BUFFER_BIT;
      NVVK_CHECK(alloc->createBuffer(attributeBuffer, std::span(data).size_bytes(), bufferUsageFlag));
      NVVK_CHECK(staging->appendBuffer(attributeBuffer, 0, std::span(data)));
      m_memoryTracker.track(kMemCategoryGeometry, attributeBuffer.allocation);
      return true;
    }
    else
    {
      staging->appendBuffer(attributeBuffer, 0, std::span(data));
    }
  }
  return false;
}

//--------------------------------------------------------------------------------------------------
// Returns the common usage flags used for all buffers.
VkBufferUsageFlags2 nvvkgltf::SceneVk::getBufferUsageFlags() const
{
  VkBufferUsageFlags2 bufferUsageFlag =
      VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT           // Buffer read/write access within shaders, without size limitation
      | VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT  // The buffer can be referred to using its address instead of a binding
      | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT           // Buffer can be copied into
      | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT;          // Buffer can be copied from (e.g. for inspection)

  if(m_rayTracingEnabled)
  {
    // Usage as a data source for acceleration structure builds
    bufferUsageFlag |= VK_BUFFER_USAGE_2_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;
  }

  return bufferUsageFlag;
}

//--------------------------------------------------------------------------------------------------
// Creating information per primitive
// - Create a buffer of Vertex and Index for each primitive
// - Each primInfo has a reference to the vertex and index buffer, and which material id it uses
//
void nvvkgltf::SceneVk::createVertexBuffers(VkCommandBuffer cmd, nvvk::StagingUploader& staging, const nvvkgltf::Scene& scn)
{
  nvutils::ScopedTimer st(__FUNCTION__);

  const auto& model = scn.getModel();

  std::vector<shaderio::GltfRenderPrimitive> renderPrim;  // The array of all primitive information

  size_t numUniquePrimitive = scn.getNumRenderPrimitives();
  m_bIndices.resize(numUniquePrimitive);
  m_vertexBuffers.resize(numUniquePrimitive);
  renderPrim.resize(numUniquePrimitive);

  for(size_t primID = 0; primID < scn.getNumRenderPrimitives(); primID++)
  {
    const tinygltf::Primitive& primitive     = *scn.getRenderPrimitive(primID).pPrimitive;
    const tinygltf::Mesh&      mesh          = model.meshes[scn.getRenderPrimitive(primID).meshID];
    VertexBuffers&             vertexBuffers = m_vertexBuffers[primID];

    updateAttributeBuffer<glm::vec3>(cmd, "POSITION", model, primitive, m_alloc, &staging, vertexBuffers.position);
    updateAttributeBuffer<glm::vec3>(cmd, "NORMAL", model, primitive, m_alloc, &staging, vertexBuffers.normal);
    updateAttributeBuffer<glm::vec2>(cmd, "TEXCOORD_0", model, primitive, m_alloc, &staging, vertexBuffers.texCoord0);
    updateAttributeBuffer<glm::vec2>(cmd, "TEXCOORD_1", model, primitive, m_alloc, &staging, vertexBuffers.texCoord1);
    updateAttributeBuffer<glm::vec4>(cmd, "TANGENT", model, primitive, m_alloc, &staging, vertexBuffers.tangent);

    if(tinygltf::utils::hasElementName(primitive.attributes, "COLOR_0"))
    {
      // For color, we need to pack it into a single int
      const tinygltf::Accessor& accessor = model.accessors[primitive.attributes.at("COLOR_0")];
      std::vector<uint32_t>     tempIntData(accessor.count);
      if(accessor.type == TINYGLTF_TYPE_VEC3)
      {
        std::vector<glm::vec3>     tempStorage;
        std::span<const glm::vec3> colors = tinygltf::utils::getAccessorData(model, accessor, &tempStorage);
        for(size_t i = 0; i < accessor.count; i++)
        {
          tempIntData[i] = glm::packUnorm4x8(glm::vec4(colors[i], 1));
        }
      }
      else if(accessor.type == TINYGLTF_TYPE_VEC4)
      {
        std::vector<glm::vec4>     tempStorage;
        std::span<const glm::vec4> colors = tinygltf::utils::getAccessorData(model, accessor, &tempStorage);
        for(size_t i = 0; i < accessor.count; i++)
        {
          tempIntData[i] = glm::packUnorm4x8(colors[i]);
        }
      }
      else
      {
        assert(!"Unknown color type");
      }

      NVVK_CHECK(m_alloc->createBuffer(vertexBuffers.color, std::span(tempIntData).size_bytes(),
                                       getBufferUsageFlags() | VK_BUFFER_USAGE_2_VERTEX_BUFFER_BIT));
      NVVK_CHECK(staging.appendBuffer(vertexBuffers.color, 0, std::span(tempIntData)));
      m_memoryTracker.track(kMemCategoryGeometry, vertexBuffers.color.allocation);
    }

    // Debug name
    if(vertexBuffers.position.buffer != VK_NULL_HANDLE)
      NVVK_DBG_NAME(vertexBuffers.position.buffer);
    if(vertexBuffers.normal.buffer != VK_NULL_HANDLE)
      NVVK_DBG_NAME(vertexBuffers.normal.buffer);
    if(vertexBuffers.texCoord0.buffer != VK_NULL_HANDLE)
      NVVK_DBG_NAME(vertexBuffers.texCoord0.buffer);
    if(vertexBuffers.texCoord1.buffer != VK_NULL_HANDLE)
      NVVK_DBG_NAME(vertexBuffers.texCoord1.buffer);
    if(vertexBuffers.tangent.buffer != VK_NULL_HANDLE)
      NVVK_DBG_NAME(vertexBuffers.tangent.buffer);
    if(vertexBuffers.color.buffer != VK_NULL_HANDLE)
      NVVK_DBG_NAME(vertexBuffers.color.buffer);


    // Buffer of indices
    std::vector<uint32_t> indexBuffer;
    if(primitive.indices > -1)
    {
      const tinygltf::Accessor& accessor = model.accessors[primitive.indices];
      bool                      ok       = tinygltf::utils::copyAccessorData(model, accessor, indexBuffer);
      assert(ok);
    }
    else
    {  // Primitive without indices, creating them
      const tinygltf::Accessor& accessor = model.accessors[primitive.attributes.at("POSITION")];

      indexBuffer.resize(accessor.count);
      for(auto i = 0; i < accessor.count; i++)
        indexBuffer[i] = i;
    }

    // Creating the buffer for the indices
    nvvk::Buffer& i_buffer = m_bIndices[primID];
    NVVK_CHECK(m_alloc->createBuffer(i_buffer, std::span(indexBuffer).size_bytes(),
                                     getBufferUsageFlags() | VK_BUFFER_USAGE_2_INDEX_BUFFER_BIT));
    NVVK_CHECK(staging.appendBuffer(i_buffer, 0, std::span(indexBuffer)));
    NVVK_DBG_NAME(i_buffer.buffer);
    m_memoryTracker.track(kMemCategoryGeometry, i_buffer.allocation);

    // Filling the primitive information
    renderPrim[primID].indices = (glm::uvec3*)i_buffer.address;

    shaderio::VertexBuffers vBuf = {};
    vBuf.positions               = (glm::vec3*)vertexBuffers.position.address;
    vBuf.normals                 = (glm::vec3*)vertexBuffers.normal.address;
    vBuf.tangents                = (glm::vec4*)vertexBuffers.tangent.address;
    vBuf.texCoords0              = (glm::vec2*)vertexBuffers.texCoord0.address;
    vBuf.texCoords1              = (glm::vec2*)vertexBuffers.texCoord1.address;
    vBuf.colors                  = (glm::uint*)vertexBuffers.color.address;

    renderPrim[primID].vertexBuffer = vBuf;
  }

  // Creating the buffer of all primitive information
  NVVK_CHECK(m_alloc->createBuffer(m_bRenderPrim, std::span(renderPrim).size_bytes(), getBufferUsageFlags()));
  NVVK_CHECK(staging.appendBuffer(m_bRenderPrim, 0, std::span(renderPrim)));
  NVVK_DBG_NAME(m_bRenderPrim.buffer);
  m_memoryTracker.track(kMemCategorySceneData, m_bRenderPrim.allocation);

  // Barrier to make sure the data is in the GPU
  VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
  barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
  if(m_rayTracingEnabled)
  {
    barrier.dstAccessMask |= VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
  }
  vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &barrier, 0,
                       nullptr, 0, nullptr);
}

// This version updates all the vertex buffers
void nvvkgltf::SceneVk::updateVertexBuffers(VkCommandBuffer cmd, nvvk::StagingUploader& staging, const nvvkgltf::Scene& scene)
{
  const auto& model = scene.getModel();

  for(size_t primID = 0; primID < scene.getNumRenderPrimitives(); primID++)
  {
    const tinygltf::Primitive& primitive     = *scene.getRenderPrimitive(primID).pPrimitive;
    VertexBuffers&             vertexBuffers = m_vertexBuffers[primID];
    bool                       newBuffer     = false;
    updateAttributeBuffer<glm::vec3>(cmd, "POSITION", model, primitive, m_alloc, &staging, vertexBuffers.position);
    newBuffer |= updateAttributeBuffer<glm::vec3>(cmd, "NORMAL", model, primitive, m_alloc, &staging, vertexBuffers.normal);
    newBuffer |= updateAttributeBuffer<glm::vec2>(cmd, "TEXCOORD_0", model, primitive, m_alloc, &staging, vertexBuffers.texCoord0);
    newBuffer |= updateAttributeBuffer<glm::vec2>(cmd, "TEXCOORD_1", model, primitive, m_alloc, &staging, vertexBuffers.texCoord1);
    newBuffer |= updateAttributeBuffer<glm::vec4>(cmd, "TANGENT", model, primitive, m_alloc, &staging, vertexBuffers.tangent);

    // A buffer was created (most likely tangent buffer), we need to update the RenderPrimitive buffer
    if(newBuffer)
    {
      shaderio::GltfRenderPrimitive renderPrim{};  // The array of all primitive information
      renderPrim.indices                 = (glm::uvec3*)m_bIndices[primID].address;
      renderPrim.vertexBuffer.positions  = (glm::vec3*)vertexBuffers.position.address;
      renderPrim.vertexBuffer.normals    = (glm::vec3*)vertexBuffers.normal.address;
      renderPrim.vertexBuffer.tangents   = (glm::vec4*)vertexBuffers.tangent.address;
      renderPrim.vertexBuffer.texCoords0 = (glm::vec2*)vertexBuffers.texCoord0.address;
      renderPrim.vertexBuffer.texCoords1 = (glm::vec2*)vertexBuffers.texCoord1.address;
      renderPrim.vertexBuffer.colors     = (glm::uint*)vertexBuffers.color.address;
      staging.appendBuffer(m_bRenderPrim, sizeof(shaderio::GltfRenderPrimitive) * primID, std::span(&renderPrim, 1));
    }
  }
}


//--------------------------------------------------------------------------------------------------------------
// Returning the Vulkan sampler information from the information in the tinygltf
//
static VkSamplerCreateInfo getSampler(const tinygltf::Model& model, int index)
{
  VkSamplerCreateInfo samplerInfo{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
  samplerInfo.minFilter  = VK_FILTER_LINEAR;
  samplerInfo.magFilter  = VK_FILTER_LINEAR;
  samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
  samplerInfo.maxLod     = VK_LOD_CLAMP_NONE;

  if(index < 0)
    return samplerInfo;

  const auto& sampler = model.samplers[index];

  const std::map<int, VkFilter> filters = {{9728, VK_FILTER_NEAREST}, {9729, VK_FILTER_LINEAR},
                                           {9984, VK_FILTER_NEAREST}, {9985, VK_FILTER_LINEAR},
                                           {9986, VK_FILTER_NEAREST}, {9987, VK_FILTER_LINEAR}};

  const std::map<int, VkSamplerMipmapMode> mipmapModes = {
      {9728, VK_SAMPLER_MIPMAP_MODE_NEAREST}, {9729, VK_SAMPLER_MIPMAP_MODE_LINEAR},
      {9984, VK_SAMPLER_MIPMAP_MODE_NEAREST}, {9985, VK_SAMPLER_MIPMAP_MODE_LINEAR},
      {9986, VK_SAMPLER_MIPMAP_MODE_NEAREST}, {9987, VK_SAMPLER_MIPMAP_MODE_LINEAR}};

  const std::map<int, VkSamplerAddressMode> wrapModes = {
      {TINYGLTF_TEXTURE_WRAP_REPEAT, VK_SAMPLER_ADDRESS_MODE_REPEAT},
      {TINYGLTF_TEXTURE_WRAP_CLAMP_TO_EDGE, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE},
      {TINYGLTF_TEXTURE_WRAP_MIRRORED_REPEAT, VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT}};

  if(sampler.minFilter > -1)
    samplerInfo.minFilter = filters.at(sampler.minFilter);
  if(sampler.magFilter > -1)
  {
    samplerInfo.magFilter  = filters.at(sampler.magFilter);
    samplerInfo.mipmapMode = mipmapModes.at(sampler.magFilter);
  }
  samplerInfo.addressModeU = wrapModes.at(sampler.wrapS);
  samplerInfo.addressModeV = wrapModes.at(sampler.wrapT);

  return samplerInfo;
}

//--------------------------------------------------------------------------------------------------------------
// This is creating all images stored in textures
//
void nvvkgltf::SceneVk::createTextureImages(VkCommandBuffer              cmd,
                                            nvvk::StagingUploader&       staging,
                                            const tinygltf::Model&       model,
                                            const std::filesystem::path& basedir)
{
  nvutils::ScopedTimer st(std::string(__FUNCTION__) + "\n");

  VkSamplerCreateInfo default_sampler{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
  default_sampler.minFilter  = VK_FILTER_LINEAR;
  default_sampler.magFilter  = VK_FILTER_LINEAR;
  default_sampler.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
  default_sampler.maxLod     = VK_LOD_CLAMP_NONE;

  // Find and all textures/images that should be sRgb encoded.
  findSrgbImages(model);

  // Make dummy image(1,1), needed as we cannot have an empty array
  auto addDefaultImage = [&](uint32_t idx, const std::array<uint8_t, 4>& color) {
    VkImageCreateInfo image_create_info = DEFAULT_VkImageCreateInfo;
    image_create_info.extent            = {1, 1, 1};
    image_create_info.usage             = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    nvvk::Image image;
    //
    NVVK_CHECK(m_alloc->createImage(image, image_create_info, DEFAULT_VkImageViewCreateInfo));
    NVVK_CHECK(staging.appendImage(image, std::span<const uint8_t>(color.data(), 4), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL));
    NVVK_DBG_NAME(image.image);
    // assert(idx < m_images.size());
    m_images[idx] = SceneImage{.imageTexture = image};
    nvvk::DebugUtil::getInstance().setObjectName(m_images[idx].imageTexture.image, "Dummy");
  };

  // Adds a texture that points to image 0, so that every texture points to some image.
  auto addDefaultTexture = [&]() {
    assert(!m_images.empty());
    nvvk::Image tex = m_images[0].imageTexture;
    NVVK_CHECK(m_samplerPool->acquireSampler(tex.descriptor.sampler));
    NVVK_DBG_NAME(tex.descriptor.sampler);
    m_textures.push_back(tex);
  };

  // Collect images that are in use by textures
  // If an image is not used, it will not be loaded. Instead, a dummy image will be created to avoid modifying the texture image source index.
  std::set<int> usedImages;
  for(const auto& texture : model.textures)
  {
    int source_image = tinygltf::utils::getTextureImageIndex(texture);
    usedImages.insert(source_image);
  }

  // Load images in parallel
  m_images.resize(model.images.size());
  uint32_t          num_threads = std::min((uint32_t)model.images.size(), std::thread::hardware_concurrency());
  const std::string indent      = st.indent();
  nvutils::parallel_batches<1>(  // Not batching
      model.images.size(),
      [&](uint64_t i) {
        if(usedImages.find(static_cast<int>(i)) == usedImages.end())
          return;  // Skip unused images
        const auto& image     = model.images[i];
        const char* imageName = image.uri.empty() ? "Embedded image" : image.uri.c_str();
        LOGI("%s(%" PRIu64 ") %s \n", indent.c_str(), i, imageName);
        loadImage(basedir, image, static_cast<int>(i));
      },
      num_threads);

  // Create Vulkan images
  for(size_t i = 0; i < m_images.size(); i++)
  {
    if(!createImage(cmd, staging, m_images[i]))
    {
      addDefaultImage((uint32_t)i, {255, 0, 255, 255});  // Image not present or incorrectly loaded (image.empty)
    }
  }

  // Add default image if nothing was loaded
  if(model.images.empty())
  {
    m_images.resize(1);
    addDefaultImage(0, {255, 255, 255, 255});
  }

  // Creating the textures using the above images
  m_textures.reserve(model.textures.size());
  for(size_t i = 0; i < model.textures.size(); i++)
  {
    const auto& texture      = model.textures[i];
    int         source_image = tinygltf::utils::getTextureImageIndex(texture);

    if(source_image >= model.images.size() || source_image < 0)
    {
      addDefaultTexture();  // Incorrect source image
      continue;
    }

    VkSamplerCreateInfo sampler = getSampler(model, texture.sampler);

    SceneImage& sceneImage = m_images[source_image];

    nvvk::Image tex = sceneImage.imageTexture;
    NVVK_CHECK(m_samplerPool->acquireSampler(tex.descriptor.sampler, sampler));
    NVVK_DBG_NAME(tex.descriptor.sampler);
    m_textures.push_back(tex);
  }

  // Add a default texture, cannot work with empty descriptor set
  if(model.textures.empty())
  {
    addDefaultTexture();
  }
}

//-------------------------------------------------------------------------------------------------
// Some images must be sRgb encoded, we find them and will be uploaded with the _SRGB format.
//
void nvvkgltf::SceneVk::findSrgbImages(const tinygltf::Model& model)
{
  // Lambda helper functions
  auto addImage = [&](int texID) {
    if(texID > -1)
    {
      const tinygltf::Texture& texture = model.textures[texID];
      m_sRgbImages.insert(tinygltf::utils::getTextureImageIndex(texture));
    }
  };

  // For images in extensions
  auto addImageFromExtension = [&](const tinygltf::Material& mat, const std::string extName, const std::string name) {
    const auto& ext = mat.extensions.find(extName);
    if(ext != mat.extensions.end())
    {
      if(ext->second.Has(name))
        addImage(ext->second.Get(name).Get<int>());
    }
  };

  // Loop over all materials and find the sRgb textures
  for(size_t matID = 0; matID < model.materials.size(); matID++)
  {
    const auto& mat = model.materials[matID];
    // https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#metallic-roughness-material
    addImage(mat.pbrMetallicRoughness.baseColorTexture.index);
    addImage(mat.emissiveTexture.index);

    // https://github.com/KhronosGroup/glTF/blob/main/extensions/2.0/Khronos/KHR_materials_specular/README.md#extending-materials
    addImageFromExtension(mat, "KHR_materials_specular", "specularColorTexture");

    // https://github.com/KhronosGroup/glTF/blob/main/extensions/2.0/Khronos/KHR_materials_sheen/README.md#sheen
    addImageFromExtension(mat, "KHR_materials_sheen", "sheenColorTexture");

    // **Deprecated** but still used with some scenes
    // https://kcoley.github.io/glTF/extensions/2.0/Khronos/KHR_materials_pbrSpecularGlossiness
    addImageFromExtension(mat, "KHR_materials_pbrSpecularGlossiness", "diffuseTexture");
    addImageFromExtension(mat, "KHR_materials_pbrSpecularGlossiness", "specularGlossinessTexture");
  }

  // Special, if the 'extra' in the texture has a gamma defined greater than 1.0, it is sRGB
  for(size_t texID = 0; texID < model.textures.size(); texID++)
  {
    const auto& texture = model.textures[texID];
    if(texture.extras.Has("gamma") && texture.extras.Get("gamma").GetNumberAsDouble() > 1.0)
    {
      m_sRgbImages.insert(tinygltf::utils::getTextureImageIndex(texture));
    }
  }
}

//--------------------------------------------------------------------------------------------------
// Loading images from disk
//
void nvvkgltf::SceneVk::loadImage(const std::filesystem::path& basedir, const tinygltf::Image& gltfImage, int imageID)
{
  namespace fs = std::filesystem;

  auto& image  = m_images[imageID];
  bool  isSrgb = m_sRgbImages.find(imageID) != m_sRgbImages.end();

  std::string uriDecoded;  // This is UTF-8, but TinyGlTF uses `char` instead of `char8_t` for it
  tinygltf::URIDecode(gltfImage.uri, &uriDecoded, nullptr);  // ex. whitespace may be represented as %20
  const fs::path uri = basedir / nvutils::pathFromUtf8(uriDecoded);
  image.imgName      = nvutils::utf8FromPath(uri.filename());

  if(nvutils::extensionMatches(uri, ".dds"))
  {
    nv_dds::Image               ddsImage{};
    nv_dds::ReadSettings        settings{};
    std::ifstream               imageFile(uri, std::ios::binary);
    const nv_dds::ErrorWithText readResult = ddsImage.readFromStream(imageFile, settings);
    if(readResult.has_value())
    {
      LOGW("Failed to read %s using nv_dds: %s\n", nvutils::utf8FromPath(uri).c_str(), readResult.value().c_str());
      return;
    }

    image.srgb        = isSrgb;
    image.size.width  = ddsImage.getWidth(0);
    image.size.height = ddsImage.getHeight(0);
    if(ddsImage.getDepth(0) > 1)
    {
      LOGW("This DDS image had a depth of %u, but loadImage() cannot handle volume textures.\n", ddsImage.getDepth(0));
      return;
    }
    if(ddsImage.getNumFaces() > 1)
    {
      LOGW("This DDS image had %u faces, but loadImage() cannot handle cubemaps.\n", ddsImage.getNumFaces());
      return;
    }
    if(ddsImage.getNumLayers() > 1)
    {
      LOGW("This DDS image had %u array elements, but loadImage() cannot handle array textures.\n", ddsImage.getNumLayers());
      return;
    }
    image.format = texture_formats::dxgiToVulkan(ddsImage.dxgiFormat);
    image.format = texture_formats::tryForceVkFormatTransferFunction(image.format, image.srgb);
    if(VK_FORMAT_UNDEFINED == image.format)
    {
      LOGW("Could not determine a VkFormat for DXGI format %u (%s).\n", ddsImage.dxgiFormat,
           texture_formats::getDXGIFormatName(ddsImage.dxgiFormat));
      return;
    }

    // Add all mip-levels. We don't need the ddsImage after this so we can move instead of copy.
    for(uint32_t i = 0; i < ddsImage.getNumMips(); i++)
    {
      std::vector<char>& mip = ddsImage.subresource(i, 0, 0).data;
      image.mipData.push_back(std::move(mip));
    }
  }
  else if(nvutils::extensionMatches(uri, ".ktx") || nvutils::extensionMatches(uri, ".ktx2"))
  {
    nv_ktx::KTXImage            ktxImage;
    const nv_ktx::ReadSettings  ktxReadSettings;
    std::ifstream               imageFile(uri, std::ios::binary);
    const nv_ktx::ErrorWithText maybeError = ktxImage.readFromStream(imageFile, ktxReadSettings);
    if(maybeError.has_value())
    {
      LOGW("Failed to read %s using nv_ktx: %s\n", nvutils::utf8FromPath(uri).c_str(), maybeError->c_str());
      return;
    }

    image.srgb        = isSrgb;
    image.size.width  = ktxImage.mip_0_width;
    image.size.height = ktxImage.mip_0_height;
    if(ktxImage.mip_0_depth > 1)
    {
      LOGW("This KTX image had a depth of %u, but loadImage() cannot handle volume textures.\n", ktxImage.mip_0_depth);
      return;
    }
    if(ktxImage.num_faces > 1)
    {
      LOGW("This KTX image had %u faces, but loadImage() cannot handle cubemaps.\n", ktxImage.num_faces);
      return;
    }
    if(ktxImage.num_layers_possibly_0 > 1)
    {
      LOGW("This KTX image had %u array elements, but loadImage() cannot handle array textures.\n", ktxImage.num_layers_possibly_0);
      return;
    }
    image.format = texture_formats::tryForceVkFormatTransferFunction(ktxImage.format, image.srgb);

    // Add all mip-levels. We don't need the ktxImage after this so we can move instead of copy.
    for(uint32_t i = 0; i < ktxImage.num_mips; i++)
    {
      std::vector<char>& mip = ktxImage.subresource(i, 0, 0);
      image.mipData.push_back(std::move(mip));
    }
  }
  else if(uri.has_extension())
  {
    // Read all contents to avoid text encoding issues with the filename
    const std::string imageFileContents = nvutils::loadFile(uri);
    if(imageFileContents.empty())
    {
      LOGW("File was empty or could not be opened: %s\n", nvutils::utf8FromPath(uri).c_str());
      return;
    }
    const stbi_uc* imageFileData = (const stbi_uc*)(imageFileContents.data());
    if(imageFileContents.size() > std::numeric_limits<int>::max())
    {
      LOGW("File too large for stb_image to read: %s\n", nvutils::utf8FromPath(uri).c_str());
      return;
    }
    const int imageFileSize = static_cast<int>(imageFileContents.size());

    // Read the header once to check how many channels it has. We can't trivially use RGB/VK_FORMAT_R8G8B8_UNORM and
    // need to set requiredComponents=4 in such cases.
    int w = 0, h = 0, comp = 0;
    if(!stbi_info_from_memory(imageFileData, imageFileSize, &w, &h, &comp))
    {
      LOGW("Failed to get info for %s\n", nvutils::utf8FromPath(uri).c_str());
      return;
    }

    // Read the header again to check if it has 16 bit data, e.g. for a heightmap.
    const bool is16Bit = stbi_is_16_bit_from_memory(imageFileData, imageFileSize);

    // Load the image
    stbi_uc* data = nullptr;
    size_t   bytesPerPixel{0};
    int      requiredComponents = comp == 1 ? 1 : 4;
    if(is16Bit)
    {
      stbi_us* data16 = stbi_load_16_from_memory(imageFileData, imageFileSize, &w, &h, &comp, requiredComponents);
      bytesPerPixel   = sizeof(*data16) * requiredComponents;
      data            = (stbi_uc*)(data16);
    }
    else
    {
      data          = stbi_load_from_memory(imageFileData, imageFileSize, &w, &h, &comp, requiredComponents);
      bytesPerPixel = sizeof(*data) * requiredComponents;
    }
    switch(requiredComponents)
    {
      case 1:
        image.format = is16Bit ? VK_FORMAT_R16_UNORM : VK_FORMAT_R8_UNORM;
        break;
      case 4:
        image.format = is16Bit ? VK_FORMAT_R16G16B16A16_UNORM :
                       isSrgb  ? VK_FORMAT_R8G8B8A8_SRGB :
                                 VK_FORMAT_R8G8B8A8_UNORM;

        break;
    }

    // Make a copy of the image data to be uploaded to vulkan later
    if(data && w > 0 && h > 0 && image.format != VK_FORMAT_UNDEFINED)
    {
      VkDeviceSize bufferSize = static_cast<VkDeviceSize>(w) * h * bytesPerPixel;
      image.size              = VkExtent2D{(uint32_t)w, (uint32_t)h};
      image.mipData           = {{data, data + bufferSize}};
    }

    stbi_image_free(data);
  }
  else if(gltfImage.width > 0 && gltfImage.height > 0 && !gltfImage.image.empty())
  {  // Loaded internally using GLB
    image.size   = VkExtent2D{(uint32_t)gltfImage.width, (uint32_t)gltfImage.height};
    image.format = isSrgb ? VK_FORMAT_R8G8B8A8_SRGB : VK_FORMAT_R8G8B8A8_UNORM;
    image.mipData.emplace_back(gltfImage.image.data(), gltfImage.image.data() + gltfImage.image.size());
  }
}

bool nvvkgltf::SceneVk::createImage(const VkCommandBuffer& cmd, nvvk::StagingUploader& staging, SceneImage& image)
{
  if(image.size.width == 0 || image.size.height == 0)
    return false;

  VkFormat   format  = image.format;
  VkExtent2D imgSize = image.size;

  // Check if we can generate mipmap with the the incoming image
  bool               canGenerateMipmaps = false;
  VkFormatProperties formatProperties;
  vkGetPhysicalDeviceFormatProperties(m_physicalDevice, format, &formatProperties);
  if((formatProperties.optimalTilingFeatures & VK_FORMAT_FEATURE_BLIT_DST_BIT) == VK_FORMAT_FEATURE_BLIT_DST_BIT)
  {
    canGenerateMipmaps = true;
  }
  VkImageCreateInfo imageCreateInfo = DEFAULT_VkImageCreateInfo;
  imageCreateInfo.extent            = VkExtent3D{imgSize.width, imgSize.height, 1};
  imageCreateInfo.format            = format;
  imageCreateInfo.usage             = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;

  // Mip-mapping images were defined (.ktx, .dds), use the number of levels defined
  if(image.mipData.size() > 1)
  {
    imageCreateInfo.mipLevels = static_cast<uint32_t>(image.mipData.size());
  }
  else if(canGenerateMipmaps && m_generateMipmaps)
  {
    // Compute the number of mipmaps levels
    imageCreateInfo.mipLevels = nvvk::mipLevels(imgSize);
  }

  nvvk::Image resultImage;
  NVVK_CHECK(m_alloc->createImage(resultImage, imageCreateInfo, DEFAULT_VkImageViewCreateInfo));
  NVVK_DBG_NAME(resultImage.image);
  NVVK_DBG_NAME(resultImage.descriptor.imageView);

  // Track the image allocation
  m_memoryTracker.track(kMemCategoryImages, resultImage.allocation);

  // Set the initial layout to TRANSFER_DST_OPTIMAL
  resultImage.descriptor.imageLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;  // Setting this, tells the appendImage that the image is in this layout (no need to transfer)
  nvvk::cmdImageMemoryBarrier(cmd, {resultImage.image, VK_IMAGE_LAYOUT_UNDEFINED, resultImage.descriptor.imageLayout});
  NVVK_CHECK(staging.appendImage(resultImage, std::span(image.mipData[0]), resultImage.descriptor.imageLayout));
  staging.cmdUploadAppended(cmd);  // Upload the first mip level

  // The image require to generate the mipmaps
  if(image.mipData.size() == 1 && (canGenerateMipmaps && m_generateMipmaps))
  {
    nvvk::cmdGenerateMipmaps(cmd, resultImage.image, imgSize, imageCreateInfo.mipLevels, 1, resultImage.descriptor.imageLayout);
  }
  else
  {
    for(uint32_t mip = 1; mip < (uint32_t)imageCreateInfo.mipLevels; mip++)
    {
      imageCreateInfo.extent.width  = std::max(1u, image.size.width >> mip);
      imageCreateInfo.extent.height = std::max(1u, image.size.height >> mip);

      VkOffset3D               offset{};
      VkImageSubresourceLayers subresource{};
      subresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
      subresource.layerCount = 1;
      subresource.mipLevel   = mip;

      if(imageCreateInfo.extent.width > 0 && imageCreateInfo.extent.height > 0)
      {
        staging.appendImageSub(resultImage, offset, imageCreateInfo.extent, subresource, std::span(image.mipData[mip]));
      }
    }
    // Upload all the mip levels
    staging.cmdUploadAppended(cmd);
  }
  // Barrier to change the layout to SHADER_READ_ONLY_OPTIMAL
  resultImage.descriptor.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
  nvvk::cmdImageMemoryBarrier(cmd, {resultImage.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, resultImage.descriptor.imageLayout});

  if(!image.imgName.empty())
  {
    nvvk::DebugUtil::getInstance().setObjectName(resultImage.image, image.imgName);
  }
  else
  {
    NVVK_DBG_NAME(resultImage.image);
  }

  // Clear image.mipData as it is no longer needed
  // image.srgb and image.imgName are preserved
  image.imageTexture = resultImage;
  image.mipData.clear();

  return true;
}

std::vector<shaderio::GltfLight> getShaderLights(const std::vector<nvvkgltf::RenderLight>& renderlights,
                                                 const std::vector<tinygltf::Light>&       gltfLights)
{
  std::vector<shaderio::GltfLight> lightsInfo;
  lightsInfo.reserve(renderlights.size());
  for(auto& l : renderlights)
  {
    const auto& gltfLight = gltfLights[l.light];

    shaderio::GltfLight info{};
    info.position   = l.worldMatrix[3];
    info.direction  = -l.worldMatrix[2];  // glm::vec3(l.worldMatrix * glm::vec4(0, 0, -1, 0));
    info.innerAngle = static_cast<float>(gltfLight.spot.innerConeAngle);
    info.outerAngle = static_cast<float>(gltfLight.spot.outerConeAngle);
    if(gltfLight.color.size() == 3)
      info.color = glm::vec3(gltfLight.color[0], gltfLight.color[1], gltfLight.color[2]);
    else
      info.color = glm::vec3(1, 1, 1);  // default color (white)
    info.intensity = static_cast<float>(gltfLight.intensity);
    info.type      = gltfLight.type == "point" ? shaderio::eLightTypePoint :
                     gltfLight.type == "spot"  ? shaderio::eLightTypeSpot :
                                                 shaderio::eLightTypeDirectional;

    info.radius = gltfLight.extras.Has("radius") ? float(gltfLight.extras.Get("radius").GetNumberAsDouble()) : 0.0f;

    if(info.type == shaderio::eLightTypeDirectional)
    {
      const double sun_distance     = 149597870.0;  // km
      double       angular_size_rad = 2.0 * std::atan(info.radius / sun_distance);
      info.angularSizeOrInvRange    = static_cast<float>(angular_size_rad);
    }
    else
    {
      info.angularSizeOrInvRange = (gltfLight.range > 0.0) ? 1.0f / static_cast<float>(gltfLight.range) : 0.0f;
    }

    lightsInfo.emplace_back(info);
  }
  return lightsInfo;
}

void nvvkgltf::SceneVk::destroy()
{
  for(auto& vertexBuffer : m_vertexBuffers)
  {
    if(vertexBuffer.position.buffer != VK_NULL_HANDLE)
    {
      m_memoryTracker.untrack(kMemCategoryGeometry, vertexBuffer.position.allocation);
      m_alloc->destroyBuffer(vertexBuffer.position);
    }
    if(vertexBuffer.normal.buffer != VK_NULL_HANDLE)
    {
      m_memoryTracker.untrack(kMemCategoryGeometry, vertexBuffer.normal.allocation);
      m_alloc->destroyBuffer(vertexBuffer.normal);
    }
    if(vertexBuffer.tangent.buffer != VK_NULL_HANDLE)
    {
      m_memoryTracker.untrack(kMemCategoryGeometry, vertexBuffer.tangent.allocation);
      m_alloc->destroyBuffer(vertexBuffer.tangent);
    }
    if(vertexBuffer.texCoord0.buffer != VK_NULL_HANDLE)
    {
      m_memoryTracker.untrack(kMemCategoryGeometry, vertexBuffer.texCoord0.allocation);
      m_alloc->destroyBuffer(vertexBuffer.texCoord0);
    }
    if(vertexBuffer.texCoord1.buffer != VK_NULL_HANDLE)
    {
      m_memoryTracker.untrack(kMemCategoryGeometry, vertexBuffer.texCoord1.allocation);
      m_alloc->destroyBuffer(vertexBuffer.texCoord1);
    }
    if(vertexBuffer.color.buffer != VK_NULL_HANDLE)
    {
      m_memoryTracker.untrack(kMemCategoryGeometry, vertexBuffer.color.allocation);
      m_alloc->destroyBuffer(vertexBuffer.color);
    }
  }
  m_vertexBuffers.clear();

  for(auto& indicesBuffer : m_bIndices)
  {
    if(indicesBuffer.buffer != VK_NULL_HANDLE)
    {
      m_memoryTracker.untrack(kMemCategoryGeometry, indicesBuffer.allocation);
      m_alloc->destroyBuffer(indicesBuffer);
    }
  }
  m_bIndices.clear();

  if(m_bMaterial.buffer != VK_NULL_HANDLE)
  {
    m_memoryTracker.untrack(kMemCategorySceneData, m_bMaterial.allocation);
    m_alloc->destroyBuffer(m_bMaterial);
  }
  if(m_bTextureInfos.buffer != VK_NULL_HANDLE)
  {
    m_memoryTracker.untrack(kMemCategorySceneData, m_bTextureInfos.allocation);
    m_alloc->destroyBuffer(m_bTextureInfos);
  }
  if(m_bLights.buffer != VK_NULL_HANDLE)
  {
    m_memoryTracker.untrack(kMemCategorySceneData, m_bLights.allocation);
    m_alloc->destroyBuffer(m_bLights);
  }
  if(m_bRenderPrim.buffer != VK_NULL_HANDLE)
  {
    m_memoryTracker.untrack(kMemCategorySceneData, m_bRenderPrim.allocation);
    m_alloc->destroyBuffer(m_bRenderPrim);
  }
  if(m_bRenderNode.buffer != VK_NULL_HANDLE)
  {
    m_memoryTracker.untrack(kMemCategorySceneData, m_bRenderNode.allocation);
    m_alloc->destroyBuffer(m_bRenderNode);
  }
  if(m_bSceneDesc.buffer != VK_NULL_HANDLE)
  {
    m_memoryTracker.untrack(kMemCategorySceneData, m_bSceneDesc.allocation);
    m_alloc->destroyBuffer(m_bSceneDesc);
  }

  for(auto& texture : m_textures)
  {
    m_samplerPool->releaseSampler(texture.descriptor.sampler);
  }
  for(auto& image : m_images)
  {
    if(image.imageTexture.image != VK_NULL_HANDLE)
    {
      m_memoryTracker.untrack(kMemCategoryImages, image.imageTexture.allocation);
      m_alloc->destroyImage(image.imageTexture);
    }
  }
  m_images.clear();
  m_textures.clear();

  m_sRgbImages.clear();
}
