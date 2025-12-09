/*
 * Copyright (c) 2019-2025, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2019-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <execution>
#include <filesystem>
#include <unordered_set>

#include <glm/gtx/norm.hpp>
#include <fmt/format.h>
#include <meshoptimizer/src/meshoptimizer.h>

#include <nvutils/file_operations.hpp>
#include <nvutils/logger.hpp>
#include <nvutils/parallel_work.hpp>
#include <nvutils/timers.hpp>

#include "scene.hpp"

// List of supported extensions
static const std::set<std::string> supportedExtensions = {
    "KHR_lights_punctual",
    "KHR_materials_anisotropy",
    "KHR_materials_clearcoat",
    "KHR_materials_displacement",
    "KHR_materials_emissive_strength",
    "KHR_materials_ior",
    "KHR_materials_iridescence",
    "KHR_materials_sheen",
    "KHR_materials_specular",
    "KHR_materials_transmission",
    "KHR_materials_unlit",
    "KHR_materials_variants",
    "KHR_materials_volume",
    "KHR_texture_transform",
    "KHR_materials_dispersion",
    "KHR_node_visibility",
    "EXT_mesh_gpu_instancing",
    "NV_attributes_iray",
    "MSFT_texture_dds",
    "KHR_materials_pbrSpecularGlossiness",
    "KHR_materials_diffuse_transmission",
    "EXT_meshopt_compression",
#ifdef USE_DRACO
    "KHR_draco_mesh_compression",
#endif
#ifdef NVP_SUPPORTS_BASISU
    "KHR_texture_basisu",
#endif
};

// Given only a normal vector, finds a valid tangent.
//
// This uses the technique from "Improved accuracy when building an orthonormal
// basis" by Nelson Max, https://jcgt.org/published/0006/01/02.
// Any tangent-generating algorithm must produce at least one discontinuity
// when operating on a sphere (due to the hairy ball theorem); this has a
// small ring-shaped discontinuity at normal.z == -0.99998796.
static glm::vec4 makeFastTangent(const glm::vec3& n)
{
  if(n.z < -0.99998796F)  // Handle the singularity
  {
    return glm::vec4(0.0F, -1.0F, 0.0F, 1.0F);
  }
  const float a = 1.0F / (1.0F + n.z);
  const float b = -n.x * n.y * a;
  return glm::vec4(1.0F - n.x * n.x * a, b, -n.x, 1.0F);
}

// Loading a GLTF file and extracting all information
bool nvvkgltf::Scene::load(const std::filesystem::path& filename)
{
  nvutils::ScopedTimer st(std::string(__FUNCTION__) + "\n");
  const std::string    filenameUtf8 = nvutils::utf8FromPath(filename);
  LOGI("%s%s\n", nvutils::ScopedTimer::indent().c_str(), filenameUtf8.c_str());

  m_filename = filename;
  m_model    = {};
  tinygltf::TinyGLTF tcontext;
  std::string        warn;
  std::string        error;
  tcontext.SetMaxExternalFileSize(-1);  // No limit for external files (images, buffers, etc.)
  const std::string ext = nvutils::utf8FromPath(filename.extension());
  bool              result{false};
  if(ext == ".gltf")
  {
    result = tcontext.LoadASCIIFromFile(&m_model, &error, &warn, filenameUtf8.c_str());
  }
  else if(ext == ".glb")
  {
    result = tcontext.LoadBinaryFromFile(&m_model, &error, &warn, filenameUtf8.c_str());
  }
  else
  {
    LOGE("%sUnknown file extension: %s\n", st.indent().c_str(), ext.c_str());
    return false;
  }

  if(!result)
  {
    LOGW("%sError loading file: %s\n", st.indent().c_str(), filenameUtf8.c_str());
    LOGW("%s%s\n", st.indent().c_str(), warn.c_str());
    // This is LOGE because the user requested to load a (probably valid)
    // glTF file, but this loader can't do what the user asked it to.
    // Only the last one is LOGE so that all the messages print before the
    // breakpoint.
    LOGE("%s%s\n", st.indent().c_str(), error.c_str());
    clearParsedData();
    //assert(!"Error while loading scene");
    return result;
  }

  // Check for required extensions
  for(auto& extension : m_model.extensionsRequired)
  {
    if(supportedExtensions.find(extension) == supportedExtensions.end())
    {
      LOGE("%sRequired extension unsupported : %s\n", st.indent().c_str(), extension.c_str());
      clearParsedData();
      return false;
    }
  }

  // Check for used extensions
  for(auto& extension : m_model.extensionsUsed)
  {
    if(supportedExtensions.find(extension) == supportedExtensions.end())
    {
      LOGW("%sUsed extension unsupported : %s\n", st.indent().c_str(), extension.c_str());
    }
  }

  // Handle EXT_meshopt_compression by decompressing all buffer data at once
  if(std::find(m_model.extensionsUsed.begin(), m_model.extensionsUsed.end(), EXT_MESHOPT_COMPRESSION_EXTENSION_NAME)
     != m_model.extensionsUsed.end())
  {
    for(tinygltf::Buffer& buffer : m_model.buffers)
    {
      if(buffer.data.empty())
      {
        buffer.data.resize(buffer.byteLength);
        buffer.extensions.erase(EXT_MESHOPT_COMPRESSION_EXTENSION_NAME);
      }
    }

    // first used to tag buffers that can be removed after decompression
    std::vector<int> isFullyCompressedBuffer(m_model.buffers.size(), 1);

    for(auto& bufferView : m_model.bufferViews)
    {
      if(bufferView.buffer < 0)
        continue;

      bool warned = false;

      EXT_meshopt_compression mcomp;
      if(tinygltf::utils::getMeshoptCompression(bufferView, mcomp))
      {
        // this decoding logic was derived from `decompressMeshopt`
        // in https://github.com/zeux/meshoptimizer/blob/master/gltf/parsegltf.cpp


        const tinygltf::Buffer& sourceBuffer = m_model.buffers[mcomp.buffer];
        const unsigned char*    source       = &sourceBuffer.data[mcomp.byteOffset];
        assert(mcomp.byteOffset + mcomp.byteLength <= sourceBuffer.data.size());

        tinygltf::Buffer& resultBuffer = m_model.buffers[bufferView.buffer];
        unsigned char*    result       = &resultBuffer.data[bufferView.byteOffset];
        assert(bufferView.byteOffset + bufferView.byteLength <= resultBuffer.data.size());

        int  rc   = -1;
        bool warn = false;

        switch(mcomp.compressionMode)
        {
          case EXT_meshopt_compression::MESHOPT_COMPRESSION_MODE_ATTRIBUTES:
            warn = meshopt_decodeVertexVersion(source, mcomp.byteLength) != 0;
            rc   = meshopt_decodeVertexBuffer(result, mcomp.count, mcomp.byteStride, source, mcomp.byteLength);
            break;

          case EXT_meshopt_compression::MESHOPT_COMPRESSION_MODE_TRIANGLES:
            warn = meshopt_decodeIndexVersion(source, mcomp.byteLength) != 1;
            rc   = meshopt_decodeIndexBuffer(result, mcomp.count, mcomp.byteStride, source, mcomp.byteLength);
            break;

          case EXT_meshopt_compression::MESHOPT_COMPRESSION_MODE_INDICES:
            warn = meshopt_decodeIndexVersion(source, mcomp.byteLength) != 1;
            rc   = meshopt_decodeIndexSequence(result, mcomp.count, mcomp.byteStride, source, mcomp.byteLength);
            break;

          default:
            break;
        }

        if(rc != 0)
        {
          LOGW("EXT_meshopt_compression decompression failed\n");
          clearParsedData();
          return false;
        }

        if(warn && !warned)
        {
          LOGW("Warning: EXT_meshopt_compression data uses versions outside of the glTF specification (vertex 0 / index 1 expected)\n");
          warned = true;
        }

        switch(mcomp.compressionFilter)
        {
          case EXT_meshopt_compression::MESHOPT_COMPRESSION_FILTER_OCTAHEDRAL:
            meshopt_decodeFilterOct(result, mcomp.count, mcomp.byteStride);
            break;

          case EXT_meshopt_compression::MESHOPT_COMPRESSION_FILTER_QUATERNION:
            meshopt_decodeFilterQuat(result, mcomp.count, mcomp.byteStride);
            break;

          case EXT_meshopt_compression::MESHOPT_COMPRESSION_FILTER_EXPONENTIAL:
            meshopt_decodeFilterExp(result, mcomp.count, mcomp.byteStride);
            break;

          default:
            break;
        }

        // remove extension for saving uncompressed
        bufferView.extensions.erase(EXT_MESHOPT_COMPRESSION_EXTENSION_NAME);
      }

      isFullyCompressedBuffer[bufferView.buffer] = 0;
    }

    // remove fully compressed buffers
    // isFullyCompressedBuffer is repurposed as buffer index remap table
    size_t writeIndex = 0;
    for(size_t readIndex = 0; readIndex < m_model.buffers.size(); readIndex++)
    {
      if(isFullyCompressedBuffer[readIndex])
      {
        // buffer is removed
        isFullyCompressedBuffer[readIndex] = -1;
      }
      else
      {
        // compacted index of buffer
        isFullyCompressedBuffer[readIndex] = int(writeIndex);

        if(readIndex != writeIndex)
        {
          m_model.buffers[writeIndex] = std::move(m_model.buffers[readIndex]);
        }
        writeIndex++;
      }
    }
    m_model.buffers.resize(writeIndex);

    // remap existing buffer views
    for(auto& bufferView : m_model.bufferViews)
    {
      if(bufferView.buffer < 0)
        continue;

      bufferView.buffer = isFullyCompressedBuffer[bufferView.buffer];
    }

    // remove extension
    std::erase(m_model.extensionsRequired, EXT_MESHOPT_COMPRESSION_EXTENSION_NAME);
    std::erase(m_model.extensionsUsed, EXT_MESHOPT_COMPRESSION_EXTENSION_NAME);
  }

  m_currentScene   = m_model.defaultScene > -1 ? m_model.defaultScene : 0;
  m_currentVariant = 0;  // Default KHR_materials_variants
  parseScene();

  return result;
}

bool nvvkgltf::Scene::save(const std::filesystem::path& filename)
{
  namespace fs = std::filesystem;

  nvutils::ScopedTimer st(std::string(__FUNCTION__) + "\n");

  std::filesystem::path saveFilename = filename;

  // Make sure the extension is correct
  if(!nvutils::extensionMatches(filename, ".gltf") && !nvutils::extensionMatches(filename, ".glb"))
  {
    // replace the extension
    saveFilename = saveFilename.replace_extension(".gltf");
  }

  const bool saveBinary = nvutils::extensionMatches(filename, ".glb");

  // Copy the images to the destination folder
  if(!m_model.images.empty() && !saveBinary)
  {
    fs::path srcPath   = m_filename.parent_path();
    fs::path dstPath   = filename.parent_path();
    int      numCopied = 0;
    for(auto& image : m_model.images)
    {
      if(image.uri.empty())
        continue;
      std::string uri_decoded;
      tinygltf::URIDecode(image.uri, &uri_decoded, nullptr);  // ex. whitespace may be represented as %20

      fs::path srcFile = srcPath / uri_decoded;
      fs::path dstFile = dstPath / uri_decoded;
      if(srcFile != dstFile)
      {
        // Create the parent directory of the destination file if it doesn't exist
        fs::create_directories(dstFile.parent_path());

        if(fs::copy_file(srcFile, dstFile, fs::copy_options::update_existing))
          numCopied++;
      }
    }
    if(numCopied > 0)
      LOGI("%sImages copied: %d\n", st.indent().c_str(), numCopied);
  }

  // Save the glTF file
  tinygltf::TinyGLTF tcontext;
  const std::string  saveFilenameUtf8 = nvutils::utf8FromPath(saveFilename);
  bool result = tcontext.WriteGltfSceneToFile(&m_model, saveFilenameUtf8, saveBinary, saveBinary, true, saveBinary);
  LOGI("%sSaved: %s\n", st.indent().c_str(), saveFilenameUtf8.c_str());
  return result;
}


void nvvkgltf::Scene::takeModel(tinygltf::Model&& model)
{
  m_model = std::move(model);
  parseScene();
}

void nvvkgltf::Scene::setCurrentScene(int sceneID)
{
  assert(sceneID >= 0 && sceneID < static_cast<int>(m_model.scenes.size()) && "Invalid scene ID");
  m_currentScene = sceneID;
  parseScene();
}

// Parses the scene from the glTF model, initializing and setting up scene elements, materials, animations, and the camera.
void nvvkgltf::Scene::parseScene()
{
  // Ensure there are nodes in the glTF model and the current scene ID is valid
  assert(m_model.nodes.size() > 0 && "No nodes in the glTF file");
  assert(m_currentScene >= 0 && m_currentScene < static_cast<int>(m_model.scenes.size()) && "Invalid scene ID");

  // Clear previous scene data and initialize scene elements
  clearParsedData();
  setSceneElementsDefaultNames();

  // Ensure only one top node per scene, creating a new node if necessary
  // This is done to be able to transform the entire scene as a single node
  for(auto& scene : m_model.scenes)
  {
    createRootIfMultipleNodes(scene);
  }
  m_sceneRootNode = m_model.scenes[m_currentScene].nodes[0];  // Set the root node of the scene

  // There must be at least one material in the scene
  if(m_model.materials.empty())
  {
    m_model.materials.emplace_back();
  }

  // Collect all draw objects; RenderNode and RenderPrimitive
  // Also it will be used  to compute the scene bounds for the camera
  for(auto& sceneNode : m_model.scenes[m_currentScene].nodes)
  {
    tinygltf::utils::traverseSceneGraph(
        m_model, sceneNode, glm::mat4(1), nullptr,
        [this](int nodeID, const glm::mat4& worldMat) { return handleLightTraversal(nodeID, worldMat); },
        [this](int nodeID, const glm::mat4& worldMat) { return handleRenderNode(nodeID, worldMat); });
  }

  // Search for the first camera in the scene and exit traversal upon finding it
  for(auto& sceneNode : m_model.scenes[m_currentScene].nodes)
  {
    tinygltf::utils::traverseSceneGraph(
        m_model, sceneNode, glm::mat4(1),
        [&](int nodeID, glm::mat4 mat) {
          m_sceneCameraNode = nodeID;
          return true;  // Stop traversal
        },
        nullptr, nullptr);
  }

  // Create a default camera if none is found in the scene
  if(m_sceneCameraNode == -1)
  {
    createSceneCamera();
  }

  // Parse various scene components
  parseVariants();
  parseAnimations();
  createMissingTangents();

  // Update the visibility of the render nodes
  uint32_t renderNodeID = 0;
  for(const int sceneNode : m_model.scenes[m_currentScene].nodes)
  {
    bool visible = tinygltf::utils::getNodeVisibility(m_model.nodes[sceneNode]).visible;
    updateVisibility(sceneNode, visible, renderNodeID);
  }

  // We are updating the scene to the first state, animation, skinning, morph, ..
  updateRenderNodes();
}


// This function recursively updates the visibility of nodes in the scene graph.
// If a node is marked as not visible, all its children will also be marked as not visible,
// regardless of their individual visibility flags.
void nvvkgltf::Scene::updateVisibility(int nodeID, bool visible, uint32_t& renderNodeID)
{
  tinygltf::Node& node = m_model.nodes[nodeID];

  if(visible)
  {
    // Changing the visibility only if the parent was visible
    visible = tinygltf::utils::getNodeVisibility(node).visible;
  }

  if(node.mesh >= 0)
  {
    // If the node has a mesh, update the visibility of all its primitives
    const tinygltf::Mesh& mesh = m_model.meshes[node.mesh];
    for(size_t j = 0; j < mesh.primitives.size(); j++)
      m_renderNodes[renderNodeID++].visible = visible;
  }

  for(auto& child : node.children)
  {
    updateVisibility(child, visible, renderNodeID);
  }
}

// Set the default names for the scene elements if they are empty
void nvvkgltf::Scene::setSceneElementsDefaultNames()
{
  auto setDefaultName = [](auto& elements, const std::string& prefix) {
    for(size_t i = 0; i < elements.size(); ++i)
    {
      if(elements[i].name.empty())
      {
        elements[i].name = fmt::format("{}-{}", prefix, i);
      }
    }
  };

  setDefaultName(m_model.scenes, "Scene");
  setDefaultName(m_model.meshes, "Mesh");
  setDefaultName(m_model.materials, "Material");
  setDefaultName(m_model.nodes, "Node");
  setDefaultName(m_model.cameras, "Camera");
  setDefaultName(m_model.lights, "Light");
}


// Creates a new root node for the scene and assigns existing top nodes as its children.
void nvvkgltf::Scene::createRootIfMultipleNodes(tinygltf::Scene& scene)
{
  // Already a single node in the scene
  if(scene.nodes.size() == 1)
    return;

  tinygltf::Node newNode;
  newNode.name = scene.name;
  newNode.children.swap(scene.nodes);   // Move the scene nodes to the new node
  m_model.nodes.emplace_back(newNode);  // Add to then to avoid invalidating any references
  scene.nodes.clear();                  // Should be already empty, due to the swap
  scene.nodes.push_back(int(m_model.nodes.size()) - 1);
}

// If there is no camera in the scene, we create one
// The camera is placed at the center of the scene, looking at the scene
void nvvkgltf::Scene::createSceneCamera()
{
  tinygltf::Camera& tcamera        = m_model.cameras.emplace_back();  // Add a camera
  int               newCameraIndex = static_cast<int>(m_model.cameras.size() - 1);
  tinygltf::Node&   tnode          = m_model.nodes.emplace_back();  // Add a node for the camera
  int               newNodeIndex   = static_cast<int>(m_model.nodes.size() - 1);
  tnode.name                       = "Camera";
  tnode.camera                     = newCameraIndex;
  int rootID                       = m_model.scenes[m_currentScene].nodes[0];
  m_model.nodes[rootID].children.push_back(newNodeIndex);  // Add the camera node to the root

  // Set the camera to look at the scene
  nvutils::Bbox bbox   = getSceneBounds();
  glm::vec3     center = bbox.center();
  glm::vec3 eye = center + glm::vec3(0, 0, bbox.radius() * 2.414f);  //2.414 units away from the center of the sphere to fit it within a 45 - degree FOV
  glm::vec3 up                    = glm::vec3(0, 1, 0);
  tcamera.type                    = "perspective";
  tcamera.name                    = "Camera";
  tcamera.perspective.aspectRatio = 16.0f / 9.0f;
  tcamera.perspective.yfov        = glm::radians(45.0f);
  tcamera.perspective.zfar        = bbox.radius() * 10.0f;
  tcamera.perspective.znear       = bbox.radius() * 0.1f;

  // Add extra information to the node/camera
  tinygltf::Value::Object extras;
  extras["camera::eye"]    = tinygltf::utils::convertToTinygltfValue(3, glm::value_ptr(eye));
  extras["camera::center"] = tinygltf::utils::convertToTinygltfValue(3, glm::value_ptr(center));
  extras["camera::up"]     = tinygltf::utils::convertToTinygltfValue(3, glm::value_ptr(up));
  tnode.extras             = tinygltf::Value(extras);

  // Set the node transformation
  tnode.translation = {eye.x, eye.y, eye.z};
  glm::quat q       = glm::quatLookAt(glm::normalize(center - eye), up);
  tnode.rotation    = {q.x, q.y, q.z, q.w};
}

// This function will update the matrices and the materials of the render nodes
void nvvkgltf::Scene::updateRenderNodes()
{
  const tinygltf::Scene& scene = m_model.scenes[m_currentScene];
  assert(scene.nodes.size() > 0 && "No nodes in the glTF file");
  //assert(scene.nodes.size() == 1 && "Only one top node per scene is supported");
  assert(m_sceneRootNode > -1 && "No root node in the scene");

  m_nodesWorldMatrices.resize(m_model.nodes.size());

  uint32_t renderNodeID = 0;  // Index of the render node
  for(auto& sceneNode : scene.nodes)
  {
    tinygltf::utils::traverseSceneGraph(
        m_model, sceneNode, glm::mat4(1),  //
        nullptr,                           // Camera fct
        // Dealing with lights
        [&](int nodeID, const glm::mat4& mat) {
          tinygltf::Node& tnode             = m_model.nodes[nodeID];
          m_lights[tnode.light].worldMatrix = mat;
          return false;  // Continue traversal
        },
        // Dealing with Nodes and Variant Materials
        [&](int nodeID, const glm::mat4& mat) {
          tinygltf::Node&       tnode = m_model.nodes[nodeID];
          const tinygltf::Mesh& mesh  = m_model.meshes[tnode.mesh];
          for(size_t j = 0; j < mesh.primitives.size(); j++)
          {
            const tinygltf::Primitive& primitive  = mesh.primitives[j];
            nvvkgltf::RenderNode&      renderNode = m_renderNodes[renderNodeID];
            renderNode.worldMatrix                = mat;
            renderNode.materialID                 = getMaterialVariantIndex(primitive, m_currentVariant);
            renderNodeID++;
          }
          return false;  // Continue traversal
        },
        [&](int nodeID, const glm::mat4& mat) {
          m_nodesWorldMatrices[nodeID] = mat;
          return false;
        });
  }

  // Update the visibility of the render nodes
  renderNodeID = 0;
  for(const int sceneNode : m_model.scenes[m_currentScene].nodes)
  {
    KHR_node_visibility nvisible = tinygltf::utils::getNodeVisibility(m_model.nodes[sceneNode]);
    updateVisibility(sceneNode, nvisible.visible, renderNodeID);
  }
}

void nvvkgltf::Scene::setCurrentVariant(int variant)
{
  m_currentVariant = variant;
  // Updating the render nodes with the new material variant
  updateRenderNodes();
}


void nvvkgltf::Scene::clearParsedData()
{
  m_cameras.clear();
  m_lights.clear();
  m_animations.clear();
  m_renderNodes.clear();
  m_renderPrimitives.clear();
  m_uniquePrimitiveIndex.clear();
  m_variants.clear();
  m_numTriangles    = 0;
  m_sceneBounds     = {};
  m_sceneCameraNode = -1;
  m_sceneRootNode   = -1;
}

void nvvkgltf::Scene::destroy()
{
  clearParsedData();
  m_filename.clear();
  m_model = {};
}


// Get the unique index of a primitive, and add it to the list if it is not already there
int nvvkgltf::Scene::getUniqueRenderPrimitive(tinygltf::Primitive& primitive, int meshID)
{
  const std::string& key = tinygltf::utils::generatePrimitiveKey(primitive);

  // Attempt to insert the key with the next available index if it doesn't exist
  auto [it, inserted] = m_uniquePrimitiveIndex.try_emplace(key, static_cast<int>(m_uniquePrimitiveIndex.size()));

  // If the primitive was newly inserted, add it to the render primitives list
  if(inserted)
  {
    nvvkgltf::RenderPrimitive renderPrim;
    renderPrim.pPrimitive  = &primitive;
    renderPrim.vertexCount = int(tinygltf::utils::getVertexCount(m_model, primitive));
    renderPrim.indexCount  = int(tinygltf::utils::getIndexCount(m_model, primitive));
    renderPrim.meshID      = meshID;
    m_renderPrimitives.push_back(renderPrim);
  }

  return it->second;
}


// Function to extract eye, center, and up vectors from a view matrix
inline void extractCameraVectors(const glm::mat4& viewMatrix, const glm::vec3& sceneCenter, glm::vec3& eye, glm::vec3& center, glm::vec3& up)
{
  eye                    = glm::vec3(viewMatrix[3]);
  glm::mat3 rotationPart = glm::mat3(viewMatrix);
  glm::vec3 forward      = -rotationPart * glm::vec3(0.0f, 0.0f, 1.0f);

  // Project sceneCenter onto the forward vector
  glm::vec3 eyeToSceneCenter = sceneCenter - eye;
  float     projectionLength = std::abs(glm::dot(eyeToSceneCenter, forward));
  center                     = eye + projectionLength * forward;

  up = glm::vec3(0.0f, 1.0f, 0.0f);  // Assume the up vector is always (0, 1, 0)
}


// Retrieve the list of render cameras in the scene.
// This function returns a vector of render cameras present in the scene. If the `force`
// parameter is set to true, it clears and regenerates the list of cameras.
//
// Parameters:
// - force: If true, forces the regeneration of the camera list.
//
// Returns:
// - A const reference to the vector of render cameras.
const std::vector<nvvkgltf::RenderCamera>& nvvkgltf::Scene::getRenderCameras(bool force /*= false*/)
{
  if(force)
  {
    m_cameras.clear();
  }

  if(m_cameras.empty())
  {
    assert(m_sceneRootNode > -1 && "No root node in the scene");
    tinygltf::utils::traverseSceneGraph(m_model, m_sceneRootNode, glm::mat4(1), [&](int nodeID, const glm::mat4& worldMatrix) {
      return handleCameraTraversal(nodeID, worldMatrix);
    });
  }
  return m_cameras;
}


bool nvvkgltf::Scene::handleCameraTraversal(int nodeID, const glm::mat4& worldMatrix)
{
  tinygltf::Node& node = m_model.nodes[nodeID];
  m_sceneCameraNode    = nodeID;

  tinygltf::Camera&      tcam = m_model.cameras[node.camera];
  nvvkgltf::RenderCamera camera;
  if(tcam.type == "perspective")
  {
    camera.type  = nvvkgltf::RenderCamera::CameraType::ePerspective;
    camera.znear = tcam.perspective.znear;
    camera.zfar  = tcam.perspective.zfar;
    camera.yfov  = tcam.perspective.yfov;
  }
  else
  {
    camera.type  = nvvkgltf::RenderCamera::CameraType::eOrthographic;
    camera.znear = tcam.orthographic.znear;
    camera.zfar  = tcam.orthographic.zfar;
    camera.xmag  = tcam.orthographic.xmag;
    camera.ymag  = tcam.orthographic.ymag;
  }

  nvutils::Bbox bbox = getSceneBounds();

  // Validate zFar
  if(camera.zfar <= camera.znear)
  {
    camera.zfar = std::max(camera.znear * 2.0, 4.0 * bbox.radius());
    LOGW("glTF: Camera zFar is less than zNear, max(zNear * 2, 4 * bbos.radius()\n");
  }

  // From the view matrix, we extract the eye, center, and up vectors
  extractCameraVectors(worldMatrix, bbox.center(), camera.eye, camera.center, camera.up);

  // If the node/camera has extras, we extract the eye, center, and up vectors from the extras
  auto& extras = node.extras;
  if(extras.IsObject())
  {
    tinygltf::utils::getArrayValue(extras, "camera::eye", camera.eye);
    tinygltf::utils::getArrayValue(extras, "camera::center", camera.center);
    tinygltf::utils::getArrayValue(extras, "camera::up", camera.up);
  }

  m_cameras.push_back(camera);
  return false;
}

bool nvvkgltf::Scene::handleLightTraversal(int nodeID, const glm::mat4& worldMatrix)
{
  tinygltf::Node&       node = m_model.nodes[nodeID];
  nvvkgltf::RenderLight renderLight;
  renderLight.light      = node.light;
  tinygltf::Light& light = m_model.lights[node.light];
  // Add a default color if the light has no color
  if(light.color.empty())
  {
    light.color = {1.0f, 1.0f, 1.0f};
  }
  // Add a default radius if the light has no radius
  if(!light.extras.Has("radius"))
  {
    if(!light.extras.IsObject())
    {  // Avoid overwriting other extras
      light.extras = tinygltf::Value(tinygltf::Value::Object());
    }
    tinygltf::Value::Object extras = light.extras.Get<tinygltf::Value::Object>();
    extras["radius"]               = tinygltf::Value(0.);
    light.extras                   = tinygltf::Value(extras);
  }
  renderLight.worldMatrix = worldMatrix;

  m_lights.push_back(renderLight);
  return false;  // Continue traversal
}


// Return the bounding volume of the scene
nvutils::Bbox nvvkgltf::Scene::getSceneBounds()
{
  if(!m_sceneBounds.isEmpty())
    return m_sceneBounds;

  for(const nvvkgltf::RenderNode& rnode : m_renderNodes)
  {
    glm::vec3 minValues = {0.f, 0.f, 0.f};
    glm::vec3 maxValues = {0.f, 0.f, 0.f};

    const nvvkgltf::RenderPrimitive& rprim    = m_renderPrimitives[rnode.renderPrimID];
    const tinygltf::Accessor&        accessor = m_model.accessors[rprim.pPrimitive->attributes.at("POSITION")];
    if(!accessor.minValues.empty())
      minValues = glm::vec3(accessor.minValues[0], accessor.minValues[1], accessor.minValues[2]);
    if(!accessor.maxValues.empty())
      maxValues = glm::vec3(accessor.maxValues[0], accessor.maxValues[1], accessor.maxValues[2]);
    nvutils::Bbox bbox(minValues, maxValues);
    bbox = bbox.transform(rnode.worldMatrix);
    m_sceneBounds.insert(bbox);
  }

  if(m_sceneBounds.isEmpty() || !m_sceneBounds.isVolume())
  {
    LOGW("glTF: Scene bounding box invalid, Setting to: [-1,-1,-1], [1,1,1]\n");
    m_sceneBounds.insert({-1.0f, -1.0f, -1.0f});
    m_sceneBounds.insert({1.0f, 1.0f, 1.0f});
  }

  return m_sceneBounds;
}

// Handles the creation of render nodes for a given primitive in the scene.
// For each primitive in the node's mesh, it:
// - Generates a unique render primitive index.
// - Creates a render node with the appropriate world matrix, material ID, render primitive ID, primitive ID, and reference node ID.
// If the primitive has the EXT_mesh_gpu_instancing extension, multiple render nodes are created for instancing.
// Otherwise, a single render node is added to the render nodes list.
// Returns false to continue traversal of the scene graph.
bool nvvkgltf::Scene::handleRenderNode(int nodeID, glm::mat4 worldMatrix)
{
  const tinygltf::Node& node = m_model.nodes[nodeID];
  tinygltf::Mesh&       mesh = m_model.meshes[node.mesh];
  for(size_t primID = 0; primID < mesh.primitives.size(); primID++)
  {
    tinygltf::Primitive& primitive    = mesh.primitives[primID];
    int                  rprimID      = getUniqueRenderPrimitive(primitive, node.mesh);
    int                  numTriangles = m_renderPrimitives[rprimID].indexCount / 3;

    nvvkgltf::RenderNode renderNode;
    renderNode.worldMatrix  = worldMatrix;
    renderNode.materialID   = getMaterialVariantIndex(primitive, m_currentVariant);
    renderNode.renderPrimID = rprimID;
    renderNode.refNodeID    = nodeID;
    renderNode.skinID       = node.skin;

    if(tinygltf::utils::hasElementName(node.extensions, EXT_MESH_GPU_INSTANCING_EXTENSION_NAME))
    {
      const tinygltf::Value& ext = tinygltf::utils::getElementValue(node.extensions, EXT_MESH_GPU_INSTANCING_EXTENSION_NAME);
      const tinygltf::Value& attributes   = ext.Get("attributes");
      size_t                 numInstances = handleGpuInstancing(attributes, renderNode, worldMatrix);
      m_numTriangles += numTriangles * static_cast<uint32_t>(numInstances);  // Statistics
    }
    else
    {
      m_renderNodes.push_back(renderNode);
      m_numTriangles += numTriangles;  // Statistics
    }
  }
  return false;  // Continue traversal
}

// Handle GPU instancing : EXT_mesh_gpu_instancing
size_t nvvkgltf::Scene::handleGpuInstancing(const tinygltf::Value& attributes, nvvkgltf::RenderNode renderNode, glm::mat4 worldMatrix)
{
  std::vector<glm::vec3> tStorage;
  std::vector<glm::quat> rStorage;
  std::vector<glm::vec3> sStorage;
  std::span<glm::vec3> translations = tinygltf::utils::getAttributeData3(m_model, attributes, "TRANSLATION", &tStorage);
  std::span<glm::quat> rotations    = tinygltf::utils::getAttributeData3(m_model, attributes, "ROTATION", &rStorage);
  std::span<glm::vec3> scales       = tinygltf::utils::getAttributeData3(m_model, attributes, "SCALE", &sStorage);

  size_t numInstances = std::max({translations.size(), rotations.size(), scales.size()});

  // Note: the specification says, that the number of elements in the attributes should be the same if they are present
  for(size_t i = 0; i < numInstances; i++)
  {
    nvvkgltf::RenderNode instNode    = renderNode;
    glm::vec3            translation = glm::vec3(0.0f, 0.0f, 0.0f);
    glm::quat            rotation    = glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
    glm::vec3            scale       = glm::vec3(1.0f, 1.0f, 1.0f);
    if(!translations.empty())
      translation = translations[i];
    if(!rotations.empty())
      rotation = rotations[i];
    if(!scales.empty())
      scale = scales[i];

    glm::mat4 mat = glm::translate(glm::mat4(1.0f), translation) * glm::mat4_cast(rotation) * glm::scale(glm::mat4(1.0f), scale);

    instNode.worldMatrix = worldMatrix * mat;
    m_renderNodes.push_back(instNode);
  }
  return numInstances;
}

//-------------------------------------------------------------------------------------------------
// Add tangents on primitives that have normal maps but no tangents
void nvvkgltf::Scene::createMissingTangents()
{
  std::vector<int> missTangentPrimitives;

  for(const auto& renderNode : m_renderNodes)
  {
    // Check for missing tangents if the primitive has normalmap
    if(m_model.materials[renderNode.materialID].normalTexture.index >= 0)
    {
      int                  renderPrimID = renderNode.renderPrimID;
      tinygltf::Primitive& primitive    = *m_renderPrimitives[renderPrimID].pPrimitive;

      if(primitive.attributes.find("TANGENT") == primitive.attributes.end())
      {
        LOGW("Render Primitive %d has a normal map but no tangents. Generating tangents.\n", renderPrimID);
        tinygltf::utils::createTangentAttribute(m_model, primitive);
        missTangentPrimitives.push_back(renderPrimID);  // Will generate the tangents later
      }
    }
  }

  // Generate the tangents in parallel
  nvutils::parallel_batches<1>(missTangentPrimitives.size(), [&](uint64_t primID) {
    tinygltf::Primitive& primitive = *m_renderPrimitives[missTangentPrimitives[primID]].pPrimitive;
    tinygltf::utils::simpleCreateTangents(m_model, primitive);
  });
}


//-------------------------------------------------------------------------------------------------
// Find which nodes are solid or translucent, helps for raster rendering
//
std::vector<uint32_t> nvvkgltf::Scene::getShadedNodes(PipelineType type)
{
  std::vector<uint32_t> result;

  for(uint32_t i = 0; i < m_renderNodes.size(); i++)
  {
    const auto& tmat               = m_model.materials[m_renderNodes[i].materialID];
    float       transmissionFactor = 0;
    if(tinygltf::utils::hasElementName(tmat.extensions, KHR_MATERIALS_TRANSMISSION_EXTENSION_NAME))
    {
      const auto& ext = tinygltf::utils::getElementValue(tmat.extensions, KHR_MATERIALS_TRANSMISSION_EXTENSION_NAME);
      tinygltf::utils::getValue(ext, "transmissionFactor", transmissionFactor);
    }
    switch(type)
    {
      case eRasterSolid:
        if(tmat.alphaMode == "OPAQUE" && !tmat.doubleSided && (transmissionFactor == 0.0F))
          result.push_back(i);
        break;
      case eRasterSolidDoubleSided:
        if(tmat.alphaMode == "OPAQUE" && tmat.doubleSided)
          result.push_back(i);
        break;
      case eRasterBlend:
        if(tmat.alphaMode != "OPAQUE" || (transmissionFactor != 0))
          result.push_back(i);
        break;
      case eRasterAll:
        result.push_back(i);
        break;
    }
  }
  return result;
}

tinygltf::Node nvvkgltf::Scene::getSceneRootNode() const
{
  const tinygltf::Scene& scene = m_model.scenes[m_currentScene];
  assert(scene.nodes.size() == 1 && "There should be exactly one node under the scene.");
  const tinygltf::Node& node = m_model.nodes[scene.nodes[0]];  // Root node

  return node;
}

void nvvkgltf::Scene::setSceneRootNode(const tinygltf::Node& node)
{
  const tinygltf::Scene& scene = m_model.scenes[m_currentScene];
  assert(scene.nodes.size() == 1 && "There should be exactly one node under the scene.");
  tinygltf::Node& rootNode = m_model.nodes[scene.nodes[0]];  // Root node
  rootNode                 = node;

  updateRenderNodes();
}

void nvvkgltf::Scene::setSceneCamera(const nvvkgltf::RenderCamera& camera)
{
  assert(m_sceneCameraNode != -1 && "No camera node found in the scene");

  // Set the tinygltf::Node
  tinygltf::Node& tnode = m_model.nodes[m_sceneCameraNode];
  glm::quat       q     = glm::quatLookAt(glm::normalize(camera.center - camera.eye), camera.up);
  tnode.translation     = {camera.eye.x, camera.eye.y, camera.eye.z};
  tnode.rotation        = {q.x, q.y, q.z, q.w};

  // Set the tinygltf::Camera
  tinygltf::Camera& tcamera = m_model.cameras[tnode.camera];
  tcamera.type              = "perspective";
  tcamera.perspective.znear = camera.znear;
  tcamera.perspective.zfar  = camera.zfar;
  tcamera.perspective.yfov  = camera.yfov;

  // Add extras to the tinygltf::Camera, to store the eye, center, and up vectors
  tinygltf::Value::Object extras;
  extras["camera::eye"]    = tinygltf::utils::convertToTinygltfValue(3, glm::value_ptr(camera.eye));
  extras["camera::center"] = tinygltf::utils::convertToTinygltfValue(3, glm::value_ptr(camera.center));
  extras["camera::up"]     = tinygltf::utils::convertToTinygltfValue(3, glm::value_ptr(camera.up));
  tnode.extras             = tinygltf::Value(extras);
}

// Collects all animation data
void nvvkgltf::Scene::parseAnimations()
{
  m_animations.clear();
  m_animations.reserve(m_model.animations.size());
  for(tinygltf::Animation& anim : m_model.animations)
  {
    Animation animation;
    animation.info.name = anim.name;
    if(animation.info.name.empty())
    {
      animation.info.name = "Animation" + std::to_string(m_animations.size());
    }

    // Samplers
    for(auto& samp : anim.samplers)
    {
      AnimationSampler sampler;

      if(samp.interpolation == "LINEAR")
      {
        sampler.interpolation = AnimationSampler::InterpolationType::eLinear;
      }
      if(samp.interpolation == "STEP")
      {
        sampler.interpolation = AnimationSampler::InterpolationType::eStep;
      }
      if(samp.interpolation == "CUBICSPLINE")
      {
        sampler.interpolation = AnimationSampler::InterpolationType::eCubicSpline;
      }

      // Read sampler input time values
      {
        const tinygltf::Accessor& accessor = m_model.accessors[samp.input];
        if(!tinygltf::utils::copyAccessorData(m_model, accessor, sampler.inputs))
        {
          LOGE("Invalid data type for animation input");
          continue;
        }

        // Protect against invalid values
        for(auto input : sampler.inputs)
        {
          if(input < animation.info.start)
          {
            animation.info.start = input;
          }
          if(input > animation.info.end)
          {
            animation.info.end = input;
          }
        }
      }

      // Read sampler output T/R/S values
      {
        const tinygltf::Accessor& accessor = m_model.accessors[samp.output];

        switch(accessor.type)
        {
          case TINYGLTF_TYPE_VEC3: {
            if(accessor.bufferView > -1)
            {
              tinygltf::utils::copyAccessorData(m_model, accessor, sampler.outputsVec3);
            }
            else
              sampler.outputsVec3.resize(accessor.count);
            break;
          }
          case TINYGLTF_TYPE_VEC4: {
            if(accessor.bufferView > -1)
            {
              tinygltf::utils::copyAccessorData(m_model, accessor, sampler.outputsVec4);
            }
            else
              sampler.outputsVec4.resize(accessor.count);
            break;
          }
          case TINYGLTF_TYPE_SCALAR: {
            // This is for `sampler.inputs` vectors of `n` elements
            sampler.outputsFloat.resize(sampler.inputs.size());
            const size_t           elemPerKey = accessor.count / sampler.inputs.size();
            std::vector<float>     storage;
            std::span<const float> val     = tinygltf::utils::getAccessorData(m_model, accessor, &storage);
            const float*           dataPtr = val.data();

            for(size_t i = 0; i < sampler.inputs.size(); i++)
            {
              for(int j = 0; j < elemPerKey; j++)
              {
                sampler.outputsFloat[i].push_back(*dataPtr++);
              }
            }
            break;
          }
          default: {
            LOGW("Unknown animation type: %d\n", accessor.type);
            break;
          }
        }
      }

      animation.samplers.emplace_back(sampler);
    }

    // Channels
    for(auto& source : anim.channels)
    {
      AnimationChannel channel;

      if(source.target_path == "rotation")
      {
        channel.path = AnimationChannel::PathType::eRotation;
      }
      else if(source.target_path == "translation")
      {
        channel.path = AnimationChannel::PathType::eTranslation;
      }
      else if(source.target_path == "scale")
      {
        channel.path = AnimationChannel::PathType::eScale;
      }
      else if(source.target_path == "weights")
      {
        channel.path = AnimationChannel::PathType::eWeights;
      }
      else if(source.target_path == "pointer")
      {
        channel.path = AnimationChannel::PathType::ePointer;
      }
      channel.samplerIndex = source.sampler;
      channel.node         = source.target_node;

      animation.channels.emplace_back(channel);
    }

    animation.info.reset();
    m_animations.emplace_back(animation);
  }

  // Find all animated primitives (morph)
  m_morphPrimitives.clear();
  for(size_t renderPrimID = 0; renderPrimID < getRenderPrimitives().size(); renderPrimID++)
  {
    const auto&                renderPrimitive = getRenderPrimitive(renderPrimID);
    const tinygltf::Primitive& primitive       = *renderPrimitive.pPrimitive;
    const tinygltf::Mesh&      mesh            = getModel().meshes[renderPrimitive.meshID];

    if(!primitive.targets.empty() && !mesh.weights.empty())
    {
      m_morphPrimitives.push_back(uint32_t(renderPrimID));
    }
  }
  // Skin animated
  m_skinNodes.clear();
  for(size_t renderNodeID = 0; renderNodeID < m_renderNodes.size(); renderNodeID++)
  {
    if(m_renderNodes[renderNodeID].skinID > -1)
    {
      m_skinNodes.push_back(uint32_t(renderNodeID));
    }
  }
}

//--------------------------------------------------------------------------------------------------
// Update the animation (index)
// The value of the animation is updated based on the current time
// - Node transformations are updated
// - Morph target weights are updated
bool nvvkgltf::Scene::updateAnimation(uint32_t animationIndex)
{
  bool       animated  = false;
  Animation& animation = m_animations[animationIndex];
  float      time      = animation.info.currentTime;

  for(auto& channel : animation.channels)
  {
    if(channel.node < 0 || channel.node >= m_model.nodes.size())  // Invalid node
      continue;

    tinygltf::Node&   gltfNode = m_model.nodes[channel.node];
    AnimationSampler& sampler  = animation.samplers[channel.samplerIndex];

    if(channel.path == AnimationChannel::PathType::ePointer)
    {
      static std::unordered_set<int> warnedAnimations;
      if(warnedAnimations.insert(animationIndex).second)
      {
        LOGE("AnimationChannel::PathType::POINTER not implemented for animation %d", animationIndex);
      }
      continue;
    }

    animated |= processAnimationChannel(gltfNode, sampler, channel, time, animationIndex);
  }

  return animated;
}

//--------------------------------------------------------------------------------------------------
// Process the animation channel
// - Interpolates the keyframes
// - Updates the node transformation
// - Updates the morph target weights
bool nvvkgltf::Scene::processAnimationChannel(tinygltf::Node&         gltfNode,
                                              AnimationSampler&       sampler,
                                              const AnimationChannel& channel,
                                              float                   time,
                                              uint32_t                animationIndex)
{
  std::unordered_set<int> warnedAnimations;

  bool animated = false;

  for(size_t i = 0; i < sampler.inputs.size() - 1; i++)
  {
    float inputStart = sampler.inputs[i];
    float inputEnd   = sampler.inputs[i + 1];

    if(inputStart <= time && time <= inputEnd)
    {
      float t  = calculateInterpolationFactor(inputStart, inputEnd, time);
      animated = true;

      switch(sampler.interpolation)
      {
        case AnimationSampler::InterpolationType::eLinear:
          handleLinearInterpolation(gltfNode, sampler, channel, t, i);
          break;
        case AnimationSampler::InterpolationType::eStep:
          handleStepInterpolation(gltfNode, sampler, channel, i);
          break;
        case AnimationSampler::InterpolationType::eCubicSpline: {
          float keyDelta = inputEnd - inputStart;
          handleCubicSplineInterpolation(gltfNode, sampler, channel, t, keyDelta, i);
          break;
        }
      }
    }
  }

  return animated;
}

//--------------------------------------------------------------------------------------------------
// Calculate the interpolation factor: [0..1] between two keyframes
float nvvkgltf::Scene::calculateInterpolationFactor(float inputStart, float inputEnd, float time)
{
  float keyDelta = inputEnd - inputStart;
  return std::clamp((time - inputStart) / keyDelta, 0.0f, 1.0f);
}

//--------------------------------------------------------------------------------------------------
// Interpolates the keyframes linearly
void nvvkgltf::Scene::handleLinearInterpolation(tinygltf::Node&         gltfNode,
                                                AnimationSampler&       sampler,
                                                const AnimationChannel& channel,
                                                float                   t,
                                                size_t                  index)
{
  switch(channel.path)
  {
    case AnimationChannel::PathType::eRotation: {
      const glm::quat q1 = glm::make_quat(glm::value_ptr(sampler.outputsVec4[index]));
      const glm::quat q2 = glm::make_quat(glm::value_ptr(sampler.outputsVec4[index + 1]));
      glm::quat       q  = glm::normalize(glm::slerp(q1, q2, t));
      gltfNode.rotation  = {q.x, q.y, q.z, q.w};
      break;
    }
    case AnimationChannel::PathType::eTranslation: {
      glm::vec3 trans      = glm::mix(sampler.outputsVec3[index], sampler.outputsVec3[index + 1], t);
      gltfNode.translation = {trans.x, trans.y, trans.z};
      break;
    }
    case AnimationChannel::PathType::eScale: {
      glm::vec3 s    = glm::mix(sampler.outputsVec3[index], sampler.outputsVec3[index + 1], t);
      gltfNode.scale = {s.x, s.y, s.z};
      break;
    }
    case AnimationChannel::PathType::eWeights: {
      {
        // Retrieve the mesh from the node
        if(gltfNode.mesh >= 0)
        {
          tinygltf::Mesh& mesh = m_model.meshes[gltfNode.mesh];

          // Make sure the weights vector is resized to match the number of morph targets
          if(mesh.weights.size() != sampler.outputsFloat[index].size())
          {
            mesh.weights.resize(sampler.outputsFloat[index].size());
          }

          // Interpolating between weights for morph targets
          for(size_t j = 0; j < mesh.weights.size(); j++)
          {
            float weight1   = sampler.outputsFloat[index][j];
            float weight2   = sampler.outputsFloat[index + 1][j];
            mesh.weights[j] = glm::mix(weight1, weight2, t);
          }
        }
        break;
      }
    }
  }
}

//--------------------------------------------------------------------------------------------------
// Interpolates the keyframes with a step interpolation
void nvvkgltf::Scene::handleStepInterpolation(tinygltf::Node& gltfNode, AnimationSampler& sampler, const AnimationChannel& channel, size_t index)
{
  switch(channel.path)
  {
    case AnimationChannel::PathType::eRotation: {
      glm::quat q       = glm::quat(sampler.outputsVec4[index]);
      gltfNode.rotation = {q.x, q.y, q.z, q.w};
      break;
    }
    case AnimationChannel::PathType::eTranslation: {
      glm::vec3 t          = glm::vec3(sampler.outputsVec3[index]);
      gltfNode.translation = {t.x, t.y, t.z};
      break;
    }
    case AnimationChannel::PathType::eScale: {
      glm::vec3 s    = glm::vec3(sampler.outputsVec3[index]);
      gltfNode.scale = {s.x, s.y, s.z};
      break;
    }
  }
}

//--------------------------------------------------------------------------------------------------
// Interpolates the keyframes with a cubic spline interpolation
void nvvkgltf::Scene::handleCubicSplineInterpolation(tinygltf::Node&         gltfNode,
                                                     AnimationSampler&       sampler,
                                                     const AnimationChannel& channel,
                                                     float                   t,
                                                     float                   keyDelta,
                                                     size_t                  index)
{
  // Implements the logic in
  // https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#interpolation-cubic
  // for quaternions (first case) and other values (second case).

  const size_t prevIndex = index * 3;
  const size_t nextIndex = (index + 1) * 3;
  const size_t A         = 0;  // Offset for the in-tangent
  const size_t V         = 1;  // Offset for the value
  const size_t B         = 2;  // Offset for the out-tangent

  const float tSq = t * t;
  const float tCb = tSq * t;
  const float tD  = keyDelta;

  // Compute each of the coefficient terms in the specification
  const float cV1 = -2 * tCb + 3 * tSq;        // -2 t^3 + 3 t^2
  const float cV0 = 1 - cV1;                   //  2 t^3 - 3 t^2 + 1
  const float cA  = tD * (tCb - tSq);          // t_d (t^3 - t^2)
  const float cB  = tD * (tCb - 2 * tSq + t);  // t_d (t^3 - 2 t^2 + t)

  if(channel.path == AnimationChannel::PathType::eRotation)
  {
    const glm::vec4& v0 = sampler.outputsVec4[prevIndex + V];  // v_k
    const glm::vec4& a  = sampler.outputsVec4[nextIndex + A];  // a_{k+1}
    const glm::vec4& b  = sampler.outputsVec4[prevIndex + B];  // b_k
    const glm::vec4& v1 = sampler.outputsVec4[nextIndex + V];  // v_{k+1}

    glm::vec4 result = cV0 * v0 + cB * b + cV1 * v1 + cA * a;

    glm::quat quatResult = glm::make_quat(glm::value_ptr(result));
    quatResult           = glm::normalize(quatResult);
    gltfNode.rotation    = {quatResult.x, quatResult.y, quatResult.z, quatResult.w};
  }
  else
  {
    const glm::vec3& v0 = sampler.outputsVec3[prevIndex + V];  // v_k
    const glm::vec3& a  = sampler.outputsVec3[nextIndex + A];  // a_{k+1}
    const glm::vec3& b  = sampler.outputsVec3[prevIndex + B];  // b_k
    const glm::vec3& v1 = sampler.outputsVec3[nextIndex + V];  // v_{k+1}

    glm::vec3 result = cV0 * v0 + cB * b + cV1 * v1 + cA * a;

    if(channel.path == AnimationChannel::PathType::eTranslation)
    {
      gltfNode.translation = {result.x, result.y, result.z};
    }
    else if(channel.path == AnimationChannel::PathType::eScale)
    {
      gltfNode.scale = {result.x, result.y, result.z};
    }
  }
}

// Parse the variants of the materials
void nvvkgltf::Scene::parseVariants()
{
  if(m_model.extensions.find(KHR_MATERIALS_VARIANTS_EXTENSION_NAME) != m_model.extensions.end())
  {
    const auto& ext = m_model.extensions.find(KHR_MATERIALS_VARIANTS_EXTENSION_NAME)->second;
    if(ext.Has("variants"))
    {
      auto& variants = ext.Get("variants");
      for(size_t i = 0; i < variants.ArrayLen(); i++)
      {
        std::string name = variants.Get(int(i)).Get("name").Get<std::string>();
        m_variants.emplace_back(name);
      }
    }
  }
}

// Return the material index based on the variant, or the material set on the primitive
int nvvkgltf::Scene::getMaterialVariantIndex(const tinygltf::Primitive& primitive, int currentVariant)
{
  if(primitive.extensions.find(KHR_MATERIALS_VARIANTS_EXTENSION_NAME) != primitive.extensions.end())
  {
    const auto& ext     = primitive.extensions.find(KHR_MATERIALS_VARIANTS_EXTENSION_NAME)->second;
    auto&       mapping = ext.Get("mappings");
    for(auto& map : mapping.Get<tinygltf::Value::Array>())
    {
      auto& variants   = map.Get("variants");
      int   materialID = map.Get("material").Get<int>();
      for(auto& variant : variants.Get<tinygltf::Value::Array>())
      {
        int variantID = variant.Get<int>();
        if(variantID == currentVariant)
          return materialID;
      }
    }
  }

  return std::max(0, primitive.material);
}
