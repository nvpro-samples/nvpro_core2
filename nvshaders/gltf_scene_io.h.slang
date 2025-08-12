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

#ifndef DH_SCN_DESC_H
#define DH_SCN_DESC_H 1

#include "slang_types.h"

NAMESPACE_SHADERIO_BEGIN()
/*-------------------------------------------------------------------------------------------------
Common structures used to store glTF scenes in GPU buffers.
-------------------------------------------------------------------------------------------------*/

// This is the GLTF Node structure, but flattened
struct GltfRenderNode
{
  float4x4 objectToWorld;
  float4x4 worldToObject;
  int      materialID;
  int      renderPrimID;
};

// This is all the information about a vertex buffer
struct VertexBuffers
{
  float3* positions;
  float3* normals;
  uint*   colors;  // packed as 4 uint8_t
  float4* tangents;
  float2* texCoords0;
  float2* texCoords1;
};

// This is the GLTF Primitive structure
struct GltfRenderPrimitive
{
  uint3*        indices;
  VertexBuffers vertexBuffer;
};


/*-------------------------------------------------------------------------------------------------
Common structures used for lights other than environment lighting.
-------------------------------------------------------------------------------------------------*/

// Light type
enum LightType
{
  eLightTypeNone,
  eLightTypeDirectional,
  eLightTypeSpot,
  eLightTypePoint,
};


// Light
struct GltfLight
{
  float3 direction;  // Direction of the light
  int    type;       // Type of the light

  float3 position;  // Position of the light
  float  radius;    // Radius of the light

  float3 color;      // Color of the light
  float  intensity;  // illuminance (lm/m2) for directional lights, luminous intensity (lm/sr) for positional lights

  float angularSizeOrInvRange;  // angular size for directional lights, 1/range for spot and point lights
  float innerAngle;             // Inner angle of the light
  float outerAngle;             // Outer angle of the light
  int   _pad;                   // Padding
};


// alphaMode
enum AlphaMode
{
  eAlphaModeOpaque,
  eAlphaModeMask,
  eAlphaModeBlend,
};

// Texture info
struct GltfTextureInfo
{
  float3x2 uvTransform = float3x2(1);  // 24 bytes (3x2 matrix)
  int      index       = -1;           // 4 bytes
  int      texCoord    = 0;            // 4 bytes
};  // Total: 32 bytes

struct GltfShadeMaterial
{
  float4 pbrBaseColorFactor;  // offset 0    - 16 bytes
  float3 emissiveFactor;      // offset 16   - 12 bytes
  float  normalTextureScale;  // offset 28   - 4 bytes

  float pbrRoughnessFactor;  // offset 32   - 4 bytes
  float pbrMetallicFactor;   // offset 36   - 4 bytes
  int   alphaMode;           // offset 40   - 4 bytes
  float alphaCutoff;         // offset 44   - 4 bytes

  float3 attenuationColor;  // offset 48   - 12 bytes
  float  ior;               // offset 60   - 4 bytes

  float transmissionFactor;   // offset 64   - 4 bytes
  float thicknessFactor;      // offset 68   - 4 bytes
  float attenuationDistance;  // offset 72   - 4 bytes
  float clearcoatFactor;      // offset 76   - 4 bytes

  float3 specularColorFactor;  // offset 80   - 12 bytes
  float  clearcoatRoughness;   // offset 92   - 4 bytes

  float specularFactor;               // offset 96   - 4 bytes
  int   unlit;                        // offset 100  - 4 bytes
  float iridescenceFactor;            // offset 104  - 4 bytes
  float iridescenceThicknessMaximum;  // offset 108  - 4 bytes

  float  iridescenceThicknessMinimum;  // offset 112  - 4 bytes
  float  iridescenceIor;               // offset 116  - 4 bytes
  float2 anisotropyRotation;           // offset 120  - 8 bytes

  float3 sheenColorFactor;    // offset 128  - 12 bytes
  float  anisotropyStrength;  // offset 140  - 4 bytes

  float sheenRoughnessFactor;      // offset 144  - 4 bytes
  float occlusionStrength;         // offset 148  - 4 bytes
  float dispersion;                // offset 152  - 4 bytes
  int   usePbrSpecularGlossiness;  // offset 156  - 4 bytes

  float4 pbrDiffuseFactor;     // offset 160  - 16 bytes
  float3 pbrSpecularFactor;    // offset 176  - 12 bytes
  float  pbrGlossinessFactor;  // offset 188  - 4 bytes

  float3 diffuseTransmissionColor;   // offset 192  - 12 bytes
  float  diffuseTransmissionFactor;  // offset 204  - 4 bytes

  int doubleSided;  // offset 208  - 4 bytes
  // Texture infos (uint16_t, 2 bytes each)
  uint16_t pbrBaseColorTexture;              // offset 212  - 2 bytes
  uint16_t normalTexture;                    // offset 214  - 2 bytes
  uint16_t pbrMetallicRoughnessTexture;      // offset 216  - 2 bytes
  uint16_t emissiveTexture;                  // offset 218  - 2 bytes
  uint16_t transmissionTexture;              // offset 220  - 2 bytes
  uint16_t thicknessTexture;                 // offset 222  - 2 bytes
  uint16_t clearcoatTexture;                 // offset 224  - 2 bytes
  uint16_t clearcoatRoughnessTexture;        // offset 226  - 2 bytes
  uint16_t clearcoatNormalTexture;           // offset 228  - 2 bytes
  uint16_t specularTexture;                  // offset 230  - 2 bytes
  uint16_t specularColorTexture;             // offset 232  - 2 bytes
  uint16_t iridescenceTexture;               // offset 234  - 2 bytes
  uint16_t iridescenceThicknessTexture;      // offset 236  - 2 bytes
  uint16_t anisotropyTexture;                // offset 238  - 2 bytes
  uint16_t sheenColorTexture;                // offset 240  - 2 bytes
  uint16_t sheenRoughnessTexture;            // offset 242  - 2 bytes
  uint16_t occlusionTexture;                 // offset 244  - 2 bytes
  uint16_t pbrDiffuseTexture;                // offset 246  - 2 bytes
  uint16_t pbrSpecularGlossinessTexture;     // offset 248  - 2 bytes
  uint16_t diffuseTransmissionTexture;       // offset 250  - 2 bytes
  uint16_t diffuseTransmissionColorTexture;  // offset 252  - 2 bytes
  uint16_t _pad1;                            // offset 254  - 2 bytes
  // Total size: 256 bytes
};

inline GltfTextureInfo defaultGltfTextureInfo()
{
  GltfTextureInfo t;
  t.uvTransform = float3x2(1);
  t.index       = -1;
  t.texCoord    = 0;
  return t;
}

inline GltfShadeMaterial defaultGltfMaterial()
{
  GltfShadeMaterial m;
  m.pbrBaseColorFactor          = float4(1, 1, 1, 1);
  m.emissiveFactor              = float3(0, 0, 0);
  m.normalTextureScale          = 1;
  m.pbrRoughnessFactor          = 1;
  m.pbrMetallicFactor           = 1;
  m.alphaMode                   = AlphaMode::eAlphaModeOpaque;
  m.alphaCutoff                 = 0.5;
  m.transmissionFactor          = 0;
  m.ior                         = 1.5;
  m.attenuationColor            = float3(1, 1, 1);
  m.thicknessFactor             = 0;
  m.attenuationDistance         = 0;
  m.clearcoatFactor             = 0;
  m.clearcoatRoughness          = 0;
  m.specularFactor              = 0;
  m.specularColorFactor         = float3(1, 1, 1);
  m.unlit                       = 0;
  m.iridescenceFactor           = 0;
  m.iridescenceThicknessMaximum = 100;
  m.iridescenceThicknessMinimum = 400;
  m.iridescenceIor              = 1.3f;
  m.anisotropyStrength          = 0;
  m.anisotropyRotation          = float2(0, 0);
  m.sheenRoughnessFactor        = 0;
  m.sheenColorFactor            = float3(0, 0, 0);
  m.occlusionStrength           = 1;
  m.dispersion                  = 0;
  m.usePbrSpecularGlossiness    = 0;
  m.pbrDiffuseFactor            = float4(1);
  m.pbrSpecularFactor           = float3(1);
  m.pbrGlossinessFactor         = 1;
  m.diffuseTransmissionColor    = float3(1, 1, 1);
  m.diffuseTransmissionFactor   = 0;
  m.doubleSided                 = 0;

  m.pbrBaseColorTexture             = -1;
  m.normalTexture                   = -1;
  m.pbrMetallicRoughnessTexture     = -1;
  m.emissiveTexture                 = -1;
  m.transmissionTexture             = -1;
  m.thicknessTexture                = -1;
  m.clearcoatTexture                = -1;
  m.clearcoatRoughnessTexture       = -1;
  m.clearcoatNormalTexture          = -1;
  m.specularTexture                 = -1;
  m.specularColorTexture            = -1;
  m.iridescenceTexture              = -1;
  m.iridescenceThicknessTexture     = -1;
  m.anisotropyTexture               = -1;
  m.sheenColorTexture               = -1;
  m.sheenRoughnessTexture           = -1;
  m.pbrDiffuseTexture               = -1;
  m.pbrSpecularGlossinessTexture    = -1;
  m.diffuseTransmissionTexture      = -1;
  m.diffuseTransmissionColorTexture = -1;

  return m;
}

// The scene description is a pointer to the material, render node and render primitive
// The buffers are all arrays of the above structures
struct GltfScene
{
  GltfShadeMaterial*   materials;
  GltfTextureInfo*     textureInfos;
  GltfRenderNode*      renderNodes;
  GltfRenderPrimitive* renderPrimitives;
  GltfLight*           lights;
  int                  numLights;  // number of punctual lights
};


#ifdef __cplusplus
inline GltfLight defaultLight()
{
  GltfLight l;
  l.position              = float3 { 5.0F, 5.F, 5.F };
  l.direction             = glm::normalize(float3 { 0.0F, -.7F, -.7F });
  l.type                  = eLightTypeDirectional;
  l.angularSizeOrInvRange = glm::radians(0.53F);
  l.color                 = { 1.0F, 1.0F, 1.0F };
  l.intensity             = 0.F;  // Dark
  l.innerAngle            = glm::radians(10.F);
  l.outerAngle            = glm::radians(30.F);
  l.radius                = 1.0F;

  return l;
}
#endif  //__cplusplus

NAMESPACE_SHADERIO_END()

#endif
