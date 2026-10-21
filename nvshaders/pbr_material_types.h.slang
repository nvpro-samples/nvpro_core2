/*
 * Copyright (c) 2022-2026, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef NVVKHL_PBR_MAT_STRUCT_H
#define NVVKHL_PBR_MAT_STRUCT_H 1

#include "functions.h.slang"

/* 
Evaluated PBR material containing all properties needed for physically based shading and sampling.
This is the result of evaluating a glTF material with its textures and parameters, ready to be used
by the rendering/ray tracing system for accurate material response simulation.
*/

struct PbrMaterial
{
  float3 baseColor;  // base color
  float  opacity;    // 1 = opaque, 0 = fully transparent
  float2 roughness;  // 0 = smooth, 1 = rough (anisotropic: x = U, y = V)
  float  metallic;   // 0 = dielectric, 1 = metallic
  float3 emissive;   // emissive color
  float  occlusion;  // occlusion factor

  float3 N;   // shading normal
  float3 T;   // tangent
  float3 B;   // bitangent
  float3 Ng;  // geometric normal


  float ior1;        // index of refraction : current medium (i.e. air)
  float ior2;        // index of refraction : the other side (i.e. glass)
  float dispersion;  // KHR_materials_specular

  float  specular;       // weight of the dielectric specular layer
  float3 specularColor;  // color of the dielectric specular layer
  float  transmission;   // KHR_materials_transmission
  // Extra tint on the specular transmission (BTDF) lobe only; the lobe is tinted by
  // `baseColor * transmissionTint`. float3(1) is plain glTF behavior. It exists because the
  // thin-walled mode of KHR_materials_scatter needs the transmission lobe and the diffuse
  // reflection lobe to carry different colors, and `baseColor` alone can only express one.
  float3 transmissionTint;

  float3 attenuationColor;     // KHR_materials_volume
  float  attenuationDistance;  //
  float  thickness;            // KHR_materials_volume thicknessFactor * thicknessTexture (object-space)

  float3 scatterCoefficient;  // KHR_materials_scatter: σ_s (scattering coefficient)
  float  scatterAnisotropy;   // KHR_materials_scatter: g parameter for Henyey-Greenstein

  float  clearcoat;           // KHR_materials_clearcoat
  float  clearcoatRoughness;  //
  float3 Nc;                  // clearcoat normal

  float iridescence;  // KHR_materials_iridescence
  float iridescenceIor;
  float iridescenceThickness;

  float3 sheenColor;  // KHR_materials_sheen
  float  sheenRoughness;

  float  diffuseTransmissionFactor;  // KHR_materials_diffuse_transmission
  float3 diffuseTransmissionColorFactor;

  float retroreflection;  // KHR_materials_retroreflection [0,1] MRM blend
};

/* 
Default PBR material with all properties set to default values.
This function initializes a PbrMaterial structure with typical values for a diffuse material.
*/
PbrMaterial defaultPbrMaterial()
{
  PbrMaterial mat;
  mat.baseColor = float3(1.0F);
  mat.opacity   = 1.0F;
  mat.roughness = float2(1.0F);
  mat.metallic  = 1.0F;
  mat.emissive  = float3(0.0F);

  mat.N  = float3(0.0F, 0.0F, 1.0F);
  mat.Ng = float3(0.0F, 0.0F, 1.0F);
  mat.T  = float3(1.0F, 0.0F, 0.0F);
  mat.B  = float3(0.0F, 1.0F, 0.0F);

  mat.ior1       = 1.0F;
  mat.ior2       = 1.5F;
  mat.dispersion = 0.0F;

  mat.specular         = 1.0F;
  mat.specularColor    = float3(1.0F);
  mat.transmission     = 0.0F;
  mat.transmissionTint = float3(1.0F);

  mat.attenuationColor    = float3(1.0F);
  mat.attenuationDistance = 1.0F;
  mat.thickness           = 0.0F;

  mat.scatterCoefficient = float3(0.0F);
  mat.scatterAnisotropy  = 0.0F;

  mat.clearcoat          = 0.0F;
  mat.clearcoatRoughness = 0.01F;
  mat.Nc                 = float3(0.0F, 0.0F, 1.0F);

  mat.iridescence          = 0.0F;
  mat.iridescenceIor       = 1.5F;
  mat.iridescenceThickness = 0.1F;

  mat.sheenColor     = float3(0.0F);
  mat.sheenRoughness = 0.0F;
  mat.occlusion      = 1.0F;

  mat.diffuseTransmissionFactor      = 0.0F;
  mat.diffuseTransmissionColorFactor = float3(1.0F);

  mat.retroreflection = 0.0F;

  return mat;
}

/* 
Default PBR material with all properties set to default values.
This function initializes a PbrMaterial structure with typical values for a diffuse material.
*/
PbrMaterial defaultPbrMaterial(float3 baseColor, float metallic, float roughness, float3 N, float3 Ng)
{
  PbrMaterial mat = defaultPbrMaterial();
  mat.baseColor   = baseColor;
  mat.metallic    = metallic;
  mat.roughness   = float2(roughness * roughness);
  mat.Ng          = Ng;
  mat.N           = N;
  orthonormalBasis(mat.N, mat.T, mat.B);

  return mat;
}

/*
Volumetric extinction coefficient `sigma_t = -log(c) / d` of a KHR_materials_volume medium, from
its attenuation color `c` and distance `d` (scene units). The one definition of sigma_t: every
consumer -- Beer's law, free-flight sampling, the scatter split -- must call it, so they all see
the same medium. The PbrMaterial overload in bsdf_functions.h.slang documents its use.

`c` is floored so a black channel stays finite (log(0) is -inf). `d` is only kept off zero: any
positive distance is valid and must be honored, however dense the medium it describes.
*/
float3 volumeExtinctionCoefficient(float3 attenuationColor, float attenuationDistance)
{
  return -log(max(attenuationColor, float3(0.001))) / max(attenuationDistance, 1e-6);
}
#endif  // NVVKHL_PBR_MAT_STRUCT_H
