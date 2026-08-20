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

/*-------------------------------------------------------------------------------------------------
This file takes the incoming `GltfShadeMaterial` (material uploaded in a buffer) and
evaluates it, basically sampling the textures, and returns the struct `PbrMaterial`
which is used by the BSDF functions to evaluate and sample the material.
-------------------------------------------------------------------------------------------------*/

#ifndef MAT_EVAL_H
#define MAT_EVAL_H 1

#include "pbr_material_types.h.slang"
#include "gltf_scene_io.h.slang"


/*-------------------------------------------------------------------------------------------------
# `MAT_EVAL_MESH_STATE_MAX_TEXCOORDS` Define
> The number of texture coordinate sets that are stored
> in the MeshState struct

Defaults to two.
-------------------------------------------------------------------------------------------------*/
#ifndef MAT_EVAL_MESH_STATE_MAX_TEXCOORDS
#define MAT_EVAL_MESH_STATE_MAX_TEXCOORDS 2
#endif

struct MeshState
{
  float3 N;   // Normal
  float3 T;   // Tangent
  float3 B;   // Bitangent
  float3 Ng;  // Geometric normal

  float2 tc[MAT_EVAL_MESH_STATE_MAX_TEXCOORDS];  // Texture coordinates

  bool   isInside = false;
  float  texGrad  = 0.0f;  // Ray footprint for SampleGrad; 0 -> Sample() (mip 0 in RT, proper mips in fragment)
};


/*-------------------------------------------------------------------------------------------------
# `MAT_EVAL_TEXTURE_ARRAY` Define
> The name of the array that contains all texture samplers.

Defaults to `texturesMap`; you can define this before including `pbr_mat_eval.h`
to use textures from a different array name.
-------------------------------------------------------------------------------------------------*/
// #ifndef MAT_EVAL_TEXTURE_ARRAY
// #define MAT_EVAL_TEXTURE_ARRAY texturesMap
// #endif

/*-------------------------------------------------------------------------------------------------
# `NO_TEXTURES` Define
> Define this before including `pbr_mat_eval.h` to use a color of `float4(1.0f)`
> for everything instead of reading textures.
-------------------------------------------------------------------------------------------------*/
#ifndef NO_TEXTURES
#define USE_TEXTURES
#endif

/*
texGrad > 0: SampleGrad (path trace ray footprint). texGrad == 0: Sample() — fragment gets real mips;
in ray trace / compute implicit LOD is undefined and effectively samples like mip 0, same as needed there.
*/
float4 getTexture(Sampler2D textures[], in GltfTextureInfo textureInfo, in float2 texCoords[MAT_EVAL_MESH_STATE_MAX_TEXCOORDS], float texGrad)
{
#ifdef USE_TEXTURES
  // KHR_texture_transform
  float2 tt = mul(float3(texCoords[textureInfo.texCoord], 1), textureInfo.uvTransform).xy;
  if(texGrad > 0.0f)
  {
    float2 ddx_uv = textureInfo.uvTransform[0] * texGrad;
    float2 ddy_uv = textureInfo.uvTransform[1] * texGrad;
    return textures[textureInfo.index].SampleGrad(tt, ddx_uv, ddy_uv);
  }
  return textures[textureInfo.index].Sample(tt);
#else
  return float4(1.0F);
#endif
}

/* 
Check if a texture is present.
This function checks if a texture is present based on the given texture info.
It returns true if the texture index is greater than -1, indicating that the texture is present.
*/
bool isTexturePresent(uint16_t tinfo)
{
  return tinfo > 0;
}

/*-------------------------------------------------------------------------------------------------
* Convert multi-scatter albedo to single-scatter albedo (Kulla-Conty mapping).
* Formula from the KHR_materials_scatter specification. The anisotropy g of the Henyey-Greenstein
* phase function enters the denominator (similarity theory), so forward- and backward-scattering
* media get the single-scatter albedo that reproduces the requested multi-scatter albedo.
* -------------------------------------------------------------------------------------------------*/
float3 multiToSingleScatterAlbedo(float3 rho_ms, float g)
{
  float3 s = 4.09712 + 4.20863 * rho_ms - sqrt(9.59217 + 41.6808 * rho_ms + 17.7126 * rho_ms * rho_ms);
  // g in (-1,1) and s in [0,1] keep the denominator strictly positive. Clamp so the scattering
  // coefficient can never exceed the total extinction.
  return clamp((1.0 - s * s) / (1.0 - g * s * s), float3(0.0), float3(1.0));
}

/*-------------------------------------------------------------------------------------------------
* Convert PBR specular glossiness to metallic-roughness
* -------------------------------------------------------------------------------------------------*/
float3 convertSGToMR(float3 diffuseColor, float3 specularColor, float glossiness, out float metallic, out float2 roughness)
{
  // Constants
  const float dielectricSpecular = 0.04f;  // F0 for dielectrics

  // Compute metallic factor
  float specularIntensity = max(specularColor.r, max(specularColor.g, specularColor.b));
  float isMetal           = smoothstep(dielectricSpecular + 0.01f, dielectricSpecular + 0.05f, specularIntensity);
  metallic                = isMetal;

  // Compute base color
  float3 baseColor;
  if(metallic > 0.0f)
  {
    // Metallic: Use specular as base color
    baseColor = specularColor;
  }
  else
  {
    // Non-metallic: Correct diffuse color for energy conservation
    baseColor = diffuseColor / (1.0f - dielectricSpecular * (1.0f - metallic));
    baseColor = clamp(baseColor, 0.0f, 1.0f);  // Ensure valid color
  }

  // Compute roughness
  float r   = 1.0f - glossiness;
  roughness = float2(r * r);

  return baseColor;
}

/*-------------------------------------------------------------------------------------------------
# `MICROFACET_MIN_ROUGHNESS` Define
> Minimum roughness for microfacet models.

This protects microfacet code from dividing by 0, as well as from numerical
instability around roughness == 0. However, it also means even roughness-0
surfaces will be rendered with a tiny amount of roughness.

This value is ad-hoc; it could probably be lowered without issue.
-------------------------------------------------------------------------------------------------*/
#define MICROFACET_MIN_ROUGHNESS 0.0014142f

/*-------------------------------------------------------------------------------------------------
# Texture-fetcher abstraction (`IGltfTextureFetcher`)

Material evaluation resolves each texture through an `IGltfTextureFetcher`. The caller supplies a
small value type that turns a `GltfTextureInfo` into a sampled color. This keeps a single
`evaluateMaterial` body that works both with a bound `Sampler2D textures[]` descriptor array and
with bindless descriptor heaps (VK_EXT_descriptor_heap):
- A shader created with VK_SHADER_CREATE_DESCRIPTOR_HEAP_BIT_EXT must have setLayoutCount == 0
  (VUID-...-11290), so it cannot bind a descriptor-set array; it provides a fetcher that samples via
  `Texture2D.Handle(...)` / `SamplerState.Handle(...)` (see samples/gltf_raytrace).
- Classic descriptor-array shaders use the built-in `GltfArrayTextureFetcher` below.

This header stays free of any descriptor-heap capability requirement — the `.Handle` usage lives in
the fetcher the caller provides.
-------------------------------------------------------------------------------------------------*/
interface IGltfTextureFetcher
{
  // Resolve and sample the texture referenced by `textureInfo` (applies KHR_texture_transform).
  // `texGrad > 0` selects gradient sampling (ray/compute footprint); `0` selects implicit-LOD Sample.
  float4 fetch(in GltfTextureInfo textureInfo, in float2 texCoords[MAT_EVAL_MESH_STATE_MAX_TEXCOORDS], float texGrad);
}

/*-------------------------------------------------------------------------------------------------
# `GltfArrayTextureFetcher`
> Adapts a bound `Sampler2D textures[]` descriptor array to `IGltfTextureFetcher`, delegating to
> `getTexture()` so the array-sampling logic lives in exactly one place.

Note: Slang cannot default-instantiate a named local of unsized type (the array member makes this
struct unsized). Construct it inline as a temporary instead — e.g. `GltfArrayTextureFetcher(textures)`
- which is exactly how the `Sampler2D textures[]` overload of `evaluateMaterial` uses it.
-------------------------------------------------------------------------------------------------*/
struct GltfArrayTextureFetcher : IGltfTextureFetcher
{
  Sampler2D textures[];
  __init(Sampler2D t[]) { textures = t; }
  float4 fetch(in GltfTextureInfo textureInfo, in float2 texCoords[MAT_EVAL_MESH_STATE_MAX_TEXCOORDS], float texGrad)
  {
    return getTexture(textures, textureInfo, texCoords, texGrad);
  }
}

/*-------------------------------------------------------------------------------------------------
# Function `evaluateMaterial` (generic over a texture fetcher)
> From the incoming `material` and `mesh` info, return a `PbrMaterial` struct for the BSDF system.
> Textures are resolved through `texFetcher`, so the material can be evaluated from a bound
> descriptor array (via `GltfArrayTextureFetcher`), a descriptor heap, or any other source.
-------------------------------------------------------------------------------------------------*/
PbrMaterial evaluateMaterial<F : IGltfTextureFetcher>(in GltfShadeMaterial material, MeshState state, F texFetcher, GltfTextureInfo* texInfos)
{
  // Material Evaluated
  PbrMaterial pbrMat;

  // pbrMetallicRoughness (standard)
  if(material.usePbrSpecularGlossiness == 0)
  {
    // Base Color/Albedo may be defined from a base texture or a flat color
    float4 baseColor = material.pbrBaseColorFactor;
    if(isTexturePresent(material.pbrBaseColorTexture))
    {
      baseColor *= texFetcher.fetch(texInfos[material.pbrBaseColorTexture], state.tc, state.texGrad);
    }
    pbrMat.baseColor = baseColor.rgb;
    pbrMat.opacity   = baseColor.a;

    // Metallic-Roughness
    float roughness = material.pbrRoughnessFactor;
    float metallic  = material.pbrMetallicFactor;
    if(isTexturePresent(material.pbrMetallicRoughnessTexture))
    {
      // Roughness is stored in the 'g' channel, metallic is stored in the 'b' channel.
      float4 metallicRoughnessSample =
          texFetcher.fetch(texInfos[material.pbrMetallicRoughnessTexture], state.tc, state.texGrad);
      roughness *= metallicRoughnessSample.g;
      metallic *= metallicRoughnessSample.b;
    }
    roughness        = max(roughness, MICROFACET_MIN_ROUGHNESS);
    pbrMat.roughness = float2(roughness * roughness);  // Square roughness for the microfacet model
    pbrMat.metallic  = clamp(metallic, 0.0F, 1.0F);
  }
  else
  {
    // KHR_materials_pbrSpecularGlossiness: deprecated but still used in many places
    float4 diffuse    = material.pbrDiffuseFactor;
    float  glossiness = material.pbrGlossinessFactor;
    float3 specular   = material.pbrSpecularFactor;

    if(isTexturePresent(material.pbrDiffuseTexture))
    {
      diffuse *= texFetcher.fetch(texInfos[material.pbrDiffuseTexture], state.tc, state.texGrad);
    }

    if(isTexturePresent(material.pbrSpecularGlossinessTexture))
    {
      float4 specularGlossinessSample =
          texFetcher.fetch(texInfos[material.pbrSpecularGlossinessTexture], state.tc, state.texGrad);
      specular *= specularGlossinessSample.rgb;
      glossiness *= specularGlossinessSample.a;
    }

    pbrMat.baseColor = convertSGToMR(diffuse.rgb, specular, glossiness, pbrMat.metallic, pbrMat.roughness);
    pbrMat.opacity   = diffuse.a;
  }

  // Occlusion Map
  pbrMat.occlusion = material.occlusionStrength;
  if(isTexturePresent(material.occlusionTexture))
  {
    float occlusion  = texFetcher.fetch(texInfos[material.occlusionTexture], state.tc, state.texGrad).r;
    pbrMat.occlusion = 1.0 + pbrMat.occlusion * (occlusion - 1.0);
  }

  // Normal Map
  pbrMat.N                = state.N;
  pbrMat.T                = state.T;
  pbrMat.B                = state.B;
  pbrMat.Ng               = state.Ng;
  bool needsTangentUpdate = false;

  if(isTexturePresent(material.normalTexture))
  {
    float3 normal_vector = texFetcher.fetch(texInfos[material.normalTexture], state.tc, state.texGrad).xyz;
    normal_vector        = normal_vector * 2.0F - 1.0F;
    normal_vector *= float3(material.normalTextureScale, material.normalTextureScale, 1.0F);
    float3x3 tbn = float3x3(state.T, state.B, state.N);
    pbrMat.N     = normalize(mul(normal_vector, tbn));

    // Mark that we need to update T and B due to normal perturbation
    needsTangentUpdate = true;
  }

  // Emissive term
  pbrMat.emissive = material.emissiveFactor;
  if(isTexturePresent(material.emissiveTexture))
  {
    pbrMat.emissive *= texFetcher.fetch(texInfos[material.emissiveTexture], state.tc, state.texGrad).rgb;
  }
  pbrMat.emissive = max(float3(0.0F), pbrMat.emissive);

  // KHR_materials_specular
  // https://github.com/KhronosGroup/glTF/tree/main/extensions/2.0/Khronos/KHR_materials_specular
  pbrMat.specularColor = material.specularColorFactor;
  if(isTexturePresent(material.specularColorTexture))
  {
    pbrMat.specularColor *= texFetcher.fetch(texInfos[material.specularColorTexture], state.tc, state.texGrad).rgb;
  }

  // KHR_materials_specular
  pbrMat.specular = material.specularFactor;
  if(isTexturePresent(material.specularTexture))
  {
    pbrMat.specular *= texFetcher.fetch(texInfos[material.specularTexture], state.tc, state.texGrad).a;
  }

  // Dielectric Specular
  float ior1 = 1.0F;                                    // IOR of the current medium (e.g., air)
  float ior2 = material.ior;                            // IOR of the material
  if(state.isInside && (material.thicknessFactor > 0))  // If the material is thin-walled, we don't need to consider the inside IOR.
  {
    ior1 = material.ior;
    ior2 = 1.0F;
  }
  pbrMat.ior1 = ior1;
  pbrMat.ior2 = ior2;


  // KHR_materials_transmission
  pbrMat.transmission = material.transmissionFactor;
  if(isTexturePresent(material.transmissionTexture))
  {
    pbrMat.transmission *= texFetcher.fetch(texInfos[material.transmissionTexture], state.tc, state.texGrad).r;
  }
  // No glTF extension tints the transmission lobe on its own; leave it neutral. `pbrMat` is not
  // zero-initialized above, so this must be assigned explicitly.
  pbrMat.transmissionTint = float3(1.0F);

  // KHR_materials_volume
  pbrMat.attenuationColor    = material.attenuationColor;
  pbrMat.attenuationDistance = material.attenuationDistance;
  pbrMat.thickness           = material.thicknessFactor;
  if(isTexturePresent(material.thicknessTexture))
  {
    pbrMat.thickness *= texFetcher.fetch(texInfos[material.thicknessTexture], state.tc, state.texGrad).g;
  }

  // KHR_materials_clearcoat
  pbrMat.clearcoat          = material.clearcoatFactor;
  pbrMat.clearcoatRoughness = material.clearcoatRoughness;
  pbrMat.Nc                 = pbrMat.N;
  if(isTexturePresent(material.clearcoatTexture))
  {
    pbrMat.clearcoat *= texFetcher.fetch(texInfos[material.clearcoatTexture], state.tc, state.texGrad).r;
  }
  if(isTexturePresent(material.clearcoatRoughnessTexture))
  {
    pbrMat.clearcoatRoughness *=
        texFetcher.fetch(texInfos[material.clearcoatRoughnessTexture], state.tc, state.texGrad).g;
  }
  if(isTexturePresent(material.clearcoatNormalTexture))
  {
    float3x3 tbn         = float3x3(pbrMat.T, pbrMat.B, pbrMat.Nc);
    float3 normal_vector = texFetcher.fetch(texInfos[material.clearcoatNormalTexture], state.tc, state.texGrad).xyz;
    normal_vector        = normal_vector * 2.0F - 1.0F;
    pbrMat.Nc            = normalize(mul(normal_vector, tbn));
  }
  pbrMat.clearcoatRoughness = max(pbrMat.clearcoatRoughness, 0.001F);

  // KHR_materials_iridescence
  float iridescence          = material.iridescenceFactor;
  float iridescenceThickness = material.iridescenceThicknessMaximum;
  pbrMat.iridescenceIor      = material.iridescenceIor;
  if(isTexturePresent(material.iridescenceTexture))
  {
    iridescence *= texFetcher.fetch(texInfos[material.iridescenceTexture], state.tc, state.texGrad).x;
  }
  if(isTexturePresent(material.iridescenceThicknessTexture))
  {
    const float t = texFetcher.fetch(texInfos[material.iridescenceThicknessTexture], state.tc, state.texGrad).y;
    iridescenceThickness = lerp(material.iridescenceThicknessMinimum, material.iridescenceThicknessMaximum, t);
  }
  pbrMat.iridescence = (iridescenceThickness > 0.0f) ? iridescence : 0.0f;  // No iridescence when the thickness is zero.
  pbrMat.iridescenceThickness = iridescenceThickness;

  // KHR_materials_anisotropy
  float anisotropyStrength = material.anisotropyStrength;
  // If the anisotropyStrength == 0.0f (default), the roughness is isotropic.
  // No need to rotate the anisotropyDirection or tangent space.
  if(anisotropyStrength > 0.0F)
  {
    float2 anisotropyDirection = float2(1.0f, 0.0f);  // By default the anisotropy strength is along the tangent.
    if(isTexturePresent(material.anisotropyTexture))
    {
      const float4 anisotropyTex = texFetcher.fetch(texInfos[material.anisotropyTexture], state.tc, state.texGrad);

      // .xy encodes the direction in (tangent, bitangent) space. Remap from [0, 1] to [-1, 1].
      anisotropyDirection = normalize(float2(anisotropyTex.xy) * 2.0f - 1.0f);
      // .z encodes the strength in range [0, 1].
      anisotropyStrength *= anisotropyTex.z;
    }

    // Adjust the roughness to account for anisotropy.
    pbrMat.roughness.x = lerp(pbrMat.roughness.y, 1.0f, anisotropyStrength * anisotropyStrength);

    // Rotate the anisotropy direction in the tangent space.
    const float s       = material.anisotropyRotation.x;  // Sin and Cos of the rotation angle.
    const float c       = material.anisotropyRotation.y;
    anisotropyDirection = float2(c * anisotropyDirection.x + s * anisotropyDirection.y,
                                 c * anisotropyDirection.y - s * anisotropyDirection.x);

    // Update the tangent to be along the anisotropy direction in tangent space.
    const float3 T_aniso = pbrMat.T * anisotropyDirection.x + pbrMat.B * anisotropyDirection.y;

    pbrMat.T           = T_aniso;
    needsTangentUpdate = true;
  }

  // Perform tangent and bitangent updates if necessary
  if(needsTangentUpdate)
  {
    // Ensure T, B, and N are orthonormal
    pbrMat.B            = normalize(cross(pbrMat.N, pbrMat.T));
    float bitangentSign = sign(dot(state.B, pbrMat.B));
    pbrMat.B            = pbrMat.B * bitangentSign;
    pbrMat.T            = normalize(cross(pbrMat.B, pbrMat.N) * bitangentSign);
  }

  // KHR_materials_sheen
  pbrMat.sheenColor = material.sheenColorFactor;
  if(isTexturePresent(material.sheenColorTexture))
  {
    pbrMat.sheenColor *= float3(texFetcher.fetch(texInfos[material.sheenColorTexture], state.tc, state.texGrad).xyz);  // sRGB
  }

  pbrMat.sheenRoughness = material.sheenRoughnessFactor;
  if(isTexturePresent(material.sheenRoughnessTexture))
  {
    pbrMat.sheenRoughness *= texFetcher.fetch(texInfos[material.sheenRoughnessTexture], state.tc, state.texGrad).w;
  }
  pbrMat.sheenRoughness = max(MICROFACET_MIN_ROUGHNESS, pbrMat.sheenRoughness);

  // KHR_materials_dispersion
  pbrMat.dispersion = material.dispersion;

  // KHR_materials_diffuse_transmission
  pbrMat.diffuseTransmissionFactor = material.diffuseTransmissionFactor;
  if(isTexturePresent(material.diffuseTransmissionTexture))
  {
    pbrMat.diffuseTransmissionFactor *=
        texFetcher.fetch(texInfos[material.diffuseTransmissionTexture], state.tc, state.texGrad).a;
  }
  pbrMat.diffuseTransmissionColor = material.diffuseTransmissionColor;
  if(isTexturePresent(material.diffuseTransmissionColorTexture))
  {
    pbrMat.diffuseTransmissionColor =
        texFetcher.fetch(texInfos[material.diffuseTransmissionColorTexture], state.tc, state.texGrad).rgb;
  }

  // KHR_materials_scatter. Evaluated after KHR_materials_volume (thickness selects the mode) and
  // after KHR_materials_diffuse_transmission, whose lobes the thin-walled mode repurposes.
  pbrMat.scatterCoefficient = float3(0.0);
  pbrMat.scatterAnisotropy  = 0.0;

  // scatterStrength = scatterStrengthFactor * scatterStrengthTexture.a
  float scatterStrength = material.scatterStrengthFactor;
  if(isTexturePresent(material.scatterStrengthTexture))
  {
    scatterStrength *= texFetcher.fetch(texInfos[material.scatterStrengthTexture], state.tc, state.texGrad).a;
  }

  // The extension is a no-op at zero strength (the spec default), so only apply it when there is
  // scattering to do.
  if(scatterStrength > 0.0)
  {
    // multiscatterColor = multiscatterColorFactor * multiscatterColorTexture.rgb
    float3 multiscatterColor = material.multiscatterColorFactor;
    if(isTexturePresent(material.multiscatterColorTexture))
    {
      multiscatterColor *= texFetcher.fetch(texInfos[material.multiscatterColorTexture], state.tc, state.texGrad).rgb;
    }

    float g = material.scatterAnisotropy;

    if(pbrMat.thickness > 0.0)
    {
      // Volumetric mode (KHR_materials_volume with thickness > 0): the surface BSDF is untouched.
      // Split the attenuation coefficient sigma_t into scattering and absorption with the
      // strength-scaled single-scatter albedo; sigma_s drives the in-medium phase function.
      // The sigma_t expression must match volumeExtinctionCoefficient() in bsdf_functions.h.slang.
      float3 singleScatterAlbedo = multiToSingleScatterAlbedo(scatterStrength * multiscatterColor, g);
      float3 attenuationCoef = -log(max(pbrMat.attenuationColor, float3(0.001))) / max(pbrMat.attenuationDistance, 0.001);
      pbrMat.scatterCoefficient = attenuationCoef * singleScatterAlbedo;
      pbrMat.scatterAnisotropy  = g;
    }
    else
    {
      // Thin-walled mode (no volume, or thickness = 0). The spec replaces the transmissive part of
      // the dielectric base with
      //   mix(specular_btdf * baseColor, scatter_bsdf(multiscatterColor, g), scatterStrength)
      // where scatter_bsdf is a Lambertian BSDF sending (1+g)/2 of its energy through the surface
      // and (1-g)/2 back. There is no dedicated scatter lobe in the BSDF, so it is expressed with
      // the diffuse-transmission / diffuse-reflection lobes, whose weights below the specular layer
      // are `transmission` for the BTDF and `1 - transmission` for the diffuse bucket. Moving
      // `scatterWeight = transmission * scatterStrength` from the BTDF into that bucket reproduces
      // the spec lobe weights exactly, for any transmission and any g. Note both factors: without
      // transmission the extension must have no effect at all.
      float scatterWeight = pbrMat.transmission * scatterStrength;
      if(scatterWeight > 0.0)
      {
        // The diffuse bucket holds four shares: the material's own diffuse reflection and diffuse
        // transmission (untouched by scatter), plus the two halves of scatter_bsdf.
        float forwardFraction = 0.5 * (1.0 + g);
        float baseDiffuse     = 1.0 - pbrMat.transmission;         // base, untouched by scatter
        float baseDtFactor    = pbrMat.diffuseTransmissionFactor;  // 0 unless KHR_materials_diffuse_transmission
        float baseTransWeight = baseDiffuse * baseDtFactor;
        float baseReflWeight  = baseDiffuse * (1.0 - baseDtFactor);
        float forwardWeight   = scatterWeight * forwardFraction;
        float backwardWeight  = scatterWeight - forwardWeight;
        float diffuseWeight   = baseDiffuse + scatterWeight;  // the four shares; >= scatterWeight > 0

        pbrMat.transmission -= scatterWeight;  // = transmission * (1 - scatterStrength)

        // Split the bucket. Note this is the forward share *of the bucket*, not the anisotropy's
        // forward fraction: the bucket also carries the untouched base, so it has to be normalised
        // by the bucket's own weight. (Using the bare fraction is only right at transmission = 1.)
        pbrMat.diffuseTransmissionFactor = (baseTransWeight + forwardWeight) / diffuseWeight;

        // Each of the two diffuse lobes now carries contributions the spec colors differently, so
        // each takes the energy-weighted average of its own two shares. Both shares of a lobe have
        // the same Lambertian shape, so averaging their colors is exact, not an approximation.
        float3 diffuseBaseColor = pbrMat.baseColor;
        pbrMat.diffuseTransmissionColor =
            lerp(pbrMat.diffuseTransmissionColor, multiscatterColor, forwardWeight / max(baseTransWeight + forwardWeight, 1e-6));
        pbrMat.baseColor =
            lerp(diffuseBaseColor, multiscatterColor, backwardWeight / max(baseReflWeight + backwardWeight, 1e-6));

        // ...but baseColor also tints the residual specular BTDF, which the spec leaves at the
        // material's baseColor. transmissionTint cancels the blend back out for that lobe alone
        // (the BSDF tints it by `baseColor * transmissionTint`), and the product is bounded by
        // baseColor so it can never add energy. It degrades in one corner: at transmission = 1 with
        // a multiscatterColor channel of exactly 0, the blended color is 0 in that channel and a
        // multiplier cannot recover baseColor from it, so the residual BTDF loses that channel.
        pbrMat.transmissionTint =
            (pbrMat.transmission > 0.0) ? diffuseBaseColor / max(pbrMat.baseColor, float3(1e-4)) : float3(1.0);
      }
    }
  }

  return pbrMat;
}

/*-------------------------------------------------------------------------------------------------
# `evaluateMaterial` overloads for a bound `Sampler2D textures[]` descriptor array
> Thin wrappers over the generic `evaluateMaterial` above; they adapt the array with
> `GltfArrayTextureFetcher` so there is a single material-evaluation body to maintain.
-------------------------------------------------------------------------------------------------*/
PbrMaterial evaluateMaterial(in GltfShadeMaterial material, MeshState state, Sampler2D textures[], GltfTextureInfo* texInfos)
{
  return evaluateMaterial(material, state, GltfArrayTextureFetcher(textures), texInfos);
}

// Compatibility function
PbrMaterial evaluateMaterial(in GltfShadeMaterial material,
                             in float3            normal,
                             in float3            tangent,
                             in float3            bitangent,
                             in float2            texCoord,
                             Sampler2D            textures[],
                             GltfTextureInfo*     texInfos)
{
  float2    tcoords[MAT_EVAL_MESH_STATE_MAX_TEXCOORDS] = { texCoord, };
  MeshState mesh       = MeshState(normal, tangent, bitangent, normal, tcoords, false, 0.0f);
  return evaluateMaterial(material, mesh, textures, texInfos);
}

#endif  // MAT_EVAL_H
