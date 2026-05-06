/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef PBR_IBL_EVAL_H
#define PBR_IBL_EVAL_H 1

/*-------------------------------------------------------------------------------------------------
Khronos split-sum image-based lighting evaluators -- the consumer side of the IBL preprocessing
shaders in `hdr_prefilter_diffuse.slang`, `hdr_prefilter_glossy.slang`, and
`hdr_integrate_brdf.slang`. Designed to match the glTF Sample Viewer reference renderer's
`ibl.glsl` layer for raster shaders that do their own deterministic split-sum lighting (path
tracers should use the per-lobe BSDF integrators in `bsdf_functions.h.slang` instead).

All evaluators take their textures/samplers as explicit parameters so this header has no
descriptor-set assumptions. World-space directions go in; the env rotation around +Y is applied
internally so callers stay in their native frame.
-------------------------------------------------------------------------------------------------*/

#include "bsdf_functions.h.slang"      // applyVolumeAttenuation
#include "constants.h.slang"
#include "functions.h.slang"
#include "pbr_ggx_microfacet.h.slang"  // Khronos analytic helpers (khronosDielectricF0,
                                       //   khronosClearcoatFresnel, layeredCompositeKhronos,
                                       //   albedoSheenScalingLUT, ...)
#include "pbr_material_types.h.slang"


/*-------------------------------------------------------------------------------------------------
# Function `getIBLDiffuse`
> Lambertian-prefiltered cubemap lookup along a world-space direction.
-------------------------------------------------------------------------------------------------*/
float3 getIBLDiffuse(SamplerCube diffuseEnv, float3 n, float envRotation, float envIntensity)
{
  float3 dir = rotate(n, float3(0, 1, 0), -envRotation);
  return diffuseEnv.SampleLevel(dir, 0.0F).rgb * envIntensity;
}

/*-------------------------------------------------------------------------------------------------
# Function `getIBLRadianceGGX`
> GGX-prefiltered cubemap lookup at the perfect-mirror reflection direction.

`perceptualRoughness` is the artist-facing roughness in [0,1] (sqrt of GGX alpha). The GGX
prefilter is mip-mapped: roughness scales linearly to `[0, mipCount-1]`.
-------------------------------------------------------------------------------------------------*/
float3 getIBLRadianceGGX(SamplerCube glossyEnv, float3 n, float3 v, float perceptualRoughness, float envRotation, float envIntensity)
{
  uint2 size;
  uint  mipCount;
  glossyEnv.GetDimensions(0, size.x, size.y, mipCount);
  float  lod        = perceptualRoughness * float(mipCount - 1);
  float3 reflection = normalize(reflect(-v, n));
  float3 dir        = rotate(reflection, float3(0, 1, 0), -envRotation);
  return glossyEnv.SampleLevel(dir, lod).rgb * envIntensity;
}

/*-------------------------------------------------------------------------------------------------
# Function `getIBLAnisotropy`
> GGX-prefiltered cubemap lookup with KHR_materials_anisotropy bent-normal compensation.

Stretches the highlight along `anisotropyDirection` (the world-space tangent on which the
highlight extends) by an amount controlled by `anisotropyStrength` in [0,1]. Falls back to the
isotropic answer when `anisotropyStrength == 0`. Reference: `ibl.glsl:118-138`.
-------------------------------------------------------------------------------------------------*/
float3 getIBLAnisotropy(SamplerCube glossyEnv,
                       float3      n,
                       float3      v,
                       float       perceptualRoughness,
                       float       anisotropyStrength,
                       float3      anisotropyDirection,
                       float       envRotation,
                       float       envIntensity)
{
  float3 anisotropicTangent = cross(anisotropyDirection, v);
  float3 anisotropicNormal  = cross(anisotropicTangent, anisotropyDirection);
  float  bendFactor         = 1.0F - anisotropyStrength * (1.0F - perceptualRoughness);
  float  bendFactor4        = bendFactor * bendFactor * bendFactor * bendFactor;
  float3 bentNormal         = normalize(lerp(anisotropicNormal, n, bendFactor4));

  uint2 size;
  uint  mipCount;
  glossyEnv.GetDimensions(0, size.x, size.y, mipCount);
  float  lod        = perceptualRoughness * float(mipCount - 1);
  float3 reflection = normalize(reflect(-v, bentNormal));
  float3 dir        = rotate(reflection, float3(0, 1, 0), -envRotation);
  return glossyEnv.SampleLevel(dir, lod).rgb * envIntensity;
}

/*-------------------------------------------------------------------------------------------------
# Function `getIBLGGXFresnelMS`
> Fdez-Aguera multiscatter Fresnel for the split-sum BRDF.

Uses the BRDF LUT baked by `hdr_integrate_brdf.slang`. The integrator stores `(scale, bias)`
with the V axis inverted (`integrateBrdf(uv.x, 1 - uv.y)`), so we flip `perceptualRoughness`
here to keep the natural caller-side convention of `(NdotV, perceptualRoughness)`.

`specularWeight` carries KHR_materials_specular's scalar (1.0 by default for metals).

Reference: `ibl.glsl:26-43`.
-------------------------------------------------------------------------------------------------*/
float3 getIBLGGXFresnelMS(Sampler2D ggxLut, float3 n, float3 v, float perceptualRoughness, float3 F0, float specularWeight)
{
  float  NdotV  = clampedDot(n, v);
  float2 uv     = clamp(float2(NdotV, 1.0F - perceptualRoughness), float2(0.0F), float2(1.0F));
  float2 f_ab   = ggxLut.SampleLevel(uv, 0.0F).rg;
  float3 Fr     = max(float3(1.0F - perceptualRoughness), F0) - F0;
  float3 k_S    = F0 + Fr * pow(1.0F - NdotV, 5.0F);
  float3 FssEss = specularWeight * (k_S * f_ab.x + f_ab.y);

  float  Ems    = (1.0F - (f_ab.x + f_ab.y));
  float3 F_avg  = specularWeight * (F0 + (1.0F - F0) / 21.0F);
  float3 FmsEms = Ems * FssEss * F_avg / (1.0F - F_avg * Ems);

  return FssEss + FmsEms;
}

/*-------------------------------------------------------------------------------------------------
# Function `getIBLCharlie`
> Sheen IBL using the runtime-baked Charlie directional-albedo LUT.

`sheenLut.r` doubles as the BRDF integrand (`ibl.glsl:149`) and as the energy-compensation factor
(Khronos `albedoSheenScalingLUT`). The Charlie-prefiltered cubemap is approximated here by
sampling the diffuse-prefiltered cubemap -- known, documented limitation: at very low sheen
roughness the highlight is overly soft. Replacing this with a real Charlie cube would need a
`hdr_prefilter_charlie.slang` plus a per-mip dispatch driver akin to the GGX prefilter.
-------------------------------------------------------------------------------------------------*/
float3 getIBLCharlie(SamplerCube diffuseEnv,
                    Sampler2D   sheenLut,
                    float3      n,
                    float3      v,
                    float       sheenRoughness,
                    float3      sheenColor,
                    float       envRotation,
                    float       envIntensity)
{
  float  NdotV      = clampedDot(n, v);
  float3 reflection = normalize(reflect(-v, n));
  float3 dir        = rotate(reflection, float3(0, 1, 0), -envRotation);
  float3 sheenLight = diffuseEnv.SampleLevel(dir, 0.0F).rgb * envIntensity;
  float2 lutUv      = clamp(float2(NdotV, sheenRoughness), float2(0.0F), float2(1.0F));
  float  brdf       = sheenLut.SampleLevel(lutUv, 0.0F).r;
  return sheenLight * sheenColor * brdf;
}


/*-------------------------------------------------------------------------------------------------
# struct `IBLLobes`
> Per-lobe IBL radiances and Fresnels packed for `composeIBLAnalyticKhronos`.

The compositor mixes these via the Khronos analytic split-sum layering (`pbr.frag:218-220`).
`computeLobeWeights` is intentionally NOT used here: the IBL Fresnel terms already encode the
diffuse/specular split for dielectrics, and applying the lobe weights on top would double-mix.
-------------------------------------------------------------------------------------------------*/
struct IBLLobes
{
  float3 f_diffuse;              // baseColor * irradiance (caller may fold in transmission / diffuse-trans)
  float3 f_metal_specular;       // metal specular IBL radiance (anisotropic-aware, pre-Fresnel)
  float3 f_dielectric_specular;  // dielectric specular IBL radiance (typically same as f_metal_specular)
  float3 f_metal_fresnel;        // Fdez-Aguera multiscatter Fresnel at NdotV, F0 = baseColor
  float3 f_dielectric_fresnel;   // Fdez-Aguera multiscatter Fresnel at NdotV, F0 = f0_dielectric, specularWeight folded in
  float3 f_sheen;                // Charlie env (stand-in) * sheenColor * sheen LUT
  float3 f_clearcoat;            // GGX env on Nc at clearcoatRoughness
  float3 clearcoatFresnel;       // F_Schlick(clearcoatF0, 1, clearcoatNdotV); precomputed by caller
  float3 irid_dielectric;        // iridescence Fresnel for the dielectric specular lobe (or 0)
  float3 irid_metallic;          // iridescence Fresnel for the metal specular lobe (or 0)
  float  iridescenceFactor;      // 0 disables the iridescence blends below
};

/*-------------------------------------------------------------------------------------------------
# Function `composeIBLAnalyticKhronos`
> Khronos analytic split-sum layering of pre-evaluated `IBLLobes`.

Layering follows `pbr.frag:218-220` and is delegated to the shared `layeredCompositeKhronos`
(in `pbr_ggx_microfacet.h.slang`), which also serves the per-light direct evaluator. This
header just resolves the per-lobe IBL BRDFs (with the iridescence override) and the sheen
scaling factor before handing off.

`sheenLut` is sampled once for the energy-compensation factor (Khronos `albedoSheenScalingLUT`).
-------------------------------------------------------------------------------------------------*/
// Internal helper: resolves per-lobe IBL BRDFs (with iridescence override) and tail-calls
// the shared layered composite with a caller-supplied sheen scale factor.
float3 _composeIBLAnalyticKhronosCore(PbrMaterial mat, IBLLobes lobes, float sheenScale)
{
  // Base per-lobe IBL BRDFs (matches `pbr.frag:197-205`).
  float3 f_metal_brdf_ibl      = lobes.f_metal_fresnel * lobes.f_metal_specular;
  float3 f_dielectric_brdf_ibl = lerp(lobes.f_diffuse, lobes.f_dielectric_specular, lobes.f_dielectric_fresnel);

  // Iridescence override on both Fresnels (`pbr.frag:203-206`).
  if(lobes.iridescenceFactor > 0.0F)
  {
    f_metal_brdf_ibl = lerp(f_metal_brdf_ibl, lobes.f_metal_specular * lobes.irid_metallic, lobes.iridescenceFactor);
    f_dielectric_brdf_ibl = lerp(f_dielectric_brdf_ibl,
                                 mix_rgb(lobes.f_diffuse, lobes.f_dielectric_specular, lobes.irid_dielectric),
                                 lobes.iridescenceFactor);
  }

  return layeredCompositeKhronos(f_dielectric_brdf_ibl, f_metal_brdf_ibl, mat.metallic, lobes.f_sheen,
                                 sheenScale, lobes.f_clearcoat, mat.clearcoat, lobes.clearcoatFresnel);
}

float3 composeIBLAnalyticKhronos(PbrMaterial mat, IBLLobes lobes, float3 V, Sampler2D sheenLut)
{
  // Sheen albedo scaling (energy-compensation LUT). Uses the shared `albedoSheenScalingLUT`
  // helper (one sample of the Charlie LUT at NdotV; the punctual evaluator samples it twice
  // for the `min(LUT(NdotV), LUT(NdotL))` bidirectional form).
  float sheenScale = 1.0F;
  if(max3(mat.sheenColor) > 0.0F)
  {
    float NdotV = clampedDot(mat.N, V);
    sheenScale  = 1.0F - max3(mat.sheenColor) * albedoSheenScalingLUT(sheenLut, NdotV, mat.sheenRoughness);
  }
  return _composeIBLAnalyticKhronosCore(mat, lobes, sheenScale);
}

// No-sheen overload: use when the scene guarantees no sheen materials. Skips the sheen LUT
// sample and leaves `sheenScale = 1`. Caller should also leave `lobes.f_sheen = float3(0)`.
float3 composeIBLAnalyticKhronos(PbrMaterial mat, IBLLobes lobes, float3 V)
{
  return _composeIBLAnalyticKhronosCore(mat, lobes, 1.0F);
}

/*-------------------------------------------------------------------------------------------------
# Function `evalKhronosIBL`
> One-call IBL evaluator mirroring `pbr.frag:162-228` for a glTF material.

Collapses the full Khronos IBL stack (diffuse + specular + sheen + clearcoat + iridescence +
diffuse-transmission + optional screen-space specular transmission) into a single call so raster
shaders match the ease of the path tracer's `bsdfSample/bsdfEvaluate` pair. Handles:

  * Anisotropy detection and bent-normal GGX when `mat.roughness.x != mat.roughness.y`.
  * KHR_materials_specular colored F0 and specularWeight.
  * KHR_materials_iridescence at NdotV (Belcour thin-film factor).
  * KHR_materials_clearcoat on a separate normal with Schlick Fresnel.
  * KHR_materials_sheen with the runtime-baked Charlie directional-albedo LUT.
  * KHR_materials_diffuse_transmission with back-hemisphere irradiance and
    KHR_materials_volume absorption along `diffuseTransmissionThickness`.
  * KHR_materials_transmission screen-space refraction via `transmissionRadiance` +
    `haveTransmissionSample`. Caller precomputes `transmissionRadiance` from the framebuffer
    (sample-specific) and passes `haveTransmissionSample = false` when no capture is available.

NOT included here (intentionally separate, matching `pbr.frag:352-356` + `pbr.frag:373`):
  * KHR_materials_volume_scatter -- add `volumeScatterApprox(mat, ...) * (1 - mat.clearcoat *
    khronosClearcoatFresnel(mat, V))` at the call site.
  * Ambient occlusion -- apply `* mat.occlusion` to the returned value.
  * Emissive -- add `mat.emissive * (1 - mat.clearcoat * khronosClearcoatFresnel(mat, V))`.

## Inputs
- `mat`:                          surface material.
- `V`:                            normalized view direction (toward camera).
- `diffuseEnv`, `glossyEnv`:      Lambert- and GGX-prefiltered HDR cubemaps.
- `ggxLut`:                       split-sum BRDF LUT (baked by `hdr_integrate_brdf.slang`, V-axis
                                  inverted convention).
- `sheenLut`:                     Charlie directional-albedo LUT (runtime-baked).
- `envRotation`, `envIntensity`:  world-space rotation around +Y and scalar intensity.
- `diffuseTransmissionThickness`: world-space thickness for KHR_materials_diffuse_transmission
                                  volume attenuation (typically `mat.thickness * model_scale_avg`).
                                  Pass `0.0` to disable.
- `transmissionRadiance`:         precomputed screen-space refraction radiance, or `float3(0)`.
- `haveTransmissionSample`:       whether `transmissionRadiance` is valid.
-------------------------------------------------------------------------------------------------*/
// Internal helper: fills everything in IBLLobes except `f_sheen`. Caller sets `f_sheen` and
// passes the sheen-free or sheen-aware composer.
void _evalKhronosIBLFillCommon(PbrMaterial mat,
                               float3      V,
                               SamplerCube diffuseEnv,
                               SamplerCube glossyEnv,
                               Sampler2D   ggxLut,
                               float       envRotation,
                               float       envIntensity,
                               float       diffuseTransmissionThickness,
                               float3      transmissionRadiance,
                               bool        haveTransmissionSample,
                               NVSHADERS_INOUT_TYPE(IBLLobes) lobes)
{
  float  NdotV               = clampedDot(mat.N, V);
  float  perceptualRoughness = sqrt(mat.roughness.y);
  bool   isAnisotropic       = abs(mat.roughness.x - mat.roughness.y) > 1e-6F;
  float  anisotropyStrength  = isAnisotropic
                                   ? sqrt(saturate((mat.roughness.x - mat.roughness.y) / max(1.0F - mat.roughness.y, 1e-5F)))
                                   : 0.0F;
  float3 f0_dielectric       = khronosDielectricF0(mat);

  lobes.f_diffuse = getIBLDiffuse(diffuseEnv, mat.N, envRotation, envIntensity) * mat.baseColor;

  // Diffuse transmission: back-hemisphere irradiance + volume absorption.
  if(mat.diffuseTransmissionFactor > 0.0F)
  {
    float3 diffuseTransIBL = getIBLDiffuse(diffuseEnv, -mat.N, envRotation, envIntensity) * mat.diffuseTransmissionColor;
    if(diffuseTransmissionThickness > 0.0F)
      diffuseTransIBL = applyVolumeAttenuation(diffuseTransIBL, diffuseTransmissionThickness, mat.attenuationColor, mat.attenuationDistance);
    lobes.f_diffuse = lerp(lobes.f_diffuse, diffuseTransIBL, mat.diffuseTransmissionFactor);
  }

  // Screen-space specular transmission overrides the diffuse slot by `transmissionFactor`.
  if(haveTransmissionSample)
    lobes.f_diffuse = lerp(lobes.f_diffuse, transmissionRadiance, mat.transmission);

  // Specular IBL -- metal and dielectric share the radiance; anisotropy bends the reflection.
  float3 f_specular_ibl = isAnisotropic ? getIBLAnisotropy(glossyEnv, mat.N, V, perceptualRoughness,
                                                           anisotropyStrength, mat.B, envRotation, envIntensity)
                                        : getIBLRadianceGGX(glossyEnv, mat.N, V, perceptualRoughness, envRotation, envIntensity);
  lobes.f_metal_specular      = f_specular_ibl;
  lobes.f_dielectric_specular = f_specular_ibl;

  // Fdez-Aguera split-sum + multiscatter Fresnels.
  lobes.f_metal_fresnel      = getIBLGGXFresnelMS(ggxLut, mat.N, V, perceptualRoughness, mat.baseColor, 1.0F);
  lobes.f_dielectric_fresnel = getIBLGGXFresnelMS(ggxLut, mat.N, V, perceptualRoughness, f0_dielectric, mat.specular);

  // Clearcoat IBL on its own normal + Schlick Fresnel the composer needs.
  lobes.f_clearcoat      = (mat.clearcoat > 0.0F)
                              ? getIBLRadianceGGX(glossyEnv, mat.Nc, V, mat.clearcoatRoughness, envRotation, envIntensity)
                              : float3(0.0F);
  lobes.clearcoatFresnel = khronosClearcoatFresnel(mat, V);

  // Iridescence Fresnel evaluated at NdotV (matches `pbr.frag:134-140`).
  lobes.iridescenceFactor = 0.0F;
  lobes.irid_dielectric   = float3(0.0F);
  lobes.irid_metallic     = float3(0.0F);
  if(mat.iridescence > 0.0F && mat.iridescenceThickness > 0.0F)
  {
    lobes.iridescenceFactor = mat.iridescence;
    lobes.irid_metallic     = thin_film_factor(mat.iridescenceThickness, mat.iridescenceIor, mat.ior2, mat.ior1, NdotV);
    lobes.irid_dielectric   = lobes.irid_metallic;
  }
}

float3 evalKhronosIBL(PbrMaterial mat,
                      float3      V,
                      SamplerCube diffuseEnv,
                      SamplerCube glossyEnv,
                      Sampler2D   ggxLut,
                      Sampler2D   sheenLut,
                      float       envRotation,
                      float       envIntensity,
                      float       diffuseTransmissionThickness,
                      float3      transmissionRadiance,
                      bool        haveTransmissionSample)
{
  IBLLobes lobes;
  _evalKhronosIBLFillCommon(mat, V, diffuseEnv, glossyEnv, ggxLut, envRotation, envIntensity,
                            diffuseTransmissionThickness, transmissionRadiance, haveTransmissionSample, lobes);
  lobes.f_sheen = (max3(mat.sheenColor) > 0.0F)
                      ? getIBLCharlie(diffuseEnv, sheenLut, mat.N, V, mat.sheenRoughness, mat.sheenColor, envRotation, envIntensity)
                      : float3(0.0F);
  return composeIBLAnalyticKhronos(mat, lobes, V, sheenLut);
}

/*-------------------------------------------------------------------------------------------------
# Function `evalKhronosIBL` (no-sheen overload)
> Same as the full `evalKhronosIBL` but skips the KHR_materials_sheen path.

Use when the scene guarantees no material carries `KHR_materials_sheen` (i.e.
`mat.sheenColor == 0` for every material). Avoids the need to bake and bind the Charlie sheen
LUT. If called on a material that does have sheen set, the sheen contribution is silently
dropped.
-------------------------------------------------------------------------------------------------*/
float3 evalKhronosIBL(PbrMaterial mat,
                      float3      V,
                      SamplerCube diffuseEnv,
                      SamplerCube glossyEnv,
                      Sampler2D   ggxLut,
                      float       envRotation,
                      float       envIntensity,
                      float       diffuseTransmissionThickness,
                      float3      transmissionRadiance,
                      bool        haveTransmissionSample)
{
  IBLLobes lobes;
  _evalKhronosIBLFillCommon(mat, V, diffuseEnv, glossyEnv, ggxLut, envRotation, envIntensity,
                            diffuseTransmissionThickness, transmissionRadiance, haveTransmissionSample, lobes);
  lobes.f_sheen = float3(0.0F);
  return composeIBLAnalyticKhronos(mat, lobes, V);
}

#endif  // PBR_IBL_EVAL_H
