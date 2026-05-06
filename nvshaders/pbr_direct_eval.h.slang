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

#ifndef PBR_DIRECT_EVAL_H
#define PBR_DIRECT_EVAL_H 1

/*-------------------------------------------------------------------------------------------------
Khronos-faithful direct (analytic-light) BSDF evaluator. Mirrors the per-light BRDF math in the
glTF Sample Renderer (`pbr.frag` / `brdf.glsl` / `punctual.glsl`) so raster shaders can match the
reference renderer pixel-for-pixel for `KHR_lights_punctual` lights (directional, point, spot).

This is the **direct lighting** counterpart to `pbr_ibl_eval.h.slang` (indirect / IBL). Together
they cover the rendering equation's two halves for the rasterizer:

  L_o = ∫ f(ωi, ωo) Li(ωi) cosθi dωi
        ↑ direct (this header) ↑   ↑ indirect (pbr_ibl_eval.h.slang) ↑

The expected pipeline per analytic light:
  GltfLight → light_contrib.h.slang::singleLightContribution(...) → LightContrib
                                                                       ↓
                                bsdfEvaluatePunctualKhronos(mat, V, pointToLight, intensity, ...)
                                                                       ↓
                                                                  shaded color

Why a dedicated evaluator (and not `bsdf_functions.h.slang::bsdfEvaluate`):
  * Fresnel is evaluated at the half-vector (`abs(VdotH)`) with vec3 F0, not at a VdotN cosine
    approximation with scalar Fresnel. This is the canonical microfacet form (Walter 2007) and
    is required for colored metals (e.g. gold whitens at grazing) and KHR_materials_specular's
    colored specular.
  * Smith-G is the height-correlated `V_GGX` (Heitz 2014), not uncorrelated `G1*G2`.
  * Sheen energy compensation uses `min(LUT(NdotV), LUT(NdotL))` from Estevez-Kulla 2017.
  * Layered material composite uses `mix(dielectric, metal, metallic) * sheenScale + sheen`,
    then `mix(., clearcoat_brdf, clearcoatFactor * clearcoatFresnel)` -- the glTF Sample
    Renderer's reference layering.

The path tracer's `bsdfEvaluate` (stochastic single-lobe picker) and `bsdfSample` keep the
older MC-friendly approximations and are paired so MIS PDFs cancel correctly; bringing those
to Khronos parity is a separate refactor.

The shared microfacet primitives (`D_GGX`, `V_GGX`, `D_Charlie`, `V_Sheen`, `BRDF_lambertian`,
`khronosDielectricF0`, `khronosClearcoatFresnel`, `albedoSheenScalingLUT`,
`layeredCompositeKhronos`) live in `pbr_ggx_microfacet.h.slang`'s "Khronos analytic helpers"
section. This header only defines per-lobe BRDFs that compose those primitives in
direct-lighting-specific ways, plus the per-light evaluator that ties them together.
-------------------------------------------------------------------------------------------------*/

#include "bsdf_functions.h.slang"      // applyIorToRoughness, applyVolumeAttenuation
#include "constants.h.slang"
#include "functions.h.slang"
#include "pbr_ggx_microfacet.h.slang"  // Khronos analytic helpers (D_GGX, V_GGX, ...) + MC primitives
#include "pbr_material_types.h.slang"


// =================================================================================================
// glTF Sample Renderer per-lobe BRDFs (compose primitives from pbr_ggx_microfacet.h.slang)
// =================================================================================================

// Isotropic GGX-Smith: D * V (Fresnel applied separately by the caller via Schlick at VdotH).
float3 BRDF_specularGGX(float alphaRoughness, float NdotL, float NdotV, float NdotH)
{
  float Vis = V_GGX(NdotL, NdotV, alphaRoughness);
  float D   = D_GGX(NdotH, alphaRoughness);
  return float3(Vis * D);
}

// Anisotropic GGX-Smith. `at` and `ab` are the perpendicular alphas (already accounting for
// KHR_materials_anisotropy stretching from `evaluateMaterial`).
float3 BRDF_specularGGXAnisotropy(float at, float ab, float3 N, float3 V, float3 L, float3 H, float3 T, float3 B)
{
  float NdotL = clamp(dot(N, L), 0.0F, 1.0F);
  float NdotH = clamp(dot(N, H), 0.001F, 1.0F);
  float NdotV = dot(N, V);
  float Vis   = V_GGX_anisotropic(NdotL, NdotV, dot(B, V), dot(T, V), dot(T, L), dot(B, L), at, ab);
  float D     = D_GGX_anisotropic(NdotH, dot(T, H), dot(B, H), at, ab);
  return float3(Vis * D);
}

// Charlie sheen BRDF (full per-light evaluation; no LUT).
float3 BRDF_specularSheen(float3 sheenColor, float sheenRoughness, float NdotL, float NdotV, float NdotH)
{
  float D = D_Charlie(sheenRoughness, NdotH);
  float V = V_Sheen(NdotL, NdotV, sheenRoughness);
  return sheenColor * D * V;
}

// Clearcoat radiance: GGX-Smith on the clearcoat normal, no Fresnel (caller multiplies by
// clearcoatFresnel during the layered composite).
float3 getPunctualRadianceClearCoat(float3 clearcoatNormal, float3 V, float3 L, float3 H, float clearcoatRoughness)
{
  float NdotL = clampedDot(clearcoatNormal, L);
  float NdotV = clampedDot(clearcoatNormal, V);
  float NdotH = clampedDot(clearcoatNormal, H);
  float aR    = clearcoatRoughness * clearcoatRoughness;
  return float3(NdotL * V_GGX(NdotL, NdotV, aR) * D_GGX(NdotH, aR));
}

// KHR_materials_transmission punctual BTDF. Mirror-reflects the light direction across the
// surface normal so a single GGX lobe approximates the refractive transmission.
// Reference: `punctual.glsl::getPunctualRadianceTransmission`.
float3 getPunctualRadianceTransmission(float3 N, float3 V, float3 L, float alphaRoughness, float3 baseColor, float ior)
{
  float  transmissionRoughness = applyIorToRoughness(alphaRoughness, ior);
  float3 lMirror               = normalize(L + 2.0F * N * dot(-L, N));
  float3 H                     = normalize(lMirror + V);
  float  NdotH                 = clamp(dot(N, H), 0.0F, 1.0F);
  float  NdotL                 = clamp(dot(N, lMirror), 0.0F, 1.0F);
  float  NdotV                 = clamp(dot(N, V), 0.0F, 1.0F);
  float  D                     = D_GGX(NdotH, transmissionRoughness);
  float  Vis                   = V_GGX(NdotL, NdotV, transmissionRoughness);
  return baseColor * D * Vis;
}


// =================================================================================================
// Khronos punctual-light evaluator
// =================================================================================================

/*-------------------------------------------------------------------------------------------------
# Function `_bsdfEvaluatePunctualKhronosCore` (private)
> Shared body of the public `bsdfEvaluatePunctualKhronos` overloads.

Computes everything except sheen; the caller passes precomputed `sheenScale` and `l_sheen`
(which the with-sheen overload computes from the Charlie LUT, and the no-sheen overload sets
to `1.0F` / `float3(0.0F)`). This mirrors the factoring used in `pbr_ibl_eval.h.slang`.

Two intentional deviations from the Khronos reference (`pbr.frag`):
1. `lightIntensity` is NOT recomputed at the volume-shifted `pointToLight` for the front-surface
   reflection lobes (specular metal, specular dielectric, sheen, clearcoat). The reference
   recomputes it (`pbr.frag:316`), which slightly mis-attenuates close point lights against
   thick transmissive surfaces -- specular reflection happens at the front surface, not at the
   refraction exit. Only the BTDF (specular-transmission) volume attenuation uses the
   transmission ray length.
2. The dead `tangentRoughness` line in the reference's anisotropy IBL path has no analog here
   (we only do the punctual path; this is informational).
-------------------------------------------------------------------------------------------------*/
float3 _bsdfEvaluatePunctualKhronosCore(PbrMaterial mat,
                                        float3      V,
                                        float3      pointToLight,
                                        float3      lightIntensity,
                                        float3      transmissionRay,
                                        float       diffuseTransmissionThickness,
                                        float       sheenScale,
                                        float3      l_sheen)
{
  float3 N     = mat.N;
  float3 L     = normalize(pointToLight);
  float  NdotV = clampedDot(N, V);

  // alphaRoughness for the isotropic path. For the anisotropic path we hand `roughness.x` and
  // `roughness.y` directly to BRDF_specularGGXAnisotropy as `at, ab` (which already encode
  // KHR_materials_anisotropy stretching from `evaluateMaterial`).
  float alphaRoughness = mat.roughness.x;
  bool  isAnisotropic  = abs(mat.roughness.x - mat.roughness.y) > 1e-6F;

  // Effective IOR-derived dielectric F0 (KHR_materials_specular layered on top).
  float  ior            = mat.ior2 / max(mat.ior1, 1e-5F);
  float3 f0_dielectric  = khronosDielectricF0(mat);
  float  specularWeight = mat.specular;
  float3 f90_dielectric = float3(specularWeight);

  // Iridescence Fresnel evaluated at NdotV (matches `pbr.frag:135-136`). Same factor used for
  // metal and dielectric; `thin_film_factor` re-uses the dielectric IOR as the substrate IOR
  // (consistent with our IBL composer and `computeLobeWeights`).
  float3 iridescenceFresnel = float3(0.0F);
  bool   hasIridescence     = mat.iridescence > 0.0F && mat.iridescenceThickness > 0.0F;
  if(hasIridescence)
  {
    iridescenceFresnel = thin_film_factor(mat.iridescenceThickness, mat.iridescenceIor, mat.ior2, mat.ior1, NdotV);
  }

  // Clearcoat Fresnel at the clearcoat normal (Schlick at NcDotV). Used in the final layered
  // composite as `mix(., clearcoat_brdf, clearcoatFactor * clearcoatFresnel)` and to attenuate
  // the front-layer composite by `(1 - clearcoatFactor * clearcoatFresnel)`.
  float3 clearcoatFresnel = (mat.clearcoat > 0.0F) ? khronosClearcoatFresnel(mat, V) : float3(0.0F);

  // ----- Front-surface direction setup -----
  float  NdotL    = clampedDot(N, L);
  float3 H        = normalize(L + V);
  float  NdotH    = clampedDot(N, H);
  float  VdotH    = clampedDot(V, H);
  float  absVdotH = abs(VdotH);

  // Front-surface dielectric / metal Fresnels (Schlick at the half vector). The dielectric
  // Fresnel may be recomputed by the diffuse-transmission-back-light branch below; the
  // iridescence override is applied later at the layered-composite stage to mirror
  // `pbr.frag:329-332` exactly.
  float3 dielectric_fresnel = schlickFresnel(f0_dielectric * specularWeight, f90_dielectric, absVdotH);
  float3 metal_fresnel      = schlickFresnel(mat.baseColor, float3(1.0F), absVdotH);

  // ----- Diffuse (Lambertian) and diffuse transmission -----
  float3 l_diffuse = lightIntensity * NdotL * BRDF_lambertian(mat.baseColor);
  l_diffuse *= (1.0F - mat.diffuseTransmissionFactor);

  if(mat.diffuseTransmissionFactor > 0.0F && dot(N, L) < 0.0F)
  {
    // Back-hemisphere Lambertian.
    float  diffuseNdotL   = clampedDot(-N, L);
    float3 l_diffuse_btdf = lightIntensity * diffuseNdotL * BRDF_lambertian(mat.diffuseTransmissionColor);

    // Recompute dielectric Fresnel at the mirror-bounced light's half vector. This represents
    // the energy that does NOT pass through the front surface to reach the diffuse-trans path.
    float3 lMirror      = normalize(L + 2.0F * N * dot(-L, N));
    float3 Hmirror      = normalize(lMirror + V);
    float  diffuseVdotH = clampedDot(V, Hmirror);
    dielectric_fresnel  = schlickFresnel(f0_dielectric * specularWeight, f90_dielectric, abs(diffuseVdotH));

    if(diffuseTransmissionThickness > 0.0F && mat.attenuationDistance > 0.0F)
    {
      l_diffuse_btdf = applyVolumeAttenuation(l_diffuse_btdf, diffuseTransmissionThickness,
                                              mat.attenuationColor, mat.attenuationDistance);
    }
    l_diffuse += l_diffuse_btdf * mat.diffuseTransmissionFactor;
  }

  // ----- Specular transmission BTDF (KHR_materials_transmission) -----
  // The BTDF is the only lobe whose L is shifted by `transmissionRay`. Reflection lobes keep
  // the original L (and original lightIntensity) -- this is the deviation from the reference
  // that fixes its mis-attenuation of close point lights against thick transmissive surfaces.
  if(mat.transmission > 0.0F)
  {
    float3 btdfL            = normalize(pointToLight - transmissionRay);
    float3 transmittedLight = lightIntensity * getPunctualRadianceTransmission(N, V, btdfL, alphaRoughness, mat.baseColor, ior);

    if(mat.thickness > 0.0F && mat.attenuationDistance > 0.0F && length(transmissionRay) > 0.0F)
    {
      transmittedLight = applyVolumeAttenuation(transmittedLight, length(transmissionRay), mat.attenuationColor, mat.attenuationDistance);
    }
    l_diffuse = lerp(l_diffuse, transmittedLight, mat.transmission);
  }

  // ----- Specular reflection (metal + dielectric, isotropic or anisotropic) -----
  float3 l_specular = isAnisotropic
                          ? lightIntensity * NdotL * BRDF_specularGGXAnisotropy(mat.roughness.x, mat.roughness.y, N, V, L, H, mat.T, mat.B)
                          : lightIntensity * NdotL * BRDF_specularGGX(alphaRoughness, NdotL, NdotV, NdotH);

  float3 l_metal_brdf      = metal_fresnel * l_specular;
  float3 l_dielectric_brdf = lerp(l_diffuse, l_specular, dielectric_fresnel);

  // Iridescence override (`pbr.frag:329-332`). Note the asymmetry: the metal layer uses a vec3
  // `lerp`, the dielectric layer uses `mix_rgb` (max-component attenuation on the diffuse base).
  if(hasIridescence)
  {
    l_metal_brdf      = lerp(l_metal_brdf, l_specular * iridescenceFresnel, mat.iridescence);
    l_dielectric_brdf = lerp(l_dielectric_brdf, mix_rgb(l_diffuse, l_specular, iridescenceFresnel), mat.iridescence);
  }

  // ----- Clearcoat reflection -----
  float3 l_clearcoat_brdf = (mat.clearcoat > 0.0F)
                                ? lightIntensity * getPunctualRadianceClearCoat(mat.Nc, V, L, H, mat.clearcoatRoughness)
                                : float3(0.0F);

  // ----- Layered composite (matches `pbr.frag:345-348`) -----
  return layeredCompositeKhronos(l_dielectric_brdf, l_metal_brdf, mat.metallic, l_sheen, sheenScale,
                                 l_clearcoat_brdf, mat.clearcoat, clearcoatFresnel);
}

/*-------------------------------------------------------------------------------------------------
# Function `bsdfEvaluatePunctualKhronos`
> Reference-faithful glTF Sample Renderer per-light BSDF evaluator.

Returns the per-light contribution color (already multiplied by `lightIntensity`). Mirrors the
per-light branch of `pbr.frag` (`for (int i = 0; i < LIGHT_COUNT; ++i)` block).

## Inputs
- `mat`:                          surface material at the front hit point.
- `V`:                            normalized view direction (toward camera).
- `pointToLight`:                 unnormalized vector from the surface to the light. Magnitude
                                  is the light distance for point/spot, conventionally a unit
                                  vector for directional lights (`-light.direction`). The
                                  function normalizes for the front-surface lobes and applies
                                  the transmission-ray shift internally for the BTDF lobe to
                                  match `pbr.frag:303` semantics.
- `lightIntensity`:               pre-attenuated light radiance (e.g.,
                                  `singleLightContribution(...).intensity`).
- `transmissionRay`:              world-space refraction offset for KHR_materials_transmission.
                                  Only the BTDF L gets shifted by this; pass `float3(0)` if no
                                  transmission lobe is in flight.
- `diffuseTransmissionThickness`: world-space thickness for KHR_materials_diffuse_transmission's
                                  volume attenuation (typically `mat.thickness *
                                  model_scale_avg`). Pass `0.0` to disable.
- `sheenLut`:                     Charlie/sheen directional-albedo LUT (`.r` channel).
-------------------------------------------------------------------------------------------------*/
float3 bsdfEvaluatePunctualKhronos(PbrMaterial mat,
                                   float3      V,
                                   float3      pointToLight,
                                   float3      lightIntensity,
                                   float3      transmissionRay,
                                   float       diffuseTransmissionThickness,
                                   Sampler2D   sheenLut)
{
  // Sheen (KHR_materials_sheen): per-light Charlie radiance + bidirectional energy-compensation
  // scaling via the directional-albedo LUT (`min(LUT(NdotV), LUT(NdotL))`). The compiler is
  // expected to CSE NdotV/NdotL/H/NdotH with the matching computations inside `_core`.
  float3 l_sheen     = float3(0.0F);
  float  sheenScale  = 1.0F;
  if(max3(mat.sheenColor) > 0.0F)
  {
    float3 N     = mat.N;
    float3 L     = normalize(pointToLight);
    float  NdotV = clampedDot(N, V);
    float  NdotL = clampedDot(N, L);
    float3 H     = normalize(L + V);
    float  NdotH = clampedDot(N, H);

    l_sheen = lightIntensity * NdotL * BRDF_specularSheen(mat.sheenColor, mat.sheenRoughness, NdotL, NdotV, NdotH);

    float maxSheen = max3(mat.sheenColor);
    sheenScale     = min(1.0F - maxSheen * albedoSheenScalingLUT(sheenLut, NdotV, mat.sheenRoughness),
                         1.0F - maxSheen * albedoSheenScalingLUT(sheenLut, NdotL, mat.sheenRoughness));
  }

  return _bsdfEvaluatePunctualKhronosCore(mat, V, pointToLight, lightIntensity, transmissionRay,
                                          diffuseTransmissionThickness, sheenScale, l_sheen);
}

/*-------------------------------------------------------------------------------------------------
# Function `bsdfEvaluatePunctualKhronos` (no-sheen overload)
> Same as the full evaluator but skips the KHR_materials_sheen lobe entirely.

Use when the scene guarantees no material carries `KHR_materials_sheen`. Avoids the need to
bake/bind the Charlie sheen LUT. If called on a material that DOES have sheen set, the sheen
contribution is silently dropped -- by design; the caller is asserting the scene has no sheen.
-------------------------------------------------------------------------------------------------*/
float3 bsdfEvaluatePunctualKhronos(PbrMaterial mat,
                                   float3      V,
                                   float3      pointToLight,
                                   float3      lightIntensity,
                                   float3      transmissionRay,
                                   float       diffuseTransmissionThickness)
{
  return _bsdfEvaluatePunctualKhronosCore(mat, V, pointToLight, lightIntensity, transmissionRay,
                                          diffuseTransmissionThickness, /*sheenScale=*/ 1.0F,
                                          /*l_sheen=*/ float3(0.0F));
}

#endif  // PBR_DIRECT_EVAL_H
