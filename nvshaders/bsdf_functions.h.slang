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

#ifndef NVVKHL_BSDF_FUNCTIONS_H
#define NVVKHL_BSDF_FUNCTIONS_H 1

/*-------------------------------------------------------------------------------------------------
> Implements bidirectional scattering distribution functions (BSDFs) for
> physically-based rendering systems.

To use this code, include `bsdf_functions.h`. When a ray hits a surface:
* Create a `PbrMaterial` describing the material at that point
* Call `bsdfEvaluate()` to evaluate the scattering from one direction to
  another, or `bsdfSample()` to choose the next ray in the light path.
  See these functions' documentation for more information on their parameters
  and return values.

See the vk_gltf_renderer and vk_mini_samples/gltf_raytrace samples for examples
where these functions are used in a Monte Carlo path tracer.

The returned BSDF values and weights have the cosine term from the rendering
equation included; e.g. the Lambert lobe returns `max(0.0f, cos(N, k1)) / pi`.

Advanced users can use some of the other functions in this file to evaluate
and sample individual lobes, or subsets of lobes.

This file also provides `bsdfEvaluateSimple()` and `bsdfSampleSimple()`, which
implement a simpler and faster, though less fully-featured BSDF model. The
simple model only has diffuse, specular, and metallic lobes, while the full
model includes diffuse, transmission, specular, metal, sheen, and clearcoat
lobes (plus support for most glTF extensions).

Since GLSL doesn't have a distinction between public and private functions,
only functions that are part of the "public API" will have annotations to
appear in README.md.

This file (and files that it depends on) should be a GLSL/C++ polyglot, and
work so long as GLM has been included:

```cpp
#undef GLM_FORCE_XYZW_ONLY
#define GLM_FORCE_SWIZZLE
#include <glm/glm.hpp>
using namespace glm;

#include <nvshaders/bsdf_functions.h>
```

## Technical Notes

NVVK's BSDFs are based on [the glTF 2.0 specification](https://github.com/KhronosGroup/glTF) and
[the NVIDIA MDL SDK's BSDF implementations](https://github.com/NVIDIA/MDL-SDK/blob/203d5140b1dee89de17b26e828c4333571878629/src/mdl/jit/libbsdf/libbsdf.cpp). 

The largest divergence from the above is that NVVK's BSDF model uses a
Fresnel term that depends only on the view and normal vectors, instead of the
half vector. This allows it to compute weights for lobes independently, while
BSDF code would normally need to sample a half vector for layer `i` to
determine the Fresnel weight for layer `i+1`. This can result in slightly
different glossy/diffuse blend weights (e.g. slightly differently shaped
highlights on leather surfaces).

All lobes are energy conserving (their integral over the sphere is at most 1)
and probability distribution functions (PDFs) integrate to 1 (except for areas
where the sampled direction results in an absorption event).

All lobes use single-scattering BSDFs. Multiple-scattering lobes are a
potential future improvement.

Most lobes use GGX normal distribution functions (NDFs) and the uncorrelated
Smith shadowing-maskiung function, except for the diffuse lobe (Lambert BRDF)
and the sheen lobe ([Conty and Kulla's "Charlie" sheen](https://blog.selfshadow.com/publications/s2017-shading-course/#course_content)
and a V-cavity shadowing-masking function).

-------------------------------------------------------------------------------------------------*/

#include "slang_types.h"  // NVSHADERS_INOUT_TYPE

#include "functions.h.slang"           // cosineSampleHemisphere
#include "pbr_ggx_microfacet.h.slang"  // brdfLambertian, ..
#include "bsdf_types.h.slang"
#include "pbr_material_types.h.slang"


#ifdef __cplusplus
#define ARRAY_TYPE(T, N, name) std::array<T, N> name
#elif defined(GLSL)
#define ARRAY_TYPE(T, N, name) T name[N]
#endif

#ifndef ARRAY_TYPE
#define ARRAY_TYPE(T, N, name) T name[N]
#endif


/*-------------------------------------------------------------------------------------------------
# Function `absorptionCoefficient`
> Returns the absorption coefficient of the material.
-------------------------------------------------------------------------------------------------*/
float3 absorptionCoefficient(PbrMaterial mat)
{
  return -log(max(mat.attenuationColor, float3(0.001))) / max(mat.attenuationDistance, 0.001);
}

/*-------------------------------------------------------------------------------------------------
# `LOBE_*` Defines
> Indices for lobe weights returned by `computeLobeWeights()`.
-------------------------------------------------------------------------------------------------*/
#define LOBE_DIFFUSE_REFLECTION 0
#define LOBE_SPECULAR_TRANSMISSION 1
#define LOBE_SPECULAR_REFLECTION 2
#define LOBE_METAL_REFLECTION 3
#define LOBE_SHEEN_REFLECTION 4
#define LOBE_CLEARCOAT_REFLECTION 5
#define LOBE_DIFFUSE_TRANSMISSION 6
#define LOBE_COUNT 7

// The Fresnel factor depends on the cosine between the view vector k1 and the
// half vector, H = normalize(k1 + k2). But during sampling, we don't have k2
// until we sample a microfacet. So instead, we approximate it.
// For a mirror surface, we have H == N. For a perfectly diffuse surface, k2
// is sampled in a cosine distribution around N, so H ~ normalize(k1 + N).
// We ad-hoc interpolate between them using the roughness.
float fresnelCosineApproximation(float VdotN, float roughness)
{
  return lerp(VdotN, sqrt(0.5F + 0.5F * VdotN), sqrt(roughness));
}

/*-------------------------------------------------------------------------------------------------
# Function `computeLobeWeights`
>Calculates the weights of the individual lobes inside the standard PBR material.

Returns an array which can be indexed using the `LOBE_*` defines. This can be
used to perform your own lobe sampling.

Note that `tint` will be changed if the material has iridescence! (It's
convenient to compute the iridescence factor here.) This means you should avoid
passing a material parameter to the `tint` field. Make a temporary instead:
```glsl
float3 tint = mat.baseColor;
float[LOBE_COUNT] weights = computeLobeWeights(mat, dot(k1, mat.N), tint);
```
-------------------------------------------------------------------------------------------------*/
ARRAY_TYPE(float, LOBE_COUNT, ) computeLobeWeights(PbrMaterial mat, float VdotN, NVSHADERS_INOUT_TYPE(float3) tint)
{
  float frCoat = 0.0F;
  if(mat.clearcoat > 0.0f)
  {
    float frCosineClearcoat = fresnelCosineApproximation(VdotN, mat.clearcoatRoughness);
    frCoat                  = mat.clearcoat * ior_fresnel(1.5f / mat.ior1, frCosineClearcoat);
  }

  // This Fresnel value defines the weighting between dielectric specular reflection and
  // the base dielectric BSDFs (diffuse reflection and specular transmission).
  float frDielectric = 0;
  if(mat.specular > 0)
  {
    float frCosineDielectric = fresnelCosineApproximation(VdotN, (mat.roughness.x + mat.roughness.y) * 0.5F);
    frDielectric             = ior_fresnel(mat.ior2 / mat.ior1, frCosineDielectric);
    frDielectric *= mat.specular;
  }

  // Estimate the iridescence Fresnel factor with the angle to the normal, and
  // blend it in. That's good enough for specular reflections.
  if(mat.iridescence > 0.0f)
  {
    // When there is iridescence enabled, use the maximum of the estimated iridescence factor. (Estimated with VdotN, no half-vector H here.)
    // With the thinfilm decision this handles the mix between non-iridescence and iridescence strength automatically.
    float3 frIridescence = thin_film_factor(mat.iridescenceThickness, mat.iridescenceIor, mat.ior2, mat.ior1, VdotN);
    frDielectric = lerp(frDielectric, max(frIridescence.x, max(frIridescence.y, frIridescence.z)), mat.iridescence);
    // Modulate the dielectric base lobe (diffuse, transmission) colors by the inverse of the iridescence factor,
    // though use the maximum component to not actually generate inverse colors.
    tint = mix_rgb(tint, mat.specularColor, frIridescence * mat.iridescence);
  }

  float sheen = 0.0f;
  if((mat.sheenColor.r != 0.0F || mat.sheenColor.g != 0.0F || mat.sheenColor.b != 0.0F))
  {
    sheen = pow(1.0f - abs(VdotN), mat.sheenRoughness);  // * luminance(mat.sheenColor);
    sheen = sheen / (sheen + 0.5F);
  }

  /*
  Our model consists of 6 layers. Each layer handles a fraction
  of light that didn't hit any layers above it:

    - Clearcoat       : clearcoat * schlickFresnel(1.5, VdotN)
    - Sheen           : sheen
    - Metal           : metallic
    - Specular        : specular * schlickFresnel(ior, VdotN)
    - Transmission    : transmission }
    - Diffuse         : 1.0          } These two are technically parallel layers
  */

  ARRAY_TYPE(float, LOBE_COUNT, weightLobe);
  weightLobe[LOBE_CLEARCOAT_REFLECTION]  = 0;
  weightLobe[LOBE_SHEEN_REFLECTION]      = 0;
  weightLobe[LOBE_METAL_REFLECTION]      = 0;
  weightLobe[LOBE_SPECULAR_REFLECTION]   = 0;
  weightLobe[LOBE_SPECULAR_TRANSMISSION] = 0;
  weightLobe[LOBE_DIFFUSE_TRANSMISSION]  = 0;
  weightLobe[LOBE_DIFFUSE_REFLECTION]    = 0;

  float weightBase = 1.0F;

  weightLobe[LOBE_CLEARCOAT_REFLECTION] = frCoat;  // BRDF clearcoat reflection (GGX-Smith)
  weightBase *= 1.0f - frCoat;

  weightLobe[LOBE_SHEEN_REFLECTION] = weightBase * sheen;  // BRDF sheen reflection (Lambert)
  weightBase *= 1.0f - sheen;

  weightLobe[LOBE_METAL_REFLECTION] = weightBase * mat.metallic;  // BRDF metal (GGX-Smith)
  weightBase *= 1.0f - mat.metallic;

  weightLobe[LOBE_SPECULAR_REFLECTION] = weightBase * frDielectric;  // BRDF dielectric specular reflection (GGX-Smith)
  weightBase *= 1.0f - frDielectric;

  weightLobe[LOBE_SPECULAR_TRANSMISSION] = weightBase * mat.transmission;  // BTDF dielectric specular transmission (GGX-Smith)

  float remainingWeight                 = weightBase * (1.0f - mat.transmission);
  weightLobe[LOBE_DIFFUSE_TRANSMISSION] = remainingWeight * mat.diffuseTransmissionFactor;
  weightLobe[LOBE_DIFFUSE_REFLECTION]   = remainingWeight * (1.0f - mat.diffuseTransmissionFactor);

  return weightLobe;
}

/*-------------------------------------------------------------------------------------------------
# Function `findLobe`
> Calculates the weights of the individual lobes inside the standard PBR material
> and randomly selects one.
-------------------------------------------------------------------------------------------------*/
int findLobe(PbrMaterial mat, float VdotN, float rndVal, NVSHADERS_INOUT_TYPE(float3) tint)
{
  ARRAY_TYPE(float, LOBE_COUNT, weightLobe) = computeLobeWeights(mat, VdotN, tint);

  int   lobe   = LOBE_COUNT;
  float weight = 0.0f;
  // Stops when lobe reaches 0! No need to look at weightLobe[0] since we know
  // light will always scatter on one lobe.
  while(--lobe > 0)
  {
    weight += weightLobe[lobe];
    if(rndVal < weight)
    {
      break;  // Sample and evaluate this lobe!
    }
  }

  return lobe;
}

// Evaluates a diffuse lobe.
void brdf_diffuse_eval(NVSHADERS_INOUT_TYPE(BsdfEvaluateData) data, PbrMaterial mat, float3 tint)
{
  // If the incoming light direction is on the backside, there is nothing to evaluate for a BRDF.
  // Note that the state normals have been flipped to the ray side by the caller.
  // Include edge-on (== 0.0f) as "no light" case.
  const float nk2 = dot(data.k2, mat.N);
  if(nk2 <= 0.0f) // if (backside)
  {
    return;  // absorb
  }
  data.pdf = nk2 * M_1_PI;

  // For a white Lambert material, the bxdf components match the evaluation pdf. (See MDL_renderer.)
  data.bsdf_diffuse = tint * data.pdf;
}

void brdf_diffuse_eval(NVSHADERS_INOUT_TYPE(BsdfEvaluateData) data, PbrMaterial mat)
{
  brdf_diffuse_eval(data, mat, mat.baseColor);
}

// Samples a diffuse lobe.
void brdf_diffuse_sample(NVSHADERS_INOUT_TYPE(BsdfSampleData) data, PbrMaterial mat, float3 tint)
{
  data.k2  = cosineSampleHemisphere(data.xi.x, data.xi.y);
  data.k2  = mat.T * data.k2.x + mat.B * data.k2.y + mat.N * data.k2.z;
  data.k2  = normalize(data.k2);
  data.pdf = dot(data.k2, mat.N) * M_1_PI;

  data.bsdf_over_pdf = tint;  // bsdf * dot(wi, normal) / pdf;
  data.event_type    = BSDF_EVENT_DIFFUSE_REFLECTION;
}

void brdf_diffuse_sample(NVSHADERS_INOUT_TYPE(BsdfSampleData) data, PbrMaterial mat)
{
  brdf_diffuse_sample(data, mat, mat.baseColor);
}

void brdf_diffuse_transmission_eval(NVSHADERS_INOUT_TYPE(BsdfEvaluateData) data, PbrMaterial mat, float3 tint)
{
  // If the incoming light direction is on the same side as the surface normal,
  // there is no diffuse transmission to evaluate. Absorb light in this case.
  float nk2 = dot(data.k2, mat.N);
  if(-nk2 <= 0.0f)
  {
    return;  // absorb
  }

  data.pdf = max(0.0f, -nk2 * M_1_PI);

  // For a white Lambert material, the bxdf components match the evaluation pdf. (See MDL_renderer.)
  data.bsdf_diffuse = tint * data.pdf * mat.diffuseTransmissionColor;
}

void brdf_diffuse_transmission_sample(NVSHADERS_INOUT_TYPE(BsdfSampleData) data, PbrMaterial mat, float3 tint)
{
  // Sample a direction on the opposite hemisphere (relative to -mat.N).
  float3 sampledDir = cosineSampleHemisphere(data.xi.x, data.xi.y);
  data.k2           = mat.T * sampledDir.x + mat.B * sampledDir.y - mat.N * sampledDir.z;  // Flip the normal
  data.k2           = normalize(data.k2);

  // Compute the PDF for this direction.
  data.pdf = dot(data.k2, -mat.N) * M_1_PI;

  // Ensure energy conservation by considering the diffuse transmission tint.
  data.bsdf_over_pdf = tint * mat.diffuseTransmissionColor;
  data.event_type = BSDF_EVENT_DIFFUSE_TRANSMISSION;
}


// Evaluates a reflective lobe with a GGX NDF and an uncorrelated Smith
// scattering-masking function.
void brdf_ggx_smith_eval(NVSHADERS_INOUT_TYPE(BsdfEvaluateData) data, PbrMaterial mat, const int lobe, float3 tint)
{
  const float nk1 = abs(dot(data.k1, mat.N));
  const float nk2 = abs(dot(data.k2, mat.N));

  // compute_half_vector() for scatter_reflect.
  const float3 h = normalize(data.k1 + data.k2);

  // Invalid for reflection / refraction?
  const float nh  = dot(mat.N, h);
  const float k1h = dot(data.k1, h);
  const float k2h = dot(data.k2, h);

  // nk1 and nh must not be 0.0f or state.pdf == NaN.
  if(nk1 <= 0.0f || nh <= 0.0f || k1h < 0.0f || k2h < 0.0f)
  {
    data.pdf         = 0.0f;
    data.bsdf_glossy = float3(0.0f);
    return;
  }

  // Compute BSDF and pdf.
  const float3 h0 = float3(dot(mat.T, h), dot(mat.B, h), nh);

  data.pdf = hvd_ggx_eval(1.0f / mat.roughness, h0);
  float G1;
  float G2;

  float G12;
  G12 = ggx_smith_shadow_mask(G1, G2, float3(dot(mat.T, data.k1), dot(mat.B, data.k1), nk1),
                              float3(dot(mat.T, data.k2), dot(mat.B, data.k2), nk2), mat.roughness);

  data.pdf *= 0.25f / (nk1 * nh);

  float3 bsdf = float3(G12 * data.pdf);

  data.pdf *= G1;

  if(mat.iridescence > 0.0f)
  {
    const float3 factor = thin_film_factor(mat.iridescenceThickness, mat.iridescenceIor, mat.ior2, mat.ior1, k1h);

    switch(lobe)
    {
      case LOBE_SPECULAR_REFLECTION:
        tint *= lerp(float3(1.f), factor, mat.iridescence);
        break;

      case LOBE_METAL_REFLECTION:
        tint = mix_rgb(tint, mat.specularColor, factor * mat.iridescence);
        break;
    }
  }

  // eval output: (glossy part of the) bsdf * dot(k2, normal)
  data.bsdf_glossy = bsdf * tint;
}

// Samples a reflective lobe with a GGX NDF and an uncorrelated Smith
// scattering-masking function.
void brdf_ggx_smith_sample(NVSHADERS_INOUT_TYPE(BsdfSampleData) data, PbrMaterial mat, const int lobe, float3 tint)
{
  // When the sampling returns eventType = BSDF_EVENT_ABSORB, the path ends inside the ray generation program.
  // Make sure the returned values are valid numbers when manipulating the PRD.
  data.bsdf_over_pdf = float3(0.0f);
  data.pdf           = 0.0f;

  // Transform to local coordinate system
  const float nk1 = dot(data.k1, mat.N);
  if(nk1 <= 0.0f)
  {
    data.event_type = BSDF_EVENT_ABSORB;
    return;
  }
  const float3 k10 = float3(dot(data.k1, mat.T), dot(data.k1, mat.B), nk1);

  // Sample half-vector, microfacet normal.
  const float3 h0 = hvd_ggx_sample_vndf(k10, mat.roughness, data.xi.xy);
  if(h0.z == 0.0f)
  {
    data.event_type = BSDF_EVENT_ABSORB;
    return;
  }

  // Transform to world
  const float3 h  = h0.x * mat.T + h0.y * mat.B + h0.z * mat.N;
  const float  kh = dot(data.k1, h);

  if(kh <= 0.0f)
  {
    data.event_type = BSDF_EVENT_ABSORB;
    return;
  }

  // BRDF: reflect
  data.k2 = (2.0f * kh) * h - data.k1;

  // Check if the resulting direction is on the correct side of the shading normal
  const float nk2 = dot(data.k2, mat.N);

  if(nk2 <= 0.0f)
  {
    data.event_type = BSDF_EVENT_ABSORB;
    return;
  }

  const float k2h = abs(dot(data.k2, h));

  float G1;
  float G2;

  float G12 = ggx_smith_shadow_mask(G1, G2, k10, float3(dot(data.k2, mat.T), dot(data.k2, mat.B), nk2), mat.roughness);

  if(G12 <= 0.0f)
  {
    data.event_type = BSDF_EVENT_ABSORB;
    return;
  }

  data.bsdf_over_pdf = float3(G2);
  data.event_type    = BSDF_EVENT_GLOSSY_REFLECTION;

  // Compute pdf
  data.pdf = hvd_ggx_eval(1.0f / mat.roughness, h0) * G1;
  data.pdf *= 0.25f / (nk1 * h0.z);

  if(mat.iridescence > 0.0f)
  {
    const float3 factor = thin_film_factor(mat.iridescenceThickness, mat.iridescenceIor, mat.ior2, mat.ior1, kh);

    switch(lobe)
    {
      case LOBE_SPECULAR_REFLECTION:
        tint *= lerp(float3(1.f), factor, mat.iridescence);
        break;

      case LOBE_METAL_REFLECTION:
        tint = mix_rgb(tint, mat.specularColor, factor * mat.iridescence);
        break;
    }
  }

  data.bsdf_over_pdf *= tint;
}

/*
 * In rare cases (mainly dispersion), we need to get an additional random
 * number. This function takes a random number and applies the hash function
 * from pcg_output_rxs_m_xs_32_32 to it.
 * Its quality hasn't been tested and it's not very principled, but it's better
 * than using a random number that's correlated with material sampling.
*/
float rerandomize(float v)
{
  uint word = asuint(v);
  word      = ((word >> ((word >> 28) + 4)) ^ word) * 277803737U;
  word      = (word >> 22) ^ word;
  return float(word) / asfloat(0x4f800000U);
}

/*-------------------------------------------------------------------------------------------------
# Function `computeDispersedIOR`
> Calculates the IOR at a given wavelength (in nanometers) given the base IOR
> and glTF dispersion factor.

See https://github.com/KhronosGroup/glTF/tree/0251c5c0cce8daec69bd54f29f891e3d0cdb52c8/extensions/2.0/Khronos/KHR_materials_dispersion.
-------------------------------------------------------------------------------------------------*/
float computeDispersedIOR(float base_ior, float dispersion, float wavelength_nm)
{
  // Since the glTF extension stores 20 / V_d
  float abbeNumber = 20.0F / dispersion;
  // Last equation of https://github.com/KhronosGroup/glTF/blob/main/extensions/2.0/Khronos/KHR_materials_dispersion
  return max(base_ior + (base_ior - 1.0F) * (523655.0F / (wavelength_nm * wavelength_nm) - 1.5168F) / abbeNumber, 1.0F);
}

/*-------------------------------------------------------------------------------------------------
# Function `wavelengthToRGB`
> Given a wavelength of light, returns an approximation to the linear RGB
> color of a D65 illuminant (sRGB whitepoint) sampled at a wavelength of `x`
> nanometers, using the CIE 2015 2-degree Standard Observer Color Matching Functions.

This is normalized so that
`sum(wavelengthToRGB(i), {i, WAVELENGTH_MIN, WAVELENGTH_MAX}) == float3(1.)`,
which means that the values it returns are usually low. You'll need to multiply
by an appropriate normalization factor if you're randomly sampling it.

The colors here are clamped to only positive sRGB values, in case renderers
have problems with colors with negative sRGB components (i.e. are valid
colors but are out-of-gamut).
-------------------------------------------------------------------------------------------------*/
#define WAVELENGTH_MIN 399.43862850585765F
#define WAVELENGTH_MAX 668.6617899434457F
float3 wavelengthToRGB(float x)
{
  // This uses a piecewise-linear approximation generated using the CIE 2015
  // 2-degree standard observer times the D65 illuminant, minimizing the L2
  // norm, then normalized to have an integral of 1.
  float3 rgb = float3(0.);
  if(399.43862850585765 < x)
  {
    if(x < 435.3450352446586)
      rgb.r = 2.6268757476158464e-05 * x + -0.010492756458829732;
    else if(x < 452.7741480943567)
      rgb.r = -5.383671438883332e-05 * x + 0.024380763013525125;
    else if(x < 550.5919453498173)
      rgb.r = 1.2536207000814165e-07 * x + -5.187018452935683e-05;
    else if(x < 600.8694441891222)
      rgb.r = 0.00032842519537482 * x + -0.18081111406184644;
    else if(x < 668.6617899434457)
      rgb.r = -0.0002438262071743009 * x + 0.16303726812428945;
  }
  if(467.41924217251835 < x)
  {
    if(x < 532.3927928594046)
      rgb.g = 0.00020126149345609334 * x + -0.0940734947497564;
    else if(x < 552.5312202450474)
      rgb.g = -4.3718474429905034e-05 * x + 0.03635207454767751;
    else if(x < 605.5304635656746)
      rgb.g = -0.00023012125757884968 * x + 0.13934543177803685;
  }
  if(400.68666327204835 < x)
  {
    if(x < 447.59688835108466)
      rgb.b = 0.00042519082480799777 * x + -0.1703682928462067;
    else if(x < 501.2110070697423)
      rgb.b = -0.00037202508909921054 * x + 0.18646306956262593;
  }
  return rgb;
}


// Evaluates the transmission lobe.
void btdf_ggx_smith_eval(NVSHADERS_INOUT_TYPE(BsdfEvaluateData) data, PbrMaterial mat, float3 tint)
{
  bool isThinWalled = (mat.isThinWalled);

  float2 ior = float2(mat.ior1, mat.ior2);
  if(mat.dispersion > 0.0f)
  {
    // Randomly choose a wavelength; for now, uniformly choose from 399-669 nm.
    float wavelength = lerp(WAVELENGTH_MIN, WAVELENGTH_MAX, rerandomize(data.xi.z));
    ior.x            = computeDispersedIOR(ior.x, mat.dispersion, wavelength);
    tint *= (WAVELENGTH_MAX - WAVELENGTH_MIN) * wavelengthToRGB(wavelength);
  }

  const float nk1 = abs(dot(data.k1, mat.N));
  float nk2       = dot(data.k2, mat.N);

  // BRDF or BTDF eval?
  // If the incoming light direction is on the backface.
  // Do NOT include edge-on (== 0.0f) as backside here to take the reflection path.
  const bool backside = (nk2 < 0.0f);
  nk2 = abs(nk2);

  const float3 h = compute_half_vector(data.k1, data.k2, mat.N, ior, nk2, backside, isThinWalled);

  // Invalid for reflection / refraction?
  const float nh  = dot(mat.N, h);
  const float k1h = dot(data.k1, h);
  const float k2h = dot(data.k2, h) * (backside ? -1.0f : 1.0f);

  // nk1 and nh must not be 0.0f or state.pdf == NaN.
  if(nk1 <= 0.0f || nh <= 0.0f || k1h < 0.0f || k2h < 0.0f)
  {
    data.pdf         = 0.0f;  // absorb
    data.bsdf_glossy = float3(0.0f);
    return;
  }

  float fr;

  if(!backside)
  {
    // For scatter_transmit: Only allow TIR with BRDF eval.
    if(!isTIR(ior, k1h))
    {
      data.pdf         = 0.0f;  // absorb
      data.bsdf_glossy = float3(0.0f);
      return;
    }
    else
    {
      fr = 1.0f;
    }
  }
  else
  {
    fr = 0.0f;
  }

  // Compute BSDF and pdf
  const float3 h0 = float3(dot(mat.T, h), dot(mat.B, h), nh);
  data.pdf        = hvd_ggx_eval(1.0f / mat.roughness, h0);

  float G1;
  float G2;
  float G12 = ggx_smith_shadow_mask(G1, G2, float3(dot(mat.T, data.k1), dot(mat.B, data.k1), nk1),
                                    float3(dot(mat.T, data.k2), dot(mat.B, data.k2), nk2), mat.roughness);

  if(!isThinWalled && backside)  // Refraction?
  {
    // Refraction pdf and BTDF
    const float tmp = k1h * ior.x - k2h * ior.y;

    data.pdf *= k1h * k2h / (nk1 * nh * tmp * tmp);
  }
  else
  {
    // Reflection pdf and BRDF (and pseudo-BTDF for thin-walled)
    data.pdf *= 0.25f / (nk1 * nh);
  }

  const float prob = (backside) ? 1.0f - fr : fr;

  const float3 bsdf = float3(prob * G12 * data.pdf);

  data.pdf *= prob * G1;

  // eval output: (glossy part of the) bsdf * dot(k2, normal)
  data.bsdf_glossy = bsdf * tint;
}


// Samples the transmission lobe.
void btdf_ggx_smith_sample(NVSHADERS_INOUT_TYPE(BsdfSampleData) data, PbrMaterial mat, float3 tint)
{
  bool isThinWalled = (mat.isThinWalled);

  // When the sampling returns eventType = BSDF_EVENT_ABSORB, the path ends inside the ray generation program.
  // Make sure the returned values are valid numbers when manipulating the PRD.
  data.bsdf_over_pdf = float3(0.0f);
  data.pdf           = 0.0f;

  float2 ior = float2(mat.ior1, mat.ior2);
  if(mat.dispersion > 0.0f)
  {
    // Randomly choose a wavelength; for now, uniformly choose from 361-716 nm.
    float wavelength = lerp(WAVELENGTH_MIN, WAVELENGTH_MAX, rerandomize(data.xi.z));
    ior.x            = computeDispersedIOR(ior.x, mat.dispersion, wavelength);
    tint *= (WAVELENGTH_MAX - WAVELENGTH_MIN) * wavelengthToRGB(wavelength);
  }

  const float nk1 = abs(dot(data.k1, mat.N));

  const float3 k10 = float3(dot(data.k1, mat.T), dot(data.k1, mat.B), nk1);

  // Sample half-vector, microfacet normal.
  const float3 h0 = hvd_ggx_sample_vndf(k10, mat.roughness, data.xi.xy);

  if(abs(h0.z) == 0.0f)
  {
    data.event_type = BSDF_EVENT_ABSORB;
    return;
  }

  // Transform to world
  const float3 h = h0.x * mat.T + h0.y * mat.B + h0.z * mat.N;

  const float kh = dot(data.k1, h);

  if(kh <= 0.0f)
  {
    data.event_type = BSDF_EVENT_ABSORB;
    return;
  }

  // Case scatter_transmit
  bool tir = false;
  if(isThinWalled)  // No refraction!
  {
    // pseudo-BTDF: flip a reflected reflection direction to the back side
    data.k2 = (2.0f * kh) * h - data.k1;
    data.k2 = normalize(data.k2 - 2.0f * mat.N * dot(data.k2, mat.N));
  }
  else
  {
    // BTDF: refract
    data.k2 = refract(data.k1, h, ior.x / ior.y, kh, tir);
  }

  data.bsdf_over_pdf = float3(1.0f);  // Was: (float3(1.0f) - fr) / prob; // PERF Always white with the original setup.
  data.event_type    = (tir) ? BSDF_EVENT_GLOSSY_REFLECTION : BSDF_EVENT_GLOSSY_TRANSMISSION;

  // Check if the resulting direction is on the correct side of the shading normal
  const float nk2 = dot(data.k2, mat.N) * (tir ? 1.0f : -1.0f);

  if(nk2 <= 0.0f || isnan(data.k2.x))
  {
    data.event_type = BSDF_EVENT_ABSORB;
    return;
  }


  const float k2h = abs(dot(data.k2, h));

  float G1;
  float G2;
  float G12 = ggx_smith_shadow_mask(G1, G2, k10, float3(dot(data.k2, mat.T), dot(data.k2, mat.B), nk2), mat.roughness);

  if(G12 <= 0.0f)
  {
    data.event_type = BSDF_EVENT_ABSORB;
    return;
  }

  data.bsdf_over_pdf *= G2;

  // Compute pdf
  data.pdf = hvd_ggx_eval(1.0f / mat.roughness, h0) * G1;  // * prob;


  if(!isThinWalled && (data.event_type == BSDF_EVENT_GLOSSY_TRANSMISSION))  // if (refraction)
  {
    const float tmp = kh * ior.x - k2h * ior.y;
    if(tmp > 0)
    {
      data.pdf *= kh * k2h / (nk1 * h0.z * tmp * tmp);
    }
  }
  else
  {
    data.pdf *= 0.25f / (nk1 * h0.z);
  }

  data.bsdf_over_pdf *= tint;
}


// Evaluates the sheen lobe.
void brdf_sheen_eval(NVSHADERS_INOUT_TYPE(BsdfEvaluateData) data, PbrMaterial mat)
{
  // BRDF or BTDF eval?
  // If the incoming light direction is on the backface.
  // Include edge-on (== 0.0f) as "no light" case.
  const float nk2      = dot(data.k2, mat.N);
  const bool  backside = (nk2 <= 0.0f);
  // Nothing to evaluate for given directions?
  if(backside)  // && scatter_reflect
  {
    return;  // absorb
  }

  const float nk1 = abs(dot(data.k1, mat.N));

  // compute_half_vector() for scatter_reflect.
  const float3 h = normalize(data.k1 + data.k2);

  // Invalid for reflection / refraction?
  const float nh  = dot(mat.N, h);
  const float k1h = dot(data.k1, h);
  const float k2h = dot(data.k2, h);

  // nk1 and nh must not be 0.0f or state.pdf == NaN.
  if(nk1 <= 0.0f || nh <= 0.0f || k1h < 0.0f || k2h < 0.0f)
  {
    return;  // absorb
  }

  const float invRoughness = 1.0f / (mat.sheenRoughness * mat.sheenRoughness);  // Perceptual roughness to alpha G.

  // Compute BSDF and pdf
  const float3 h0 = float3(dot(mat.T, h), dot(mat.B, h), nh);

  data.pdf = hvd_sheen_eval(invRoughness, h0.z);

  float G1;
  float G2;

  const float G12 = vcavities_shadow_mask(G1, G2, h0.z, float3(dot(mat.T, data.k1), dot(mat.B, data.k1), nk1), k1h,
                                          float3(dot(mat.T, data.k2), dot(mat.B, data.k2), nk2), k2h);
  data.pdf *= 0.25f / (nk1 * nh);

  const float3 bsdf = float3(G12 * data.pdf);

  data.pdf *= G1;

  // eval output: (glossy part of the) bsdf * dot(k2, normal)
  data.bsdf_glossy = bsdf * mat.sheenColor;
}


// Samples the sheen lobe.
void brdf_sheen_sample(NVSHADERS_INOUT_TYPE(BsdfSampleData) data, PbrMaterial mat)
{
  // When the sampling returns eventType = BSDF_EVENT_ABSORB, the path ends inside the ray generation program.
  // Make sure the returned values are valid numbers when manipulating the PRD.
  data.bsdf_over_pdf = float3(0.0f);
  data.pdf           = 0.0f;

  const float invRoughness = 1.0f / (mat.sheenRoughness * mat.sheenRoughness);  // Perceptual roughness to alpha G.

  const float nk1 = abs(dot(data.k1, mat.N));

  const float3 k10 = float3(dot(data.k1, mat.T), dot(data.k1, mat.B), nk1);

  float        xiFlip = data.xi.z;
  const float3 h0     = flip(hvd_sheen_sample(data.xi.xy, invRoughness), k10, xiFlip);

  if(abs(h0.z) == 0.0f)
  {
    data.event_type = BSDF_EVENT_ABSORB;
    return;
  }

  // Transform to world
  const float3 h = h0.x * mat.T + h0.y * mat.B + h0.z * mat.N;

  const float k1h = dot(data.k1, h);

  if(k1h <= 0.0f)
  {
    data.event_type = BSDF_EVENT_ABSORB;
    return;
  }

  // BRDF: reflect
  data.k2            = (2.0f * k1h) * h - data.k1;
  data.bsdf_over_pdf = float3(1.0f);  // PERF Always white with the original setup.
  data.event_type    = BSDF_EVENT_GLOSSY_REFLECTION;

  // Check if the resulting reflection direction is on the correct side of the shading normal.
  const float nk2 = dot(data.k2, mat.N);

  if(nk2 <= 0.0f)
  {
    data.event_type = BSDF_EVENT_ABSORB;
    return;
  }

  const float k2h = abs(dot(data.k2, h));

  float G1;
  float G2;

  const float G12 = vcavities_shadow_mask(G1, G2, h0.z, k10, k1h, float3(dot(data.k2, mat.T), dot(data.k2, mat.B), nk2), k2h);
  if(G12 <= 0.0f)
  {
    data.event_type = BSDF_EVENT_ABSORB;
    return;
  }

  data.bsdf_over_pdf *= G12 / G1;

  // Compute pdf.
  data.pdf = hvd_sheen_eval(invRoughness, h0.z) * G1;

  data.pdf *= 0.25f / (nk1 * h0.z);

  data.bsdf_over_pdf *= mat.sheenColor;
}


/*-------------------------------------------------------------------------------------------------
# Function `bsdfEvaluate`
> Evaluates the full BSDF model for the given material and set of directions.

You must provide `BsdfEvaluateData`'s `k1`, `k2`, and `xi` parameters.
(Evaluation is stochastic because this code randomly samples lobes depending on
`xi`; this is valid to do in a Monte Carlo path tracer.)

The diffuse lobe evaluation and the sum of the specular lobe evaluations
(including the cosine term from the rendering equation) will be returned in
`data.bsdf_diffuse` and `data.bsdf_glossy`. Additionally, the probability that
the sampling code will return this direction will be returned in `data.pdf`.
-------------------------------------------------------------------------------------------------*/
void bsdfEvaluate(NVSHADERS_INOUT_TYPE(BsdfEvaluateData) data, PbrMaterial mat)
{
  float3 tint       = mat.baseColor;
  float  VdotN      = dot(data.k1, mat.N);
  int    lobe       = findLobe(mat, VdotN, data.xi.z, tint);
  data.bsdf_diffuse = float3(0, 0, 0);
  data.bsdf_glossy  = float3(0, 0, 0);
  data.pdf          = 0.0;

  if(lobe == LOBE_DIFFUSE_REFLECTION)
  {
    brdf_diffuse_eval(data, mat, tint);
  }
  else if(lobe == LOBE_DIFFUSE_TRANSMISSION)
  {
    brdf_diffuse_transmission_eval(data, mat, tint);
  }
  else if(lobe == LOBE_SPECULAR_REFLECTION)
  {
    brdf_ggx_smith_eval(data, mat, LOBE_SPECULAR_REFLECTION, mat.specularColor);
  }
  else if(lobe == LOBE_SPECULAR_TRANSMISSION)
  {
    btdf_ggx_smith_eval(data, mat, tint);
  }
  else if(lobe == LOBE_METAL_REFLECTION)
  {
    brdf_ggx_smith_eval(data, mat, LOBE_METAL_REFLECTION, mat.baseColor);
  }
  else if(lobe == LOBE_CLEARCOAT_REFLECTION)
  {
    mat.roughness   = float2(mat.clearcoatRoughness * mat.clearcoatRoughness);
    mat.N           = mat.Nc;
    mat.iridescence = 0.0f;
    brdf_ggx_smith_eval(data, mat, LOBE_CLEARCOAT_REFLECTION, float3(1, 1, 1));
  }
  else if(lobe == LOBE_SHEEN_REFLECTION)
  {
    brdf_sheen_eval(data, mat);
  }

  // Occlusion effect (glTF occlusion texture)
  data.bsdf_diffuse *= mat.occlusion;
  data.bsdf_glossy *= mat.occlusion;
}

/*-------------------------------------------------------------------------------------------------
# Function `bsdfSample`

> Samples the full BSDF model for the given material and input direction.

You must provide `BsdfSampleData`'s `k1` and `xi` parameters. This function
will set the other parameters of `BsdfSampleData`.

There are two things you should check after calling this function:
* Is `data.event_type` equal to `BSDF_EVENT_ABSORB`? If so, the sampler sampled
  an output direction that would be absorbed by the material (e.g. it chose a
  reflective lobe but sampled a vector below the surface).
  The light path ends here.
* Is `data.pdf` equal to `DIRAC`? If so, this sampled a perfectly specular lobe.
  If you're using Multiple Importance Sampling weights, you should compute them
  as if `data.pdf` was infinity.
  
-------------------------------------------------------------------------------------------------*/
void bsdfSample(NVSHADERS_INOUT_TYPE(BsdfSampleData) data, PbrMaterial mat)
{
  float3 tint        = mat.baseColor;
  float  VdotN       = dot(data.k1, mat.N);
  int    lobe        = findLobe(mat, VdotN, data.xi.z, tint);
  data.pdf           = 0;
  data.bsdf_over_pdf = float3(0.0F);
  data.event_type    = BSDF_EVENT_ABSORB;

  if(lobe == LOBE_DIFFUSE_REFLECTION)
  {
    brdf_diffuse_sample(data, mat, tint);
  }
  else if(lobe == LOBE_DIFFUSE_TRANSMISSION)
  {
    brdf_diffuse_transmission_sample(data, mat, tint);
  }
  else if(lobe == LOBE_SPECULAR_REFLECTION)
  {
    brdf_ggx_smith_sample(data, mat, LOBE_SPECULAR_REFLECTION, mat.specularColor);
  }
  else if(lobe == LOBE_SPECULAR_TRANSMISSION)
  {
    btdf_ggx_smith_sample(data, mat, tint);
  }
  else if(lobe == LOBE_METAL_REFLECTION)
  {
    brdf_ggx_smith_sample(data, mat, LOBE_METAL_REFLECTION, mat.baseColor);
  }
  else if(lobe == LOBE_CLEARCOAT_REFLECTION)
  {
    mat.roughness   = float2(mat.clearcoatRoughness * mat.clearcoatRoughness);
    mat.N           = mat.Nc;
    mat.B           = normalize(cross(mat.N, mat.T));  // Assumes Nc and Tc are not collinear!
    mat.T           = cross(mat.B, mat.N);
    mat.iridescence = 0.0f;
    brdf_ggx_smith_sample(data, mat, LOBE_CLEARCOAT_REFLECTION, float3(1, 1, 1));
  }
  else if(lobe == LOBE_SHEEN_REFLECTION)
  {
    // Sheen is using the state.sheenColor and state.sheenInvRoughness values directly.
    // Only brdf_sheen_sample needs a third random sample for the v-cavities flip. Put this as argument.
    brdf_sheen_sample(data, mat);
  }

  // Avoid internal reflection
  if(data.pdf <= 0.00001F || any(isnan(data.bsdf_over_pdf)))
  {
    data.event_type = BSDF_EVENT_ABSORB;
  }
  if((isnan(data.pdf) || isinf(data.pdf)) && data.event_type != BSDF_EVENT_ABSORB)
  {
    // Treat as a perfectly specular bounce; change GLOSSY to IMPULSE
    data.event_type = (data.event_type & (~BSDF_EVENT_GLOSSY)) | BSDF_EVENT_IMPULSE;
    data.pdf        = DIRAC;
  }
}

/*-------------------------------------------------------------------------------------------------
# Simple BSDF Model
The functions below are used to evaluate and sample the BSDF for a simple PBR material,
without any additional lobes like clearcoat, sheen, etc. and without the need of random numbers.
This is based on the metallic/roughness BRDF in Appendix B of the glTF specification.
-------------------------------------------------------------------------------------------------*/

// Returns the probability that bsdfSampleSimple samples a glossy lobe.
float bsdfSimpleGlossyProbability(float NdotV, float metallic)
{
  return lerp(schlickFresnel(0.04F, 1.0F, NdotV), 1.0F, metallic);
}

/*-------------------------------------------------------------------------------------------------
# Function `bsdfEvaluateSimple`
> Evaluates the simple BSDF model using the given material and input and output directions.

You must provide `data.k1` and `data.k2`, but do not need to provide `data.xi`.
-------------------------------------------------------------------------------------------------*/
void bsdfEvaluateSimple(NVSHADERS_INOUT_TYPE(BsdfEvaluateData) data, PbrMaterial mat)
{
  // Specular reflection
  float3 H     = normalize(data.k1 + data.k2);
  float  NdotV = clampedDot(mat.N, data.k1);
  float  NdotL = clampedDot(mat.N, data.k2);
  float  VdotH = clampedDot(data.k1, H);
  float  NdotH = clampedDot(mat.N, H);

  if(NdotV == 0.0f || NdotL == 0.0f || VdotH == 0.0f || NdotH == 0.0f)
  {
    data.bsdf_diffuse = data.bsdf_glossy = float3(0.0f);
    data.pdf                             = 0.0f;
    return;
  }

  // We combine the metallic and specular lobes into a single glossy lobe.
  // The metallic weight is     metallic *    fresnel(f0 = baseColor)
  // The specular weight is (1-metallic) *    fresnel(f0 = c_min_reflectance)
  // The diffuse weight is  (1-metallic) * (1-fresnel(f0 = c_min_reflectance)) * baseColor

  // Fresnel terms
  float  c_min_reflectance = 0.04F;
  float3 f0                = lerp(float3(c_min_reflectance), mat.baseColor, mat.metallic);
  float3 fGlossy           = schlickFresnel(f0, float3(1.0F), VdotH);  // Metallic + specular
  float  fDiffuse          = schlickFresnel(1.0F - c_min_reflectance, 0.0F, VdotH) * (1.0F - mat.metallic);

  // Specular GGX
  float3 localH  = float3(dot(mat.T, H), dot(mat.B, H), NdotH);
  float  d       = hvd_ggx_eval(1.0f / mat.roughness, localH);
  float3 localK1 = float3(dot(mat.T, data.k1), dot(mat.B, data.k1), NdotV);
  float3 localK2 = float3(dot(mat.T, data.k2), dot(mat.B, data.k2), NdotL);
  float  G1 = 0.0f, G2 = 0.0f;
  ggx_smith_shadow_mask(G1, G2, localK1, localK2, mat.roughness);

  float diffusePdf  = M_1_PI * NdotL;
  float specularPdf = G1 * d * 0.25f / (NdotV * NdotH);
  data.pdf          = lerp(diffusePdf, specularPdf, bsdfSimpleGlossyProbability(NdotV, mat.metallic));

  data.bsdf_diffuse = mat.baseColor * fDiffuse * diffusePdf;  // Lambertian
  data.bsdf_glossy  = fGlossy * G2 * specularPdf;             // GGX-Smith
}

/*-------------------------------------------------------------------------------------------------
# Function `bsdfSampleSimple`
> Evaluates the simple BSDF model using the given material and input and output directions.

You must provide `data.k1` and `data.xi`. For one sample of pure reflection
(e.g. vk_mini_samples/ray_trace), use `data.xi == float2(0,0)`.

After calling this function, you should check if `data.event_type` is
`BSDF_EVENT_ABSORB`. If so, the sampling code sampled a direction below the
surface, and the light path ends here (it should be treated as a reflectance of 0).

This code cannot currently return a PDF of `DIRAC`, but that might change in
the future.
-------------------------------------------------------------------------------------------------*/
void bsdfSampleSimple(NVSHADERS_INOUT_TYPE(BsdfSampleData) data, PbrMaterial mat)
{
  float3 tint        = mat.baseColor;
  data.bsdf_over_pdf = float3(0.0F);

  float nk1 = clampedDot(mat.N, data.k1);
  if(data.xi.z <= bsdfSimpleGlossyProbability(nk1, mat.metallic))
  {
    // Glossy GGX
    data.event_type = BSDF_EVENT_GLOSSY_REFLECTION;
    // Transform to local space
    float3 localK1    = float3(dot(mat.T, data.k1), dot(mat.B, data.k1), nk1);
    float3 halfVector = hvd_ggx_sample_vndf(localK1, mat.roughness, data.xi.xy);
    // Transform from local space
    halfVector = mat.T * halfVector.x + mat.B * halfVector.y + mat.N * halfVector.z;
    data.k2    = reflect(-data.k1, halfVector);
  }
  else
  {
    // Diffuse
    data.event_type = BSDF_EVENT_DIFFUSE_REFLECTION;
    float3 localDir = cosineSampleHemisphere(data.xi.x, data.xi.y);
    data.k2         = mat.T * localDir.x + mat.B * localDir.y + mat.N * localDir.z;
  }

  BsdfEvaluateData evalData;
  evalData.k1 = data.k1;
  evalData.k2 = data.k2;
  bsdfEvaluateSimple(evalData, mat);
  data.pdf          = evalData.pdf;
  float3 bsdf_total = evalData.bsdf_diffuse + evalData.bsdf_glossy;
  if(data.pdf <= 0.00001F || any(isnan(bsdf_total)))
  {
    data.bsdf_over_pdf = float3(0.0f);
    data.event_type    = BSDF_EVENT_ABSORB;
  }
  else
  {
    data.bsdf_over_pdf = bsdf_total / data.pdf;
  }
}

/*-------------------------------------------------------------------------------------------------
# Function `bsdfSimpleAverageReflectance`
> Returns the approximate average reflectance of the Simple BSDF -- that is,
> `average_over_k2(f(k1, k2))` -- if GGX didn't lose energy.

This is useful for things like the variance reduction algorithm in
Tomasz Stachowiak's *Stochastic Screen-Space Reflections*; see also
Ray-Tracing Gems 1, chapter 32, *Accurate Real-Time Specular Reflections
with Radiance Caching*.
-------------------------------------------------------------------------------------------------*/
float3 bsdfSimpleAverageReflectance(float3 k1, PbrMaterial mat)
{
  float  NdotV             = clampedDot(mat.N, k1);
  float  c_min_reflectance = 0.04F;
  float3 f0                = lerp(float3(c_min_reflectance), mat.baseColor, mat.metallic);
  // This is approximate because
  // average_over_k2(fresnel(f0, 1.0f, VdotH)) != fresnel(f0, 1.0f, NdotV).
  float3 bsdf_glossy_average = schlickFresnel(f0, float3(1.0F), NdotV);
  float3 bsdf_diffuse_average = mat.baseColor * schlickFresnel(1.0F - c_min_reflectance, 0.0F, NdotV) * (1.0F - mat.metallic);
  return bsdf_glossy_average + bsdf_diffuse_average;
}


#endif  // NVVKHL_BSDF_FUNCTIONS_H
