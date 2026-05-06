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


#ifndef NVVKHL_GGX_H
#define NVVKHL_GGX_H 1

/*-------------------------------------------------------------------------------------------------
Lower-level mathematical functions used for BSDFs. This file is named after the
GGX normal distribution, which is the basis for much of nvvk's BSDF code, but
includes Fresnel, shadowing-masking, sheen, and thin-film functions as well.
-------------------------------------------------------------------------------------------------*/

#include "constants.h.slang"
#include "pbr_material_types.h.slang"  // PbrMaterial (used by the Khronos analytic helpers below)
#include "slang_types.h"


// Schlick Fresnel approximation
float schlickFresnel(float F0, float F90, float VdotH)
{
  return F0 + (F90 - F0) * pow(1.0F - VdotH, 5.0F);
}

// Schlick Fresnel approximation for vectors
float3 schlickFresnel(float3 F0, float3 F90, float VdotH)
{
  return F0 + (F90 - F0) * pow(1.0F - VdotH, 5.0F);
}

// Schlick Fresnel approximation for float
float schlickFresnel(float ior, float VdotH)
{
  // Calculate reflectance at normal incidence (R0)
  float R0 = pow((1.0F - ior) / (1.0F + ior), 2.0F);

  // Fresnel reflectance using Schlick's approximation
  return R0 + (1.0F - R0) * pow(1.0F - VdotH, 5.0F);
}


//-----------------------------------------------------------------------
// MDL-based functions
//-----------------------------------------------------------------------

// Mix two colors with a factor
float3 mix_rgb(const float3 base, const float3 layer, const float3 factor)
{
  return (1.0f - max(factor.x, max(factor.y, factor.z))) * base + factor * layer;
}

// Square the input
float sqr(float x)
{
  return x * x;
}

// Check for total internal reflection. `kh` is the dot product of the view
// vector and the half vector.
bool isTIR(const float2 ior, const float kh)
{
  const float b = ior.x / ior.y;
  return (1.0f < (b * b * (1.0f - kh * kh)));
}


/*-------------------------------------------------------------------------------------------------
# Function `hvd_ggx_eval`
> Evaluates anisotropic GGX distribution on the non-projected hemisphere (i.e.
> +z is up)
-------------------------------------------------------------------------------------------------*/
float hvd_ggx_eval(const float2 invRoughness,
                   const float3 h)  // == float3(dot(tangent, h), dot(bitangent, h), dot(normal, h))
{
  const float x     = h.x * invRoughness.x;
  const float y     = h.y * invRoughness.y;
  const float aniso = x * x + y * y;
  const float f     = aniso + h.z * h.z;

  return M_1_PI * invRoughness.x * invRoughness.y * h.z / (f * f);
}

/*-------------------------------------------------------------------------------------------------
# Function `hvd_ggx_sample_vndf`
> Samples Samples a visible (Smith-masked) half vector according to the anisotropic GGX distribution.

See Eric Heitz, "A Simpler and Exact Sampling Routine for the GGX Distribution of Visible Normals".

The input and output will be in local space:
`float3(dot(T, k1), dot(B, k1), dot(N, k1))`.
-------------------------------------------------------------------------------------------------*/
float3 hvd_ggx_sample_vndf(float3 k, float2 roughness, float2 xi)
{
  const float3 v = normalize(float3(k.x * roughness.x, k.y * roughness.y, k.z));

  const float3 t1 = (v.z < 0.99999f) ? normalize(cross(v, float3(0.0f, 0.0f, 1.0f))) : float3(1.0f, 0.0f, 0.0f);
  const float3 t2 = cross(t1, v);

  const float a = 1.0f / (1.0f + v.z);
  const float r = sqrt(xi.x);

  const float phi = (xi.y < a) ? xi.y / a * M_PI : M_PI + (xi.y - a) / (1.0f - a) * M_PI;
  float       sp  = sin(phi);
  float       cp  = cos(phi);
  const float p1  = r * cp;
  const float p2  = r * sp * ((xi.y < a) ? 1.0f : v.z);

  float3 h = p1 * t1 + p2 * t2 + sqrt(max(0.0f, 1.0f - p1 * p1 - p2 * p2)) * v;

  h.x *= roughness.x;
  h.y *= roughness.y;
  h.z = max(0.0f, h.z);
  return normalize(h);
}

float smith_shadow_or_mask(const float3 k, const float2 roughness)
{
  float kz2 = k.z * k.z;
  if(kz2 == 0.0f)
  {
    return 0.0f;  // Totally shadowed
  }
  const float ax     = k.x * roughness.x;
  const float ay     = k.y * roughness.y;
  const float inv_a2 = (ax * ax + ay * ay) / kz2;

  return 2.0f / (1.0f + sqrt(1.0f + inv_a2));
}

/*-------------------------------------------------------------------------------------------------
# Function `ggx_smith_shadow_mask`
> The uncorrelated Smith shadowing-masking function.

Note that the joint Smith shadowing-masking function may be more realistic.
-------------------------------------------------------------------------------------------------*/
float ggx_smith_shadow_mask(NVSHADERS_OUT_TYPE(float) G1, NVSHADERS_OUT_TYPE(float) G2, const float3 k1, const float3 k2, const float2 roughness)
{
  G1 = smith_shadow_or_mask(k1, roughness);
  G2 = smith_shadow_or_mask(k2, roughness);

  return G1 * G2;
}

// Computes the squared norm of s/p polarized Fresnel reflection coefficients and phase shifts in complex unit circle.
//
// See Max Born and Emil Wolf, "Principles of Optics", section 13.4.1,
// "An absorbing film on a transparent substrate".
//
// This comes from MDL, and is a different approach than the example
// implementation in KHR_materials_iridescence.
float2 fresnel_conductor(NVSHADERS_OUT_TYPE(float2) phase_shift_sin,
                         NVSHADERS_OUT_TYPE(float2) phase_shift_cos,
                         const float n_a,
                         const float n_b,
                         const float k_b,
                         const float cos_a,
                         const float sin_a_sqd)
{
  const float k_b2   = k_b * k_b;
  const float n_b2   = n_b * n_b;
  const float n_a2   = n_a * n_a;
  const float tmp0   = n_b2 - k_b2;
  const float half_U = 0.5f * (tmp0 - n_a2 * sin_a_sqd);
  const float half_V = sqrt(max(0.0f, half_U * half_U + k_b2 * n_b2));

  const float u_b2 = half_U + half_V;
  const float v_b2 = half_V - half_U;
  const float u_b  = sqrt(max(0.0f, u_b2));
  const float v_b  = sqrt(max(0.0f, v_b2));

  const float tmp1 = tmp0 * cos_a;
  const float tmp2 = n_a * u_b;
  const float tmp3 = (2.0f * n_b * k_b) * cos_a;
  const float tmp4 = n_a * v_b;
  const float tmp5 = n_a * cos_a;

  const float tmp6 = (2.0f * tmp5) * v_b;
  const float tmp7 = (u_b2 + v_b2) - tmp5 * tmp5;

  const float tmp8 = (2.0f * tmp5) * ((2.0f * n_b * k_b) * u_b - tmp0 * v_b);
  const float tmp9 = sqr((n_b2 + k_b2) * cos_a) - n_a2 * (u_b2 + v_b2);

  const float tmp67      = tmp6 * tmp6 + tmp7 * tmp7;
  const float inv_sqrt_x = (0.0f < tmp67) ? (1.0f / sqrt(tmp67)) : 0.0f;
  const float tmp89      = tmp8 * tmp8 + tmp9 * tmp9;
  const float inv_sqrt_y = (0.0f < tmp89) ? (1.0f / sqrt(tmp89)) : 0.0f;

  phase_shift_cos = float2(tmp7 * inv_sqrt_x, tmp9 * inv_sqrt_y);
  phase_shift_sin = float2(tmp6 * inv_sqrt_x, tmp8 * inv_sqrt_y);

  return float2((sqr(tmp5 - u_b) + v_b2) / (sqr(tmp5 + u_b) + v_b2),
                (sqr(tmp1 - tmp2) + sqr(tmp3 - tmp4)) / (sqr(tmp1 + tmp2) + sqr(tmp3 + tmp4)));
}


// Simplified for dielectric, no phase shift computation.
float2 fresnel_dielectric(const float n_a, const float n_b, const float cos_a, const float cos_b)
{
  const float naca = n_a * cos_a;
  const float nbcb = n_b * cos_b;
  const float r_s  = (naca - nbcb) / (naca + nbcb);

  const float nacb = n_a * cos_b;
  const float nbca = n_b * cos_a;
  const float r_p  = (nbca - nacb) / (nbca + nacb);

  return float2(r_s * r_s, r_p * r_p);
}


float3 thin_film_factor(float coating_thickness, const float coating_ior, const float base_ior, const float incoming_ior, const float kh)
{
  coating_thickness = max(0.0f, coating_thickness);

  const float sin0_sqr  = max(0.0f, 1.0f - kh * kh);
  const float eta01     = incoming_ior / coating_ior;
  const float eta01_sqr = eta01 * eta01;
  const float sin1_sqr  = eta01_sqr * sin0_sqr;

  if(1.0f < sin1_sqr)  // TIR at first interface
  {
    return float3(1.0f);
  }

  const float  cos1 = sqrt(max(0.0f, 1.0f - sin1_sqr));
  const float2 R01  = fresnel_dielectric(incoming_ior, coating_ior, kh, cos1);

  float2 phi12_sin, phi12_cos;
  const float2 R12 = fresnel_conductor(phi12_sin, phi12_cos, coating_ior, base_ior, /* base_k = */ 0.0f, cos1, sin1_sqr);

  const float tmp = (4.0f * M_PI) * coating_ior * coating_thickness * cos1;

  const float R01R12_s = max(0.0f, R01.x * R12.x);
  const float r01r12_s = sqrt(R01R12_s);

  const float R01R12_p = max(0.0f, R01.y * R12.y);
  const float r01r12_p = sqrt(R01R12_p);

  float3 xyz = float3(0.0f);

  //!! using low res color matching functions here
  float lambda_min  = 400.0f;
  float lambda_step = ((700.0f - 400.0f) / 16.0f);

  const float3 cie_xyz[16] = { { 0.02986f, 0.00310f, 0.13609f }, { 0.20715f, 0.02304f, 0.99584f },
                               { 0.36717f, 0.06469f, 1.89550f }, { 0.28549f, 0.13661f, 1.67236f },
                               { 0.08233f, 0.26856f, 0.76653f }, { 0.01723f, 0.48621f, 0.21889f },
                               { 0.14400f, 0.77341f, 0.05886f }, { 0.40957f, 0.95850f, 0.01280f },
                               { 0.74201f, 0.97967f, 0.00060f }, { 1.03325f, 0.84591f, 0.00000f },
                               { 1.08385f, 0.62242f, 0.00000f }, { 0.79203f, 0.36749f, 0.00000f },
                               { 0.38751f, 0.16135f, 0.00000f }, { 0.13401f, 0.05298f, 0.00000f },
                               { 0.03531f, 0.01375f, 0.00000f }, { 0.00817f, 0.00317f, 0.00000f } };

  float lambda = lambda_min + 0.5f * lambda_step;

  for(int i = 0; i < 16; ++i)
  {
    const float phi = tmp / lambda;

    float phi_s = sin(phi);
    float phi_c = cos(phi);

    const float cos_phi_s = phi_c * phi12_cos.x - phi_s * phi12_sin.x;  // cos(a+b) = cos(a) * cos(b) - sin(a) * sin(b)
    const float tmp_s     = 2.0f * r01r12_s * cos_phi_s;
    const float R_s       = (R01.x + R12.x + tmp_s) / (1.0f + R01R12_s + tmp_s);

    const float cos_phi_p = phi_c * phi12_cos.y - phi_s * phi12_sin.y;  // cos(a+b) = cos(a) * cos(b) - sin(a) * sin(b)
    const float tmp_p     = 2.0f * r01r12_p * cos_phi_p;
    const float R_p       = (R01.y + R12.y + tmp_p) / (1.0f + R01R12_p + tmp_p);

    xyz += cie_xyz[i] * (0.5f * (R_s + R_p));

    lambda += lambda_step;
  }

  xyz *= (1.0f / 16.0f);

  // ("normalized" such that the loop for no shifted wave gives reflectivity (1,1,1))
  return clamp(float3(xyz.x * (3.2406f / 0.433509f) + xyz.y * (-1.5372f / 0.433509f) + xyz.z * (-0.4986f / 0.433509f),
                      xyz.x * (-0.9689f / 0.341582f) + xyz.y * (1.8758f / 0.341582f) + xyz.z * (0.0415f / 0.341582f),
                      xyz.x * (0.0557f / 0.32695f) + xyz.y * (-0.204f / 0.32695f) + xyz.z * (1.057f / 0.32695f)),
               0.0f, 1.0f);
}


// Compute half vector (convention: pointing to outgoing direction, like shading normal)
float3 compute_half_vector(const float3 k1, const float3 k2, const float3 normal, const float2 ior, const float nk2, const bool transmission, const bool thinwalled)
{
  float3 h;

  if(transmission)
  {
    if(thinwalled)  // No refraction!
    {
      h = k1 + (normal * (nk2 + nk2) + k2);  // Use corresponding reflection direction.
    }
    else
    {
      h = k2 * ior.y + k1 * ior.x;  // Points into thicker medium.

      if(ior.y > ior.x)
      {
        h *= -1.0f;  // Make pointing to outgoing direction's medium.
      }
    }
  }
  else
  {
    h = k1 + k2;  // unnormalized half-vector
  }

  return normalize(h);
}

float3 refract(const float3 k,                // direction (pointing from surface)
               const float3 n,                // normal
               const float  b,                // (reflected side IOR) / (transmitted side IOR)
               const float  nk,               // dot(n, k)
               NVSHADERS_OUT_TYPE(bool) tir)  // total internal reflection
{
  const float refraction = b * b * (1.0f - nk * nk);

  tir = (1.0f <= refraction);

  return (tir) ? (n * (nk + nk) - k) : normalize((-k * b + n * (b * nk - sqrt(1.0f - refraction))));
}


/*-------------------------------------------------------------------------------------------------
# Function `ior_fresnel`
> Fresnel equation for an equal mix of polarization.
-------------------------------------------------------------------------------------------------*/
float ior_fresnel(const float eta,  // refracted / reflected ior
                  const float kh)   // cosine of angle between normal/half-vector and direction
{
  float costheta = 1.0f - (1.0f - kh * kh) / (eta * eta);

  if(costheta <= 0.0f)
  {
    return 1.0f;
  }

  costheta = sqrt(costheta);  // refracted angle cosine

  const float n1t1 = kh;
  const float n1t2 = costheta;
  const float n2t1 = kh * eta;
  const float n2t2 = costheta * eta;
  const float r_p  = (n1t2 - n2t1) / (n1t2 + n2t1);
  const float r_o  = (n1t1 - n2t2) / (n1t1 + n2t2);

  const float fres = 0.5f * (r_p * r_p + r_o * r_o);

  return clamp(fres, 0.0f, 1.0f);
}

/*-------------------------------------------------------------------------------------------------
# Function `hvd_sheen_eval`
> Evaluates the anisotropic sheen half-vector distribution on the non-projected hemisphere.
-------------------------------------------------------------------------------------------------*/
float hvd_sheen_eval(const float invRoughness,
                     const float nh)  // dot(shading_normal, h)
{
  const float sinTheta2 = max(0.0f, 1.0f - nh * nh);
  const float sinTheta  = sqrt(sinTheta2);

  return (invRoughness + 2.0f) * pow(sinTheta, invRoughness) * 0.5f * M_1_PI * nh;
}


// Just the masking term for the v-cavity model.
float vcavities_mask(const float nh,  // abs(dot(normal, half))
                     const float kh,  // abs(dot(dir, half))
                     const float nk)  // abs(dot(normal, dir))
{
  return min(2.0f * nh * nk / kh, 1.0f);
}

/*-------------------------------------------------------------------------------------------------
# Function `vcavities_shadow_mask`
> Cook-Torrance style v-cavities shadowing-masking term.
-------------------------------------------------------------------------------------------------*/
float vcavities_shadow_mask(NVSHADERS_OUT_TYPE(float) G1,
                            NVSHADERS_OUT_TYPE(float) G2,
                            const float  nh,
                            const float3 k1,
                            const float  k1h,
                            const float3 k2,
                            const float  k2h)
{
  G1 = vcavities_mask(nh, k1h, k1.z);  // In my renderer the z-coordinate is the normal!
  G2 = vcavities_mask(nh, k2h, k2.z);

  //return (refraction) ? fmaxf(0.0f, G1 + G2 - 1.0f) : fminf(G1, G2);
  return min(G1, G2);  // PERF Need reflection only.
}


/*-------------------------------------------------------------------------------------------------
# Function `hvd_sheen_sample`
> Samples a half-vector according to an anisotropic sheen distribution.
-------------------------------------------------------------------------------------------------*/
float3 hvd_sheen_sample(const float2 xi, const float invRoughness)
{
  const float phi = 2.0f * M_PI * xi.x;

  float sinPhi = sin(phi);
  float cosPhi = cos(phi);

  const float sinTheta = pow(1.0f - xi.y, 1.0f / (invRoughness + 2.0f));
  const float cosTheta = sqrt(1.0f - sinTheta * sinTheta);

  return normalize(float3(cosPhi * sinTheta, sinPhi * sinTheta,
                          cosTheta));  // In my renderer the z-coordinate is the normal!
}

float3 flip(const float3 h, const float3 k, float xi)
{
  const float a = h.z * k.z;
  const float b = h.x * k.x + h.y * k.y;

  const float kh   = max(0.0f, a + b);
  const float kh_f = max(0.0f, a - b);

  const float p_flip = kh_f / (kh + kh_f);

  // PERF xi is not used after this operation by the only caller brdf_sheen_sample(),
  // so there is no need to scale the sample.
  //if (xi < p_flip)
  //{
  //  xi /= p_flip;
  //  return make_float3(-h.x, -h.y, h.z);
  //}
  //else
  //{
  //  xi = (xi - p_flip) / (1.0f - p_flip);
  //  return h;
  //}

  return (xi < p_flip) ? float3(-h.x, -h.y, h.z) : h;
}


// =================================================================================================
// Khronos analytic helpers (deterministic per-light / IBL evaluators)
// =================================================================================================
//
// Everything below mirrors the per-lobe BRDF math in the glTF Sample Renderer
// (`pbr.frag` / `brdf.glsl` / `punctual.glsl` / `ibl.glsl`) so the rasterizer-side evaluators
// in `pbr_direct_eval.h.slang` and `pbr_ibl_eval.h.slang` can match the reference renderer
// pixel-for-pixel.
//
// These primitives use a different mathematical model than the Monte Carlo PT helpers above:
//   * Smith G is height-correlated `V_GGX` (Heitz 2014), not uncorrelated `G1*G2`.
//   * The NDF is given in non-projected form (`D_GGX(NdotH, alpha)`), not the VNDF projected form
//     that pairs with `hvd_ggx_sample_vndf`.
//   * Sheen uses Estevez & Kulla 2017's `D_Charlie` / `lambdaSheen` / `V_Sheen` family rather
//     than v-cavities + `hvd_sheen_eval`.
//
// They are NOT used by the Monte Carlo path tracer's `bsdfEvaluate` / `bsdfSample` -- those keep
// the MC-friendly approximations above so MIS PDFs cancel correctly.

// -------------------------------------------------------------------------------------------------
// Microfacet primitives (Khronos `brdf.glsl` ports)
// -------------------------------------------------------------------------------------------------

// GGX/Trowbridge-Reitz NDF (non-projected hemisphere).
float D_GGX(float NdotH, float alphaRoughness)
{
  float alphaRoughnessSq = alphaRoughness * alphaRoughness;
  float f                = (NdotH * NdotH) * (alphaRoughnessSq - 1.0F) + 1.0F;
  return alphaRoughnessSq / (M_PI * f * f);
}

// Smith-GGX height-correlated visibility term `Vis = G / (4 NdotL NdotV)` (Heitz 2014).
float V_GGX(float NdotL, float NdotV, float alphaRoughness)
{
  float alphaRoughnessSq = alphaRoughness * alphaRoughness;
  float GGXV             = NdotL * sqrt(NdotV * NdotV * (1.0F - alphaRoughnessSq) + alphaRoughnessSq);
  float GGXL             = NdotV * sqrt(NdotL * NdotL * (1.0F - alphaRoughnessSq) + alphaRoughnessSq);
  float GGX              = GGXV + GGXL;
  return (GGX > 0.0F) ? (0.5F / GGX) : 0.0F;
}

// Anisotropic GGX NDF (Disney/Burley addenda).
float D_GGX_anisotropic(float NdotH, float TdotH, float BdotH, float at, float ab)
{
  float  a2 = at * ab;
  float3 f  = float3(ab * TdotH, at * BdotH, a2 * NdotH);
  float  w2 = a2 / dot(f, f);
  return a2 * w2 * w2 / M_PI;
}

// Anisotropic GGX height-correlated visibility (Heitz).
float V_GGX_anisotropic(float NdotL, float NdotV, float BdotV, float TdotV, float TdotL, float BdotL, float at, float ab)
{
  float GGXV = NdotL * length(float3(at * TdotV, ab * BdotV, NdotV));
  float GGXL = NdotV * length(float3(at * TdotL, ab * BdotL, NdotL));
  float v    = 0.5F / max(GGXV + GGXL, 1e-7F);
  return clamp(v, 0.0F, 1.0F);
}


// -------------------------------------------------------------------------------------------------
// Sheen primitives (Estevez & Kulla 2017, Khronos port)
// -------------------------------------------------------------------------------------------------

float D_Charlie(float sheenRoughness, float NdotH)
{
  sheenRoughness = max(sheenRoughness, 1e-6F);
  float alphaG   = sheenRoughness * sheenRoughness;
  float invR     = 1.0F / alphaG;
  float cos2h    = NdotH * NdotH;
  float sin2h    = max(1.0F - cos2h, 0.0F);
  return (2.0F + invR) * pow(sin2h, invR * 0.5F) / (2.0F * M_PI);
}

float lambdaSheenNumericHelper(float x, float alphaG)
{
  float oneMinusAlphaSq = (1.0F - alphaG) * (1.0F - alphaG);
  float a               = lerp(21.5473F, 25.3245F, oneMinusAlphaSq);
  float b               = lerp(3.82987F, 3.32435F, oneMinusAlphaSq);
  float c               = lerp(0.19823F, 0.16801F, oneMinusAlphaSq);
  float d               = lerp(-1.97760F, -1.27393F, oneMinusAlphaSq);
  float e               = lerp(-4.32054F, -4.85967F, oneMinusAlphaSq);
  return a / (1.0F + b * pow(x, c)) + d * x + e;
}

float lambdaSheen(float cosTheta, float alphaG)
{
  if(abs(cosTheta) < 0.5F)
    return exp(lambdaSheenNumericHelper(cosTheta, alphaG));
  return exp(2.0F * lambdaSheenNumericHelper(0.5F, alphaG) - lambdaSheenNumericHelper(1.0F - cosTheta, alphaG));
}

float V_Sheen(float NdotL, float NdotV, float sheenRoughness)
{
  sheenRoughness = max(sheenRoughness, 1e-6F);
  float alphaG   = sheenRoughness * sheenRoughness;
  float denom    = (1.0F + lambdaSheen(NdotV, alphaG) + lambdaSheen(NdotL, alphaG)) * (4.0F * NdotV * NdotL);
  return clamp(1.0F / max(denom, 1e-7F), 0.0F, 1.0F);
}


// -------------------------------------------------------------------------------------------------
// Lambertian
// -------------------------------------------------------------------------------------------------

// Lambertian: c / pi.
float3 BRDF_lambertian(float3 diffuseColor)
{
  return diffuseColor * M_1_PI;
}


// -------------------------------------------------------------------------------------------------
// Sheen energy-compensation factor (Khronos `albedoSheenScalingLUT`)
// -------------------------------------------------------------------------------------------------

// Reads `.r` from the runtime-baked Charlie directional-albedo LUT
// (see `nvshaders/hdr_charlie_brdf_lut.slang` and `nvshaders_host/pbr_sheen_lut.hpp`).
#ifdef __SLANG__
float albedoSheenScalingLUT(Sampler2D sheenLut, float NdotX, float sheenRoughness)
{
  float2 uv = clamp(float2(NdotX, sheenRoughness), float2(0.0F), float2(1.0F));
  return sheenLut.SampleLevel(uv, 0.0F).r;
}
#endif


// -------------------------------------------------------------------------------------------------
// Khronos material-derived helpers
// -------------------------------------------------------------------------------------------------

/*-------------------------------------------------------------------------------------------------
# Function `khronosDielectricF0`
> Returns the vec3 dielectric F0 used by the Khronos glTF renderer's Fresnel math.

Derived from the KHR_materials_ior ratio (`ior2/ior1`, default 1.5/1.0 -> F0 ~ 0.04) and tinted
by KHR_materials_specular's `specularColor` (default `vec3(1)`), clamped to `vec3(1)` to keep
the Fresnel physically bounded.
-------------------------------------------------------------------------------------------------*/
float3 khronosDielectricF0(PbrMaterial mat)
{
  float ior         = mat.ior2 / max(mat.ior1, 1e-5F);
  float f0_from_ior = square((ior - 1.0F) / (ior + 1.0F));
  return min(float3(f0_from_ior) * mat.specularColor, float3(1.0F));
}

/*-------------------------------------------------------------------------------------------------
# Function `khronosClearcoatFresnel`
> Schlick Fresnel at the clearcoat normal used by `pbr.frag:157` / `pbr.frag:220`.

Returns `F_Schlick(clearcoatF0, vec3(1), NcDotV)`. Clearcoat uses the same IOR-derived F0 as the
dielectric base (Khronos `getClearCoatInfo`: `clearcoatF0 = pow((ior-1)/(ior+1), 2)`), so swapping
the base IOR via KHR_materials_ior also shifts the clearcoat's reflectance. Used by:
  * The IBL composer (final `mix(., f_clearcoat, clearcoatFactor * clearcoatFresnel)` step).
  * The emissive attenuation `emissive * (1 - clearcoatFactor * clearcoatFresnel)` (`pbr.frag:373`).
  * The KHR_materials_volume_scatter mask (`pbr.frag:355`).
-------------------------------------------------------------------------------------------------*/
float3 khronosClearcoatFresnel(PbrMaterial mat, float3 V)
{
  float3 cF0    = float3(square((mat.ior2 - mat.ior1) / max(mat.ior2 + mat.ior1, 1e-5F)));
  float  NcDotV = clampedDot(mat.Nc, V);
  return cF0 + (1.0F - cF0) * pow(1.0F - NcDotV, 5.0F);
}


// -------------------------------------------------------------------------------------------------
// Layered composite (Khronos `pbr.frag:218-220` / `:345-348`)
// -------------------------------------------------------------------------------------------------

/*-------------------------------------------------------------------------------------------------
# Function `layeredCompositeKhronos`
> Final per-pixel layering used by both the per-light direct evaluator
> (`pbr_direct_eval.h.slang::bsdfEvaluatePunctualKhronos`) and the IBL evaluator
> (`pbr_ibl_eval.h.slang::composeIBLAnalyticKhronos`). Mirrors `pbr.frag:218-220`:
```
color = lerp(dielectric, metal, metallic);
color = sheen + color * sheenScale;
color = lerp(color, clearcoat, clearcoat * clearcoatFresnel);
```
For the no-sheen path, pass `f_sheen = float3(0)` and `sheenScale = 1.0F`.
-------------------------------------------------------------------------------------------------*/
float3 layeredCompositeKhronos(float3 f_dielectric_brdf,
                               float3 f_metal_brdf,
                               float  metallic,
                               float3 f_sheen,
                               float  sheenScale,
                               float3 f_clearcoat_brdf,
                               float  clearcoat,
                               float3 clearcoatFresnel)
{
  float3 color = lerp(f_dielectric_brdf, f_metal_brdf, metallic);
  color        = f_sheen + color * sheenScale;
  color        = lerp(color, f_clearcoat_brdf, clearcoat * clearcoatFresnel);
  return color;
}


#endif  // NVVKHL_GGX_H
