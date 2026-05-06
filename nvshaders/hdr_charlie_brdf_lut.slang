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

// Bakes the Charlie sheen directional albedo into a 2D LUT keyed on (NdotV, sheenRoughness).
// Companion to the GGX BRDF LUT baked by `hdr_integrate_brdf.slang`; consumed at sample time by
// `pbr_ibl_eval.h.slang::getIBLCharlie`/`composeIBLAnalyticKhronos` and by
// `pbr_ggx_microfacet.h.slang::albedoSheenScalingLUT` (which `pbr_direct_eval.h.slang`'s
// punctual evaluator calls for the bidirectional sheen scaling factor).
//
// The same integral serves two roles in the Khronos glTF Sample Renderer (it ships two LUTs
// differing only by a constant normalization):
//   * `u_CharlieLUT.b` sampled in `getIBLRadianceCharlie` (ibl.glsl:149) to scale the Charlie
//     env contribution.
//   * `u_SheenELUT.r` sampled in `albedoSheenScalingLUT` (material_info.glsl:407-410) for the
//     sheen energy-compensation factor.
//
// We bake the directional albedo (∫ D_Charlie · V_Sheen · NdotL dω in [0,1]) and use it for
// both roles. Same value in `.r` and `.g` so callers can pick whichever channel suits their
// sampling code.
//
// D_Charlie / lambdaSheen / V_Sheen are duplicated here from
// `pbr_ggx_microfacet.h.slang::"Khronos analytic helpers"` to keep this compute shader
// self-contained (avoids pulling in the full PbrMaterial stack through its include chain for a
// one-time bake).

#include "constants.h.slang"

[[vk::binding(0, 0)]] RWTexture2D<float4> gOutLut;

[[vk::push_constant]]
cbuffer PushConst
{
  uint2 lutSize;
};

float radinv(uint bits)
{
  bits = (bits << 16u) | (bits >> 16u);
  bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
  bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
  bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
  bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
  return float(bits) * 2.3283064365386963e-10;
}

float2 hammersley2D(uint i, uint N)
{
  return float2(float(i) / float(N), radinv(i));
}

// Charlie NDF, lambdaSheen, V_Sheen — line-for-line ports of Khronos brdf.glsl.
float D_Charlie(float sheenRoughness, float NdotH)
{
  float r2    = max(sheenRoughness * sheenRoughness, 1e-6);
  float invR  = 1.0 / r2;
  float cos2h = NdotH * NdotH;
  float sin2h = max(1.0 - cos2h, 0.0);
  return (2.0 + invR) * pow(sin2h, invR * 0.5) / (2.0 * M_PI);
}

float lambdaSheenNumericHelper(float x, float alphaG)
{
  float oneMinusAlphaSq = (1.0 - alphaG) * (1.0 - alphaG);
  float a = lerp(21.5473, 25.3245, oneMinusAlphaSq);
  float b = lerp(3.82987, 3.32435, oneMinusAlphaSq);
  float c = lerp(0.19823, 0.16801, oneMinusAlphaSq);
  float d = lerp(-1.97760, -1.27393, oneMinusAlphaSq);
  float e = lerp(-4.32054, -4.85967, oneMinusAlphaSq);
  return a / (1.0 + b * pow(x, c)) + d * x + e;
}

float lambdaSheen(float cosTheta, float alphaG)
{
  if(abs(cosTheta) < 0.5)
    return exp(lambdaSheenNumericHelper(cosTheta, alphaG));
  return exp(2.0 * lambdaSheenNumericHelper(0.5, alphaG) - lambdaSheenNumericHelper(1.0 - cosTheta, alphaG));
}

float V_Sheen(float NdotL, float NdotV, float sheenRoughness)
{
  float r      = max(sheenRoughness, 1e-6);
  float alphaG = r * r;
  float denom  = (1.0 + lambdaSheen(NdotV, alphaG) + lambdaSheen(NdotL, alphaG)) * (4.0 * NdotV * NdotL);
  return clamp(1.0 / max(denom, 1e-7), 0.0, 1.0);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 tid: SV_DispatchThreadID)
{
  if(tid.x >= lutSize.x || tid.y >= lutSize.y)
    return;

  // (u, v) maps to (NdotV, sheenRoughness). Same layout as the existing GGX BRDF LUT.
  float2 uv             = (float2(tid.xy) + 0.5) / float2(lutSize);
  float  NdotV          = max(uv.x, 1e-4);
  float  sheenRoughness = uv.y;

  // View vector in tangent frame.
  float3 view = float3(sqrt(max(1.0 - NdotV * NdotV, 0.0)), 0.0, NdotV);

  const uint nsamples = 1024u;
  float      result   = 0.0;

  // Cosine-weighted hemisphere sampling over the half vector (no closed-form VNDF for Charlie).
  // PDF(h) = NdotH/π → PDF(l) = (NdotH/π)·(1/(4·VdotH)).
  // MC estimate of ∫ D·V·NdotL dω_l ≈ (1/N) Σ (D·V·NdotL) / pdf(l).
  for(uint i = 0u; i < nsamples; ++i)
  {
    float2 xi   = hammersley2D(i, nsamples);
    float  phi  = 2.0 * M_PI * xi.x;
    float  cosH = sqrt(max(1.0 - xi.y, 0.0));
    float  sinH = sqrt(max(1.0 - cosH * cosH, 0.0));
    float3 h    = float3(cos(phi) * sinH, sin(phi) * sinH, cosH);

    float3 light = normalize(2.0 * dot(view, h) * h - view);

    float NdotL = max(light.z, 0.0);
    float NdotH = max(h.z, 0.0);
    float VdotH = max(dot(view, h), 0.0);

    if(NdotL > 0.0 && NdotH > 0.0 && VdotH > 0.0)
    {
      float pdf_l = (NdotH / M_PI) * 0.25 / VdotH;
      float D     = D_Charlie(sheenRoughness, NdotH);
      float V     = V_Sheen(NdotL, NdotV, sheenRoughness);
      result += (D * V * NdotL) / max(pdf_l, 1e-7);
    }
  }

  result /= float(nsamples);
  result = clamp(result, 0.0, 1.0);

  // Same value in .r and .g; callers can pick whichever suits their existing sampling code.
  gOutLut[int2(tid.xy)] = float4(result, result, 0.0, 1.0);
}
