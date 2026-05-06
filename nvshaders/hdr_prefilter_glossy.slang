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


// GGX split-sum IBL prefilter (Karis / Unreal).
//
//   filtered(R, α) = ∫ L(ω)·(n·l) D(α) dω  /  ∫ (n·l) D(α) dω
//                  ≈ Σ L(ω_i)·(n·l_i)  /  Σ (n·l_i)        with samples drawn from D(α).
//
// Direct port of Khronos `glTF-Sample-Renderer/source/shaders/ibl_filtering.frag`, adapted
// to our equirectangular HDR input (the reference uses a cubemap).
//
// Conventions (matches glTF / Khronos):
//   * `pc.roughness` is perceptualRoughness ∈ [0, 1].
//   * Internal GGX α = perceptualRoughness² (squared here).
//   * Consumers sample at lod = perceptualRoughness · (mipCount - 1)
//     (see `pbr_ibl_eval.h.slang::getIBLRadianceGGX`).
//
// Variance reduction:
//   * Samples read a mip of the HDR chosen via the Krivanek & Colbert formula (GPU Gems 3,
//     Ch. 20). Low-pdf samples (wide cones / rare directions) read blurrier mips, which
//     averages bright HDR texels over neighbors and kills fireflies on environments with
//     concentrated peaks (sun, windows). The HDR mip chain is generated up front by the
//     host (see `vk_gltf_renderer/src/renderer.cpp` around `cmdGenerateMipmaps`).

#include "hdr_io.h.slang"
#include "constants.h.slang"
#include "functions.h.slang"

// clang-format off
[[vk::binding(EnvDomeDraw::eHdrImage, 0)]]      RWTexture2D<float4>          g_outColor;
[[vk::binding(EnvBindings::eHdr, 1)]]           Sampler2D                    hdrIBL;
[[vk::push_constant]]                           ConstantBuffer<HdrPushBlock> pc;
// clang-format on


// Inverse CDF of D(α) in solid angle (no n·h weighting). Karis-style GGX importance sample.
float3 ggxSample(float2 xi, float alpha)
{
  float phi       = 2.0F * M_PI * xi.x;
  float cos_theta = sqrt((1.0F - xi.y) / (1.0F + (alpha * alpha - 1.0F) * xi.y));
  float sin_theta = sqrt(1.0F - cos_theta * cos_theta);
  return float3(cos(phi) * sin_theta, sin(phi) * sin_theta, cos_theta);
}

// Filament-style compact GGX NDF.
float ggxEval(float n_dot_h, float alpha)
{
  float a = n_dot_h * alpha;
  float k = alpha / (1.0F - n_dot_h * n_dot_h + a * a);
  return k * k * (1.0F / M_PI);
}

[shader("compute")]
[numthreads(HDR_WORKGROUP_SIZE, HDR_WORKGROUP_SIZE, 1)]
void main(uint3 threadIdx: SV_DispatchThreadID)
{
  const float2 pixel_center = float2(threadIdx.xy) + float2(0.5F);
  const float2 in_uv        = pixel_center / float2(pc.size);
  const float2 d            = in_uv * 2.0F - 1.0F;

  float4 v         = float4(d.x, d.y, 1.0F, 1.0F);
  v                = mul(v, pc.mvp);
  float3 direction = float3(v.xyz);

  float3 tangent, bitangent;
  float3 normal = normalize(float3(direction.x, -direction.y, direction.z));  // Flipping Y
  orthonormalBasis(normal, tangent, bitangent);

  // perceptualRoughness in pc.roughness; α is squared per Khronos / glTF convention.
  const float alpha    = pc.roughness * pc.roughness;
  const uint  nsamples = alpha > 0.0F ? 1024u : 1u;

  // Equirect texel count for Krivanek–Colbert LOD (cube uses 6·W², we use W·H).
  uint envW, envH, envMips;
  hdrIBL.GetDimensions(0, envW, envH, envMips);
  const float texel_count = float(envW) * float(envH);

  float3 result     = float3(0.0F);
  float  weight_sum = 0.0F;

  for(uint i = 0u; i < nsamples; ++i)
  {
    float2 xi = hammersley2D(i, nsamples);
    float3 h0 = alpha > 0.0F ? ggxSample(xi, alpha) : float3(0.0F, 0.0F, 1.0F);
    float3 h  = tangent * h0.x + bitangent * h0.y + normal * h0.z;

    // Karis assumes V == N for the prefilter (the "split" in split-sum).
    float3 l = normalize(2.0F * dot(normal, h) * h - normal);

    float n_dot_l = dot(normal, l);
    if(n_dot_l > 0.0F)
    {
      // Solid-angle pdf of l. Starting from the half-vector pdf D(α, n·h)·n·h and applying
      // the reflection Jacobian 1/(4·v·h), with V == N ⇒ v·h = n·h, this reduces to D(α, n·h)/4.
      float pdf = ggxEval(h0.z, alpha) * 0.25F;

      // Krivanek & Colbert, GPU Gems 3 Ch. 20. Low-pdf samples fetch a blurrier mip.
      float lod = (pc.roughness == 0.0F) ? 0.0F :
                  0.5F * log2(texel_count / (float(nsamples) * max(pdf, 1e-8F)));

      float2 uv = getSphericalUv(l);
      float3 L  = hdrIBL.SampleLevel(uv, lod).rgb;

      result     += L * n_dot_l;
      weight_sum += n_dot_l;
    }
  }

  // weight_sum > 0 is guaranteed for the mirror path (n_dot_l = 1); for non-mirror, at least
  // some Hammersley samples land in the upper hemisphere.
  float4 result_color            = float4(result / max(weight_sum, 1e-8F), 1.0F);
  g_outColor[int2(threadIdx.xy)] = result_color;
}
