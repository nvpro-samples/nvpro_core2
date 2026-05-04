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


// Lambertian (diffuse) IBL prefilter.
//
//   E(n) = ∫ L(ω)·(n·l)/π dω        (cosine-weighted irradiance over the upper hemisphere)
//
// Direct port of Khronos `glTF-Sample-Renderer/source/shaders/ibl_filtering.frag` (Lambertian
// path), adapted for our equirectangular HDR input (the reference uses a cubemap).
//
// Estimator: cosine-hemisphere importance sampling. With pdf(l) = (n·l)/π, the Monte-Carlo
// estimator collapses to a plain average of `L` along the sampled directions, because the
// Lambert BRDF weight (n·l)/π cancels the pdf exactly:
//
//   (1/N) Σ L(l_i) · (n·l_i)/π / pdf(l_i)  =  (1/N) Σ L(l_i)
//
// Variance reduction: samples read a mip of the HDR chosen via the Krivanek & Colbert LOD
// (GPU Gems 3, Ch. 20). Wide-cone / low-pdf samples fetch a blurrier mip, which averages
// bright HDR texels over neighbors and kills fireflies on HDRs with concentrated peaks.
// The HDR mip chain is generated up front by the host (see `vk_gltf_renderer/src/renderer.cpp`
// around `cmdGenerateMipmaps`).

#include "hdr_io.h.slang"
#include "constants.h.slang"
#include "functions.h.slang"

// clang-format off
[[vk::binding(EnvDomeDraw::eHdrImage, 0)]]      RWTexture2D<float4>          g_outColor;
[[vk::binding(EnvBindings::eHdr, 1)]]           Sampler2D                    hdrTexture;
[[vk::push_constant]]                           ConstantBuffer<HdrPushBlock> pc;
// clang-format on


[shader("compute")]
[numthreads(HDR_WORKGROUP_SIZE, HDR_WORKGROUP_SIZE, 1)]
void main(uint3 threadIdx: SV_DispatchThreadID)
{
  const float2 pixelCenter = float2(threadIdx.xy) + float2(0.5F);
  const float2 inUV        = pixelCenter / pc.size;
  const float2 d           = inUV * 2.0F - 1.0F;
  float3       direction   = float3(mul(float4(d.x, d.y, 1.0F, 1.0F), pc.mvp).xyz);

  float3 tangent, bitangent;
  float3 normal = normalize(float3(direction.x, -direction.y, direction.z));  // Flipping Y
  orthonormalBasis(normal, tangent, bitangent);

  const uint nsamples = 2048u;

  // Equirect texel count for Krivanek–Colbert LOD (cube uses 6·W², we use W·H).
  uint envW, envH, envMips;
  hdrTexture.GetDimensions(0, envW, envH, envMips);
  const float texel_count = float(envW) * float(envH);

  float3 result = float3(0.0F);
  for(uint i = 0u; i < nsamples; ++i)
  {
    // Cosine-hemisphere sample (Malley's method).
    float2 xi      = hammersley2D(i, nsamples);
    float  phi     = 2.0F * M_PI * xi.x;
    float  cos_th  = sqrt(xi.y);
    float  sin_th  = sqrt(1.0F - xi.y);
    float3 hLocal  = float3(sin_th * cos(phi), sin_th * sin(phi), cos_th);
    float3 l       = tangent * hLocal.x + bitangent * hLocal.y + normal * hLocal.z;

    // pdf(l) = cos_th / π. Krivanek & Colbert, GPU Gems 3 Ch. 20: low-pdf samples (wide
    // cones at grazing angles) fetch a blurrier mip.
    float pdf = cos_th * M_1_OVER_PI;
    float lod = 0.5F * log2(texel_count / (float(nsamples) * max(pdf, 1e-8F)));

    float2 uv = getSphericalUv(l);
    result += hdrTexture.SampleLevel(uv, lod).rgb;
  }

  // (cos_th/π) BRDF weight cancels the pdf exactly, leaving a plain average.
  float4 frag_color        = float4(result / float(nsamples), 1.0F);
  g_outColor[threadIdx.xy] = frag_color;
}
