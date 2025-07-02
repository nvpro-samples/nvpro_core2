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


// This shader computes a glossy IBL map to be used with the Unreal 4 PBR shading model as
// described in
//
// "Real Shading in Unreal Engine 4" by Brian Karis
// http://blog.selfshadow.com/publications/s2013-shading-course/karis/s2013_pbs_epic_notes_v2.pdf
//
// As an extension to the original it uses multiple importance sampling weighted BRDF importance
// sampling and environment map importance sampling to yield good results for high dynamic range
// lighting.

#include "hdr_io.h.slang"
#include "constants.h.slang"
#include "functions.h.slang"
#include "random.h.slang"
#include "hdr_env_sampling.h.slang"

// clang-format off
[[vk::binding(EnvDomeDraw::eHdrImage, 0)]]      RWTexture2D<float4> g_outColor;
[[vk::binding(EnvBindings::eHdr, 1)]]           Sampler2D hdrIBL;
[[vk::binding(EnvBindings::eImpSamples, 1)]]    StructuredBuffer<EnvAccel> envSamplingData;
[[vk::push_constant]]                           ConstantBuffer<HdrPushBlock> pc;
// clang-format on


// Importance sample a GGX microfacet distribution.
float3 ggxSample(float2 xi, float alpha)
{
  float phi       = 2.0F * M_PI * xi.x;
  float cos_theta = sqrt((1.0F - xi.y) / (1.0F + (alpha * alpha - 1.0F) * xi.y));
  float sin_theta = sqrt(1.0F - cos_theta * cos_theta);

  return float3(cos(phi) * sin_theta, sin(phi) * sin_theta, cos_theta);
}

// Evaluate a GGX microfacet distribution.
float ggxEval(float alpha, float nh)
{
  float a2   = alpha * alpha;
  float nh2  = nh * nh;
  float tan2 = (1.0f - nh2) / nh2;
  float f    = a2 + tan2;
  return a2 / (f * f * M_PI * nh2 * nh);
}


struct EnvmapSampleValue
{
  float3 dir;
  float3 value;
  float  pdf;
};

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

  float alpha    = pc.roughness;
  uint  nsamples = alpha > 0.0F ? 512u : 1u;

  uint state = xxhash32(uint3(threadIdx.xy, uint(pc.roughness * 10.0F)));

  // The integrals are additionally weighted by the cosine and normalized using the average cosine of
  // the importance sampled BRDF directions (as in the Unreal publication).
  float weight_sum = 0.0f;

  float3 result       = float3(0.0F);
  float  inv_nsamples = 1.0F / float(nsamples);
  for(uint i = 0u; i < nsamples; ++i)
  {
    // Importance sample BRDF.
    {
      float xi0 = (float(i) + 0.5F) * inv_nsamples;
      float xi1 = rand(state);

      float3 h0 = alpha > 0.0f ? ggxSample(float2(xi0, xi1), alpha) : float3(0.0F, 0.0F, 1.0F);
      float3 h  = tangent * h0.x + bitangent * h0.y + normal * h0.z;

      float3 direction = normalize(2.0 * dot(normal, h) * h - normal);

      float cos_theta = dot(normal, direction);
      if(cos_theta > 0.0F)
      {
        float2 uv = getSphericalUv(direction);
        float  w  = 1.0F;
        if(alpha > 0.0F)
        {
          float pdf_brdf_sqr = ggxEval(alpha, h0.z) * 0.25F / dot(direction, h);
          pdf_brdf_sqr *= pdf_brdf_sqr;
          float pdf_env = hdrIBL.Sample(uv).a;
          w             = pdf_brdf_sqr / (pdf_brdf_sqr + pdf_env * pdf_env);
        }
        result += w * hdrIBL.Sample(uv).rgb * cos_theta;
        weight_sum += cos_theta;
      }
    }

    // Importance sample environment.
    if(alpha > 0.0f)
    {
      float3 randVal = float3(rand(state), rand(state), rand(state));

      EnvmapSampleValue val;
      float4            radPdf = environmentSample(hdrIBL, envSamplingData, randVal, val.dir);
      val.pdf                  = radPdf.a;
      val.value                = radPdf.rgb / val.pdf;


      float3 h  = normalize(normal + val.dir);
      float  nh = dot(h, normal);
      float  kh = dot(val.dir, h);
      float  nk = dot(val.dir, normal);
      if(kh > 0.0F && nh > 0.0F && nk > 0.0F)
      {
        float pdf_env_sqr = val.pdf * val.pdf;
        float pdf_brdf    = ggxEval(alpha, nh) * 0.25F / kh;

        float w = pdf_env_sqr / (pdf_env_sqr + pdf_brdf * pdf_brdf);
        result += w * val.value * pdf_brdf * nk * nk;
      }
    }
  }

  float4 result_color            = float4(result / float(weight_sum), 1.0F);
  g_outColor[int2(threadIdx.xy)] = result_color;
}
