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


#include "hdr_io.h.slang"
#include "constants.h.slang"
#include "functions.h.slang"
#include "random.h.slang"
#include "hdr_env_sampling.h.slang"

// clang-format off
[[vk::binding(EnvDomeDraw::eHdrImage, 0)]]      RWTexture2D<float4> g_outColor;
[[vk::binding(EnvBindings::eHdr, 1)]]           Sampler2D hdrTexture;
[[vk::binding(EnvBindings::eImpSamples, 1)]]    StructuredBuffer<EnvAccel> envSamplingData;
[[vk::push_constant]]                           ConstantBuffer<HdrPushBlock> pc;
// clang-format on


[shader("compute")]
[numthreads(HDR_WORKGROUP_SIZE, HDR_WORKGROUP_SIZE, 1)]
void main(uint3 threadIdx: SV_DispatchThreadID)
{
  // Finding the world direction
  const float2 pixelCenter = float2(threadIdx.xy) + float2(0.5F);
  const float2 inUV        = pixelCenter / pc.size;
  const float2 d           = inUV * 2.0F - 1.0F;
  float3       direction   = float3(mul(float4(d.x, d.y, 1.0F, 1.0F), pc.mvp).xyz);

  // Getting axis
  float3 tangent, bitangent;
  float3 normal = normalize(float3(direction.x, -direction.y, direction.z));  // Flipping Y
  orthonormalBasis(normal, tangent, bitangent);

  // Random seed
  uint seed = xxhash32(uint3(threadIdx.xyz));


  float3 result      = float3(0.0f);
  uint   nsamples    = 512u;
  float  inv_samples = 1.0f / float(nsamples);
  for(uint i = 0u; i < nsamples; ++i)
  {
    // Importance sample diffuse BRDF.
    {
      float xi0 = (float(i) + 0.5f) * inv_samples;
      float xi1 = rand(seed);

      float phi     = 2.0f * M_PI * xi0;
      float sin_phi = sin(phi);
      float cos_phi = cos(phi);

      float sin_theta = sqrt(1.0f - xi1);
      float cos_theta = sqrt(xi1);

      float3 d = float3(sin_theta * cos_phi, sin_theta * sin_phi, cos_theta);

      float3 direction = d.x * tangent + d.y * bitangent + d.z * normal;
      float2 uv        = getSphericalUv(direction);

      float p_brdf_sqr = cos_theta * cos_theta * (M_1_OVER_PI * M_1_OVER_PI);

      float4 rad_pdf = hdrTexture.Sample(uv);
      float  p_env   = rad_pdf.a;
      float  w       = p_brdf_sqr / (p_brdf_sqr + p_env * p_env);
      result += rad_pdf.rgb * w * M_PI;
    }

    // Importance sample environment.
    {
      float3 dir;
      float3 rand_val = float3(rand(seed), rand(seed), rand(seed));
      float4 rad_pdf  = environmentSample(hdrTexture, envSamplingData, rand_val, dir);
      float  pdf      = rad_pdf.a;
      float3 value    = rad_pdf.rgb / pdf;

      float cosine     = dot(dir, normal);
      float p_brdf_sqr = cosine * cosine * (M_1_OVER_PI * M_1_OVER_PI);
      float p_env_sqr  = pdf * pdf;
      if(cosine > 0.0f)
      {
        float w = p_env_sqr / (p_env_sqr + p_brdf_sqr);
        result += w * value * cosine;
      }
    }
  }

  float4 frag_color        = float4(result * (1.0f / float(nsamples)) / M_PI, 1.0f);
  g_outColor[threadIdx.xy] = frag_color;
}
