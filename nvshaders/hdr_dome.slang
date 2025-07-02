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
#include "sample_blur.h.slang"

// clang-format off
[[vk::binding(EnvDomeDraw::eHdrImage, 0)]]  RWTexture2D<float4> gOutHdr;
[[vk::binding(EnvBindings::eHdr, 1)]]       Sampler2D gInHdr;
[[vk::push_constant]]                       ConstantBuffer<HdrDomePushConstant> pc;
// clang-format on


[shader("compute")]
[numthreads(HDR_WORKGROUP_SIZE, HDR_WORKGROUP_SIZE, 1)]
void main(uint3 threadIdx: SV_DispatchThreadID)
{
  const float2 pixel_center = float2(threadIdx.xy) + float2(0.5);

  int2 image_size;
  gOutHdr.GetDimensions(image_size.x, image_size.y);

  const float2 in_uv     = pixel_center / float2(image_size);
  const float2 d         = in_uv * 2.0F - 1.0F;
  float3       direction = float3(mul(float4(d.x, d.y, 1.0F, 1.0F), pc.mvp).xyz);

  direction = rotate(direction, float3(0.0F, 1.0F, 0.0F), -pc.rotation);

  const float2 uv = getSphericalUv(normalize(direction.xyz));
  float3       color;
  if(pc.blur > 0.0F)
  {
    color = smoothHDRBlur(gInHdr, uv, pc.blur).rgb;
  }
  else
  {
    color = gInHdr.SampleLevel(uv, 0.0f).rgb;
  }
  color *= pc.multColor.rgb;
  gOutHdr[threadIdx.xy] = float4(color, 1.0);
}
