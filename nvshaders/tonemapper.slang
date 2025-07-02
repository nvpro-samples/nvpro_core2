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

#define WORKGROUP_SIZE 16

#include "nvshaders/tonemap_functions.h.slang"

// clang-format off
[[vk::push_constant]]                       ConstantBuffer<TonemapperData>  tm;
[[vk::binding(TonemapBinding::eInput)]]     Texture2D<float4>               inImage;
[[vk::binding(TonemapBinding::eOutput)]]    RWTexture2D<float4>             outImage;
// clang-format on

[shader("compute")]
[numthreads(WORKGROUP_SIZE, WORKGROUP_SIZE, 1)]
void main(uint3 globalThreadID: SV_DispatchThreadID)
{
  uint2 imageSize;
  outImage.GetDimensions(imageSize.x, imageSize.y);

  if(globalThreadID.x >= imageSize.x || globalThreadID.y >= imageSize.y)
    return;

  float4 color = inImage[int2(globalThreadID.xy)];

  if(tm.isActive == 1)
  {
    float2 screenPos = float2(globalThreadID.xy) / float2(imageSize);
    color.xyz        = applyTonemap(tm, color.xyz, screenPos);
  }

  outImage[int2(globalThreadID.xy)] = color;
}

