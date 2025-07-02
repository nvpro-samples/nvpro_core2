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

#include "sky_functions.h.slang"

// Push Constant block (combined struct to match Vulkan)
struct PushConstants
{
    SkyPhysicalParameters skyParams;
    float4x4 transformMatrix; // This starts at offset sizeof(SkySimpleParameters)
};

// clang-format off
[[vk::push_constant]]                       ConstantBuffer<PushConstants>   pushConst;
[[vk::binding(SkyBindings::eSkyOutImage)]]  RWTexture2D<float4>             outImage;
// clang-format on

[shader("compute")]
[numthreads(WORKGROUP_SIZE, WORKGROUP_SIZE, 1)]
void main(uint3 globalThreadID: SV_DispatchThreadID)
{
    uint2 imageSize;
    outImage.GetDimensions(imageSize.x, imageSize.y);

    if (globalThreadID.x >= imageSize.x || globalThreadID.y >= imageSize.y)
        return;

    float2 screenPos = (float2(globalThreadID.xy) + 0.5F) / float2(imageSize) * 2.0 - 1.0;

    float4 transformed = mul(float4(screenPos, 1.0F, 1.0F), pushConst.transformMatrix);
    float3 rayDir      = normalize(transformed.xyz);

    // Evaluate sky color
    float3 skyColor = evalPhysicalSky(pushConst.skyParams, rayDir);

    // Store result
    outImage[int2(globalThreadID.xy)] = float4(skyColor, 1);
}

