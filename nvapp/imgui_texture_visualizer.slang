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

// Shader used by nvapp::ImTextureVisualizer. It draws a single screen-space
// quad (triangle strip, 4 vertices, no vertex buffer) and maps the texture
// channels according to the push constants.
//
// There are three fragment entry points, sampling a 2D texture, a layer of a 2D
// array texture, or a slice of a 3D texture. The sampler type is part of the
// shader, it can be neither a push nor a specialization constant, so the host
// creates one pipeline per entry point.
//
// This file is NOT compiled by the build system. The SPIR-V of all entry points
// lives in imgui_texture_visualizer.slang.h; after editing this file
// regenerate that header with (flags match cmake/CompilerSlangShader.cmake):
//
//   slangc -emit-spirv-directly -matrix-layout-row-major -force-glsl-scalar-layout -fvk-use-entrypoint-name -lang slang -profile spirv_1_6 -target spirv -g0 -O0 -source-embed-name imgui_texture_visualizer_slang -source-embed-style text -o imgui_texture_visualizer.slang.h imgui_texture_visualizer.slang

// Must match nvapp::ImTextureVisualizer::PushConstants
struct PushConstants
{
  float4 posRect;  // clip space rectangle, xy = min corner, zw = max corner
  float4 uvRect;   // texture coordinates, xy at posRect.xy, zw at posRect.zw
  float4 powExp;   // per channel exponent, applied last (gamma correction)
  float4 scale;    // per channel multiplier
  float4 add;      // per channel offset
  uint   swizzle;  // per channel source, one byte each (0:R 1:G 2:B 3:A 4:Zero 5:One)
  uint   invert;   // per channel bit, 1 means the channel is replaced by 1 - channel
  float  lod;      // mip level to sample
  uint   layer;    // array layer to sample, only used by fragmentArrayMain
  float  zCoord;   // normalized z coordinate to sample, only used by fragment3DMain
};

[[vk::push_constant]]
ConstantBuffer<PushConstants> pc;

// Both alias the same binding, each entry point only ever references one of them
[[vk::binding(0, 0)]]
Sampler2D texImage2D;
[[vk::binding(0, 0)]]
Sampler2DArray texImage2DArray;
[[vk::binding(0, 0)]]
Sampler3D texImage3D;

struct QuadVertex
{
  float4 position : SV_Position;
  float2 uv       : TEXCOORD0;
};

[shader("vertex")]
QuadVertex vertexMain(uint vertexIndex: SV_VertexID)
{
  // 0:(0,0) 1:(1,0) 2:(0,1) 3:(1,1), which is a valid triangle strip quad
  const float2 corner = float2(vertexIndex & 1, (vertexIndex >> 1) & 1);

  QuadVertex output;
  output.uv       = lerp(pc.uvRect.xy, pc.uvRect.zw, corner);
  output.position = float4(lerp(pc.posRect.xy, pc.posRect.zw, corner), 0.0, 1.0);
  return output;
}

float sourceChannel(float4 texel, uint source)
{
  switch(source)
  {
    case 0:
      return texel.r;
    case 1:
      return texel.g;
    case 2:
      return texel.b;
    case 3:
      return texel.a;
    case 4:
      return 0.0;
    default:
      return 1.0;
  }
}

// swizzle, then scale/bias, then gamma, then invert
float4 mapTexel(float4 texel)
{
  float4 color;
  color.r = sourceChannel(texel, (pc.swizzle >> 0) & 0xFF);
  color.g = sourceChannel(texel, (pc.swizzle >> 8) & 0xFF);
  color.b = sourceChannel(texel, (pc.swizzle >> 16) & 0xFF);
  color.a = sourceChannel(texel, (pc.swizzle >> 24) & 0xFF);

  color = color * pc.scale + pc.add;
  color = pow(max(color, float4(0.0)), pc.powExp);

  const bool4 invert = bool4((pc.invert & 1) != 0, (pc.invert & 2) != 0, (pc.invert & 4) != 0, (pc.invert & 8) != 0);
  return lerp(color, float4(1.0) - color, float4(invert));
}

[shader("fragment")]
float4 fragmentMain(QuadVertex input)
    : SV_Target
{
  return mapTexel(texImage2D.SampleLevel(input.uv, pc.lod));
}

[shader("fragment")]
float4 fragmentArrayMain(QuadVertex input)
    : SV_Target
{
  return mapTexel(texImage2DArray.SampleLevel(float3(input.uv, float(pc.layer)), pc.lod));
}

[shader("fragment")]
float4 fragment3DMain(QuadVertex input)
    : SV_Target
{
  return mapTexel(texImage3D.SampleLevel(float3(input.uv, pc.zCoord), pc.lod));
}
