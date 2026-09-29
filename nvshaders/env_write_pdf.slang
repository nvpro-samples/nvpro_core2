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


// PDF writer for the GPU-image environment-sampling path.
//
// Companion to `nvvk::HdrIbl::updateFromGpuImage`: after the alias table has been built on the
// CPU from the per-cell importance and the integral is known, this pass writes into the lat-long
// image's alpha the PDF of the grid cell each texel falls in, so the sampling shader
// (`hdr_env_sampling.h.slang`) and the MIS weight on a BSDF ray that escapes both read the density
// the alias table actually samples.
//
// Sampling picks a cell with probability importance(cell) / integral, then a direction uniformly
// in solid angle within it, so the density per steradian is constant over the cell:
//
//   PDF = importance(cell) / (integral * solidAngle(cell))
//
// With the grid equal to the image, importance(cell) = solidAngle * max(rgb) and this reduces to
// max(rgb) / integral -- the formula of `createEnvironmentAccel` (nvvk/hdr_ibl.cpp), so the two
// producers agree.

#include "hdr_io.h.slang"
#include "functions.h.slang"  // texelSolidAngle

// clang-format off
[[vk::binding(EnvWritePdfBindings::eEnvWritePdfImage, 0)]] RWTexture2D<float4>                        g_image;
[[vk::binding(EnvWritePdfBindings::eEnvWritePdfCells, 0)]] StructuredBuffer<float>                    g_cells;
[[vk::push_constant]]                                       ConstantBuffer<EnvWritePdfPushConstant>   pc;
// clang-format on


[shader("compute")]
[numthreads(HDR_WORKGROUP_SIZE, HDR_WORKGROUP_SIZE, 1)]
void main(uint3 threadIdx : SV_DispatchThreadID)
{
  if(any(threadIdx.xy >= pc.imageSize))
    return;

  const uint2 block = pc.imageSize / pc.gridSize;  // texels per cell; exact, the host checks
  const uint2 cell  = threadIdx.xy / block;

  const float importance = g_cells[cell.y * pc.gridSize.x + cell.x];
  const float solidAngle = texelSolidAngle(cell.y, pc.gridSize.x, pc.gridSize.y);

  float4 c              = g_image[threadIdx.xy];
  c.a                   = importance * pc.invIntegral / solidAngle;
  g_image[threadIdx.xy] = c;
}
