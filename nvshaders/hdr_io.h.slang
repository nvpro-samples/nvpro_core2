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

#ifndef DH_HDR_H
#define DH_HDR_H 1

#include "slang_types.h"

NAMESPACE_SHADERIO_BEGIN()

#define HDR_WORKGROUP_SIZE 32


// Common structures used for HDR environment maps.

// Environment acceleration structure - computed in hdr_env
struct EnvAccel
{
  uint  alias;
  float q;
};

struct HdrPushBlock
{
  float4x4 mvp;
  float2   size;
  float    roughness;
  uint     numSamples;
};


struct HdrDomePushConstant
{
  float4x4 mvp;
  float4   multColor;
  float    blur;
};

enum EnvBindings
{
  eHdr        = 0,
  eImpSamples = 1
};

enum EnvDomeBindings
{
  eHdrBrdf     = 0,
  eHdrDiffuse  = 1,
  eHdrSpecular = 2
};

enum EnvDomeDraw
{
  eHdrImage = 0
};

// The GPU-image path samples from a *grid* that may be coarser than the image: each grid cell
// covers a block of image texels, and the alias table has one entry per cell. A smooth source (a
// sky with its sun sampled separately) loses nothing to a coarse grid, and the alias build costs
// what the grid costs rather than what the image costs. The grid must divide the image exactly, so
// every texel belongs to one cell; a grid equal to the image is the per-texel case. The producer
// supplies the per-cell importance -- see nvvk::HdrIbl::updateFromGpuImage.

// env_write_pdf.slang: writes into the lat-long image's alpha the PDF of the grid cell each texel
// falls in. With a grid equal to the image this is the per-texel formula of `createEnvironmentAccel`
// (nvvk/hdr_ibl.cpp), so the file-load and GPU-image paths agree for the sampling shader.
enum EnvWritePdfBindings
{
  eEnvWritePdfImage = 0,
  eEnvWritePdfCells = 1,  // StructuredBuffer<float>, the per-cell importance, read
};

struct EnvWritePdfPushConstant
{
  uint2 imageSize;    // width, height of the lat-long image
  uint2 gridSize;     // width, height of the sampling grid; divides imageSize
  float invIntegral;  // 1 / integral of the emitted radiance (energy normalization)
};

NAMESPACE_SHADERIO_END()

#endif  // DH_HDR_H
