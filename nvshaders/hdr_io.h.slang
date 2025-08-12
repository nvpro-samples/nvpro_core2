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
  float    rotation;
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

NAMESPACE_SHADERIO_END()

#endif  // DH_HDR_H
