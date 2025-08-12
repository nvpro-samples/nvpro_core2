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

#ifndef SKY_SHADERIO_H
#define SKY_SHADERIO_H 1

#include "slang_types.h"

NAMESPACE_SHADERIO_BEGIN()

/*-------------------------------------------------------------------------------------------------
> Contains structures and functions for procedural sky models.

This file includes two sky models: a simple sky that is fast to compute, and
a more complex "physical sky" model based on a model from Mental Ray and later
modernized in the MDL SDK starting [here](https://github.com/NVIDIA/MDL-SDK/blob/203d5140b1dee89de17b26e828c4333571878629/src/shaders/mdl/base/base.mdl#L1180)
customized for nvpro-samples.
-------------------------------------------------------------------------------------------------*/

#if defined(GL_core_profile)  // GLSL
#define eSkyOutImage 0
#else
enum SkyBindings
{
  eSkyOutImage = 0,
};
#endif

/*-------------------------------------------------------------------------------------------------
# Struct `SkySamplingResult`
> Contains the resulting direction, probability density function, and radiance
> from sampling the procedural sky.
-------------------------------------------------------------------------------------------------*/
struct SkySamplingResult
{
  float3 direction;  // Direction to the sampled light
  float  pdf;        // Probability Density Function value
  float3 radiance;   // Light intensity
};


/*-------------------------------------------------------------------------------------------------
# Struct `SkySimpleParameters`
> Parameters for the simple sky model.
-------------------------------------------------------------------------------------------------*/
struct SkySimpleParameters
{
  float3 sunDirection      SLANG_DEFAULT(float3(-1.23413404e-08F, 0.707106829F, 0.707106709F));
  float angularSizeOfLight SLANG_DEFAULT(0.059F);

  float3 sunColor SLANG_DEFAULT(float3(1.0F, 1.0F, 1.0F));
  float glowSize  SLANG_DEFAULT(0.091F);

  float3 skyColor     SLANG_DEFAULT(float3(0.17F, 0.37F, 0.65F));
  float glowIntensity SLANG_DEFAULT(0.9F);

  float3 horizonColor SLANG_DEFAULT(float3(0.50F, 0.70F, 0.92F));
  float horizonSize   SLANG_DEFAULT(0.5F);

  float3 groundColor  SLANG_DEFAULT(float3(0.62F, 0.59F, 0.55F));
  float glowSharpness SLANG_DEFAULT(4.F);

  float3 directionUp SLANG_DEFAULT(float3(0.F, 1.F, 0.F));
  float sunIntensity SLANG_DEFAULT(1.0f);

  float3 lightRadiance SLANG_DEFAULT(float3(1.0F, 1.0F, 1.0F));
  float brightness     SLANG_DEFAULT(1.0F);
};


////////////////////////////////////////////////////////////////////////
// Physical Sky
////////////////////////////////////////////////////////////////////////

/*-------------------------------------------------------------------------------------------------
# Struct `SkyPhysicalParameters`
> Parameters for the physical sky model.
-------------------------------------------------------------------------------------------------*/
struct SkyPhysicalParameters
{
  float3 rgbUnitConversion SLANG_DEFAULT(float3(1.0f / 80000.0f));
  float multiplier         SLANG_DEFAULT(0.1f);

  float haze          SLANG_DEFAULT(0.1f);
  float redblueshift  SLANG_DEFAULT(0.1f);
  float saturation    SLANG_DEFAULT(1.0f);
  float horizonHeight SLANG_DEFAULT(0.0f);

  float3 groundColor SLANG_DEFAULT(float3(0.4f));
  float horizonBlur  SLANG_DEFAULT(0.3f);

  float3 nightColor      SLANG_DEFAULT(float3(0.0f, 0.0f, 0.01f));
  float sunDiskIntensity SLANG_DEFAULT(1.0f);

  float3 sunDirection SLANG_DEFAULT(float3(-1.23413404e-08F, 0.707106829F, 0.707106709F));
  float sunDiskScale  SLANG_DEFAULT(1.0f);

  float sunGlowIntensity SLANG_DEFAULT(1.0f);
  int yIsUp              SLANG_DEFAULT(1);
};

NAMESPACE_SHADERIO_END()

#endif  // SKY_SHADERIO_H
