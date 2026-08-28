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


#ifndef TONEMAMP_SHADERIO_H
#define TONEMAMP_SHADERIO_H 1

#include "slang_types.h"

NAMESPACE_SHADERIO_BEGIN()

#define TONEMAP_WORKGROUP_SIZE 16


#if defined(GL_core_profile)  // GLSL
// GLSL has no enums, so the tone map methods become macros. They carry the enum name as
// prefix to keep the global macro namespace clean; TONEMAP_METHOD(eFilmic) resolves to
// the macro here and to ToneMapMethod::eFilmic everywhere else.
#define TONEMAP_METHOD(name) ToneMapMethod_##name
#define ToneMapMethod_eFilmic 0
#define ToneMapMethod_eUncharted2 1
#define ToneMapMethod_eClip 2
#define ToneMapMethod_eACES 3
#define ToneMapMethod_eAgX 4
#define ToneMapMethod_eKhronosPBR 5
#define ToneMapMethod_eCount 6
#else
#define TONEMAP_METHOD(name) ToneMapMethod::name

enum ToneMapMethod
{
  eFilmic = 0,
  eUncharted2,
  eClip,
  eACES,
  eAgX,
  eKhronosPBR,
  eCount
};


// Bindings only relevant to `tonemapper.slang`
enum TonemapBinding
{
  eImageInput = 0,
  eImageOutput,
  eHistogramInputOutput,
  eLuminanceInputOutput,
};
#endif


// Tonemapper settings.
struct TonemapperData
{
  int isActive SLANG_DEFAULT(1);
  int method   SLANG_DEFAULT(TONEMAP_METHOD(eFilmic));

  // Input color correction
  float exposure SLANG_DEFAULT(1.0f);
  float temperature SLANG_DEFAULT(6506.11144f);  // Color temperature in degrees Kelvin. The default produces sRGB white for our approximation
  float tint SLANG_DEFAULT(3.25895312e-3f);  // Tint adjustment as an ANSI C78.377-2008 Duv value. The default produces sRGB white for our approximation
  float3x3 inputMatrix SLANG_DEFAULT(float3x3(0));  // Will be computed from the above

  // Post effects
  float contrast   SLANG_DEFAULT(1.0f);
  float brightness SLANG_DEFAULT(1.0f);
  float saturation SLANG_DEFAULT(1.0f);
  float vignette   SLANG_DEFAULT(0.0f);

  // Advanced color grading
  float vibrance      SLANG_DEFAULT(0.0f);  // Boosts desaturated colors while preserving saturated ones
  float shadowBias    SLANG_DEFAULT(0.0f);  // -1 to 1: Adjust shadow tones
  float midtoneBias   SLANG_DEFAULT(0.0f);  // -1 to 1: Adjust midtone brightness
  float highlightBias SLANG_DEFAULT(0.0f);  // -1 to 1: Adjust highlight tones
  float3 coolColor    SLANG_DEFAULT(float3(1.0f, 1.0f, 1.0f));  // Tint for shadows in split toning
  float3 warmColor    SLANG_DEFAULT(float3(1.0f, 1.0f, 1.0f));  // Tint for highlights in split toning
  float splitBalance  SLANG_DEFAULT(0.0f);  // -0.5 to 0.5: Balance between cool and warm tones

  // Auto-Exposure
  int      autoExposure         SLANG_DEFAULT(0);      // 0: off, 1: on
  float    autoExposureSpeed    SLANG_DEFAULT(5.0F);   // Speed of auto-exposure adjustment
  float    evMinValue           SLANG_DEFAULT(-5.0F);  // Minimum EV value for auto-exposure
  float    evMaxValue           SLANG_DEFAULT(10.0F);  // Maximum EV value for auto-exposure
  uint32_t enableCenterMetering SLANG_DEFAULT(0);      // 0 = off, 1 = on
  float    centerMeteringSize   SLANG_DEFAULT(0.5F);   // Size of the center metering area (0.0 to 1.0, where 1.0 is the full screen)
  uint32_t averageMode          SLANG_DEFAULT(1);      // 0 = Mean, 1 = Median
  // Dither
  int dither SLANG_DEFAULT(1);  // 0: no dither, 1: dither
};


#define EXPOSURE_HISTOGRAM_SIZE 256

NAMESPACE_SHADERIO_END()


#endif  // TONEMAMP_SHADERIO_H
