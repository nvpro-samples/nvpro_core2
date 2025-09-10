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


#ifndef TONEMAMP_SHADERIO_H
#define TONEMAMP_SHADERIO_H 1

#include "slang_types.h"

NAMESPACE_SHADERIO_BEGIN()

#define TONEMAP_WORKGROUP_SIZE 16


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


// Bindings
enum TonemapBinding
{
  eImageInput = 0,
  eImageOutput,
  eHistogramInputOutput,
  eLuminanceInputOutput,
};


// Tonemapper settings.
struct TonemapperData
{
  int isActive = 1;
  int method   = ToneMapMethod::eFilmic;

  // Input color correction
  float exposure = 1.0f;
  float temperature = 6506.11144f;  // Color temperature in degrees Kelvin. The default produces sRGB white for our approximation
  float tint = 3.25895312e-3f;  // Tint adjustment as an ANSI C78.377-2008 Duv value. The default produces sRGB white for our approximation
  float3x3 inputMatrix = {};  // Will be computed from the above

  // Post effects
  float contrast   = 1.0f;
  float brightness = 1.0f;
  float saturation = 1.0f;
  float vignette   = 0.0f;

  // Auto-Exposure
  int      autoExposure         = 0;      // 0: off, 1: on
  float    autoExposureSpeed    = 5.0F;   // Speed of auto-exposure adjustment
  float    evMinValue           = -5.0F;  // Minimum EV value for auto-exposure
  float    evMaxValue           = 10.0F;  // Maximum EV value for auto-exposure
  uint32_t enableCenterMetering = 0;      // 0 = off, 1 = on
  float    centerMeteringSize   = 0.5F;   // Size of the center metering area (0.0 to 1.0, where 1.0 is the full screen)
  uint32_t averageMode          = 1;      // 0 = Mean, 1 = Median
  // Dither
  int dither = 1;  // 0: no dither, 1: dither
};


#define EXPOSURE_HISTOGRAM_SIZE 256

NAMESPACE_SHADERIO_END()


#endif  // TONEMAMP_SHADERIO_H
