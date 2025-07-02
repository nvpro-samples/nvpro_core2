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

#ifndef SIMPLE_SHADING_H
#define SIMPLE_SHADING_H

#include "slang_types.h"

// Performs shading calculations for a 3D surface point, taking into account light direction, view
// direction, surface color, and surface normal. It combines diffuse and specular components to
// determine the light intensity, multiplies it with the surface color, adds an ambient term (sky),
// and returns the final shaded color.
float3 simpleShading(float3 viewDir, float3 lightDir, float3 normal, float3 color SLANG_DEFAULT(float3(1, 1, 1)), float expo SLANG_DEFAULT(16.0))
{
  // Diffuse + Specular
  float3 reflDir = normalize(-reflect(lightDir, normal));
  float  lt      = saturate(dot(normal, lightDir)) + pow(max(0, dot(reflDir, viewDir)), expo);
  color *= lt;

  // Slight ambient term (sky effect)
  float3 skyUpDir    = float3(0, 1, 0);
  float3 groundColor = float3(0.1, 0.1, 0.4);
  float3 skyColor    = float3(0.8, 0.6, 0.2);
  color += lerp(skyColor, groundColor, dot(normal, skyUpDir.xyz) * 0.5 + 0.5) * 0.2;

  return color;
}

#endif