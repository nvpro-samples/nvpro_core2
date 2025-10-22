/*
 * Copyright (c) 2023-2025, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

/// Computes the sum of absolute screen-space derivatives in both x and y directions
/// This function calculates the rate of change of a value across the screen, which is useful for
/// anti-aliasing, edge detection, and texture filtering. It follows the OpenGL fwidth specification.
///
/// @param bary Input value (typically barycentric coordinates or any per-pixel varying value)
/// @return The sum of absolute derivatives: abs(ddx(bary)) + abs(ddy(bary))
///
/// Reference: https://registry.khronos.org/OpenGL-Refpages/gl4/html/fwidth.xhtml
/// Note: This function has the same shader stage validity as ddx and ddy:
/// Reference: https://docs.shader-slang.org/en/latest/external/core-module-reference/global-decls/ddx.html

float3 fwidth(float3 bary)
{
  return abs(ddx(bary)) + abs(ddy(bary));
}
