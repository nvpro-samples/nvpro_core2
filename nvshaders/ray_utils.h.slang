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

#ifndef RAY_UTIL_H
#define RAY_UTIL_H 1

/*-------------------------------------------------------------------------------------------------
# Function `offsetRay`
> Adjusts the origin `p` of the ray `p + t*n` so that it is unlikely to
> intersect with a triangle that passes through `p`, but tries not to
> noticeably affect visual results.

For a more sophisticated algorithm, see
"A Fast and Robust Method for Avoiding Self-Intersection" by
Carsten Wachter and Nikolaus Binder in Ray Tracing Gems volume 1.
-------------------------------------------------------------------------------------------------*/
float3 offsetRay(in float3 p, in float3 n)
{
  // Smallest epsilon that can be added without losing precision is 1.19209e-07, but we play safe
  const float epsilon = 1.0f / 65536.0f;  // Safe epsilon

  float magnitude = length(p);
  float offset    = epsilon * magnitude;
  // multiply the direction vector by the smallest offset
  float3 offsetVector = n * offset;
  // add the offset vector to the starting point
  float3 offsetPoint = p + offsetVector;

  return offsetPoint;
}

/*-------------------------------------------------------------------------------------------------
# Function `pointOffset`
> Adjusts a position so that shadows match interpolated normals more closely.

This technique comes from ["Hacking the shadow terminator"](https://jo.dreggn.org/home/2021_terminator.pdf) by Johannes Hanika.

Inputs:
- `p`: Point of intersection on a triangle.
- `p[a..c]`: Positions of the triangle at each vertex.
- `n[a..c]`: Normals of the triangle at each vertex.
- `bary`: Barycentric coordinate of the hit position.

Returns the new position.
-------------------------------------------------------------------------------------------------*/
float3 pointOffset(float3 p, float3 pa, float3 pb, float3 pc, float3 na, float3 nb, float3 nc, float3 bary)
{
  float3 tmpu = p - pa;
  float3 tmpv = p - pb;
  float3 tmpw = p - pc;

  float dotu = min(0.0F, dot(tmpu, na));
  float dotv = min(0.0F, dot(tmpv, nb));
  float dotw = min(0.0F, dot(tmpw, nc));

  tmpu -= dotu * na;
  tmpv -= dotv * nb;
  tmpw -= dotw * nc;

  float3 pP = p + tmpu * bary.x + tmpv * bary.y + tmpw * bary.z;

  return pP;
}


#endif
