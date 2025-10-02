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

#ifndef NORMAL_COMPRESS_H
#define NORMAL_COMPRESS_H 1


#include "slang_types.h"

NAMESPACE_SHADERIO_BEGIN()


//
// Normal vector compression utility using octahedral mapping.
//
// This encoding projects a unit-length normal vector onto the surface of an octahedron,
// then flattens the octahedron to 2D, mapping the normal to a pair of signed 16-bit integers.
// These are packed into a single 32-bit unsigned integer. For normals in the negative Z hemisphere,
// a reflection is performed to preserve bijectivity. This approach provides efficient storage
// and good angular precision for normal vectors, as described in "Octahedron Environment Maps"
// by Engelhardt & Dachsbacher.
//

#define C_Stack_Max 3.402823466e+38f  // Maximum float value used as invalid marker

// Compresses a unit normal vector into a 32-bit unsigned integer.
// Returns ~0u for invalid input.
uint compressUnitVec(float3 nv)
{
  // Validate input vector
  if((nv.x < C_Stack_Max) && !isinf(nv.x))
  {
    // Project to octahedron surface and normalize to [-32767, 32767] range
    const float d = 32767.0f / (abs(nv.x) + abs(nv.y) + abs(nv.z));
    int         x = int(round(nv.x * d));
    int         y = int(round(nv.y * d));
    
    // Handle negative Z hemisphere by reflecting across octahedron edges
    if(nv.z < 0.0f)
    {
      const int maskx = x >> 31;  // Sign bit mask
      const int masky = y >> 31;
      const int tmp   = 32767 + maskx + masky;
      const int tmpx  = x;
      x               = (tmp - (y ^ masky)) ^ maskx;
      y               = (tmp - (tmpx ^ maskx)) ^ masky;
    }
    
    // Pack into 32-bit uint: Y in upper 16 bits, X in lower 16 bits
    uint packed = (uint(y + 32767) << 16) | uint(x + 32767);
    
    // Avoid reserved invalid value
    if(packed == ~0u)
      return ~0x1u;
    return packed;
  }
  else
  {
    return ~0u;  // Invalid input marker
  }
}

// Converts a 16-bit signed integer to float in range [-1, 1].
// Uses bit manipulation for efficient conversion without division.
float ShortToFloatM11(const int v)
{
  return (v >= 0) ? (asfloat(0x3F800000u | (uint(v) << 8)) - 1.0f) :
                    (asfloat((0x80000000u | 0x3F800000u) | (uint(-v) << 8)) + 1.0f);
}

// Decompresses a 32-bit packed normal back to a unit vector.
// Reverses the octahedral mapping compression process.
float3 decompressUnitVec(uint packed)
{
  if(packed != ~0u)  // Check for invalid marker
  {
    // Unpack X and Y coordinates from 16-bit components
    int       x     = int(packed & 0xFFFFu) - 32767;
    int       y     = int(packed >> 16) - 32767;
    
    // Reconstruct Z coordinate using octahedral math
    const int maskx = x >> 31;  // Sign bit masks
    const int masky = y >> 31;
    const int tmp0  = 32767 + maskx + masky;
    const int ymask = y ^ masky;
    const int tmp1  = tmp0 - (x ^ maskx);
    const int z     = tmp1 - ymask;
    
    float zf;
    if(z < 0)
    {
      // Negative Z hemisphere: apply reflection transformation
      x  = (tmp0 - ymask) ^ maskx;
      y  = tmp1 ^ masky;
      zf = asfloat((0x80000000u | 0x3F800000u) | (uint(-z) << 8)) + 1.0f;
    }
    else
    {
      // Positive Z hemisphere: direct conversion
      zf = asfloat(0x3F800000u | (uint(z) << 8)) - 1.0f;
    }
    
    // Convert back to float coordinates and normalize to ensure unit length
    return normalize(float3(ShortToFloatM11(x), ShortToFloatM11(y), zf));
  }
  else
  {
    return float3(C_Stack_Max);  // Return invalid marker
  }
}

NAMESPACE_SHADERIO_END()
#endif
