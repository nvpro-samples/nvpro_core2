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

#ifndef SAMPLE_BLUR_H
#define SAMPLE_BLUR_H 1


/*-------------------------------------------------------------------------------------------------
# Function `sampleBlur`
> Samples a texture with a Gaussian blur kernel.
-------------------------------------------------------------------------------------------------*/

#ifndef __cplusplus
float4 sampleBlur(Sampler2D texture, float2 uv, float lodLevel)
{
  // G(x, y) = (1 / (2 * pi * sigma^2)) * exp(-(x^2 + y^2) / (2 * sigma^2))
  // Gaussian blur kernel normalized
  const float3x3 WEIGHTS_2D = float3x3(0.0625, 0.125, 0.0625, 0.125, 0.25, 0.125, 0.0625, 0.125, 0.0625);

  uint2 image_size;
  uint  numberOfLevels;
  texture.GetDimensions(uint(lodLevel), image_size.x, image_size.y, numberOfLevels);

  float2 texelSize = 1.0 / float2(image_size);

  float4 color = float4(0.0);
  [[unroll]] for(int i = 0; i < 3; i++)
  {
    [[unroll]] for(int j = 0; j < 3; j++)
    {
      float2 offsetUV = float2(i - 1, j - 1) * texelSize;
      color += texture.SampleLevel(uv + offsetUV, lodLevel) * WEIGHTS_2D[i][j];
    }
  }
  return color;
}

/*-------------------------------------------------------------------------------------------------
# Function `smoothHDRBlur`
> Samples a texture with a Gaussian blur kernel, using multiple LOD levels.
* The blur amount controls the blending between the two LOD levels.
-------------------------------------------------------------------------------------------------*/
float4 smoothHDRBlur(Sampler2D texture, float2 uv, float blurAmount)
{
  // Calculate the maximum LOD level
  uint width, height, mipLevels;
  texture.GetDimensions(0, width, height, mipLevels);

  float maxLOD = float(mipLevels) - 1.0;

  // Calculate two adaptive LOD levels
  float lod0 = max(0, (maxLOD * blurAmount) - 2);
  float lod1 = maxLOD * blurAmount;

  // Sample multiple adaptive mip levels
  float4 color0 = sampleBlur(texture, uv, lod0);
  float4 color1 = sampleBlur(texture, uv, lod1);

  // Blend between two mip levels, each of which depend on `blurAmount`.
  float  blurMix      = 1.0 - pow(1.0 - blurAmount, 1.0 / 1.5);
  float4 blendedColor = lerp(color0, color1, 0.5);

  return blendedColor;
}

#endif  // __cplusplus

#endif  // SAMPLE_BLUR_H
