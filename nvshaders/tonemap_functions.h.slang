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


#ifndef TONEMAMP_FUNCTIONS_H
#define TONEMAMP_FUNCTIONS_H 1

#include "slang_types.h"
#include "tonemap_io.h.slang"


/*-------------------------------------------------------------------------------------------------
# Function `toSrgb`
> Converts a color from linear RGB to sRGB.
-------------------------------------------------------------------------------------------------*/
inline float3 toSrgb(float3 rgb)
{
  float3 low  = rgb * 12.92f;
  float3 high = fma(pow(rgb, float3(1.0F / 2.4F)), float3(1.055F), float3(-0.055F));
  return lerp(low, high, float3(greaterThan(rgb, float3(0.0031308F))));
}

/*-------------------------------------------------------------------------------------------------
# Function `toLinear`
> Converts a color from sRGB to linear RGB.
-------------------------------------------------------------------------------------------------*/
inline float3 toLinear(float3 srgb)
{
  float3 low  = srgb / 12.92F;
  float3 high = pow((srgb + float3(0.055F)) / 1.055F, float3(2.4F));
  return lerp(low, high, float3(greaterThan(srgb, float3(0.04045F))));
}

/*-------------------------------------------------------------------------------------------------
# Function `tonemapFilmic`
> Filmic tonemapping operator by Jim Hejl and Richard Burgess-Dawson,
> approximating the Digital Fusion Cineon mode, but more saturated and with
> darker midtones. sRGB correction is built in.

http://filmicworlds.com/blog/filmic-tonemapping-operators/
-------------------------------------------------------------------------------------------------*/
inline float3 tonemapFilmic(float3 color)
{
  float3 temp = max(float3(0.0F), color - float3(0.004F));
  float3 result = (temp * (float3(6.2F) * temp + float3(0.5F))) / (temp * (float3(6.2F) * temp + float3(1.7F)) + float3(0.06F));
  return result;
}

/*-------------------------------------------------------------------------------------------------
# Function `tonemapUncharted`
> Tone mapping operator from Uncharted 2 by John Hable. sRGB correction is built in.

See: http://filmicworlds.com/blog/filmic-tonemapping-operators/
-------------------------------------------------------------------------------------------------*/

inline float3 tonemapUncharted2Impl(float3 color)
{
  const float a = 0.15F;
  const float b = 0.50F;
  const float c = 0.10F;
  const float d = 0.20F;
  const float e = 0.02F;
  const float f = 0.30F;
  return ((color * (a * color + c * b) + d * e) / (color * (a * color + b) + d * f)) - e / f;
}

inline float3 tonemapUncharted2(float3 color)
{
  const float W             = 11.2F;
  const float exposure_bias = 2.0F;
  color                     = tonemapUncharted2Impl(color * exposure_bias);
  float3 white_scale        = float3(1.0F) / tonemapUncharted2Impl(float3(W));
  // We apply pow() here instead of calling toSrgb to match the
  // original implementation.
  return pow(color * white_scale, float3(1.0F / 2.2F));
}

/*-------------------------------------------------------------------------------------------------
# Function `tonemapACES`
> An approximation by Stephen Hill to the Academy Color Encoding System's
> filmic curve for displaying HDR images on LDR output devices.

From https://github.com/TheRealMJP/BakingLab/blob/master/BakingLab/ACES.hlsl,
via https://knarkowicz.wordpress.com/2016/01/06/aces-filmic-tone-mapping-curve/
-------------------------------------------------------------------------------------------------*/
inline float3 tonemapACES(float3 color)
{
  // Input transform
  const float3x3 ACESInputMat = float3x3(0.59719F, 0.07600F, 0.02840F,   // Row 1
                                         0.35458F, 0.90834F, 0.13383F,   // Row 2
                                         0.04823F, 0.01566F, 0.83777F);  // Row 3
  color                       = mul(color, ACESInputMat);

  // RRT and ODT fit
  float3 a = color * (color + float3(0.0245786F)) - float3(0.000090537F);
  float3 b = color * (float3(0.983729F) * color + float3(0.4329510F)) + float3(0.238081F);
  color    = a / b;  // Always OK because of the large constant term in b's polynomial
  // Output transform
  const float3x3 ACESOutputMat = float3x3(1.60475F, -0.10208F, -0.00327F,  //
                                          -0.53108F, 1.10813F, -0.07276F,  //
                                          -0.07367F, -0.00605F, 1.07602F);
  color                        = mul(color, ACESOutputMat);
  return toSrgb(color);
}

/*-------------------------------------------------------------------------------------------------
# Function `tonemapAgX`
> Benjamin Wrensch's approximation to the AgX tone mapping curve by Troy Sobotka.

From https://iolite-engine.com/blog_posts/minimal_agx_implementation
-------------------------------------------------------------------------------------------------*/
inline float3 tonemapAgX(float3 color)
{
  // Input transform
  const float3x3 agx_mat = float3x3(0.842479062253094F, 0.0423282422610123F, 0.0423756549057051F,  //
                                    0.0784335999999992F, 0.878468636469772F, 0.0784336F,           //
                                    0.0792237451477643F, 0.0791661274605434F, 0.879142973793104F);
  color                  = mul(color, agx_mat);

  // Log2 space encoding
  const float min_ev = -12.47393f;
  const float max_ev = 4.026069f;
  color              = clamp(log2(color), min_ev, max_ev);
  color              = (color - min_ev) / (max_ev - min_ev);

  // Apply 6th-order sigmoid function approximation
  float3 v = fma(float3(15.5F), color, float3(-40.14F));
  v        = fma(color, v, float3(31.96F));
  v        = fma(color, v, float3(-6.868F));
  v        = fma(color, v, float3(0.4298F));
  v        = fma(color, v, float3(0.1191F));
  v        = fma(color, v, float3(-0.0023F));

  // Output transform
  const float3x3 agx_mat_inv = float3x3(1.19687900512017F, -0.0528968517574562F, -0.0529716355144438F,  //
                                        -0.0980208811401368F, 1.15190312990417F, -0.0980434501171241F,  //
                                        -0.0990297440797205F, -0.0989611768448433F, 1.15107367264116F);
  v                          = mul(v, agx_mat_inv);

  // Skip the pow(..., float3(2.2)), because we want sRGB output here.
  return v;
}

/*-------------------------------------------------------------------------------------------------
# Function `tonemapKhronosPBR`
> The Khronos PBR neutral tone mapper.

Adapted from https://github.com/KhronosGroup/ToneMapping/blob/main/PBR_Neutral/pbrNeutral.glsl
-------------------------------------------------------------------------------------------------*/
inline float3 tonemapKhronosPBR(float3 color)
{
  const float startCompression = 0.8F - 0.04F;
  const float desaturation     = 0.15F;

#ifdef __cplusplus
  float x    = glm::min(color.x, glm::min(color.y, color.z));
  float peak = glm::max(color.x, glm::max(color.y, color.z));
#else
  float x    = min(color.x, min(color.y, color.z));
  float peak = max(color.x, max(color.y, color.z));
#endif

  float offset = x < 0.08F ? x * (-6.25F * x + 1.F) : 0.04F;
  color -= offset;

  if(peak >= startCompression)
  {
    const float d       = 1.F - startCompression;
    float       newPeak = 1.F - d * d / (peak + d - startCompression);
    color *= newPeak / peak;

    float g = 1.F - 1.F / (desaturation * (peak - newPeak) + 1.F);
    color   = lerp(color, float3(newPeak), float3(g));
  }
  return toSrgb(color);
}

/*-------------------------------------------------------------------------------------------------
> Applies the given tone mapper and color grading settings to a given color.

Requires the UV coordinate so that it can apply vignetting.
-------------------------------------------------------------------------------------------------*/
inline float3 applyTonemap(TonemapperData tm, float3 color, float2 uv)
{
  // Exposure
  color *= tm.exposure;
  float3 c;
  // Tonemap
  switch(tm.method)
  {
    case ToneMapMethod::eFilmic:
      c = tonemapFilmic(color);
      break;
    case ToneMapMethod::eUncharted2:
      c = tonemapUncharted2(color);
      break;
    case ToneMapMethod::eClip:
      c = toSrgb(color);
      break;
    case ToneMapMethod::eACES:
      c = tonemapACES(color);
      break;
    case ToneMapMethod::eAgX:
      c = tonemapAgX(color);
      break;
    case ToneMapMethod::eKhronosPBR:
      c = tonemapKhronosPBR(color);
      break;
  }

  // contrast and clamp
  c = clamp(lerp(float3(0.5F), c, float3(tm.contrast)), float3(0.F), float3(1.F));
  // brightness
  c = pow(c, float3(1.0F / tm.brightness));
  // saturation
  float3 i = float3(dot(c, float3(0.299F, 0.587F, 0.114F)));
  c        = lerp(i, c, float3(tm.saturation));
  // vignette
  float2 center_uv = ((uv)-float2(0.5F)) * float2(2.0F);
  c *= 1.0F - dot(center_uv, center_uv) * tm.vignette;

  return c;
}

#endif  // TONEMAMP_FUNCTIONS_H
