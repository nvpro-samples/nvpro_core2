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

NAMESPACE_SHADERIO_BEGIN()

// Converts a color from linear RGB to sRGB.
inline float3 toSrgb(float3 rgb)
{
  float3 low  = rgb * 12.92f;
  float3 high = fma(pow(rgb, float3(1.0F / 2.4F)), float3(1.055F), float3(-0.055F));
  return lerp(low, high, float3(greaterThan(rgb, float3(0.0031308F))));
}

// Converts a color from sRGB to linear RGB.
inline float3 toLinear(float3 srgb)
{
  float3 low  = srgb / 12.92F;
  float3 high = pow((srgb + float3(0.055F)) / 1.055F, float3(2.4F));
  return lerp(low, high, float3(greaterThan(srgb, float3(0.04045F))));
}

// Converts from logarithmic photographer's EV100 ("exposure in stops") to
// linear scene luminance in candela-seconds/m^2.
inline float ev100Luminance(float ev100)
{
  // The constant factor doesn't really matter; the auto-exposure code only needs
  // a logarithmic scale, and using EV100 here gives values photographers are familiar with.
  // See https://en.wikipedia.org/wiki/Exposure_value#Relationship_of_EV_to_lighting_conditions
  // for the equation source, and historical background as to why this is called EV100.
  return exp2(ev100) / float(100.0f / 12.5f);
}

// Converts from linear scene luminance to photographer's EV100 ("exposure in stops").
// See ev100Luminance().
inline float luminanceEv100(float luminance)
{
  return log2(luminance * float(100.0f / 12.5f));
}

// Returns the luminance of a linear color in the BT.709 standard.
inline float bt709Luminance(float3 linearColor)
{
  return dot(linearColor, float3(0.2126, 0.7152, 0.0722));
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

// Computes the CIE 1960 chromaticity coordinates of a light source given
// temperature and tint.
//
// ANSI C78.377-2008 defines light colors in terms of a correlated color
// temperature `t` of a blackbody, plus a tint shift `duv`, which is the
// distance from the "Planckian locus" of blackbody chromaticities in the
// CIE 1960 color space; negative values are more magenta.
//
// Note that the D65 Illuminant D white point used for sRGB cooresponds to
// `t == 6506.11144` and `duv = 0.00325895312`, i.e. daylight is very slightly
// green compared to a blackbody, which matches the standard.
//
// This parameterization of white balance/tint is rather physically based, but
// might not be the best for artists -- in your own apps, you may wish to
// provide RGB white balance parameters instead.
inline float2 cie1960LightColor(float t, float duv)
{
  // The approximation for the locus we use here is from Michael P.Krystek,
  // "An algorithm to calculate correlated colour temperature"(1985),
  // as it appears in https: // en.wikipedia.org/wiki/Planckian_locus#Approximation .
  // We compute the derivative analytically.
  // As a result of this approximation, D65 is at 6506 rather than the usual
  // 6504 K.
  float  rcpUDenom = 1.0F / ((7.08145163e-7F * t + 8.42420235e-4F) * t + 1.F);
  float  rcpVDenom = 1.0F / ((1.61456053e-7F * t - 2.89741816e-5F) * t + 1.F);
  float2 uv        = float2(((1.28641212e-7F * t + 1.54118254e-4F) * t + 0.860117757F) * rcpUDenom,  //
                            ((4.20481691e-8F * t + 4.22806245e-5F) * t + 0.317398726F) * rcpVDenom);

  // For the derivative, note that we can re - use the denominator, since
  // d/dt f(t) / g(t) = (f'(t)g(t) - f(t)g'(t)) / g(t)^2:
  float2 derivative =
      float2(((-7.68136056380576e-13F * t - 9.60894034459919e-7F) * t - 0.000570462348979613F) * rcpUDenom * rcpUDenom,  //
             ((-8.04477403759601e-12F * t - 1.8395552854377e-8F) * t + 5.14769928267326e-5F) * rcpVDenom * rcpVDenom);
  derivative = normalize(derivative);  // Normalize it

  // Apply the duv offset
  uv += duv * float2(derivative.y, -derivative.x);
  return uv;
}

// Converts from CIE 1960 (u, v) chromaticity to CIE 1931 XYZ, normalizing
// to a brightness of Y=1.
inline float3 cie1931XyzFromCie1960UV(float2 uv)
{
  // From https://en.wikipedia.org/wiki/CIE_1960_color_space#Relation_to_CIE_XYZ
  float2 xy = uv * float2(3.f, 2.f) / (2.f * uv.x - 8.f * uv.y + 4.f);
  // From https://en.wikipedia.org/wiki/CIE_1931_color_space#CIE_xyY_color_space :
  // For low color temperatures and high Duv, the light color might get a
  // negative z = 1 - x - y value, which is totally impossible.
  // Scale the xy value back in to prevent this:
  xy /= max(1.f, xy.x + xy.y);
  // Convert to (x, y, 1-x-y) then divide by y
  float rcpY = 1.f / xy.y;
  return float3(xy.x * rcpY, 1.f, (1.f - xy.x) * rcpY - 1.f);
}

// Returns a matrix that corrects a light source with the given Kelvin
// temperature and Duv tint to RGB (1,1,1), then multiplies by exposure.
// Use it like mul(inputMatrix, color).
inline float3x3 getColorCorrectionMatrix(float exposure, float temperature, float tint)
{
  // Suesstrunk, Holm, and Finlayson's paper "Chromatic Adaptation
  // Performance of Different RGB Sensors" (2001,
  // https://web.archive.org/web/20061018020916/http://infoscience.epfl.ch/getfile.py?mode=best&recid=34049 )
  // found that the CMCCAT2000 space performed the best at mapping colors
  // between scenes with different illuminations, based on data from 16
  // data sets.
  //
  // Following their approach, we'll convert colors and the light source
  // color to CMCCAT2000, divide by the whitepoint and apply exposure, and
  // then convert back to linear RGB.
  // Since these are all linear operations, it's the same as multiplying by
  // a single 3x3 matrix.
  //
  // Here are the matrices we'll need:
  const float3x3 kCmccat2000FromCie1931Xyz = float3x3(0.7982F, 0.3389F, -0.1371F,              // Row 1
                                                      -0.5918F, 1.5512F, 0.0406F,              // Row 2
                                                      0.0008F, 0.0239F, 0.9753F);              // Row 3
  const float3x3 kCmccat2000FromRgb        = float3x3(0.39858179F, 0.51147528F, 0.03823013F,   //
                                                      0.08651038F, 0.90263008F, 0.04376704F,   //
                                                      0.02423435F, 0.13363512F, 0.92889263F);  //
  const float3x3 kRgbFromCmccat2000        = float3x3(2.86209568F, -1.61563711F, -0.04166966,  //
                                                      -0.27259146F, 1.26953249F, -0.04859816,  //
                                                      -0.0354543F, -0.14049009F, 1.0846294);

  // Whitepoint to CMCCAT2000
  float3 whitepointXYZ        = cie1931XyzFromCie1960UV(cie1960LightColor(temperature, tint));
  float3 whitepointCmccat2000 = mul(kCmccat2000FromCie1931Xyz, whitepointXYZ);
  // Componentwise multiplier.
  // Note that even for D65, whitepointCmccat2000 isn't [1,1,1], so we need to
  // apply a normalization factor. This is mul(kCmccat2000FromRgb, float3(1,1,1)):
  float3   middleFactor = exposure * (float3(0.948287308f, 1.03290749f, 1.08676255f) / whitepointCmccat2000);
  float3x3 right        = float3x3(middleFactor[0] * kCmccat2000FromRgb[0],  //
                                   middleFactor[1] * kCmccat2000FromRgb[1],  //
                                   middleFactor[2] * kCmccat2000FromRgb[2]);
  // Fold all the matrices together
  float3x3 result = mul(kRgbFromCmccat2000, right);
  // Note that for the default exposure, temperature, and tint we now have the
  // identity matrix to within 2e-7.
  return result;
}

// Applies the given tone mapper and color grading settings to a given color.
// Requires the pixel index so it can dither,
// and the screen size so that it can apply vignetting.
inline float3 applyTonemap(TonemapperData tm, float3 color, float2 pixel, float2 imageSize)
{
  // Exposure and white balance.
  // These are combined into a matrix and applied together before tonemapping
  // because they're correcting for scene lighting and camera parameters.
  color = mul(tm.inputMatrix, color);

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

  // SDR color grading.
  // contrast and clamp (just in case)
  c = clamp(lerp(float3(0.5F), c, float3(tm.contrast)), float3(0.F), float3(1.F));
  // brightness
  c = pow(c, float3(1.0F / tm.brightness));
  // saturation
  float3 i = float3(dot(c, float3(0.299F, 0.587F, 0.114F)));
  c        = lerp(i, c, float3(tm.saturation));
  // vignette
  float2 center_uv = (pixel / imageSize) * 2.0F - 1.0F;
  c *= 1.0F - dot(center_uv, center_uv) * tm.vignette;

  // Dither assuming 8-bit output.
  if(tm.dither == 1)
  {
    const float levelsMinus1 = 255.F;
    // Based on Fardan Niqbal's survey at https://www.shadertoy.com/view/cl2GRm,
    // we use R2 noise in [0, 1]:
    float noise = fract(dot(pixel, float2(0.245122331F, 0.430159704F)));
    // routed through a triangular wave to [-.5,.5]:
    noise = .5F - 2.F * abs(noise - .5F);
    // shaped so its probability distribution function is a triangle over [-1, 1]:
    const float trinoise = sign(noise) * (1.F - sqrt(1.F - 2.F * abs(noise)));
    // We use triangular noise for most of the range (so that the amount of
    // noise appears uniform) but switch to uniform at the ends:
    const bool3 useUniform = greaterThan(float3(.5F / levelsMinus1), c)  //
                             || greaterThan(c, float3(1.F - .5F / levelsMinus1));
    // Note that because we add noise in sRGB space, we gain an average of
    // 4 * 10^-6 luminance (roughly 0.279/L^2 where L is the number of levels).
    // http://www.thetenthplanet.de/archives/5367 does energy-preserving dither,
    // but with a box instead of a triangular kernel, so the amount of noise
    // has a banding pattern over the image.
    c += lerp(float3(trinoise), float3(noise), float3(useUniform)) / levelsMinus1;
  }

  return c;
}

NAMESPACE_SHADERIO_END()

#endif  // TONEMAMP_FUNCTIONS_H
