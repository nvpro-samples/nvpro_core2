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

#ifndef SKY_FUNCTIONS_H
#define SKY_FUNCTIONS_H 1

#include "constants.h.slang"
#include "sky_io.h.slang"

/*-------------------------------------------------------------------------------------------------
# Function `evalSimpleSky`
> Returns the radiance of the simple sky model in a given view direction.
-------------------------------------------------------------------------------------------------*/
inline float3 evalSimpleSky(SkySimpleParameters params, float3 direction)
{
  float3 skyColor     = params.skyColor * params.brightness;
  float3 horizonColor = params.horizonColor * params.brightness;
  float3 groundColor  = params.groundColor * params.brightness;

  // Sky
  float  elevation   = asin(clamp(dot(direction, params.directionUp), -1.0F, 1.0F));
  float  top         = smoothstep(0.F, params.horizonSize, elevation);
  float  bottom      = smoothstep(0.F, params.horizonSize, -elevation);
  float3 environment = lerp(lerp(horizonColor, groundColor, bottom), skyColor, top);

  // Sun
  float angleToLight    = acos(clamp(dot(direction, params.sunDirection), 0.0F, 1.0F));
  float halfAngularSize = params.angularSizeOfLight * 0.5F;
  float glowInput =
      clamp(2.0F * (1.0F - smoothstep(halfAngularSize - params.glowSize, halfAngularSize + params.glowSize, angleToLight)),
            0.0F, 1.0F);
  float  glowIntensity = params.glowIntensity * pow(glowInput, params.glowSharpness);
  float3 sunLight      = glowIntensity * params.lightRadiance;

  return environment + sunLight;
}

/*-------------------------------------------------------------------------------------------------
# Function `sampleSimpleSky`
> Samples the simple sky model using two random values, returning a `SkySamplingResult`.
-------------------------------------------------------------------------------------------------*/
inline SkySamplingResult sampleSimpleSky(SkySimpleParameters params, float2 randVal)
{
  SkySamplingResult result;

  // Constants
  const float sunWeight = 0.95f;  // 95.0% weight for the sun
  const float skyWeight = 1.0f - sunWeight;

  // Decide whether to sample the sun or the sky
  if(randVal.x < sunWeight)
  {
    float sunAngularRadius = params.angularSizeOfLight * 0.5f;

    // Sample the sun
    float theta = sunAngularRadius * sqrt(randVal.y);
    float phi   = 2.0f * M_PI * randVal.x;

    float3 sunLocalDir;
    float  sinTheta = sin(theta);
    sunLocalDir.x   = sinTheta * cos(phi);
    sunLocalDir.y   = sinTheta * sin(phi);
    sunLocalDir.z   = cos(theta);

    // Transform sun_local_dir to world space
    float3 up    = float3(0, 0, 1);
    float3 right = cross(up, params.sunDirection);
    up           = cross(params.sunDirection, right);

    result.direction = sunLocalDir.x * right + sunLocalDir.y * up + sunLocalDir.z * params.sunDirection;

    // Compute the PDF
    result.pdf = sunWeight;  // / (M_TWO_PI * (1.0f - cos(angular_size)));

    // Compute the light intensity (assuming uniform sun disk)
    result.radiance = params.lightRadiance;
  }
  else
  {
    // Sample the sky (simple uniform sampling of the upper hemisphere)
    float cosTheta = sqrt(1.0f - randVal.y);
    float sinTheta = sqrt(randVal.y);
    float phi      = 2.0f * M_PI * randVal.x;

    result.direction = float3(sinTheta * cos(phi), cosTheta, sinTheta * sin(phi));

    // Ensure the sampled direction is in the upper hemisphere
    if(dot(result.direction, params.directionUp) < 0.0f)
    {
      result.direction.y = -result.direction.y;
    }

    // Compute the PDF (uniform sampling of hemisphere)
    result.pdf = skyWeight;  // / M_TWO_PI;

    // Compute sky color (simplified version of proceduralSky function)
    float elevation = asin(clamp(dot(result.direction, params.directionUp), -1.0f, 1.0f));
    float t         = smoothstep(0.0f, params.horizonSize, elevation);
    result.radiance = lerp(params.horizonColor, params.skyColor, t);
  }

  return result;
}


////////////////////////////////////////////////////////////////////////
// Physical Sky
////////////////////////////////////////////////////////////////////////

/*helper functions for sun_and_sky*/

// Function to calculate the luminance of an RGB color
inline float rgbLuminance(float3 rgb)
{
  return 0.2126f * rgb.x + 0.7152f * rgb.y + 0.0722f * rgb.z;  // Rec. 709 luminance coefficients
}

// Function to transform local coordinates (x, y, z) to a direction vector aligned with in_main
inline float3 localCoordsToDir(float3 mainVec, float x, float y, float z)
{
  float3 u =
      normalize(abs(mainVec.x) < abs(mainVec.y) ? float3(0.0f, -mainVec.z, mainVec.y) : float3(mainVec.z, 0.0f, -mainVec.x));
  float3 v = cross(mainVec, u);
  return x * u + y * v + z * mainVec;
}

// Equal-area transformation of the square [0,1]^2 to the disk
// {(x, y) | x^2 + y^2 <= 1}.
inline float2 squareToDisk(float inX, float inY)
{
  float localX = 2.0f * inX - 1.0f;
  float localY = 2.0f * inY - 1.0f;
  if(localX == 0.0f && localY == 0.0f)
    return float2(0.0f);

  float r, phi;
  if(localX > -localY)
  {
    if(localX > localY)
    {
      r   = localX;
      phi = (M_PI / 4.0f) * (1.0f + localY / localX);
    }
    else
    {
      r   = localY;
      phi = (M_PI / 4.0f) * (3.0f - localX / localY);
    }
  }
  else
  {
    if(localX < localY)
    {
      r   = -localX;
      phi = (M_PI / 4.0f) * (5.0f + localY / localX);
    }
    else
    {
      r   = -localY;
      phi = (M_PI / 4.0f) * (7.0f - localX / localY);
    }
  }
  return float2(r, phi);
}

// Function to calculate a diffuse reflection direction based on an input normal and sample
inline float3 reflectionDirDiffuseX(float3 inNormal, float2 inSample)
{
  float2 rPhi = squareToDisk(inSample.x, inSample.y);   // Map the 2D sample point to a disk
  float  x    = rPhi.x * cos(rPhi.y);                   // Calculate the x component of the reflection direction
  float  y    = rPhi.x * sin(rPhi.y);                   // Calculate the y component of the reflection direction
  float  z    = sqrt(max(0.0f, 1.0f - x * x - y * y));  // Calculate the z component, ensuring it is non-negative
  return localCoordsToDir(inNormal, x, y, z);  // Convert the local direction to the world space direction using the normal
}

// Function to calculate the color of the sun based on its direction and atmospheric turbidity
inline float3 calcSunColor(float3 sunDir, float turbidity)
{
  if(sunDir.z <= 0.0f)
    return float3(0.0f);

  float3 ko         = float3(12.0f, 8.5f, 0.9f);       // Optical depth constants for ozone
  float3 wavelength = float3(0.610f, 0.550f, 0.470f);  // Wavelength of light (in micrometers) for RGB channels
  float3 solRad = float3(1.0f, 0.992f, 0.911f) * (127500.0f / 0.9878f);  // Solar radiance adjusted for atmospheric effects

  float m = 1.0f / (sunDir.z + 0.15f * pow(93.885f - degrees(acos(sunDir.z)), -1.253f));  // Optical air mass calculation (simplified relative air mass formula)
  float  beta = 0.04608f * turbidity - 0.04586f;  // Beta coefficient for Rayleigh scattering based on turbidity
  float3 ta   = exp(-m * beta * pow(wavelength, float3(-1.3f)));  // Attenuation due to aerosol (Rayleigh) scattering
  float3 to   = exp(-m * ko * 0.0035f);                           // Attenuation due to ozone absorption
  float3 tr   = exp(-m * 0.008735f * pow(wavelength, float3(-4.08f)));  // Attenuation due to Rayleigh scattering

  return tr * ta * to * solRad;  // Calculate the final sun color based on the combined effects
}

// Function to calculate the color of the sky based on the sun's direction and atmospheric turbidity
inline float3 skyColorXyz(float3 inDir, float3 inSunPos, float inTurbidity, float inLuminance)
{
  float3 xyz;
  float  A, B, C, D, E;
  float  cosGamma = dot(inSunPos, inDir);
  if(cosGamma > 1.0f)
  {
    cosGamma = 2.0f - cosGamma;
  }
  float gamma       = acos(cosGamma);
  float cosTheta    = inDir.z;
  float cosThetaSun = inSunPos.z;
  float thetaSun    = acos(cosThetaSun);
  float t2          = inTurbidity * inTurbidity;
  float ts2         = thetaSun * thetaSun;
  float ts3         = ts2 * thetaSun;
  // determine x and y at zenith
  float zenithX = ((+0.001650f * ts3 - 0.003742f * ts2 + 0.002088f * thetaSun + 0) * t2
                   + (-0.029028f * ts3 + 0.063773f * ts2 - 0.032020f * thetaSun + 0.003948f) * inTurbidity
                   + (+0.116936f * ts3 - 0.211960f * ts2 + 0.060523f * thetaSun + 0.258852f));
  float zenithY = ((+0.002759f * ts3 - 0.006105f * ts2 + 0.003162f * thetaSun + 0) * t2
                   + (-0.042149f * ts3 + 0.089701f * ts2 - 0.041536f * thetaSun + 0.005158f) * inTurbidity
                   + (+0.153467f * ts3 - 0.267568f * ts2 + 0.066698f * thetaSun + 0.266881f));
  xyz.y         = inLuminance;
  // TODO: Preetham/Utah

  // Precompute common terms
  float invCosTheta = 1.0f / max(cosTheta, 0.001f);  // Avoid division by zero

  // use flags (see above)
  A = -0.019257f * inTurbidity - (0.29f - sqrt(cosThetaSun) * 0.09f);
  B = -0.066513f * inTurbidity + 0.000818f;
  C = -0.000417f * inTurbidity + 0.212479f;
  D = -0.064097f * inTurbidity - 0.898875f;
  E = -0.003251f * inTurbidity + 0.045178f;

  float expBInvCosTheta = exp(B * invCosTheta);
  float expB            = exp(B);
  float expDGamma       = exp(D * gamma);
  float expDThetaSun    = exp(D * thetaSun);

  float x = (((1.f + A * expBInvCosTheta) * (1.f + C * expDGamma + E * cosGamma * cosGamma))
             / ((1 + A * expB) * (1 + C * expDThetaSun + E * cosThetaSun * cosThetaSun)));

  A = -0.016698f * inTurbidity - 0.260787f;
  B = -0.094958f * inTurbidity + 0.009213f;
  C = -0.007928f * inTurbidity + 0.210230f;
  D = -0.044050f * inTurbidity - 1.653694f;
  E = -0.010922f * inTurbidity + 0.052919f;

  // Recompute for y calculation
  expBInvCosTheta = exp(B * invCosTheta);
  expB            = exp(B);
  expDGamma       = exp(D * gamma);
  expDThetaSun    = exp(D * thetaSun);

  float y = (((1 + A * expBInvCosTheta) * (1 + C * expDGamma + E * cosGamma * cosGamma))
             / ((1 + A * expB) * (1 + C * expDThetaSun + E * cosThetaSun * cosThetaSun)));

  float localSaturation = 1.0f;

  x = zenithX * ((x * localSaturation) + (1.0f - localSaturation));
  y = zenithY * ((y * localSaturation) + (1.0f - localSaturation));

  // convert chromaticities x and y to CIE
  xyz.x = (x / y) * xyz.y;
  xyz.z = ((1.0f - x - y) / y) * xyz.y;
  return xyz;
}

inline float skyLuminance(float3 inDir, float3 inSunPos, float inTurbidity)
{
  float cosGamma    = clamp(dot(inSunPos, inDir), 0.0f, 1.0f);
  float gamma       = acos(cosGamma);
  float cosTheta    = inDir.z;
  float cosThetaSun = inSunPos.z;
  float thetaSun    = acos(cosThetaSun);

  float A = 0.178721f * inTurbidity - 1.463037f;
  float B = -0.355402f * inTurbidity + 0.427494f;
  float C = -0.022669f * inTurbidity + 5.325056f;
  float D = 0.120647f * inTurbidity - 2.577052f;
  float E = -0.066967f * inTurbidity + 0.370275f;

  // Precompute exp terms to reduce function calls
  float invCosTheta     = 1.0f / max(cosTheta, 0.001f);
  float expBInvCosTheta = exp(B * invCosTheta);
  float expB            = exp(B);
  float expDGamma       = exp(D * gamma);
  float expDThetaSun    = exp(D * thetaSun);

  return (((1 + A * expBInvCosTheta) * (1 + C * expDGamma + E * cosGamma * cosGamma))
          / ((1 + A * expB) * (1 + C * expDThetaSun + E * cosThetaSun * cosThetaSun)));
}

inline float3 calcSkyColor(float3 inSunDir, float3 inDir, float inTurbidity)
{
  float thetaSun  = acos(inSunDir.z);
  float chi       = (4.0f / 9.0f - inTurbidity / 120.0f) * (M_PI - 2.0f * thetaSun);
  float luminance = 1000.0f * ((4.0453f * inTurbidity - 4.9710f) * tan(chi) - 0.2155f * inTurbidity + 2.4192f);
  luminance *= skyLuminance(inDir, inSunDir, inTurbidity);

  float3 xyz = skyColorXyz(inDir, inSunDir, inTurbidity, luminance);
  float3 envColor = float3(3.241f * xyz.x - 1.537f * xyz.y - 0.499f * xyz.z, -0.969f * xyz.x + 1.876f * xyz.y + 0.042f * xyz.z,
                           0.056f * xyz.x - 0.204f * xyz.y + 1.057f * xyz.z);
  return envColor * M_PI;
}

inline float3 calcSkyIrradiance(float3 inDataSunDir, float inDataSunDirHaze)
{
  float3 colSum        = float3(0.0f);
  float3 nuStateNormal = float3(0.0f, 0.0f, 1.0f);

  for(float u = 0.125f; u < 1.0f; u += 0.25f)
  {
    for(float v = 0.125f; v < 1.0f; v += 0.25f)
    {
      float3 diff = reflectionDirDiffuseX(nuStateNormal, float2(u, v));
      colSum += calcSkyColor(inDataSunDir, diff, inDataSunDirHaze);
    }
  }
  return colSum / 16.0f;
}

inline float tweakSaturation(float inSaturation, float inHaze)
{
  if(inSaturation > 1.0f)
    return 1.0f;

  float lowSat    = inSaturation * inSaturation * inSaturation;  // Compute the low saturation value
  float localHaze = clamp((inHaze - 2.0f) / 15.0f, 0.0f, 1.0f);  // Normalize the local haze value
  localHaze *= localHaze * localHaze;                            // Adjust haze to the power of 3
  return lerp(inSaturation, lowSat, localHaze);  // Blend the input saturation with the low saturation value based on local haze
}


inline float3 tweakVector(float3 dir, int yIsUp, float horizHeight)
{
  float3 outDir = dir;
  if(yIsUp == 1)
  {
    outDir = float3(dir.x, dir.z, dir.y);
  }
  if(horizHeight != 0)
  {
    outDir.z -= horizHeight;
    outDir = normalize(outDir);
  }
  return outDir;
}

inline float3 tweakColor(float3 tint, float saturation, float redness)
{
  float  intensity = rgbLuminance(tint);
  float3 outTint   = (saturation <= 0.0f) ? float3(intensity) : lerp(float3(intensity), tint, saturation);
  outTint *= float3(1.0f + redness, 1.0f, 1.0f - redness);
  return max(outTint, float3(0.0f));
}

inline float2 calcPhysicalScale(float sunDiskScale, float sunGlowIntensity, float sunDiskIntensity)
{
  float sunAngularRadius = 0.00465f;
  float sunDiskRadius    = sunAngularRadius * sunDiskScale;
  float sunGlowRadius    = sunDiskRadius * 10.0f;

  // Precompute constants to reduce redundant calculations
  float sunGlowRadiusSq    = sunGlowRadius * sunGlowRadius;
  float sunGlowRadiusCubed = sunGlowRadiusSq * sunGlowRadius;

  /* We calculate the integral of the glow intensity function */
  float glowFuncIntegral =
      sunGlowIntensity * ((4.0f * M_PI) - (24.0f * M_PI) / sunGlowRadiusSq + (24.0f * M_PI) * sin(sunGlowRadius) / sunGlowRadiusCubed);


  /* Calculate the target sun disk intensity integral (the value towards which
    we must scale to attain a physically-scaled sun intensity */
  float targetSundiskIntegral = sunDiskIntensity * M_PI;

  /* Subtract the glow integral from the target disk integral,
    limiting the glow power to 50% of the sun disk */
  float skySunglowScale = 1.0f;
  float maxGlowIntegral = 0.5f * targetSundiskIntegral;
  if(glowFuncIntegral > maxGlowIntegral)
  {
    skySunglowScale *= maxGlowIntegral / glowFuncIntegral;
    targetSundiskIntegral -= maxGlowIntegral;
  }
  else
  {
    targetSundiskIntegral -= glowFuncIntegral;
  }

  float cosSunDiskRadius       = cos(sunDiskRadius);
  float sundiskArea            = 2 * M_PI * (1 - cosSunDiskRadius);
  float targetSundiskIntensity = targetSundiskIntegral / sundiskArea;

  /* Calculate the actual sun disk intensity, before scaling is applied */
  float actualSundiskIntegral = 1.0f * sundiskArea;
  /* approximation! needs to be re-calculated from the integral of the function */
  float actualSundiskIntensity = sunDiskIntensity * 100.0f * actualSundiskIntegral / sundiskArea;
  /* Apply the proper scaling to get to the target value */
  return float2((targetSundiskIntensity == 0.0f) ? 0.0f : targetSundiskIntensity / actualSundiskIntensity, skySunglowScale);
}

inline float nightBrightnessAdjustment(float3 sunDir)
{
  float lmt = 0.30901699437494742410229341718282f;  // ~ cos(72°) or 18° below the horizon
  if(sunDir.z <= -lmt)
    return 0.0f;
  float factor = (sunDir.z + lmt) / lmt;
  factor *= factor;
  factor *= factor;
  return factor;
}

/*-------------------------------------------------------------------------------------------------
# Function `evalPhysicalSky`
> Returns the radiance of the physical sky model in a given direction.
-------------------------------------------------------------------------------------------------*/
inline float3 evalPhysicalSky(SkyPhysicalParameters ss, float3 inDirection)
{
  if(ss.multiplier <= 0.0f)
    return float3(0.0f);

  float3 result          = float3(0.0f);
  float  factor          = 1.0f;
  float  nightFactor     = 1.0f;
  float3 outColor        = float3(0.0f);
  float3 rgbScale        = ss.rgbUnitConversion * ss.multiplier;
  float  heightAdjusted  = (ss.horizonHeight + ss.horizonBlur) / 10.0f;
  float3 dir             = tweakVector(inDirection, ss.yIsUp, heightAdjusted);
  float  localHaze       = max(2.0f, 2.0f + ss.haze);
  float  localSaturation = tweakSaturation(ss.saturation, localHaze);

  // Calculate the "downness" of the direction (how much it points downward)
  float  downness = dir.z;
  float3 realDir  = dir;
  if(dir.z < 0.001f)
  {
    dir.z = 0.001f;
    dir   = normalize(dir);
  }

  // Adjust the sun direction similarly to the input direction
  float3 sunDir     = ss.sunDirection;
  sunDir            = tweakVector(sunDir, ss.yIsUp, heightAdjusted);
  float3 realSunDir = sunDir;
  if(sunDir.z < 0.001f)
  {
    // Adjust brightness for night time
    factor   = nightBrightnessAdjustment(sunDir);
    sunDir.z = 0.001f;
    sunDir   = normalize(sunDir);
  }

  // Calculate the sun and sky color
  float3 tint         = (factor > 0.0f) ? calcSkyColor(sunDir, dir, localHaze) * factor : float3(0.0f);
  float3 dataSunColor = calcSunColor(sunDir, (downness > 0) ? localHaze : 2.0f);

  // Add the sun disk and glow if enabled
  if(ss.sunDiskIntensity > 0.0f && ss.sunDiskScale > 0.0f)
  {
    float sunAngle   = acos(dot(realDir, realSunDir));
    float glowRadius = 0.00465f * ss.sunDiskScale * 10.0f;
    if(sunAngle < glowRadius)
    {
      float2 scales = calcPhysicalScale(ss.sunDiskScale, ss.sunGlowIntensity, ss.sunDiskIntensity);
      // A value of 0 is at the edge of the glow disk; a value of 1 is in the
      // center of the sun.
      float centerProximity      = (1.0f - sunAngle / glowRadius);
      float centerProximitySq    = centerProximity * centerProximity;
      float centerProximityCubed = centerProximitySq * centerProximity;
      float glowFactor           = centerProximityCubed * 2.0f * ss.sunGlowIntensity * scales.y;
      float diskFactor =
          smoothstep(0.85f, 0.95f + (localHaze / 500.0f), centerProximity) * 100.0f * ss.sunDiskIntensity * scales.x;
      tint += dataSunColor * (glowFactor + diskFactor);
    }
  }
  outColor = tint * rgbScale;

  // Add ground color if the direction is pointing downward
  if(downness <= 0.0f)
  {
    float3 irrad     = calcSkyIrradiance(sunDir, 2.0f);
    float3 downColor = ss.groundColor * (irrad + dataSunColor * sunDir.z) * rgbScale;
    downColor *= factor;
    float horBlur = ss.horizonBlur / 10.0f;
    if(horBlur > 0.0f)
    {
      // Blend between sky and ground color at the horizon
      float dness = smoothstep(0.0f, 1.0f, -downness / horBlur);
      outColor    = lerp(outColor, downColor, dness);
      nightFactor = 1.0f - dness;
    }
    else
    {
      outColor    = downColor;
      nightFactor = 0.0f;
    }
  }

  // Adjusting the color based on the saturation and red-blue shift
  outColor = tweakColor(outColor, localSaturation, ss.redblueshift);
  result   = outColor * M_PI;

  // Adding the night sky color
  if(nightFactor > 0.0f)
  {
    float3 night = ss.nightColor * nightFactor;
    result       = max(result, night);
  }

  return result;
}

// Uniformly samples a spherical cap: the part of the surface of a sphere
// where z ranges from z_min to 1. If randomSample.y is sampled in the closed
// interval [0,1], this samples z in [z_min, 1] and a closed cap will be sampled;
// if randomSample.y is sampled in the half-open interval [0,1), then z is in
// (z_min, 1] and an open cap will be sampled.
inline float3 sampleSphericalCap(float z_min, float2 xi)
{
  float z   = lerp(1.F, z_min, xi.y);
  float r   = sqrt(max(0.F, 1.F - z * z));
  float phi = 2.0f * M_PI * xi.x;
  float x   = r * cos(phi);
  float y   = r * sin(phi);
  return float3(x, y, z);
}

// Probability that samplePhysicalSky samples the sun.
// If the sun is too small, we never sample it.
inline float physicalSkySunProbability(SkyPhysicalParameters ss)
{
  float sunElevation = ss.sunDirection.z;
  return (ss.sunDiskScale > 1e-5f) ? clamp(ss.sunDiskIntensity * sunElevation * 0.5f + 0.5f, 0.1f, 0.9f) : 0.0f;
}

/*-------------------------------------------------------------------------------------------------
# Function `samplePhysicalSkyPDF`
> Returns the probability that `samplePhysicalSky` samples a certain direction.
-------------------------------------------------------------------------------------------------*/
inline float samplePhysicalSkyPDF(SkyPhysicalParameters ss, float3 inDirection)
{
  const float sunAngularRadius = 0.00465f * ss.sunDiskScale;
  // If we choose the sky, this is the probability of choosing a given direction:
  const float skyPdf = 1.0f / (2.0f * M_PI);
  // This factor of 1.5f is because of the lower bound on the sun's
  // smoothstep when computing `diskFactor` in `evalPhysicalSky`.
  const float sunSampleAngularRadius    = 1.5f * sunAngularRadius;
  const float cosSunSampleAngularRadius = cos(sunSampleAngularRadius);

  // Use first-degree Taylor series expansion around 0 for better precision
  const float sunSampleSolidAngle = (sunSampleAngularRadius < 0.001f) ? M_PI * sunSampleAngularRadius * sunSampleAngularRadius :
                                                                        2.0f * M_PI * (1.0f - cosSunSampleAngularRadius);
  // If we choose the sun, this is the probability of choosing a given direction:
  const float sunPdf = (dot(inDirection, ss.sunDirection) >= cosSunSampleAngularRadius) ? 1.0f / sunSampleSolidAngle : 0.0f;
  return lerp(skyPdf, sunPdf, physicalSkySunProbability(ss));
}

/*-------------------------------------------------------------------------------------------------
# Function `samplePhysicalSky`
> Samples the physical sky model using two random values, returning a `SkySamplingResult`.
-------------------------------------------------------------------------------------------------*/
inline SkySamplingResult samplePhysicalSky(SkyPhysicalParameters ss, float2 randomSample)
{
  SkySamplingResult result;

  // Decide whether to sample sun or sky
  float sunProb   = physicalSkySunProbability(ss);
  float z_min     = 0.0f;  // Minimum z-value of the spherical cap we'll sample
  bool  sampleSun = randomSample.x < sunProb;
  if(sampleSun)
  {
    // Adjust the range of the random value so we can use it again:
    randomSample.x = randomSample.x / sunProb;
    // Sample the sun by doing uniform spherical cap sampling around the +z
    // axis, then rotating the +z axis so that it points towards the sun.
    const float sunSampleAngularRadius = 1.5f * 0.00465f * ss.sunDiskScale;
    z_min                              = cos(sunSampleAngularRadius);
  }
  else
  {
    // Adjust the range of the random value so we can use it again:
    randomSample.x = (randomSample.x - sunProb) / (1.0f - sunProb);
  }

  result.direction = sampleSphericalCap(z_min, randomSample);

  if(sampleSun)
  {
    // Transform the sun from +z to ss.sunDirection
    float3 up    = float3(0, 0, 1);
    float3 right = normalize(cross(up, ss.sunDirection));
    up           = cross(ss.sunDirection, right);

    result.direction = result.direction.x * right  //
                       + result.direction.y * up   //
                       + result.direction.z * ss.sunDirection;
  }

  // Evaluate the sky model
  result.radiance = evalPhysicalSky(ss, result.direction);
  result.pdf      = samplePhysicalSkyPDF(ss, result.direction);
  return result;
}


#endif  // SKY_FUNCTIONS_H
