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

#ifndef LIGHT_CONTRIB_H
#define LIGHT_CONTRIB_H 1

#include "functions.h.slang"
#include "gltf_scene_io.h.slang"

struct LightContrib
{
  float3 incidentVector;  // Direction from surface to light source (normalized)
  float3 intensity;       // RGB intensity of the light contribution
  float  distance;        // Distance from surface to light source
  float  pdf;             // Sampling PDF for MIS (DIRAC for punctual lights, 1/solidAngle for area)
};

/*-----------------------------------------------------------------------------------------------------------------
# Function `singleLightContribution`
> Calculates the contribution of a light source to the illumination at a specific point on a surface.

## Parameters:
- `light`: A `GltfLight` structure containing the light's properties (defined in gltf_scene_io.h)
- `surfacePos`: The world-space position of the point on the surface being illuminated
- `surfaceNormal`: The normalized surface normal at the point being illuminated
- `randVal`: A 2D random value used for importance sampling of area lights (disk-shaped)

## Returns:
A `LightContrib` structure containing the calculated light contribution information,
including incident direction, light intensity, distance, and direction PDF.

## Notes:
For area lights, using random sampling will result in more realistic but potentially
noisy illumination. The function handles directional, point, and spot lights with
proper attenuation and falloff calculations.
----------------------------------------------------------------------------------------------------------------- */
LightContrib singleLightContribution(in GltfLight light, in float3 surfacePos, in float3 surfaceNormal, in float2 randVal)
{
  LightContrib contrib;
  contrib.incidentVector     = float3(0.0F);
  contrib.intensity          = float3(0.0F);
  contrib.distance           = INFINITE;
  contrib.pdf                = DIRAC;  // Default to punctual light (delta distribution)
  float irradiance           = 0.0F;
  float oneMinusCosHalfAngle = 0.0F;

  if(light.type == LightType::eLightTypeDirectional)
  {
    if(dot(surfaceNormal, -light.direction) <= 0.0)
      return contrib;

    contrib.incidentVector = light.direction;
    // Approximation to 1.F - cos(light.angularSizeOrInvRange * 0.5F),
    // accurate to within 0.5% over [0, pi/2]
    float temp           = light.angularSizeOrInvRange * light.angularSizeOrInvRange;
    oneMinusCosHalfAngle = fma(-1.F / 384.F, temp, 0.125F) * temp;
    irradiance           = light.intensity;
  }
  else if(light.type == LightType::eLightTypeSpot || light.type == LightType::eLightTypePoint)
  {
    float3  light_to_surface = surfacePos - light.position;
    float distance         = sqrt(dot(light_to_surface, light_to_surface));
    float r_distance       = 1.0F / distance;

    contrib.distance       = distance;
    contrib.incidentVector = light_to_surface * r_distance;

    float attenuation = 1.F;
    if(light.angularSizeOrInvRange > 0.0F)
    {
      attenuation = square(saturate(1.0F - square(square(distance * light.angularSizeOrInvRange))));

      if(attenuation == 0.0F)
        return contrib;
    }

    float spotlight = 1.0F;
    if(light.type == LightType::eLightTypeSpot)
    {
      float lDotD           = dot(contrib.incidentVector, light.direction);
      float direction_angle = acos(lDotD);
      spotlight             = 1.0F - smoothstep(light.innerAngle, light.outerAngle, direction_angle);

      if(spotlight == 0.0F)
        return contrib;
    }

    if(light.radius > 0.0F)
    {
      // More numerically stable version of 1 - cos(atan(min(light.radius * r_distance, 1.0F)));
      float r              = min(light.radius * r_distance, 1.0F);
      float r2p1           = fma(r, r, 1.F);
      oneMinusCosHalfAngle = r * r / (r2p1 + sqrt(r2p1));

      float solidAngleOverPi = 2.f * oneMinusCosHalfAngle;

      float radianceTimesPi = light.intensity / square(light.radius);

      irradiance = radianceTimesPi * solidAngleOverPi;
    }
    else
    {
      irradiance = light.intensity * square(r_distance);
    }

    irradiance *= spotlight * attenuation;
  }

  contrib.intensity = irradiance * light.color;


  if(oneMinusCosHalfAngle > 0.0F)
  {  
    // <----- Area lights: spherical (point/spot with radius) or directional with angular size (sun/moon)
    // Uniform cone sampling - section 34 https://people.cs.kuleuven.be/~philip.dutre/GI/TotalCompendium.pdf
    
    float3 dir;
    float tmp   = 1.0F - randVal.y * oneMinusCosHalfAngle;
    float tmp2  = tmp * tmp;
    float theta = sqrt(1.0F - tmp2);
    dir.x = cos(M_TWO_PI * randVal.x) * theta;
    dir.y = sin(M_TWO_PI * randVal.x) * theta;
    dir.z = tmp;
    
    float3 light_dir = -contrib.incidentVector;
    float3 tangent, binormal;
    orthonormalBasis(light_dir, tangent, binormal);
    float3x3 tbn = float3x3(tangent, binormal, light_dir);
    light_dir    = normalize(mul(dir, tbn));

    contrib.incidentVector = -light_dir;
    
    // PDF for uniform cone sampling: 1 / solidAngle, where solidAngle = 2π(1 - cos(θ))
    contrib.pdf = (1.0F / M_TWO_PI) / max(oneMinusCosHalfAngle, 1e-10F);
  }

  return contrib;
}

/*-----------------------------------------------------------------------------------------------------------------
# Function `singleLightContribution` (three-argument overload)
> Like `singleLightContribution` above, but without using random values.

Because this is equivalent to `singleLightContribution(..., float2(0.0F))`, area
lights won't be noisy, but will have harder falloff than they would
otherwise have.
----------------------------------------------------------------------------------------------------------------- */
LightContrib singleLightContribution(in GltfLight light, in float3 surfacePos, in float3 surfaceNormal)
{
  return singleLightContribution(light, surfacePos, surfaceNormal, float2(0.0F));
}
#endif