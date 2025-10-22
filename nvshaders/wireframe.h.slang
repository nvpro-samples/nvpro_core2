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

#include "nvshaders/constants.h.slang"

/// Configuration structure for wireframe rendering
/// Controls the appearance and behavior of wireframe edges on triangles
struct WireframeSettings
{
  float  thickness;       ///< Base thickness of the wireframe lines (multiplied by 0.5 internally)
  float2 thicknessVar;    ///< Thickness variation along edges: x = min thickness, y = max thickness
  float  smoothing;       ///< Anti-aliasing smoothing factor (multiplied by thickness)
  bool   stipple;         ///< Enable stipple/dash pattern on wireframe lines
  float  stippleLength;   ///< Length of dashed segments in stipple pattern [0..1]
  float  stippleRepeats;  ///< Number of stipple pattern repeats along the edge
  bool   screenSpace;     ///< If true, use screen-space derivatives; if false, use provided deltas
};

/// Enumeration of available wireframe style presets
[UnscopedEnum]
enum WireframePreset
{
  WIREFRAME_DEFAULT = 0,  ///< Standard wireframe with uniform thickness
  WIREFRAME_WIRE_DOT,     ///< Wire with softer smoothing
  WIREFRAME_STAR,         ///< Varying thickness, object-space, creates star-like appearance
  WIREFRAME_FLAKE,        ///< Thin varying lines with stipple, object-space
  WIREFRAME_THIN,         ///< Thin smooth wireframe with heavy smoothing
  WIREFRAME_WIRE_LINE,    ///< Solid continuous dashed line
  WIREFRAME_STIPPLE,      ///< Stippled/dashed wireframe pattern
  WIREFRAME_PRESET_COUNT
};

/// Preset wireframe configurations for quick style setup
/// Access using WireframePreset enum: WIREFRAME_PRESETS[WIREFRAME_DEFAULT], etc.
static const WireframeSettings WIREFRAME_PRESETS[WIREFRAME_PRESET_COUNT] = {
  // WIREFRAME_DEFAULT - Standard wireframe with uniform thickness
  { 1.0, float2(1.0, 1.0), 1.0, false, 0.5, 5.0, true },

  // WIREFRAME_WIRE_DOT - Wire with softer smoothing for a dotted appearance
  { 1.0, float2(1.0, 1.0), 0.5, false, 0.5, 5.0, true },

  // WIREFRAME_STAR - Varying thickness in object-space creates star-like effect
  { 0.1, float2(0.0, 1.0), 0.1, false, 0.5, 5.0, false },

  // WIREFRAME_FLAKE - Thin varying lines with stipple pattern, object-space
  { 0.1, float2(0.7, 0.0), 0.1, true, 0.8, 10.0, false },

  // WIREFRAME_THIN - Thin smooth wireframe with heavy anti-aliasing
  { 0.3, float2(1.0, 1.0), 2.0, false, 0.4, 8.0, true },

  // WIREFRAME_WIRE_LINE - Solid continuous line with minimal dashing
  { 0.5, float2(1.0, 1.0), 1.0, true, 1.0, 1.0, true },

  // WIREFRAME_STIPPLE - Classic stippled/dashed wireframe pattern
  { 0.5, float2(1.0, 1.0), 1.0, true, 0.5, 20.0, true }
};

/// Returns default wireframe settings with sensible values
/// @return WireframeSettings with thickness=1.0, smoothing=1.0, screenSpace=true, stipple=false
WireframeSettings wireframeDefaultSettings()
{
  return WIREFRAME_PRESETS[WIREFRAME_DEFAULT];
}

/// Main wireframe processing function - computes wireframe contribution at a given point
/// This is the primary entry point for wireframe rendering in both raster and raytracing pipelines
/// @param settings Wireframe configuration controlling appearance and behavior
/// @param barys Barycentric coordinates of the current fragment/hit point (w, u, v)
/// @param deltas Barycentric derivatives for edge detection:
///               - When settings.screenSpace = true, deltas are ignored (can pass float3(0))
///               - In raster pipeline: use fwidth(barys), see fwidth.h.slang
///               - In raytracing pipeline: use computeDerivative() result
/// @return Wireframe contribution in [0..1] where 0=no wireframe, 1=full wireframe
///         This value can be used to blend wireframe color with surface color
float processWireframe(WireframeSettings settings, in float3 barys, in float3 deltas)
{

  // Wireframe Settings
  float thickness = settings.thickness * 0.5;  // Thickness for both side of the edge, must be divided by 2
  float smoothing = settings.thickness * settings.smoothing;  // Could be thickness

  // Uniform position on the edge [0, 1]
  float edgePos = edgePosition(barys);

  // Vary the thickness along the edge
  thickness *= edgeThickness(settings.thicknessVar, edgePos);

  // Get the wireframe line width
  float lineWidth = getLineWidth(deltas, thickness, smoothing, barys);

  if(settings.stipple)
  {
    float stippleFact = stipple(settings.stippleRepeats, settings.stippleLength, edgePos);
    lineWidth *= stippleFact;  // 0 or 1
  }

  return lineWidth;
}

/// Computes the wireframe line width based on barycentric coordinates
/// @param deltas Barycentric coordinate derivatives (use fwidth(barys) in raster, or computeDerivative() in raytracing)
/// @param thickness Half-thickness of the wireframe line
/// @param smoothing Smoothing width for anti-aliasing
/// @param barys Barycentric coordinates of the current fragment/hit point
/// @return Line width value in [0..1] where 0=no line, 1=full line intensity
float getLineWidth(in float3 deltas, in float thickness, in float smoothing, in float3 barys)
{
  barys         = smoothstep(deltas * (thickness), deltas * (thickness + smoothing), barys);
  float minBary = min(barys.x, min(barys.y, barys.z));
  return 1.0 - minBary;
}

/// Computes the normalized position along the triangle edge
/// @param baryWeights Barycentric coordinates (w, u, v)
/// @return Position along the edge in [0..1] where closest edge has highest weight
float edgePosition(float3 baryWeights)
{
  return max(baryWeights.z, max(baryWeights.y, baryWeights.x));
}

/// Computes stipple/dash pattern mask for wireframe edges
/// @param stippleRepeats Number of times the dash pattern repeats along the edge
/// @param stippleLength Length of each dash as a fraction of the pattern [0..1]
/// @param edgePos Position along the edge [0..1]
/// @return 1.0 if the edge should be displayed at this position, 0.0 otherwise
float stipple(in float stippleRepeats, in float stippleLength, in float edgePos)
{
  float offset = 1.0 / stippleRepeats;
  offset *= 0.5 * stippleLength;
  float pattern = fract((edgePos + offset) * stippleRepeats);
  return 1.0 - step(stippleLength, pattern);
}

/// Modulates wireframe thickness along the edge using sinusoidal interpolation
/// Creates thicker lines at edge endpoints and thinner in the middle (or vice versa)
/// @param thicknessVar Thickness range: x = minimum thickness, y = maximum thickness
/// @param edgePos Position along the edge [0..1]
/// @return Interpolated thickness value based on sine wave modulation
float edgeThickness(in float2 thicknessVar, in float edgePos)
{
  return lerp(thicknessVar.x, thicknessVar.y, (1.0 - sin(edgePos * M_PI)));
}

/// Computes ray differentials for a given pixel position in raytracing pipelines
/// These differentials are used by computeDerivative() to calculate barycentric derivatives
/// Typically called once per ray (e.g., in raygen shader) and stored in ray payload
/// @param pixelPos Current pixel coordinates (x, y)
/// @param imageSize Image/framebuffer dimensions (width, height)
/// @param viewInverse Inverse view matrix (camera to world transform)
/// @param projInverse Inverse projection matrix
/// @param differentialX [out] Ray differential in X direction (world space)
/// @param differentialY [out] Ray differential in Y direction (world space)
void computeDifferentials(float2 pixelPos, float2 imageSize, float4x4 viewInverse, float4x4 projInverse, out float3 differentialX, out float3 differentialY)
{
  const float2 pixelCenter = pixelPos + float2(0.5);
  const float2 inUV        = pixelCenter / imageSize;
  const float2 d           = inUV * 2.0 - 1.0;
  // We compute and store the differentials in the payload
  const float2 uvOffset       = (pixelPos + float2(1.5, 1.5)) / imageSize;
  const float2 dOffset        = uvOffset * 2.0 - 1.0;
  const float4 targetOffsetX  = normalize(mul(float4(dOffset.x, d.y, 1, 1), projInverse));
  const float4 targetOffsetY  = normalize(mul(float4(d.x, dOffset.y, 1, 1), projInverse));
  const float4 _differentialX = normalize(mul(float4(targetOffsetX.xyz, 0), viewInverse));
  const float4 _differentialY = normalize(mul(float4(targetOffsetY.xyz, 0), viewInverse));
  // output
  differentialX = _differentialX.xyz;
  differentialY = _differentialY.xyz;
}

/// Computes barycentric coordinate derivatives for raytracing wireframe rendering
/// This is the raytracing equivalent of fwidth(bary) used in raster pipelines
/// Call this function in closest-hit or any-hit shaders to get derivatives for processWireframe()
///
/// Usage example in a closest-hit shader:
///   float3 triangleVertexPositions[3] = { HitTriangleVertexPosition(0),
///                                          HitTriangleVertexPosition(1),
///                                          HitTriangleVertexPosition(2) };
///   float3 deltas = computeDerivative(WorldToObject4x3(), WorldRayOrigin(), ObjectRayOrigin(),
///                                      triangleVertexPositions, differentialX, differentialY,
///                                      HitBarycentrics());
///
/// @param worldToObject World-to-object space transformation matrix (4x3)
/// @param worldRayOrigin Ray origin in world space
/// @param objectRayOrigin Ray origin in object space
/// @param triangleVertexPositions Array of 3 triangle vertices in object space
/// @param offsetX Ray differential in X direction (from computeDifferentials)
/// @param offsetY Ray differential in Y direction (from computeDifferentials)
/// @param bary Current hit point barycentric coordinates (w, u, v)
/// @return Barycentric derivatives (absolute difference) analogous to fwidth(bary)
float3 computeDerivative(in float4x3 worldToObject,
                         in float3   worldRayOrigin,
                         in float3   objectRayOrigin,
                         in float3   triangleVertexPositions[3],
                         in float3   offsetX,
                         in float3   offsetY,
                         in float3   bary)
{
  float3 derivativeTargetX = mul(float4(worldRayOrigin + offsetX, 1), worldToObject);
  float3 derivativeDirX    = derivativeTargetX - objectRayOrigin;
  float3 derivativeX       = intersectRayTriangle(objectRayOrigin, derivativeDirX, triangleVertexPositions[0],
                                                  triangleVertexPositions[1], triangleVertexPositions[2]);
  derivativeX              = abs(derivativeX - float3(bary));

  float3 derivativeTargetY = mul(float4(worldRayOrigin + offsetY, 1), worldToObject);
  float3 derivativeDirY    = derivativeTargetY - objectRayOrigin;
  float3 derivativeY       = intersectRayTriangle(objectRayOrigin, derivativeDirY, triangleVertexPositions[0],
                                                  triangleVertexPositions[1], triangleVertexPositions[2]);
  derivativeY              = abs(derivativeY - float3(bary));

  return float3(derivativeX + derivativeY);
}

/// High-precision version of computeDerivative using double-precision arithmetic
/// Use this when rendering scenes with extreme scale differences or when numerical precision
/// issues cause wireframe artifacts. Slightly more expensive than computeDerivative()
/// @param worldToObject World-to-object space transformation matrix (4x3)
/// @param worldRayOrigin Ray origin in world space
/// @param objectRayOrigin Ray origin in object space
/// @param triangleVertexPositions Array of 3 triangle vertices in object space
/// @param offsetX Ray differential in X direction (from computeDifferentials)
/// @param offsetY Ray differential in Y direction (from computeDifferentials)
/// @param bary Current hit point barycentric coordinates (w, u, v)
/// @return Barycentric derivatives with improved numerical accuracy
float3 computeDerivativeHighp(in float4x3 worldToObject,
                              in float3   worldRayOrigin,
                              in float3   objectRayOrigin,
                              in float3   triangleVertexPositions[3],
                              in float3   offsetX,
                              in float3   offsetY,
                              in float3   bary)
{
  double3 derivativeTargetX = mul(float4(worldRayOrigin + offsetX, 1), worldToObject);
  double3 derivativeDirX    = derivativeTargetX - objectRayOrigin;
  double3 derivativeX       = intersectRayTriangle(objectRayOrigin, derivativeDirX, double3(triangleVertexPositions[0]),
                                                   double3(triangleVertexPositions[1]), double3(triangleVertexPositions[2]));
  derivativeX               = abs(derivativeX - double3(bary));

  double3 derivativeTargetY = mul(float4(worldRayOrigin + offsetY, 1), worldToObject);
  double3 derivativeDirY    = derivativeTargetY - objectRayOrigin;
  double3 derivativeY       = intersectRayTriangle(objectRayOrigin, derivativeDirY, triangleVertexPositions[0],
                                                   triangleVertexPositions[1], triangleVertexPositions[2]);
  derivativeY               = abs(derivativeY - double3(bary));

  return float3(derivativeX + derivativeY);
}

/// Computes the scalar triple product (determinant) of three 3D vectors
/// Helper function for ray-triangle intersection barycentric coordinate calculation
/// @param a First vector
/// @param b Second vector
/// @param c Third vector
/// @return Determinant value: dot(cross(a, b), c)
T determinant<T : __BuiltinFloatingPointType>(vector<T, 3> a, vector<T, 3> b, vector<T, 3> c)
{
  return dot<T>(cross<T>(a, b), c);
}

/// Intersects a ray with a triangle and returns barycentric coordinates
/// Generic implementation supporting both float and double precision
/// Used internally by computeDerivative() and computeDerivativeHighp()
/// @param origin Ray origin point
/// @param direction Ray direction vector (does not need to be normalized)
/// @param v0 First triangle vertex
/// @param v1 Second triangle vertex
/// @param v2 Third triangle vertex
/// @return Barycentric coordinates (w, u, v) where w+u+v=1, corresponding to vertices (v0, v1, v2)
vector<T, 3> intersectRayTriangle<T : __BuiltinFloatingPointType>(vector<T, 3> origin,
                                                                  vector<T, 3> direction,
                                                                  vector<T, 3> v0,
                                                                  vector<T, 3> v1,
                                                                  vector<T, 3> v2)
{
  // Edge vectors
  vector<T, 3> e1 = v1 - v0;
  vector<T, 3> e2 = v2 - v0;

  // Plane normal (cross product of edge vectors)
  vector<T, 3> planeNormal = cross(e1, e2);

  // Check for parallelism (ray parallel to the plane)
  T nDotDir = dot(planeNormal, direction);

  // Distance from ray origin to the plane
  T t = dot(planeNormal, v0 - origin) / nDotDir;

  // Intersection point
  vector<T, 3> p = origin + t * direction;

  // Compute barycentric coordinates using determinant
  vector<T, 3> temp = p - v0;

  T det = determinant(e1, e2, planeNormal);
  T u   = dot(cross(temp, e2), planeNormal) / det;
  T v   = dot(cross(e1, temp), planeNormal) / det;
  T w   = T(1.0) - u - v;

  return vector<T, 3>(w, u, v);
}

