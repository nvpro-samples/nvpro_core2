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

/*
 * Tonemapper Compute Shader
 * 
 * Performs HDR to LDR conversion with automatic exposure control.
 * 
 * Pipeline:
 * 1. Input: Linear HDR texture (RGB32F) 
 * 2. Auto-exposure calculation via luminance histogram analysis
 * 3. Exposure compensation and tonemapping application
 * 4. Output: sRGB LDR texture (RGB8)
 * 
 * Auto-exposure algorithm:
 * - Generates luminance histogram from input buffer
 * - Supports center-weighted metering (optional)
 * - Computes exposure value targeting middle grey (18%)
 * - Applies temporal smoothing to prevent flickering
 * 
 * Histogram processing:
 * - Maps EV100 values to histogram buckets
 * - Supports both mean and median averaging modes
 * - Excludes near-black pixels from exposure calculation
 * - Uses parallel prefix sum for efficient histogram analysis
 */


#include "nvshaders/tonemap_functions.h.slang"
#include "nvshaders/functions.h.slang"

layout(push_constant) ConstantBuffer<TonemapperData> tm;

layout(binding = TonemapBinding::eImageInput) Texture2D<float4> InColorBuffer;
layout(binding = TonemapBinding::eImageOutput) RWTexture2D<float4> outImage;
layout(binding = TonemapBinding::eHistogramInputOutput) RWStructuredBuffer<uint> InOutHistogram;
layout(binding = TonemapBinding::eLuminanceInputOutput) RWStructuredBuffer<float> OutLuminance;

groupshared float g_localFloatData[EXPOSURE_HISTOGRAM_SIZE];  // shared memory for the prefix sum
groupshared uint  g_localUintData[EXPOSURE_HISTOGRAM_SIZE];   // shared memory for the histogram

//////////////////////////////////////////////////////////////////////////////
// Tonemapping
/////////////////////////////////////////////////////////////////////////////

[shader("compute")]
[numthreads(TONEMAP_WORKGROUP_SIZE, TONEMAP_WORKGROUP_SIZE, 1)]
void Tonemap(uint2 globalThreadID: SV_DispatchThreadID)
{
  uint2 imageSize;
  outImage.GetDimensions(imageSize.x, imageSize.y);

  // Check if the thread is within the image bounds
  if(any(globalThreadID >= imageSize))
    return;

  // Get the color from the input texture
  float4 color = InColorBuffer[int2(globalThreadID.xy)];

  // If the tonemapper is active
  if(tm.isActive == 1)
  {
    // If auto-exposure is enabled
    if(tm.autoExposure == 1)
    {
      // Apply auto-exposure compensation
      // OutLuminance[0] contains the target luminance calculated from histogram analysis
      // We want to expose the scene so that the target luminance maps to middle grey (0.18)
      const float middleGrey         = 0.18f;
      float       exposureMultiplier = middleGrey / max(0.001f, OutLuminance[0]);
      color.xyz *= exposureMultiplier;
    }

    // Apply the tonemapping pipeline
    color.xyz = applyTonemap(tm, color.xyz, float2(globalThreadID.xy), float2(imageSize));
  }

  // Write the color to the output texture
  outImage[int2(globalThreadID.xy)] = color;
}


//////////////////////////////////////////////////////////////////////////////
// Histogram Tonemapping
/////////////////////////////////////////////////////////////////////////////

/* 
Histogram Construction:

- Builds a 1D luminance histogram (EXPOSURE_HISTOGRAM_SIZE bins) mapping pixels to EV100-based bins
- Bin 0 reserved for near-black pixels (excluded from exposure calculations)
- Bins represent quantized EV100 intervals using evMinValue and evMaxValue parameters
- Supports median (50th percentile) and mean modes for exposure estimation
- Enables real-time adaptive exposure control for HDR rendering
*/


// This function classifies a pixel into a histogram bucket based on its luminance.
uint inputToHistogramBucket(const float3 inputColor)
{
  const float inputLuminance = bt709Luminance(inputColor);

  // Classify near-black pixels (excluded from exposure calculation)
  const float kEpsilon = 0.00001f;
  if(inputLuminance < kEpsilon)
  {
    return 0;
  }

  // Convert luminance to EV100 and normalize to [0,1] range
  const float ev100        = luminanceEv100(inputLuminance);
  const float evRange      = tm.evMaxValue - tm.evMinValue;
  const float normalizedEV = saturate((ev100 - tm.evMinValue) / evRange);

  // Map to histogram bucket [1, size-2]
  return uint(normalizedEVToHistogramOffset(normalizedEV));
}

// This function calculates a weight for a pixel based on its distance from the center of the screen.
// It is used to weight the pixels in the histogram calculation.
float centerMeteringWeight(float2 screenUV, float aspect)
{
  if(tm.enableCenterMetering == 0)
  {
    return 1.f;  // Uniform weighting when center metering disabled
  }

  // Calculate distance from screen center, normalized by aspect ratio
  const float weight = saturate(length((screenUV - float2(0.5f)) / float2(1.0f, aspect)) / tm.centerMeteringSize);
  return lerp(1.0f, 0.0f, weight);  // Full weight at center, zero at edges
}

// This function builds the histogram of the input image.
// It is used to determine the target luminance for the auto-exposure algorithm.
// The histogram is a 1D texture with 256 bins, each bin representing a range of EV100 values.
// The histogram is then used to determine the target luminance for the auto-exposure algorithm.
// The target luminance is calculated by finding the median of the histogram.
[shader("compute")]
[numthreads(TONEMAP_WORKGROUP_SIZE, TONEMAP_WORKGROUP_SIZE, 1)]
void Histogram(uint2 threadId: SV_DispatchThreadID, uint linearIndex: SV_GroupIndex)
{
  g_localUintData[linearIndex] = 0;

  // Wait for all threads to finish their local calculations
  GroupMemoryBarrierWithGroupSync();

  // Get the dimensions of the input image
  uint2 dimensions;
  InColorBuffer.GetDimensions(dimensions.x, dimensions.y);

  // If the thread is within the image bounds
  if(all(threadId < dimensions))
  {
    const float3 inputColor = InColorBuffer[threadId].xyz;

    // Center-weighted metering (optional)
    const float weight = centerMeteringWeight(float2(threadId) / dimensions, (float)dimensions.x / dimensions.y);

    // Classify pixel into histogram bucket based on luminance
    const uint bucketIdx = inputToHistogramBucket(inputColor);

    // Weight the contribution to the histogram
    InterlockedAdd(g_localUintData[bucketIdx], uint(weight * 255.0f));
  }

  // Wait for all threads to finish their local calculations
  GroupMemoryBarrierWithGroupSync();

  // Propagate local histogram to global histogram buffer
  uint localBinValue = g_localUintData[linearIndex];
  if(localBinValue != 0)
  {
    InterlockedAdd(InOutHistogram[linearIndex], localBinValue);
  }
}


/////////////////////////////////////////////////////////////////////////////////////////
// Auto Exposure
/////////////////////////////////////////////////////////////////////////////////////////

/*
Auto-Exposure Calculation:

- Computes target luminance from histogram using median or mean mode
- Applies optional exposure compensation weighting
- Uses parallel prefix sum for efficient histogram analysis
- Temporally smooths luminance to prevent flickering
- Writes target luminance to OutLuminance[0] for use in tonemapping

*/

// This function maps a normalized EV100 value to a histogram bucket index.
// It is used to map the exposure value to a histogram bucket.
float normalizedEVToHistogramOffset(float normalizedEV)
{
  // Map normalized EV100 [0,1] to histogram bucket index [1, size-2]
  // Bucket 0 reserved for near-black pixels, bucket size-1 for overexposed
  return (saturate(normalizedEV) * float(EXPOSURE_HISTOGRAM_SIZE - 2)) + 1;
}

// This function calculates an "inclusive prefix sum" (also called a scan) across all threads in the group.
// For example, given input values [3, 10, 5, 1], the output will be [3, 13, 18, 19].
// Each output element is the sum of all input elements up to and including its own position.
// This is useful for things like cumulative histograms or parallel reductions.
// The implementation first uses a fast "wave-level" prefix sum within small groups of threads (waves).
// Then, it combines the results from each wave in a hierarchical way so that the sum is correct across the entire group.
// This approach is much faster than having each thread sum all previous values by itself.
// Note that this is not the standard Hillis-Steele short-span prefix sum; that one requires double-buffering
// the data array to avoid race conditions, while this one doesn't.
float computePrefixSum(float val, const uint linearIndex)
{
  // Wait for all threads to finish their local calculations
  GroupMemoryBarrierWithGroupSync();

  // Convert exclusive to inclusive prefix sum (add current value)
  val = WavePrefixSum(val) + val;
  // Each thread writes to its own slot in shared memory
  g_localFloatData[linearIndex] = val;
  // Wait for all threads to finish their local calculations
  GroupMemoryBarrierWithGroupSync();

  // Hierarchical propagation: double block size each iteration
  // This is a way to efficiently calculate the prefix sum across all threads in the group.
  // It is done by doubling the block size -- and the distance by which sums are propagated -- each iteration.
  // This is a much faster way to calculate the prefix sum than having each thread sum all previous values by itself.
  // https://en.wikipedia.org/wiki/Prefix_sum#Algorithm_1:_Shorter_span,_more_parallel
  for(uint k = WaveGetLaneCount(); k < EXPOSURE_HISTOGRAM_SIZE; k *= 2)
  {
    uint blockIndex = linearIndex / k;
    // For each odd-numbered block, add the sum of the elements from the partial prefix sum of the last block
    // (which will be the last element of that block)
    if((blockIndex % 2) == 1)
      g_localFloatData[linearIndex] += g_localFloatData[k * blockIndex - 1];

    GroupMemoryBarrierWithGroupSync();
  }

  return g_localFloatData[EXPOSURE_HISTOGRAM_SIZE - 1];
}

// This function returns a weight for a histogram bucket.
// It is used to weight the pixels in the histogram calculation.
float getHistogramWeight(uint index)
{
  return 1.0;
  // float weightUV = (index + 0.5) / float(EXPOSURE_HISTOGRAM_SIZE);
  // return InExposureCompensation.SampleLevel(weightUV, 0).r;
}

// This function converts a histogram bucket index to a normalized EV100 value.
// It is used to convert the exposure value to a normalized EV100 value.
float histogramOffsetToNormalizedEV(float offset)
{
  // Inverse mapping: histogram bucket index -> normalized EV100 [0,1]
  return saturate((offset - 1) / float(EXPOSURE_HISTOGRAM_SIZE - 2));
}

// This function calculates the target luminance for the auto-exposure algorithm.
// linearIndex is the index of the thread in the group.
[shader("compute")]
[numthreads(EXPOSURE_HISTOGRAM_SIZE, 1, 1)]
void AutoExposure(uint2 threadId: SV_DispatchThreadID, uint linearIndex: SV_GroupIndex)
{
  // Apply optional exposure compensation weighting to histogram bins
  float weight                 = 1.0;  // (tm.useExposureCompensation == 1) ? getHistogramWeight(linearIndex) : 1.0f;
  float countThisBucket        = (float)InOutHistogram[linearIndex] * weight;
  float totalWeight            = computePrefixSum(countThisBucket, linearIndex);
  float adaptedCountThisBucket = countThisBucket;

  if(tm.averageMode == 0)  // mean mode
  {
    // Weight by bucket index for mean calculation: sum(index * count) / sum(count)
    adaptedCountThisBucket *= linearIndex;
  }

  // Compute weighted sum for exposure calculation
  const float weightedSum = computePrefixSum(adaptedCountThisBucket, linearIndex);

  float normalizedEV = 0;
  if(linearIndex == 0)
  {
    uint discardSampleCount = uint(countThisBucket);  // Near-black pixels in bucket 0 (excluded from exposure calc)

    float weightedAverage;

    if(tm.averageMode == 1)  // median mode
    {
      // Find median: 50th percentile of non-black pixels
      float median    = (weightedSum - discardSampleCount) / 2.f;
      weightedAverage = 0.5f;  // Initialize to first bucket center

      for(uint i = 1; i < EXPOSURE_HISTOGRAM_SIZE; i++)
      {
        if(g_localFloatData[i] > median)
        {
          break;
        }

        weightedAverage = i + 0.5f;  // Use bucket center (index + 0.5)
      }
    }
    else  // mean mode
    {
      // Calculate weighted mean: sum(index * count) / sum(count)
      // Excludes near-black pixels from denominator
      weightedAverage = weightedSum / totalWeight;
    }

    // Convert histogram bucket back to EV100 scale
    normalizedEV = histogramOffsetToNormalizedEV(weightedAverage);

    // Convert normalized EV100 to EV100
    const float evRange         = tm.evMaxValue - tm.evMinValue;
    const float currentEV100    = normalizedEV * evRange + tm.evMinValue;
    const float targetLuminance = ev100Luminance(currentEV100);

    // Get previous luminance
    const float previousLuminance = OutLuminance[0];

    // Smooth luminance adjustment using time-based adaptation
    const float adaptedLuminance =
        previousLuminance + (targetLuminance - previousLuminance) * (1.f - exp2(-tm.autoExposureSpeed));

    if(linearIndex == 0)  // Only the first thread writes the final result
    {
      // Write luminance value to output
      OutLuminance[0] = adaptedLuminance;
    }
  }

  // Clear histogram for next frame
  InOutHistogram[linearIndex] = 0;
}
