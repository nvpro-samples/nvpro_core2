/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */


// CPU implementation of Vose's alias method for environment-map importance sampling. See
// `env_aliasmap.hpp` for the rationale for owning its own translation unit. Kept dependency-
// free (no Vulkan, no VMA, no descriptors) so CPU tests can link it directly.

#include "env_aliasmap.hpp"

#include <memory>
#include <numeric>

namespace nvvk {

float buildEnvAliasmap(std::span<const float> data, std::span<shaderio::EnvAccel> accel)
{
  auto size = static_cast<uint32_t>(data.size());

  // Compute the integral of the emitted radiance of the environment map. Since each element in
  // `data` is already weighted by its solid angle, the integral is a simple sum.
  float sum = std::accumulate(data.begin(), data.end(), 0.F);
  if(sum == 0.0f)
  {
    // Sentinel to keep the sampling shader safe from division by zero when the caller feeds a
    // fully-black environment. Every accel entry stays as its own alias with q == 0, so the
    // sampler will simply pick the uniform draw and return that texel (which is black).
    sum = 1.0f;
  }

  // For each texel, compute the ratio q between the emitted radiance of the texel and the
  // average emitted radiance over the entire sphere. We also initialise the aliases to
  // identity, i.e. each texel is its own alias.
  auto  f_size          = static_cast<float>(size);
  float inverse_average = f_size / sum;
  for(uint32_t i = 0; i < size; ++i)
  {
    accel[i].q     = data[i] * inverse_average;
    accel[i].alias = i;
  }

  // Partition the texels according to their emitted radiance ratio wrt. average. Texels with
  // a value q < 1 (i.e., below average) are stored incrementally from the beginning of the
  // array; texels emitting higher-than-average radiance are stored from the end of the array.
  // Every element is assigned before it is read, so zero-initialising it is wasted work -- and at
  // a 2048x1024 bake that is 8 MB of memset on the critical path of an interactive commit.
  std::unique_ptr<uint32_t[]> partition_table = std::make_unique_for_overwrite<uint32_t[]>(size);
  uint32_t                    s               = 0U;
  uint32_t                    large           = size;
  for(uint32_t i = 0; i < size; ++i)
  {
    if(accel[i].q < 1.F)
      partition_table[s++] = i;
    else
      partition_table[--large] = i;
  }

  // Associate the lower-energy texels to higher-energy ones. Since the emission of a
  // high-energy texel may be vastly superior to the average, a single high-energy texel may
  // absorb the deficit of many low-energy neighbours before its own q drops below 1.
  for(s = 0; s < large && large < size; ++s)
  {
    const uint32_t small_energy_index = partition_table[s];
    const uint32_t high_energy_index  = partition_table[large];

    // Associate the texel to its higher-energy alias.
    accel[small_energy_index].alias = high_energy_index;

    // Compute the difference between the lower-energy texel and the average.
    const float difference_with_average = 1.F - accel[small_energy_index].q;

    // The goal is to obtain texel couples whose combined intensity is close to the average.
    // Some texels may have very low energies, others may have very high intensity (a sunset:
    // the sky is dark, but the sun is very bright). In this case it may not be possible to
    // obtain a value close to average by combining only two texels; we then associate a
    // single high-energy texel with many smaller-energy ones. We keep track of the combined
    // average by subtracting the deficit from the ratio stored in the high-energy texel.
    accel[high_energy_index].q -= difference_with_average;

    // If the combined ratio to average of the higher-energy texel reaches 1, a balance has
    // been found; move to the next higher-energy texel in the partition on the next iteration.
    if(accel[high_energy_index].q < 1.0F)
      large++;
  }
  // Return the integral of the emitted radiance. The caller uses it to normalise the PDF of
  // each texel.
  return sum;
}

}  // namespace nvvk
