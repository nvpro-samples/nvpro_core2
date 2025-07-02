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

#include <third_party/hash_combine/hash_combine.hpp>

namespace nvutils {

//---- Hash Combination ----
template <typename T>
void hashCombine(std::size_t& seed, const T& val)
{
  boost::hash_combine(seed, val);
}
// Auxiliary generic functions to create a hash value using a seed
template <typename T, typename... Types>
void hashCombine(std::size_t& seed, const T& val, const Types&... args)
{
  hashCombine(seed, val);
  hashCombine(seed, args...);
}
// Optional auxiliary generic functions to support hash_val() without arguments
inline void hashCombine(std::size_t& seed) {}
// Generic function to create a hash value out of a heterogeneous list of arguments
template <typename... Types>
std::size_t hashVal(const Types&... args)
{
  std::size_t seed = 0;
  hashCombine(seed, args...);
  return seed;
}

}  // namespace nvutils
