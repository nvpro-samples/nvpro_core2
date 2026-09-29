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


#pragma once

// Vose alias-table builder for environment-map importance sampling.
//
// Split out of `hdr_ibl.hpp` so CPU unit tests (and any host-side tool) can pin the algorithm
// without linking against the full Vulkan / VMA / descriptor stack. It has zero Vulkan
// dependency by design -- the only type it references is the shader-side `shaderio::EnvAccel`
// POD carried by the `hdr_io.h.slang` header (a plain struct of two uint32_t + one float).
//
// The same helper backs both `HdrIbl::loadEnvironment` (file path) and
// `HdrIbl::updateFromGpuImage` (GPU-image path), so any test that pins this function also
// pins both producer flows.

#include <span>
#include <vector>

#include "../nvshaders/hdr_io.h.slang"  // shaderio::EnvAccel

namespace nvvk {

// Build a Vose alias table from a per-texel importance array. `accel` must be sized
// `importance.size()` on entry; on exit it holds one `shaderio::EnvAccel` entry per texel.
// Returns the integral (sum of importance) so callers can normalise the PDF against it.
// If the input importance sums to zero, returns 1.0 (sentinel) so downstream code does not
// divide by zero.
//
// Spans rather than vectors so a caller can hand over memory it already has -- a mapped staging
// buffer, say -- instead of copying into a vector first. A `std::vector` still converts
// implicitly, so existing callers are unaffected.
//
// `accel` is written before it is read, so it does not need to be zero-initialised; and it is
// read back during the build, so it must not be write-combined memory.
float buildEnvAliasmap(std::span<const float> importance, std::span<shaderio::EnvAccel> accel);

}  // namespace nvvk
