/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

/*-------------------------------------------------------------------------------------------------
# Basic types

Included by every nvdx12 header: Windows.h with the lean macros set
(NOMINMAX keeps `std::min` / `std::max` usable), d3d12.h, and the COM smart
pointer the helpers hold their objects with. Include it before any header
that uses `std::numeric_limits<>::min/max()`.
-------------------------------------------------------------------------------------------------*/

#ifndef NOMINMAX
#define NOMINMAX
#endif

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif

#ifndef WIN32_EXTRA_LEAN
#define WIN32_EXTRA_LEAN
#endif

#include <d3d12.h>
#include <Windows.h>
#include <wrl/client.h>

namespace nvdx12 {

// See https://github.com/microsoft/DirectXTK/wiki/ComPtr
template <typename T>
using ComPtr = Microsoft::WRL::ComPtr<T>;

}  // namespace nvdx12
