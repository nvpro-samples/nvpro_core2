/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

// The D3D12 Memory Allocator implementation (third_party/d3d12ma), compiled
// into nvdx12. Unfreed allocations of its heaps are reported through the
// nvutils logger when the allocator is released, as nvvk's
// VMA_LEAK_LOG_FORMAT does for VMA; nvdx12::ResourceAllocator names them
// "nvdx12AllocID: <id>" (see setLeakID). Unfreed committed allocations only
// trip D3D12MA's assert in debug builds.

#include "basic_types.hpp"  // first: Windows headers with NOMINMAX

#include <cwchar>

#include <nvutils/logger.hpp>

#define D3D12MA_DEBUG_LOG(format, ...)                                                                                 \
  do                                                                                                                   \
  {                                                                                                                    \
    wchar_t d3d12maMessage[512];                                                                                       \
    swprintf_s(d3d12maMessage, format, __VA_ARGS__);                                                                   \
    LOGW("D3D12MA: %ls\n", d3d12maMessage);                                                                            \
  } while(false)

#include <third_party/d3d12ma/src/D3D12MemAlloc.cpp>
