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

/*

Utilities for working with SPIR-V data.

*/

#include <filesystem>
#include <cstdint>
#include <cstddef>
#include <span>

namespace nvutils {

// Hash the SPIR-V code
std::size_t hashSpirv(const uint32_t* spirvData, size_t spirvSize);

// Dump the SPIR-V code to a file with a hashed name
std::filesystem::path dumpSpirvName(const std::filesystem::path& filename, const uint32_t* spirvData, size_t spirvSize);

// Dump the SPIR-V code to a file
void dumpSpirv(const std::filesystem::path& filename, const uint32_t* spirvData, size_t spirvSize);

// Dump the SPIR-V code to a file with a hashed name
void dumpSpirvWithHashedName(const std::filesystem::path& sourceFile, const uint32_t* spirvData, size_t spirvSize);

}  // namespace nvutils