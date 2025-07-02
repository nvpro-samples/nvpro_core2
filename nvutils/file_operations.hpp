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
#include <string>
#include <vector>
#include <filesystem>

namespace nvutils {

// Return the path to a file if it exists in one of the search paths.
// if `reportError` is true, will report missing file as error, otherwise as warning.
std::filesystem::path findFile(const std::filesystem::path&              filename,
                               const std::vector<std::filesystem::path>& searchPaths,
                               bool                                      reportError = true);

// Open a file and return its content as a string. On error, returns an
// empty string.
std::string loadFile(const std::filesystem::path& filePath);

// Return the path to the executable
std::filesystem::path getExecutablePath();

// Converts an `std::filesystem::path` to an `std::string` in UTF-8 encoding.
// On error (such as if the path contains unpaired surrogates), prints an
// error message and returns an empty string.
std::string utf8FromPath(const std::filesystem::path& path) noexcept;

// Converts a UTF-8 `std::string to an `std::filesystem::path`.
// On error (such as if the string is not valid UTF-8), prints an error message
// and returns an empty path.
std::filesystem::path        pathFromUtf8(const char* utf8) noexcept;
inline std::filesystem::path pathFromUtf8(const std::string& utf8) noexcept
{
  return pathFromUtf8(utf8.c_str());
}

// Returns whether a path has the given extension. The comparison is case-insensitive.
// If the path has no extension, returns (extension == "").
// For example, extensionMatches("foo.txt", ".txt") returns `true`.
bool extensionMatches(const std::filesystem::path& path, const char* extension);

}  // namespace nvutils
