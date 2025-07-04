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


#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <libloaderapi.h>  // GetModuleFileNameA
#include <stringapiset.h>  // WideChartoMultiByte
#else
#include <limits.h>     // PATH_MAX
#include <strings.h>    // strcasecmp
#include <sys/types.h>  // ssize_t
#include <unistd.h>     // readlink
#endif

#include "file_operations.hpp"
#include "logger.hpp"

#include <fstream>
#include <limits>
#include <tuple>


std::filesystem::path nvutils::findFile(const std::filesystem::path&              filename,
                                        const std::vector<std::filesystem::path>& searchPaths,
                                        bool                                      reportError)
{
  for(const auto& path : searchPaths)
  {
    const std::filesystem::path filePath = path / filename;
    if(std::filesystem::exists(filePath))
    {
      return filePath;
    }
  }
  nvutils::Logger::getInstance().log(reportError ? nvutils::Logger::LogLevel::eERROR : nvutils::Logger::LogLevel::eWARNING,
                                     "File not found: %s\n", utf8FromPath(filename).c_str());
  LOGI("Searched under: \n");
  for(const auto& path : searchPaths)
  {
    LOGI("  %s\n", utf8FromPath(path).c_str());
  }
  return std::filesystem::path();
}


std::string nvutils::loadFile(const std::filesystem::path& filePath)
{
  std::ifstream file(filePath, std::ios::binary | std::ios::ate);
  if(!file)
  {
    LOGW("Could not open file: %s\n", utf8FromPath(filePath).c_str());
    return "";
  }

  std::streamsize size = file.tellg();
  file.seekg(0, std::ios::beg);

  std::vector<char> buffer(size);
  if(!file.read(buffer.data(), size))
  {
    LOGW("Error reading file: %s\n", utf8FromPath(filePath).c_str());
    return "";
  }

  return std::string(buffer.begin(), buffer.end());
}

std::filesystem::path nvutils::getExecutablePath()
{
#ifdef _WIN32
  wchar_t     buffer[MAX_PATH + 1]{};
  const DWORD count = GetModuleFileNameW(NULL, buffer, sizeof(buffer) / sizeof(buffer[0]));
  // `buffer` is guaranteed to be null-terminated on Windows Vista and later
  return std::filesystem::path(buffer, buffer + count);
#else
  char          buffer[PATH_MAX + 1]{};
  const ssize_t count = readlink("/proc/self/exe", buffer, sizeof(buffer) / sizeof(buffer[0]));
  return std::filesystem::path(buffer, buffer + ((count > 0) ? count : 0));
#endif
}

std::string nvutils::utf8FromPath(const std::filesystem::path& path) noexcept
{
  try
  {
#ifdef _WIN32
    // On Windows, paths are UTF-16, possibly with unpaired surrogates.
    // There's several possible routes to convert to UTF-8; we use WideCharToMultiByte because
    // - <codecvt> is deprecated and scheduled to be removed from C++ (see https://github.com/microsoft/STL/issues/443)
    // -  _wcstombs_s_l requires a locale object
    // We choose to reject unpaired surrogates using WC_ERR_INVALID_CHARS,
    // in case downstream code expects fully valid UTF-8.
    if(path.empty())  // Convert empty strings without producing errors
    {
      return "";
    }
    const wchar_t* utf16Str        = path.c_str();
    const size_t   utf16Characters = wcslen(utf16Str);
    // Cast to int for WideCharToMultiByte
    if(utf16Characters > std::numeric_limits<int>::max())
    {
      LOGE("%s: Input had too many characters to store in an int.\n", __func__);
      return "";
    }
    const int utf16CharactersI = static_cast<int>(utf16Characters);
    // Get output size (does not include terminating null since we specify cchWideChar)
    const int utf8Bytes = WideCharToMultiByte(CP_UTF8,               // CodePage
                                              WC_ERR_INVALID_CHARS,  // dwFlags
                                              utf16Str,              // lpWideCharStr
                                              utf16CharactersI,      // cchWideChar
                                              nullptr, 0,            // Output
                                              nullptr, nullptr);     // lpDefaultChar, lpUsedDefaultChar
    // WideCharToMultiByte returns 0 on failure. Check for that plus negative
    // values (which chould never happen):
    if(utf8Bytes <= 0)
    {
      LOGE("%s: WideCharToMultiByte failed. The path is probably not valid UTF-16.\n", __func__);
      return "";
    }
    std::string result(utf8Bytes, char(0));
    std::ignore = WideCharToMultiByte(CP_UTF8, 0, utf16Str, utf16CharactersI, result.data(), utf8Bytes, nullptr, nullptr);
    return result;
#else
    // On other platforms, paths are UTF-8 already.
    // Catch exceptions just to make this function 100% noexcept.
    static_assert(std::is_same_v<std::filesystem::path::string_type::value_type, char>);
    return path.string();
#endif
  }
  catch(const std::exception& e)  // E.g. out of memory
  {
    LOGE("%s threw an exception: %s\n", __func__, e.what());
    return "";
  }
}

std::filesystem::path nvutils::pathFromUtf8(const char* utf8) noexcept
{
  // Since this is the inverse of pathToUtf8, see pathToUtf8 for implementation
  // notes.
  try
  {
#ifdef _WIN32
    const size_t utf8Bytes = strlen(utf8);
    if(utf8Bytes == 0)
    {
      return L"";
    }
    if(utf8Bytes > std::numeric_limits<int>::max())
    {
      LOGE("%s: Input had too many characters to store in an int.\n", __func__);
      return L"";
    }
    const int utf8BytesI      = static_cast<int>(utf8Bytes);
    const int utf16Characters = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, utf8, utf8BytesI, nullptr, 0);
    if(utf16Characters <= 0)
    {
      LOGE("%s: MultiByteToWideChar failed. The input is probably not valid UTF-8.\n", __func__);
      return L"";
    }
    static_assert(std::is_same_v<std::filesystem::path::string_type, std::wstring>);
    std::wstring result(utf16Characters, wchar_t(0));
    std::ignore = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, utf8, utf8BytesI, result.data(), utf16Characters);
    return std::filesystem::path(std::move(result));
#else
    return std::filesystem::path(utf8);
#endif
  }
  catch(const std::exception& e)
  {
    LOGE("%s threw an exception: %s\n", __func__, e.what());
#ifdef _WIN32  // Avoid _convert_narrow_to_wide, just in case
    return L"";
#else
    return "";
#endif
  }
}

bool nvutils::extensionMatches(const std::filesystem::path& path, const char* extension)
{
  // The standard implementation of this, tolower(path.extension()) == extension,
  // would use 3 allocations: path.extension(), a copy for tolower, and one for
  // fs::path(extension) (since == is only implemented for path == path).
  // We can bring this down to 1 on Windows and 0 on Linux with a custom implementation.

  // First, find the character where the extension starts.
  // Because we're just testing whether the extension matches, we don't need
  // to handle things like Windows' NTFS Alternate Data Streams.
  const std::filesystem::path::string_type&   native = path.native();
  constexpr std::filesystem::path::value_type dot    = '.';  // Same value in UTF-8 and UTF-16
  const auto                                  dotPos = native.rfind(dot);
  if(dotPos == native.npos)  // No extension?
  {
    return extension == nullptr || extension[0] == '\0';
  }

#ifdef _WIN32
  // UTF-8 to UTF-16 copy
  const std::filesystem::path              extensionUTF16 = pathFromUtf8(extension);
  const std::filesystem::path::value_type* extensionBytes = extensionUTF16.native().c_str();
  return 0 == _wcsicmp(native.c_str() + dotPos, extensionBytes);
#else
  return 0 == strcasecmp(native.c_str() + dotPos, extension);
#endif
}
