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

#include "text_encoding.hpp"

#include "basic_types.hpp"  // first: Windows headers with NOMINMAX

#include <filesystem>

#include <dxgi1_6.h>

#include <nvutils/file_operations.hpp>

namespace nvdx12 {

// Both conversions go through std::filesystem::path, whose native encoding
// on Windows is the wide format; nvutils owns the UTF-8 side.

std::string utf8FromWtf16(const wchar_t* wtf16)
{
  return wtf16 ? nvutils::utf8FromPath(std::filesystem::path(wtf16)) : std::string();
}

std::string utf8FromWtf16(const std::wstring& wtf16)
{
  return nvutils::utf8FromPath(std::filesystem::path(wtf16));
}

std::wstring wtf16FromUtf8(const char* utf8)
{
  return utf8 ? nvutils::pathFromUtf8(utf8).native() : std::wstring();
}

std::wstring wtf16FromUtf8(const std::string& utf8)
{
  return nvutils::pathFromUtf8(utf8).native();
}

}  // namespace nvdx12

//--------------------------------------------------------------------------------------------------
// Usage example
//--------------------------------------------------------------------------------------------------
[[maybe_unused]] static void usage_TextEncoding()
{
  // D3D12 and DXGI speak wide text; the rest of nvpro_core2 speaks UTF-8.
  ID3D12Object* object = nullptr;  // EX: any device child
  object->SetName(nvdx12::wtf16FromUtf8("Scene vertices").c_str());

  DXGI_ADAPTER_DESC1 desc{};  // EX: from IDXGIAdapter1::GetDesc1
  const std::string  adapterName = nvdx12::utf8FromWtf16(desc.Description);
}
