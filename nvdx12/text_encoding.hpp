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

#include <string>

namespace nvdx12 {

/*-------------------------------------------------------------------------------------------------
# Text encoding

Conversions between UTF-8 (nvpro_core2, the logger, file names) and the
wide text of the Windows API and D3D12 (`SetName`, adapter descriptions,
dxc arguments). The wide format is a superset of UTF-16 that allows
unpaired surrogates, hence WTF-16. A null pointer converts to an empty string.

See usage_TextEncoding() in text_encoding.cpp.
-------------------------------------------------------------------------------------------------*/

std::string  utf8FromWtf16(const wchar_t* wtf16);
std::string  utf8FromWtf16(const std::wstring& wtf16);
std::wstring wtf16FromUtf8(const char* utf8);
std::wstring wtf16FromUtf8(const std::string& utf8);

}  // namespace nvdx12
