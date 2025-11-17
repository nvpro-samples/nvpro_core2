/*
* Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
* SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
* SPDX-License-Identifier: Apache-2.0
*/

#pragma once

#include <functional>
#include <string>
#include <limits>
#include <vector>
#include <span>
#include <filesystem>

#include <fmt/format.h>

namespace nvutils {

// Parameters store information about tweakable values within an application.
// A Parameter contains a pointer to the destination variable that must be valid while the
// Parameter is used.
// Parameters can only be constructed through the `ParameterRegistry`

class ParameterBase
{
public:
  enum class Type
  {
    BOOL8,
    BOOL8_TRIGGER,
    FLOAT32,
    INT8,
    INT16,
    INT32,
    UINT8,
    UINT16,
    UINT32,
    STRING,
    FILENAME,
    CUSTOM,
    INVALID,
  };

  // this basic callback is triggered after successful parsing
  using CallbackSuccess = std::function<void(const ParameterBase* const)>;

  // custom parameter callback
  // `filenameBasePath` is provided through the parser and typically the working directory or a file being parsed
  using CallbackCustom =
      std::function<bool(const ParameterBase* const, std::span<char const* const> args, const std::filesystem::path& filenameBasePath)>;

  struct Info
  {
    std::string     name;              // required, parser prefixes "--"
    std::string     help;              // optional
    std::string     shortName;         // optional, parser prefixes "-"
    std::string     guiName;           // optional, defaults to regular help
    std::string     guiHelp;           // optional, defaults to regular name
    uint32_t        visibility = ~0u;  // optional, allows custom filtering for parameters (TBD)
    CallbackSuccess callbackSuccess;   // optional, triggers after parsing was completed successfully
  };

  static constexpr size_t MAX_ARRAY_LENGTH = 16;

  Type type = Type::INVALID;
  Info info;

  // how many arguments this parameter needs for parsing
  uint32_t argCount = 0;

  // custom callback for CUSTOM
  CallbackCustom callbackCustom;

  // special case, allows parser to trigger this parameter without
  // a leading "--name" keyword, just tests the parameter suffix
  // uses lowered string!
  std::vector<std::string> extensions;

  // for all others pointers are used during parsing
  union
  {
    bool*                  b8;
    float*                 f32;
    int8_t*                i8;
    int16_t*               i16;
    int32_t*               i32;
    uint8_t*               u8;
    uint16_t*              u16;
    uint32_t*              u32;
    void*                  raw;
    std::string*           string;
    std::filesystem::path* filename;
  } destination;

  // parsing can enforce a per-component min/max logic
  union MinMaxData
  {
    float    f32[MAX_ARRAY_LENGTH];
    int8_t   i8[MAX_ARRAY_LENGTH];
    int16_t  i16[MAX_ARRAY_LENGTH];
    int32_t  i32[MAX_ARRAY_LENGTH];
    uint8_t  u8[MAX_ARRAY_LENGTH];
    uint16_t u16[MAX_ARRAY_LENGTH];
    uint32_t u32[MAX_ARRAY_LENGTH];
  } minMaxValues[2];

  // basic type string
  // e.g. `float[3]` or `bool`
  std::string getTypeString() const
  {
    const char* typeString = toString(type);
    return argCount ? fmt::format("{}[{}]", typeString, argCount) : fmt::format("{}", typeString);
  }

  static const char* toString(Type type);

private:
  ParameterBase() = default;

  friend class ParameterRegistry;
};

// template wrapper class to access the destination pointer in a type-safe fashion
template <class T>
class Parameter : public ParameterBase
{
  T* data() const { return (T*)destination.raw; }
  T  min() const { return *(const T*)&minMaxValues[0].u32[0]; }
  T  max() const { return *(const T*)&minMaxValues[1].u32[0]; }
};

}  // namespace nvutils
