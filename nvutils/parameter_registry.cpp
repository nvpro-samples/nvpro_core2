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

#include <cassert>
#include <cstring>

#include <fmt/format.h>

#include "parameter_registry.hpp"

namespace nvutils {

ParameterBase* ParameterRegistry::addNewBase(const ParameterBase::Info& info, ParameterBase::Type type, uint32_t argCount, void* destination)
{
  ParameterBase* parameter = new ParameterBase;
  m_parameters.push_back(parameter);

  parameter->info = info;

  if(info.guiName.empty())
    parameter->info.guiName = info.name;
  if(info.guiHelp.empty())
    parameter->info.guiHelp = info.help;

  parameter->type            = type;
  parameter->argCount        = argCount;
  parameter->callbackCustom  = {};
  parameter->destination.raw = destination;
  memset(&parameter->minMaxValues, 0, sizeof(ParameterBase::MinMaxData) * 2);

  return parameter;
}

const Parameter<bool>* ParameterRegistry::add(const ParameterBase::Info& info, bool* destination)
{
  ParameterBase* parameter = addNewBase(info, ParameterBase::Type::BOOL8, 1, destination);

  return static_cast<Parameter<bool>*>(parameter);
}

const Parameter<bool>* ParameterRegistry::add(const ParameterBase::Info& info, bool* destination, bool value)
{
  ParameterBase* parameter = addNewBase(info, ParameterBase::Type::BOOL8_TRIGGER, 0, destination);

  parameter->minMaxValues[0].u32[0] = value ? 1 : 0;

  return static_cast<Parameter<bool>*>(parameter);
}

const Parameter<float>* ParameterRegistry::add(const ParameterBase::Info& info,
                                               float*                     destination,
                                               float minValue /*= std::numeric_limits<float>::min*/,
                                               float maxValue /*= std::numeric_limits<float>::max*/)
{
  ParameterBase* parameter = addNewBase(info, ParameterBase::Type::FLOAT32, 1, destination);

  parameter->minMaxValues[0].f32[0] = minValue;
  parameter->minMaxValues[1].f32[0] = maxValue;

  return static_cast<Parameter<float>*>(parameter);
}

const Parameter<int8_t>* ParameterRegistry::add(const ParameterBase::Info& info,
                                                int8_t*                    destination,
                                                int8_t minValue /*= std::numeric_limits<int8_t>::min*/,
                                                int8_t maxValue /*= std::numeric_limits<int8_t>::max*/)
{
  ParameterBase* parameter = addNewBase(info, ParameterBase::Type::INT8, 1, destination);

  parameter->minMaxValues[0].i8[0] = minValue;
  parameter->minMaxValues[1].i8[0] = maxValue;

  return static_cast<Parameter<int8_t>*>(parameter);
}


const Parameter<int16_t>* ParameterRegistry::add(const ParameterBase::Info& info,
                                                 int16_t*                   destination,
                                                 int16_t minValue /*= std::numeric_limits<int16_t>::min*/,
                                                 int16_t maxValue /*= std::numeric_limits<int16_t>::max*/)
{
  ParameterBase* parameter = addNewBase(info, ParameterBase::Type::INT16, 1, destination);

  parameter->minMaxValues[0].i16[0] = minValue;
  parameter->minMaxValues[1].i16[0] = maxValue;

  return static_cast<Parameter<int16_t>*>(parameter);
}

const Parameter<int32_t>* ParameterRegistry::add(const ParameterBase::Info& info,
                                                 int32_t*                   destination,
                                                 int32_t minValue /*= std::numeric_limits<int32_t>::min*/,
                                                 int32_t maxValue /*= std::numeric_limits<int32_t>::max*/)
{
  ParameterBase* parameter = addNewBase(info, ParameterBase::Type::INT32, 1, destination);

  parameter->minMaxValues[0].i32[0] = minValue;
  parameter->minMaxValues[1].i32[0] = maxValue;

  return static_cast<Parameter<int32_t>*>(parameter);
}

const Parameter<uint8_t>* ParameterRegistry::add(const ParameterBase::Info& info,
                                                 uint8_t*                   destination,
                                                 uint8_t minValue /*= std::numeric_limits<uint8_t>::min*/,
                                                 uint8_t maxValue /*= std::numeric_limits<uint8_t>::max*/)
{
  ParameterBase* parameter = addNewBase(info, ParameterBase::Type::UINT8, 1, destination);

  parameter->minMaxValues[0].u8[0] = minValue;
  parameter->minMaxValues[1].u8[0] = maxValue;

  return static_cast<Parameter<uint8_t>*>(parameter);
}


const Parameter<uint16_t>* ParameterRegistry::add(const ParameterBase::Info& info,
                                                  uint16_t*                  destination,
                                                  uint16_t minValue /*= std::numeric_limits<uint16_t>::min*/,
                                                  uint16_t maxValue /*= std::numeric_limits<uint16_t>::max*/)
{
  ParameterBase* parameter = addNewBase(info, ParameterBase::Type::UINT16, 1, destination);

  parameter->minMaxValues[0].u16[0] = minValue;
  parameter->minMaxValues[1].u16[0] = maxValue;

  return static_cast<Parameter<uint16_t>*>(parameter);
}


const Parameter<uint32_t>* ParameterRegistry::add(const ParameterBase::Info& info,
                                                  uint32_t*                  destination,
                                                  uint32_t minValue /*= std::numeric_limits<uint32_t>::min*/,
                                                  uint32_t maxValue /*= std::numeric_limits<uint32_t>::max*/)
{
  ParameterBase* parameter = addNewBase(info, ParameterBase::Type::UINT32, 1, destination);

  parameter->minMaxValues[0].u32[0] = minValue;
  parameter->minMaxValues[1].u32[0] = maxValue;

  return static_cast<Parameter<uint32_t>*>(parameter);
}

const ParameterBase* ParameterRegistry::addArray(const ParameterBase::Info& info,
                                                 uint32_t                   arrayLength,
                                                 float*                     destination,
                                                 const float*               minValues /*= nullptr*/,
                                                 const float*               maxValues /*= nullptr*/)
{
  assert(arrayLength <= ParameterBase::MAX_ARRAY_LENGTH);

  ParameterBase* parameter = addNewBase(info, ParameterBase::Type::FLOAT32, arrayLength, destination);
  for(uint32_t i = 0; i < arrayLength; i++)
  {
    parameter->minMaxValues[0].f32[i] = minValues ? minValues[i] : std::numeric_limits<float>::min();
    parameter->minMaxValues[1].f32[i] = maxValues ? maxValues[i] : std::numeric_limits<float>::min();
  }

  return parameter;
}

const ParameterBase* ParameterRegistry::addArray(const ParameterBase::Info& info,
                                                 uint32_t                   arrayLength,
                                                 int32_t*                   destination,
                                                 const int32_t*             minValues /*= nullptr*/,
                                                 const int32_t*             maxValues /*= nullptr*/)
{
  assert(arrayLength <= ParameterBase::MAX_ARRAY_LENGTH);

  ParameterBase* parameter = addNewBase(info, ParameterBase::Type::INT32, arrayLength, destination);
  for(uint32_t i = 0; i < arrayLength; i++)
  {
    parameter->minMaxValues[0].i32[i] = minValues ? minValues[i] : std::numeric_limits<int32_t>::min();
    parameter->minMaxValues[1].i32[i] = maxValues ? maxValues[i] : std::numeric_limits<int32_t>::max();
  }

  return parameter;
}

const ParameterBase* ParameterRegistry::addArray(const ParameterBase::Info& info,
                                                 uint32_t                   arrayLength,
                                                 uint32_t*                  destination,
                                                 const uint32_t*            minValues /*= nullptr*/,
                                                 const uint32_t*            maxValues /*= nullptr*/)
{
  assert(arrayLength <= ParameterBase::MAX_ARRAY_LENGTH);

  ParameterBase* parameter = addNewBase(info, ParameterBase::Type::UINT32, arrayLength, destination);
  for(uint32_t i = 0; i < arrayLength; i++)
  {
    parameter->minMaxValues[0].u32[i] = minValues ? minValues[i] : std::numeric_limits<uint32_t>::min();
    parameter->minMaxValues[1].u32[i] = maxValues ? maxValues[i] : std::numeric_limits<uint32_t>::max();
  }

  return parameter;
}

const Parameter<std::string>* ParameterRegistry::add(const ParameterBase::Info& info, std::string* destination)
{
  ParameterBase* parameter = addNewBase(info, ParameterBase::Type::STRING, 1, destination);

  return static_cast<Parameter<std::string>*>(parameter);
}

const Parameter<std::filesystem::path>* ParameterRegistry::add(const ParameterBase::Info& info, std::filesystem::path* destination)
{
  ParameterBase* parameter = addNewBase(info, ParameterBase::Type::FILENAME, 1, destination);

  return static_cast<Parameter<std::filesystem::path>*>(parameter);
}

const nvutils::Parameter<std::filesystem::path>* ParameterRegistry::add(const ParameterBase::Info&      info,
                                                                        const std::vector<std::string>& extensions,
                                                                        std::filesystem::path*          destination)
{
  ParameterBase* parameter = addNewBase(info, ParameterBase::Type::FILENAME, 1, destination);
  parameter->extensions    = extensions;

  return static_cast<Parameter<std::filesystem::path>*>(parameter);
}

const ParameterBase* ParameterRegistry::addCustom(const ParameterBase::Info&      info,
                                                  uint32_t                        argCount,
                                                  ParameterBase::CallbackCustom   custom,
                                                  const std::vector<std::string>& extensions)
{
  assert(extensions.empty() || argCount == 1);

  ParameterBase* parameter  = addNewBase(info, ParameterBase::Type::CUSTOM, argCount, nullptr);
  parameter->callbackCustom = std::move(custom);
  parameter->extensions     = extensions;

  return parameter;
}

ParameterRegistry::~ParameterRegistry()
{
  for(auto& it : m_parameters)
  {
    delete it;
  }
}


}  // namespace nvutils


//--------------------------------------------------------------------------------------------------
// Usage example
//--------------------------------------------------------------------------------------------------
[[maybe_unused]] static void usage_ParameterRegistry()
{
  nvutils::ParameterRegistry registry;

  bool     blubb = false;
  uint32_t blah  = 123;

  // register some parameters
  const nvutils::Parameter<bool>* blubbParameter =
      registry.add({"blubb", "triggering this parameter enables blubb"}, &blubb, true);

  const nvutils::Parameter<uint32_t>* blahParameter = registry.add({"blah", "modifies blah, clamped to [0,10]"}, &blah, 0, 10);

  // later you can use the parameters to generate UI elements or command line parser options
  // see `ParameterParser`
}
