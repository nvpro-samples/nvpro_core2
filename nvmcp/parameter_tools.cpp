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

#include "parameter_tools.hpp"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <mcp_message.h>
#include <nvutils/file_operations.hpp>
#include <nvutils/parameter_registry.hpp>

namespace nvmcp::detail {
namespace {

using Parameter = nvutils::ParameterBase;

bool isSupported(const Parameter& parameter)
{
  // Triggers and custom parsers are actions rather than inspectable state;
  // samples should expose those deliberately as named tools.
  return parameter.destination.raw && parameter.type != Parameter::Type::BOOL8_TRIGGER
         && parameter.type != Parameter::Type::CUSTOM && parameter.type != Parameter::Type::INVALID;
}

const Parameter& findParameter(std::span<const Parameter* const> parameters, std::string_view name)
{
  for(size_t parameterIndex = 0; parameterIndex < parameters.size(); parameterIndex++)
  {
    const Parameter* parameter = parameters[parameterIndex];
    if(parameter->info.name == name)
      return *parameter;
  }

  throw std::invalid_argument("unknown exposed parameter: " + std::string(name));
}

template <typename T>
mcp::json numberJson(T value)
{
  if constexpr(std::is_floating_point_v<T>)
    return value;
  else if constexpr(std::is_signed_v<T>)
    return static_cast<int64_t>(value);
  else
    return static_cast<uint64_t>(value);
}

template <typename T>
mcp::json numericData(const T* values, uint32_t count)
{
  if(count == 1)
    return numberJson(values[0]);

  mcp::json result = mcp::json::array();
  for(uint32_t i = 0; i < count; i++)
  {
    result.push_back(numberJson(values[i]));
  }
  return result;
}

template <typename T>
void addNumericData(mcp::json& result, const Parameter& parameter, const T* destination, const T* minimum, const T* maximum)
{
  result["value"]   = numericData(destination, parameter.argCount);
  result["minimum"] = numericData(minimum, parameter.argCount);
  result["maximum"] = numericData(maximum, parameter.argCount);
}

mcp::json parameterJson(const Parameter& parameter)
{
  mcp::json result{
      {"name", parameter.info.name},
      {"description", parameter.info.help},
      {"type", parameter.getTypeString()},
  };

  switch(parameter.type)
  {
    case Parameter::Type::BOOL8:
      result["value"] = parameter.destination.b8[0];
      break;
    case Parameter::Type::FLOAT32:
      addNumericData(result, parameter, parameter.destination.f32, parameter.minMaxValues[0].f32,
                     parameter.minMaxValues[1].f32);
      break;
    case Parameter::Type::INT8:
      addNumericData(result, parameter, parameter.destination.i8, parameter.minMaxValues[0].i8,
                     parameter.minMaxValues[1].i8);
      break;
    case Parameter::Type::INT16:
      addNumericData(result, parameter, parameter.destination.i16, parameter.minMaxValues[0].i16,
                     parameter.minMaxValues[1].i16);
      break;
    case Parameter::Type::INT32:
      addNumericData(result, parameter, parameter.destination.i32, parameter.minMaxValues[0].i32,
                     parameter.minMaxValues[1].i32);
      break;
    case Parameter::Type::UINT8:
      addNumericData(result, parameter, parameter.destination.u8, parameter.minMaxValues[0].u8,
                     parameter.minMaxValues[1].u8);
      break;
    case Parameter::Type::UINT16:
      addNumericData(result, parameter, parameter.destination.u16, parameter.minMaxValues[0].u16,
                     parameter.minMaxValues[1].u16);
      break;
    case Parameter::Type::UINT32:
      addNumericData(result, parameter, parameter.destination.u32, parameter.minMaxValues[0].u32,
                     parameter.minMaxValues[1].u32);
      break;
    case Parameter::Type::STRING:
      result["value"] = *parameter.destination.string;
      break;
    case Parameter::Type::FILENAME:
      result["value"] = nvutils::utf8FromPath(*parameter.destination.filename);
      break;
    default:
      throw std::logic_error("unsupported exposed parameter: " + parameter.info.name);
  }

  return result;
}

const mcp::json& component(const mcp::json& value, uint32_t count, uint32_t index, std::string_view name)
{
  if(count == 1)
  {
    if(value.is_array())
      throw std::invalid_argument("parameter " + std::string(name) + " requires a scalar value");
    return value;
  }

  if(!value.is_array() || value.size() != count)
  {
    throw std::invalid_argument("parameter " + std::string(name) + " requires an array of " + std::to_string(count) + " values");
  }
  return value[index];
}

template <typename T>
T numericValue(const mcp::json& value, T minimum, T maximum, std::string_view name)
{
  if constexpr(std::is_floating_point_v<T>)
  {
    if(!value.is_number())
      throw std::invalid_argument("parameter " + std::string(name) + " requires a number");
  }
  else
  {
    if(!value.is_number_integer() && !value.is_number_unsigned())
      throw std::invalid_argument("parameter " + std::string(name) + " requires an integer");
  }

  const long double number = value.get<long double>();
  if(number < static_cast<long double>(minimum) || number > static_cast<long double>(maximum))
  {
    throw std::out_of_range("parameter " + std::string(name) + " is outside its supported range");
  }
  return static_cast<T>(number);
}

template <typename T>
void validateNumeric(const Parameter& parameter, const mcp::json& value, const T* minimum, const T* maximum)
{
  for(uint32_t i = 0; i < parameter.argCount; i++)
  {
    numericValue(component(value, parameter.argCount, i, parameter.info.name), minimum[i], maximum[i], parameter.info.name);
  }
}

void validateValue(const Parameter& parameter, const mcp::json& value)
{
  switch(parameter.type)
  {
    case Parameter::Type::BOOL8:
      if(!value.is_boolean())
        throw std::invalid_argument("parameter " + parameter.info.name + " requires a boolean");
      break;
    case Parameter::Type::FLOAT32:
      validateNumeric(parameter, value, parameter.minMaxValues[0].f32, parameter.minMaxValues[1].f32);
      break;
    case Parameter::Type::INT8:
      validateNumeric(parameter, value, parameter.minMaxValues[0].i8, parameter.minMaxValues[1].i8);
      break;
    case Parameter::Type::INT16:
      validateNumeric(parameter, value, parameter.minMaxValues[0].i16, parameter.minMaxValues[1].i16);
      break;
    case Parameter::Type::INT32:
      validateNumeric(parameter, value, parameter.minMaxValues[0].i32, parameter.minMaxValues[1].i32);
      break;
    case Parameter::Type::UINT8:
      validateNumeric(parameter, value, parameter.minMaxValues[0].u8, parameter.minMaxValues[1].u8);
      break;
    case Parameter::Type::UINT16:
      validateNumeric(parameter, value, parameter.minMaxValues[0].u16, parameter.minMaxValues[1].u16);
      break;
    case Parameter::Type::UINT32:
      validateNumeric(parameter, value, parameter.minMaxValues[0].u32, parameter.minMaxValues[1].u32);
      break;
    case Parameter::Type::STRING:
    case Parameter::Type::FILENAME:
      if(!value.is_string())
        throw std::invalid_argument("parameter " + parameter.info.name + " requires a string");
      break;
    default:
      throw std::logic_error("unsupported exposed parameter: " + parameter.info.name);
  }
}

template <typename T>
void applyNumeric(const Parameter& parameter, const mcp::json& value, T* destination, const T* minimum, const T* maximum)
{
  for(uint32_t i = 0; i < parameter.argCount; i++)
  {
    destination[i] = numericValue(component(value, parameter.argCount, i, parameter.info.name), minimum[i], maximum[i],
                                  parameter.info.name);
  }
}

void applyValue(const Parameter& parameter, const mcp::json& value)
{
  switch(parameter.type)
  {
    case Parameter::Type::BOOL8:
      parameter.destination.b8[0] = value.get<bool>();
      break;
    case Parameter::Type::FLOAT32:
      applyNumeric(parameter, value, parameter.destination.f32, parameter.minMaxValues[0].f32, parameter.minMaxValues[1].f32);
      break;
    case Parameter::Type::INT8:
      applyNumeric(parameter, value, parameter.destination.i8, parameter.minMaxValues[0].i8, parameter.minMaxValues[1].i8);
      break;
    case Parameter::Type::INT16:
      applyNumeric(parameter, value, parameter.destination.i16, parameter.minMaxValues[0].i16, parameter.minMaxValues[1].i16);
      break;
    case Parameter::Type::INT32:
      applyNumeric(parameter, value, parameter.destination.i32, parameter.minMaxValues[0].i32, parameter.minMaxValues[1].i32);
      break;
    case Parameter::Type::UINT8:
      applyNumeric(parameter, value, parameter.destination.u8, parameter.minMaxValues[0].u8, parameter.minMaxValues[1].u8);
      break;
    case Parameter::Type::UINT16:
      applyNumeric(parameter, value, parameter.destination.u16, parameter.minMaxValues[0].u16, parameter.minMaxValues[1].u16);
      break;
    case Parameter::Type::UINT32:
      applyNumeric(parameter, value, parameter.destination.u32, parameter.minMaxValues[0].u32, parameter.minMaxValues[1].u32);
      break;
    case Parameter::Type::STRING:
      *parameter.destination.string = value.get<std::string>();
      break;
    case Parameter::Type::FILENAME:
      *parameter.destination.filename = nvutils::pathFromUtf8(value.get<std::string>());
      break;
    default:
      throw std::logic_error("unsupported exposed parameter: " + parameter.info.name);
  }
}

mcp::json parseObject(std::string_view arguments)
{
  const mcp::json result = arguments.empty() ? mcp::json::object() : mcp::json::parse(arguments);
  if(!result.is_object())
    throw std::invalid_argument("tool arguments must be a JSON object");
  return result;
}

}  // namespace

ParameterList selectParameters(const nvutils::ParameterRegistry& registry, std::initializer_list<std::string_view> names)
{
  if(names.size() == 0)
    throw std::invalid_argument("at least one MCP parameter name is required");

  ParameterList result;
  result.reserve(names.size());
  for(size_t nameIndex = 0; nameIndex < names.size(); nameIndex++)
  {
    const std::string_view                  name = *(names.begin() + nameIndex);
    const Parameter*                        selected{};
    const std::span<const Parameter* const> registryParameters = registry.getParameters();
    for(size_t parameterIndex = 0; parameterIndex < registryParameters.size(); parameterIndex++)
    {
      const Parameter* parameter = registryParameters[parameterIndex];
      if(parameter->info.name == name)
      {
        selected = parameter;
        break;
      }
    }

    if(!selected)
      throw std::invalid_argument("unknown parameter selected for MCP: " + std::string(name));
    if(!isSupported(*selected))
      throw std::invalid_argument("unsupported parameter selected for MCP: " + std::string(name));
    bool duplicate{};
    for(size_t resultIndex = 0; resultIndex < result.size(); resultIndex++)
    {
      if(result[resultIndex] == selected)
      {
        duplicate = true;
        break;
      }
    }
    if(duplicate)
      throw std::invalid_argument("duplicate parameter selected for MCP: " + std::string(name));
    result.push_back(selected);
  }
  return result;
}

#if defined(NVPRO2_TRACK_PARAMETER_REGISTRIES)
ParameterSelection selectAllParameters(std::span<const Parameter* const> parameters)
{
  ParameterSelection selection;
  for(size_t parameterIndex = 0; parameterIndex < parameters.size(); parameterIndex++)
  {
    const Parameter* parameter = parameters[parameterIndex];
    if(!isSupported(*parameter))
      continue;

    bool duplicate{};
    for(size_t otherIndex = 0; otherIndex < parameters.size(); otherIndex++)
    {
      if(otherIndex != parameterIndex && isSupported(*parameters[otherIndex])
         && parameters[otherIndex]->info.name == parameter->info.name)
      {
        duplicate = true;
        break;
      }
    }
    if(duplicate)
    {
      bool nameRecorded{};
      for(size_t duplicateIndex = 0; duplicateIndex < selection.duplicateNames.size(); duplicateIndex++)
      {
        if(selection.duplicateNames[duplicateIndex] == parameter->info.name)
        {
          nameRecorded = true;
          break;
        }
      }
      if(!nameRecorded)
        selection.duplicateNames.push_back(parameter->info.name);
      continue;
    }
    selection.parameters.push_back(parameter);
  }
  return selection;
}
#endif

ToolResult getParameters(std::span<const Parameter* const> parameters, std::string_view arguments)
{
  const mcp::json input = parseObject(arguments);
  if(input.size() > 1 || (input.size() == 1 && !input.contains("names")))
    throw std::invalid_argument("nvpro_get_parameters accepts only an optional names array");

  mcp::json result = mcp::json::array();
  if(!input.contains("names"))
  {
    for(size_t parameterIndex = 0; parameterIndex < parameters.size(); parameterIndex++)
    {
      result.push_back(parameterJson(*parameters[parameterIndex]));
    }
  }
  else
  {
    const mcp::json& names = input["names"];
    if(!names.is_array() || names.empty())
      throw std::invalid_argument("nvpro_get_parameters names must be a non-empty array");

    std::vector<std::string> selectedNames;
    selectedNames.reserve(names.size());
    for(size_t nameIndex = 0; nameIndex < names.size(); nameIndex++)
    {
      if(!names[nameIndex].is_string())
        throw std::invalid_argument("nvpro_get_parameters names must contain strings");
      const std::string name = names[nameIndex].get<std::string>();
      if(std::find(selectedNames.begin(), selectedNames.end(), name) != selectedNames.end())
        throw std::invalid_argument("duplicate nvpro_get_parameters name: " + name);
      selectedNames.push_back(name);
      result.push_back(parameterJson(findParameter(parameters, name)));
    }
  }

  return ToolResult::success(mcp::json{{"parameters", std::move(result)}}.dump(2));
}

ToolResult setParameters(std::span<const Parameter* const> parameters, std::string_view arguments)
{
  const mcp::json input = parseObject(arguments);
  if(input.size() != 1 || !input.contains("values") || !input["values"].is_object() || input["values"].empty())
    throw std::invalid_argument("nvpro_set_parameters requires a non-empty values object");

  struct Change
  {
    const Parameter* parameter;
    mcp::json        value;
  };

  std::vector<Change> changes;
  changes.reserve(input["values"].size());
  // Validate the complete agent request before mutating sample state so a
  // rejected batch cannot leave a partially applied configuration.
  for(mcp::json::const_iterator valueIt = input["values"].cbegin(); valueIt != input["values"].cend(); valueIt++)
  {
    const std::string& name      = valueIt.key();
    const mcp::json&   value     = valueIt.value();
    const Parameter&   parameter = findParameter(parameters, name);
    validateValue(parameter, value);
    changes.push_back({&parameter, value});
  }

  for(size_t changeIndex = 0; changeIndex < changes.size(); changeIndex++)
  {
    const Change& change = changes[changeIndex];
    applyValue(*change.parameter, change.value);
  }
  for(size_t changeIndex = 0; changeIndex < changes.size(); changeIndex++)
  {
    const Change& change = changes[changeIndex];
    if(change.parameter->info.callbackSuccess)
      change.parameter->info.callbackSuccess(change.parameter);
  }

  mcp::json updated = mcp::json::array();
  for(size_t changeIndex = 0; changeIndex < changes.size(); changeIndex++)
  {
    const Change& change = changes[changeIndex];
    updated.push_back(parameterJson(*change.parameter));
  }
  return ToolResult::success(mcp::json{{"updated", std::move(updated)}}.dump(2));
}

}  // namespace nvmcp::detail
