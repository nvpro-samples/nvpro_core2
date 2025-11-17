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

#include "parameter.hpp"

namespace nvutils {

const char* ParameterBase::toString(Type type)
{
  switch(type)
  {
    case ParameterBase::Type::BOOL8:
      return "bool";
    case ParameterBase::Type::BOOL8_TRIGGER:
      return "bool_trigger";
    case ParameterBase::Type::FLOAT32:
      return "float";
    case ParameterBase::Type::INT8:
      return "int8";
    case ParameterBase::Type::INT16:
      return "int16";
    case ParameterBase::Type::INT32:
      return "int32";
    case ParameterBase::Type::UINT8:
      return "uint8";
    case ParameterBase::Type::UINT16:
      return "uint16";
    case ParameterBase::Type::UINT32:
      return "uint32";
    case ParameterBase::Type::STRING:
      return "string";
    case ParameterBase::Type::FILENAME:
      return "filename";
    case ParameterBase::Type::CUSTOM:
      return "custom";
    default:
      return "";
  }
}
}  // namespace nvutils
