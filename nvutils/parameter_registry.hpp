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

#include "parameter.hpp"

namespace nvutils {


// The ParameterRegistry serves as a central place to register tweakable parameters
// within an application. This allows to reduce describing such parameters multiple times
// redundantly, for example for command line arguments or GUI.
// A Parameter contains a pointer to the destination variable that must be valid while the
// Parameter is used.
//
// All Parameters must be constructed through the registry and their lifetime is linked to the
// lifetime of the registry. They cannot be destroyed individually.
//
// There is a close interaction with the `ParameterParser` class.
class ParameterRegistry
{
public:
  ~ParameterRegistry();


  // retrieve all parameters stored within
  std::span<const ParameterBase* const> getParameters() const
  {
    return std::span(m_parameters.data(), m_parameters.size());
  }

  // Add new parameters to the registry.
  // All destination pointers are copied and must be kept alive.

  const Parameter<bool>* add(const ParameterBase::Info& info, bool* destination);

  // if this parameter is parsed, it will always force setting the provided `triggerValue`
  const Parameter<bool>* add(const ParameterBase::Info& info, bool* destination, bool triggerValue);

  const Parameter<std::string>*           add(const ParameterBase::Info& info, std::string* destination);
  const Parameter<std::filesystem::path>* add(const ParameterBase::Info& info, std::filesystem::path* destination);

  // special filename that can be triggered without parameter name based on argument extension alone
  // extensions must be provided in lowercase
  const Parameter<std::filesystem::path>* add(const ParameterBase::Info&      info,
                                              const std::vector<std::string>& extensions,
                                              std::filesystem::path*          destination);

  // if `extensions` are provided then parser can trigger this based on string suffix alone, rather than requiring the parameter name
  // in that case argCount must be 1 and extensions must be provided in lowercase
  const ParameterBase* addCustom(const ParameterBase::Info&      info,
                                 uint32_t                        argCount,
                                 ParameterBase::CallbackCustom   custom,
                                 const std::vector<std::string>& extensions = {});

  const Parameter<float>* add(const ParameterBase::Info& info,
                              float*                     destination,
                              const float                minValue = std::numeric_limits<float>::min(),
                              const float                maxValue = std::numeric_limits<float>::max());

  // if min/max are left nullptr defaults to min/max numeric_limits
  // length of vector must be within
  const ParameterBase* addArray(const ParameterBase::Info& info,
                                uint32_t                   arrayLength,
                                float*                     destination,
                                const float*               minValues = nullptr,
                                const float*               maxValues = nullptr);

  const Parameter<int32_t>* add(const ParameterBase::Info& info,
                                int32_t*                   destination,
                                const int32_t              minValue = std::numeric_limits<int32_t>::min(),
                                const int32_t              maxValue = std::numeric_limits<int32_t>::max());

  const Parameter<int16_t>* add(const ParameterBase::Info& info,
                                int16_t*                   destination,
                                const int16_t              minValue = std::numeric_limits<int16_t>::min(),
                                const int16_t              maxValue = std::numeric_limits<int16_t>::max());

  const Parameter<int8_t>* add(const ParameterBase::Info& info,
                               int8_t*                    destination,
                               const int8_t               minValue = std::numeric_limits<int8_t>::min(),
                               const int8_t               maxValue = std::numeric_limits<int8_t>::max());

  // if min/max are left nullptr defaults to min/max numeric_limits
  const ParameterBase* addArray(const ParameterBase::Info& info,
                                uint32_t                   arrayLength,
                                int32_t*                   destination,
                                const int32_t*             minValues = nullptr,
                                const int32_t*             maxValues = nullptr);

  const Parameter<uint32_t>* add(const ParameterBase::Info& info,
                                 uint32_t*                  destination,
                                 const uint32_t             minValue = std::numeric_limits<uint32_t>::min(),
                                 const uint32_t             maxValue = std::numeric_limits<uint32_t>::max());

  const Parameter<uint16_t>* add(const ParameterBase::Info& info,
                                 uint16_t*                  destination,
                                 const uint16_t             minValue = std::numeric_limits<uint16_t>::min(),
                                 const uint16_t             maxValue = std::numeric_limits<uint16_t>::max());

  const Parameter<uint8_t>* add(const ParameterBase::Info& info,
                                uint8_t*                   destination,
                                const uint8_t              minValue = std::numeric_limits<uint8_t>::min(),
                                const uint8_t              maxValue = std::numeric_limits<uint8_t>::max());

  // if min/max are left nullptr defaults to min/max numeric_limits
  const ParameterBase* addArray(const ParameterBase::Info& info,
                                uint32_t                   arrayLength,
                                uint32_t*                  destination,
                                const uint32_t*            minValues = nullptr,
                                const uint32_t*            maxValues = nullptr);


  //////////////////////////////////////////////////////////////////////////

  // convenience wrapper to automatically infer the vector dimensions from a glm vector type
  template <typename GLMVEC>
  const Parameter<GLMVEC>* addVector(const ParameterBase::Info& info,
                                     GLMVEC*                    destination,
                                     GLMVEC minValue = GLMVEC(std::numeric_limits<typename GLMVEC::value_type>::min()),
                                     GLMVEC maxValue = GLMVEC(std::numeric_limits<typename GLMVEC::value_type>::max()))
  {
    static_assert(sizeof(GLMVEC) <= sizeof(ParameterBase::MinMaxData));
    return static_cast<const Parameter<GLMVEC>*>(
        addArray(info, GLMVEC::length(), (typename GLMVEC::value_type*)destination, &minValue.x, &maxValue.x));
  }

  template <typename GLMMAT>
  const Parameter<GLMMAT>* addMatrix(const ParameterBase::Info& info, GLMMAT* destination)
  {
    static_assert(sizeof(GLMMAT) <= sizeof(ParameterBase::MinMaxData));
    return static_cast<const Parameter<GLMMAT>*>(addArray(info, GLMMAT::length() * (GLMMAT::col_type::length()),
                                                          (typename GLMMAT::value_type*)destination, nullptr, nullptr));
  }

protected:
  ParameterBase* addNewBase(const ParameterBase::Info& info, ParameterBase::Type type, uint32_t argCount, void* destination);

  // Contains the pointers to all allocated parameters.
  // We allocate parameters individually so their pointers can be used more persistently.
  std::vector<const ParameterBase*> m_parameters;
};

}  // namespace nvutils
