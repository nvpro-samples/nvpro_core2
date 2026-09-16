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

#pragma once

#include "element_mcp.hpp"

#include <span>
#include <vector>

namespace nvutils {
class ParameterBase;
class ParameterRegistry;
}  // namespace nvutils

namespace nvmcp::detail {

using ParameterList = std::vector<const nvutils::ParameterBase*>;

ParameterList selectParameters(const nvutils::ParameterRegistry& registry, std::initializer_list<std::string_view> names);
#if defined(NVPRO2_TRACK_PARAMETER_REGISTRIES)
struct ParameterSelection
{
  ParameterList            parameters;
  std::vector<std::string> duplicateNames;
};
ParameterSelection selectAllParameters(std::span<const nvutils::ParameterBase* const> parameters);
#endif
ToolResult getParameters(std::span<const nvutils::ParameterBase* const> parameters, std::string_view arguments);
ToolResult setParameters(std::span<const nvutils::ParameterBase* const> parameters, std::string_view arguments);

}  // namespace nvmcp::detail
