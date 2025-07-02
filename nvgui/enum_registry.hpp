/*
 * Copyright (c) 2018-2025, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2018-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include <imgui/imgui.h>

#include <string>
#include <vector>

namespace nvgui {

// This helper allows to use ImGui::BeginCombo/EndCombo that return pre recorded values
// usage: see usage_EnumRegistry in .cpp
class EnumRegistry
{
private:
  enum ValueType
  {
    TYPE_INT,
    TYPE_FLOAT
  };

  struct Enum
  {
    union
    {
      int   ivalue = 0;
      float fvalue;
    };
    std::string name;
    bool        disabled = false;
  };

  struct Entry
  {
    std::vector<Enum> enums;
    ValueType         valueType    = ValueType::TYPE_INT;
    bool              valueChanged = false;
  };
  std::vector<Entry> entries;

public:
  const std::vector<Enum>& getEnums(uint32_t type) const { return entries[type].enums; }

  void enumAdd(uint32_t type, int value, const char* name, bool disabled = false)
  {
    if(type >= entries.size())
    {
      entries.resize(type + 1ULL);
    }
    entries[type].enums.push_back({{value}, name, disabled});
    entries[type].valueChanged = false;
    entries[type].valueType = TYPE_INT;  // the user must be consistent so that he adds only the same type for the same combo !
  }

  void enumAdd(uint32_t type, float value, const char* name, bool disabled = false)
  {
    if(type >= entries.size())
    {
      entries.resize(type + 1ULL);
    }
    Enum e;
    e.fvalue   = value;
    e.name     = name;
    e.disabled = disabled;
    entries[type].enums.push_back(e);
    entries[type].valueChanged = false;
    entries[type].valueType = TYPE_FLOAT;  // the user must be consistent so that he adds only the same type for the same combo !
  }

  void enumReset(uint32_t type)
  {
    if(type < entries.size())
    {
      entries[type].enums.clear();
      entries[type].valueChanged = false;
      entries[type].valueType    = TYPE_INT;
    }
  }

  bool enumCombobox(uint32_t type, const char* label, void* value, ImGuiComboFlags flags = 0, bool* valueChanged = NULL)
  {
    bool bRes = Combo(label, entries[type].enums.size(), entries[type].enums.data(), value, flags,
                      entries[type].valueType, &entries[type].valueChanged);
    if(valueChanged)
    {
      *valueChanged = entries[type].valueChanged;
    }
    return bRes;
  }

private:
  bool Combo(const char*     label,
             size_t          numEnums,
             const Enum*     enums,
             void*           value,
             ImGuiComboFlags flags        = 0,
             ValueType       valueType    = TYPE_INT,
             bool*           valueChanged = NULL);

};
}  // namespace nvgui