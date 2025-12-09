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

#define GLFW_INCLUDE_NONE
#include "nvgui/enum_registry.hpp"
#include <backends/imgui_impl_glfw.h>
#include <GLFW/glfw3.h>
#include <math.h>


#include <fstream>

namespace nvgui {

bool EnumRegistry::Combo(const char* label, size_t numEnums, const Enum* enums, void* valuePtr, ImGuiComboFlags flags, ValueType valueType, bool* valueChanged)
{
  int*   ivalue = (int*)valuePtr;
  float* fvalue = (float*)valuePtr;

  size_t idx     = 0;
  bool   found   = false;
  bool   changed = false;
  for(size_t i = 0; i < numEnums; i++)
  {
    switch(valueType)
    {
      case TYPE_INT:
        if(enums[i].ivalue == *ivalue)
        {
          idx   = i;
          found = true;
        }
        break;
      case TYPE_FLOAT:
        if(enums[i].fvalue == *fvalue)
        {
          idx   = i;
          found = true;
        }
        break;
      default:
        break;
    }
  }

  if(!found)
  {
    assert(!"No such value in combo!");
    return false;
  }

  // The second parameter is the label previewed before opening the combo.
  if(ImGui::BeginCombo(label, enums[idx].name.c_str(), flags))
  {
    for(size_t i = 0; i < numEnums; i++)
    {
      ImGui::BeginDisabled(enums[i].disabled);
      bool is_selected = i == idx;
      if(ImGui::Selectable(enums[i].name.c_str(), is_selected))
      {
        switch(valueType)
        {
          case TYPE_INT:
            *ivalue = enums[i].ivalue;
            break;
          case TYPE_FLOAT:
            *fvalue = enums[i].fvalue;
            break;
        }

        changed = true;
      }
      if(is_selected)
      {
        // Set the initial focus when opening the combo (scrolling +
        // for keyboard navigation support in the upcoming navigation branch)
        ImGui::SetItemDefaultFocus();
      }
      ImGui::EndDisabled();
    }
    ImGui::EndCombo();
  }

  if(valueChanged)
    *valueChanged = changed;

  return changed;
}
}  // namespace nvgui

[[maybe_unused]] static void usage_EnumRegistry()
{
  // A - Register the text item selectors, each list associated with an enum
  enum
  {
    MY_SELECTOR1,
    MY_SELECTOR2,
  };
  nvgui::EnumRegistry registry;
  // First selector
  registry.enumAdd(MY_SELECTOR1, 0, "Buffers");
  registry.enumAdd(MY_SELECTOR1, 1, "Textures");
  // Second selector (can register any integer as second value)
  registry.enumAdd(MY_SELECTOR2, 10, "First choice label");
  registry.enumAdd(MY_SELECTOR2, 24, "Second choice label");
  // etc.

  // B - Then in ui render, display the choice list for a given enum
  int choiceResultInteger;
  if(registry.enumCombobox(MY_SELECTOR2, "##ImGuiID", &choiceResultInteger))
  {
    // then choiceResultInteger recieved the value associated with selected text label
  }

  // C - same but used within a PropertyEditor
  // if(nvgui::PropertyEditor::entry(
  //        "Storage", [&] { return registry.enumCombobox(MY_SELECTOR2, "##ImGuiID", &choiceResultInteger); },
  //        "tooltip text"))
  // {
  //   // then choiceResultInteger recieved the value associated with selected text label
  // }
}