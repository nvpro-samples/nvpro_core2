/*
 * Copyright (c) 2023-2025, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */


#include "property_editor.hpp"
#include "tooltip.hpp"

namespace nvgui {
namespace PropertyEditor {

template <typename T>
bool Clamped(bool changed, T* value, T min, T max)
{
  *value = std::max(min, std::min(max, *value));
  return changed;
}

// Beginning the Property Editor
bool begin(const char* label, ImGuiTableFlags flag)
{
  ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(2, 2));
  bool result = ImGui::BeginTable(label, 2, flag);
  if(!result)
  {
    ImGui::PopStyleVar();
  }
  return result;
}

// Generic entry, the lambda function should return true if the widget changed
bool entry(const std::string& property_name, const std::function<bool()>& content_fct, const std::string& tooltip)
{
  ImGui::PushID(property_name.c_str());
  ImGui::TableNextRow();
  ImGui::TableNextColumn();
  ImGui::AlignTextToFramePadding();
  ImGui::Text("%s", property_name.c_str());
  if(!tooltip.empty())
    nvgui::tooltip(tooltip.c_str(), false, 0);
  ImGui::TableNextColumn();
  ImGui::SetNextItemWidth(-FLT_MIN);
  bool result = content_fct();
  if(!tooltip.empty())
    nvgui::tooltip(tooltip.c_str());
  ImGui::PopID();
  return result;  // returning if the widget changed
}

// Text specialization
void entry(const std::string& property_name, const std::string& value)
{
  entry(property_name, [&] {
    ImGui::Text("%s", value.c_str());
    return false;  // dummy, no change
  });
}

bool treeNode(const std::string& name, ImGuiTreeNodeFlags flags)
{
  ImGui::TableNextRow();
  ImGui::TableNextColumn();
  ImGui::AlignTextToFramePadding();
  return ImGui::TreeNodeEx(name.c_str(), flags);
}
void treePop()
{
  ImGui::TreePop();
}

// Ending the Editor
void end()
{
  ImGui::EndTable();
  ImGui::PopStyleVar();
}

bool Button(const char* label, const ImVec2& size, const std::string& tooltip)
{
  return PropertyEditor::entry(label, [&] { return ImGui::Button("##hidden", size); }, tooltip);
}
bool SmallButton(const char* label, const std::string& tooltip)
{
  return entry(label, [&] { return ImGui::SmallButton("##hidden"); }, tooltip);
}
bool Checkbox(const char* label, bool* v, const std::string& tooltip)
{
  return entry(label, [&] { return ImGui::Checkbox("##hidden", v); }, tooltip);
}
bool RadioButton(const char* label, bool active, const std::string& tooltip)
{
  return entry(label, [&] { return ImGui::RadioButton("##hidden", active); }, tooltip);
}
bool RadioButton(const char* label, int* v, int v_button, const std::string& tooltip)
{
  return entry(label, [&] { return ImGui::RadioButton("##hidden", v, v_button); }, tooltip);
}
bool Combo(const char* label, int* current_item, const char* const items[], int items_count, int popup_max_height_in_items, const std::string& tooltip)
{
  return entry(label,
               [&] { return ImGui::Combo("##hidden", current_item, items, items_count, popup_max_height_in_items); });
}
bool Combo(const char* label, int* current_item, const char* items_separated_by_zeros, int popup_max_height_in_items, const std::string& tooltip)
{

  return entry(
      label, [&] { return ImGui::Combo("##hidden", current_item, items_separated_by_zeros, popup_max_height_in_items); }, tooltip);
}
bool Combo(const char* label,
           int*        current_item,
           const char* (*getter)(void* user_data, int idx),
           void*              user_data,
           int                items_count,
           int                popup_max_height_in_items,
           const std::string& tooltip)
{
  return entry(
      label,
      [&] { return ImGui::Combo("##hidden", current_item, getter, user_data, items_count, popup_max_height_in_items); }, tooltip);
}
bool SliderFloat(const char* label, float* v, float v_min, float v_max, const char* format, ImGuiSliderFlags flags, const std::string& tooltip)
{
  return entry(label, [&] { return ImGui::SliderFloat("##hidden", v, v_min, v_max, format, flags); }, tooltip);
}
bool SliderFloat2(const char* label, float v[2], float v_min, float v_max, const char* format, ImGuiSliderFlags flags, const std::string& tooltip)
{
  return entry(label, [&] { return ImGui::SliderFloat2("##hidden", v, v_min, v_max, format, flags); }, tooltip);
}
bool SliderFloat3(const char* label, float v[3], float v_min, float v_max, const char* format, ImGuiSliderFlags flags, const std::string& tooltip)
{
  return entry(label, [&] { return ImGui::SliderFloat3("##hidden", v, v_min, v_max, format, flags); }, tooltip);
}
bool SliderFloat4(const char* label, float v[4], float v_min, float v_max, const char* format, ImGuiSliderFlags flags, const std::string& tooltip)
{
  return entry(label, [&] { return ImGui::SliderFloat4("##hidden", v, v_min, v_max, format, flags); }, tooltip);
}
bool SliderAngle(const char*        label,
                 float*             v_rad,
                 float              v_degrees_min,
                 float              v_degrees_max,
                 const char*        format,
                 ImGuiSliderFlags   flags,
                 const std::string& tooltip)
{
  return entry(label, [&] { return ImGui::SliderAngle("##hidden", v_rad, v_degrees_min, v_degrees_max, format, flags); }, tooltip);
}
bool SliderInt(const char* label, int* v, int v_min, int v_max, const char* format, ImGuiSliderFlags flags, const std::string& tooltip)
{
  return entry(label, [&] { return ImGui::SliderInt("##hidden", v, v_min, v_max, format, flags); }, tooltip);
}
bool SliderInt2(const char* label, int v[2], int v_min, int v_max, const char* format, ImGuiSliderFlags flags, const std::string& tooltip)
{
  return entry(label, [&] { return ImGui::SliderInt2("##hidden", v, v_min, v_max, format, flags); }, tooltip);
}
bool SliderInt3(const char* label, int v[3], int v_min, int v_max, const char* format, ImGuiSliderFlags flags, const std::string& tooltip)
{
  return entry(label, [&] { return ImGui::SliderInt3("##hidden", v, v_min, v_max, format, flags); }, tooltip);
}
bool SliderInt4(const char* label, int v[4], int v_min, int v_max, const char* format, ImGuiSliderFlags flags, const std::string& tooltip)
{
  return entry(label, [&] { return ImGui::SliderInt4("##hidden", v, v_min, v_max, format, flags); }, tooltip);
}
bool SliderScalar(const char*        label,
                  ImGuiDataType      data_type,
                  void*              p_data,
                  const void*        p_min,
                  const void*        p_max,
                  const char*        format,
                  ImGuiSliderFlags   flags,
                  const std::string& tooltip)
{
  return entry(label, [&] { return ImGui::SliderScalar("##hidden", data_type, p_data, p_min, p_max, format, flags); }, tooltip);
}

bool DragFloat(const char* label, float* v, float v_speed, float v_min, float v_max, const char* format, ImGuiSliderFlags flags, const std::string& tooltip)
{
  return entry(label, [&] { return ImGui::DragFloat("##hidden", v, v_speed, v_min, v_max, format, flags); }, tooltip);
}
bool DragFloat2(const char* label, float v[2], float v_speed, float v_min, float v_max, const char* format, ImGuiSliderFlags flags, const std::string& tooltip)
{
  return entry(label, [&] { return ImGui::DragFloat2("##hidden", v, v_speed, v_min, v_max, format, flags); }, tooltip);
}
bool DragFloat3(const char* label, float v[3], float v_speed, float v_min, float v_max, const char* format, ImGuiSliderFlags flags, const std::string& tooltip)
{
  return entry(label, [&] { return ImGui::DragFloat3("##hidden", v, v_speed, v_min, v_max, format, flags); }, tooltip);
}
bool DragFloat4(const char* label, float v[4], float v_speed, float v_min, float v_max, const char* format, ImGuiSliderFlags flags, const std::string& tooltip)
{
  return entry(label, [&] { return ImGui::DragFloat4("##hidden", v, v_speed, v_min, v_max, format, flags); }, tooltip);
}
bool DragInt(const char* label, int* v, float v_speed, int v_min, int v_max, const char* format, ImGuiSliderFlags flags, const std::string& tooltip)
{
  return entry(label, [&] { return ImGui::DragInt("##hidden", v, v_speed, v_min, v_max, format, flags); }, tooltip);
}
bool DragInt2(const char* label, int v[2], float v_speed, int v_min, int v_max, const char* format, ImGuiSliderFlags flags, const std::string& tooltip)
{
  return entry(label, [&] { return ImGui::DragInt2("##hidden", v, v_speed, v_min, v_max, format, flags); }, tooltip);
}
bool DragInt3(const char* label, int v[3], float v_speed, int v_min, int v_max, const char* format, ImGuiSliderFlags flags, const std::string& tooltip)
{
  return entry(label, [&] { return ImGui::DragInt3("##hidden", v, v_speed, v_min, v_max, format, flags); }, tooltip);
}
bool DragInt4(const char* label, int v[4], float v_speed, int v_min, int v_max, const char* format, ImGuiSliderFlags flags, const std::string& tooltip)
{
  return entry(label, [&] { return ImGui::DragInt4("##hidden", v, v_speed, v_min, v_max, format, flags); }, tooltip);
}
bool DragScalar(const char*        label,
                ImGuiDataType      data_type,
                void*              p_data,
                float              v_speed /*= 1.0f*/,
                const void*        p_min /*= NULL*/,
                const void*        p_max /*= NULL*/,
                const char*        format /*= NULL*/,
                ImGuiSliderFlags   flags /*= 0*/,
                const std::string& tooltip /*= {}*/)
{
  return entry(
      label, [&] { return ImGui::DragScalar("##hidden", data_type, p_data, v_speed, p_min, p_max, format, flags); }, tooltip);
}
bool InputText(const char* label, char* buf, size_t buf_size, ImGuiInputTextFlags flags, const std::string& tooltip)
{
  return entry(label, [&] { return ImGui::InputText("##hidden", buf, buf_size, flags); }, tooltip);
}
bool InputTextMultiline(const char* label, char* buf, size_t buf_size, const ImVec2& size, ImGuiInputTextFlags flags, const std::string& tooltip)
{
  return entry(label, [&] { return ImGui::InputTextMultiline("##hidden", buf, buf_size, size, flags); }, tooltip);
}
bool InputFloat(const char* label, float* v, float step, float step_fast, const char* format, ImGuiInputTextFlags flags, const std::string& tooltip)
{
  float tv      = *v;
  bool  changed = entry(
      label,
      [&] {
        return ImGui::InputFloat("##hidden", &tv, step, step_fast, format, (flags & ~ImGuiInputTextFlags_EnterReturnsTrue));
      },
      tooltip);

  if(changed && (!(flags & ImGuiInputTextFlags_EnterReturnsTrue) || (ImGui::IsItemDeactivatedAfterEdit() || ImGui::IsItemClicked())))
  {
    *v = tv;
    return true;
  }
  else
  {
    return false;
  }
}
bool InputFloat2(const char* label, float v[2], const char* format, ImGuiInputTextFlags flags, const std::string& tooltip)
{
  return entry(label, [&] { return ImGui::InputFloat2("##hidden", v, format, flags); }, tooltip);
}
bool InputFloat3(const char* label, float v[3], const char* format, ImGuiInputTextFlags flags, const std::string& tooltip)
{
  return entry(label, [&] { return ImGui::InputFloat3("##hidden", v, format, flags); }, tooltip);
}
bool InputFloat4(const char* label, float v[4], const char* format, ImGuiInputTextFlags flags, const std::string& tooltip)
{
  return entry(label, [&] { return ImGui::InputFloat4("##hidden", v, format, flags); }, tooltip);
}
bool InputInt(const char* label, int* v, int step, int step_fast, ImGuiInputTextFlags flags, const std::string& tooltip)
{
  int  tv      = *v;
  bool changed = entry(
      label,
      [&] { return ImGui::InputInt("##hidden", &tv, step, step_fast, (flags & ~ImGuiInputTextFlags_EnterReturnsTrue)); }, tooltip);

  if(changed && (!(flags & ImGuiInputTextFlags_EnterReturnsTrue) || (ImGui::IsItemDeactivatedAfterEdit() || ImGui::IsItemClicked())))
  {
    *v = tv;
    return true;
  }
  else
  {
    return false;
  }
}
bool InputIntClamped(const char* label, int* v, int min, int max, int step, int step_fast, ImGuiInputTextFlags flags, const std::string& tooltip)
{
  int  tv      = *v;
  bool changed = entry(
      label,
      [&] { return ImGui::InputInt("##hidden", &tv, step, step_fast, (flags & ~ImGuiInputTextFlags_EnterReturnsTrue)); }, tooltip);

  changed = changed
            && (!(flags & ImGuiInputTextFlags_EnterReturnsTrue) || (ImGui::IsItemDeactivatedAfterEdit() || ImGui::IsItemClicked()));
  if(changed)
  {
    *v = tv;
  }
  return Clamped(changed, v, min, max);
}
bool InputInt2(const char* label, int v[2], ImGuiInputTextFlags flags, const std::string& tooltip)
{
  return entry(label, [&] { return ImGui::InputInt2("##hidden", v, flags); }, tooltip);
}
bool InputInt3(const char* label, int v[3], ImGuiInputTextFlags flags, const std::string& tooltip)
{
  return entry(label, [&] { return ImGui::InputInt3("##hidden", v, flags); }, tooltip);
}
bool InputInt4(const char* label, int v[4], ImGuiInputTextFlags flags, const std::string& tooltip)
{
  return entry(label, [&] { return ImGui::InputInt4("##hidden", v, flags); }, tooltip);
}
bool InputDouble(const char* label, double* v, double step, double step_fast, const char* format, ImGuiInputTextFlags flags, const std::string& tooltip)
{
  return entry(label, [&] { return ImGui::InputDouble("##hidden", v, step, step_fast, format, flags); }, tooltip);
}
bool InputScalar(const char*         label,
                 ImGuiDataType       data_type,
                 void*               p_data,
                 const void*         p_step,
                 const void*         p_step_fast,
                 const char*         format,
                 ImGuiInputTextFlags flags,
                 const std::string&  tooltip)
{
  return entry(
      label, [&] { return ImGui::InputScalar("##hidden", data_type, p_data, p_step, p_step_fast, format, flags); }, tooltip);
}

bool ColorEdit3(const char* label, float col[3], ImGuiColorEditFlags flags, const std::string& tooltip)
{
  return entry(label, [&] { return ImGui::ColorEdit3("##hidden", col, flags); }, tooltip);
}
bool ColorEdit4(const char* label, float col[4], ImGuiColorEditFlags flags, const std::string& tooltip)
{
  return entry(label, [&] { return ImGui::ColorEdit4("##hidden", col, flags); }, tooltip);
}
bool ColorPicker3(const char* label, float col[3], ImGuiColorEditFlags flags, const std::string& tooltip)
{
  return entry(label, [&] { return ImGui::ColorPicker3("##hidden", col, flags); }, tooltip);
}
bool ColorPicker4(const char* label, float col[4], ImGuiColorEditFlags flags, const std::string& tooltip)
{
  return entry(label, [&] { return ImGui::ColorPicker4("##hidden", col, flags); }, tooltip);
}
bool ColorButton(const char* label, const ImVec4& col, ImGuiColorEditFlags flags, const ImVec2& size, const std::string& tooltip)
{
  return entry(label, [&] { return ImGui::ColorButton("##hidden", col, flags, size); }, tooltip);
}
void Text(const char* label, const std::string& text)
{
  entry(label, [&] {
    ImGui::Text("%s", text.c_str());
    return false;  // dummy, no change
  });
}
void Text(const char* label, const char* fmt, ...)
{
  va_list args;
  va_start(args, fmt);
  entry(label, [&] {
    ImGui::TextV(fmt, args);
    return false;  // dummy, no change
  });
  va_end(args);
}

}  // namespace PropertyEditor
}  // namespace nvgui
