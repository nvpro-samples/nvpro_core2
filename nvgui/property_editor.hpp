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

#pragma once
#include <functional>
#include <string>

#include <imgui/imgui.h>


// clang-format off
namespace nvgui {

//--------------------------------------------------------------------------------------------------
// This is a helper to create a nice property editor with ImGui, where the name of the property
// is on the left, while all values are on the right.
//
// To use:
// - Call PropertyEditor::begin() to start the section of the editor and PropertyEditor::end() to
//   close it.
// - Use the same as with ImGui, but use the namespace ImGuiH::PropertyEditor instead
// - For special cases, use the ImGui property in the lambda function
//   Ex: PropertyEditor::entry("My Prop", [&](){return ImGui::DragFloat(...);});
// - Note, each function has an extra argument, which will display the string as Tooltip
//
namespace PropertyEditor {
bool begin(const char* label = "PE::Table", ImGuiTableFlags flag = ImGuiTableFlags_BordersOuter | ImGuiTableFlags_Resizable);
void end();
bool entry(const std::string& property_name, const std::function<bool()>& content_fct, const std::string& tooltip = {});
void entry(const std::string& property_name, const std::string& value);
bool treeNode(const std::string& name, ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_SpanFullWidth);
void treePop();


bool Button(const char* label, const ImVec2& size = ImVec2(0, 0), const std::string& tooltip = {});
bool SmallButton(const char* label, const std::string& tooltip = {});
bool Checkbox(const char* label, bool* v, const std::string& tooltip = {});
bool RadioButton(const char* label, bool active, const std::string& tooltip = {});
bool RadioButton(const char* label, int* v, int v_button, const std::string& tooltip = {});
bool Combo(const char* label, int* current_item, const char* const items[], int items_count, int popup_max_height_in_items = -1, const std::string& tooltip = {});
bool Combo(const char* label, int* current_item, const char* items_separated_by_zeros, int popup_max_height_in_items = -1, const std::string& tooltip = {});
bool Combo(const char* label, int* current_item, const char* (*getter)(void* user_data, int idx), void* user_data, int items_count, int popup_max_height_in_items = -1, const std::string& tooltip = {});
bool SliderFloat(const char* label, float* v, float v_min, float v_max, const char* format = "%.3f", ImGuiSliderFlags flags = 0, const std::string& tooltip = {});
bool SliderFloat2(const char* label, float v[2], float v_min, float v_max, const char* format = "%.3f", ImGuiSliderFlags flags = 0, const std::string& tooltip = {});
bool SliderFloat3(const char* label, float v[3], float v_min, float v_max, const char* format = "%.3f", ImGuiSliderFlags flags = 0, const std::string& tooltip = {});
bool SliderFloat4(const char* label, float v[4], float v_min, float v_max, const char* format = "%.3f", ImGuiSliderFlags flags = 0, const std::string& tooltip = {});
bool SliderAngle(const char* label, float* v_rad, float v_degrees_min = -360.0f, float v_degrees_max = +360.0f, const char* format = "%.0f deg", ImGuiSliderFlags flags = 0, const std::string& tooltip = {});
bool SliderInt(const char* label, int* v, int v_min, int v_max, const char* format = "%d", ImGuiSliderFlags flags = 0, const std::string& tooltip = {});
bool SliderInt2(const char* label, int v[2], int v_min, int v_max, const char* format = "%d", ImGuiSliderFlags flags = 0, const std::string& tooltip = {});
bool SliderInt3(const char* label, int v[3], int v_min, int v_max, const char* format = "%d", ImGuiSliderFlags flags = 0, const std::string& tooltip = {});
bool SliderInt4(const char* label, int v[4], int v_min, int v_max, const char* format = "%d", ImGuiSliderFlags flags = 0, const std::string& tooltip = {});
bool SliderScalar(const char* label, ImGuiDataType data_type, void* p_data, const void* p_min, const void* p_max, const char* format = NULL, ImGuiSliderFlags flags = 0, const std::string& tooltip = {});
bool DragFloat(const char* label, float* v, float v_speed = 1.0f, float v_min = 0.0f, float v_max = 0.0f, const char* format = "%.3f", ImGuiSliderFlags flags = 0, const std::string& tooltip = {});
bool DragFloat2(const char* label, float v[2], float v_speed = 1.0f, float v_min = 0.0f, float v_max = 0.0f, const char* format = "%.3f", ImGuiSliderFlags flags = 0, const std::string& tooltip = {});
bool DragFloat3(const char* label, float v[3], float v_speed = 1.0f, float v_min = 0.0f, float v_max = 0.0f, const char* format = "%.3f", ImGuiSliderFlags flags = 0, const std::string& tooltip = {});
bool DragFloat4(const char* label, float v[4], float v_speed = 1.0f, float v_min = 0.0f, float v_max = 0.0f, const char* format = "%.3f", ImGuiSliderFlags flags = 0, const std::string& tooltip = {});
bool DragInt(const char* label, int* v, float v_speed = 1.0f, int v_min = 0, int v_max = 0, const char* format = "%d", ImGuiSliderFlags flags = 0, const std::string& tooltip = {});
bool DragInt2(const char* label, int v[2], float v_speed = 1.0f, int v_min = 0, int v_max = 0, const char* format = "%d", ImGuiSliderFlags flags = 0, const std::string& tooltip = {});
bool DragInt3(const char* label, int v[3], float v_speed = 1.0f, int v_min = 0, int v_max = 0, const char* format = "%d", ImGuiSliderFlags flags = 0, const std::string& tooltip = {});
bool DragInt4(const char* label, int v[4], float v_speed = 1.0f, int v_min = 0, int v_max = 0, const char* format = "%d", ImGuiSliderFlags flags = 0, const std::string& tooltip = {});
bool DragScalar(const char* label, ImGuiDataType data_type, void* p_data, float v_speed = 1.0f, const void* p_min = NULL, const void* p_max = NULL, const char* format = NULL, ImGuiSliderFlags flags = 0, const std::string& tooltip = {});
bool InputText(const char* label, char* buf, size_t buf_size, ImGuiInputTextFlags flags = 0, const std::string& tooltip = {});
bool InputTextMultiline(const char* label, char* buf, size_t buf_size, const ImVec2& size = ImVec2(0, 0), ImGuiInputTextFlags flags = 0, const std::string& tooltip = {});
// does support workaround for ImGuiInputTextFlags_EnterReturnsTrue
bool InputFloat(const char* label, float* v, float step = 0.0f, float step_fast = 0.0f, const char* format = "%.3f", ImGuiInputTextFlags flags = 0, const std::string& tooltip = {});
bool InputFloat2(const char* label, float v[2], const char* format = "%.3f", ImGuiInputTextFlags flags = 0, const std::string& tooltip = {});
bool InputFloat3(const char* label, float v[3], const char* format = "%.3f", ImGuiInputTextFlags flags = 0, const std::string& tooltip = {});
bool InputFloat4(const char* label, float v[4], const char* format = "%.3f", ImGuiInputTextFlags flags = 0, const std::string& tooltip = {});
// does support workaround for ImGuiInputTextFlags_EnterReturnsTrue
bool InputInt(const char* label, int* v, int step = 1, int step_fast = 100, ImGuiInputTextFlags flags = 0, const std::string& tooltip = {});
// does support workaround for ImGuiInputTextFlags_EnterReturnsTrue
bool InputIntClamped(const char* label, int* v, int min, int max, int step, int step_fast, ImGuiInputTextFlags flags = 0, const std::string& tooltip={});
bool InputInt2(const char* label, int v[2], ImGuiInputTextFlags flags = 0, const std::string& tooltip = {});
bool InputInt3(const char* label, int v[3], ImGuiInputTextFlags flags = 0, const std::string& tooltip = {});
bool InputInt4(const char* label, int v[4], ImGuiInputTextFlags flags = 0, const std::string& tooltip = {});
bool InputDouble(const char* label, double* v, double step = 0.0, double step_fast = 0.0, const char* format = "%.6f", ImGuiInputTextFlags flags = 0, const std::string& tooltip = {});
bool InputScalar(const char* label, ImGuiDataType data_type, void* p_data, const void* p_step = NULL, const void* p_step_fast = NULL, const char* format = NULL, ImGuiInputTextFlags flags = 0, const std::string& tooltip = {});
bool ColorEdit3(const char* label, float col[3], ImGuiColorEditFlags flags = 0, const std::string& tooltip = {});
bool ColorEdit4(const char* label, float col[4], ImGuiColorEditFlags flags = 0, const std::string& tooltip = {});
bool ColorPicker3(const char* label, float col[3], ImGuiColorEditFlags flags = 0, const std::string& tooltip = {});
bool ColorPicker4(const char* label, float col[4], ImGuiColorEditFlags flags = 0, const std::string& tooltip = {});
bool ColorButton(const char* label, const ImVec4& col, ImGuiColorEditFlags flags = 0, const ImVec2& size = ImVec2(0, 0), const std::string& tooltip = {});
void Text(const char* label, const std::string& text);
void Text(const char* label, const char* fmt, ...);
}  // namespace PropertyEditor
}  // namespace ImGui
// clang-format on
