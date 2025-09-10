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

// Defines default and monospace fonts.
//
// To switch to the monospace font:
// ```
// ImGui::PushFont(ImGuiH::getMonospaceFont());
// ImGui::Text("%f ms", frameTime);
// ImGui::PopFont();
// ```
//
// The default font includes icons within the Unicode Private Use Area.
// You can use them using the ICON_MS_* definitions, like this:
//
//	std::string buttonLabel = std::string("Login " + ICON_MS_LOGIN);
//	ImGui::Button(buttonLabel.c_str());
//  ImGui::Button(fmt::format("Login {}", ICON_MS_LOGIN).c_str());
//	ImGui::Button(ICON_MS_LOGIN);
//
// The list of all icons can be seen online at https://fonts.google.com/icons.


#pragma once

#include "IconsMaterialSymbols.h"  // ICON_MS definitions

struct ImFont;

namespace nvgui {

void    addDefaultFont(float fontSize = 15.F, bool appendIcons = true);  // Initializes the default font.
ImFont* getDefaultFont();                                                // Returns the default font.
void    addMonospaceFont(float fontSize = 15.F);                         // Initializes the monospace font.
ImFont* getMonospaceFont();                                              // Returns the monospace font

}  // namespace nvgui
