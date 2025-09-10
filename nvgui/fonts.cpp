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

#include <cstdint>
#include <string>

#include <imgui/imgui.h>
#include <roboto/roboto_mono.h>
#include <roboto/roboto_regular.h>

#include <material_symbols/material_symbols_rounded_regular.h>
#define MATERIAL_SYMBOLS_DATA g_materialSymbolsRounded_compressed_data
#define MATERIAL_SYMBOLS_SIZE g_materialSymbolsRounded_compressed_size

#include "fonts.hpp"


namespace nvgui {
ImFont* g_defaultFont   = nullptr;
ImFont* g_iconicFont    = nullptr;
ImFont* g_monospaceFont = nullptr;
}  // namespace nvgui

static ImFontConfig getDefaultConfig()
{
  ImFontConfig config{};
  config.OversampleH = 3;
  config.OversampleV = 3;
  return config;
}

// Helper function to append a font with embedded Material Symbols icons
// Icon fonts: https://fonts.google.com/icons?icon.set=Material+Symbols
static ImFont* appendFontWithMaterialSymbols(const void* fontData, int fontDataSize, float fontSize)
{
  // Configure Material Symbols icon font for merging
  ImFontConfig iconConfig = getDefaultConfig();
  iconConfig.MergeMode    = true;
  iconConfig.PixelSnapH   = true;

  // Material Symbols specific configuration
  float iconFontSize       = 1.28571429f * fontSize;  // Material Symbols work best at 9/7x the base font size
  iconConfig.GlyphOffset.x = iconFontSize * 0.01f;
  iconConfig.GlyphOffset.y = iconFontSize * 0.2f;

  // Define the Material Symbols character range
  static const ImWchar materialSymbolsRange[] = {ICON_MIN_MS, ICON_MAX_MS, 0};

  // Load embedded Material Symbols
  ImFont* font = ImGui::GetIO().Fonts->AddFontFromMemoryCompressedTTF(fontData, fontDataSize, iconFontSize, &iconConfig);

  return font;
}


// Add default Roboto fonts with the option to merge Material Symbols (icons)
void nvgui::addDefaultFont(float fontSize, bool appendIcons)
{
  if(g_defaultFont == nullptr)
  {
    ImFontConfig fontConfig = getDefaultConfig();
    g_defaultFont           = ImGui::GetIO().Fonts->AddFontFromMemoryCompressedTTF(g_roboto_regular_compressed_data,
                                                                                   g_roboto_regular_compressed_size, fontSize, &fontConfig);

    if(appendIcons)  // If appendIcons is true, merge Material Symbols into the default font
    {
      g_defaultFont = appendFontWithMaterialSymbols(MATERIAL_SYMBOLS_DATA, MATERIAL_SYMBOLS_SIZE, fontSize);
    }
  }
}

ImFont* nvgui::getDefaultFont()
{
  return g_defaultFont;
}

void nvgui::addMonospaceFont(float fontSize)
{
  if(g_monospaceFont == nullptr)
  {
    ImFontConfig fontConfig = getDefaultConfig();
    g_monospaceFont         = ImGui::GetIO().Fonts->AddFontFromMemoryCompressedTTF(g_roboto_mono_compressed_data,
                                                                                   g_roboto_mono_compressed_size, fontSize, &fontConfig);
  }
}

ImFont* nvgui::getMonospaceFont()
{
  return g_monospaceFont;
}
