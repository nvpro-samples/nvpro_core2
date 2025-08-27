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
#include <open_iconic/open_iconic.h>
#include <roboto/roboto_mono.h>
#include <roboto/roboto_regular.h>
#include <material_symbols/material_symbols_rounded_filled_regular.h>

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
  float iconFontSize       = 1.28571429f * fontSize;  // Material Symbols work best at ~1.29x the base font size
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
      g_defaultFont = appendFontWithMaterialSymbols(g_materialSymbolsRounded_filled_compressed_data,
                                                    g_materialSymbolsRounded_filled_compressed_size, fontSize);
    }
  }
}

ImFont* nvgui::getDefaultFont()
{
  return g_defaultFont;
}

void nvgui::addIconicFont(float fontSize)
{
  if(g_iconicFont == nullptr)
  {
    ImFontConfig          fontConfig = getDefaultConfig();
    static uint16_t const range[]    = {0xE000, 0xE0DF, 0};  // Up to 0xE0 characters in the Private Use Area
    g_iconicFont = ImGui::GetIO().Fonts->AddFontFromMemoryCompressedTTF(g_open_iconic_compressed_data, open_iconic_compressed_size,
                                                                        fontSize, &fontConfig, (const ImWchar*)range);
  }
}

ImFont* nvgui::getIconicFont()
{
  return g_iconicFont;
}

void nvgui::showDemoIcons()
{
  // clang-format off
   static const char* text_icon_[] = {
    "account_login", "account_logout", "action_redo", "action_undo", "align_center",
    "align_left","align_right","aperture","arrow_bottom","arrow_circle_bottom","arrow_circle_left",
    "arrow_circle_right","arrow_circle_top","arrow_left","arrow_right","arrow_thick_bottom","arrow_thick_left",
    "arrow_thick_right","arrow_thick_top","arrow_top","audio","audio_spectrum","badge","ban","bar_chart","basket",
    "battery_empty","battery_full","beaker","bell","bluetooth","bold","bolt","book","bookmark","box","briefcase",
    "british_pound","browser","brush","bug","bullhorn","calculator","calendar","camera_slr","caret_bottom",
    "caret_left","caret_right","caret_top","cart","chat","check","chevron_bottom","chevron_left","chevron_right",
    "chevron_top","circle_check","circle_x","clipboard","clock","cloud","cloud_download","cloud_upload","cloudy",
    "code","cog","collapse_down","collapse_left","collapse_right","collapse_up","command","comment_square",
    "compass","contrast","copywriting","credit_card","crop","dashboard","data_transfer_download","data_transfer_upload",
    "delete","dial","document","dollar","double_quote_sans_left","double_quote_sans_right","double_quote_serif_left",
    "double_quote_serif_right","droplet","eject","elevator","ellipses","envelope_closed","envelope_open","euro",
    "excerpt", "expend_down", "expend_left", "expend_right", "expend_up", "external_link", "eye", "eyedropper", "file", "fire", "flag", "flash",
    "folder", "fork", "fullscreen_enter", "fullscreen_exit", "globe", "graph", "grid_four_up", "grid_three_up", "grid_two_up", "hard_drive", "header",
    "headphones", "heart", "home", "image", "inbox", "infinity", "info", "italic", "justify_center", "justify_left", "justify_right",
    "key", "laptop", "layers", "lightbulb", "link_broken", "link_intact", "list", "list_rich", "location", "lock_locked", "lock_unlocked", "loop_circular",
    "loop_square", "loop", "magnifying_glass",
    "map", "map_marquer", "media_pause", "media_play", "media_record", "media_skip_backward", "media_skip_forward", "media_step_backward", "media_step_forward",
    "media_stop", "medical_cross", "menu", "microphone", "minus", "monitor", "moon", "move", "musical_note", "paperclip",
    "pencil", "people", "person", "phone", "pie_chart", "pin", "play_circle", "plus", "power_standby", "print", "project", "pulse", "puzzle_piece",
    "question_mark", "rain", "random", "reload", "resize_both", "resize_height",
    "resize_width", "rss", "rss_alt", "script", "share", "share_boxed", "shield", "signal", "signpost", "sort_ascending", "sort_descending", "spreadsheet",
    "star", "sun", "tablet", "tag", "tags", "target", "task", "terminal",
    "text", "thumb_down", "thumb_up", "timer", "transfer", "trash", "underline", "vertical_align_bottom", "vertical_align_center", "vertical_align_top", "video",
    "volume_high", "volume_low", "volume_off", "warning", "wifi", "wrench", "x", "yen", "zoom_in", "zoom_out"
    };
  // clang-format on

  ImGui::SetNextWindowSize(ImVec2(700, 500), ImGuiCond_FirstUseEver);
  if(!ImGui::Begin("Icons"))
  {
    ImGui::End();
    return;
  }

  // From 0xE000 to 0xE0DF
  for(int i = 0; i < 223; i++)
  {
    std::string utf8String;
    int         codePoint = i + 0xE000;
    utf8String += static_cast<char>(0xE0 | (codePoint >> 12));
    utf8String += static_cast<char>(0x80 | ((codePoint >> 6) & 0x3F));
    utf8String += static_cast<char>(0x80 | (codePoint & 0x3F));

    ImGui::PushFont(nvgui::getIconicFont());
    ImGui::Text("%s", utf8String.c_str());  // Show icon
    if(((i + 1) % 20) != 0)
      ImGui::SameLine();
    ImGui::PopFont();
    ImGui::SetItemTooltip("%s", text_icon_[i]);
  }
  ImGui::End();
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
