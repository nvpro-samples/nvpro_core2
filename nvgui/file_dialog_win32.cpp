/*
 * Copyright (c) 2022-2025, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifdef _WIN32

#include <cassert>
#include <filesystem>

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3native.h>

#include <nvutils/file_operations.hpp>

#include "file_dialog.hpp"


static std::filesystem::path fileDialog(struct GLFWwindow* glfwin, std::wstring title, std::wstring exts, bool openToLoad)
{
  if(!glfwin)
  {
    assert(!"Attempted to call fileDialog() on null window!");
    return std::filesystem::path();
  }
  HWND hwnd = glfwGetWin32Window(glfwin);

  std::vector<wchar_t> extsfixed;
  for(size_t i = 0; i < exts.size(); i++)
  {
    if(exts[i] == L'|')
    {
      extsfixed.push_back(0);
    }
    else
    {
      extsfixed.push_back(exts[i]);
    }
  }
  extsfixed.push_back(0);
  extsfixed.push_back(0);

  OPENFILENAMEW ofn;             // common dialog box structure
  wchar_t       szFile[1024]{};  // buffer for file name

  // Initialize OPENFILENAME
  ZeroMemory(&ofn, sizeof(ofn));
  ofn.lStructSize     = sizeof(ofn);
  ofn.hwndOwner       = hwnd;
  ofn.lpstrFile       = szFile;
  ofn.lpstrFile[0]    = L'\0';
  ofn.nMaxFile        = sizeof(szFile) / sizeof(wchar_t);
  ofn.lpstrFilter     = extsfixed.data();
  ofn.nFilterIndex    = 1;
  ofn.lpstrFileTitle  = NULL;
  ofn.nMaxFileTitle   = 0;
  ofn.lpstrInitialDir = NULL;
  ofn.Flags           = OFN_PATHMUSTEXIST | OFN_NOCHANGEDIR;
  ofn.lpstrTitle      = title.c_str();

  if(openToLoad)
  {
    ofn.Flags |= OFN_FILEMUSTEXIST;
    if(GetOpenFileNameW(&ofn) == TRUE)
    {
      return std::filesystem::path(ofn.lpstrFile);
    }
  }
  else
  {
    ofn.Flags |= OFN_OVERWRITEPROMPT;
    if(GetSaveFileNameW(&ofn) == TRUE)
    {
      return std::filesystem::path(ofn.lpstrFile);
    }
  }

  return std::filesystem::path();
}

std::filesystem::path nvgui::windowOpenFileDialog(struct GLFWwindow* glfwin, const char* title, const char* exts)
{
  return fileDialog(glfwin, nvutils::pathFromUtf8(title).native(), nvutils::pathFromUtf8(exts).native(), true);
}

std::filesystem::path nvgui::windowSaveFileDialog(struct GLFWwindow* glfwin, const char* title, const char* exts)
{
  return fileDialog(glfwin, nvutils::pathFromUtf8(title).native(), nvutils::pathFromUtf8(exts).native(), false);
}

#endif