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
#include <shlobj.h>
#include <wrl.h>  // Microsoft::WRL::ComPtr

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3native.h>

#include <nvutils/file_operations.hpp>

#include "file_dialog.hpp"

// Unified dialog mode enum (should match header)
enum class DialogMode
{
  OpenFile,
  SaveFile,
  OpenFolder
};

static std::filesystem::path unifiedDialog(struct GLFWwindow*           glfwin,
                                           std::wstring                 title,
                                           std::wstring                 exts,
                                           DialogMode                   mode,
                                           const std::filesystem::path& initialDir = {})
{
  if(!glfwin)
  {
    assert(!"Attempted to call dialog() on null window!");
    return {};
  }
  HWND hwnd = glfwGetWin32Window(glfwin);

  // Initialize COM for this thread if not already
  HRESULT hr         = CoInitializeEx(NULL, COINIT_APARTMENTTHREADED | COINIT_DISABLE_OLE1DDE);
  bool    needUninit = SUCCEEDED(hr);

  std::filesystem::path result;

  Microsoft::WRL::ComPtr<IFileDialog> pfd;
  if(mode == DialogMode::SaveFile)
    hr = CoCreateInstance(CLSID_FileSaveDialog, NULL, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&pfd));
  else
    hr = CoCreateInstance(CLSID_FileOpenDialog, NULL, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&pfd));

  if(SUCCEEDED(hr))
  {
    DWORD dwOptions;
    hr = pfd->GetOptions(&dwOptions);
    if(SUCCEEDED(hr))
    {
      if(mode == DialogMode::OpenFolder)
        pfd->SetOptions(dwOptions | FOS_PICKFOLDERS | FOS_FORCEFILESYSTEM);
      else
        pfd->SetOptions(dwOptions | FOS_FORCEFILESYSTEM);

      if(!title.empty())
        pfd->SetTitle(title.c_str());

      // Set file type filters for file dialogs
      if(mode != DialogMode::OpenFolder && !exts.empty())
      {
        // exts format: "Text Files|*.txt|All Files|*.*"
        std::vector<COMDLG_FILTERSPEC> filters;
        std::vector<std::wstring>      filterNames;
        std::vector<std::wstring>      filterSpecs;
        size_t                         start = 0;
        while(start < exts.size())
        {
          size_t sep = exts.find(L'|', start);
          if(sep == std::wstring::npos)
            break;
          std::wstring name = exts.substr(start, sep - start);
          start             = sep + 1;
          sep               = exts.find(L'|', start);
          std::wstring spec = (sep == std::wstring::npos) ? exts.substr(start) : exts.substr(start, sep - start);
          start             = (sep == std::wstring::npos) ? exts.size() : sep + 1;
          filterNames.push_back(name);
          filterSpecs.push_back(spec);
        }
        for(size_t i = 0; i < filterNames.size(); ++i)
        {
          filters.push_back({filterNames[i].c_str(), filterSpecs[i].c_str()});
        }
        if(!filters.empty())
          pfd->SetFileTypes((UINT)filters.size(), filters.data());
      }

      // Set initial directory if provided
      if(!initialDir.empty() && std::filesystem::exists(initialDir))
      {
        Microsoft::WRL::ComPtr<IShellItem> pFolder;
        hr = SHCreateItemFromParsingName(initialDir.c_str(), nullptr, IID_PPV_ARGS(&pFolder));
        if(SUCCEEDED(hr))
        {
          pfd->SetFolder(pFolder.Get());
        }
      }

      // Show the dialog
      hr = pfd->Show(hwnd);
      if(SUCCEEDED(hr))
      {
        Microsoft::WRL::ComPtr<IShellItem> pItem;
        hr = pfd->GetResult(&pItem);
        if(SUCCEEDED(hr))
        {
          PWSTR pszFilePath = nullptr;
          hr                = pItem->GetDisplayName(SIGDN_FILESYSPATH, &pszFilePath);
          if(SUCCEEDED(hr))
          {
            result = std::filesystem::path(pszFilePath);
            CoTaskMemFree(pszFilePath);
          }
        }
      }
    }
  }

  if(needUninit)
    CoUninitialize();

  return result;
}


std::filesystem::path nvgui::windowOpenFileDialog(struct GLFWwindow* glfwin, const char* title, const char* exts)
{
  return unifiedDialog(glfwin, nvutils::pathFromUtf8(title).native(), nvutils::pathFromUtf8(exts).native(), DialogMode::OpenFile);
}

std::filesystem::path nvgui::windowOpenFileDialog(struct GLFWwindow* glfwin, const char* title, const char* exts, std::filesystem::path& initialDir)
{
  std::filesystem::path result = unifiedDialog(glfwin, nvutils::pathFromUtf8(title).native(),
                                               nvutils::pathFromUtf8(exts).native(), DialogMode::OpenFile, initialDir);
  // Update the initial directory to the directory of the selected file
  if(!result.empty())
  {
    initialDir = result.parent_path();
  }
  return result;
}

std::filesystem::path nvgui::windowSaveFileDialog(struct GLFWwindow* glfwin, const char* title, const char* exts)
{
  return unifiedDialog(glfwin, nvutils::pathFromUtf8(title).native(), nvutils::pathFromUtf8(exts).native(), DialogMode::SaveFile);
}

std::filesystem::path nvgui::windowOpenFolderDialog(struct GLFWwindow* glfwin, const char* title)
{
  return unifiedDialog(glfwin, nvutils::pathFromUtf8(title), {}, DialogMode::OpenFolder);
}

#endif