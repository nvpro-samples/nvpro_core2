/*
 * Copyright (c) 2019-2025, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2019-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
//--------------------------------------------------------------------

#ifdef __APPLE__

#import <Cocoa/Cocoa.h>
#import <UniformTypeIdentifiers/UniformTypeIdentifiers.h>

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include <string>
#include <vector>
#include <sstream>

#include "file_dialog.hpp"

// Parse the extension filter string into a list of file extensions
// Format: "Description|*.ext1;*.ext2|Description2|*.ext3"
static std::vector<std::string> parseExtensions(const char* exts)
{
  std::vector<std::string> extensions;
  if (!exts || *exts == '\0') {
    return extensions;
  }

  std::string extStr(exts);
  std::istringstream stream(extStr);
  std::string token;
  int index = 0;

  while (std::getline(stream, token, '|')) {
    // Only process filter strings (odd indices), skip descriptions (even indices)
    if (index % 2 == 1) {
      // Split by semicolon for multiple extensions
      std::istringstream extStream(token);
      std::string ext;
      while (std::getline(extStream, ext, ';')) {
        // Remove leading *. if present
        if (ext.size() > 2 && ext[0] == '*' && ext[1] == '.') {
          ext = ext.substr(2);
        } else if (ext.size() > 1 && ext[0] == '.') {
          ext = ext.substr(1);
        }
        // Skip wildcard
        if (ext != "*" && !ext.empty()) {
          extensions.push_back(ext);
        }
      }
    }
    index++;
  }

  return extensions;
}

// Convert file extensions to UTTypes for macOS file dialogs
static NSArray<UTType*>* extensionsToUTTypes(const std::vector<std::string>& extensions)
{
  if (extensions.empty()) {
    return nil;
  }

  NSMutableArray<UTType*>* types = [NSMutableArray array];

  for (const auto& ext : extensions) {
    NSString* nsExt = [NSString stringWithUTF8String:ext.c_str()];
    UTType* type = [UTType typeWithFilenameExtension:nsExt];
    if (type) {
      [types addObject:type];
    }
  }

  return types.count > 0 ? types : nil;
}

std::filesystem::path nvgui::windowOpenFileDialog(struct GLFWwindow* glfwin, const char* title, const char* exts)
{
  std::filesystem::path initialDir;
  return windowOpenFileDialog(glfwin, title, exts, initialDir);
}

std::filesystem::path nvgui::windowOpenFileDialog(struct GLFWwindow* glfwin, const char* title, const char* exts, std::filesystem::path& initialDir)
{
  @autoreleasepool {
    NSOpenPanel* panel = [NSOpenPanel openPanel];

    [panel setTitle:[NSString stringWithUTF8String:title]];
    [panel setCanChooseFiles:YES];
    [panel setCanChooseDirectories:NO];
    [panel setAllowsMultipleSelection:NO];

    // Set initial directory if provided
    if (!initialDir.empty()) {
      NSString* dirPath = [NSString stringWithUTF8String:initialDir.c_str()];
      [panel setDirectoryURL:[NSURL fileURLWithPath:dirPath]];
    }

    // Set allowed file types
    std::vector<std::string> extensions = parseExtensions(exts);
    NSArray<UTType*>* allowedTypes = extensionsToUTTypes(extensions);
    if (allowedTypes) {
      [panel setAllowedContentTypes:allowedTypes];
    }

    if ([panel runModal] == NSModalResponseOK) {
      NSURL* url = [[panel URLs] firstObject];
      if (url) {
        std::filesystem::path result([[url path] UTF8String]);
        // Update initial directory
        initialDir = result.parent_path();
        return result;
      }
    }

    return std::filesystem::path();
  }
}

std::filesystem::path nvgui::windowSaveFileDialog(struct GLFWwindow* glfwin, const char* title, const char* exts)
{
  @autoreleasepool {
    NSSavePanel* panel = [NSSavePanel savePanel];

    [panel setTitle:[NSString stringWithUTF8String:title]];
    [panel setCanCreateDirectories:YES];

    // Set allowed file types
    std::vector<std::string> extensions = parseExtensions(exts);
    NSArray<UTType*>* allowedTypes = extensionsToUTTypes(extensions);
    if (allowedTypes) {
      [panel setAllowedContentTypes:allowedTypes];
    }

    if ([panel runModal] == NSModalResponseOK) {
      NSURL* url = [panel URL];
      if (url) {
        return std::filesystem::path([[url path] UTF8String]);
      }
    }

    return std::filesystem::path();
  }
}

std::filesystem::path nvgui::windowOpenFolderDialog(struct GLFWwindow* glfwin, const char* title)
{
  @autoreleasepool {
    NSOpenPanel* panel = [NSOpenPanel openPanel];

    [panel setTitle:[NSString stringWithUTF8String:title]];
    [panel setCanChooseFiles:NO];
    [panel setCanChooseDirectories:YES];
    [panel setAllowsMultipleSelection:NO];

    if ([panel runModal] == NSModalResponseOK) {
      NSURL* url = [[panel URLs] firstObject];
      if (url) {
        return std::filesystem::path([[url path] UTF8String]);
      }
    }

    return std::filesystem::path();
  }
}

#endif // __APPLE__
