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

#pragma once

#include <string>
#include <filesystem>

// The API

struct GLFWwindow;

namespace nvgui {

// opens a file chooser dialog and returns the path to the selected file
std::filesystem::path windowOpenFileDialog(GLFWwindow* glfwin, const char* title, const char* exts);
// opens a file chooser dialog with an initial directory and returns the path to the selected file, and initialDir is updated to the directory of the selected file
std::filesystem::path windowOpenFileDialog(GLFWwindow* glfwin, const char* title, const char* exts, std::filesystem::path& initialDir);
// opens a file save dialog and returns the path to the saved file
std::filesystem::path windowSaveFileDialog(GLFWwindow* glfwin, const char* title, const char* exts);
// opens a folder chooser dialog and returns the path to the selected directory
std::filesystem::path windowOpenFolderDialog(struct GLFWwindow* glfwin, const char* title);

};  // namespace nvgui
