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

#pragma once

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>

/*-------------------------------------------------------------------------------------------------
# class NVPSystem
>  NVPSystem is a utility class to handle some basic system
functionality that all projects likely make use of.
It does not require any window to be opened.
Typical usage is calling init right after main and deinit
in the end, or use the NVPSystem object for that.
init
- calls glfwInit and registers the error callback for it
- sets up and log filename based on projectName via nvprintSetLogFileName
-------------------------------------------------------------------------------------------------*/

class NVPSystem
{
public:
  static void init(const char* projectName);
  static void deinit();

  static void pollEvents(); ///< polls events. Non blicking
  static void waitEvents(); ///< wait for events. Will return when at least 1 event happened

  static double getTime();  ///< returns time in seconds
  static void   sleep(double seconds);

  static std::string exePath(); ///< exePath() can be called without init called before

  static bool isInited();

  /// for sake of debugging/automated testing
  static void windowScreenshot(struct GLFWwindow* glfwin, const char* filename);
  static void windowClear(struct GLFWwindow* glfwin, uint32_t r, uint32_t g, uint32_t b);
  /// \namesimple modal dialog
  /// @{
  static std::filesystem::path windowOpenFileDialog(struct GLFWwindow* glfwin, const char* title, const char* exts);
  static std::filesystem::path windowSaveFileDialog(struct GLFWwindow* glfwin, const char* title, const char* exts);
  /// @}
  NVPSystem(const char* projectName) { init(projectName); }
  ~NVPSystem() { deinit(); }

private:
  static void platformInit();
  static void platformDeinit();
};
