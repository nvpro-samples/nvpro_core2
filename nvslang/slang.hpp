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
#include <array>
#include <vector>
#include <string>
#include <filesystem>

#pragma push_macro("None")
#pragma push_macro("Bool")
#undef None
#undef Bool
#include <slang.h>
#pragma pop_macro("None")
#pragma pop_macro("Bool")

#include <slang-com-ptr.h>
#include <nvutils/logger.hpp>
#include <nvutils/file_operations.hpp>
#include "nvutils/spirv.hpp"

//--------------------------------------------------------------------------------------------------
// Slang Compiler
//
// Usage:
//   see usage_SlangCompiler
//--------------------------------------------------------------------------------------------------

namespace nvslang {

// A class responsible for compiling Slang source code.
class SlangCompiler
{
public:
  // Initializes the SlangCompiler.
  //
  // Set `enableGLSL` to `true` to enable the Slang compatibility module (which is loaded when a Slang file includes a `#version` directive).
  // If enabled, you will also need to add `FILES ${Slang_GLSL_MODULE}` to your CMake `copy_to_runtime_and_install` call.
  SlangCompiler(bool enableGLSL = false);
  ~SlangCompiler() = default;

  void defaultTarget();   // Default target is SPIR-V
  void defaultOptions();  // Default options are EmitSpirvDirectly, VulkanUseEntryPointName

  void addOption(const slang::CompilerOptionEntry& option) { m_options.push_back(option); }
  void clearOptions() { m_options.clear(); }
  std::vector<slang::CompilerOptionEntry>& options() { return m_options; }

  void                            addTarget(const slang::TargetDesc& target) { m_targets.push_back(target); }
  void                            clearTargets() { m_targets.clear(); }
  std::vector<slang::TargetDesc>& targets() { return m_targets; }

  void addSearchPaths(const std::vector<std::filesystem::path>& searchPaths);
  void clearSearchPaths();
  // This is const because modifiying the search paths requires extra work.
  const std::vector<std::filesystem::path>& searchPaths() const { return m_searchPaths; }

  void addMacro(const slang::PreprocessorMacroDesc& macro) { m_macros.push_back(macro); }
  void clearMacros() { m_macros.clear(); }
  std::vector<slang::PreprocessorMacroDesc>& macros() { return m_macros; }

  // Compile a file or source
  bool compileFile(const std::filesystem::path& filename);
  bool loadFromSourceString(const std::string& moduleName, const std::string& slangSource);

  // Get result of the compilation
  const uint32_t* getSpirv() const;
  // Get the number of bytes in the compiled SPIR-V.
  size_t getSpirvSize() const;
  // Gets the linked Slang program; does not add a reference to it.
  // This is usually what you want for reflection.
  slang::IComponentType* getSlangProgram() const;
  // Gets the Slang module; does not add a reference to it. This is usually
  // useful for reflection if you need a list of functions.
  slang::IModule* getSlangModule() const;

  // Use for Dump or Aftermath
  void setCompileCallback(std::function<void(const std::filesystem::path& sourceFile, const uint32_t* spirvCode, size_t spirvSize)> callback)
  {
    m_callback = callback;
  }

  // Get the last diagnostic message (error or warning).
  // Multiple diagnostics are each separated by a single newline.
  const std::string& getLastDiagnosticMessage() const { return m_lastDiagnosticMessage; }

private:
  void createSession();
  void logAndAppendDiagnostics(slang::IBlob* diagnostic);

  Slang::ComPtr<slang::IGlobalSession>      m_globalSession;
  std::vector<slang::TargetDesc>            m_targets;
  std::vector<slang::CompilerOptionEntry>   m_options;
  std::vector<std::filesystem::path>        m_searchPaths;
  std::vector<std::string>                  m_searchPathsUtf8;
  std::vector<const char*>                  m_searchPathsUtf8Pointers;
  Slang::ComPtr<slang::ISession>            m_session;
  Slang::ComPtr<slang::IModule>             m_module;
  Slang::ComPtr<slang::IComponentType>      m_linkedProgram;
  Slang::ComPtr<ISlangBlob>                 m_spirv;
  std::vector<slang::PreprocessorMacroDesc> m_macros;

  std::function<void(const std::filesystem::path& sourceFile, const uint32_t* spirvCode, size_t spirvSize)> m_callback;

  // Store the last diagnostic message
  std::string m_lastDiagnosticMessage;
};

}  // namespace nvslang