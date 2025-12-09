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

#include "slang.hpp"


nvslang::SlangCompiler::SlangCompiler(bool enableGLSL)
{
  SlangGlobalSessionDesc desc{.enableGLSL = enableGLSL};
  slang::createGlobalSession(&desc, m_globalSession.writeRef());
}

void nvslang::SlangCompiler::defaultTarget()
{
  m_targets.push_back({
      .format                      = SLANG_SPIRV,
      .profile                     = m_globalSession->findProfile("spirv_1_6+vulkan_1_4"),
      .flags                       = SLANG_TARGET_FLAG_GENERATE_SPIRV_DIRECTLY,
      .forceGLSLScalarBufferLayout = true,
  });
}

void nvslang::SlangCompiler::defaultOptions()
{
  m_options.push_back({slang::CompilerOptionName::EmitSpirvDirectly, {slang::CompilerOptionValueKind::Int, 1}});
  m_options.push_back({slang::CompilerOptionName::VulkanUseEntryPointName, {slang::CompilerOptionValueKind::Int, 1}});
  // m_options.push_back({slang::CompilerOptionName::AllowGLSL, {slang::CompilerOptionValueKind::Int, 1}});
}

void nvslang::SlangCompiler::addSearchPaths(const std::vector<std::filesystem::path>& searchPaths)
{
  for(auto& str : searchPaths)
  {
    m_searchPaths.push_back(str);                             // For nvutils::findFile()
    m_searchPathsUtf8.push_back(nvutils::utf8FromPath(str));  // Need to keep the UTF-8 allocation alive
    // Slang expects const char* to UTF-8; see implementation of Slang's FileStream::_init().
    m_searchPathsUtf8Pointers.push_back(m_searchPathsUtf8.back().c_str());
  }
}

void nvslang::SlangCompiler::clearSearchPaths()
{
  m_searchPaths.clear();
  m_searchPathsUtf8.clear();
  m_searchPathsUtf8Pointers.clear();
}

const uint32_t* nvslang::SlangCompiler::getSpirv() const
{
  if(!m_spirv)
  {
    return nullptr;
  }
  return reinterpret_cast<const uint32_t*>(m_spirv->getBufferPointer());
}

size_t nvslang::SlangCompiler::getSpirvSize() const
{
  if(!m_spirv)
  {
    return 0;
  }
  return m_spirv->getBufferSize();
}

slang::IComponentType* nvslang::SlangCompiler::getSlangProgram() const
{
  if(!m_linkedProgram)
  {
    return nullptr;
  }
  return m_linkedProgram.get();
}

slang::IModule* nvslang::SlangCompiler::getSlangModule() const
{
  if(!m_module)
  {
    return nullptr;
  }
  return m_module.get();
}

bool nvslang::SlangCompiler::compileFile(const std::filesystem::path& filename)
{
  const std::filesystem::path sourceFile = nvutils::findFile(filename, m_searchPaths);
  if(sourceFile.empty())
  {
    m_lastDiagnosticMessage = "File not found: " + nvutils::utf8FromPath(filename);
    LOGW("%s\n", m_lastDiagnosticMessage.c_str());
    return false;
  }
  bool success = loadFromSourceString(nvutils::utf8FromPath(sourceFile.stem()), nvutils::loadFile(sourceFile));
  if(success)
  {
    if(m_callback)
    {
      m_callback(sourceFile, getSpirv(), getSpirvSize());
    }
  }

  return success;
}

void nvslang::SlangCompiler::logAndAppendDiagnostics(slang::IBlob* diagnostics)
{
  if(diagnostics)
  {
    const char* message = reinterpret_cast<const char*>(diagnostics->getBufferPointer());
    // Since these are often multi-line, we want to print them with extra spaces:
    LOGW("\n%s\n", message);
    // Append onto m_lastDiagnosticMessage, separated by a newline:
    if(m_lastDiagnosticMessage.empty())
    {
      m_lastDiagnosticMessage += '\n';
    }
    m_lastDiagnosticMessage += message;
  }
}

bool nvslang::SlangCompiler::loadFromSourceString(const std::string& moduleName, const std::string& slangSource)
{
  createSession();

  // Clear any previous compilation
  m_spirv = nullptr;
  m_lastDiagnosticMessage.clear();

  Slang::ComPtr<slang::IBlob> diagnostics;
  // From source code to Slang module
  m_module = m_session->loadModuleFromSourceString(moduleName.c_str(), nullptr, slangSource.c_str(), diagnostics.writeRef());
  logAndAppendDiagnostics(diagnostics);
  if(!m_module)
  {
    return false;
  }

  // In order to get entrypoint shader reflection, it seems like one must go
  // through the additional step of listing every entry point in the composite
  // type. This matches the docs, but @nbickford wonders if there's a simpler way.
  const SlangInt32                               definedEntryPointCount = m_module->getDefinedEntryPointCount();
  std::vector<Slang::ComPtr<slang::IEntryPoint>> entryPoints(definedEntryPointCount);
  std::vector<slang::IComponentType*>            components(1 + definedEntryPointCount);
  components[0] = m_module;
  for(SlangInt32 i = 0; i < definedEntryPointCount; i++)
  {
    m_module->getDefinedEntryPoint(i, entryPoints[i].writeRef());
    components[1 + i] = entryPoints[i];
  }

  Slang::ComPtr<slang::IComponentType> composedProgram;
  SlangResult result = m_session->createCompositeComponentType(components.data(), components.size(),
                                                               composedProgram.writeRef(), diagnostics.writeRef());
  logAndAppendDiagnostics(diagnostics);
  if(SLANG_FAILED(result) || !composedProgram)
  {
    return false;
  }

  // From composite component type to linked program
  result = composedProgram->link(m_linkedProgram.writeRef(), diagnostics.writeRef());
  logAndAppendDiagnostics(diagnostics);
  if(SLANG_FAILED(result) || !m_linkedProgram)
  {
    return false;
  }

  // From linked program to SPIR-V
  result = m_linkedProgram->getTargetCode(0, m_spirv.writeRef(), diagnostics.writeRef());
  logAndAppendDiagnostics(diagnostics);
  if(SLANG_FAILED(result) || nullptr == m_spirv)
  {
    return false;
  }
  return true;
}

void nvslang::SlangCompiler::createSession()
{
  m_session = {};

  slang::SessionDesc desc{
      .targets                  = m_targets.data(),
      .targetCount              = SlangInt(m_targets.size()),
      .searchPaths              = m_searchPathsUtf8Pointers.data(),
      .searchPathCount          = SlangInt(m_searchPathsUtf8Pointers.size()),
      .preprocessorMacros       = m_macros.data(),
      .preprocessorMacroCount   = SlangInt(m_macros.size()),
      .allowGLSLSyntax          = true,
      .compilerOptionEntries    = m_options.data(),
      .compilerOptionEntryCount = uint32_t(m_options.size()),
  };
  m_globalSession->createSession(desc, m_session.writeRef());
}

//--------------------------------------------------------------------------------------------------
// Usage example
//--------------------------------------------------------------------------------------------------
[[maybe_unused]] static void usage_SlangCompiler()
{
  nvslang::SlangCompiler slangCompiler;
  slangCompiler.defaultTarget();
  slangCompiler.defaultOptions();

  // Configure compiler settings as you wish
  const std::vector<std::filesystem::path> shadersPaths = {"include/shaders"};
  slangCompiler.addSearchPaths(shadersPaths);
  slangCompiler.addOption({slang::CompilerOptionName::DebugInformation,
                           {slang::CompilerOptionValueKind::Int, SLANG_DEBUG_INFO_LEVEL_MAXIMAL}});
  slangCompiler.addMacro({"MY_DEFINE", "1"});

  // Compile a shader file
  bool success = slangCompiler.compileFile("shader.slang");

  // Check if compilation was successful
  if(!success)
  {
    // Get the error message
    const std::string& errorMessages = slangCompiler.getLastDiagnosticMessage();
    LOGE("Compilation failed: %s\n", errorMessages.c_str());
  }
  else
  {
    // Get the compiled SPIR-V code
    const uint32_t* spirv     = slangCompiler.getSpirv();
    size_t          spirvSize = slangCompiler.getSpirvSize();

    // Check if there were any warnings
    const std::string& warningMessages = slangCompiler.getLastDiagnosticMessage();
    if(!warningMessages.empty())
    {
      LOGW("Compilation succeeded with warnings: %s\n", warningMessages.c_str());
    }
  }
}
