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

#include <volk.h>
#include <nvutils/logger.hpp>
#include <nvutils/file_operations.hpp>
#include <nvvk/check_error.hpp>
#include <nvutils/spirv.hpp>

#include "glsl.hpp"


// Implementation of the libshaderc includer interface.
class GlslIncluder : public shaderc::CompileOptions::IncluderInterface
{
public:
  GlslIncluder(const std::vector<std::filesystem::path>& searchPaths)
      : m_searchPaths(searchPaths)
  {
  }

  // Subtype of shaderc_include_result that holds the include data we found;
  // MUST be static_cast to this type before deleting as shaderc_include_result lacks virtual destructor.
  struct IncludeResult : public shaderc_include_result
  {
    IncludeResult(const std::string& content, const std::string& filenameFoundUtf8)
        : m_content(content)
        , m_filenameFoundUtf8(filenameFoundUtf8)
    {
      this->source_name        = m_filenameFoundUtf8.data();
      this->source_name_length = m_filenameFoundUtf8.size();
      this->content            = m_content.data();
      this->content_length     = m_content.size();
      this->user_data          = nullptr;
    }
    const std::string m_content;
    const std::string m_filenameFoundUtf8;
  };

  shaderc_include_result* GetInclude(const char* requested_source, shaderc_include_type type, const char* requesting_source, size_t include_depth) override
  {
    // Check the relative path for #include "quotes"
    std::filesystem::path find_name;
    if(type == shaderc_include_type_relative)
    {
      std::filesystem::path relative_path = std::filesystem::path(requesting_source).parent_path() / requested_source;
      if(std::filesystem::exists(relative_path))
        find_name = relative_path;
    }

    // If nothing found yet, search include directories
    // TODO: skip nvutils::findFile searching the current working directory
    if(find_name.empty())
    {
      find_name = nvutils::findFile(requested_source, m_searchPaths);
    }

    std::string src_code;
    if(find_name.empty())
    {
      // [shaderc.h] For a failed inclusion, this contains the error message.
      src_code = "Could not find include file in any include path.";
    }
    else
    {
      src_code = nvutils::loadFile(find_name);
    }
    return new IncludeResult(src_code, nvutils::utf8FromPath(find_name));
  }

  // Handles shaderc_include_result_release_fn callbacks.
  void ReleaseInclude(shaderc_include_result* data) override { delete static_cast<IncludeResult*>(data); };

  const std::vector<std::filesystem::path>& m_searchPaths;
};


nvvkglsl::GlslCompiler::GlslCompiler()
{
  m_compilerOptions = std::move(makeOptions());
}

void nvvkglsl::GlslCompiler::addSearchPaths(const std::vector<std::filesystem::path>& paths)
{
  for(const auto& p : paths)
    m_searchPaths.push_back(p);
}

std::unique_ptr<shaderc::CompileOptions> nvvkglsl::GlslCompiler::makeOptions()
{
  std::unique_ptr<shaderc::CompileOptions> options = std::make_unique<shaderc::CompileOptions>();
  options->SetIncluder(std::make_unique<GlslIncluder>(m_searchPaths));
  options->AddMacroDefinition("__GLSL__", "1");
  return options;
}

shaderc::SpvCompilationResult nvvkglsl::GlslCompiler::compileFile(const std::filesystem::path& filename,
                                                                  shaderc_shader_kind          shader_kind,
                                                                  shaderc::CompileOptions*     overrideOptions)
{
  std::filesystem::path sourceFile = nvutils::findFile(filename, m_searchPaths);
  if(sourceFile.empty())
    return {};
  std::string                   sourceCode = nvutils::loadFile(sourceFile);
  shaderc::SpvCompilationResult compResult =
      CompileGlslToSpv(sourceCode, shader_kind, nvutils::utf8FromPath(sourceFile.filename()).c_str(),
                       overrideOptions ? *overrideOptions : *m_compilerOptions);
  if(compResult.GetCompilationStatus() == shaderc_compilation_status_success)
  {
    if(m_callback)
    {
      m_callback(sourceFile, getSpirv(compResult), getSpirvSize(compResult));
    }
  }

  return compResult;
}


bool nvvkglsl::GlslCompiler::isValid(const shaderc::SpvCompilationResult& compResult)
{
  if(compResult.GetCompilationStatus() != shaderc_compilation_status_success)
  {
    const std::string err = compResult.GetErrorMessage();
    LOGW("Shader compilation error: %s\n", err.c_str());
    return false;
  }
  return true;
}

// =================================================================================================
// Example
// =================================================================================================
[[maybe_unused]] static void usage_GlslCompiler()
{
#define PROJECT_EXE_TO_SOURCE_DIRECTORY "../myproject/shaders"
  std::filesystem::path exePath = nvutils::getExecutablePath().parent_path();
  const std::vector<std::filesystem::path> searchPaths = {std::filesystem::path(PROJECT_EXE_TO_SOURCE_DIRECTORY) / "shaders",
                                                          exePath / "shaders", exePath};

  nvvkglsl::GlslCompiler glslCompiler = {};
  glslCompiler.addSearchPaths(searchPaths);
  glslCompiler.defaultOptions();
  glslCompiler.defaultTarget();
  glslCompiler.options().SetGenerateDebugInfo();
  glslCompiler.options().SetOptimizationLevel(shaderc_optimization_level_zero);
  glslCompiler.options().AddMacroDefinition("MY_DEFINE", "1");

  VkShaderCreateInfoEXT shaderCreateInfos{
      .sType    = VK_STRUCTURE_TYPE_SHADER_CREATE_INFO_EXT,
      .stage    = VK_SHADER_STAGE_COMPUTE_BIT,
      .codeType = VK_SHADER_CODE_TYPE_SPIRV_EXT,
      .pName    = "main",
  };

  auto shaderComp = glslCompiler.compileFile("shader.comp.glsl", shaderc_glsl_compute_shader);
  if(shaderComp.GetCompilationStatus() == shaderc_compilation_status_success)
  {
    shaderCreateInfos.codeSize = glslCompiler.getSpirvSize(shaderComp);
    shaderCreateInfos.pCode    = glslCompiler.getSpirv(shaderComp);
  }
  VkShaderEXT computeShader{};  // The compute shader
  VkDevice    device = nullptr;
  NVVK_CHECK(vkCreateShadersEXT(device, 1, &shaderCreateInfos, NULL, &computeShader));
}
