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

/*-------------------------------------------------------------------------------------------------
# class nvvkglsl::GlslCompiler

>  This class is a wrapper around the shaderc compiler to help compiling GLSL to Spir-V using Shaderc


Example: 
    see usage_GlslCompiler() in glsl.cpp


-------------------------------------------------------------------------------------------------*/

#pragma once

#include <filesystem>
#include <functional>
#include <memory>
#include <span>

#include <shaderc/shaderc.hpp>
#include <vulkan/vulkan_core.h>

namespace nvvkglsl {

inline shaderc_shader_kind getShaderKind(VkShaderStageFlagBits shaderStage)
{
  switch(shaderStage)
  {
    case VK_SHADER_STAGE_VERTEX_BIT:
      return shaderc_glsl_vertex_shader;
    case VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT:
      return shaderc_glsl_tess_control_shader;
    case VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT:
      return shaderc_glsl_tess_evaluation_shader;
    case VK_SHADER_STAGE_GEOMETRY_BIT:
      return shaderc_glsl_geometry_shader;
    case VK_SHADER_STAGE_FRAGMENT_BIT:
      return shaderc_glsl_fragment_shader;
    case VK_SHADER_STAGE_COMPUTE_BIT:
      return shaderc_glsl_compute_shader;
    case VK_SHADER_STAGE_RAYGEN_BIT_KHR:
      return shaderc_glsl_raygen_shader;
    case VK_SHADER_STAGE_ANY_HIT_BIT_KHR:
      return shaderc_glsl_anyhit_shader;
    case VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR:
      return shaderc_glsl_closesthit_shader;
    case VK_SHADER_STAGE_MISS_BIT_KHR:
      return shaderc_glsl_miss_shader;
    case VK_SHADER_STAGE_INTERSECTION_BIT_KHR:
      return shaderc_glsl_intersection_shader;
    case VK_SHADER_STAGE_CALLABLE_BIT_KHR:
      return shaderc_glsl_callable_shader;
    case VK_SHADER_STAGE_TASK_BIT_EXT:
      return shaderc_glsl_task_shader;
    case VK_SHADER_STAGE_MESH_BIT_EXT:
      return shaderc_glsl_mesh_shader;
    default:
      return shaderc_glsl_infer_from_source;
  }
}

class GlslCompiler : public shaderc::Compiler
{
public:
  GlslCompiler();
  ~GlslCompiler() = default;

  // Adds a path to the include paths.
  void                                addSearchPaths(const std::vector<std::filesystem::path>& paths);
  std::vector<std::filesystem::path>& searchPaths() { return m_searchPaths; }
  void                                clearSearchPaths() { m_searchPaths.clear(); }

  // Accesses the Shaderc compile options. You can use this for preprocessor macros,
  // for instance; see the code sample.
  shaderc::CompileOptions& options() { return *m_compilerOptions; }
  void                     clearOptions() { m_compilerOptions = makeOptions(); }

  // Compiles a GLSL shader to SPIR-V. The file is found using the given
  // filename and include paths. `shaderKind` must be the correct type
  // of shader.
  // The output is a full SpvCompilationResult object. You can use the
  // getSpirv* helpers below to access it easily.
  shaderc::SpvCompilationResult compileFile(const std::filesystem::path& filename,
                                            shaderc_shader_kind          shader_kind,
                                            shaderc::CompileOptions*     overrideOptions = nullptr);

  // Returns a pointer to the SPIR-V.
  static const uint32_t* getSpirv(const shaderc::SpvCompilationResult& compResult)
  {
    return reinterpret_cast<const uint32_t*>(compResult.begin());
  }
  // Returns the size of the SPIR-V in bytes.
  static size_t getSpirvSize(const shaderc::SpvCompilationResult& compResult)
  {
    return (compResult.end() - compResult.begin()) * sizeof(uint32_t);
  }

  static std::span<const uint32_t> getSpirvData(const shaderc::SpvCompilationResult& compResult)
  {
    return std::span<const uint32_t>(compResult.begin(), compResult.end());
  }

  static VkShaderModuleCreateInfo makeShaderModuleCreateInfo(const shaderc::SpvCompilationResult& compResult,
                                                             VkShaderModuleCreateFlags            flags = 0)
  {
    VkShaderModuleCreateInfo createInfo = {
        .sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .flags    = flags,
        .codeSize = getSpirvSize(compResult),
        .pCode    = getSpirv(compResult),
    };

    return createInfo;
  }

  static bool isValid(const shaderc::SpvCompilationResult& compResult);


  // The compile callback is called on every successful compilation with the
  // input file and its SPIR-V result. You can use this, for instance,
  // to register shaders with Nsight Aftermath.
  void setCompileCallback(std::function<void(const std::filesystem::path& sourceFile, const uint32_t* spirvCode, size_t spirvSize)> callback)
  {
    m_callback = callback;
  }

  // Sets the most typical compilation target.
  void defaultTarget()
  {
    m_compilerOptions->SetTargetSpirv(shaderc_spirv_version::shaderc_spirv_version_1_6);
    m_compilerOptions->SetTargetEnvironment(shaderc_target_env_vulkan, shaderc_env_version_vulkan_1_4);
  }

  // Sets the most typical compilation options. Note that without this, the
  // compiler options are very minimal.
  void defaultOptions()
  {
    m_compilerOptions->SetGenerateDebugInfo();
    m_compilerOptions->SetOptimizationLevel(shaderc_optimization_level_zero);
  }

private:
  std::unique_ptr<shaderc::CompileOptions> makeOptions();

  std::vector<std::filesystem::path>       m_searchPaths{};
  std::unique_ptr<shaderc::CompileOptions> m_compilerOptions;

  std::function<void(const std::filesystem::path& sourceFile, const uint32_t* spirvCode, size_t spirvSize)> m_callback;
};


}  // namespace nvvkglsl
