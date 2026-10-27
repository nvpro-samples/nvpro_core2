/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "basic_types.hpp"

namespace nvdx12 {

/*-------------------------------------------------------------------------------------------------
# class nvdx12::ShaderCompiler

> Runtime HLSL compilation through dxcompiler.dll with an on-disk DXIL
> cache, the D3D12 counterpart of nvvkglsl::GlslCompiler and
> nvslang::SlangCompiler.

`compileFile` and `compileSource` return a `ShaderBlob`: the DXIL, the dxc
diagnostics and the files the compilation read (for hot reload). Every
compilation passes `-HV 2021`, `-enable-16bit-types`, the search paths as
`-I` and the global defines; library profiles add
`-disable-payload-qualifiers`.

The cache key covers the compiler version, the arguments and the
preprocessed text, so an edit in any included file produces a new entry.
Profiles above the device's shader model (`setMaxShaderModel`) are clamped
with a one-time warning.

Compilations are serialized; the setters, `findFile` and the cache counters
take the same lock, so they may be called from any thread. The references
returned by `searchPaths`, `globalDefines` and `cacheDir` are not protected:
read them while no other thread changes the settings.

See usage_ShaderCompiler() in shader_compiler.cpp.
-------------------------------------------------------------------------------------------------*/

// Result of one compilation; `valid()` is the only status check call sites need.
struct ShaderBlob
{
  std::vector<uint8_t>               dxil;
  std::string                        errors;  // dxc diagnostics (warnings when valid, errors otherwise)
  std::string                        entryPoint;
  std::string                        profile;
  std::vector<std::filesystem::path> dependencies;  // source + every included file, for hot reload
  bool                               fromCache{false};

  bool                  valid() const { return !dxil.empty(); }
  D3D12_SHADER_BYTECODE bytecode() const { return D3D12_SHADER_BYTECODE{dxil.data(), dxil.size()}; }
  explicit              operator bool() const { return valid(); }
};

struct ShaderDefine
{
  std::string name;
  std::string value;  // empty = defined without a value
};

class ShaderCompiler
{
public:
  // Called after every compilation that produced DXIL, cache hits included
  // (shader registration with Nsight Aftermath, dumps).
  using CompileCallback = std::function<void(const std::filesystem::path& sourceFile, const void* dxil, size_t dxilSize)>;

  ShaderCompiler();  // both out of line: Impl is complete in the .cpp alone
  ~ShaderCompiler();
  ShaderCompiler(const ShaderCompiler&)            = delete;
  ShaderCompiler& operator=(const ShaderCompiler&) = delete;

  // Loads dxcompiler.dll: `dxcompilerDll` empty = next to the executable, then
  // the library search path. `cacheDir` empty = caching disabled.
  HRESULT init(const std::filesystem::path& dxcompilerDll = {}, const std::filesystem::path& cacheDir = {});
  void    deinit();

  // Directories searched in order for #include and for relative source paths, before the working directory.
  void                                      addSearchPath(const std::filesystem::path& path);
  void                                      addSearchPaths(const std::vector<std::filesystem::path>& paths);
  void                                      clearSearchPaths();
  const std::vector<std::filesystem::path>& searchPaths() const { return m_searchPaths; }

  // Defines applied to every compilation.
  void                             addGlobalDefine(const std::string& name, const std::string& value = {});
  void                             clearGlobalDefines();
  const std::vector<ShaderDefine>& globalDefines() const { return m_globalDefines; }

  // Debug info (-Zi, -Qembed_debug) and optimization level (-Od, -O1..-O3). Defaults: no debug info, -O3.
  void setDebugInfo(bool enable);
  void setOptimizationLevel(uint32_t level);
  // Highest shader model the device supports; profiles above it are clamped.
  void setMaxShaderModel(D3D_SHADER_MODEL sm);

  // DXIL cache: `setCacheDir` redirects it (created if missing; an empty
  // path disables it), `setCacheEnabled` toggles lookups without
  // forgetting the directory.
  void                         setCacheDir(const std::filesystem::path& cacheDir);
  void                         setCacheEnabled(bool enable);
  const std::filesystem::path& cacheDir() const { return m_cacheDir; }
  uint32_t                     cacheHits() const;
  uint32_t                     cacheMisses() const;

  // Called after each successful compilation, outside the lock.
  void setCompileCallback(CompileCallback callback);

  // Compiles `file` (absolute, or relative to a search path) for `profile`
  // ("cs_6_6", "lib_6_9", ...). `entryPoint` is ignored for lib profiles.
  // `extraArgs` are appended to the dxc command line verbatim.
  ShaderBlob compileFile(const std::filesystem::path&  file,
                         const char*                   profile,
                         const char*                   entryPoint = "main",
                         std::span<const ShaderDefine> defines    = {},
                         std::span<const std::wstring> extraArgs  = {});

  // Same for in-memory source; `name` labels diagnostics and the cache entry.
  ShaderBlob compileSource(std::string_view              source,
                           const char*                   name,
                           const char*                   profile,
                           const char*                   entryPoint = "main",
                           std::span<const ShaderDefine> defines    = {},
                           std::span<const std::wstring> extraArgs  = {});

  // Resolves a relative shader path against the search paths, then the working directory; empty when not found.
  std::filesystem::path findFile(const std::filesystem::path& file) const;

  bool     isValid() const { return m_impl != nullptr; }
  explicit operator bool() const { return isValid(); }

private:
  struct Impl;  // holds the dxc interfaces (dxcapi.h stays out of this header)

  std::unique_ptr<Impl>              m_impl;
  std::vector<std::filesystem::path> m_searchPaths;
  std::vector<ShaderDefine>          m_globalDefines;
  std::filesystem::path              m_cacheDir;
  bool                               m_cacheEnabled{true};
  bool                               m_debugInfo{false};
  uint32_t                           m_optimizationLevel{3};
  D3D_SHADER_MODEL                   m_maxShaderModel{D3D_SHADER_MODEL(0x6A)};  // 6.10
  uint32_t                           m_cacheHits{0};
  uint32_t                           m_cacheMisses{0};
  CompileCallback                    m_callback;
  mutable std::mutex                 m_mutex;  // serializes compilations, the settings and the cache counters
};

// True when every blob is valid or default-constructed (never compiled).
bool verifyShaders(std::span<const ShaderBlob> blobs);

}  // namespace nvdx12
