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

#include "shader_compiler.hpp"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <fstream>

#include <dxcapi.h>

#include <nvutils/file_operations.hpp>
#include <nvutils/logger.hpp>

#include "check_error.hpp"
#include "text_encoding.hpp"

namespace nvdx12 {

namespace fs = std::filesystem;

namespace {

// FNV-1a 64 over the pieces that make a compilation unique.
struct Fnv1a64
{
  uint64_t hash{0xcbf29ce484222325ull};
  void     add(const void* data, size_t size)
  {
    const uint8_t* bytes = static_cast<const uint8_t*>(data);
    for(size_t i = 0; i < size; i++)
    {
      hash ^= bytes[i];
      hash *= 0x100000001b3ull;
    }
  }
  void add(std::string_view s) { add(s.data(), s.size()); }
  void add(std::wstring_view s) { add(s.data(), s.size() * sizeof(wchar_t)); }
};

bool readFile(const fs::path& path, std::vector<uint8_t>& out)
{
  std::ifstream in(path, std::ios::binary | std::ios::ate);
  if(!in)
    return false;
  std::streamsize size = in.tellg();
  if(size < 0)
    return false;
  out.resize(size_t(size));
  in.seekg(0);
  return size == 0 || bool(in.read(reinterpret_cast<char*>(out.data()), size));
}

// Writes through a temporary so a concurrent reader never sees a partial file.
bool writeFileAtomic(const fs::path& path, const std::vector<uint8_t>& bytes)
{
  fs::path tmp = path;
  tmp += L".tmp";
  {
    std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
    if(!out || !out.write(reinterpret_cast<const char*>(bytes.data()), std::streamsize(bytes.size())))
      return false;
  }
  std::error_code ec;
  fs::rename(tmp, path, ec);
  if(ec)
    fs::remove(tmp, ec);
  return !ec;
}

fs::path canonicalOrNormal(const fs::path& path)
{
  std::error_code ec;
  fs::path        canonical = fs::weakly_canonical(path, ec);
  return ec ? path.lexically_normal() : canonical;
}

// Resolves include candidates against the search paths and records every
// file handed to dxc, so hot reload knows the dependency set.
class IncludeHandler : public IDxcIncludeHandler
{
public:
  IncludeHandler(IDxcUtils* utils, const std::vector<fs::path>& searchPaths, std::vector<fs::path>& dependencies)
      : m_utils(utils)
      , m_searchPaths(searchPaths)
      , m_dependencies(dependencies)
  {
  }

  // IUnknown; the object lives on the stack of the compile call.
  HRESULT STDMETHODCALLTYPE QueryInterface(REFIID riid, void** object) override
  {
    if(!object)
      return E_POINTER;
    if(riid == __uuidof(IDxcIncludeHandler) || riid == __uuidof(IUnknown))
    {
      *object = static_cast<IDxcIncludeHandler*>(this);
      AddRef();
      return S_OK;
    }
    *object = nullptr;
    return E_NOINTERFACE;
  }
  ULONG STDMETHODCALLTYPE AddRef() override { return ++m_refCount; }
  ULONG STDMETHODCALLTYPE Release() override { return --m_refCount; }

  HRESULT STDMETHODCALLTYPE LoadSource(LPCWSTR filename, IDxcBlob** includeSource) override
  {
    if(!includeSource)
      return E_POINTER;
    fs::path resolved = resolve(filename);
    if(resolved.empty())
      return HRESULT_FROM_WIN32(ERROR_FILE_NOT_FOUND);
    ComPtr<IDxcBlobEncoding> blob;
    HRESULT                  hr = m_utils->LoadFile(resolved.c_str(), nullptr, &blob);
    if(FAILED(hr))
      return hr;
    record(resolved);
    *includeSource = blob.Detach();
    return S_OK;
  }

private:
  void record(const fs::path& path)
  {
    fs::path canonical = canonicalOrNormal(path);
    if(std::find(m_dependencies.begin(), m_dependencies.end(), canonical) == m_dependencies.end())
      m_dependencies.push_back(canonical);
  }

  fs::path resolve(const wchar_t* filename) const
  {
    // dxc joins the include directive with the including file's directory
    // or one of the -I paths; "./" prefixes come from that join.
    std::wstring_view name(filename);
    while(name.size() >= 2 && name[0] == L'.' && (name[1] == L'/' || name[1] == L'\\'))
      name.remove_prefix(2);
    // Relative names try the search paths before the working directory, as findFile does.
    fs::path        candidate(name);
    std::error_code ec;
    if(candidate.is_relative())
    {
      for(size_t i = 0; i < m_searchPaths.size(); i++)
      {
        fs::path joined = m_searchPaths[i] / candidate;
        if(fs::is_regular_file(joined, ec))
          return joined;
      }
    }
    return fs::is_regular_file(candidate, ec) ? candidate : fs::path{};
  }

  IDxcUtils*                   m_utils;
  const std::vector<fs::path>& m_searchPaths;
  std::vector<fs::path>&       m_dependencies;
  std::atomic<ULONG>           m_refCount{1};
};

bool isLibraryProfile(std::string_view profile)
{
  return profile.substr(0, 4) == "lib_";
}

// Minor version of "xx_6_N"; -1 when the profile is not shader model 6.
int profileMinor(std::string_view profile)
{
  size_t pos = profile.find("_6_");
  if(pos == std::string_view::npos)
    return -1;
  int minor = 0;
  for(size_t i = pos + 3; i < profile.size() && profile[i] >= '0' && profile[i] <= '9'; i++)
    minor = minor * 10 + (profile[i] - '0');
  return minor;
}

std::string diagnosticsOf(IDxcResult* result)
{
  ComPtr<IDxcBlobUtf8> errors;
  if(result && SUCCEEDED(result->GetOutput(DXC_OUT_ERRORS, IID_PPV_ARGS(&errors), nullptr)) && errors
     && errors->GetStringLength() > 0)
  {
    return std::string(errors->GetStringPointer(), errors->GetStringLength());
  }
  return {};
}

std::wstring toWide(std::string_view s)
{
  return wtf16FromUtf8(std::string(s).c_str());
}

// Creates `cacheDir` and returns it, or an empty path (caching disabled) when that fails.
fs::path prepareCacheDir(const fs::path& cacheDir)
{
  if(cacheDir.empty())
    return {};
  std::error_code ec;
  fs::create_directories(cacheDir, ec);
  if(ec)
  {
    LOGW("Shader cache directory %s cannot be created (%s); caching disabled\n",
         nvutils::utf8FromPath(cacheDir).c_str(), ec.message().c_str());
    return {};
  }
  return cacheDir;
}

}  // namespace

// Holds the dxc interfaces and the compile routine shared by compileFile
// and compileSource; as a nested class it reaches the owner's settings.
struct ShaderCompiler::Impl
{
  HMODULE               dll{nullptr};
  DxcCreateInstanceProc createInstance{nullptr};
  ComPtr<IDxcUtils>     utils;
  ComPtr<IDxcCompiler3> compiler;
  std::string           compilerVersion;  // "major.minor hash" of dxcompiler.dll, part of the cache key
  bool                  warnedClamp{false};

  ~Impl()
  {
    compiler.Reset();
    utils.Reset();
    if(dll)
      FreeLibrary(dll);
  }

  struct Request
  {
    DxcBuffer                 source{};
    std::vector<std::wstring> args;
    std::wstring              displayName;
  };

  // Runs dxc with the request arguments plus `extra`; null when the call
  // itself failed (diagnostics are in the result otherwise).
  ComPtr<IDxcResult> run(const Request& req, std::initializer_list<const wchar_t*> extra, IDxcIncludeHandler* includes)
  {
    std::vector<LPCWSTR> argv;
    argv.reserve(req.args.size() + extra.size() + 1);
    argv.push_back(req.displayName.c_str());
    for(size_t i = 0; i < req.args.size(); i++)
      argv.push_back(req.args[i].c_str());
    for(const wchar_t* a : extra)
      argv.push_back(a);
    ComPtr<IDxcResult> result;
    HRESULT hr = compiler->Compile(&req.source, argv.data(), UINT32(argv.size()), includes, IID_PPV_ARGS(&result));
    if(FAILED(hr))
    {
      LOGE("dxc Compile() failed for %s: %s\n", utf8FromWtf16(req.displayName.c_str()).c_str(), hresultToString(hr).c_str());
      return nullptr;
    }
    return result;
  }

  // Caller holds owner.m_mutex.
  static ShaderBlob compile(ShaderCompiler&               owner,
                            std::string_view              source,
                            const char*                   name,
                            const char*                   profile,
                            const char*                   entryPoint,
                            std::span<const ShaderDefine> defines,
                            std::span<const std::wstring> extraArgs);
};

ShaderBlob ShaderCompiler::Impl::compile(ShaderCompiler&               owner,
                                         std::string_view              source,
                                         const char*                   name,
                                         const char*                   profile,
                                         const char*                   entryPoint,
                                         std::span<const ShaderDefine> defines,
                                         std::span<const std::wstring> extraArgs)
{
  ShaderBlob blob;
  blob.profile    = profile ? profile : "";
  blob.entryPoint = entryPoint ? entryPoint : "";
  name            = name ? name : "<memory>";

  Impl* impl = owner.m_impl.get();
  if(!impl || !impl->compiler)
  {
    blob.errors = "ShaderCompiler is not initialized";
    LOGE("%s (%s)\n", blob.errors.c_str(), name);
    return blob;
  }

  // Clamp the profile to the device's shader model (6.10 -> 6.9 ...).
  std::string effectiveProfile = blob.profile;
  int         minor            = profileMinor(effectiveProfile);
  int         maxMinor         = int(owner.m_maxShaderModel) & 0xF;
  if(minor > maxMinor)
  {
    std::string clamped = effectiveProfile.substr(0, effectiveProfile.find("_6_") + 3) + std::to_string(maxMinor);
    if(!impl->warnedClamp)
    {
      LOGW("Shader model 6.%d is not supported by the device; compiling %s as %s (reported once)\n", minor,
           effectiveProfile.c_str(), clamped.c_str());
      impl->warnedClamp = true;
    }
    effectiveProfile = clamped;
  }

  Request req;
  req.displayName = toWide(name);
  req.source      = DxcBuffer{source.data(), source.size(), DXC_CP_UTF8};
  req.args.push_back(L"-T");
  req.args.push_back(toWide(effectiveProfile));
  if(!isLibraryProfile(effectiveProfile) && !blob.entryPoint.empty())
  {
    req.args.push_back(L"-E");
    req.args.push_back(toWide(blob.entryPoint));
  }
  req.args.push_back(L"-HV");
  req.args.push_back(L"2021");
  req.args.push_back(L"-enable-16bit-types");
  if(isLibraryProfile(effectiveProfile))
  {
    // Payload structs are plain structs shared across pipelines; SM 6.7+
    // libraries would otherwise demand [raypayload] access qualifiers.
    req.args.push_back(L"-disable-payload-qualifiers");
  }
  for(size_t i = 0; i < owner.m_searchPaths.size(); i++)
  {
    req.args.push_back(L"-I");
    req.args.push_back(owner.m_searchPaths[i].wstring());
  }
  auto addDefine = [&](const ShaderDefine& d) {
    req.args.push_back(L"-D");
    req.args.push_back(d.value.empty() ? toWide(d.name) : toWide(d.name + "=" + d.value));
  };
  for(size_t i = 0; i < owner.m_globalDefines.size(); i++)
    addDefine(owner.m_globalDefines[i]);
  for(size_t i = 0; i < defines.size(); i++)
    addDefine(defines[i]);
  if(owner.m_optimizationLevel == 0)
    req.args.push_back(L"-Od");
  else
    req.args.push_back(L"-O" + std::to_wstring(std::min(owner.m_optimizationLevel, 3u)));
  if(owner.m_debugInfo)
  {
    req.args.push_back(L"-Zi");
    req.args.push_back(L"-Qembed_debug");
  }
  for(size_t i = 0; i < extraArgs.size(); i++)
    req.args.push_back(extraArgs[i]);

  IncludeHandler includes(impl->utils.Get(), owner.m_searchPaths, blob.dependencies);

  // Cache lookup: the key covers the arguments and the preprocessed text,
  // so an edit in any included file produces a new entry.
  bool     useCache = owner.m_cacheEnabled && !owner.m_cacheDir.empty();
  fs::path cachePath;
  if(useCache)
  {
    Fnv1a64 key;
    key.add(impl->compilerVersion);
    key.add(effectiveProfile);
    key.add(blob.entryPoint);
    for(size_t i = 0; i < req.args.size(); i++)
      key.add(req.args[i]);

    bool               hashed = false;
    ComPtr<IDxcResult> pre    = impl->run(req, {L"-P"}, &includes);
    HRESULT            status = E_FAIL;
    if(pre && SUCCEEDED(pre->GetStatus(&status)) && SUCCEEDED(status))
    {
      ComPtr<IDxcBlobUtf8> hlsl;
      if(pre->HasOutput(DXC_OUT_HLSL) && SUCCEEDED(pre->GetOutput(DXC_OUT_HLSL, IID_PPV_ARGS(&hlsl), nullptr)) && hlsl)
      {
        key.add(hlsl->GetStringPointer(), hlsl->GetStringLength());
        hashed = true;
      }
      else
      {
        // No preprocessed text: hash the source and every file the preprocess pass pulled in.
        key.add(source);
        hashed = true;
        std::vector<uint8_t> bytes;
        for(size_t i = 0; i < blob.dependencies.size(); i++)
        {
          if(!readFile(blob.dependencies[i], bytes))
          {
            hashed = false;
            break;
          }
          key.add(bytes.data(), bytes.size());
        }
      }
    }
    if(hashed)
    {
      char hex[32];
      std::snprintf(hex, sizeof(hex), "%016llx", static_cast<unsigned long long>(key.hash));
      cachePath = owner.m_cacheDir / (std::string(hex) + ".dxil");
      if(readFile(cachePath, blob.dxil) && !blob.dxil.empty())
      {
        blob.fromCache = true;
        owner.m_cacheHits++;
        return blob;
      }
      blob.dxil.clear();
    }
  }

  ComPtr<IDxcResult> result = impl->run(req, {}, &includes);
  if(!result)
  {
    blob.errors = "dxc did not return a result";
    return blob;
  }
  blob.errors    = diagnosticsOf(result.Get());
  HRESULT status = E_FAIL;
  result->GetStatus(&status);
  if(FAILED(status))
  {
    LOGE("Shader compilation failed: %s (%s)\n%s\n", name, effectiveProfile.c_str(), blob.errors.c_str());
    return blob;
  }
  if(!blob.errors.empty())
    LOGW("Shader warnings: %s\n%s\n", name, blob.errors.c_str());

  ComPtr<IDxcBlob> object;
  if(FAILED(result->GetOutput(DXC_OUT_OBJECT, IID_PPV_ARGS(&object), nullptr)) || !object || object->GetBufferSize() == 0)
  {
    blob.errors += "\ndxc produced no object";
    LOGE("Shader compilation produced no object: %s\n", name);
    return blob;
  }
  const uint8_t* begin = static_cast<const uint8_t*>(object->GetBufferPointer());
  blob.dxil.assign(begin, begin + object->GetBufferSize());

  if(useCache)
  {
    owner.m_cacheMisses++;
    if(!cachePath.empty() && !writeFileAtomic(cachePath, blob.dxil))
      LOGW("Could not write shader cache entry %s\n", nvutils::utf8FromPath(cachePath).c_str());
  }
  return blob;
}

ShaderCompiler::ShaderCompiler()  = default;
ShaderCompiler::~ShaderCompiler() = default;

HRESULT ShaderCompiler::init(const fs::path& dxcompilerDll, const fs::path& cacheDir)
{
  deinit();
  std::lock_guard<std::mutex> lock(m_mutex);
  std::unique_ptr<Impl>       impl = std::make_unique<Impl>();

  if(dxcompilerDll.empty())
  {
    impl->dll = LoadLibraryExW(L"dxcompiler.dll", nullptr, LOAD_LIBRARY_SEARCH_APPLICATION_DIR);
    if(!impl->dll)
    {
      LOGW("dxcompiler.dll not found next to the executable, trying the library search path\n");
    }
  }
  else
  {
    impl->dll = LoadLibraryW(dxcompilerDll.c_str());
    if(!impl->dll)
    {
      LOGW("dxcompiler.dll not found at %s, trying the library search path\n", nvutils::utf8FromPath(dxcompilerDll).c_str());
    }
  }
  if(!impl->dll)
  {
    impl->dll = LoadLibraryW(L"dxcompiler.dll");
  }
  if(!impl->dll)
  {
    LOGE("Could not load dxcompiler.dll\n");
    return HRESULT_FROM_WIN32(ERROR_MOD_NOT_FOUND);
  }
  impl->createInstance = reinterpret_cast<DxcCreateInstanceProc>(GetProcAddress(impl->dll, "DxcCreateInstance"));
  if(!impl->createInstance)
  {
    LOGE("dxcompiler.dll has no DxcCreateInstance entry point\n");
    return HRESULT_FROM_WIN32(ERROR_PROC_NOT_FOUND);
  }
  NVDX12_FAIL_RETURN(impl->createInstance(CLSID_DxcUtils, IID_PPV_ARGS(&impl->utils)));
  NVDX12_FAIL_RETURN(impl->createInstance(CLSID_DxcCompiler, IID_PPV_ARGS(&impl->compiler)));

  ComPtr<IDxcVersionInfo> versionInfo;
  UINT32                  major = 0, minor = 0;
  if(SUCCEEDED(impl->compiler.As(&versionInfo)) && SUCCEEDED(versionInfo->GetVersion(&major, &minor)))
  {
    impl->compilerVersion = std::to_string(major) + "." + std::to_string(minor);
    ComPtr<IDxcVersionInfo2> versionInfo2;
    UINT32                   commitCount = 0;
    char*                    commitHash  = nullptr;
    if(SUCCEEDED(versionInfo.As(&versionInfo2)) && SUCCEEDED(versionInfo2->GetCommitInfo(&commitCount, &commitHash)) && commitHash)
    {
      impl->compilerVersion += std::string(" ") + commitHash;
      CoTaskMemFree(commitHash);
    }
  }

  m_impl        = std::move(impl);
  m_cacheDir    = prepareCacheDir(cacheDir);
  m_cacheHits   = 0;
  m_cacheMisses = 0;
  return S_OK;
}

void ShaderCompiler::deinit()
{
  std::lock_guard<std::mutex> lock(m_mutex);
  m_impl.reset();
}

void ShaderCompiler::setCompileCallback(CompileCallback callback)
{
  std::lock_guard<std::mutex> lock(m_mutex);
  m_callback = std::move(callback);
}

void ShaderCompiler::addSearchPath(const fs::path& path)
{
  std::error_code             ec;
  fs::path                    abs = fs::absolute(path, ec);
  std::lock_guard<std::mutex> lock(m_mutex);
  m_searchPaths.push_back(ec ? path : abs.lexically_normal());
}

void ShaderCompiler::clearSearchPaths()
{
  std::lock_guard<std::mutex> lock(m_mutex);
  m_searchPaths.clear();
}

void ShaderCompiler::addSearchPaths(const std::vector<fs::path>& paths)
{
  for(size_t i = 0; i < paths.size(); i++)
    addSearchPath(paths[i]);
}

void ShaderCompiler::addGlobalDefine(const std::string& name, const std::string& value)
{
  std::lock_guard<std::mutex> lock(m_mutex);
  m_globalDefines.push_back({name, value});
}

void ShaderCompiler::clearGlobalDefines()
{
  std::lock_guard<std::mutex> lock(m_mutex);
  m_globalDefines.clear();
}

void ShaderCompiler::setDebugInfo(bool enable)
{
  std::lock_guard<std::mutex> lock(m_mutex);
  m_debugInfo = enable;
}

void ShaderCompiler::setOptimizationLevel(uint32_t level)
{
  std::lock_guard<std::mutex> lock(m_mutex);
  m_optimizationLevel = level;
}

void ShaderCompiler::setMaxShaderModel(D3D_SHADER_MODEL sm)
{
  std::lock_guard<std::mutex> lock(m_mutex);
  m_maxShaderModel = sm;
}

void ShaderCompiler::setCacheEnabled(bool enable)
{
  std::lock_guard<std::mutex> lock(m_mutex);
  m_cacheEnabled = enable;
}

uint32_t ShaderCompiler::cacheHits() const
{
  std::lock_guard<std::mutex> lock(m_mutex);
  return m_cacheHits;
}

uint32_t ShaderCompiler::cacheMisses() const
{
  std::lock_guard<std::mutex> lock(m_mutex);
  return m_cacheMisses;
}

void ShaderCompiler::setCacheDir(const fs::path& cacheDir)
{
  std::lock_guard<std::mutex> lock(m_mutex);
  m_cacheDir = prepareCacheDir(cacheDir);
}

fs::path ShaderCompiler::findFile(const fs::path& file) const
{
  std::error_code ec;
  if(file.is_absolute())
    return fs::is_regular_file(file, ec) ? file : fs::path{};
  std::lock_guard<std::mutex> lock(m_mutex);
  for(size_t i = 0; i < m_searchPaths.size(); i++)
  {
    fs::path candidate = m_searchPaths[i] / file;
    if(fs::is_regular_file(candidate, ec))
      return candidate.lexically_normal();
  }
  if(fs::is_regular_file(file, ec))
    return fs::absolute(file, ec).lexically_normal();
  return {};
}

ShaderBlob ShaderCompiler::compileFile(const fs::path&               file,
                                       const char*                   profile,
                                       const char*                   entryPoint,
                                       std::span<const ShaderDefine> defines,
                                       std::span<const std::wstring> extraArgs)
{
  ShaderBlob blob;
  blob.profile    = profile ? profile : "";
  blob.entryPoint = entryPoint ? entryPoint : "";

  fs::path resolved = findFile(file);
  if(resolved.empty())
  {
    blob.errors = "shader file not found: " + nvutils::utf8FromPath(file);
    LOGE("%s\n", blob.errors.c_str());
    return blob;
  }
  std::vector<uint8_t> source;
  if(!readFile(resolved, source))
  {
    blob.errors = "cannot read shader file: " + nvutils::utf8FromPath(resolved);
    LOGE("%s\n", blob.errors.c_str());
    return blob;
  }
  std::string_view text(reinterpret_cast<const char*>(source.data()), source.size());
  std::string      name = nvutils::utf8FromPath(resolved);
  CompileCallback  callback;
  {
    std::lock_guard<std::mutex> lock(m_mutex);
    blob     = Impl::compile(*this, text, name.c_str(), profile, entryPoint, defines, extraArgs);
    callback = m_callback;
  }
  // The include handler records the headers; the main file is a dependency too.
  blob.dependencies.insert(blob.dependencies.begin(), canonicalOrNormal(resolved));
  if(blob.valid() && callback)
    callback(resolved, blob.dxil.data(), blob.dxil.size());
  return blob;
}

ShaderBlob ShaderCompiler::compileSource(std::string_view              source,
                                         const char*                   name,
                                         const char*                   profile,
                                         const char*                   entryPoint,
                                         std::span<const ShaderDefine> defines,
                                         std::span<const std::wstring> extraArgs)
{
  ShaderBlob      blob;
  CompileCallback callback;
  {
    std::lock_guard<std::mutex> lock(m_mutex);
    blob     = Impl::compile(*this, source, name, profile, entryPoint, defines, extraArgs);
    callback = m_callback;
  }
  if(blob.valid() && callback)
    callback(fs::path(name ? name : ""), blob.dxil.data(), blob.dxil.size());
  return blob;
}

bool verifyShaders(std::span<const ShaderBlob> blobs)
{
  for(size_t i = 0; i < blobs.size(); i++)
  {
    const ShaderBlob& blob          = blobs[i];
    bool              neverCompiled = blob.profile.empty() && blob.errors.empty() && blob.dxil.empty();
    if(!blob.valid() && !neverCompiled)
      return false;
  }
  return true;
}

}  // namespace nvdx12

//--------------------------------------------------------------------------------------------------
// Usage example
//--------------------------------------------------------------------------------------------------
[[maybe_unused]] static void usage_ShaderCompiler()
{
  const std::filesystem::path exePath            = std::filesystem::current_path();
  const std::filesystem::path sourceDir          = exePath / "../myproject";
  D3D_SHADER_MODEL            highestShaderModel = D3D_SHADER_MODEL_6_6;  // from D3D12_FEATURE_SHADER_MODEL

  // dxcompiler.dll next to the executable, DXIL cache in a writable folder.
  nvdx12::ShaderCompiler compiler;
  NVDX12_CHECK(compiler.init({}, exePath / "shader_cache"));
  compiler.addSearchPaths({sourceDir / "shaders", exePath / "shaders", exePath});
  compiler.addGlobalDefine("MY_DEFINE", "1");
  compiler.setMaxShaderModel(highestShaderModel);  // lib_6_9 requests compile as lib_6_6 here
  compiler.setDebugInfo(true);
  compiler.setOptimizationLevel(3);

  // Compute shader from a file; the defines of this call come after the global ones.
  const nvdx12::ShaderDefine defines[] = {{"WORKGROUP_SIZE", "256"}, {"USE_FAST_PATH", ""}};
  nvdx12::ShaderBlob         compute   = compiler.compileFile("shader.comp.hlsl", "cs_6_6", "main", defines);
  if(!compute.valid())
  {
    LOGE("Compilation failed: %s\n", compute.errors.c_str());
  }
  else if(!compute.errors.empty())
  {
    LOGW("Compilation succeeded with warnings: %s\n", compute.errors.c_str());
  }

  // `dependencies` lists the source and every included file: watch them for hot reload.
  for(size_t i = 0; i < compute.dependencies.size(); i++)
    (void)compute.dependencies[i];

  // Ray-tracing library (entry points are exported by name, no -E) and in-memory source.
  nvdx12::ShaderBlob library = compiler.compileFile("pathtrace.rgen.hlsl", "lib_6_6");
  nvdx12::ShaderBlob inline_ =
      compiler.compileSource("[numthreads(64,1,1)] void main(uint3 id : SV_DispatchThreadID) {}", "inline.hlsl", "cs_6_6");

  // createComputePipeline (compute_pipeline.hpp) and createRayTracingPipeline (raytracing_pipeline.hpp) consume the blobs.
  const nvdx12::ShaderBlob all[] = {compute, library, inline_};
  if(!nvdx12::verifyShaders(all))
    LOGE("Some shaders failed to compile\n");
  LOGI("Shader cache: %u hits, %u misses\n", compiler.cacheHits(), compiler.cacheMisses());

  compiler.deinit();
}
