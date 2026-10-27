# FindDX12AgilitySdk.cmake
#
# Downloads the DirectX 12 Agility SDK, or uses the one at DX12AgilitySdk_ROOT.
#
# The default version is shared by every sample; update it here so all of
# them move together. Setting DX12AgilitySdk_VERSION before finding this
# package downloads that version instead. nvdx12 compiles against a single
# SDK, so one version is in use per build.
set(DX12AgilitySdk_VERSION_DEFAULT "1.721.3-preview")

# Sets the following cache variables, refreshed on every configure so that a
# change of DX12AgilitySdk_VERSION or DX12AgilitySdk_ROOT takes effect:
# DX12AgilitySdk_ROOT: Path to the SDK root directory.
# DX12AgilitySdk_INCLUDE_DIR: Directory that includes d3d12.h.
# DX12AgilitySdk_LIBRARIES: Linker library/libraries.
# DX12AgilitySdk_DLLS: D3D12Core.dll, and d3d12SDKLayers.dll when the SDK has it.
# DX12AgilitySdk_CORE_DLL, DX12AgilitySdk_LAYERS_DLL (advanced): Each of the two DLLs.
# DX12AgilitySdk_SDK_VERSION: D3D12SDKVersion as used in https://devblogs.microsoft.com/directx/gettingstarted-dx12agility/#howto
#   Derived from DX12AgilitySdk_VERSION for a downloaded SDK. With your own
#   DX12AgilitySdk_ROOT, set it to that SDK's D3D12SDKVersion: d3d12.h cannot
#   tell, since retail headers also declare a D3D12_PREVIEW_SDK_VERSION.
#
# The runtime is loaded from the directory the executable exports as
# D3D12SDKPath; nvdx12 exports ".\D3D12\" (NVDX12_EXPORT_AGILITY_SDK), so
# samples copy the DLLs into that subdirectory:
# copy_to_runtime_and_install(...
#   FILES ${DX12AgilitySdk_DLLS}
#   SUBDIRECTORY "D3D12"
#   ...
# )
# With the DLLs anywhere else, device creation fails with
# D3D12_ERROR_INVALID_HOST_EXE_SDK_VERSION; nvdx12::Context::init names the fix.

if(NOT DX12AgilitySdk_VERSION)
  set(DX12AgilitySdk_VERSION ${DX12AgilitySdk_VERSION_DEFAULT})
endif()

# Each version downloads into its own folder. A root inside these folders was
# set by this module and follows DX12AgilitySdk_VERSION; any other root is the
# caller's and is used as is.
set(_DX12AgilitySdk_DOWNLOAD_PREFIX "${CMAKE_BINARY_DIR}/_deps/DX12AgilitySdk-")
string(FIND "${DX12AgilitySdk_ROOT}" "${_DX12AgilitySdk_DOWNLOAD_PREFIX}" _DX12AgilitySdk_DOWNLOAD_POS)
if(NOT DX12AgilitySdk_ROOT OR _DX12AgilitySdk_DOWNLOAD_POS EQUAL 0)
  # Note that here we download a NuGet package and extract it!
  # This works because .nuget packages are really .zip files, and everything we
  # need is in the /build/native subdirectory.
  include(DownloadPackage)
  download_package(
    NAME "DX12AgilitySdk"
    URLS "https://www.nuget.org/api/v2/package/Microsoft.Direct3D.D3D12/${DX12AgilitySdk_VERSION}"
    VERSION ${DX12AgilitySdk_VERSION}
    LOCATION _DX12AgilitySdk_SOURCE_DIR
  )
  set(DX12AgilitySdk_ROOT "${_DX12AgilitySdk_SOURCE_DIR}/build/native" CACHE PATH "Path to the DirectX Agility SDK directory containing `bin`, `include`, and `src` folders." FORCE)

  # NuGet versions are 1.<D3D12SDKVersion>.<patch>[-preview].
  string(REGEX MATCH "^[0-9]+\\.([0-9]+)" _DX12AgilitySdk_MATCH "${DX12AgilitySdk_VERSION}")
  set(DX12AgilitySdk_SDK_VERSION "${CMAKE_MATCH_1}" CACHE STRING "D3D12SDKVersion, as used in https://devblogs.microsoft.com/directx/gettingstarted-dx12agility/#howto" FORCE)
  set(_DX12AgilitySdk_DERIVED_SDK_VERSION "${CMAKE_MATCH_1}" CACHE INTERNAL "")
elseif(_DX12AgilitySdk_DERIVED_SDK_VERSION AND DX12AgilitySdk_SDK_VERSION STREQUAL _DX12AgilitySdk_DERIVED_SDK_VERSION)
  # The caller's root replaced a download: forget the download's version, the caller sets its own.
  unset(DX12AgilitySdk_SDK_VERSION CACHE)
  unset(_DX12AgilitySdk_DERIVED_SDK_VERSION CACHE)
endif()
mark_as_advanced(DX12AgilitySdk_ROOT)

# NuGet packages keep the headers under include/ and the binaries under
# bin/x64; preview drops are flat, with the binaries under runtime/.
unset(DX12AgilitySdk_INCLUDE_DIR CACHE)
find_path(DX12AgilitySdk_INCLUDE_DIR
  d3d12.h
  HINTS "${DX12AgilitySdk_ROOT}/include" "${DX12AgilitySdk_ROOT}"
  NO_DEFAULT_PATH
  DOC "Directory that includes d3d12.h from the DX12 Agility SDK."
)
mark_as_advanced(DX12AgilitySdk_INCLUDE_DIR)

# The linker libraries for the Agility SDK are the regular D3D12 libraries.
# These are a bit unusual because Windows' linker finds them automatically.
set(DX12AgilitySdk_LIBRARIES d3d12.lib dxgi.lib CACHE STRING "The D3D12 linker library; the linker finds this automatically.")
mark_as_advanced(DX12AgilitySdk_LIBRARIES)

# DLLs: the runtime, and the debug layer when the SDK has it.
set(_DX12AgilitySdk_SEARCH_DIRS "${DX12AgilitySdk_ROOT}/bin/x64" "${DX12AgilitySdk_ROOT}/runtime" "${DX12AgilitySdk_ROOT}/runtime/D3D12" "${DX12AgilitySdk_ROOT}/bin")
unset(DX12AgilitySdk_CORE_DLL CACHE)
find_file(DX12AgilitySdk_CORE_DLL
  NAMES D3D12Core.dll
  HINTS ${_DX12AgilitySdk_SEARCH_DIRS}
  NO_DEFAULT_PATH
)
mark_as_advanced(DX12AgilitySdk_CORE_DLL)

unset(DX12AgilitySdk_LAYERS_DLL CACHE)
find_file(DX12AgilitySdk_LAYERS_DLL
  NAMES d3d12SDKLayers.dll
  HINTS ${_DX12AgilitySdk_SEARCH_DIRS}
  NO_DEFAULT_PATH
)
mark_as_advanced(DX12AgilitySdk_LAYERS_DLL)

set(_DX12AgilitySdk_FOUND_DLLS)
foreach(_DX12AgilitySdk_DLL IN ITEMS DX12AgilitySdk_CORE_DLL DX12AgilitySdk_LAYERS_DLL)
  if(${_DX12AgilitySdk_DLL})
    list(APPEND _DX12AgilitySdk_FOUND_DLLS "${${_DX12AgilitySdk_DLL}}")
  endif()
endforeach()
set(DX12AgilitySdk_DLLS ${_DX12AgilitySdk_FOUND_DLLS} CACHE STRING "D3D12 Agility SDK DLLs" FORCE)

message(STATUS "--> using DX12 Agility SDK (D3D12SDKVersion ${DX12AgilitySdk_SDK_VERSION}) under: ${DX12AgilitySdk_ROOT}")

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(DX12AgilitySdk
  REQUIRED_VARS
    DX12AgilitySdk_ROOT
    DX12AgilitySdk_LIBRARIES
    DX12AgilitySdk_INCLUDE_DIR
    DX12AgilitySdk_CORE_DLL
  VERSION_VAR
    DX12AgilitySdk_VERSION
)
