# FindDXC.cmake
#
# Downloads the DirectX Shader Compiler (the Microsoft.Direct3D.DXC NuGet
# package), or uses the one at DXC_ROOT: dxc.exe for build-time compilation
# and dxcompiler.dll / dxil.dll for nvdx12::ShaderCompiler at run time.
#
# The default version is shared by every sample; update it here so all of
# them move together. Setting DXC_VERSION before finding this package
# downloads that version instead.
set(DXC_VERSION_DEFAULT "1.9.2609.5")

# Sets the following cache variables, refreshed on every configure so that a
# change of DXC_VERSION or DXC_ROOT takes effect:
# DXC_ROOT: Path to the package's build/native directory (bin, include).
# DXC_INCLUDE_DIR: Directory that includes dxcapi.h.
# DXC_EXECUTABLE: dxc.exe, for build-time compilation and validation.
# DXC_COMPILER_DLL, DXC_DXIL_DLL (advanced): dxcompiler.dll and dxil.dll.
# DXC_DLLS (a normal variable): Both DLLs.
#
# nvdx12::ShaderCompiler loads dxcompiler.dll from the executable's directory,
# so samples copy the DLLs next to the executable:
# copy_to_runtime_and_install(...
#   FILES ${DXC_DLLS}
#   ...
# )
# The DXIL compilation of the shared shaders (NVPRO2_NVSHADERS_HOST_DX12) takes
# the directory of the DLLs: get_filename_component(NVPRO2_DXC_DIR "${DXC_EXECUTABLE}" DIRECTORY).

if(NOT DXC_VERSION)
  set(DXC_VERSION ${DXC_VERSION_DEFAULT})
endif()

# Each version downloads into its own folder. A root inside these folders was
# set by this module and follows DXC_VERSION; any other root is the caller's
# and is used as is.
set(_DXC_DOWNLOAD_PREFIX "${CMAKE_BINARY_DIR}/_deps/DXC-")
string(FIND "${DXC_ROOT}" "${_DXC_DOWNLOAD_PREFIX}" _DXC_DOWNLOAD_POS)
if(NOT DXC_ROOT OR _DXC_DOWNLOAD_POS EQUAL 0)
  # A NuGet package is a zip; everything needed is under build/native.
  include(DownloadPackage)
  download_package(
    NAME "DXC"
    URLS "https://www.nuget.org/api/v2/package/Microsoft.Direct3D.DXC/${DXC_VERSION}"
    VERSION ${DXC_VERSION}
    LOCATION _DXC_SOURCE_DIR
  )
  set(DXC_ROOT "${_DXC_SOURCE_DIR}/build/native" CACHE PATH "Path to the DirectX Shader Compiler directory containing `bin` and `include` folders." FORCE)
endif()
mark_as_advanced(DXC_ROOT)

unset(DXC_INCLUDE_DIR CACHE)
find_path(DXC_INCLUDE_DIR
  dxcapi.h
  HINTS "${DXC_ROOT}/include" "${DXC_ROOT}/include/dxc"
  NO_DEFAULT_PATH
  DOC "Directory that includes dxcapi.h from the DirectX Shader Compiler."
)
mark_as_advanced(DXC_INCLUDE_DIR)

set(_DXC_BIN_DIR "${DXC_ROOT}/bin/x64")
unset(DXC_EXECUTABLE CACHE)
find_program(DXC_EXECUTABLE
  NAMES dxc.exe
  HINTS "${_DXC_BIN_DIR}"
  NO_DEFAULT_PATH
  DOC "dxc.exe of the DirectX Shader Compiler."
)
mark_as_advanced(DXC_EXECUTABLE)

unset(DXC_COMPILER_DLL CACHE)
find_file(DXC_COMPILER_DLL
  NAMES dxcompiler.dll
  HINTS "${_DXC_BIN_DIR}"
  NO_DEFAULT_PATH
)
mark_as_advanced(DXC_COMPILER_DLL)
unset(DXC_DXIL_DLL CACHE)
find_file(DXC_DXIL_DLL
  NAMES dxil.dll
  HINTS "${_DXC_BIN_DIR}"
  NO_DEFAULT_PATH
)
mark_as_advanced(DXC_DXIL_DLL)
set(DXC_DLLS ${DXC_COMPILER_DLL} ${DXC_DXIL_DLL})

message(STATUS "--> using DirectX Shader Compiler version ${DXC_VERSION} under: ${DXC_ROOT}")

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(DXC
  REQUIRED_VARS
    DXC_ROOT
    DXC_INCLUDE_DIR
    DXC_EXECUTABLE
    DXC_COMPILER_DLL
    DXC_DXIL_DLL
  VERSION_VAR
    DXC_VERSION
)
