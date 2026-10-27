# nvshaders_host contains host code for running some shaders in nvshaders/.
# This also sets up some utilities for nvshaders/:
# * NVSHADERS_DIR is the path to nvshaders/ for other scripts
# * nvshaders_files will appear as a folder in Visual Studio.

#-------------------------------------------------------------------------------
set(NVSHADERS_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/.."
    CACHE PATH "Path to the directory that contains the nvshaders directory"
)
mark_as_advanced(NVSHADERS_DIR)

file(GLOB SHADER_FILES "${CMAKE_CURRENT_SOURCE_DIR}/../nvshaders/*")
source_group("Shaders" FILES ${SHADER_FILES})

add_custom_target(nvshaders_files SOURCES ${SHADER_FILES})
set_property(TARGET nvshaders_files PROPERTY FOLDER "nvpro_core2")

#-------------------------------------------------------------------------------
# Collect all source files in the utils directory
file(GLOB LIB_SOURCES "*.cpp" "*.h" "*.hpp")

# The Vulkan hosts need nvvk. The Direct3D 12 hosts (*_dx12.*) need nvdx12 and
# their shaders as DXIL, which slangc produces through a dxcompiler.dll the
# Slang package does not ship: they are opt-in, with the DLL's directory.
option(NVPRO2_NVSHADERS_HOST_DX12 "Build the Direct3D 12 hosts of the shared shaders (needs nvdx12 and NVPRO2_DXC_DIR)" OFF)
set(NVPRO2_DXC_DIR "" CACHE PATH "Directory of dxcompiler.dll and dxil.dll for the DXIL compilation of the shared shaders")
set(_NVSHADERS_HOST_DX12 OFF)
if(NVPRO2_NVSHADERS_HOST_DX12)
  if(NOT WIN32 OR NOT NVPRO2_ENABLE_nvdx12)
    message(FATAL_ERROR "NVPRO2_NVSHADERS_HOST_DX12 needs Windows and NVPRO2_ENABLE_nvdx12")
  endif()
  if(NOT EXISTS "${NVPRO2_DXC_DIR}/dxcompiler.dll")
    message(FATAL_ERROR "NVPRO2_NVSHADERS_HOST_DX12 needs NVPRO2_DXC_DIR, the directory of dxcompiler.dll and dxil.dll "
                        "(with FindDXC: the directory of DXC_EXECUTABLE)")
  endif()
  set(_NVSHADERS_HOST_DX12 ON)
else()
  list(FILTER LIB_SOURCES EXCLUDE REGEX "_dx12\\.(cpp|hpp)$")
endif()
if(NOT NVPRO2_ENABLE_nvvk)
  list(FILTER LIB_SOURCES EXCLUDE REGEX "(tonemapper|sky|hdr_env_dome|pbr_sheen_lut)\\.(cpp|hpp)$")
endif()
if(NOT LIB_SOURCES)
  message(FATAL_ERROR "nvshaders_host has no hosts to build: enable NVPRO2_ENABLE_nvvk or NVPRO2_NVSHADERS_HOST_DX12, or disable NVPRO2_ENABLE_nvshaders_host")
endif()
source_group("Source Files" FILES ${LIB_SOURCES})

# Define the utils library
add_library(nvshaders_host STATIC ${LIB_SOURCES})

# All libraries use nvpro_core2 as their include root
cmake_path(GET CMAKE_CURRENT_LIST_DIR PARENT_PATH PARENT_DIR)
target_include_directories(nvshaders_host PUBLIC ${PARENT_DIR})

target_link_libraries(nvshaders_host PUBLIC nvutils glm)
if(NVPRO2_ENABLE_nvvk)
  target_link_libraries(nvshaders_host PUBLIC nvvk)
endif()

if(_NVSHADERS_HOST_DX12)
  # The shared shaders compiled to DXIL and embedded in the Direct3D 12 hosts.
  # compile_slang is included here because Setup.cmake adds it after the libraries.
  find_package(Slang)
  include("${CMAKE_CURRENT_LIST_DIR}/../cmake/CompilerSlangShader.cmake")
  set(_DX12_AUTOGEN "${CMAKE_CURRENT_BINARY_DIR}/_autogen")
  set(_DX12_INCLUDES -I${NVSHADERS_DIR} -I${NVSHADERS_DIR}/nvshaders)
  # The entry points are named: without them slangc compiles DXIL as a library,
  # which compute pipelines reject.
  compile_slang(
    "${NVSHADERS_DIR}/nvshaders/sky_simple.slang;${NVSHADERS_DIR}/nvshaders/sky_physical.slang"
    "${_DX12_AUTOGEN}"
    HEADERS_VAR _DX12_SKY_HEADERS
    TARGET dxil
    DXC_DIR "${NVPRO2_DXC_DIR}"
    ENTRY_POINTS main
    EXTRA_FLAGS ${_DX12_INCLUDES}
  )
  # One blob per compute entry point. Slang 2026.13 emits HLSL fma(), which dxc
  # accepts for doubles only; mad() is its float form.
  compile_slang(
    "${NVSHADERS_DIR}/nvshaders/tonemapper.slang"
    "${_DX12_AUTOGEN}"
    HEADERS_VAR _DX12_TONEMAPPER_HEADERS
    TARGET dxil
    DXC_DIR "${NVPRO2_DXC_DIR}"
    ENTRY_POINTS Tonemap Histogram AutoExposure
    EXTRA_FLAGS ${_DX12_INCLUDES} -Dfma=mad
  )
  target_sources(nvshaders_host PRIVATE ${_DX12_SKY_HEADERS} ${_DX12_TONEMAPPER_HEADERS})
  target_include_directories(nvshaders_host PRIVATE "${_DX12_AUTOGEN}")
  target_link_libraries(nvshaders_host PUBLIC nvdx12)
endif()

target_precompile_headers(nvshaders_host PRIVATE
  <filesystem>
)

add_library(nvpro2::nvshaders_host ALIAS nvshaders_host)
set_property(TARGET nvshaders_host PROPERTY FOLDER "nvpro_core2")
