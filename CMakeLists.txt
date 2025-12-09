# This CMake file only creates targets for the nvpro_core libraries.
# Samples using nvpro_core will usually want to include cmake/Setup.cmake instead,
# which sets properties most samples will need and includes all helper scripts.
#
# After you've added nvpro_core, get any packages and link with any libraries
# you need.
#
# Here's an example.
set(_NVPRO_CORE_CMAKE_USAGE
    [=[
# Look for nvpro_core as a sub-folder and in other locations.
find_path(NVPRO_CORE_DIR
  NAMES cmake/Setup.cmake
  PATHS ${CMAKE_CURRENT_LIST_DIR}/nvpro_core2
        ${CMAKE_CURRENT_LIST_DIR}/../nvpro_core2
        ${CMAKE_CURRENT_LIST_DIR}/../../nvpro_core2
  REQUIRED
  DOC "Path to nvpro_core"
)
include(${NVPRO_CORE_DIR}/cmake/Setup.cmake)

# Add the sample:
add_executable(${PROJECT_NAME} main.cpp)

# If you need to set the PROJECT_NAME, PROJECT_EXE_TO_SOURCE_DIRECTORY, or NVSHADERS_DIR
# C++ macro definitions, use this (from cmake/Utilities.cmake):
add_project_definitions(${PROJECT_NAME})

# Compile offline shaders using compile_slang (from cmake/CompilerSlangShader.cmake)
set(_SHADER_OUTPUT_DIR "${CMAKE_BINARY_DIR}/_autogen")
file(GLOB _SHADER_SLANG_FILES "shaders/*.slang")
file(GLOB _SHADER_H_FILES "shaders/*.h")
compile_slang(
  ${_SHADER_SLANG_FILES}
  ${_SHADER_OUTPUT_DIR}
  _GENERATED_SHADER_HEADERS
  EXTRA_FLAGS "-I${NVSHADERS_DIR}" # If your shaders import from nvshaders/
)
source_group("Shaders" FILES ${_SHADER_SLANG_FILES} ${_SHADER_H_FILES})
source_group("Shaders/Compiled" FILES ${_GENERATED_SHADER_HEADERS})

# Link with some libraries - you won't always need all of these:
target_link_libraries(${PROJECT_NAME} PRIVATE
  nvpro2::nvapp
  nvpro2::nvutils
  nvpro2::nvvk
  nvpro2::nvgui
  nvpro2::nvshaders_host
  tinygltf
)

# Use copy_to_runtime_and_install (from cmake/Utilities.cmake) to copy files
# needed to the output, whether building a sample by itself or building the
# INSTALL target:
copy_to_runtime_and_install(${PROJECT_NAME}
  DIRECTORIES "${CMAKE_CURRENT_LIST_DIR}/shaders"
              "${NVSHADERS_DIR}"
  AUTO
)

]=]
) # End example

#-------------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.22...3.31)
project(
  nvpro_core2
  VERSION 1.0
  LANGUAGES CXX
)

# For this and subdirectories, make find_package find scripts in the `cmake/` directory:
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")

# Test if the local nvpro_core2 is up to date with the remote.
include(CheckLatest)

# And enable C++20:
set(CMAKE_CXX_STANDARD 20)

if(MSVC)
  # Enable parallel builds for this directory and subdirectories:
  add_compile_options(/MP)
else()
  # Place each function in its own section. This allows -Wl,--gc-sections in
  # Setup.cmake to remove functions that aren't used.
  add_compile_options(-ffunction-sections)
endif()

# No target in this or a subdirectory currently uses C++ modules.
# Setting this to OFF tells Visual Studio to skip scanning for modules at the
# start of every build. This saves about 5-10 seconds of build time as of 2025-03-24.
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)

find_package(VulkanExtra REQUIRED)

# We might compile with Clang on Windows, but while '_WIN32' is defined by the compiler,
# we mostly use 'WIN32', which apparently will not be defined by default under Clang
if(WIN32 AND NOT MSVC)
  add_compile_definitions(WIN32)
endif()

if(WIN32)
  add_compile_definitions(_CRT_SECURE_NO_WARNINGS=1)
endif()

if((CMAKE_CXX_COMPILER_ID MATCHES "Clang") AND MSVC)
  # VMA throws tons of warnings in the vk_mem_alloc.h header
  add_compile_options(-Wno-nullability-completeness)
endif()

# Set up targets for all filtered packages first:
add_subdirectory(third_party)

# Enable MSVC warning level 3 for this directory and subdirectories.
# We don't set this above in case third-party projects emit warnings.
if(MSVC)
  add_compile_options(/W3)
endif()

# Then set up targets for all nvpro_core libraries.
# These are in a specific order because of interdependencies.
foreach(_LIBRARY IN ITEMS nvimageformats nvutils nvapp nvgui nvgpu_monitor nvvk nvgl nvshaders_host nvaftermath nvslang nvvkglsl nvvkgltf nvnsight)
  option(NVPRO2_ENABLE_${_LIBRARY} "Enable ${_LIBRARY}" ON)
  
  if(NVPRO2_ENABLE_${_LIBRARY})
    add_subdirectory(${_LIBRARY})

    # If nvpro_core is being built by itself -- not included by a sample --
    # set up an INSTALL target that copies libraries to the output.
    # TODO: Set up an EXPORT library that also copies headers to the output.
    if(PROJECT_IS_TOP_LEVEL)
      if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
        set(CMAKE_INSTALL_PREFIX "${CMAKE_SOURCE_DIR}/_install" CACHE PATH [=[
Location for nvpro_core2's INSTALL target to write its files.
It's usually easier to call add_subdirectory(nvpro_core2) instead of using this.
]=] FORCE)
      endif()
      install(TARGETS ${_LIBRARY})
    endif()
  endif()
endforeach()

# Custom target to see files in Solution
file(GLOB CMAKE_FILES CMakeLists.txt cmake/*.cmake)
add_custom_target(NvPro2CMakeFiles SOURCES ${CMAKE_FILES})
set_property(TARGET NvPro2CMakeFiles PROPERTY FOLDER "nvpro_core2/cmake")
# Automatically create folders based on file structure
source_group(TREE ${CMAKE_CURRENT_LIST_DIR} FILES ${CMAKE_FILES})

# Uncomment to build the template sample - This is not needed for the library
# add_subdirectory(project_template)
