# This file is the default CMake entry point for samples;
# it sets useful defaults, adds nvpro_core, and automatically includes helpers.

# Default to C++20.
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Turn on support for folders.
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

if(MSVC)
  # Enable parallel builds, and set warning level 3 for MSVC.
  add_compile_options($<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:/MP>)
  add_compile_options($<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:/W3>)
else()
  # Remove unused sections to save space (and remove usage_* doc functions).
  # You can add -Wl,--print-gc-sections to see what was removed.
  add_link_options(-Wl,--gc-sections)
endif()

# We might compile with Clang on Windows, but while '_WIN32' is defined by the compiler,
# we mostly use 'WIN32', which apparently will not be defined by default under Clang
if(WIN32 AND NOT MSVC)
  add_compile_definitions(WIN32)
endif()

if((CMAKE_CXX_COMPILER_ID MATCHES "Clang") AND MSVC)
  # VMA throws tons of warnings in the vk_mem_alloc.h header
  add_compile_options(-Wno-nullability-completeness)
endif()

# This saves some time scanning source files for imported C++ modules.
# If your sample uses modules, you must explicitly turn on this property on your
# sample, or set this variable to ON before including Setup.cmake.
if(NOT DEFINED CMAKE_CXX_SCAN_FOR_MODULES)
  set(CMAKE_CXX_SCAN_FOR_MODULES OFF)
endif()

# Make find_package find scripts in this directory.
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}")

# Built samples will be written to ${NVPRO_CORE2_OUTPUT_DIR}/_bin, and
# packaged/installed samples will be written to
# ${NVPRO_CORE2_OUTPUT_DIR}/_include.
# Samples should leave this at its default for consistency, but metaprojects
# like build_all can set it to custom values.
if(NOT NVPRO_CORE2_OUTPUT_DIR)
  set(NVPRO_CORE2_OUTPUT_DIR ${CMAKE_SOURCE_DIR})
endif()
if(NOT NVPRO_CORE2_DOWNLOAD_DIR)
  set(NVPRO_CORE2_DOWNLOAD_DIR ${CMAKE_SOURCE_DIR}/_downloaded_resources)
endif()
if (NOT NVPRO_CORE2_DOWNLOAD_SITE)
  set(NVPRO_CORE2_DOWNLOAD_SITE http://developer.download.nvidia.com/ProGraphics/nvpro-samples)
endif()

if(NOT EXISTS ${NVPRO_CORE2_DOWNLOAD_DIR})
  file(MAKE_DIRECTORY ${NVPRO_CORE2_DOWNLOAD_DIR})
endif()

# Set a default installation prefix.
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX "${NVPRO_CORE2_OUTPUT_DIR}/_install" CACHE PATH "Default install path" FORCE)
  # Tell subprojects that we've set the install prefix to a non-default value now
  set(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT OFF)
endif()

# The INSTALL target should generate portable executables. On Linux, this means
# we want to package some libraries (like Slang) alongside the executable, and
# use an RPATH that lets us find them.
# This interacts with COPY_AUTO in Utilities.cmake.
if(NOT DEFINED CMAKE_INSTALL_RPATH)
  set(CMAKE_INSTALL_RPATH "$ORIGIN")
endif()

# Set the output directory for executables
if(GENERATOR_IS_MULTI_CONFIG)
  # CMake appends the build config for us
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${NVPRO_CORE2_OUTPUT_DIR}/_bin)
else()
  # Manually append config, e.g. for Linux Makefiles
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${NVPRO_CORE2_OUTPUT_DIR}/_bin/${CMAKE_BUILD_TYPE})
endif()

# Add nvpro_core.
# Since multiple samples can include Setup.cmake, avoid doing this if we've
# already added it.
if(NOT TARGET nvpro2::nvvk)
  add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/.. ${CMAKE_CURRENT_BINARY_DIR}/nvpro_core2)
endif()

# Include custom modules so that samples using this don't need to explicitly include them.
include(DownloadPackage)
include(CompilerGlslShader)
include(CompilerSlangShader)
include(Utilities)

