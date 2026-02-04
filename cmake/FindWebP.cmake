# FindWebP.cmake
#
# Downloads libwebp, Google's library for reading and writing WebP images.
# To use it, call this script and then link against the webp target:
# ```cmake
# find_package(WebP REQUIRED)
# target_link_libraries(${PROJECT_NAME} PRIVATE webp)
# ```
#
# We download libwebp instead of including it as a filtered package because
# it is a relatively large dependency and few samples use it.
# In addition, we set various CMake options by default, much like
# nvpro_core2/third_party/SetKnownToolchainProperties.cmake, to speed up
# initial CMake configuration on systems we recognize.
#
# Option(s) (all cached):
# - WebP_VERSION: The SDK version. Defaults to "1.6.0".
#     While you can change this per-project, we recommend changing it in this
#     file since if WebP is added multiple times, the first declaration wins.
#
# Defines the following variable:
# - WebP_ROOT: Path to the libwebp root directory.
#
# And the following target:
# - webp: Static library for webp.
#      If multiple versions of webp are pulled, first one wins.

#-------------------------------------------------------------------------------
# Download libwebp.
if(NOT WebP_VERSION)
  set(WebP_VERSION "1.6.0" CACHE STRING "LibWebP version to download.")
endif()
set(_WebP_URL "https://github.com/webmproject/libwebp/archive/refs/tags/v${WebP_VERSION}.zip")
include(DownloadPackage)
download_package(
  NAME libwebp
  URLS ${_WebP_URL}
  VERSION ${WebP_VERSION}
  LOCATION WebP_SOURCE_DIR
)

set(WebP_ROOT ${WebP_SOURCE_DIR}/libwebp-${WebP_VERSION})
message(STATUS "--> using LibWebP under: ${WebP_ROOT}")

#-------------------------------------------------------------------------------
# Set up the webp target.
if(NOT TARGET webp)
  # Turn off libraries we don't need.
  option(WEBP_BUILD_ANIM_UTILS "Build animation utilities." OFF)
  option(WEBP_BUILD_CWEBP "Build the cwebp command line tool." OFF)
  option(WEBP_BUILD_DWEBP "Build the dwebp command line tool." OFF)
  option(WEBP_BUILD_GIF2WEBP "Build the gif2webp conversion tool." OFF)
  option(WEBP_BUILD_IMG2WEBP "Build the img2webp animation tool." OFF)
  option(WEBP_BUILD_VWEBP "Build the vwebp viewer tool." OFF)
  option(WEBP_BUILD_WEBPINFO "Build the webpinfo command line tool." OFF)
  option(WEBP_BUILD_LIBWEBPMUX "Build the webpmux library." OFF)
  option(WEBP_BUILD_WEBPMUX "Build the webpmux command line tool." OFF)
  option(WEBP_BUILD_EXTRAS "Build extras." OFF)

  # Set whether features are available if we recognize the system.
  # WebP does these tests in cmake/cpu.cmake, by the way.
  set(NVPRO2_WEBP_SKIP_COMPILER_FEATURE_DETECTION_DEFAULT OFF)
  set(_IS_X86 OFF)
  set(_IS_AARCH64 OFF)
  if("${CMAKE_SYSTEM_PROCESSOR}" MATCHES "(x86_64|AMD64|aarch64)") # Is x86_64 or aarch64?
    if(WIN32 OR UNIX) # One of the OSes we know?
      set(NVPRO2_WEBP_SKIP_COMPILER_FEATURE_DETECTION_DEFAULT ON)
	  if("${CMAKE_SYSTEM_PROCESSOR}" MATCHES "(x86_64|AMD64)")
		set(_IS_X86 ON)
	  elseif("${CMAKE_SYSTEM_PROCESSOR}" MATCHES "aarch64")
	    set(_IS_AARCH64 ON)
	  endif()
    endif()
  endif()
  option(NVPRO2_WEBP_SKIP_COMPILER_FEATURE_DETECTION
         "Assume minimum compiler features exist instead of running test compiles"
         ${NVPRO2_WEBP_SKIP_COMPILER_FEATURE_DETECTION_DEFAULT}
  )
  if(NVPRO2_WEBP_SKIP_COMPILER_FEATURE_DETECTION)
    set(WEBP_HAVE_FLAG_AVX2 ${_IS_X86})
	set(WEBP_HAVE_FLAG_SSE41 ${_IS_X86})
	set(WEBP_HAVE_FLAG_SSE2 ${_IS_X86})
	set(WEBP_HAVE_FLAG_MIPS32 OFF)
	set(WEBP_HAVE_FLAG_MIPS_DSP_R2 OFF)
	set(WEBP_HAVE_FLAG_NEON ${_IS_AARCH64})
    set(WEBP_HAVE_FLAG_MSA OFF)

	# On MSVC we also need to tell libwebp that we really can compile with these SIMD architectures.
	# libwebp checks if the CPU supports these features before executing code with them, so this is OK even if target CPUs only have SSE4.1.
	if(MSVC)
	  if(${_IS_X86})
        add_definitions(-D__SSE2__ -D__SSE4_1__ -D__AVX2__)
	  else()
	    add_definitions(-D__ARM_NEON__)
	  endif()
	endif()
  endif()

  # LibWebP tries to only build in single-configuration mode if CMAKE_BUILD_TYPE isn't defined. Try to prevent this.
  set(_PREVIOUS_CMAKE_BUILD_TYPE ${CMAKE_BUILD_TYPE})
  set(CMAKE_BUILD_TYPE "<any string works here to prevent libwebp overriding CMAKE_BUILD_TYPE>")
  add_subdirectory(${WebP_ROOT} EXCLUDE_FROM_ALL)
  set(CMAKE_BUILD_TYPE ${_PREVIOUS_CMAKE_BUILD_TYPE})

  # In VS, sort webp and the targets it defines under the ThirdParty folder.
  foreach(_LIB IN ITEMS webp sharpyuv webpdecode webpdecoder webpdemux webpdsp webpdspdecode webpencode webputils webputilsdecode)
    set_property(TARGET ${_LIB} PROPERTY FOLDER "ThirdParty")
  endforeach()
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(WebP
  REQUIRED_VARS
    WebP_ROOT
  VERSION_VAR WebP_VERSION
)