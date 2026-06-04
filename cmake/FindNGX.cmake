# FindNGX.cmake
#
# Downloads the DLSS-RR SDK. Link with the `ngx` target, then copy `DLSS_DLLS`
# to the output directory. In the future we might add more optional components,
# but you'll still only need to link against ngx -- just copy different DLLs
# into the output.
#
# Usage:
# ```cmake
# find_package(NGX REQUIRED)
# target_link_libraries(${PROJECT_NAME} PRIVATE ngx)
# copy_to_runtime_and_install(${PROJECT_NAME} FILES ${DLSS_DLLS})
# ```
#
# Options (all cached):
# - DLSS_ROOT_OVERRIDE: Optional path to a pre-existing DLSS SDK. If set,
#     skips downloading and uses this path instead. Useful for architectures
#     not available in the public SDK (e.g., Linux aarch64).
# - DLSS_VERSION: The SDK version. Defaults to "310.4.0".
#     While you can change this per-project, we recommend changing it in this
#     file instead so that DLSS updates apply to all nvpro-samples uniformly.
# - DLSS_USE_DEVELOP_LIBRARIES: Defaults to OFF.
#     Whether to use the dev/ DLL/SO files from DLSS instead of rel/.
#     The dev/ files have a debug overlay (on Windows, press Ctrl-Alt-F12 on to
#     enable the overlay and Ctrl-Alt-F11 to switch between views), but
#     shouldn't be shipped; they also have debug text and may print additional
#     messages to the console.
# - NGX_USE_STATIC_MSVCRT: Defaults to OFF.
#     Whether to use NGX libraries linked for the static MSVC runtime library.
#     Has no effect on Linux.
#
# Defines the following variables:
# - DLSS_ROOT: Path to the DLSS SDK root directory.
# - DLSS_DLLS: DLL/SO files used by DLSS. Because these are loaded at
#     runtime, you'll need to copy them to the output using
#       copy_to_runtime_and_install(${PROJECT_NAME} FILES ${DLSS_DLLS}) .
# - NGX_INCLUDE_DIR (cached): NGX SDK include directory.
#
# And the following target:
# - ngx: IMPORTED library for NGX.
#     If multiple versions of NGX are pulled, first one wins.

#-------------------------------------------------------------------------------
# Detect platform library subdirectory.
string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" _NGX_ARCH_PROC)
if(_NGX_ARCH_PROC MATCHES "^(arm64|aarch64)$")
  set(_NGX_ARCH "arm64")
  if(WIN32)
    set(_NGX_PLATFORM "Windows_arm64")
  else()
    set(_NGX_PLATFORM "Linux_aarch64")
  endif()
elseif(_NGX_ARCH_PROC MATCHES "^(x86_64|amd64)$")
  set(_NGX_ARCH "x64")
  if(WIN32)
    set(_NGX_PLATFORM "Windows_x86_64")
  else()
    set(_NGX_PLATFORM "Linux_x86_64")
  endif()
else()
  message(FATAL_ERROR "FindNGX: unhandled architecture '${CMAKE_SYSTEM_PROCESSOR}'")
endif()

#-------------------------------------------------------------------------------
# Download the DLSS SDK, or use a user-provided override.
set(DLSS_ROOT_OVERRIDE "" CACHE PATH "Path to pre-existing DLSS SDK (skips download if set).")
set(DLSS_VERSION "310.4.0" CACHE STRING "DLSS version to download.")

if(DLSS_ROOT_OVERRIDE)
  set(DLSS_ROOT ${DLSS_ROOT_OVERRIDE})
  message(STATUS "--> using DLSS-RR under (user-provided): ${DLSS_ROOT}")
else()
  set(_DLSS_URL "https://github.com/NVIDIA/DLSS/archive/refs/tags/v${DLSS_VERSION}.zip")
  include(DownloadPackage)
  download_package(
    NAME DLSSRR
    URLS ${_DLSS_URL}
    VERSION ${DLSS_VERSION}
    LOCATION DLSS_SOURCE_DIR
  )

  set(DLSS_ROOT ${DLSS_SOURCE_DIR}/DLSS-${DLSS_VERSION})
  message(STATUS "--> using DLSS-RR under: ${DLSS_ROOT}")
endif()

# Validate DLSS_ROOT contents. Failures here drive NGX_FOUND=FALSE via the
# find_package_handle_standard_args call below; we leave the diagnostic message
# to that call so REQUIRED/QUIET are honored uniformly.
set(_NGX_VALID TRUE)
if(NOT EXISTS "${DLSS_ROOT}/include")
  message(STATUS "FindNGX: DLSS_ROOT does not contain an include/ directory: ${DLSS_ROOT}")
  set(_NGX_VALID FALSE)
endif()
if(NOT EXISTS "${DLSS_ROOT}/lib/${_NGX_PLATFORM}")
  message(STATUS "FindNGX: DLSS_ROOT does not contain libs for this platform: ${DLSS_ROOT}/lib/${_NGX_PLATFORM}")
  set(_NGX_VALID FALSE)
endif()

if(_NGX_VALID)
  # Collect DLSS DLLs that need to be copied.
  option(DLSS_USE_DEVELOP_LIBRARIES "Use non-distributable DLSS libraries with a debug overlay. On Windows, press Ctrl-Alt-F12 to enable the debug overlay and Ctrl-Alt-F11 to switch views." OFF)
  if(DLSS_USE_DEVELOP_LIBRARIES)
    set(_NGX_DLL_SUBDIR "dev")
  else()
    set(_NGX_DLL_SUBDIR "rel")
  endif()
  set(_NGX_DLL_DIR "${DLSS_ROOT}/lib/${_NGX_PLATFORM}/${_NGX_DLL_SUBDIR}")

  if(WIN32)
    file(GLOB DLSS_DLLS "${_NGX_DLL_DIR}/nvngx_*.dll")
  else()
    file(GLOB DLSS_DLLS "${_NGX_DLL_DIR}/libnvidia-ngx-*.so.*")
  endif()

  if(NOT DLSS_DLLS)
    message(STATUS "FindNGX: no DLSS runtime libraries found in ${_NGX_DLL_DIR}/ (DLSS_USE_DEVELOP_LIBRARIES=${DLSS_USE_DEVELOP_LIBRARIES})")
  endif()

  #-----------------------------------------------------------------------------
  # Set up the NGX target. These are static libraries we link to that load
  # modules like DLSS.
  if(NOT TARGET ngx)
    add_library(ngx IMPORTED STATIC GLOBAL)

    # Map MinSizeRel and RelWithDebInfo to Release.
    set_property(TARGET ngx APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
    set_property(TARGET ngx APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
    set_target_properties(ngx PROPERTIES
      MAP_IMPORTED_CONFIG_MINSIZEREL Release
      MAP_IMPORTED_CONFIG_RELWITHDEBINFO Release
    )

    # STATIC IMPORTED targets only consume IMPORTED_LOCATION_*; IMPORTED_IMPLIB_*
    # is for SHARED targets on Windows (the .lib stub paired with a runtime DLL).
    if (WIN32)
      option(NGX_USE_STATIC_MSVCRT "[Deprecated?]Use NGX libs with static VC runtime (/MT), otherwise dynamic (/MD)" OFF)

      if(NGX_USE_STATIC_MSVCRT)
        set_target_properties(ngx PROPERTIES IMPORTED_LOCATION_DEBUG ${DLSS_ROOT}/lib/${_NGX_PLATFORM}/${_NGX_ARCH}/nvsdk_ngx_s_dbg.lib)
        set_target_properties(ngx PROPERTIES IMPORTED_LOCATION_RELEASE ${DLSS_ROOT}/lib/${_NGX_PLATFORM}/${_NGX_ARCH}/nvsdk_ngx_s.lib)
      else()
        set_target_properties(ngx PROPERTIES IMPORTED_LOCATION_DEBUG ${DLSS_ROOT}/lib/${_NGX_PLATFORM}/${_NGX_ARCH}/nvsdk_ngx_d_dbg.lib)
        set_target_properties(ngx PROPERTIES IMPORTED_LOCATION_RELEASE ${DLSS_ROOT}/lib/${_NGX_PLATFORM}/${_NGX_ARCH}/nvsdk_ngx_d.lib)
      endif()
    else ()
      set_target_properties(ngx PROPERTIES IMPORTED_LOCATION_DEBUG ${DLSS_ROOT}/lib/${_NGX_PLATFORM}/libnvsdk_ngx.a)
      set_target_properties(ngx PROPERTIES IMPORTED_LOCATION_RELEASE ${DLSS_ROOT}/lib/${_NGX_PLATFORM}/libnvsdk_ngx.a)
    endif()

    set(NGX_INCLUDE_DIR "${DLSS_ROOT}/include" CACHE PATH "NGX SDK include directory.")
    mark_as_advanced(NGX_INCLUDE_DIR)
    target_include_directories(ngx INTERFACE "${NGX_INCLUDE_DIR}")
  endif()
else()
  # Make the REQUIRED_VARS check below reflect reality. DLSS_ROOT is left set
  # so it appears in the find_package_handle_standard_args error message.
  set(DLSS_DLLS "")
  unset(NGX_INCLUDE_DIR CACHE)
endif()

include(FindPackageHandleStandardArgs)
# Only assert a version when we downloaded the SDK ourselves. With
# DLSS_ROOT_OVERRIDE there is no reliable way to detect what version actually
# lives at the override path (the SDK ships no version file, the header carries
# an unrelated NGX API version, and only Linux SO filenames carry the SDK
# version), so we omit VERSION_VAR entirely on that path. find_package's
# version-check is skipped when VERSION_VAR is absent, which lets callers using
# find_package(NGX <ver> REQUIRED) keep working under an override; passing
# VERSION_VAR with an empty variable instead would fail those callers.
set(_NGX_FIND_PACKAGE_ARGS REQUIRED_VARS NGX_INCLUDE_DIR DLSS_ROOT DLSS_DLLS)
if(NOT DLSS_ROOT_OVERRIDE)
  list(APPEND _NGX_FIND_PACKAGE_ARGS VERSION_VAR DLSS_VERSION)
endif()
find_package_handle_standard_args(NGX ${_NGX_FIND_PACKAGE_ARGS})
