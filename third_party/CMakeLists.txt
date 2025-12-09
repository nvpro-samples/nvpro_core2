# This CMake file sets up targets for all third-party libraries included within
# this folder.
project(third_party LANGUAGES C CXX)

# We want static libraries for all packages defined here.
set(BUILD_SHARED_LIBS OFF)

# Speed up CMake configuration by setting some properties if we recognize the toolchain.
include(SetKnownToolchainProperties.cmake)

# dxh
if(NOT TARGET dxh)
  set(_DXH_DIR ${CMAKE_CURRENT_SOURCE_DIR}/dxh)
  file(GLOB_RECURSE _DXH_FILES "${_DXH_DIR}/include/*")
  add_library(dxh INTERFACE ${_DXH_FILES})
  target_include_directories(dxh INTERFACE ${_DXH_DIR}/include)
  set_property(TARGET dxh PROPERTY FOLDER "ThirdParty")
endif()

# GLFW
if(NOT TARGET glfw)
  set(GLFW_USE_HYBRID_HPG ON)
  set(GLFW_VULKAN_STATIC OFF)
  set(GLFW_BUILD_DOCS OFF)
  set(GLFW_BUILD_EXAMPLES OFF)
  set(GLFW_BUILD_TESTS OFF)
  set(GLFW_BUILD_WAYLAND OFF)
  set(GLFW_INSTALL OFF)
  add_subdirectory(glfw)
  set_target_properties(glfw update_mappings PROPERTIES FOLDER "ThirdParty/GLFW")
endif()

# volk
if(NOT TARGET volk)
  if(Vulkan_VOLK_DIR)
    set(VOLK_DIR ${Vulkan_VOLK_DIR})
    message(STATUS "Using Vulkan::volk target from Vulkan package")
  else()
    set(VOLK_DIR ${CMAKE_CURRENT_LIST_DIR}/volk)
    message(STATUS "Using fallback volk version")
  endif()
  add_library(volk STATIC ${VOLK_DIR}/volk.h ${VOLK_DIR}/volk.c)
  add_library(volk::volk ALIAS volk)
  target_include_directories(volk PUBLIC ${VOLK_DIR} ${Vulkan_INCLUDE_DIRS})
  # We set these compile options on volk so that it sees platform-specific
  # extension macros in vulkan.h.
  if(WIN32)
    target_compile_definitions(volk PUBLIC VK_USE_PLATFORM_WIN32_KHR NOMINMAX)
  else()
    target_link_libraries(volk PUBLIC ${CMAKE_DL_LIBS})
    target_compile_definitions(
      volk PUBLIC VK_USE_PLATFORM_XCB_KHR VK_USE_PLATFORM_XLIB_KHR
    )
  endif()
  # Define VK_NO_PROTOTYPES to avoid including Vulkan prototypes
  # This is necessary because we are using volk to load Vulkan functions
  target_compile_definitions(volk PUBLIC VK_NO_PROTOTYPES)

  set_property(TARGET volk PROPERTY FOLDER "ThirdParty")
endif()

# imgui
set(IMGUI_DIR "${CMAKE_CURRENT_SOURCE_DIR}/imgui")
if(NOT TARGET imgui)
  set(IMGUI_FILES
      "${IMGUI_DIR}/imgui.cpp"
      "${IMGUI_DIR}/imgui_demo.cpp"
      "${IMGUI_DIR}/imgui_draw.cpp"
      "${IMGUI_DIR}/imgui_tables.cpp"
      "${IMGUI_DIR}/imgui_widgets.cpp"
      "${IMGUI_DIR}/imconfig.h"
      "${IMGUI_DIR}/imgui.h"
      "${IMGUI_DIR}/imgui_internal.h"
      "${IMGUI_DIR}/imstb_rectpack.h"
      "${IMGUI_DIR}/imstb_textedit.h"
      "${IMGUI_DIR}/imstb_truetype.h"
      # For GLFW, Vulkan, and Volk:
      "${IMGUI_DIR}/backends/imgui_impl_glfw.h"
      "${IMGUI_DIR}/backends/imgui_impl_glfw.cpp"
      "${IMGUI_DIR}/backends/imgui_impl_vulkan.h"
      "${IMGUI_DIR}/backends/imgui_impl_vulkan.cpp"
      "${IMGUI_DIR}/backends/imgui_impl_opengl3_loader.h"
      "${IMGUI_DIR}/backends/imgui_impl_opengl3.h"
      "${IMGUI_DIR}/backends/imgui_impl_opengl3.cpp"
  )

  add_library(imgui STATIC ${IMGUI_FILES})
  target_include_directories(imgui PUBLIC 
    ${IMGUI_DIR} # for those, such as implot, including as <imgui.h>
    ${CMAKE_CURRENT_SOURCE_DIR}) # for those including as <imgui/imgui.h>

  # Link against GLFW and Volk
  target_link_libraries(imgui PUBLIC glfw volk)
  # Mandatory for Imgui to work with Volk
  target_compile_definitions(imgui PRIVATE IMGUI_IMPL_VULKAN_USE_VOLK)

  set_property(TARGET imgui PROPERTY FOLDER "ThirdParty")
endif()

# implot
if(NOT TARGET implot)
  set(IMPLOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/implot")
  set(IMPLOT_FILES
      "${IMPLOT_DIR}/implot.cpp" "${IMPLOT_DIR}/implot.h"
      "${IMPLOT_DIR}/implot_demo.cpp" "${IMPLOT_DIR}/implot_internal.h"
      "${IMPLOT_DIR}/implot_items.cpp"
  )

  add_library(implot STATIC ${IMPLOT_FILES})
  target_include_directories(implot PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
  # This reduces the binary size of ImPlot by about 75%.
  target_compile_definitions(implot PRIVATE
    "-DIMPLOT_NO_FORCE_INLINE=1"
    "-DIMPLOT_CUSTOM_NUMERIC_TYPES=(ImU32)(ImU64)(float)"
  )
  target_link_libraries(implot PRIVATE imgui)
  set_property(TARGET implot PROPERTY FOLDER "ThirdParty")
endif()

# glm
if(NOT TARGET glm)
  set(GLM_DIR ${CMAKE_CURRENT_SOURCE_DIR}/glm)
  add_library(glm INTERFACE ${GLM_DIR}/glm/glm.hpp)
  target_include_directories(glm INTERFACE ${GLM_DIR})
  target_compile_definitions(
    glm INTERFACE GLM_FORCE_RADIANS GLM_ENABLE_EXPERIMENTAL GLM_FORCE_XYZW_ONLY
  )
  set_property(TARGET glm PROPERTY FOLDER "ThirdParty")
endif()

# stb
if(NOT TARGET stb)
  set(STB_DIR ${CMAKE_CURRENT_SOURCE_DIR}/stb)
  add_library(stb INTERFACE ${STB_DIR}/stb_image.h ${STB_DIR}/stb_image_write.h)
  target_include_directories(stb INTERFACE "${STB_DIR}")
  set_property(TARGET stb PROPERTY FOLDER "ThirdParty")
endif()

# tinygltf
if(NOT TARGET tinygltf)
  set(TINYGLTF_DIR ${CMAKE_CURRENT_SOURCE_DIR}/tinygltf)
  add_library(
    tinygltf INTERFACE ${TINYGLTF_DIR}/tiny_gltf.h ${TINYGLTF_DIR}/json.hpp
  )
  target_include_directories(
    tinygltf INTERFACE ${CMAKE_CURRENT_SOURCE_DIR} ${STB_DIR}
  )
  set_property(TARGET tinygltf PROPERTY FOLDER "ThirdParty")
endif()

#fmt
if(NOT TARGET fmt)
  add_library(fmt STATIC fmt/src/format.cc)
  target_include_directories(fmt PUBLIC fmt/include)
  set_property(TARGET fmt PROPERTY FOLDER "ThirdParty")
  # On MSVC, fmt requires either FMT_UNICODE to be defined as 0 or to enable
  # UTF-8 for source and execution character sets; see
  # https://github.com/fmtlib/fmt/commit/077e4ae746cde56559d72babe278e3d189bb2b3a.
  if(MSVC)
    target_compile_options(fmt PUBLIC $<$<COMPILE_LANGUAGE:CXX>:/utf-8>)
  endif()
endif()

# tinyobjloader
if(NOT TARGET tinyobjloader)
  set(TINYOBJLOADER_DIR ${CMAKE_CURRENT_SOURCE_DIR}/tinyobjloader)
  add_library(tinyobjloader INTERFACE ${TINYOBJLOADER_DIR}/tiny_obj_loader.h)
  target_include_directories(
    tinyobjloader INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}
  )
  set_property(TARGET tinyobjloader PROPERTY FOLDER "ThirdParty")
endif()

# vma
if(NOT TARGET vma AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/vma)
  set(VMA_DIR ${CMAKE_CURRENT_SOURCE_DIR}/vma/include)
  add_library(vma INTERFACE ${VMA_DIR}/vk_mem_alloc.h)
  target_include_directories(vma INTERFACE ${VMA_DIR})
  set_property(TARGET vma PROPERTY FOLDER "ThirdParty")
endif()

# thread_pool
if(NOT TARGET thread_pool)
  add_library(thread_pool INTERFACE)
  target_include_directories(
    thread_pool INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/thread_pool/include
  )
endif()

# offsetallocator
if(NOT TARGET offsetAllocator)
  set(OFFSETALLOCATOR_DIR ${CMAKE_CURRENT_SOURCE_DIR}/offsetallocator)
  add_library(
    offsetAllocator STATIC ${OFFSETALLOCATOR_DIR}/offsetAllocator.cpp
                           ${OFFSETALLOCATOR_DIR}/offsetAllocator.hpp
  )
  target_include_directories(offsetAllocator PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
  set_property(TARGET offsetAllocator PROPERTY FOLDER "ThirdParty")
endif()

# KHR/GL headers
if(NOT TARGET gl_headers)
  add_library(gl_headers INTERFACE)
  target_include_directories(
    gl_headers INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/gl_headers
  )
endif()

# zlib-ng
if(NOT TARGET zlibstatic)
  set(SKIP_INSTALL_ALL ON)
  set(WITH_GTEST OFF)
  set(ZLIB_COMPAT ON)
  set(ZLIB_ENABLE_TESTS OFF)
  set(ZLIBNG_ENABLE_TESTS OFF)
  # zlib-ng's CMakeLists is pretty noisy.
  # Make it so that only warning messages appear from this subdirectory.
  set(_ORIGINAL_CMAKE_MESSAGE_LOG_LEVEL ${CMAKE_MESSAGE_LOG_LEVEL})
  set(CMAKE_MESSAGE_LOG_LEVEL "WARNING")
  add_subdirectory(zlib-ng)
  set(CMAKE_MESSAGE_LOG_LEVEL ${_ORIGINAL_CMAKE_MESSAGE_LOG_LEVEL})
  set_property(TARGET zlib PROPERTY FOLDER "ThirdParty")
endif()

# zstd
if(NOT TARGET zstd)
  set(ZSTD_BUILD_PROGRAMS OFF)
  set(ZSTD_BUILD_SHARED OFF)
  set(ZSTD_BUILD_STATIC ON)
  set(ZSTD_BUILD_TESTS OFF)
  set(ZSTD_USE_STATIC_RUNTIME ON)
  set(_ZSTD_DIR ${CMAKE_CURRENT_LIST_DIR}/zstd)
  # The EXCLUDE_FROM_ALL here removes Zstd from the INSTALL target
  add_subdirectory(${_ZSTD_DIR}/build/cmake ${CMAKE_BINARY_DIR}/zstd EXCLUDE_FROM_ALL)
  # zstd defines clean-all and uninstall targets we can't seem to avoid.
  # Remove them from BUILD_ALL so that it doesn't fail:
  set_target_properties(clean-all uninstall PROPERTIES EXCLUDE_FROM_DEFAULT_BUILD 1)
  set_property(TARGET libzstd_static clean-all uninstall PROPERTY FOLDER "ThirdParty")
endif()

# Basis Universal
if(NOT TARGET basisu)
  set(_BASISU_DIR ${CMAKE_CURRENT_LIST_DIR}/basis_universal)
  file(GLOB_RECURSE _BASISU_FILES "${_BASISU_DIR}/transcoder/*.*" "${_BASISU_DIR}/encoder/*.*")
  add_library(basisu STATIC ${_BASISU_FILES})
  target_include_directories(basisu INTERFACE "${_BASISU_DIR}/transcoder" "${_BASISU_DIR}/encoder")
  # Turn off some transcoding formats we don't use to reduce code size by about
  # 500 KB. We require BC4 (DXT5A), BC5, BC7, and ASTC (in case a user sets
  # device_supports_astc).
  target_compile_definitions(basisu PRIVATE
    BASISD_SUPPORT_ATC=0
    BASISD_SUPPORT_DXT1=0
    BASISD_SUPPORT_ETC2_EAC_A8=0
    BASISD_SUPPORT_ETC2_EAC_RG11=0
    BASISD_SUPPORT_FXT1=0
    BASISD_SUPPORT_PVRTC1=0
    BASISD_SUPPORT_PVRTC2=0
  )
  #basisu is using deprecated/unsecure POSIX functions.
  if(WIN32 AND NOT MSVC)
    add_compile_definitions(_CRT_NONSTDC_NO_DEPRECATE _CRT_INSECURE_NO_DEPRECATE)
  endif()
  # On x86, our minimum compilers support SSE 4.2, so Basis Universal can
  # compile with SSE intrinsics.
  if("${CMAKE_SYSTEM_PROCESSOR}" MATCHES "(x86_64|AMD64|i[3-6]86)")
    target_compile_definitions(basisu PRIVATE BASISU_SUPPORT_SSE=1)
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID MATCHES "GNU")
      # We need to add a flag to make GCC allow SSE 4.2:
      target_compile_options(basisu PRIVATE -msse4.2)
    endif()
  endif()
  # Set up linking between basisu and its dependencies, so that we always get
  # a correct linking order on Linux:
  target_link_libraries(basisu PUBLIC libzstd_static)
  set_property(TARGET basisu PROPERTY FOLDER "ThirdParty")
endif()

# cgltf
if (NOT TARGET cgltf AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/cgltf)
  add_library(cgltf INTERFACE)
  target_include_directories(cgltf INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/cgltf")
  set_property(TARGET cgltf PROPERTY FOLDER "ThirdParty")
endif()

# NVTX
if (NOT TARGET nvtx3-cpp AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/NVTX/c/CMakeLists.txt)
  add_subdirectory(NVTX/c)
  # Set folder property for NVTX targets if they exist
  if(TARGET nvtx3-cpp)
    set_property(TARGET nvtx3-cpp PROPERTY FOLDER "ThirdParty")
  endif()
  if(TARGET nvtx3-c)
    set_property(TARGET nvtx3-c PROPERTY FOLDER "ThirdParty")
  endif()
endif()

# meshoptimizer
if (NOT TARGET meshoptimizer)
  set(MESHOPT_INSTALL OFF)
  set(MESHOPT_BUILD_DEMO OFF)
  set(MESHOPT_BUILD_SHARED_LIBS OFF)
  set(MESHOPT_STABLE_EXPORTS OFF)
  set(MESHOPT_WERROR OFF)
  add_subdirectory(meshoptimizer)
  set_property(TARGET meshoptimizer PROPERTY FOLDER "ThirdParty")
endif()