set(LIB_NAME nvutils)

# Collect all source files in the utils directory
file(GLOB LIB_SOURCES "*.cpp" "*.h" "*.hpp")
source_group("Source Files" FILES ${LIB_SOURCES})

# Define the utils library
add_library(${LIB_NAME} STATIC ${LIB_SOURCES})

# All libraries use nvpro_core2 as their include root
cmake_path(GET CMAKE_CURRENT_LIST_DIR PARENT_PATH PARENT_DIR)
target_include_directories(${LIB_NAME} PUBLIC ${PARENT_DIR})

# Check if TBB is present and add it (for parallel_batches)
if(UNIX)
  find_package(TBB)
  if(NOT TBB_FOUND)
    message(
      FATAL_ERROR
        "Could not find TBB. Try installing the TBB development \
library using your package manager: `sudo apt install libtbb-dev` on Ubuntu, \
`sudo dnf install tbb-devel` on Fedora, `sudo pacman -S onetbb` on Arch, or on \
NixOS, add `tbb` to your environment."
    )
  endif()
  
  # Produce a config error on TBB 2021.6+ and libstdc++ < 11 as they almost
  # certainly won't compile together, producing a somewhat arcane error; see
  # https://github.com/nvpro-samples/vk_gaussian_splatting/issues/9#issuecomment-3199120202
  if((CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
     AND (CMAKE_CXX_COMPILER_VERSION VERSION_LESS "11.0.0")
     AND (TBB_VERSION VERSION_GREATER_EQUAL "2021.6")
  )
    message(FATAL_ERROR "It looks like you're using GCC < 11.0.0 and TBB >= 2021.6. \
TBB 2021.6 and newer broke the API that libstdc++ < 11.0.0's parallel STL relied on. \
To fix this, try deleting ${CMAKE_BINARY_DIR}/CMakeCache.txt (if it exists) and then \
re-configuring with GCC 11 or newer (e.g. `CC=gcc-11 CXX=g++-11 cmake ...`). For more information, please see \
https://github.com/nvpro-samples/vk_gaussian_splatting/issues/9#issuecomment-3199120202 .")
  endif()

  target_link_libraries(${LIB_NAME} PUBLIC TBB::tbb)
endif()

target_link_libraries(
  ${LIB_NAME}
  PUBLIC fmt         # Formatting library
         glm         # Math library
         thread_pool # for parallel_for
)

# On Windows, link with Windows 10's general .lib file so timers.cpp can use
# QueryUnbiasedInterruptTimePrecise. This only uses kernelbase.dll.
if(WIN32)
  target_link_libraries(${LIB_NAME} PUBLIC OneCore.lib)
endif()

target_precompile_headers(${LIB_NAME} PRIVATE
  <filesystem>
)

add_library(nvpro2::${LIB_NAME} ALIAS ${LIB_NAME})
set_property(TARGET ${LIB_NAME} PROPERTY FOLDER "nvpro_core2")
