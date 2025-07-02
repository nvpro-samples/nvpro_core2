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
  if(TBB_FOUND)
    target_link_libraries(${LIB_NAME} PUBLIC TBB::tbb)
  else()
    message(
      FATAL_ERROR
        "Could not find TBB. Try installing the TBB development \
library using your package manager: `sudo apt install libtbb-dev` on Ubuntu, \
`sudo dnf install tbb-devel` on Fedora, `sudo pacman -S onetbb` on Arch, or on \
NixOS, add `tbb` to your environment."
    )
  endif()
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
