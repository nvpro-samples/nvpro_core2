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
source_group("Source Files" FILES ${LIB_SOURCES})

# Define the utils library
add_library(nvshaders_host STATIC ${LIB_SOURCES})

# All libraries use nvpro_core2 as their include root
cmake_path(GET CMAKE_CURRENT_LIST_DIR PARENT_PATH PARENT_DIR)
target_include_directories(nvshaders_host PUBLIC ${PARENT_DIR})

target_link_libraries(nvshaders_host PUBLIC nvvk glm)

target_precompile_headers(nvshaders_host PRIVATE
  <filesystem>
)

add_library(nvpro2::nvshaders_host ALIAS nvshaders_host)
set_property(TARGET nvshaders_host PROPERTY FOLDER "nvpro_core2")
