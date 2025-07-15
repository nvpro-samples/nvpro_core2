# nvcompilers links with the Slang and ShaderC runtime compilers,
# and includers helpers for using these runtime compilers.

set(LIB_NAME nvslang)

# Collect all source files
file(GLOB LIB_SOURCES "*.cpp" "*.hpp")
source_group("Source Files" FILES ${LIB_SOURCES})

# Define the library
add_library(${LIB_NAME} STATIC ${LIB_SOURCES})

# All libraries use nvpro_core2 as their include root
cmake_path(GET CMAKE_CURRENT_LIST_DIR PARENT_PATH PARENT_DIR)
target_include_directories(${LIB_NAME} PUBLIC ${PARENT_DIR})

# For compiler_slang.hpp:
find_package(Slang)
target_include_directories(${LIB_NAME} PUBLIC ${Slang_INCLUDE_DIR})

target_link_libraries(
  ${LIB_NAME}
  PUBLIC nvutils
         Slang
)

target_precompile_headers(${LIB_NAME} PRIVATE
  <filesystem>
)

add_library(nvpro2::${LIB_NAME} ALIAS ${LIB_NAME})
set_property(TARGET ${LIB_NAME} PROPERTY FOLDER "nvpro_core2")
