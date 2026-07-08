set(LIB_NAME nvimageformats)

# Collect all source files in the utils directory
file(GLOB LIB_SOURCES "*.cpp" "*.h" "*.hpp")
source_group("Source" FILES ${LIB_SOURCES})

# Define the utils library
add_library(${LIB_NAME} STATIC ${LIB_SOURCES})

# All libraries use nvpro_core2 as their include root
cmake_path(GET CMAKE_CURRENT_LIST_DIR PARENT_PATH PARENT_DIR)
target_include_directories(${LIB_NAME} PUBLIC ${PARENT_DIR})

target_include_directories(${LIB_NAME} PUBLIC ${Vulkan_INCLUDE_DIRS}) # VkFormat

target_link_libraries(
  ${LIB_NAME}
  PUBLIC dxh # DXGI_FORMAT
         libzstd_static
         zlib
)

target_compile_definitions(${LIB_NAME} PUBLIC
                           NVP_SUPPORTS_GZLIB
                           NVP_SUPPORTS_ZSTD
)

if(NVPRO2_ENABLE_nvimageformats_basisu)
  if(NOT TARGET basisu)
    message(FATAL_ERROR "NVPRO2_ENABLE_nvimageformats_basisu is ON, but the basisu target is not available.")
  endif()
  target_link_libraries(${LIB_NAME} PUBLIC basisu)
  target_compile_definitions(${LIB_NAME} PUBLIC NVP_SUPPORTS_BASISU)
endif()

add_library(nvpro2::${LIB_NAME} ALIAS ${LIB_NAME})
set_property(TARGET ${LIB_NAME} PROPERTY FOLDER "nvpro_core2")
