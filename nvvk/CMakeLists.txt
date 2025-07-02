# NVVK contains Vulkan helpers.
set(LIB_NAME nvvk)

# Collect all source files in the utils directory
file(GLOB LIB_SOURCES "*.cpp" "*.hpp")
source_group("Source Files" FILES ${LIB_SOURCES})

# Define the utils library
add_library(${LIB_NAME} STATIC ${LIB_SOURCES})

# All libraries use nvpro_core2 as their include root
cmake_path(GET CMAKE_CURRENT_LIST_DIR PARENT_PATH PARENT_DIR)
target_include_directories(${LIB_NAME} PUBLIC ${PARENT_DIR})

target_link_libraries(
  ${LIB_NAME}
  PUBLIC nvutils
         glm # Math library
         volk # The Vulkan loader
         vma # Vulkan Memory Allocator
         fmt # Formatting library
         offsetAllocator # OffsetAllocator library
)

target_precompile_headers(${LIB_NAME} PRIVATE
  <filesystem>
)

add_library(nvpro2::${LIB_NAME} ALIAS ${LIB_NAME})
set_property(TARGET ${LIB_NAME} PROPERTY FOLDER "nvpro_core2")
