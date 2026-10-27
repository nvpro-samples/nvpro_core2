# Window and GUI library: the Vulkan nvapp::Application and its elements.
# Without nvvk the library is the API-agnostic part alone (application_base,
# detail and frame_pacer), which nvdx12app builds on.
set(LIB_NAME nvapp)

# Collect all source files in the utils directory
file(GLOB LIB_SOURCES "*.cpp" "*.hpp")

# Shaders that are not compiled by the build system, together with the generated
# headers holding their SPIR-V. Listed so they show up in the IDE.
file(GLOB LIB_SHADERS "*.slang" "*.slang.h")

if(NOT NVPRO2_ENABLE_nvvk)
  set(LIB_SOURCES
      ${CMAKE_CURRENT_LIST_DIR}/application_base.cpp
      ${CMAKE_CURRENT_LIST_DIR}/application_base.hpp
      ${CMAKE_CURRENT_LIST_DIR}/detail.cpp
      ${CMAKE_CURRENT_LIST_DIR}/detail.hpp
      ${CMAKE_CURRENT_LIST_DIR}/frame_pacer.cpp
      ${CMAKE_CURRENT_LIST_DIR}/frame_pacer.hpp
  )
  set(LIB_SHADERS)
endif()
source_group("Source Files" FILES ${LIB_SOURCES})
source_group("Shaders" FILES ${LIB_SHADERS})

# Define the utils library
add_library(${LIB_NAME} STATIC ${LIB_SOURCES} ${LIB_SHADERS})

if(LIB_SHADERS)
  set_source_files_properties(${LIB_SHADERS} PROPERTIES HEADER_FILE_ONLY TRUE)
endif()

# All libraries use nvpro_core2 as their include root
cmake_path(GET CMAKE_CURRENT_LIST_DIR PARENT_PATH PARENT_DIR)
target_include_directories(${LIB_NAME} PUBLIC ${PARENT_DIR})

target_link_libraries(
  ${LIB_NAME}
  PUBLIC
    nvutils
    nvgui
    glfw   # Windowing library (Application needs it)
    glm    # Math library
    imgui  # ImGui Vulkan backend: Application init + ImGui_ImplVulkan_AddTexture (also nvapp::ImTexture)
    implot # For Implot
    fmt    # For formatting strings
)
if(NVPRO2_ENABLE_nvvk)
  target_link_libraries(${LIB_NAME} PUBLIC
    nvvk
    vma    # Vulkan Memory Allocator
  )
endif()

target_precompile_headers(${LIB_NAME} PRIVATE
  <filesystem>
)

add_library(nvpro2::${LIB_NAME} ALIAS ${LIB_NAME})
set_property(TARGET ${LIB_NAME} PROPERTY FOLDER "nvpro_core2")
