# Direct3D 12 back end of nvapp (window, swapchain, ImGui renderer).
set(LIB_NAME nvdx12app)

# Collect all source files in the library directory
file(GLOB LIB_SOURCES CONFIGURE_DEPENDS "*.cpp" "*.hpp")
source_group("Source Files" FILES ${LIB_SOURCES})

# Define the library
add_library(${LIB_NAME} STATIC ${LIB_SOURCES})

# All libraries use nvpro_core2 as their include root
cmake_path(GET CMAKE_CURRENT_LIST_DIR PARENT_PATH PARENT_DIR)
set(IMGUI_BACKENDS_DIR "${PARENT_DIR}/third_party/imgui/backends")
target_sources(${LIB_NAME} PRIVATE "${IMGUI_BACKENDS_DIR}/imgui_impl_dx12.cpp" "${IMGUI_BACKENDS_DIR}/imgui_impl_dx12.h")
target_include_directories(${LIB_NAME} PUBLIC ${PARENT_DIR})

target_link_libraries(
  ${LIB_NAME}
  PUBLIC
    nvutils
    nvdx12
    nvgui
    nvapp
    glfw   # Windowing library (Application needs it)
    glm    # Math library
    imgui  # ImGui DX12 renderer and ImTextureID handles
    implot # Elements plot through ImPlot
    stb    # screenshots (stb_image_write)
    fmt    # For formatting strings
)

target_precompile_headers(${LIB_NAME} PRIVATE
  <filesystem>
)

add_library(nvpro2::${LIB_NAME} ALIAS ${LIB_NAME})
set_property(TARGET ${LIB_NAME} PROPERTY FOLDER "nvpro_core2")
