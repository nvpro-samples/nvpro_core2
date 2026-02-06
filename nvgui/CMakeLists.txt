# GUI widgets.
set(LIB_NAME nvgui)

# Collect all source files in the utils directory
file(GLOB LIB_SOURCES "*.cpp" "*.h" "*.hpp")

# On macOS, also include Objective-C++ files
if(APPLE)
  file(GLOB LIB_SOURCES_MM "*.mm")
  list(APPEND LIB_SOURCES ${LIB_SOURCES_MM})
endif()

source_group("Source Files" FILES ${LIB_SOURCES})

# Define the utils library
add_library(${LIB_NAME} STATIC ${LIB_SOURCES})

# All libraries use nvpro_core2 as their include root
cmake_path(GET CMAKE_CURRENT_LIST_DIR PARENT_PATH PARENT_DIR)
target_include_directories(${LIB_NAME} PUBLIC ${PARENT_DIR})

target_link_libraries(
  ${LIB_NAME}
  PUBLIC
    imgui
    glfw   # for file_dialog
    fmt
    glm
)

# On macOS, link against AppKit and UniformTypeIdentifiers for file dialogs
if(APPLE)
  target_link_libraries(${LIB_NAME} PRIVATE "-framework AppKit" "-framework UniformTypeIdentifiers")
  # Disable precompiled headers for Objective-C++ files (incompatible with C++ PCH)
  set_source_files_properties(${LIB_SOURCES_MM} PROPERTIES SKIP_PRECOMPILE_HEADERS ON)
endif()

target_precompile_headers(${LIB_NAME} PRIVATE
  <filesystem>
)

add_library(nvpro2::${LIB_NAME} ALIAS ${LIB_NAME})
set_property(TARGET ${LIB_NAME} PROPERTY FOLDER "nvpro_core2")
