# NVGL contains OpenGL helpers.
set(LIB_NAME nvgl)

find_package(OpenGL)

set(PLATFORM_LIBRARIES)

if(UNIX)
  find_package(X11 REQUIRED)
  LIST(APPEND PLATFORM_LIBRARIES ${X11_LIBRARIES})
endif()

# Collect all source files in the utils directory
file(GLOB LIB_SOURCES "*.cpp" "*.hpp" "*.h")
source_group("Source Files" FILES ${LIB_SOURCES})

# Define the utils library
add_library(${LIB_NAME} STATIC ${LIB_SOURCES})

# All libraries use nvpro_core2 as their include root
cmake_path(GET CMAKE_CURRENT_LIST_DIR PARENT_PATH PARENT_DIR)
target_include_directories(${LIB_NAME} PUBLIC ${PARENT_DIR})

if(WIN32)
  target_compile_definitions(${LIB_NAME} PRIVATE WIN32_LEAN_AND_MEAN)
endif()

target_link_libraries(
  ${LIB_NAME}
  PUBLIC nvutils
         glfw               # Windowing library
         glm                # Math library
         imgui              # UI library
         gl_headers         # gl headers library
         stb                # stb image library
         ${OPENGL_LIBRARY}  # OpenGL library
         ${PLATFORM_LIBRARIES}
)

target_precompile_headers(${LIB_NAME} PRIVATE
  <filesystem>
)

add_library(nvpro2::${LIB_NAME} ALIAS ${LIB_NAME})
set_property(TARGET ${LIB_NAME} PROPERTY FOLDER "nvpro_core2")
