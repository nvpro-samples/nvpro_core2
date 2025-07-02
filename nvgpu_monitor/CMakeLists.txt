set(LIB_NAME nvgpu_monitor)

# Collect all source files in the utils directory
file(GLOB LIB_SOURCES "*.cpp" "*.h" "*.hpp")
source_group("Source Files" FILES ${LIB_SOURCES})

# Define the utils library
add_library(${LIB_NAME} STATIC ${LIB_SOURCES})

# All libraries use nvpro_core2 as their include root
cmake_path(GET CMAKE_CURRENT_LIST_DIR PARENT_PATH PARENT_DIR)
target_include_directories(${LIB_NAME} PUBLIC ${PARENT_DIR})

# Check if CUDA is present and add it
find_package(CUDAToolkit COMPONENTS CUDA::nvml)
if(TARGET CUDA::nvml)
  target_link_libraries(${LIB_NAME} PUBLIC CUDA::nvml)
  target_compile_definitions(${LIB_NAME} PUBLIC NVML_SUPPORTED)
  message(STATUS "NVML: Supported")
else()
  message(STATUS "CUDA (NVML) Toolkit not found.")
endif()

target_link_libraries(${LIB_NAME} PUBLIC 
  nvpro2::nvgui
  fmt
  imgui
  implot
  glm
)

target_precompile_headers(${LIB_NAME} PRIVATE
  <filesystem>
)

add_library(nvpro2::${LIB_NAME} ALIAS ${LIB_NAME})
set_property(TARGET ${LIB_NAME} PROPERTY FOLDER "nvpro_core2")
