set(LIB_NAME nvmcp)

file(GLOB LIB_SOURCES "*.cpp" "*.hpp")
source_group("Source Files" FILES ${LIB_SOURCES})

add_library(${LIB_NAME} STATIC ${LIB_SOURCES})

cmake_path(GET CMAKE_CURRENT_LIST_DIR PARENT_PATH PARENT_DIR)
target_include_directories(${LIB_NAME} PUBLIC ${PARENT_DIR})

if(NVPRO2_ENABLE_MCP_AUTO_PARAMETER_REGISTRY)
  target_compile_definitions(${LIB_NAME} PRIVATE NVPRO2_TRACK_PARAMETER_REGISTRIES)
endif()

target_link_libraries(${LIB_NAME}
  PUBLIC nvapp
  PRIVATE mcp
)

add_library(nvpro2::${LIB_NAME} ALIAS ${LIB_NAME})
set_property(TARGET ${LIB_NAME} PROPERTY FOLDER "nvpro_core2")
