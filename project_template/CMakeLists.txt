cmake_minimum_required(VERSION 3.22 FATAL_ERROR)
get_filename_component(PROJECT_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)
project(${PROJECT_NAME} LANGUAGES C CXX)

# Prior adding the package one can disable certain libraries
# e.g.
option(NVPRO2_ENABLE_nvgl "" OFF)
option(NVPRO2_ENABLE_nvgpu_monitor "" OFF)

# The Direct3D 12 back end is opt-in and Windows only; it pulls in the Agility SDK.
option(BUILD_DX12 "Also build the Direct3D 12 variant of this sample" OFF)
if(BUILD_DX12 AND WIN32)
  option(NVPRO2_ENABLE_nvdx12 "" ON)
  option(NVPRO2_ENABLE_nvdx12app "" ON)
endif()

# Call FindNvproCore2 to download nvpro_core2 or find it if it is already
# downloaded, or is located as a subdirectory or up to two levels up.
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
find_package(NvproCore2 REQUIRED)

#####################################################################################
# Source files for this project
#

set(SOURCE_FILES
  main.cpp
)

source_group("Source Files" FILES ${SOURCE_FILES})

#####################################################################################
# Executable
#

add_executable(${PROJECT_NAME} ${SOURCE_FILES})
set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT ${PROJECT_NAME})

target_link_libraries(${PROJECT_NAME} PRIVATE
  nvpro2::nvapp
)

add_project_definitions(${PROJECT_NAME})

# This sample doesn't need addtional files, but one might need to
# copy required dlls, additional commands etc. through this command
copy_to_runtime_and_install(${PROJECT_NAME} AUTO)

# The same sample on the Direct3D 12 back end: main_dx12.cpp on
# nvdx12app::Application, where main.cpp uses nvapp::Application.
if(BUILD_DX12 AND WIN32)
  source_group("Source Files" FILES main_dx12.cpp)

  add_executable(${PROJECT_NAME}_dx12 main_dx12.cpp)

  target_link_libraries(${PROJECT_NAME}_dx12 PRIVATE
    nvpro2::nvdx12app
  )

  add_project_definitions(${PROJECT_NAME}_dx12)

  # The Agility SDK runtime must sit in the D3D12 folder next to the executable,
  # which is what the sample exports as D3D12SDKPath.
  copy_to_runtime_and_install(${PROJECT_NAME}_dx12 AUTO
    FILES ${DX12AgilitySdk_DLLS}
    SUBDIRECTORY "D3D12"
  )
endif()

