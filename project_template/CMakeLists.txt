cmake_minimum_required(VERSION 3.22 FATAL_ERROR)
get_filename_component(PROJECT_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)
project(${PROJECT_NAME} LANGUAGES C CXX)

# Prior adding the package one can disable certain libraries
# e.g.
option(NVPRO2_ENABLE_nvgl "" OFF)
option(NVPRO2_ENABLE_nvgpu_monitor "" OFF)

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

