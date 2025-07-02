# Provides nvpro2::nvaftermath.
# Depends on nvvk.

set(LIB_NAME nvaftermath)

# ~~~
# Find Nsight Aftermath SDK Headers and Library
# ~~~
set(NsightAftermath_SDK
    ""
    CACHE PATH "Point to top directory of nSight Aftermath SDK"
)
if(NOT NsightAftermath_SDK)
  set(NsightAftermath_SDK "$ENV{NSIGHT_AFTERMATH_SDK}")
endif()
if(NsightAftermath_SDK)
  find_package(NsightAftermath)
  get_filename_component(
    NsightAftermath_LIBRARY_DIR ${NsightAftermath_LIBRARY} DIRECTORY
  )
endif()

if(NsightAftermath_FOUND)
  message(STATUS "Nsight Aftermath found")
  message(STATUS "  Library Directory : ${NsightAftermath_LIBRARY_DIR}")
  message(STATUS "  Library           : ${NsightAftermath_LIBRARY}")
  message(STATUS "  DLL               : ${NsightAftermath_DLLS}")
  message(STATUS "  Include           : ${NsightAftermath_INCLUDE_DIR}")
else()
  message(STATUS "Nsight Aftermath SDK not found.")
endif()

# If Aftermath isn't available, aftermath.{cpp, hpp} becomes a stub library.
add_library(
  ${LIB_NAME} STATIC ${CMAKE_CURRENT_LIST_DIR}/aftermath.cpp
                     ${CMAKE_CURRENT_LIST_DIR}/aftermath.hpp
)

# All libraries use nvpro_core2 as their include root
cmake_path(GET CMAKE_CURRENT_LIST_DIR PARENT_PATH PARENT_DIR)
target_include_directories(${LIB_NAME} PUBLIC ${PARENT_DIR})

target_link_libraries(${LIB_NAME} PUBLIC nvvk)

target_precompile_headers(${LIB_NAME} PRIVATE
  <filesystem>
)

if(NsightAftermath_FOUND)
  # Enable full Aftermath support.
  target_sources(
    ${LIB_NAME}
    PRIVATE ${CMAKE_CURRENT_LIST_DIR}/NsightAftermathGpuCrashTracker.cpp
            ${CMAKE_CURRENT_LIST_DIR}/NsightAftermathGpuCrashTracker.h
            ${CMAKE_CURRENT_LIST_DIR}/NsightAftermathHelpers.h
  )
  target_compile_definitions(${LIB_NAME} PUBLIC AFTERMATH_AVAILABLE=1)
  target_link_libraries(${LIB_NAME} PUBLIC NsightAftermath::NsightAftermath)
endif()

add_library(nvpro2::${LIB_NAME} ALIAS ${LIB_NAME})
set_property(TARGET ${LIB_NAME} PROPERTY FOLDER "nvpro_core2")
