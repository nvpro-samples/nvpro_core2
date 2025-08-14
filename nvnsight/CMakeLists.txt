# NVTX/Nsight integration for nvpro_core2
# Provides conditional NVTX support for NVIDIA Nsight Graphics profiling

# Add CMake option for NVTX support (disabled by default)
option(NSIGHT_ENABLE_NVTX "Enable NVTX profiling support for NVIDIA Nsight Graphics" OFF)

# Create a STATIC library for nvnsight (so it shows up in Visual Studio)
# We'll add a dummy source file to make it a proper library
add_library(nvnsight STATIC 
    ${CMAKE_CURRENT_SOURCE_DIR}/example_usage.cpp 
    ${CMAKE_CURRENT_SOURCE_DIR}/nsightevents.hpp)

# Add include directory for nsightevents.hpp
target_include_directories(nvnsight PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)

# Create an alias for easier usage
add_library(nvpro2::nvnsight ALIAS nvnsight)

if(NSIGHT_ENABLE_NVTX)
  # Check if NVTX target is available (from third_party)
  if(TARGET nvtx3-cpp)
    # Link against NVTX C++ API (includes both C and C++ APIs)
    target_link_libraries(nvnsight PUBLIC nvtx3-cpp)
    
    # Define the macro to enable NVTX functionality in nsightevents.hpp
    target_compile_definitions(nvnsight PUBLIC NVTX_AVAILABLE)
    
    message(STATUS "NVTX support enabled - nsightevents.hpp will provide NVTX functionality")
  else()
    # NVTX not available, fall back to disabled mode
    message(WARNING "NVTX support requested but NVTX target not found. NVTX will be disabled.")
    message(WARNING "Run the repro script to download NVTX: python dev-rebuild-filtered-repos.py")
    message(STATUS "NVTX support disabled - nsightevents.hpp will provide no-op macros")
  endif()
else()
  message(STATUS "NVTX support disabled - nsightevents.hpp will provide no-op macros")
  # If NVTX is not enabled, define the macro to disable NVTX functionality
  target_compile_definitions(nvnsight PUBLIC NVTX_DISABLE) 
endif()

# Set the folder for IDE organization
set_property(TARGET nvnsight PROPERTY FOLDER "nvpro_core2")
