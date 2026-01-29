# This function compiles Slang shaders into C++ headers and .spv files using slangc.

# Usage:
#
#   # List of shaders
#   set(SHADER_FILES
#     ${CMAKE_SOURCE_DIR}/shaders/shader1.glsl
#     ${CMAKE_SOURCE_DIR}/shaders/shader2.glsl
#   )
#   
#   # Define the output directory for generated headers
#   set(SHADER_OUTPUT_DIR "${CMAKE_BINARY_DIR}/_autogen")
#   
#   # Call the function to compile shaders
#   compile_slang(
#     "${SHADER_FILES}"
#     "${SHADER_OUTPUT_DIR}"
#     GENERATED_SHADER_HEADERS
# ...
#     # Optional arguments
#     TARGET_ENV "vulkan1.1"
#     EXTRA_FLAGS "-I<dir>" "-Dfoo=1"
#     NAME_SUFFIX ".combination_1"
#   )
#   
#   # Use the generated headers as needed (example: link them to a target)
#   target_sources(MyProg PRIVATE ${GENERATED_SHADER_HEADERS})



function(compile_slang SHADER_FILES OUTPUT_DIR SHADER_HEADERS_VAR)
  # Optional arguments for flexibility
  set(options )
  set(oneValueArgs TARGET_ENV DEBUG_LEVEL OPTIMIZATION_LEVEL NAME_SUFFIX)
  set(multiValueArgs EXTRA_FLAGS)
  cmake_parse_arguments(COMPILE_SHADER "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  # Set defaults for optional arguments
  set(TARGET_ENV ${COMPILE_SHADER_TARGET_ENV})
  if(NOT TARGET_ENV)
    set(TARGET_ENV "vulkan1.4")
  endif()
  set(EXTRA_FLAGS ${COMPILE_SHADER_EXTRA_FLAGS})
  set(NAME_SUFFIX ${COMPILE_SHADER_NAME_SUFFIX})

  set(DEBUG_LEVEL ${COMPILE_SHADER_DEBUG_LEVEL})
  if(NOT COMPILE_SHADER_DEBUG_LEVEL)
    set(DEBUG_LEVEL 1)
  endif()

  set(OPTIMIZATION_LEVEL ${COMPILE_SHADER_OPTIMIZATION_LEVEL})
  if(NOT COMPILE_SHADER_OPTIMIZATION_LEVEL)
    set(OPTIMIZATION_LEVEL 0)
  endif()
  
  # Ensure the output directory exists
  file(MAKE_DIRECTORY ${OUTPUT_DIR})

  # Initialize the output variable
  set(SHADER_HEADERS "")
  set(SHADER_SPVS "")


  set(_SLANG_FLAGS
        -profile sm_6_6+spirv_1_6 # Target SM 6.6 and SPIR-V 1.6
        # Enable default capabilities for our shaders; more can be added using EXTRA_FLAGS.
        -capability spvInt64Atomics+spvShaderClockKHR+spvRayTracingMotionBlurNV+spvRayQueryKHR+SPV_KHR_compute_shader_derivatives
        -target spirv             # Target SPIR-V
        -emit-spirv-directly      # Emit SPIR-V directly without intermediate files
        -force-glsl-scalar-layout # Force scalar layout for Vulkan shaders
        -fvk-use-entrypoint-name  # Use the entrypoint name as the shader name
        -g${DEBUG_LEVEL}          # Enable debug info
        -O${OPTIMIZATION_LEVEL}   # Optimization level
        -lang slang               # Force all files to use slang language
        -matrix-layout-row-major  # Explicitly set row-major layout (Slang default is row-major)
        # -allow-glsl
    )

  # Compile Slang shaders using slangc
  foreach(SHADER ${SHADER_FILES})
      get_filename_component(SHADER_NAME ${SHADER} NAME)
      string(APPEND SHADER_NAME ${NAME_SUFFIX})
      string(REPLACE "." "_" VN_SHADER_NAME ${SHADER_NAME})
      set(OUTPUT_FILE ${OUTPUT_DIR}/${SHADER_NAME})
      set(_COMMAND_H ${Slang_SLANGC_EXECUTABLE}
        ${_SLANG_FLAGS} ${EXTRA_FLAGS}
        -source-embed-name ${VN_SHADER_NAME}
        -source-embed-style text
        -depfile "${OUTPUT_FILE}.dep"
        -o "${OUTPUT_FILE}.h" ${SHADER}
      )
      set(_COMMAND_S ${Slang_SLANGC_EXECUTABLE}
        ${_SLANG_FLAGS} ${EXTRA_FLAGS}
        -o "${OUTPUT_FILE}.spv" ${SHADER}
      )
      add_custom_command(
        # PRE_BUILD
        OUTPUT ${OUTPUT_FILE}.h ${OUTPUT_FILE}.spv
        COMMAND echo ${_COMMAND_H}
        COMMAND ${_COMMAND_H}
        COMMAND echo ${_COMMAND_S}
        COMMAND ${_COMMAND_S}
        MAIN_DEPENDENCY ${SHADER}
        DEPFILE "${OUTPUT_FILE}.dep"
        COMMENT "Compiling Slang shader ${SHADER_NAME}"
      )
      list(APPEND SHADER_HEADERS "${OUTPUT_FILE}.h")
  endforeach()

  # Set the output variables for the caller
  set(${SHADER_HEADERS_VAR} ${SHADER_HEADERS} PARENT_SCOPE)
endfunction()

