# Miscellaneous CMake utilities.
# This should not include variable defaults (those go in Setup.cmake)
# or packages (those go in Find scripts).

#-------------------------------------------------------------------------------
# Sets up the standard PROJECT_NAME and PROJECT_EXE_TO_SOURCE_DIRECTORY C++ macros for
# a target.
# * PROJECT_NAME is the name of the target
# * PROJECT_EXE_TO_SOURCE_DIRECTORY is the relative path from the project output directory to CMAKE_CURRENT_SOURCE_DIR.
# * NVSHADERS_DIR is the absolute path to nvshaders/
macro(add_project_definitions TARGET_NAME)
  file(RELATIVE_PATH TO_CURRENT_SOURCE_DIR "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${CMAKE_CFG_INTDIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
  file(RELATIVE_PATH TO_DOWNLOAD_SOURCE_DIR "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${CMAKE_CFG_INTDIR}" "${NVPRO_CORE2_DOWNLOAD_DIR}")
  file(RELATIVE_PATH TO_ROOT_DIR "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${CMAKE_CFG_INTDIR}" "${CMAKE_SOURCE_DIR}")

  target_compile_definitions(${PROJECT_NAME} PRIVATE
    PROJECT_NAME="${TARGET_NAME}"
    PROJECT_EXE_TO_SOURCE_DIRECTORY="${TO_CURRENT_SOURCE_DIR}"
    PROJECT_EXE_TO_DOWNLOAD_DIRECTORY="${TO_DOWNLOAD_SOURCE_DIR}"
    PROJECT_EXE_TO_ROOT_DIRECTORY="${TO_ROOT_DIR}"
  )

  # This should always be set when the nvshaders_host target is defined -- but
  # check in case someone included Utilities.cmake without nvpro_core.
  # It might be better design to define an nvshaders utility target that sets
  # this.
  if(NVSHADERS_DIR)
    file(RELATIVE_PATH TO_NVSHADERS_SOURCE_DIR "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${CMAKE_CFG_INTDIR}" "${NVSHADERS_DIR}")
    target_compile_definitions(${PROJECT_NAME} PRIVATE
      NVSHADERS_DIR="${NVSHADERS_DIR}"
      PROJECT_EXE_TO_NVSHADERS_DIRECTORY="${TO_NVSHADERS_SOURCE_DIR}"
    )
  endif()
endmacro()

#-------------------------------------------------------------------------------
# Sets up install() for the given target.
# Optionally, copies files and directories to the executable output directory
# when building a sample or building INSTALL.
#
# Required argument:
# * TARGET_NAME: Name of a CMake target to install().
# Optional arguments:
# * INSTALL_DIR: The DESTINATION directory used for install().
#     Defaults to an empty string.
#     Convention when installing to a subfolder is to use `${PROJECT_NAME}_files`.
# * DIRECTORIES: List of directories to copy. Tree structure will be preserved.
# * FILES: List of files to copy to the root of the install directory.
# * NVSHADERS_FILES: List of files to copy to a subfolder named after the target / nvshaders.
# * LOCAL_DIRS: List of directories to copy to a subfolder named after the target.
# * AUTO: Copies all DLLs CMake knows the target links with, using
#     $<TARGET_RUNTIME_DLLS>.
function(copy_to_runtime_and_install TARGET_NAME)
    # Parse the arguments
    set(options AUTO)
    set(oneValueArgs INSTALL_DIR)
    set(multiValueArgs DIRECTORIES FILES LOCAL_DIRS NVSHADERS_FILES)
    cmake_parse_arguments(COPY "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    # Make sure the install directory isn't set to the target name; this will
    # cause conflicts on Linux.
    if(COPY_INSTALL_DIR STREQUAL "${TARGET_NAME}")
        message(ERROR "INSTALL_DIR was set to the project name. If you're making a subfolder for your project, try using `${PROJECT_NAME}_files` instead.")
    endif()
    
    if(NOT COPY_INSTALL_DIR)
      set(COPY_INSTALL_DIR ".")
    endif()


    install(TARGETS ${TARGET_NAME} RUNTIME DESTINATION ${COPY_INSTALL_DIR})

    # Handle directories
    foreach(DIR ${COPY_DIRECTORIES})
        # Ensure directory exists
        if(NOT EXISTS "${DIR}")
            message(WARNING "Directory does not exist: ${DIR}")
            continue()
        endif()


        # Copy for installation
        install(
            DIRECTORY ${DIR}
            DESTINATION ${COPY_INSTALL_DIR}
        )
    endforeach()

    # Handle individual files
    foreach(_FILE ${COPY_FILES})
        # Ensure file exists
        if(NOT EXISTS "${_FILE}")
            message(WARNING "File does not exist: ${_FILE}")
            continue()
        endif()

        # Copy for runtime (post-build)
        add_custom_command(
            TARGET ${TARGET_NAME} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different ${_FILE} "$<TARGET_FILE_DIR:${TARGET_NAME}>" VERBATIM
        )

        # Copy for installation
        install(
            FILES ${_FILE}
            DESTINATION ${COPY_INSTALL_DIR}
        )
    endforeach()
    
    # Handle nvshaders files
    if(COPY_NVSHADERS_FILES)
        install(
            FILES ${COPY_NVSHADERS_FILES}
            DESTINATION "${COPY_INSTALL_DIR}/nvshaders"
        )
    endif()
    
    # Handle local directories
    foreach(LOCAL ${COPY_LOCAL_DIRS})              
        
        # Windows, install to a subfolder named after the target
        if(MSVC)
            install(
                DIRECTORY ${LOCAL}
                DESTINATION "${COPY_INSTALL_DIR}/${TARGET_NAME}"
            )
        else()
        # Linux, install to the root of the install directory 
        # This is because there is an issue with the install command on Linux
            install(
                    DIRECTORY ${LOCAL}
                    DESTINATION "${COPY_INSTALL_DIR}"
                )
        endif()
    endforeach()


    if(COPY_AUTO)
        # On Windows, we want to copy the DLLs when building.
        # This isn't necessary on Linux, so long as the RPATH isn't changed.
        # That's good, since TARGET_RUNTIME_DLLS isn't available on Linux.
        if(MSVC)
            # We use copy -t here so that this works if TARGET_RUNTIME_DLLS is empty.
            add_custom_command(
                TARGET ${TARGET_NAME} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy -t $<TARGET_FILE_DIR:${TARGET_NAME}> $<TARGET_RUNTIME_DLLS:${TARGET_NAME}>
                COMMAND_EXPAND_LISTS
            )
          
            # This works better than install(... RUNTIME_DEPENDENCIES ...)
            # on Windows.
            install(
                FILES $<TARGET_RUNTIME_DLLS:${TARGET_NAME}>
                DESTINATION ${COPY_INSTALL_DIR}
            )
        else()
            # Although TARGET_RUNTIME_DLLS isn't available, we can do this:
            install(TARGETS ${TARGET_NAME}
                    RUNTIME_DEPENDENCIES
                    DESTINATION ${COPY_INSTALL_DIR})
        endif()
    endif()
endfunction()
