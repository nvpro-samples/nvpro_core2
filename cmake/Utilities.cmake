# Miscellaneous CMake utilities.
# This should not include variable defaults (those go in Setup.cmake)
# or packages (those go in Find scripts).

#-------------------------------------------------------------------------------
# Sets up the standard TARGET_NAME and TARGET_EXE_TO_SOURCE_DIRECTORY C++ macros for
# a target.
# * TARGET_NAME is the name of the target
# * TARGET_EXE_TO_SOURCE_DIRECTORY is the relative path from the project output directory to CMAKE_CURRENT_SOURCE_DIR.
# * NVSHADERS_DIR is the absolute path to nvshaders/
macro(add_project_definitions TARGET_NAME)
  file(RELATIVE_PATH TO_CURRENT_SOURCE_DIR "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${CMAKE_CFG_INTDIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
  file(RELATIVE_PATH TO_DOWNLOAD_SOURCE_DIR "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${CMAKE_CFG_INTDIR}" "${NVPRO_CORE2_DOWNLOAD_DIR}")
  file(RELATIVE_PATH TO_ROOT_DIR "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${CMAKE_CFG_INTDIR}" "${CMAKE_SOURCE_DIR}")

  target_compile_definitions(${TARGET_NAME} PRIVATE
    TARGET_NAME="${TARGET_NAME}"
    TARGET_EXE_TO_SOURCE_DIRECTORY="${TO_CURRENT_SOURCE_DIR}"
    TARGET_EXE_TO_DOWNLOAD_DIRECTORY="${TO_DOWNLOAD_SOURCE_DIR}"
    TARGET_EXE_TO_ROOT_DIRECTORY="${TO_ROOT_DIR}"
  )

  # This should always be set when the nvshaders_host target is defined -- but
  # check in case someone included Utilities.cmake without nvpro_core.
  # It might be better design to define an nvshaders utility target that sets
  # this.
  if(NVSHADERS_DIR)
    file(RELATIVE_PATH TO_NVSHADERS_SOURCE_DIR "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${CMAKE_CFG_INTDIR}" "${NVSHADERS_DIR}")
    target_compile_definitions(${TARGET_NAME} PRIVATE
      NVSHADERS_DIR="${NVSHADERS_DIR}"
      TARGET_EXE_TO_NVSHADERS_DIRECTORY="${TO_NVSHADERS_SOURCE_DIR}"
    )
  endif()
endmacro()

#-------------------------------------------------------------------------------
# Sets up install() for the given target.
# Optionally, copies files and directories to the executable output directory
# when building a sample or building INSTALL.
#
# Required argument:
# * TARGET_NAME: Name of a CMake target to install() to the root of the install directory.
# Optional arguments:
# * INSTALL_DIR: The DESTINATION directory used for install().
#     Defaults to an empty string.
# * DIRECTORIES: List of directories to copy to INSTALL_DIR. Tree structure will be preserved.
# * LOCAL_DIRS: List of directories to copy to INSTALL_DIR/TARGET_NAME_files. Tree structure will be preserved.
# * FILES: List of files to copy to the build directory and INSTALL_DIR.
# * PROGRAMS: Same as FILES except it uses install(PROGRAMS), which sets a+x on Linux.
# * NVSHADERS_FILES: List of files to copy to INSTALL_DIR/nvshaders.
# * AUTO: Copies all shared libraries CMake knows the target links with to the build directory and INSTALL_DIR.
#
# Example:
# copy_to_runtime_and_install(
#   example
#   DIRECTORIES "${NVSHADERS_DIR}/nvshaders"
#   LOCAL_DIRS models
#   FILES spruit_sunrise.hdr
#   AUTO
# )
# - installs nvpro_core2/nvshaders to _install/nvshaders
# - installs models to _install/example_files/models
# - copies spruit_sunrise.hdr to _bin/spruit_sunrise.hdr and installs to _install/spruit_sunrise.hdr
# - copies shared libraries `example` depends on to _bin and installs them to _install
function(copy_to_runtime_and_install TARGET_NAME)
    # Parse the arguments
    set(options AUTO)
    set(oneValueArgs INSTALL_DIR)
    set(multiValueArgs DIRECTORIES FILES PROGRAMS LOCAL_DIRS NVSHADERS_FILES)
    cmake_parse_arguments(COPY "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if(${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.31")
        cmake_policy(SET CMP0177 NEW) # Normalize DESTINATION paths
    endif()

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
            USE_SOURCE_PERMISSIONS
        )
    endforeach()


    # Handle local directories
    foreach(LOCAL_DIR ${COPY_LOCAL_DIRS})
        # Ensure directory exists
        if(NOT EXISTS "${LOCAL_DIR}")
            message(WARNING "Directory does not exist: ${LOCAL_DIR}")
            continue()
        endif()

        install(
            DIRECTORY ${LOCAL_DIR}
            DESTINATION "${COPY_INSTALL_DIR}/${TARGET_NAME}_files"
        )
    endforeach()


    # Handle individual files
    foreach(_FILETYPE "FILES" "PROGRAMS")
        foreach(_FILE ${COPY_${_FILETYPE}})
            # If it's not a generator expression, make sure it exists
            if((NOT (_FILE MATCHES "^\\$")) AND (NOT EXISTS "${_FILE}"))
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
                ${_FILETYPE} ${_FILE}
                DESTINATION ${COPY_INSTALL_DIR}
            )
        endforeach()
    endforeach()

    # Handle nvshaders files
    if(COPY_NVSHADERS_FILES)
        install(
            FILES ${COPY_NVSHADERS_FILES}
            DESTINATION "${COPY_INSTALL_DIR}/nvshaders"
        )
    endif()


    if(COPY_AUTO)
        # On Windows, we want to copy the DLLs when building.
        # This isn't necessary on Linux, so long as the RPATH isn't changed.
        # That's good, since TARGET_RUNTIME_DLLS isn't available on Linux.
        if(WIN32)
            # Avoid emitting a command if we have no DLLs to copy.
            set(_HAVE_RUNTIME_DLLS $<BOOL:$<TARGET_RUNTIME_DLLS:${TARGET_NAME}>>)
            set(_COMMAND ${CMAKE_COMMAND} -E copy_if_different "$<TARGET_RUNTIME_DLLS:${TARGET_NAME}>" "$<TARGET_FILE_DIR:${TARGET_NAME}>")
            add_custom_command(
                TARGET ${TARGET_NAME} POST_BUILD
                COMMAND "$<${_HAVE_RUNTIME_DLLS}:${_COMMAND}>"
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
                        POST_EXCLUDE_REGEXES "^/lib" "^/usr/lib"
                    DESTINATION ${COPY_INSTALL_DIR})
        endif()
    endif()
endfunction()
