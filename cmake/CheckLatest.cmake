# This file is used to check if the local nvpro_core2 is up to date with the remote.
# If the local nvpro_core2 is behind the remote, it will print a warning message.



# Check if remote exists and local is behind
if(EXISTS ${NvproCore2_ROOT}/nvpro_core2/.git)
    # First check if we're on the main branch
    execute_process(
        COMMAND git rev-parse --abbrev-ref HEAD
        WORKING_DIRECTORY ${NvproCore2_ROOT}/nvpro_core2
        OUTPUT_VARIABLE CURRENT_BRANCH
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE BRANCH_RESULT
    )

    # Only check for updates if we're on the main branch
    if(BRANCH_RESULT EQUAL 0 AND CURRENT_BRANCH STREQUAL "main")
        message(STATUS "Checking for nvpro_core2 updates")
        
        execute_process(
            COMMAND git ls-remote --heads origin main
            WORKING_DIRECTORY ${NvproCore2_ROOT}/nvpro_core2
            OUTPUT_VARIABLE REMOTE_MAIN_LINE
            OUTPUT_STRIP_TRAILING_WHITESPACE
            RESULT_VARIABLE REMOTE_RESULT
        )
        
        if(REMOTE_RESULT EQUAL 0 AND REMOTE_MAIN_LINE)
            # Extract just the hash (first word)
            string(REGEX MATCH "^[0-9a-fA-F]+" REMOTE_MAIN_HASH "${REMOTE_MAIN_LINE}")

            # Validate that we got a valid hash
            if(REMOTE_MAIN_HASH)
                execute_process(
                    COMMAND git rev-parse HEAD
                    WORKING_DIRECTORY ${NvproCore2_ROOT}/nvpro_core2
                    OUTPUT_VARIABLE LOCAL_HASH
                    OUTPUT_STRIP_TRAILING_WHITESPACE
                    RESULT_VARIABLE LOCAL_RESULT
                )
                
                if(LOCAL_RESULT EQUAL 0 AND LOCAL_HASH)
                    if(NOT LOCAL_HASH STREQUAL REMOTE_MAIN_HASH)
                        message("WARNING: Local nvpro_core2 may be outdated. Consider updating from remote.")
                        message("  Local:  ${LOCAL_HASH}")
                        message("  Remote: ${REMOTE_MAIN_HASH}")
                    endif()
                endif()
            endif()
        endif()
    else()
        message(STATUS "Skipping update check - not on main branch (current: ${CURRENT_BRANCH})")
    endif()
endif()
