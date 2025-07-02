# This calls find_package(Vulkan), and finds some additional components that
# CMake doesn't yet provide. Also prints out diagnostic information.
find_package(Vulkan COMPONENTS glslangValidator)
get_filename_component(_Vulkan_LIB_DIR ${Vulkan_LIBRARY} DIRECTORY)
set(_Vulkan_BIN_DIR ${Vulkan_INCLUDE_DIRS}/../bin)
# Vulkan - Volk
if(NOT Vulkan_VOLK_DIR)
  find_path(Vulkan_VOLK_DIR volk.h HINTS ${Vulkan_INCLUDE_DIRS}/volk)
endif()

# Finds a library named NAME and sets Vulkan_NAME_LIBRARY and
# Vulkan_NAME_DLL (converting the name to uppercase and hyphens to underscores).
# Also sets up an IMPORTED SHARED library named ${NAME}.
macro(nvpro2_find_vulkan_shared_library NAME)
  string(TOUPPER ${NAME} VARNAME)
  string(REPLACE "-" "_" VARNAME ${VARNAME})

  # CMake note:
  # ${_LIB_VAR}    == Vulkan_${VARNAME}_LIBRARY
  # ${${_LIB_VAR}} == ${Vulkan_${VARNAME}_LIBRARY}
  set(_LIB_VAR Vulkan_${VARNAME}_LIBRARY)
  set(_DLL_VAR Vulkan_${VARNAME}_DLL)

  if(NOT ${_LIB_VAR})
    find_library(${_LIB_VAR}
      NAMES ${NAME}
      HINTS ${_Vulkan_LIB_DIR}
    )
  endif()

  if(NOT ${_DLL_VAR})
    if(WIN32)
      find_file(${_DLL_VAR}
        NAMES ${NAME}.dll
        HINTS ${_Vulkan_BIN_DIR}
      )
    else()
      set(${_DLL_VAR} ${${_LIB_VAR}})
    endif()
  endif()

  # Imported library
  if(NOT TARGET ${NAME})
    add_library(${NAME} SHARED IMPORTED)
    set_target_properties(${NAME} PROPERTIES
      IMPORTED_LOCATION ${${_DLL_VAR}}
      INTERFACE_INCLUDE_DIRECTORIES ${Vulkan_INCLUDE_DIRS}
    )
    if(WIN32)
      set_target_properties(${NAME} PROPERTIES IMPORTED_IMPLIB ${${_LIB_VAR}})
    endif()
  endif()
endmacro()

nvpro2_find_vulkan_shared_library(shaderc_shared) # ShaderC library/SO and DLL
nvpro2_find_vulkan_shared_library(SPIRV-Tools-shared) # SPIRV-Tools library/SO and DLL

if(Vulkan_FOUND)
  message(STATUS "Vulkan found:")
  message(STATUS "  Version                   : ${Vulkan_VERSION}")
  message(STATUS "  Include Directory         : ${Vulkan_INCLUDE_DIRS}")
  message(STATUS "  Library Directory         : ${Vulkan_LIBRARY}")
  message(STATUS "  GLSLANG Validator         : ${Vulkan_GLSLANG_VALIDATOR_EXECUTABLE}")
  message(STATUS "  ShaderC Import Library    : ${Vulkan_SHADERC_SHARED_LIBRARY}")
  message(STATUS "  ShaderC Shared Library    : ${Vulkan_SHADERC_SHARED_DLL}")
  message(STATUS "  SPIRV-Tools Import Library: ${Vulkan_SPIRV_TOOLS_SHARED_LIBRARY}")
  message(STATUS "  SPIRV-Tools Shared Library: ${Vulkan_SPIRV_TOOLS_SHARED_DLL}")
  if(Vulkan_VOLK_DIR)
    message(STATUS "  Volk Directory        : ${Vulkan_VOLK_DIR}")
  else()
    message(STATUS "  Volk Directory        : Using fallback")
  endif()
else()
	message(FATAL_ERROR "Vulkan not found.")
endif()

set(VulkanExtra_FOUND ${Vulkan_FOUND})
