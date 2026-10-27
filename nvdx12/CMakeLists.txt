# NVDX12 contains Direct3D 12 helpers.
set(LIB_NAME nvdx12)

# The Agility SDK provides d3d12.h and the runtime (D3D12Core.dll); the find
# module downloads the pinned NuGet package or uses DX12AgilitySdk_ROOT.
find_package(DX12AgilitySdk REQUIRED)

# nvdx12 exports D3D12SDKVersion and D3D12SDKPath ".\D3D12\": the runtime
# (${DX12AgilitySdk_DLLS}) goes to the D3D12 folder next to the executable
# (copy_to_runtime_and_install ... SUBDIRECTORY "D3D12"). Applications that
# export the two symbols themselves (for instance to select a preview SDK)
# turn this off.
option(NVDX12_EXPORT_AGILITY_SDK "Export the Agility SDK version and path from nvdx12" ON)
if(NVDX12_EXPORT_AGILITY_SDK AND NOT DX12AgilitySdk_SDK_VERSION)
  message(FATAL_ERROR
    "nvdx12 exports D3D12SDKVersion but DX12AgilitySdk_SDK_VERSION is empty. "
    "Set it to the SDK's D3D12SDKVersion or turn NVDX12_EXPORT_AGILITY_SDK off."
  )
endif()

# Collect all source files in the library directory
file(GLOB LIB_SOURCES CONFIGURE_DEPENDS "*.cpp" "*.hpp")
source_group("Source Files" FILES ${LIB_SOURCES})

# HLSL side of the shader contract (shaders/bindless.h), listed for the IDE
file(GLOB LIB_SHADERS CONFIGURE_DEPENDS "shaders/*.h")
source_group("Shaders" FILES ${LIB_SHADERS})
set_source_files_properties(${LIB_SHADERS} PROPERTIES HEADER_FILE_ONLY TRUE)

# Define the library
add_library(${LIB_NAME} STATIC ${LIB_SOURCES} ${LIB_SHADERS})

# All libraries use nvpro_core2 as their include root
cmake_path(GET CMAKE_CURRENT_LIST_DIR PARENT_PATH PARENT_DIR)

# context.cpp exports D3D12SDKVersion of the SDK found above; the built-in
# kernels include <nvdx12/shaders/bindless.h> from the nvpro_core2 root.
target_compile_definitions(${LIB_NAME} PRIVATE
  "NVDX12_D3D12SDKVERSION=${DX12AgilitySdk_SDK_VERSION}"
  "NVDX12_EXPORT_AGILITY_SDK=$<BOOL:${NVDX12_EXPORT_AGILITY_SDK}>"
  "NVDX12_SHADER_INCLUDE_DIR=\"${PARENT_DIR}\"")

# d3dx12 helpers: next to d3d12.h (flat SDK), in the NuGet d3dx12/ folder,
# or the copy under third_party/dxh.
set(_NVDX12_D3DX12_DIRS)
if(EXISTS "${DX12AgilitySdk_INCLUDE_DIR}/d3dx12/d3dx12.h")
  list(APPEND _NVDX12_D3DX12_DIRS "${DX12AgilitySdk_INCLUDE_DIR}/d3dx12")
elseif(NOT EXISTS "${DX12AgilitySdk_INCLUDE_DIR}/d3dx12.h")
  list(APPEND _NVDX12_D3DX12_DIRS "${PARENT_DIR}/third_party/dxh/include/directx")
endif()

target_include_directories(${LIB_NAME}
  PUBLIC
    ${PARENT_DIR}
    ${DX12AgilitySdk_INCLUDE_DIR}
    ${_NVDX12_D3DX12_DIRS}
)

target_link_libraries(
  ${LIB_NAME}
  PUBLIC
    nvutils
    glm                         # Math library
    ${DX12AgilitySdk_LIBRARIES} # d3d12.lib, dxgi.lib
    dxguid                      # IIDs and the experimental-feature GUIDs
    d3d12ma                     # D3D12 Memory Allocator (header; d3d12ma.cpp compiles it)
    offsetAllocator             # OffsetAllocator library (BufferSubAllocator)
)

add_library(nvpro2::${LIB_NAME} ALIAS ${LIB_NAME})
set_property(TARGET ${LIB_NAME} PROPERTY FOLDER "nvpro_core2")
