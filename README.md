# nvpro_core2

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

A modular C++ framework for Vulkan 1.4+ and OpenGL 4.6 development, designed as the successor to [nvpro_core](https://github.com/nvpro-samples/nvpro_core). This framework provides a collection of static libraries that make it easier to create high-performance research and sample graphics applications.

## Overview

`nvpro_core2` is a collection of modular libraries that provide essential functionality for graphics development.

Vulkan:
- [**nvvk**](nvvk): Vulkan helper functions and abstractions
- [**nvvkglsl**](nvvkglsl): Vulkan GLSL compiler support (uses shaderc from Vulkan SDK)
- [**nvvkgltf**](nvvkgltf): GLTF model loading and rendering support
- [**nvapp**](nvapp): Application framework and window management
- [**nvshaders**](nvshaders): Useful shaders and functions for BxDFs, skies, tonemapping, and more. Many functions can also be used from GLSL or C++ using slang_types.h
- [**nvshaders_host**](nvshaders_host): Host code for some pre-defined shader pipelines

OpenGL:
- [**nvgl**](nvgl): OpenGL helper functions and legacy application framework

Generic:
- [**nvutils**](nvutils): Utility functions and common data structures
- [**nvgui**](nvgui): GUI components and ImGui integration
- [**nvslang**](nvslang): Slang compiler support (downloads appropriate slang library)
- [**nvimageformats**](nvimageformats): DDS and KTX2 image libraries
- [**nvaftermath**](nvaftermath): NVIDIA Aftermath integration for crash analysis (library has additional features if NVIDIA Aftermath SDK is found, otherwise basic functionality)
- [**nvgpu_monitor**](nvgpu_monitor): GPU performance monitoring using NVML (requires the component from CUDA Toolkit)
- [**nvnsight**](nvnsight): NVIDIA Nsight Graphics integration with NVTX profiling support (conditional, can be disabled)

`nvpro_core2`'s code is designed so you can extract and use functions from it in your own projects without too much modification.

Key improvements in `nvpro_core2` compared to previous [nvpro_core](https://github.com/nvpro-samples/nvpro_core):
* Vulkan `1.4` usage throughout all `nvvk` helpers
  * Latest core functions, structs and 64-bit flags
  * Adds support for dynamic rendering and states, timeline semaphores, and more
* Slang support for common shading functions and a rich set of glTF-compatible materials
* CMake: cleaner and faster system with separate static libraries, rather than a single monolithic library
* CMake: `copy_to_runtime_and_install(AUTO)` eases dealing with DLL dependencies
* Unified classes that had competing APIs in the past
* `nvapp` serves as centralized structure of Vulkan applications

## Requirements

- [Vulkan 1.4 SDK](https://vulkan.lunarg.com/sdk/home)
- [CMake 3.22 or higher](https://cmake.org/download/)
- 64-bit Windows or Linux OS
- Compiler supporting basic C++20 features 
  - On Windows, MSVC 2019 is our minimum compiler.
  - On Linux, GCC 10.5 or Clang 14 are our minimum compilers.
  - Other compilers supporting their features should also work.
- On Linux, you'll also need a few system libraries and headers. The following line installs the required libraries on distros that use `apt` as their package manager; on other distros, similar commands should work:

```
sudo apt install libx11-dev libxcb1-dev libxcb-keysyms1-dev libxcursor-dev libxi-dev libxinerama-dev libxrandr-dev libxxf86vm-dev libtbb-dev libgl-dev
```

## Building a Sample

1. Install the required development tools (see [Requirements](#Requirements) above).

2. Optionally, clone `nvpro_core2` next to your samples or add it as a submodule.

    This is optional because samples automatically download nvpro_core2 if they can't find it. If you're building multiple samples, though, it's more efficient to clone nvpro_core2 once where samples can find it instead of downloading it multiple times.

3. Configure and build:

    ```bash
    cd TheSampleFolder

    # Run CMake's configure step; this will generate your build system's files
    cmake -S . -B build

    # Build the project
    cmake --build build --config Release --parallel
    ```

4. Optionally, build a portable version by running the `install` target:
   
    ```bash
    cmake --build build --parallel --target install
    ```
    This will be created in the `_install` folder, and includes all runtime dependencies so it can be copied to other computers.

## Coding with nvpro_core2

### Starting a New Sample

For new samples, the easiest way to get started is to copy the [project template](./project_template). It's set up to build right away, and is easy to customize
to your project's needs.

### Adding `nvpro_core2` to an Existing CMake Project

First, find and include nvpro_core2. There's an easy way to do this, and a
manual but more customizable way.

* Easy Way:
  
    1. Copy [`FindNvproCore2.cmake`](./project_template/cmake/FindNvproCore2.cmake) to a `cmake/` subdirectory of your project.

    2. Add the following code to your `CMakeLists.txt`. This will automatically download `nvpro_core2` if not found, add its targets, and set appropriate defaults for most samples:

        ```cmake
        # Add the cmake folder to the module path
        list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
        find_package(NvproCore2 REQUIRED)
        ```

* Manual Way:

    1. Clone `nvpro_core2` next to your project (`git clone https://github.com/nvpro-samples/nvpro_core2.git`) or add it as a submodule (`git submodule add https://github.com/nvpro-samples/nvpro_core2.git`).

    2. Manually find and include its `Setup.cmake` file:

        ```cmake
        find_path(NVPRO_CORE2_DIR
          NAMES cmake/Setup.cmake
          PATHS ${CMAKE_CURRENT_LIST_DIR}/nvpro_core2
                ${CMAKE_CURRENT_LIST_DIR}/../nvpro_core2
                ${CMAKE_CURRENT_LIST_DIR}/../../nvpro_core2
          REQUIRED
          DOC "Path to nvpro_core2"
        )
        include(${NVPRO_CORE2_DIR}/cmake/Setup.cmake)
        ```

        If you find the defaults in Setup.cmake cause issues, you can instead call `add_subdirectory(${NVPRO_CORE2_DIR})` to only add `nvpro_core2`'s targets.

Whichever option you choose, you'll then need to link against the libraries you
use:

```cmake
target_link_libraries(${PROJECT_NAME} PRIVATE
  nvpro2::nvapp
  nvpro2::nvutils
  nvpro2::nvvk
  # Add other libraries as needed
)
```

Finally, if you're using libraries that rely on DLLs like nvslang, you'll need
to copy them to the bin and install directories.

1. If you didn't use `FindNvproCore2` or `Setup.cmake`, add a call to `include(${NVPRO_CORE2_DIR}/cmake/Setup.cmake)` before the next statement. 

2. Then at the end of your script, call:

```cmake
copy_to_runtime_and_install(${PROJECT_NAME} AUTO)
```

## License

`nvpro_core2` is licensed under [Apache 2.0](LICENSE).

## Third-Party Libraries

This project embeds or includes (as filtered [third_party](third_party) subfolders) several open-source libraries and/or code derived from them. All such libraries' licenses are included in the [PACKAGE-LICENSES](PACKAGE-LICENSES) folder.

Key third-party dependencies include:
- [{fmt}](https://github.com/fmtlib/fmt) - Formatting library
- [Dear ImGui](https://github.com/ocornut/imgui) - Immediate mode GUI
- [GLFW](https://github.com/glfw/glfw) - Window management
- [GLM](https://github.com/g-truc/glm) - GLSL-style mathematics library
- [OffsetAllocator](https://github.com/sebbbi/OffsetAllocator) - Realtime O(1) offset allocator
- [stb](https://github.com/NBickford-NV/stb) - Multiple utilities including image loading
- [tinygltf](https://github.com/syoyo/tinygltf) - GLTF loading
- [volk](https://github.com/zeux/volk) - Vulkan loader
- [Vulkan Memory Allocator](https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator) - Vulkan memory management
- [zstd](https://github.com/facebook/zstd) - Compression library

## Contributing

Merge requests to `nvpro_core2` are welcome, and use the Developer Certificate of Origin (https://developercertificate.org; included in [CONTRIBUTING](CONTRIBUTING)). When committing, please certify that your contribution adheres to the DCO and use `git commit --sign-off`. Thank you!

## Support

- For bug reports and feature requests, please use the [GitHub Issues](https://github.com/nvpro-samples/nvpro_core2/issues) page.
- For general questions and discussions, please use the [GitHub Discussions](https://github.com/nvpro-samples/nvpro_core2/discussions) page.