/*
 * Copyright (c) 2023-2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */


/*
 * Dummy source file for nvnsight library
 * 
 * This file exists solely to make nvnsight a STATIC library so it shows up
 * in Visual Studio. The actual functionality is provided by the header-only
 * nsightevents.hpp file.
 */


/*
 * To use NVTX tools:
 * 
 * 1. Include the nvnsight module in your CMakeLists.txt:
 *    target_link_libraries(your_target PRIVATE nvpro2::nvnsight)
 * 
 * 2. Build and run the application
 * 
 * 3. To see the NVTX markers in NVIDIA Nsight Graphics:
 *    - Launch Nsight Graphics
 *    - Start a new capture
 *    - Run your application
 *    - The markers and ranges will appear in the timeline view
 * 
 * 4. When NVTX is disabled (NSIGHT_ENABLE_NVTX=OFF), all macros become no-ops
 *    with zero performance impact.
 */

// Include the main header to ensure it compiles correctly
#include "nvnsight/nsightevents.hpp"

#include <thread>
#include <chrono>

// Dummy function to prevent linker warnings
[[maybe_unused]] static void usage_NVTX()
{
  // This function is never called, but prevents the library from being empty
  // which could cause linker issues on some platforms

  // Example function that demonstrates various NVTX usage patterns
  {
    // Automatic function profiling - creates a range for the entire function
    NXPROFILEFUNC("exampleRenderFrame");

    // Simple marker
    NX_MARK("Frame Start");

    // Colored range for clear operation
    NX_RANGEPUSHCOL("Clear Screen", 0xFF0000FF);  // Red
    {
      // Simulate clear operation
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    NX_RANGEPOP();

    // Colored range for geometry rendering
    NX_RANGEPUSHCOL("Render Geometry", 0xFF00FF00);  // Green
    {
      // Simulate geometry rendering
      std::this_thread::sleep_for(std::chrono::milliseconds(5));

      // Nested range for specific geometry operations
      NX_RANGEPUSHCOL("Process Vertices", 0xFFFF0000);  // Blue
      {
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
      }
      NX_RANGEPOP();
    }
    NX_RANGEPOP();

    // Range with payload for detailed profiling
    {
      NXPROFILEFUNCCOL2("Post Processing", 0xFFFFFF00, 123);  // Yellow with payload
      {
        std::this_thread::sleep_for(std::chrono::milliseconds(3));
      }
    }

    // Simple marker at the end
    NX_MARK("Frame Complete");
  }

  // Example of using range start/end pattern
  {
    NXPROFILEFUNC("exampleRangePattern");

    // Start a range and get an ID
    NX_RANGE rangeId = NX_RANGESTART("Custom Range");

    // Do some work
    std::this_thread::sleep_for(std::chrono::milliseconds(10));

    // End the range using the ID
    NX_RANGEEND(rangeId);
  }

  // Example of automatic function profiling with different colors
  {
    // Red function (default)
    {
      NXPROFILEFUNC("Red Function");
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    // Green function
    {
      NXPROFILEFUNCCOL("Green Function", 0xFF00FF00);
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    // Blue function with payload
    {
      NXPROFILEFUNCCOL2("Blue Function", 0xFF0000FF, 456);
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  }
}
