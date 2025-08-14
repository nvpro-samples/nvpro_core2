# nvnsight - NVIDIA Nsight Graphics Integration

This directory provides integration with NVIDIA Nsight Graphics for profiling and debugging GPU applications.

## Overview

The `nvnsight` module provides conditional NVTX (NVIDIA Tools Extension) support for adding performance markers and ranges to your code. These markers appear in NVIDIA Nsight Graphics timeline view, helping you profile and debug GPU applications.

## Features

- **Conditional NVTX Support**: NVTX functionality can be enabled/disabled via CMake option
- **Header-only Integration**: No runtime dependencies when NVTX is disabled
- **Automatic No-ops**: All NVTX macros become no-ops when not running through NVIDIA tools
- **Easy Integration**: Simple include and link against `nvpro2::nvnsight`

## Usage

### 1. Include the Header

```cpp
#include "nvnsight/nsightevents.hpp"
```

### 2. Link Against nvnsight

In your CMakeLists.txt:
```cmake
target_link_libraries(your_target PRIVATE nvpro2::nvnsight)
```

### 3. Use NVTX Macros

```cpp
void renderFrame() {
    NXPROFILEFUNC("renderFrame");
    
    NX_RANGEPUSHCOL("Clear", 0xFF0000FF);
    clearScreen();
    NX_RANGEPOP();
    
    NX_RANGEPUSHCOL("Geometry", 0xFF00FF00);
    renderGeometry();
    NX_RANGEPOP();
    
    NX_MARK("Frame Complete");
}
```

## Available Macros

- `NX_MARK(name)` - Add a simple marker
- `NX_RANGESTART(name)` / `NX_RANGEEND(id)` - Start/end a named range
- `NX_RANGEPUSH(name)` / `NX_RANGEPOP()` - Push/pop ranges (stack-based)
- `NX_RANGEPUSHCOL(name, color)` - Push a colored range
- `NXPROFILEFUNC(name)` - Automatic function profiling (red color)
- `NXPROFILEFUNCCOL(name, color)` - Colored function profiling
- `NXPROFILEFUNCCOL2(name, color, payload)` - Function profiling with payload

## Configuration

### Enable/Disable NVTX

By default, NVTX support is enabled. To disable it:

```cmake
set(NVP_ENABLE_NVTX OFF)
```

### CMake Option

- `NVP_ENABLE_NVTX` (default: ON) - Enable NVTX profiling support

## How It Works

1. **When NVTX is enabled**: The module links against NVTX3 and defines `NVP_SUPPORTS_NVTOOLSEXT`
2. **When NVTX is disabled**: All macros become no-ops with zero performance impact
3. **Runtime behavior**: NVTX calls are automatically redirected to NVIDIA tools when available

## Benefits

- **Visual Timeline**: See your markers and ranges in Nsight Graphics timeline
- **Performance Analysis**: Identify bottlenecks and measure execution time
- **Debugging**: Correlate GPU events with CPU code execution
- **Color Coding**: Use colors to categorize different types of operations
- **Zero Overhead**: No performance impact when not using NVIDIA tools

## Dependencies

- NVTX3 (automatically downloaded via repro_rules.json)
- NVIDIA Nsight Graphics (for visualization)

## Setup

If NVTX is not found during CMake configuration, you may need to run the repro script to download it:

```bash
cd nvpro_core2/third_party
python dev-rebuild-filtered-repos.py
```

## Example

See the main nvpro_core2 documentation for complete examples of using NVTX profiling in your applications.
