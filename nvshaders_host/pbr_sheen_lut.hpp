/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <span>

#include <vulkan/vulkan_core.h>
#include "nvvk/context.hpp"
#include "nvvk/resource_allocator.hpp"
#include "nvvk/sampler_pool.hpp"


namespace nvshaders {

/*-------------------------------------------------------------------------------------------------
# Function `bakeCharlieSheenLut`
> One-time bake of the Charlie sheen directional-albedo LUT keyed on (NdotV, sheenRoughness).

Runs the `nvshaders/hdr_charlie_brdf_lut.slang` compute shader once to fill a `dim.width x
dim.height` R16G16_SFLOAT image, then transitions it to `VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL`
with a linear sampler attached. The bake stores a single scalar — the Charlie sheen directional
albedo in [0,1] — duplicated into both `.r` and `.g`; `.b`/`.a` are not present in the storage
format (sampling them returns 0/1). The R16G16 format is chosen for parity with the companion
GGX BRDF LUT (`hdr_integrate_brdf.slang`, which legitimately needs two channels) and for
broader storage-image support than R16_SFLOAT; the duplicated `.g` also lets ports of Khronos
shader code keep their original `.r`/`.g` swizzle without modification (the Khronos reference
samples `.b` from one of its two LUTs — that swizzle must become `.r` or `.g` here).

Consumed at shade time by:
  * `nvshaders/pbr_ibl_eval.h.slang::getIBLCharlie` (IBL sheen).
  * `nvshaders/pbr_ibl_eval.h.slang::composeIBLAnalyticKhronos` (IBL sheen energy compensation).
  * `nvshaders/pbr_ggx_microfacet.h.slang::albedoSheenScalingLUT` (per-light sheen scaling, called
    from `pbr_direct_eval.h.slang::bsdfEvaluatePunctualKhronos`).

## Inputs
- `allocator`:   resource allocator that owns the returned image.
- `samplerPool`: sampler pool that provides the linear sampler (refcounted; release via
                 `samplerPool.releaseSampler(image.descriptor.sampler)` on teardown).
- `queueInfo`:   graphics/compute queue used for the single-time dispatch.
- `spirvCode`:   compiled SPIR-V of `nvshaders/hdr_charlie_brdf_lut.slang` (typically embedded
                 via the sample's autogen headers).
- `dim`:         LUT resolution. 64x64 is the typical choice (matches the glTF Sample Renderer
                 and keeps interpolation tight across `[0, 1]^2`).

## Returns
An `nvvk::Image` ready to be written to a descriptor set binding. The caller OWNS the image and
must release it at shutdown:
```cpp
samplerPool.releaseSampler(lut.descriptor.sampler);
allocator.destroyImage(lut);
```

## Threading
Dispatches a single compute shader on a transient command pool/queue submit and blocks until
completion. Safe to call during scene load; NOT safe during rendering.
-------------------------------------------------------------------------------------------------*/
nvvk::Image bakeCharlieSheenLut(nvvk::ResourceAllocator&         allocator,
                                nvvk::SamplerPool&               samplerPool,
                                const nvvk::QueueInfo&           queueInfo,
                                const std::span<const uint32_t>& spirvCode,
                                VkExtent2D                       dim = {64, 64});

}  // namespace nvshaders
