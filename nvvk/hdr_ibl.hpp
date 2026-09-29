/*
 * Copyright (c) 2022-2026, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */


#pragma once
//////////////////////////////////////////////////////////////////////////


#include <array>
#include <vector>
#include <assert.h>
#include <filesystem>

#include <span>

#include <vulkan/vulkan_core.h>

#include "../nvshaders/hdr_io.h.slang"  // shaderio::EnvAccel (used by buildEnvAliasmap)
#include "descriptors.hpp"
#include "env_aliasmap.hpp"  // re-export nvvk::buildEnvAliasmap for callers already including hdr_ibl.hpp
#include "resource_allocator.hpp"
#include "sampler_pool.hpp"
#include "uploader_interface.hpp"


namespace nvvk {
class Context;


/*-------------------------------------------------------------------------------------------------
# class nvvkhl::HdrEnv

High-Dynamic-Range (HDR) environment map used for Image-Based Lighting (IBL).

>  Load an environment image (HDR) and create an acceleration structure for important light sampling.
  
-------------------------------------------------------------------------------------------------*/
class HdrIbl
{
public:
  HdrIbl() = default;
  ~HdrIbl() { assert(m_device == VK_NULL_HANDLE); }

  void init(nvvk::ResourceAllocator* allocator, nvvk::SamplerPool* samplerPool);
  void deinit();

  void loadEnvironment(VkCommandBuffer cmd, nvvk::CmdUploaderInterface& staging, const std::filesystem::path& hdrImage, bool enableMipmaps = false);

  // Same, from pixels the caller has already decoded: tightly packed RGBA32F, `size.width *
  // size.height * 4` floats, row-major from the top-left.
  //
  // **The span is written to.** Building the alias table replaces each texel's alpha with its
  // sampling PDF, in place -- the CPU counterpart of what env_write_pdf.slang does on the GPU.
  // Pass a buffer you own and do not need afterwards; the file overload passes stb's.
  //
  // This exists so a caller can support an image format `nvvk` does not. Radiance .hdr is decoded
  // here because stb_image is header-only and costs this library no dependency; a format whose
  // decoder is a compiled library (OpenEXR, for one) would drag that library into every consumer
  // of nvvk, which is too high a price for a file format. Decode it on your side and hand the
  // floats over instead.
  //
  // An empty span, a zero extent, or a length that disagrees with the extent produces the same
  // 1x1 dummy environment a failed file load does, so callers need no separate failure path.
  void loadEnvironment(VkCommandBuffer cmd, nvvk::CmdUploaderInterface& staging, std::span<float> rgbaPixels, VkExtent2D size, bool enableMipmaps = false);
  void destroyEnvironment();

  // Additional one-time setup for the GPU-image code path (updateFromGpuImage): the SPIR-V of
  // nvshaders/env_write_pdf.slang. Callers that only ever load HDRs from a file can skip this and
  // pay no GPU cost. Idempotent -- calling it twice is a no-op after the pipeline has been created.
  void initGpuWriter(std::span<const uint32_t> spirvWritePdf);

  // Refresh the HDR + importance-sampling descriptor set to sample from a caller-owned lat-long
  // image, and rebuild the alias table from a caller-owned importance storage buffer. Requires
  // a prior initGpuWriter() call.
  //
  // Blocking on the host: importance is copied to a host-visible staging buffer, the queue is
  // waited on, and the CPU Vose builds the alias table. See § Preview vs Commit in the plan
  // -- this is a parameter-commit path (~20 ms on a ~1024x512 image), not a per-frame path.
  //
  // Neither `latlongImage` nor `importanceBuf` is taken over. The producer (typically
  // EnvBaker) retains ownership across the lifetime of this HdrIbl. `latlongImage` must be in
  // `VK_IMAGE_LAYOUT_GENERAL` on entry, with `latlongImage.descriptor.imageLayout` reflecting
  // this: `env_write_pdf.slang` binds it as `VK_DESCRIPTOR_TYPE_STORAGE_IMAGE` to write the
  // PDF into the alpha channel, which per spec requires `GENERAL`. The descriptor's `sampler`
  // must be set too, since the image is rebound into `eHdr`, a combined image sampler. The
  // writes that produced the image may come from any stage, as long as they were submitted
  // earlier on the queue `queueInfo` names. On exit the image is still
  // in `GENERAL` -- the producer pipeline (bake write -> env_write_pdf write -> sampler read)
  // keeps it there end-to-end. The internal `m_accelImpSmpl` alias buffer is recreated
  // on every call and rebound into `eImpSamples`; the descriptor for `eHdr` is rebound to
  // point at the external `latlongImage`.
  //
  // `importanceBuf` must hold one float per cell of `samplingGrid`, row-major: the sum, over the
  // cell's texels, of `solidAngle(row) * max(color.rgb)` -- the per-texel importance metric the CPU
  // path computes in `createEnvironmentAccel`. With the grid equal to the image, that is one value
  // per texel. Summing is the producer's job because the producer already has every texel in
  // hand: a workgroup-local reduction in the bake costs next to nothing, where a separate pass here
  // would read an image-sized buffer back in. `texelSolidAngle` and `latlongToDir` in
  // `nvshaders/functions.h.slang` are what a producing shader must use to end up with an image and
  // a weighting this call can consume; how it binds its own outputs is its business.
  //
  // `samplingGrid` is the resolution the alias table is built at, independently of the image's.
  // The CPU builds the alias table over the grid's cells, and a sample picks a cell and then a
  // direction uniformly in solid angle within it -- so the build costs what the grid costs, not
  // what the image costs. It suits a smooth source, such as a sky whose sun is sampled as a light
  // of its own; a small bright feature would have its probability spread over the whole cell. The
  // grid must divide `imageSize` exactly -- a grid that does not is refused with an error and the
  // current environment stays bound -- and a zero extent means per-texel. Shaders must sample with
  // the grid, not the image size -- see getSamplingGrid().
  //
  // `queueInfo` is used for the internal transient command pool that carries the readback copy
  // and the env_write_pdf dispatch. It must reference the same queue family the caller uses for
  // compute submits.
  void updateFromGpuImage(const nvvk::QueueInfo& queueInfo,
                          nvvk::Image&           latlongImage,
                          const nvvk::Buffer&    importanceBuf,
                          VkExtent2D             imageSize,
                          VkExtent2D             samplingGrid = {0, 0});

  float              getIntegral() const { return m_integral; }
  float              getAverage() const { return m_average; }
  bool               isValid() const { return m_valid; }
  const nvvk::Buffer getEnvAccel() const { return m_accelImpSmpl; }

  // HDR + importance sampling
  inline VkDescriptorSetLayout getDescriptorSetLayout() const { return m_descPack.getLayout(); }
  inline VkDescriptorSet       getDescriptorSet() const { return m_descPack.getSet(0); }
  const nvvk::Image&           getHdrImage() { return m_texHdr; }  // The loaded HDR texture
  VkExtent2D                   getHdrImageSize() const { return m_hdrImageSize; }
  // Resolution of the alias table: what a shader passes to environmentSample() as the sampling
  // grid. The image size for an environment loaded from a file; possibly coarser after
  // updateFromGpuImage().
  VkExtent2D getSamplingGrid() const { return m_samplingGrid; }

private:
  VkDevice                 m_device{VK_NULL_HANDLE};
  nvvk::ResourceAllocator* m_alloc{nullptr};
  nvvk::SamplerPool*       m_samplerPool{};

  float      m_integral{1.F};
  float      m_average{1.F};
  bool       m_valid{false};
  VkExtent2D m_hdrImageSize{1, 1};
  VkExtent2D m_samplingGrid{1, 1};

  // Resources
  nvvk::Image          m_texHdr;
  nvvk::Buffer         m_accelImpSmpl;
  nvvk::DescriptorPack m_descPack;
  // True when m_texHdr was allocated by loadEnvironment() and this class owns it; false when
  // updateFromGpuImage() has bound an externally owned image into the descriptor set instead.
  bool m_hdrImageIsOwned{false};

  // env_write_pdf.slang pipeline; created by initGpuWriter(), reused by every
  // updateFromGpuImage() call. Empty (VK_NULL_HANDLE) if initGpuWriter was never called.
  VkPipeline           m_writePdfPipeline{VK_NULL_HANDLE};
  VkPipelineLayout     m_writePdfLayout{VK_NULL_HANDLE};
  nvvk::DescriptorPack m_writePdfPack;

  void createDescriptorSetLayout();
  // Rewrite `m_descPack.getSet(0)` so `eHdr` samples from `envImage` and `eImpSamples` reads
  // from `aliasBuf`. Called by both loadEnvironment (owned image + alias) and
  // updateFromGpuImage (external image + regenerated alias).
  void writeDescriptorSet(const nvvk::Image& envImage, const nvvk::Buffer& aliasBuf);
};

}  // namespace nvvk
