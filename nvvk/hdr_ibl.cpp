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


/*
 *  HDR sampling is loading an HDR image and creating an acceleration structure for 
 *  sampling the environment. 
 */

#include <array>
#include <cstring>
#include <memory>
#include <span>
#include <numeric>
#include <limits>

#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>

#include "nvshaders/slang_types.h"
#include "nvshaders/hdr_io.h.slang"

#include <nvutils/file_operations.hpp>
#include <nvutils/logger.hpp>
#include <nvutils/timers.hpp>
#include <stb/stb_image.h>

#include "check_error.hpp"
#include "commands.hpp"
#include "command_pools.hpp"
#include "compute_pipeline.hpp"
#include "debug_util.hpp"
#include "default_structs.hpp"
#include "descriptors.hpp"
#include "hdr_ibl.hpp"
#include "mipmaps.hpp"


namespace nvvk {
// Forward declaration -- definition lives at the bottom of the file. Kept as a forward decl so
// the members of HdrIbl that call it below don't need to know its full body up here.
std::vector<shaderio::EnvAccel> createEnvironmentAccel(float*& pixels, const uint32_t& width, const uint32_t& height, float& average, float& integral);


//--------------------------------------------------------------------------------------------------
// Initialize the HdrIbl and create the descriptor set layout
// The layout is created once and remains stable throughout the lifetime of the object
//
void HdrIbl::init(nvvk::ResourceAllocator* allocator, nvvk::SamplerPool* samplerPool)
{
  m_device      = allocator->getDevice();
  m_alloc       = allocator;
  m_samplerPool = samplerPool;

  // Create the descriptor set layout once - it describes the structure, not the content
  // This layout will remain stable even when loading different HDR environments
  createDescriptorSetLayout();
}

//--------------------------------------------------------------------------------------------------
// Clean up all resources including the descriptor set layout
//
void HdrIbl::deinit()
{
  destroyEnvironment();

  // The GPU-image pipelines are optional -- only populated when initGpuWriter() was called.
  // Vulkan destroys are no-ops on VK_NULL_HANDLE.
  vkDestroyPipeline(m_device, m_writePdfPipeline, nullptr);
  vkDestroyPipelineLayout(m_device, m_writePdfLayout, nullptr);
  m_writePdfPack.deinit();

  m_descPack.deinit();  // Destroy the descriptor set layout last
  m_device      = {};
  m_alloc       = nullptr;
  m_samplerPool = nullptr;
}

//--------------------------------------------------------------------------------------------------
// Loading the HDR environment texture (HDR) and create the important accel structure
//
// Note: enableMipmaps will create a mipmap chain for the environment texture, but does not generate the
//       mipmaps
void HdrIbl::loadEnvironment(VkCommandBuffer cmd, nvvk::CmdUploaderInterface& staging, const std::filesystem::path& hdrImage, bool enableMipmaps)
{
  nvutils::ScopedTimer st(__FUNCTION__);

  // Decode only. Everything downstream -- image, alias table, sampler, descriptors -- lives in the
  // pixels overload, so a caller decoding a format this library does not know goes through exactly
  // the same code from here on.
  int32_t     width{0};
  int32_t     height{0};
  int32_t     component{0};
  std::string fileContents;
  float*      pixels = nullptr;
  bool        decoded = !hdrImage.empty();

  if(decoded)
  {
    // Read the contents into memory so that we don't have to worry about text
    // encoding in the stbi filename API
    fileContents = nvutils::loadFile(hdrImage);
    if(fileContents.empty())
    {
      LOGW("File does not exist or is empty: %s\n", nvutils::utf8FromPath(hdrImage).c_str());
      decoded = false;
    }
    else if(fileContents.size() > std::numeric_limits<int>::max())
    {
      LOGW("File is too large for stb_image to load: %s\n", nvutils::utf8FromPath(hdrImage).c_str());
      decoded = false;
    }
  }

  if(decoded)
  {
    const stbi_uc* fileData = reinterpret_cast<const stbi_uc*>(fileContents.data());
    const int      fileSize = static_cast<int>(fileContents.size());

    if(!stbi_is_hdr_from_memory(fileData, fileSize))
    {
      LOGW("File is not HDR: %s\n", nvutils::utf8FromPath(hdrImage).c_str());
      decoded = false;
    }
    else
    {
      nvutils::ScopedTimer stLoad("Load image");
      pixels = stbi_loadf_from_memory(fileData, fileSize, &width, &height, &component, STBI_rgb_alpha);
      if(!pixels)
      {
        LOGW("stbi_loadf_from_memory failed: %s\n", nvutils::utf8FromPath(hdrImage).c_str());
        decoded = false;
      }
    }
  }

  // A failed decode passes an empty span, which the overload turns into the dummy environment.
  const size_t floatCount = decoded ? size_t(width) * size_t(height) * 4 : 0;
  loadEnvironment(cmd, staging, std::span<float>(pixels, floatCount), VkExtent2D{uint32_t(width), uint32_t(height)}, enableMipmaps);

  if(pixels)
    stbi_image_free(pixels);
}

void HdrIbl::loadEnvironment(VkCommandBuffer cmd, nvvk::CmdUploaderInterface& staging, std::span<float> rgbaPixels, VkExtent2D size, bool enableMipmaps)
{
  // Reject a span that disagrees with the extent rather than reading past it: the caller decoded
  // this, so a mismatch is their bug, and a dummy environment plus a warning beats a crash.
  const size_t expected = size_t(size.width) * size_t(size.height) * 4;
  m_valid               = !rgbaPixels.empty() && size.width > 0 && size.height > 0;
  if(m_valid && rgbaPixels.size() != expected)
  {
    LOGW("HdrIbl::loadEnvironment: %zu floats supplied for a %ux%u RGBA image, expected %zu\n", rgbaPixels.size(),
         size.width, size.height, expected);
    m_valid = false;
  }

  float*        pixels = m_valid ? rgbaPixels.data() : nullptr;
  const int32_t width  = int32_t(size.width);
  const int32_t height = int32_t(size.height);

  if(m_valid)
  {
    assert(pixels);
    VkDeviceSize buffer_size = width * height * 4 * sizeof(float);
    VkExtent2D   imgSize{static_cast<uint32_t>(width), static_cast<uint32_t>(height)};

    m_hdrImageSize = imgSize;
    m_samplingGrid = imgSize;  // createEnvironmentAccel builds one alias entry per texel

    VkFormat          format    = VK_FORMAT_R32G32B32A32_SFLOAT;
    VkImageCreateInfo imageInfo = DEFAULT_VkImageCreateInfo;
    imageInfo.extent            = {imgSize.width, imgSize.height, 1};
    imageInfo.format            = format;
    imageInfo.usage     = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    imageInfo.mipLevels = enableMipmaps ? nvvk::mipLevels(imgSize) : 1;

    {
      nvutils::ScopedTimer st("Generating Acceleration structure");
      {
        // Creating the importance sampling for the HDR and storing the info in the m_accelImpSmpl buffer
        std::vector<shaderio::EnvAccel> envAccel =
            createEnvironmentAccel(pixels, imgSize.width, imgSize.height, m_average, m_integral);

        NVVK_CHECK(m_alloc->createBuffer(m_accelImpSmpl, std::span(envAccel).size_bytes(), VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT));
        NVVK_CHECK(staging.appendBuffer(m_accelImpSmpl, 0, std::span(envAccel)));
        NVVK_DBG_NAME(m_accelImpSmpl.buffer);

        NVVK_CHECK(m_alloc->createImage(m_texHdr, imageInfo, DEFAULT_VkImageViewCreateInfo));
        NVVK_CHECK(staging.appendImage(m_texHdr, std::span(pixels, buffer_size / sizeof(float)), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL));
        NVVK_DBG_NAME(m_texHdr.image);
      }
    }
  }
  else
  {  // Create a Dummy image and buffer, such that the code can still run
    VkImageCreateInfo imageInfo = DEFAULT_VkImageCreateInfo;
    imageInfo.extent            = {1, 1, 1};
    imageInfo.format            = VK_FORMAT_R8G8B8A8_UNORM;
    imageInfo.usage     = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    imageInfo.mipLevels = 1;


    std::vector<uint8_t> color{255, 255, 255, 255};
    NVVK_CHECK(m_alloc->createImage(m_texHdr, imageInfo, DEFAULT_VkImageViewCreateInfo));
    NVVK_CHECK(staging.appendImage(m_texHdr, std::span(color), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL));
    NVVK_DBG_NAME(m_texHdr.image);
    m_hdrImageSize = {1, 1};
    m_samplingGrid = {1, 1};

    // One whole alias entry that always picks its own texel. The buffer used to be sized from the
    // 4-byte colour above, half an entry, so a shader sampling the dummy read past its end.
    std::vector<shaderio::EnvAccel> accel{{.alias = 0, .q = 1.0F}};
    NVVK_CHECK(m_alloc->createBuffer(m_accelImpSmpl, std::span(accel).size_bytes(), VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT));
    NVVK_CHECK(staging.appendBuffer(m_accelImpSmpl, 0, std::span(accel)));
    NVVK_DBG_NAME(m_accelImpSmpl.buffer);
  }

  // Sampler for the HDR
  // The map is parameterized with the U axis corresponding to the azimuthal angle, and V to the polar angle
  // Therefore, in U the sampler will use VK_SAMPLER_ADDRESS_MODE_REPEAT (default), but V needs to use
  // CLAMP_TO_EDGE to avoid having light leaking from one pole to another.
  VkSamplerCreateInfo samplerInfo{
      .sType        = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
      .magFilter    = VK_FILTER_LINEAR,
      .minFilter    = VK_FILTER_LINEAR,
      .mipmapMode   = VK_SAMPLER_MIPMAP_MODE_LINEAR,
      .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
      .maxLod       = VK_LOD_CLAMP_NONE,
  };
  NVVK_CHECK(m_samplerPool->acquireSampler(m_texHdr.descriptor.sampler, samplerInfo));

  // The HDR image + alias buffer both belong to this HdrIbl. Flip the ownership flag so
  // destroyEnvironment() knows to release them (updateFromGpuImage would set it back to
  // false for the external-image path).
  m_hdrImageIsOwned = true;

  // Update the descriptor set to point to the new resources
  // The layout was already created in init() and remains stable
  writeDescriptorSet(m_texHdr, m_accelImpSmpl);
}

// Destroy the resources for the environment
void HdrIbl::destroyEnvironment()
{
  if(m_alloc != nullptr)
  {
    // m_texHdr is only released when loadEnvironment() allocated it; updateFromGpuImage()
    // binds a caller-owned image and leaves m_hdrImageIsOwned false so we don't double-free.
    if(m_hdrImageIsOwned)
    {
      m_samplerPool->releaseSampler(m_texHdr.descriptor.sampler);
      m_alloc->destroyImage(m_texHdr);
    }
    m_texHdr          = {};
    m_hdrImageIsOwned = false;
    m_alloc->destroyBuffer(m_accelImpSmpl);
  }
}

//--------------------------------------------------------------------------------------------------
// Create the descriptor set layout structure
// This is called once during init() and defines the binding structure that remains stable
//
void HdrIbl::createDescriptorSetLayout()
{
  nvvk::DescriptorBindings bindings;
  bindings.addBinding(shaderio::EnvBindings::eHdr, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_ALL);  // HDR image
  bindings.addBinding(shaderio::EnvBindings::eImpSamples, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_ALL);  // importance sampling
  NVVK_CHECK(m_descPack.init(bindings, m_device, 1));
  NVVK_DBG_NAME(m_descPack.getLayout());
  NVVK_DBG_NAME(m_descPack.getPool());
  NVVK_DBG_NAME(m_descPack.getSet(0));
}

//--------------------------------------------------------------------------------------------------
// Rewrite descriptor set 0 so `eHdr` samples from `envImage` and `eImpSamples` reads from
// `aliasBuf`. Called by loadEnvironment (bound resources are member-owned) and by
// updateFromGpuImage (envImage is external, aliasBuf is member-owned). The bindings enum lives
// in `nvshaders/hdr_io.h.slang` so the shader and host descriptor layouts stay in sync.
//
void HdrIbl::writeDescriptorSet(const nvvk::Image& envImage, const nvvk::Buffer& aliasBuf)
{
  nvvk::DescriptorBindings bindings;
  bindings.addBinding(shaderio::EnvBindings::eHdr, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_ALL);
  bindings.addBinding(shaderio::EnvBindings::eImpSamples, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_ALL);

  nvvk::WriteSetContainer writeContainer;
  writeContainer.append(bindings.getWriteSet(shaderio::EnvBindings::eHdr, m_descPack.getSet(0)), envImage);
  writeContainer.append(bindings.getWriteSet(shaderio::EnvBindings::eImpSamples, m_descPack.getSet(0)), aliasBuf);
  vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(writeContainer.size()), writeContainer.data(), 0, nullptr);
}


//--------------------------------------------------------------------------------------------------
// One-time setup of the env_write_pdf compute pipeline. Callers only need this if they intend
// to use updateFromGpuImage(); the file-load code path never touches this pipeline. Idempotent
// -- a second call is a no-op after the first successful creation, so callers can invoke this
// unconditionally after init() without tracking state.
//
void HdrIbl::initGpuWriter(std::span<const uint32_t> spirvWritePdf)
{
  if(m_writePdfPipeline != VK_NULL_HANDLE)
    return;  // Already initialized

  // Descriptor set: the storage image env_write_pdf writes into (alpha channel), and the per-cell
  // importance it reads the PDF from.
  nvvk::DescriptorBindings bindings;
  bindings.addBinding(shaderio::EnvWritePdfBindings::eEnvWritePdfImage, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT);
  bindings.addBinding(shaderio::EnvWritePdfBindings::eEnvWritePdfCells, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT);
  NVVK_CHECK(m_writePdfPack.init(bindings, m_device, 1));
  NVVK_DBG_NAME(m_writePdfPack.getLayout());
  NVVK_DBG_NAME(m_writePdfPack.getPool());
  NVVK_DBG_NAME(m_writePdfPack.getSet(0));

  // Pipeline layout: descriptor set + push constant (imageSize + gridSize + invIntegral).
  const VkPushConstantRange pushConstantRange{
      .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT, .offset = 0, .size = sizeof(shaderio::EnvWritePdfPushConstant)};
  NVVK_CHECK(nvvk::createPipelineLayout(m_device, &m_writePdfLayout, {m_writePdfPack.getLayout()}, {pushConstantRange}));
  NVVK_DBG_NAME(m_writePdfLayout);

  VkShaderModuleCreateInfo moduleInfo = {
      .sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = spirvWritePdf.size_bytes(),
      .pCode    = spirvWritePdf.data(),
  };

  VkPipelineShaderStageCreateInfo stageInfo{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
  stageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  stageInfo.pName = "main";
  stageInfo.pNext = &moduleInfo;

  VkComputePipelineCreateInfo compInfo{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
  compInfo.layout = m_writePdfLayout;
  compInfo.stage  = stageInfo;

  NVVK_CHECK(vkCreateComputePipelines(m_device, {}, 1, &compInfo, nullptr, &m_writePdfPipeline));
  NVVK_DBG_NAME(m_writePdfPipeline);
}


//--------------------------------------------------------------------------------------------------
// Refresh the sampling descriptor set to point at a caller-owned lat-long image + regenerate
// the alias table from a caller-owned importance buffer. See the header comment on the
// declaration for the caller contract (blocking on host, ownership, layout expectations,
// importance-buffer format, sampling grid).
//
// Flow:
//   1. Check the grid, then release any HDR image + alias buffer the previous load owned.
//   2. Copy importanceBuf -- one value per grid cell -- into a host-visible staging buffer,
//      submit + wait.
//   3. Build the alias table over the cells on the CPU (Vose) via the same buildEnvAliasmap the
//      file path uses. Its cost is the grid's, not the image's.
//   4. Allocate a new host-visible m_accelImpSmpl and memcpy the alias entries into it.
//   5. Dispatch env_write_pdf to write each texel's cell PDF into the image's alpha channel,
//      submit + wait.
//   6. Rebind descriptors: eHdr -> latlongImage (external), eImpSamples -> m_accelImpSmpl.
//
void HdrIbl::updateFromGpuImage(const nvvk::QueueInfo& queueInfo,
                                nvvk::Image&           latlongImage,
                                const nvvk::Buffer&    importanceBuf,
                                VkExtent2D             imageSize,
                                VkExtent2D             samplingGrid)
{
  assert(m_writePdfPipeline != VK_NULL_HANDLE && "initGpuWriter() must be called before updateFromGpuImage()");
  assert(imageSize.width > 0 && imageSize.height > 0);

  nvutils::ScopedTimer st(__FUNCTION__);

  // A zero extent means per-texel. Otherwise the grid must divide the image, or texels would
  // straddle two cells and their PDF would match neither -- and the caller has already summed its
  // importance into this grid, so there is nothing sound to fall back to. Refuse, and leave the
  // current environment bound.
  if(samplingGrid.width == 0 || samplingGrid.height == 0)
    samplingGrid = imageSize;
  if(imageSize.width % samplingGrid.width != 0 || imageSize.height % samplingGrid.height != 0)
  {
    LOGE("HdrIbl::updateFromGpuImage: sampling grid %ux%u does not divide the %ux%u image; environment not updated\n",
         samplingGrid.width, samplingGrid.height, imageSize.width, imageSize.height);
    assert(false && "updateFromGpuImage: the sampling grid must divide the image");
    return;
  }

  // Free any owned image + the previous alias buffer. m_hdrImageIsOwned is reset to false.
  destroyEnvironment();

  m_hdrImageSize                = imageSize;
  m_samplingGrid                = samplingGrid;
  const uint32_t     texelCount = imageSize.width * imageSize.height;
  const uint32_t     cellCount  = samplingGrid.width * samplingGrid.height;
  const VkDeviceSize cellBytes  = VkDeviceSize(cellCount) * sizeof(float);

  // Transient command pool -- created and destroyed per call. Matches the HdrEnvDome::create
  // pattern; updateFromGpuImage runs on parameter commit, not per frame, so pool churn is fine.
  VkCommandPool                 transientCmdPool = VK_NULL_HANDLE;
  const VkCommandPoolCreateInfo poolInfo{
      .sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .flags            = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,
      .queueFamilyIndex = queueInfo.familyIndex,
  };
  NVVK_CHECK(vkCreateCommandPool(m_device, &poolInfo, nullptr, &transientCmdPool));
  NVVK_DBG_NAME(transientCmdPool);

  // -- Step 2: readback importanceBuf --------------------------------------------------------
  nvvk::Buffer stagingReadback;
  NVVK_CHECK(m_alloc->createBuffer(stagingReadback, cellBytes, VK_BUFFER_USAGE_2_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
                                   VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT));
  NVVK_DBG_NAME(stagingReadback.buffer);

  {
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    NVVK_CHECK(nvvk::beginSingleTimeCommands(cmd, m_device, transientCmdPool));

    // Whatever wrote the importance buffer (a bake's compute dispatch in a preceding submit,
    // typically) -> the copy. Same scope reasoning as the barrier in step 5.
    VkMemoryBarrier toCopy{
        .sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
    };
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &toCopy, 0,
                         nullptr, 0, nullptr);

    VkBufferCopy copy{.srcOffset = 0, .dstOffset = 0, .size = cellBytes};
    vkCmdCopyBuffer(cmd, importanceBuf.buffer, stagingReadback.buffer, 1, &copy);

    // Make the copy visible to host reads.
    VkMemoryBarrier toHost{
        .sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_HOST_READ_BIT,
    };
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &toHost, 0, nullptr, 0, nullptr);

    nvvk::endSingleTimeCommands(cmd, m_device, transientCmdPool, queueInfo.queue);
  }

  // -- Step 3: build alias table on CPU ------------------------------------------------------
  // Each cell's value is the sum of `solidAngle(row) * max(color.rgb)` over its texels -- the
  // per-texel metric the CPU path computes -- so the readback feeds buildEnvAliasmap without any
  // host-side re-weighting, and with a grid equal to the image this is the file path's table.
  //
  // The pipeline barrier above orders TRANSFER_WRITE against HOST_READ, but the Vulkan spec
  // additionally requires an invalidate on non-coherent mapped memory before the host reads
  // it. VMA's choice of memory type for `AUTO_PREFER_HOST + HOST_ACCESS_RANDOM` is a hint --
  // usually coherent on desktop with resizable BAR, not guaranteed everywhere. autoInvalidateBuffer
  // is a no-op when the range is already coherent, so this costs nothing in the common case.
  m_alloc->autoInvalidateBuffer(stagingReadback);

  // Read the importance straight out of the staging mapping rather than copying it into a vector
  // first: buildEnvAliasmap only streams it, and the buffer is released below once it is done.
  const std::span<const float> importance(reinterpret_cast<const float*>(stagingReadback.mapping), cellCount);

  // Not a vector: this is written before it is read, and value-initialising it would be a memset
  // for nothing. It stays in ordinary memory rather than being built directly in the alias buffer's
  // mapping, because the Vose merge reads entries back as it goes and that mapping may be
  // write-combined.
  std::unique_ptr<shaderio::EnvAccel[]> envAccel = std::make_unique_for_overwrite<shaderio::EnvAccel[]>(cellCount);
  m_integral                                     = buildEnvAliasmap(importance, std::span(envAccel.get(), cellCount));
  m_alloc->destroyBuffer(stagingReadback);
  if(m_integral == 0.0f)
  {
    m_integral = 1.0f;
  }
  // Approximation of the CIE-luminance average the file path computes: this integral is in
  // max-channel * solid-angle units rather than CIE-luminance units, but the ratio is roughly
  // constant across physical HDRs and only feeds tonemapper auto-exposure heuristics. Divided by
  // the image's texel count, not the grid's: the integral is the same either way, so this keeps
  // the value independent of the grid.
  m_average = m_integral / static_cast<float>(texelCount);
  m_valid   = true;

  // -- Step 4: allocate + upload alias buffer -----------------------------------------------
  // Host-visible so we can memcpy directly without a second staging round-trip. For discrete
  // GPUs with a resizable BAR this lands in device-local memory; otherwise it stays in system
  // memory. Either way the shader access pattern (one read per ray) is not the hot path.
  const VkDeviceSize aliasBytes = VkDeviceSize(cellCount) * sizeof(shaderio::EnvAccel);
  NVVK_CHECK(m_alloc->createBuffer(m_accelImpSmpl, aliasBytes, VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_AUTO,
                                   VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT));
  NVVK_DBG_NAME(m_accelImpSmpl.buffer);
  std::memcpy(m_accelImpSmpl.mapping, envAccel.get(), aliasBytes);

  // Symmetric to the invalidate in Step 3: the Vulkan spec requires a flush on non-coherent
  // host-visible memory after the CPU write, before any queue submit reads the buffer via the
  // `eImpSamples` descriptor. The queue-submit host-write-ordering guarantee only makes
  // already-flushed writes visible to the device; it does not itself flush a non-coherent CPU
  // cache. autoFlushBuffer is a no-op on the coherent memory VMA typically picks for
  // AUTO + HOST_ACCESS_SEQUENTIAL_WRITE on desktop with ReBAR.
  m_alloc->autoFlushBuffer(m_accelImpSmpl);

  // -- Step 5: dispatch env_write_pdf ------------------------------------------------------
  // The caller must have transitioned latlongImage to VK_IMAGE_LAYOUT_GENERAL (a bake writes it
  // as a storage image, so it is already GENERAL on entry). env_write_pdf binds the image as
  // VK_DESCRIPTOR_TYPE_STORAGE_IMAGE for its alpha-channel write, which per Vulkan spec requires
  // GENERAL -- assert loudly so a caller passing anything else does not silently launder its
  // descriptor.imageLayout through this path.
  assert(latlongImage.descriptor.imageLayout == VK_IMAGE_LAYOUT_GENERAL
         && "updateFromGpuImage requires latlongImage in VK_IMAGE_LAYOUT_GENERAL on entry");

  // Step 6 binds this same image into `eHdr`, a COMBINED_IMAGE_SAMPLER, so the caller's descriptor
  // must carry a sampler. We never fill one in -- the image is the caller's, and so is the
  // filtering/addressing choice -- and a null sampler would otherwise surface much later as a
  // validation error on an unrelated draw.
  assert(latlongImage.descriptor.sampler != VK_NULL_HANDLE
         && "updateFromGpuImage requires latlongImage.descriptor.sampler to be set by the caller");

  nvvk::WriteSetContainer writeContainer;
  writeContainer.append(m_writePdfPack.makeWrite(shaderio::EnvWritePdfBindings::eEnvWritePdfImage), latlongImage);
  writeContainer.append(m_writePdfPack.makeWrite(shaderio::EnvWritePdfBindings::eEnvWritePdfCells), importanceBuf);
  vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(writeContainer.size()), writeContainer.data(), 0, nullptr);

  {
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    NVVK_CHECK(nvvk::beginSingleTimeCommands(cmd, m_device, transientCmdPool));

    // The label must close before endSingleTimeCommands, which ends, submits, waits and then
    // *frees* this command buffer. At the outer scope, ScopedCmdLabel's destructor would run after
    // that free and call vkCmdEndDebugUtilsLabelEXT on a dead handle -- invisible in an ordinary
    // run, because the entry point is null unless VK_EXT_debug_utils is enabled and the destructor
    // checks for that, and an access violation inside the validation layer the moment it is not.
    // Same inner-block shape as HdrEnvDome::integrateBrdf in nvshaders_host/hdr_env_dome.cpp.
    {
      NVVK_DBG_SCOPE(cmd);

      // Barrier for whatever produced latlongImage (a bake's compute dispatch in a preceding
      // submit, typically) -> our compute read-modify-write. The source scope is deliberately
      // ALL_COMMANDS/MEMORY_WRITE rather than COMPUTE_SHADER/SHADER_WRITE: this is a public entry
      // point and the contract names no producing stage, so a caller that filled the image with a
      // transfer or a render pass would fall outside a compute-only source scope. The widening is
      // free -- this runs once per parameter commit, on an otherwise idle queue.
      VkMemoryBarrier memoryBarrier{
          .sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
          .srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT,
          .dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
      };
      vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1,
                           &memoryBarrier, 0, nullptr, 0, nullptr);

      const shaderio::EnvWritePdfPushConstant push{
          .imageSize   = {imageSize.width, imageSize.height},
          .gridSize    = {samplingGrid.width, samplingGrid.height},
          .invIntegral = 1.0F / m_integral,
      };
      vkCmdPushConstants(cmd, m_writePdfLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
      vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_writePdfLayout, 0, 1, m_writePdfPack.getSetPtr(), 0, nullptr);
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_writePdfPipeline);

      const VkExtent2D groupCounts = nvvk::getGroupCounts(imageSize, HDR_WORKGROUP_SIZE);
      vkCmdDispatch(cmd, groupCounts.width, groupCounts.height, 1);

    }  // the label closes here, while the command buffer is still alive

    nvvk::endSingleTimeCommands(cmd, m_device, transientCmdPool, queueInfo.queue);
  }

  vkDestroyCommandPool(m_device, transientCmdPool, nullptr);

  // -- Step 6: adopt the image and rebind sampling descriptors ------------------------------
  // m_texHdr is what getHdrImage() returns, and downstream consumers bind it into their own
  // descriptor sets (vk_gltf_renderer puts it in `texturesHdr[HDR_IMAGE_INDEX]`). It has to name
  // the image we just bound, or getHdrImage() keeps returning the zeroed handle destroyEnvironment
  // left behind and the caller writes a dangling descriptor.
  //
  // m_hdrImageIsOwned stays false: we reference latlongImage, we do not own it, so
  // destroyEnvironment() must not free it on the next teardown.
  m_texHdr = latlongImage;
  writeDescriptorSet(latlongImage, m_accelImpSmpl);
}


//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////

// The Vose alias-map builder used to live here; it moved to `env_aliasmap.{hpp,cpp}` so CPU
// tests can link it directly without dragging in the whole Vulkan/VMA stack. The declaration
// stays reachable through `hdr_ibl.hpp` (which now re-includes `env_aliasmap.hpp`).

// CIE luminance
inline static float luminance(const float* color)
{
  return color[0] * 0.2126F + color[1] * 0.7152F + color[2] * 0.0722F;
}

//--------------------------------------------------------------------------------------------------
// Create acceleration data for importance sampling
// See:  https://arxiv.org/pdf/1901.05423.pdf
// And store the PDF into the ALPHA channel of pixels
//
inline std::vector<shaderio::EnvAccel> createEnvironmentAccel(float*&         pixels,
                                                              const uint32_t& width,
                                                              const uint32_t& height,
                                                              float&          average,
                                                              float&          integral)
{
  const uint32_t rx = width;
  const uint32_t ry = height;

  // Create importance sampling data
  std::vector<shaderio::EnvAccel> env_accel(rx * ry);
  std::vector<float>              importance_data(rx * ry);
  float                           cos_theta0 = 1.0F;
  const float                     step_phi   = glm::two_pi<float>() / static_cast<float>(rx);
  const float                     step_theta = glm::pi<float>() / static_cast<float>(ry);
  double                          total      = 0.0;

  // For each texel of the environment map, we compute the related solid angle
  // subtended by the texel, and store the weighted luminance in importance_data,
  // representing the amount of energy emitted through each texel.
  // Also compute the average CIE luminance to drive the tonemapping of the final image
  for(uint32_t y = 0; y < ry; ++y)
  {
    const float theta1     = static_cast<float>(y + 1) * step_theta;
    const float cos_theta1 = std::cos(theta1);
    const float area       = (cos_theta0 - cos_theta1) * step_phi;  // solid angle
    cos_theta0             = cos_theta1;

    for(uint32_t x = 0; x < rx; ++x)
    {
      const uint32_t idx           = y * rx + x;
      const uint32_t idx4          = idx * 4;
      float          cie_luminance = luminance(&pixels[idx4]);
      importance_data[idx]         = area * std::max(pixels[idx4], std::max(pixels[idx4 + 1], pixels[idx4 + 2]));
      total += cie_luminance;
    }
  }

  average = static_cast<float>(total) / static_cast<float>(rx * ry);

  // Build the alias map, which aims at creating a set of texel couples
  // so that all couples emit roughly the same amount of energy. To this aim,
  // each smaller radiance texel will be assigned an "alias" with higher emitted radiance
  // As a byproduct this function also returns the integral of the radiance emitted by the environment
  integral = buildEnvAliasmap(importance_data, env_accel);
  if(integral == 0.0f)
  {
    integral = 1.0f;
  }

  // We deduce the PDF of each texel by normalizing its emitted radiance by the radiance integral
  const float inv_env_integral = 1.0F / integral;
  for(uint32_t i = 0; i < rx * ry; ++i)
  {
    const uint32_t idx4 = i * 4;
    pixels[idx4 + 3]    = std::max(pixels[idx4], std::max(pixels[idx4 + 1], pixels[idx4 + 2])) * inv_env_integral;
  }

  return env_accel;
}

}  // namespace nvvk
