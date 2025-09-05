/*
 * Copyright (c) 2022-2025, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */


/*
 *  HDR sampling is loading an HDR image and creating an acceleration structure for 
 *  sampling the environment. 
 */

#include <array>
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
#include "debug_util.hpp"
#include "default_structs.hpp"
#include "descriptors.hpp"
#include "hdr_ibl.hpp"
#include "mipmaps.hpp"
#include "staging.hpp"


namespace nvvk {
// Forward declaration
std::vector<shaderio::EnvAccel> createEnvironmentAccel(float*& pixels, const uint32_t& width, const uint32_t& height, float& average, float& integral);


//--------------------------------------------------------------------------------------------------
//
//
void HdrIbl::init(nvvk::ResourceAllocator* allocator, nvvk::SamplerPool* samplerPool)
{
  m_device      = allocator->getDevice();
  m_alloc       = allocator;
  m_samplerPool = samplerPool;
}

//--------------------------------------------------------------------------------------------------
//
//
void HdrIbl::deinit()
{
  destroyEnvironment();
  m_device      = {};
  m_alloc       = nullptr;
  m_samplerPool = nullptr;
}

//--------------------------------------------------------------------------------------------------
// Loading the HDR environment texture (HDR) and create the important accel structure
//
// Note: enableMipmaps will create a mipmap chain for the environment texture, but does not generate the
//       mipmaps
void HdrIbl::loadEnvironment(VkCommandBuffer cmd, nvvk::StagingUploader& staging, const std::filesystem::path& hdrImage, bool enableMipmaps)
{
  nvutils::ScopedTimer st(__FUNCTION__);

  m_valid = !hdrImage.empty();

  int32_t     width{0};
  int32_t     height{0};
  int32_t     component{0};
  std::string fileContents;
  float*      pixels = nullptr;
  if(m_valid)
  {
    // Read the contents into memory so that we don't have to worry about text
    // encoding in the stbi filename API
    fileContents = nvutils::loadFile(hdrImage);
    if(fileContents.empty())
    {
      LOGW("File does not exist or is empty: %s\n", nvutils::utf8FromPath(hdrImage).c_str());
      m_valid = false;
    }
    else if(fileContents.size() > std::numeric_limits<int>::max())
    {
      LOGW("File is too large for stb_image to load: %s\n", nvutils::utf8FromPath(hdrImage).c_str());
      m_valid = false;
    }
  }

  if(m_valid)
  {
    const stbi_uc* fileData = reinterpret_cast<const stbi_uc*>(fileContents.data());
    const int      fileSize = static_cast<int>(fileContents.size());

    if(!stbi_is_hdr_from_memory(fileData, fileSize))
    {
      LOGW("File is not HDR: %s\n", nvutils::utf8FromPath(hdrImage).c_str());
      m_valid = false;
    }
    else
    {
      nvutils::ScopedTimer st("Load image");
      pixels = stbi_loadf_from_memory(fileData, fileSize, &width, &height, &component, STBI_rgb_alpha);
      if(!pixels)
      {
        LOGW("stbi_loadf_from_memory failed: %s\n", nvutils::utf8FromPath(hdrImage).c_str());
        m_valid = false;
      }
    }
  }

  if(m_valid)
  {
    assert(pixels);
    VkDeviceSize buffer_size = width * height * 4 * sizeof(float);
    VkExtent2D   imgSize{static_cast<uint32_t>(width), static_cast<uint32_t>(height)};

    m_hdrImageSize = imgSize;

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

    stbi_image_free(pixels);
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

    NVVK_CHECK(m_alloc->createBuffer(m_accelImpSmpl, std::span(color).size_bytes(), VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT));
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

  // Create the descriptor set layout
  createDescriptorSetLayout();
}

// Destroy the resources for the environment
void HdrIbl::destroyEnvironment()
{
  m_descPack.deinit();

  if(m_alloc != nullptr)
  {
    m_samplerPool->releaseSampler(m_texHdr.descriptor.sampler);
    m_alloc->destroyImage(m_texHdr);
    m_alloc->destroyBuffer(m_accelImpSmpl);
  }
}

//--------------------------------------------------------------------------------------------------
// Descriptors of the HDR and the acceleration structure
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

  nvvk::WriteSetContainer writeContainer;
  writeContainer.append(bindings.getWriteSet(shaderio::EnvBindings::eHdr, m_descPack.getSet(0)), m_texHdr);
  writeContainer.append(bindings.getWriteSet(shaderio::EnvBindings::eImpSamples, m_descPack.getSet(0)), m_accelImpSmpl);
  vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(writeContainer.size()), writeContainer.data(), 0, nullptr);
}


//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////

//--------------------------------------------------------------------------------------------------
// Build alias map for the importance sampling: Each texel is associated to another texel, or alias,
// so that their combined intensities are a close as possible to the average of the environment map.
// This will later allow the sampling shader to uniformly select a texel in the environment, and
// select either that texel or its alias depending on their relative intensities
//
inline static float buildAliasmap(const std::vector<float>& data, std::vector<shaderio::EnvAccel>& accel)
{
  auto size = static_cast<uint32_t>(data.size());

  // Compute the integral of the emitted radiance of the environment map
  // Since each element in data is already weighted by its solid angle
  // the integral is a simple sum
  float sum = std::accumulate(data.begin(), data.end(), 0.F);
  if(sum == 0.0f)
  {
    sum = 1.0f;
  }

  // For each texel, compute the ratio q between the emitted radiance of the texel and the average
  // emitted radiance over the entire sphere
  // We also initialize the aliases to identity, ie. each texel is its own alias
  auto  f_size          = static_cast<float>(size);
  float inverse_average = f_size / sum;
  for(uint32_t i = 0; i < size; ++i)
  {
    accel[i].q     = data[i] * inverse_average;
    accel[i].alias = i;
  }

  // Partition the texels according to their emitted radiance ratio wrt. average.
  // Texels with a value q < 1 (ie. below average) are stored incrementally from the beginning of the
  // array, while texels emitting higher-than-average radiance are stored from the end of the array
  std::vector<uint32_t> partition_table(size);
  uint32_t              s     = 0U;
  uint32_t              large = size;
  for(uint32_t i = 0; i < size; ++i)
  {
    if(accel[i].q < 1.F)
      partition_table[s++] = i;
    else
      partition_table[--large] = i;
  }

  // Associate the lower-energy texels to higher-energy ones. Since the emission of a high-energy texel may
  // be vastly superior to the average,
  for(s = 0; s < large && large < size; ++s)
  {
    // Index of the smaller energy texel
    const uint32_t small_energy_index = partition_table[s];

    // Index of the higher energy texel
    const uint32_t high_energy_index = partition_table[large];

    // Associate the texel to its higher-energy alias
    accel[small_energy_index].alias = high_energy_index;

    // Compute the difference between the lower-energy texel and the average
    const float difference_with_average = 1.F - accel[small_energy_index].q;

    // The goal is to obtain texel couples whose combined intensity is close to the average.
    // However, some texels may have low energies, while others may have very high intensity
    // (for example a sunset: the sky is quite dark, but the sun is still visible). In this case
    // it may not be possible to obtain a value close to average by combining only two texels.
    // Instead, we potentially associate a single high-energy texel to many smaller-energy ones until
    // the combined average is similar to the average of the environment map.
    // We keep track of the combined average by subtracting the difference between the lower-energy texel and the average
    // from the ratio stored in the high-energy texel.
    accel[high_energy_index].q -= difference_with_average;

    // If the combined ratio to average of the higher-energy texel reaches 1, a balance has been found
    // between a set of low-energy texels and the higher-energy one. In this case, we will use the next
    // higher-energy texel in the partition when processing the next texel.
    if(accel[high_energy_index].q < 1.0F)
      large++;
  }
  // Return the integral of the emitted radiance. This integral will be used to normalize the probability
  // distribution function (PDF) of each pixel
  return sum;
}

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
  integral = buildAliasmap(importance_data, env_accel);
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
