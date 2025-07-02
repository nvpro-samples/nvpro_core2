/*
* Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
* SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION.
* SPDX-License-Identifier: Apache-2.0
*/

#include <cinttypes>
#include <string>
#include <string.h>
#include <vector>

#include <volk.h>

#include "pipeline.hpp"

namespace nvvk {

void dumpPipelineStats(VkDevice device, VkPipeline pipeline, const std::filesystem::path& fileName)
{
  VkPipelineInfoKHR pipeInfo = {VK_STRUCTURE_TYPE_PIPELINE_INFO_KHR};
  pipeInfo.pipeline          = pipeline;
  if(!pipeline)
    return;

#ifdef _WIN32
  FILE* fdump = nullptr;
  if(_wfopen_s(&fdump, fileName.c_str(), L"wt") != 0)
    return;
#else
  FILE* fdump = fopen(fileName.c_str(), "wt");
#endif
  if(!fdump)
    return;


  std::vector<VkPipelineExecutablePropertiesKHR> props;
  uint32_t                                       executableCount = 0;
  vkGetPipelineExecutablePropertiesKHR(device, &pipeInfo, &executableCount, nullptr);
  props.resize(executableCount, {VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_PROPERTIES_KHR});
  vkGetPipelineExecutablePropertiesKHR(device, &pipeInfo, &executableCount, props.data());

  fprintf(fdump, "VkPipeline stats for %p\n", pipeline);
  fprintf(fdump, "-----------------------\n");
  for(uint32_t i = 0; i < executableCount; i++)
  {
    const VkPipelineExecutablePropertiesKHR& prop = props[i];
    fprintf(fdump, "- Executable: %s\n", prop.name);
    fprintf(fdump, "  (%s)\n", prop.description);
    fprintf(fdump, "  - stages: 0x%08X\n", prop.stages);
    fprintf(fdump, "  - subgroupSize: %2d\n", prop.subgroupSize);
    VkPipelineExecutableInfoKHR execInfo = {VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_INFO_KHR};
    execInfo.pipeline                    = pipeline;
    execInfo.executableIndex             = i;

    uint32_t                                      statsCount = 0;
    std::vector<VkPipelineExecutableStatisticKHR> stats;
    vkGetPipelineExecutableStatisticsKHR(device, &execInfo, &statsCount, nullptr);
    stats.resize(statsCount, {VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_STATISTIC_KHR});
    vkGetPipelineExecutableStatisticsKHR(device, &execInfo, &statsCount, stats.data());

    for(uint32_t s = 0; s < statsCount; s++)
    {
      const VkPipelineExecutableStatisticKHR& stat = stats[s];
      switch(stat.format)
      {
        case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_BOOL32_KHR:
          fprintf(fdump, "  - %s: %d\n", stat.name, stat.value.b32);
          break;
        case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_INT64_KHR:
          fprintf(fdump, "  - %s: %" PRIi64 "\n", stat.name, stat.value.i64);
          break;
        case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_UINT64_KHR:
          fprintf(fdump, "  - %s: %" PRIu64 "\n", stat.name, stat.value.u64);
          break;
        case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_FLOAT64_KHR:
          fprintf(fdump, "  - %s: %f\n", stat.name, stat.value.f64);
          break;
      }
      fprintf(fdump, "    (%s)\n", stat.description);
    }
  }
  fprintf(fdump, "\n");
  fclose(fdump);
}

// creates multiple files, one for each pipe executable and representation.
// The baseFilename will get appended along the lines of ".some details.bin"
// pipeline must have been created with VK_PIPELINE_CREATE_CAPTURE_INTERNAL_REPRESENTATIONS_BIT_KHR
void dumpPipelineInternals(VkDevice device, VkPipeline pipeline, const std::filesystem::path& baseFileName)
{
  VkPipelineInfoKHR pipeInfo = {VK_STRUCTURE_TYPE_PIPELINE_INFO_KHR};
  pipeInfo.pipeline          = pipeline;
  if(!pipeline)
    return;

  std::vector<VkPipelineExecutablePropertiesKHR> props;
  uint32_t                                       executableCount = 0;
  vkGetPipelineExecutablePropertiesKHR(device, &pipeInfo, &executableCount, nullptr);
  props.resize(executableCount, {VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_PROPERTIES_KHR});
  vkGetPipelineExecutablePropertiesKHR(device, &pipeInfo, &executableCount, props.data());

  for(uint32_t e = 0; e < executableCount; e++)
  {
    const VkPipelineExecutablePropertiesKHR& prop     = props[e];
    VkPipelineExecutableInfoKHR              execInfo = {VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_INFO_KHR};
    execInfo.pipeline                                 = pipeline;
    execInfo.executableIndex                          = e;

    uint32_t internalCount = 0;
    vkGetPipelineExecutableInternalRepresentationsKHR(device, &execInfo, &internalCount, nullptr);
    if(internalCount)
    {
      std::vector<VkPipelineExecutableInternalRepresentationKHR> internals(
          internalCount, {VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_INTERNAL_REPRESENTATION_KHR});
      vkGetPipelineExecutableInternalRepresentationsKHR(device, &execInfo, &internalCount, internals.data());

      size_t offset = 0;
      for(uint32_t i = 0; i < internalCount; i++)
      {
        offset += internals[i].dataSize;
      }

      std::vector<uint8_t> rawBytes(offset);

      offset = 0;
      for(uint32_t i = 0; i < internalCount; i++)
      {
        internals[i].pData = rawBytes.data() + offset;
        offset += internals[i].dataSize;
      }

      vkGetPipelineExecutableInternalRepresentationsKHR(device, &execInfo, &internalCount, internals.data());
      for(uint32_t i = 0; i < internalCount; i++)
      {
        bool isText = strstr(internals[i].name, "text") != nullptr;

        std::filesystem::path fileName = baseFileName;
        fileName += std::string(".") + prop.name + "." + std::to_string(e) + "." + internals[i].name + "."
                    + std::to_string(i) + "." + (isText ? "txt" : "bin");

#ifdef _WIN32
        FILE* fdump = nullptr;
        if(_wfopen_s(&fdump, fileName.c_str(), L"wt") != 0)
          fdump = nullptr;
#else
        FILE* fdump = fopen(fileName.c_str(), "wt");
#endif
        if(fdump)
        {
          fwrite(internals[i].pData, internals[i].dataSize, 1, fdump);
          fclose(fdump);
        }
      }
    }
  }
}

}  // namespace nvvk
