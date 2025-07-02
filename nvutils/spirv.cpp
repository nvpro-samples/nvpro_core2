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

#include "spirv.hpp"
#include "file_operations.hpp"
#include "hash_operations.hpp"
#include "logger.hpp"

std::size_t nvutils::hashSpirv(const uint32_t* spirvData, size_t spirvSize)
{
  std::size_t seed = 0;
  for(size_t i = 0; i < spirvSize / sizeof(uint32_t); ++i)
  {
    nvutils::hashCombine(seed, spirvData[i]);
  }
  return seed;
}

std::filesystem::path nvutils::dumpSpirvName(const std::filesystem::path& filename, const uint32_t* spirvData, size_t spirvSize)
{
  return nvutils::getExecutablePath().parent_path()
         / (filename.filename().replace_extension(std::to_string(hashSpirv(spirvData, spirvSize)) + ".spv"));
}

void nvutils::dumpSpirv(const std::filesystem::path& filename, const uint32_t* spirvData, size_t spirvSize)
{
  std::ofstream file(filename, std::ios::binary);
  if(!file)
  {
    LOGE("Failed to open file for writing: %s\n", nvutils::utf8FromPath(filename).c_str());
    return;
  }

  file.write(reinterpret_cast<const char*>(spirvData), spirvSize);
  if(!file)
  {
    LOGE("Failed to write SPIR-V data to file: %s\n", nvutils::utf8FromPath(filename).c_str());
  }
}

void nvutils::dumpSpirvWithHashedName(const std::filesystem::path& sourceFile, const uint32_t* spirvData, size_t spirvSize)
{
  dumpSpirv(dumpSpirvName(sourceFile, spirvData, spirvSize), spirvData, spirvSize);
}
