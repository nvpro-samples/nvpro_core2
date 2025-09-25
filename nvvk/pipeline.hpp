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

#pragma once

#include <filesystem>
#include <vulkan/vulkan_core.h>


namespace nvvk {

// writes stats into single file
// pipeline must have been created with VK_PIPELINE_CREATE_CAPTURE_STATISTICS_BIT_KHR
void dumpPipelineStats(VkDevice device, VkPipeline pipeline, const std::filesystem::path& fileName);

// creates multiple files, one for each pipe executable and representation.
// The baseFilename will get appended along the lines of ".some details.bin"
// pipeline must have been created with VK_PIPELINE_CREATE_CAPTURE_INTERNAL_REPRESENTATIONS_BIT_KHR
void dumpPipelineInternals(VkDevice device, VkPipeline pipeline, const std::filesystem::path& baseFileName);

}  // namespace nvvk
