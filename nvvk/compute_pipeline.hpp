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
* SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
* SPDX-License-Identifier: Apache-2.0
*/

#pragma once

#include <vulkan/vulkan_core.h>

namespace nvvk {

// Return the number of workgroups needed for a given size
// Used for compute shaders in combination with vkCmdDispatch
inline VkExtent2D getGroupCounts(const VkExtent2D& size, uint32_t workgroupSize)
{
  return VkExtent2D{(size.width + (workgroupSize - 1)) / workgroupSize, (size.height + (workgroupSize - 1)) / workgroupSize};
}

inline VkExtent2D getGroupCounts(const VkExtent2D& size, const VkExtent2D& workgroupSize)
{
  return VkExtent2D{(size.width + (workgroupSize.width - 1)) / workgroupSize.width,
                    (size.height + (workgroupSize.height - 1)) / workgroupSize.height};
}

inline VkExtent3D getGroupCounts(const VkExtent3D& size, const VkExtent3D& workgroupSize)
{
  return VkExtent3D{(size.width + (workgroupSize.width - 1)) / workgroupSize.width,
                    (size.height + (workgroupSize.height - 1)) / workgroupSize.height,
                    (size.depth + (workgroupSize.depth - 1)) / workgroupSize.depth};
}

inline uint32_t getGroupCounts(uint32_t size, uint32_t workgroupSize)
{
  return (size + (workgroupSize - 1)) / workgroupSize;
}

}  // namespace nvvk
