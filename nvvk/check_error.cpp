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

#include "check_error.hpp"

#include "nvutils/logger.hpp"

#include <cassert>
#include <cstdlib>
#include <vulkan/vk_enum_string_helper.h>  // For string_VkResult

namespace nvvk {

void CheckError::check(VkResult result, const char* expression, const char* file, int line)
{
  if(result < 0)
  {
    const char* errMsg = string_VkResult(result);
    LOGE("Vulkan error: %s from %s:%d\n", errMsg, file, line);
    if(m_callback)
    {
      m_callback(result);
    }
    assert((result >= 0) && errMsg);
    exit(EXIT_FAILURE);
  }
}

VkResult CheckError::report(VkResult result, const char* expression, const char* file, int line)
{
  if(result < 0)
  {
    const char* errMsg = string_VkResult(result);
    LOGE("Vulkan error: %s from %s, %s:%d\n", errMsg, expression, file, line);
  }
  return result;
}

}  // namespace nvvk