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

#pragma once

#include <functional>
#include <vulkan/vulkan_core.h>

namespace nvvk {
class CheckError
{
public:
  static CheckError& getInstance()
  {
    static CheckError instance;
    return instance;
  }

  using Callback = std::function<void(VkResult)>;
  void setCallbackFunction(Callback&& callback) { m_callback = std::forward<Callback>(callback); }

  // If result is an error, logs an error with the given expression, file,
  // and line, calls the callback, and asserts false and always calls exit(EXIT_FAILURE).
  void check(VkResult result, const char* expression, const char* file, int line);

  // Same as `check`, but is "recoverable"; prints an error and returns the value.
  VkResult report(VkResult result, const char* expression, const char* file, int line);

private:
  CheckError()                             = default;
  ~CheckError()                            = default;
  CheckError(const CheckError&)            = delete;
  CheckError& operator=(const CheckError&) = delete;

  Callback m_callback;
};

}  // namespace nvvk

// Use NVVK_CHECK to check the result of a Vulkan function call.
// If the input is an error, it will print an error message, call the
// callback function, and assert as well as call exit(EXIT_FAILURE)
// -- basically treating errors as fatal.
#define NVVK_CHECK(vkFnc)                                                                                              \
  {                                                                                                                    \
    const VkResult checkResult = (vkFnc);                                                                              \
    nvvk::CheckError::getInstance().check(checkResult, #vkFnc, __FILE__, __LINE__);                                    \
  }

// If the input VkResult is an error, this prints an error message and returns
// from the current function with the result.
//
// For example:
// ```
// VkResult foo()
// {
//   NVVK_FAIL_RETURN(vkDeviceWaitIdle(m_device));
//   return VK_SUCCESS;
// }
// ```
// is a shorter way to write
// ```
// VkResult foo()
// {
//   const VkResult result = vkDeviceWaitIdle(m_device);
//   if(result < 0)
//   {
//     LOGE("Vulkan error: <result, expression, file, and line info...>");
//     return result;
//   }
//   return VK_SUCCESS;
// }
// ```
#define NVVK_FAIL_RETURN(vkFnc)                                                                                        \
  {                                                                                                                    \
    const VkResult checkResult = (vkFnc);                                                                              \
    if(checkResult < 0)                                                                                                \
    {                                                                                                                  \
      return nvvk::CheckError::getInstance().report(checkResult, #vkFnc, __FILE__, __LINE__);                          \
    }                                                                                                                  \
  }

// If the input VkResult is an error, prints an error message. Passes the input
// through.
//
// This can be used as an expression; for example:
// ```
// const VkResult result = NVVK_FAIL_REPORT(vkDeviceWaitIdle(m_device));
// ```
// or
// ```
// if(VK_SUCCESS != NVVK_FAIL_REPORT(vkDeviceWaitIdle(m_device)))
// {
//   // custom cleanup...
// }
// ```
#define NVVK_FAIL_REPORT(vkFnc) nvvk::CheckError::getInstance().report(vkFnc, #vkFnc, __FILE__, __LINE__)
