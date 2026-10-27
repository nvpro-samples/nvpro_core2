/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <functional>
#include <string>

#include "basic_types.hpp"

namespace nvdx12 {

/*-------------------------------------------------------------------------------------------------
# class nvdx12::CheckError

> Reports failed HRESULTs, the D3D12 counterpart of nvvk::CheckError.

`NVDX12_CHECK(expr)` treats a failure as fatal: it logs the expression, file
and line, calls the optional callback and exits. `NVDX12_FAIL_RETURN(expr)`
logs and returns the HRESULT from the enclosing function, and
`NVDX12_FAIL_REPORT(expr)` logs and evaluates to the HRESULT so it can be
tested inline. `hresultToString` is the counterpart of `string_VkResult`.

```cpp
HRESULT createResources(ID3D12Device* device)
{
  NVDX12_FAIL_RETURN(device->CreateCommittedResource3(...));
  if(FAILED(NVDX12_FAIL_REPORT(device->CreateFence(...))))
  {
    // custom cleanup
  }
  return S_OK;
}
```

Differences with nvvk that follow from D3D12:
- The macros and the callback take an HRESULT instead of a VkResult;
  FAILED() decides, so success codes such as S_FALSE pass.
- `hresultToString` adds the constant name of common DXGI / D3D12 / COM
  codes and the system message to the code (string_VkResult has the enum
  name alone).

See usage_CheckError() in check_error.cpp.
-------------------------------------------------------------------------------------------------*/

// Human-readable HRESULT: the hex code, the constant name of common DXGI /
// D3D12 / COM codes, and the system message when there is one.
std::string hresultToString(HRESULT result);

class CheckError
{
public:
  static CheckError& getInstance()
  {
    static CheckError instance;
    return instance;
  }

  // Called by `check` with the failed HRESULT before the process exits.
  using Callback = std::function<void(HRESULT)>;
  void setCallbackFunction(Callback&& callback) { m_callback = std::forward<Callback>(callback); }

  // If result is an error, logs an error with the given expression, file
  // and line, calls the callback, asserts false and calls exit(EXIT_FAILURE).
  void check(HRESULT result, const char* expression, const char* file, int line);

  // Recoverable `check`: logs a failed result (no callback, no exit) and returns `result`.
  HRESULT report(HRESULT result, const char* expression, const char* file, int line);

private:
  CheckError()                             = default;
  ~CheckError()                            = default;
  CheckError(const CheckError&)            = delete;
  CheckError& operator=(const CheckError&) = delete;

  Callback m_callback;
};

}  // namespace nvdx12

// Checks the result of a D3D12 call and treats an error as fatal: logs,
// calls the callback, asserts and calls exit(EXIT_FAILURE).
#define NVDX12_CHECK(winFnc)                                                                                           \
  do                                                                                                                   \
  {                                                                                                                    \
    const HRESULT checkResult = (winFnc);                                                                              \
    nvdx12::CheckError::getInstance().check(checkResult, #winFnc, __FILE__, __LINE__);                                 \
  } while(0)

// If the HRESULT is an error, logs it and returns it from the current function.
#define NVDX12_FAIL_RETURN(winFnc)                                                                                     \
  do                                                                                                                   \
  {                                                                                                                    \
    const HRESULT checkResult = (winFnc);                                                                              \
    if(FAILED(checkResult))                                                                                            \
    {                                                                                                                  \
      return nvdx12::CheckError::getInstance().report(checkResult, #winFnc, __FILE__, __LINE__);                       \
    }                                                                                                                  \
  } while(0)

// If the HRESULT is an error, logs it. Passes the input through, so it can
// be used as an expression: `if(FAILED(NVDX12_FAIL_REPORT(...)))`.
#define NVDX12_FAIL_REPORT(winFnc) nvdx12::CheckError::getInstance().report(winFnc, #winFnc, __FILE__, __LINE__)
