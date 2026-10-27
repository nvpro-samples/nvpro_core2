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

#include "check_error.hpp"

#include <cassert>
#include <cstdio>
#include <cstdlib>

#include <dxgi1_6.h>

#include <nvutils/logger.hpp>

namespace nvdx12 {

// Constant names of common DXGI / D3D12 / COM codes (FormatMessage renders
// DXGI_ERROR_DEVICE_REMOVED as "Access is denied").
static const char* hresultName(HRESULT result)
{
  switch(result)
  {
    case DXGI_ERROR_DEVICE_REMOVED:
      return "DXGI_ERROR_DEVICE_REMOVED";
    case DXGI_ERROR_DEVICE_HUNG:
      return "DXGI_ERROR_DEVICE_HUNG";
    case DXGI_ERROR_DEVICE_RESET:
      return "DXGI_ERROR_DEVICE_RESET";
    case DXGI_ERROR_DRIVER_INTERNAL_ERROR:
      return "DXGI_ERROR_DRIVER_INTERNAL_ERROR";
    case DXGI_ERROR_INVALID_CALL:
      return "DXGI_ERROR_INVALID_CALL";
    case DXGI_ERROR_NOT_FOUND:
      return "DXGI_ERROR_NOT_FOUND";
    case DXGI_ERROR_UNSUPPORTED:
      return "DXGI_ERROR_UNSUPPORTED";
    case DXGI_ERROR_SDK_COMPONENT_MISSING:
      return "DXGI_ERROR_SDK_COMPONENT_MISSING";
    case DXGI_ERROR_ACCESS_LOST:
      return "DXGI_ERROR_ACCESS_LOST";
    case DXGI_ERROR_WAS_STILL_DRAWING:
      return "DXGI_ERROR_WAS_STILL_DRAWING";
    case D3D12_ERROR_ADAPTER_NOT_FOUND:
      return "D3D12_ERROR_ADAPTER_NOT_FOUND";
    case D3D12_ERROR_DRIVER_VERSION_MISMATCH:
      return "D3D12_ERROR_DRIVER_VERSION_MISMATCH";
    case E_INVALIDARG:
      return "E_INVALIDARG";
    case E_OUTOFMEMORY:
      return "E_OUTOFMEMORY";
    case E_NOTIMPL:
      return "E_NOTIMPL";
    case E_NOINTERFACE:
      return "E_NOINTERFACE";
    case E_POINTER:
      return "E_POINTER";
    case E_ACCESSDENIED:
      return "E_ACCESSDENIED";
    case E_FAIL:
      return "E_FAIL";
    default:
      return nullptr;
  }
}

std::string hresultToString(HRESULT result)
{
  char code[32];
  snprintf(code, sizeof(code), "0x%08X", static_cast<unsigned>(result));
  std::string text = code;
  if(const char* name = hresultName(result))
  {
    text += " ";
    text += name;
  }

  char* systemMessage = nullptr;
  DWORD length = FormatMessageA(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
                                nullptr, static_cast<DWORD>(result), 0, reinterpret_cast<LPSTR>(&systemMessage), 0, nullptr);
  if(length && systemMessage)
  {
    // Drop the trailing line break of the message table.
    while(length > 0 && (systemMessage[length - 1] == '\n' || systemMessage[length - 1] == '\r' || systemMessage[length - 1] == ' '))
    {
      systemMessage[--length] = '\0';
    }
    if(length)
    {
      text += " (";
      text += systemMessage;
      text += ")";
    }
  }
  if(systemMessage)
  {
    LocalFree(systemMessage);
  }
  return text;
}

void CheckError::check(HRESULT result, const char* expression, const char* file, int line)
{
  if(FAILED(result))
  {
    LOGE("%s(%d): %s failed: %s\n", file, line, expression, hresultToString(result).c_str());
    if(m_callback)
    {
      m_callback(result);
    }
    assert(SUCCEEDED(result));
    exit(EXIT_FAILURE);
  }
}

HRESULT CheckError::report(HRESULT result, const char* expression, const char* file, int line)
{
  if(FAILED(result))
  {
    LOGE("%s(%d): %s failed: %s\n", file, line, expression, hresultToString(result).c_str());
  }
  return result;
}

}  // namespace nvdx12

//--------------------------------------------------------------------------------------------------
// Usage example
//--------------------------------------------------------------------------------------------------
[[maybe_unused]] static HRESULT usage_CheckError()
{
  ID3D12Device* device = nullptr;  // EX: from nvdx12::Context::getDevice()

  // Optional: run application code (dump the frame, flush the log) before a
  // fatal check exits the process.
  nvdx12::CheckError::getInstance().setCallbackFunction(
      [](HRESULT hr) { LOGE("fatal: %s\n", nvdx12::hresultToString(hr).c_str()); });

  // Fatal: a failure logs, calls the callback and exits.
  nvdx12::ComPtr<ID3D12Fence> fence;
  NVDX12_CHECK(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)));

  // Recoverable: a failure logs and is returned from this function.
  nvdx12::ComPtr<ID3D12CommandAllocator> allocator;
  NVDX12_FAIL_RETURN(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator)));

  // Recoverable, inline: the HRESULT is logged and passed through.
  if(FAILED(NVDX12_FAIL_REPORT(allocator->Reset())))
  {
    // custom cleanup
  }
  return S_OK;
}
