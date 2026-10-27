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

#include "debug_util.hpp"

#include <cstring>

#include "text_encoding.hpp"

namespace nvdx12 {

void setDebugName(ID3D12Object* object, const char* name)
{
  if(!object || !name || !*name)
  {
    return;
  }
  object->SetName(wtf16FromUtf8(name).c_str());
}

void setDebugName(ID3D12Object* object, const std::string& name)
{
  setDebugName(object, name.c_str());
}

// Metadata of BeginEvent / SetMarker for a null-terminated ANSI string
// (PIX_EVENT_ANSI_VERSION in pix3.h); PIX and Nsight Graphics decode it
// without the PIX runtime.
static constexpr UINT kPixEventAnsiVersion = 1;

void cmdBeginLabel(ID3D12GraphicsCommandList* cmd, const char* label)
{
  if(!cmd)
  {
    return;
  }
  label = label ? label : "";
  cmd->BeginEvent(kPixEventAnsiVersion, label, static_cast<UINT>(strlen(label) + 1));
}

void cmdEndLabel(ID3D12GraphicsCommandList* cmd)
{
  if(cmd)
  {
    cmd->EndEvent();
  }
}

}  // namespace nvdx12

//--------------------------------------------------------------------------------------------------
// Usage example
//--------------------------------------------------------------------------------------------------
[[maybe_unused]] static void usage_DebugUtil()
{
  ID3D12Device*              device = nullptr;  // EX: from nvdx12::Context::getDevice()
  ID3D12GraphicsCommandList* cmd    = nullptr;  // EX: the command list being recorded

  nvdx12::ComPtr<ID3D12Fence> fence;
  device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence));
  NVDX12_DBG_NAME(fence);                            // "usage_DebugUtil:fence(NN)"
  NVDX12_DBG_CUSTOM_NAME(fence, "frame fence");      // explicit name through the singleton
  nvdx12::setDebugName(fence.Get(), "frame fence");  // the same without it

  {
    // Events named after the function and with a custom name, both ending with the block.
    NVDX12_DBG_SCOPE(cmd);
    NVDX12_DBG_SCOPE_NAMED(cmd, "clear targets");
    // record commands
  }

  // The same event without a scope object.
  nvdx12::cmdBeginLabel(cmd, "post processing");
  nvdx12::cmdEndLabel(cmd);
}
