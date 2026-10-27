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

#include <string>

#include "basic_types.hpp"

namespace nvdx12 {

/*-------------------------------------------------------------------------------------------------
# class nvdx12::DebugUtil

> Debug names and command-list events, the D3D12 counterpart of
> nvvk::DebugUtil.

`setDebugName` attaches a UTF-8 name to any D3D12 object (null-safe); PIX,
Nsight Graphics, the debug layer and DRED report it. `cmdBeginLabel` /
`cmdEndLabel` bracket command-list work with a PIX event in the ANSI
encoding the tools decode without the PIX runtime. The macros mirror nvvk's:
`NVDX12_DBG_NAME(obj)` names an object after the enclosing function and
line, `NVDX12_DBG_CUSTOM_NAME(obj, name)` names it explicitly,
`NVDX12_DBG_SCOPE(cmd)` brackets the enclosing scope with an event named
after the function and `NVDX12_DBG_SCOPE_NAMED(cmd, name)`, an addition, with a
custom name.
`DBGUTIL_FILE_LINE()` is nvvk's "file:line" string helper.

Differences with nvvk that follow from D3D12:
- No `DebugUtil::init(device)`: names are set on the objects themselves
  (ID3D12Object::SetName) and events on the command lists, so the singleton
  needs no device and is usable before any device exists.
- `setObjectName` takes any ID3D12Object (or a ComPtr to one) instead of a
  Vulkan handle and its VkObjectType.
- Labels are PIX events (BeginEvent / EndEvent) instead of
  VK_EXT_debug_utils labels.

See usage_DebugUtil() in debug_util.cpp.
-------------------------------------------------------------------------------------------------*/

void setDebugName(ID3D12Object* object, const char* name);
void setDebugName(ID3D12Object* object, const std::string& name);

// PIX event around the commands recorded between the two calls (null-safe).
void cmdBeginLabel(ID3D12GraphicsCommandList* cmd, const char* label);
void cmdEndLabel(ID3D12GraphicsCommandList* cmd);

class DebugUtil
{
public:
  static DebugUtil& getInstance()
  {
    static DebugUtil instance;
    return instance;
  }

  void setObjectName(ID3D12Object* object, const std::string& name) const { setDebugName(object, name); }

  template <class T>
  void setObjectName(const ComPtr<T>& object, const std::string& name) const
  {
    setDebugName(object.Get(), name);
  }

  // Command-list event for the lifetime of the object (BeginEvent / EndEvent).
  class ScopedCmdLabel
  {
  public:
    ScopedCmdLabel(ID3D12GraphicsCommandList* cmd, const char* label)
        : m_cmd(cmd)
    {
      cmdBeginLabel(m_cmd, label);
    }
    ScopedCmdLabel(ID3D12GraphicsCommandList* cmd, const std::string& label)
        : ScopedCmdLabel(cmd, label.c_str())
    {
    }
    ~ScopedCmdLabel() { cmdEndLabel(m_cmd); }
    ScopedCmdLabel(const ScopedCmdLabel&)            = delete;
    ScopedCmdLabel& operator=(const ScopedCmdLabel&) = delete;

  private:
    ID3D12GraphicsCommandList* m_cmd;
  };

private:
  DebugUtil() = default;
};

}  // namespace nvdx12

#ifndef DBGUTIL_FILE_LINE  // also defined, identically, by nvvk/debug_util.hpp
#define DBGUTIL_FILE_LINE() (__FILE__ ":" + std::to_string(__LINE__))
#endif

#define NVDX12_DBG_CUSTOM_NAME(object, name) nvdx12::DebugUtil::getInstance().setObjectName((object), (name))
#define NVDX12_DBG_NAME(obj)                                                                                           \
  nvdx12::DebugUtil::getInstance().setObjectName(obj, std::string(__FUNCTION__) + ":" #obj "(" + std::to_string(__LINE__) + ")")

// The scope variable carries the line number so two scopes may share a block.
#define NVDX12_DBG_SCOPE_CONCAT_(a, b) a##b
#define NVDX12_DBG_SCOPE_CONCAT(a, b) NVDX12_DBG_SCOPE_CONCAT_(a, b)
#define NVDX12_DBG_SCOPE(cmd)                                                                                          \
  nvdx12::DebugUtil::ScopedCmdLabel NVDX12_DBG_SCOPE_CONCAT(scopedCmdLabel_, __LINE__)((cmd), __FUNCTION__)
#define NVDX12_DBG_SCOPE_NAMED(cmd, name)                                                                              \
  nvdx12::DebugUtil::ScopedCmdLabel NVDX12_DBG_SCOPE_CONCAT(scopedCmdLabel_, __LINE__)((cmd), (name))
