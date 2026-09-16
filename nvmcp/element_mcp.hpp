/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <chrono>
#include <cstdint>
#include <functional>
#include <initializer_list>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <nvapp/application.hpp>

namespace nvutils {
class ParameterRegistry;
class ProfilerManager;
}  // namespace nvutils

namespace nvmcp {

struct ImageContent
{
  std::string data;      // Base64-encoded bytes as required by MCP image content
  std::string mimeType;  // MIME type describing the decoded bytes
};

// JSON and encoded images cross the public boundary as standard-library types
// so sample tools can return native MCP content without inheriting cpp-mcp's
// bundled nlohmann::json version.
struct ToolResult
{
  std::string               json;
  std::vector<ImageContent> images;
  bool                      isError{false};

  static ToolResult success(std::string jsonText) { return {.json = std::move(jsonText)}; }
  static ToolResult success(std::string jsonText, ImageContent image)
  {
    return {.json = std::move(jsonText), .images = {std::move(image)}};
  }
  static ToolResult error(std::string jsonText) { return {.json = std::move(jsonText), .isError = true}; }
};

struct ToolAnnotations
{
  bool readOnlyHint{false};
  // Match MCP's conservative defaults so omitted sample annotations never
  // make an unknown action look safer or more isolated than it is.
  bool destructiveHint{true};
  bool idempotentHint{false};
  bool openWorldHint{true};
};

struct Tool
{
  std::string                                 name;
  std::string                                 description;
  std::string                                 inputSchema{R"({"type":"object","additionalProperties":false})"};
  std::function<ToolResult(std::string_view)> handler;
  ToolAnnotations                             annotations;
  // Worker threads are the default so tool work does not interrupt rendering.
  // Opt in only for short handlers that access application-owned state which
  // is not safe to use concurrently with the application loop.
  bool runOnApplicationThread{false};
};

struct ElementCreateInfo
{
  std::string serverName{"nvpro_core2"};
  std::string serverVersion{"0.1.0"};
  std::string instructions{"Tools inspect and control a live development application. Follow each tool's description and annotations."};
  // The first implementation is intentionally unauthenticated and therefore
  // accepts loopback hosts only.
  std::string               host{"127.0.0.1"};
  uint16_t                  port{7671};
  std::string               endpoint{"/mcp"};
  std::chrono::milliseconds toolCallTimeout{std::chrono::seconds(300)};
  uint32_t                  sessionTimeoutSeconds{600};
};

// An nvapp element that owns a local Streamable HTTP MCP endpoint.
//
// Register sample tools before adding the element to Application. Handlers run
// on MCP worker threads by default. A tool can explicitly request the
// application thread when it needs short, synchronized access to sample or
// render state.
class Element : public nvapp::IAppElement
{
public:
  explicit Element(ElementCreateInfo info = {});
  ~Element() override;

  Element(const Element&)            = delete;
  Element& operator=(const Element&) = delete;

  void registerTool(Tool tool);
  // Reuses existing ParameterRegistry metadata without exposing every command
  // line option or sample setting by default. Call once before attachment; the
  // registry and all parameter destinations must outlive this element.
  void registerParameters(const nvutils::ParameterRegistry& registry, std::initializer_list<std::string_view> names);
  // Exposes the profiler's existing thread-safe snapshots without introducing
  // a second timing or aggregation path. The manager must outlive this element.
  void registerProfiler(const nvutils::ProfilerManager& profiler);

  bool running() const;

  void onAttach(nvapp::Application* app) override;
  void onUpdate() override;
  void onPreRender() override;
  void onPostRender() override;
  void onDetach() override;

private:
  class Impl;
  std::unique_ptr<Impl> m_impl;
};

}  // namespace nvmcp
