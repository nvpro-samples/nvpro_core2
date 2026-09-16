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

#include "element_mcp.hpp"
#include "parameter_tools.hpp"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <deque>
#include <filesystem>
#include <future>
#include <mutex>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

#include <base64.hpp>
#include <mcp_server.h>
#include <mcp_tool.h>
#include <nvutils/file_operations.hpp>
#include <nvutils/logger.hpp>
#include <nvutils/parameter_registry.hpp>
#include <nvutils/profiler.hpp>

namespace nvmcp {
namespace {

constexpr std::string_view noArgumentsSchema  = R"({"type":"object","additionalProperties":false})";
constexpr size_t           maxLogEntries      = 2000;
constexpr size_t           maxLogBytes        = 1024 * 1024;
constexpr size_t           maxLogQueryEntries = 1000;

const char* logLevelName(nvutils::Logger::LogLevel level)
{
  switch(level)
  {
    case nvutils::Logger::eDEBUG:
      return "debug";
    case nvutils::Logger::eSTATS:
      return "stats";
    case nvutils::Logger::eOK:
      return "ok";
    case nvutils::Logger::eINFO:
      return "info";
    case nvutils::Logger::eWARNING:
      return "warning";
    case nvutils::Logger::eERROR:
      return "error";
    default:
      return "unknown";
  }
}

nvutils::Logger::LogLevel parseLogLevel(std::string_view name)
{
  for(int level = nvutils::Logger::eDEBUG; level <= nvutils::Logger::eERROR; level++)
  {
    const nvutils::Logger::LogLevel logLevel = static_cast<nvutils::Logger::LogLevel>(level);
    if(name == logLevelName(logLevel))
      return logLevel;
  }
  throw std::invalid_argument("minimumLevel must be debug, stats, ok, info, warning, or error");
}

mcp::json textContent(const std::string& text)
{
  return mcp::json::array({mcp::json{{"type", "text"}, {"text", text}}});
}

mcp::json annotationsJson(const ToolAnnotations& annotations)
{
  return {
      {"readOnlyHint", annotations.readOnlyHint},
      {"destructiveHint", annotations.destructiveHint},
      {"idempotentHint", annotations.idempotentHint},
      {"openWorldHint", annotations.openWorldHint},
  };
}

void requireEmptyArguments(std::string_view arguments, std::string_view toolName)
{
  const mcp::json parsed = arguments.empty() ? mcp::json::object() : mcp::json::parse(arguments);
  if(!parsed.is_object() || !parsed.empty())
    throw std::invalid_argument(std::string(toolName) + " arguments must be an empty object");
}

mcp::json timerStatsJson(const nvutils::ProfilerTimeline::TimerStats& stats)
{
  return {
      {"last", stats.last},
      {"average", stats.average},
      {"minimum", stats.absMinValue},
      {"maximum", stats.absMaxValue},
  };
}

mcp::json timelineJson(const nvutils::ProfilerTimeline::Snapshot& snapshot)
{
  mcp::json timers = mcp::json::array();
  for(size_t i = 0; i < snapshot.timerInfos.size(); i++)
  {
    const nvutils::ProfilerTimeline::TimerInfo& info = snapshot.timerInfos[i];
    mcp::json                                   timer;
    timer["name"]        = snapshot.timerNames[i];
    timer["samples"]     = info.numAveraged;
    timer["accumulated"] = info.accumulated;
    timer["async"]       = info.async;
    timer["cpu"]         = timerStatsJson(info.cpu);
    timer["gpu"]         = timerStatsJson(info.gpu);
    if(!info.async)
      timer["level"] = info.level;
    if(!snapshot.timerApiNames[i].empty())
      timer["gpuApi"] = snapshot.timerApiNames[i];
    timers.push_back(std::move(timer));
  }

  return {
      {"name", snapshot.name},
      {"timers", std::move(timers)},
  };
}

}  // namespace

class Element::Impl
{
public:
  explicit Impl(ElementCreateInfo createInfo)
      : info(std::move(createInfo))
  {
    if(info.serverName.empty())
      throw std::invalid_argument("MCP server name must not be empty");
    if(info.host != "127.0.0.1" && info.host != "localhost" && info.host != "::1")
      throw std::invalid_argument("the unauthenticated MCP server must bind to a loopback host");
    if(info.port == 0)
      throw std::invalid_argument("MCP server port must not be zero");
    if(info.endpoint.empty() || info.endpoint.front() != '/')
      throw std::invalid_argument("MCP endpoint must start with '/'");
    if(info.toolCallTimeout <= std::chrono::milliseconds::zero())
      throw std::invalid_argument("MCP tool timeout must be positive");

    logObserver = nvutils::Logger::getInstance().addLogObserver(
        [this](nvutils::Logger::LogLevel level, const std::string& message) { appendLog(level, message); });
  }

  ~Impl() { nvutils::Logger::getInstance().removeLogObserver(logObserver); }

  struct LogEntry
  {
    uint64_t                  sequence{};
    nvutils::Logger::LogLevel level{};
    std::string               message;
  };

  void appendLog(nvutils::Logger::LogLevel level, const std::string& message) noexcept
  {
    try
    {
      LogEntry entry{.level = level, .message = message};
      if(entry.message.size() > maxLogBytes)
      {
        constexpr std::string_view suffix = "\n[log entry truncated]";
        entry.message.resize(maxLogBytes - suffix.size());
        entry.message.append(suffix);
      }

      std::lock_guard lock(logMutex);
      entry.sequence = nextLogSequence++;
      logBytes += entry.message.size();
      logEntries.push_back(std::move(entry));
      while(logEntries.size() > maxLogEntries || logBytes > maxLogBytes)
      {
        logBytes -= logEntries.front().message.size();
        logEntries.pop_front();
        droppedLogEntries++;
      }
    }
    catch(...)
    {
      std::lock_guard lock(logMutex);
      droppedLogEntries++;
    }
  }

  struct PendingCall
  {
    std::string              name;
    std::string              arguments;
    std::promise<ToolResult> promise;
    std::atomic<bool>        canceled{false};
  };

  void registerTool(Tool tool)
  {
    if(attached)
      throw std::logic_error("MCP tools must be registered before the element is attached");
    if(tool.name.empty() || tool.description.empty() || !tool.handler)
      throw std::invalid_argument("MCP tools require a name, description, and handler");
    if(tools.contains(tool.name))
      throw std::invalid_argument("duplicate MCP tool: " + tool.name);

    // destructiveHint is only meaningful for tools that modify state. Emit an
    // unambiguous false value for read-only tools because clients may still
    // use every advertised hint when deciding whether approval is required.
    if(tool.annotations.readOnlyHint)
      tool.annotations.destructiveHint = false;

    mcp::json schema;
    try
    {
      schema = mcp::json::parse(tool.inputSchema);
    }
    catch(const mcp::json::exception& e)
    {
      throw std::invalid_argument("invalid input schema for MCP tool " + tool.name + ": " + e.what());
    }
    if(!schema.is_object())
      throw std::invalid_argument("input schema for MCP tool " + tool.name + " must be a JSON object");

    toolOrder.push_back(tool.name);
    tools.emplace(tool.name, std::move(tool));
  }

  void registerParameterTools(detail::ParameterList selected)
  {
    if(attached)
      throw std::logic_error("MCP parameters must be registered before the element is attached");
    if(parameterToolsRegistered)
      throw std::logic_error("an MCP parameter registry is already registered");
    if(tools.contains("nvpro_get_parameters") || tools.contains("nvpro_set_parameters"))
      throw std::logic_error("MCP parameter tool names are already registered");

    registerTool({
        .name        = "nvpro_get_parameters",
        .description = "Return all exposed sample parameters, or only the requested names.",
        .inputSchema = R"({"type":"object","properties":{"names":{"type":"array","items":{"type":"string"},"minItems":1,"uniqueItems":true}},"additionalProperties":false})",
        .handler                = [this](std::string_view arguments) { return getParameters(arguments); },
        .annotations            = {.readOnlyHint = true, .idempotentHint = true, .openWorldHint = false},
        .runOnApplicationThread = true,
    });
    registerTool({
        .name        = "nvpro_set_parameters",
        .description = "Validate and update one or more exposed sample parameters.",
        .inputSchema = R"({"type":"object","properties":{"values":{"type":"object","minProperties":1}},"required":["values"],"additionalProperties":false})",
        .handler                = [this](std::string_view arguments) { return setParameters(arguments); },
        .annotations            = {.openWorldHint = false},
        .runOnApplicationThread = true,
    });
    parameters               = std::move(selected);
    parameterToolsRegistered = true;
  }

  void registerParameters(const nvutils::ParameterRegistry& registry, std::initializer_list<std::string_view> names)
  {
    registerParameterTools(detail::selectParameters(registry, names));
  }

#if defined(NVPRO2_TRACK_PARAMETER_REGISTRIES)
  void registerAutomaticParameters()
  {
    if(automaticParameters)
      return;

    if(parameterToolsRegistered)
    {
      LOGW("Automatic MCP parameter exposure replaces the explicit parameter allowlist\n");
    }
    else
    {
      // Register the tools even if no parameters exist yet. Automatic mode
      // resolves live registries when each call runs on the application thread.
      registerParameterTools({});
    }
    // Parameter metadata does not say whether a value is read after startup.
    // Tell agents not to assume that a successful write changed live behavior.
    tools.at("nvpro_get_parameters").description =
        "Return all automatically exposed parameters, or only the requested names. Some may only be read at launch; MCP cannot determine which values affect the running application.";
    tools.at("nvpro_set_parameters").description =
        "Validate and update automatically exposed parameter storage. Some values may only be read at launch, so a successful write may not affect the running application.";
    automaticParameters = true;
  }

  detail::ParameterSelection automaticParameterSelection()
  {
    detail::ParameterSelection selection = detail::selectAllParameters(nvutils::getActiveParameterRegistryParameters());
    for(size_t duplicateIndex = 0; duplicateIndex < selection.duplicateNames.size(); duplicateIndex++)
    {
      const std::string& duplicateName = selection.duplicateNames[duplicateIndex];
      bool               warned{};
      for(size_t warnedIndex = 0; warnedIndex < warnedDuplicateParameters.size(); warnedIndex++)
      {
        if(warnedDuplicateParameters[warnedIndex] == duplicateName)
        {
          warned = true;
          break;
        }
      }
      if(!warned)
      {
        LOGW("Automatic MCP parameter exposure skipped duplicate name '%s'\n", duplicateName.c_str());
        warnedDuplicateParameters.push_back(duplicateName);
      }
    }
    return selection;
  }
#endif

  ToolResult getParameters(std::string_view arguments)
  {
#if defined(NVPRO2_TRACK_PARAMETER_REGISTRIES)
    if(automaticParameters)
    {
      detail::ParameterSelection selection = automaticParameterSelection();
      return detail::getParameters(selection.parameters, arguments);
    }
#endif
    return detail::getParameters(parameters, arguments);
  }

  ToolResult setParameters(std::string_view arguments)
  {
#if defined(NVPRO2_TRACK_PARAMETER_REGISTRIES)
    if(automaticParameters)
    {
      detail::ParameterSelection selection = automaticParameterSelection();
      return detail::setParameters(selection.parameters, arguments);
    }
#endif
    return detail::setParameters(parameters, arguments);
  }

  void registerProfiler(const nvutils::ProfilerManager& manager)
  {
    if(attached)
      throw std::logic_error("MCP profiler must be registered before the element is attached");
    if(profiler)
      throw std::logic_error("an MCP profiler is already registered");
    if(tools.contains("nvpro_get_profiler_snapshot"))
      throw std::logic_error("MCP profiler tool name is already registered");

    profiler = &manager;
    registerTool({
        .name        = "nvpro_get_profiler_snapshot",
        .description = "Return the latest CPU and GPU timing snapshots from the application's existing profiler.",
        .handler     = [this](std::string_view arguments) { return profilerSnapshot(arguments); },
        .annotations = {.readOnlyHint = true, .idempotentHint = true, .openWorldHint = false},
    });
  }

  ToolResult enqueueAndWait(const std::string& name, const std::string& arguments)
  {
    auto call                      = std::make_shared<PendingCall>();
    call->name                     = name;
    call->arguments                = arguments;
    std::future<ToolResult> result = call->promise.get_future();

    {
      std::lock_guard lock(queueMutex);
      if(!accepting)
        throw std::runtime_error("application MCP endpoint is stopping");
      pendingCalls.push_back(call);
    }

    if(result.wait_for(info.toolCallTimeout) != std::future_status::ready)
    {
      call->canceled.store(true);
      throw std::runtime_error("timed out waiting for the application thread");
    }
    return result.get();
  }

  ToolResult executeTool(const std::string& name, const std::string& arguments)
  {
    try
    {
      const std::unordered_map<std::string, Tool>::iterator it = tools.find(name);
      if(it == tools.end())
        throw std::runtime_error("unknown MCP tool: " + name);
      if(it->second.inputSchema == noArgumentsSchema)
        requireEmptyArguments(arguments, name);
      return it->second.handler(arguments);
    }
    catch(const std::exception& e)
    {
      return ToolResult::error(mcp::json{{"error", e.what()}, {"tool", name}}.dump());
    }
    catch(...)
    {
      return ToolResult::error(mcp::json{{"error", "unknown tool failure"}, {"tool", name}}.dump());
    }
  }

  void completeCall(const std::shared_ptr<PendingCall>& call)
  {
    if(call->canceled.load())
      return;
    call->promise.set_value(executeTool(call->name, call->arguments));
  }

  void drain()
  {
    std::deque<std::shared_ptr<PendingCall>> calls;
    {
      std::lock_guard lock(queueMutex);
      calls.swap(pendingCalls);
    }

    for(size_t callIndex = 0; callIndex < calls.size(); callIndex++)
    {
      const std::shared_ptr<PendingCall>& call = calls[callIndex];
      if(app && !app->isHeadless() && (call->name == "nvpro_capture_screenshot" || call->name == "nvpro_save_screenshot"))
      {
        // Swapchain pixels are only owned between submission and presentation.
        // Defer these calls to that boundary instead of making an otherwise
        // responsive onUpdate() path access an image owned by presentation.
        deferredScreenshotCalls.push_back(call);
        continue;
      }
      completeCall(call);
    }
  }

  void completeDeferredScreenshots()
  {
    std::deque<std::shared_ptr<PendingCall>> calls;
    calls.swap(deferredScreenshotCalls);
    for(size_t callIndex = 0; callIndex < calls.size(); callIndex++)
    {
      completeCall(calls[callIndex]);
    }
  }

  void failPending(std::string_view reason)
  {
    std::deque<std::shared_ptr<PendingCall>> calls;
    {
      std::lock_guard lock(queueMutex);
      accepting = false;
      calls.swap(pendingCalls);
    }
    calls.insert(calls.end(), deferredScreenshotCalls.begin(), deferredScreenshotCalls.end());
    deferredScreenshotCalls.clear();

    for(size_t callIndex = 0; callIndex < calls.size(); callIndex++)
    {
      const std::shared_ptr<PendingCall>& call = calls[callIndex];
      if(!call->canceled.load())
      {
        call->promise.set_value(ToolResult::error(mcp::json{{"error", reason}, {"tool", call->name}}.dump()));
      }
    }
  }

  ToolResult applicationState(std::string_view arguments) const
  {
    requireEmptyArguments(arguments, "nvpro_get_application_state");
    if(!app)
      return ToolResult::error(R"({"error":"application is not attached"})");

    const VkExtent2D viewport = app->getViewportSize();
    const VkExtent2D window   = app->getWindowSize();
    return ToolResult::success(mcp::json{
        {"server", {{"name", info.serverName}, {"version", info.serverVersion}}},
        {"application",
         {
             {"headless", app->isHeadless()},
             {"vsync", app->isVsync()},
             {"frameIndex", frameIndex},
             {"viewport", {{"width", viewport.width}, {"height", viewport.height}}},
             {"window", {{"width", window.width}, {"height", window.height}}},
         }},
    }
                                   .dump(2));
  }

  ToolResult shutdown(std::string_view arguments)
  {
    requireEmptyArguments(arguments, "nvpro_shutdown");
    if(!app)
      return ToolResult::error(R"({"error":"application is not attached"})");
    app->close();
    return ToolResult::success(R"({"status":"shutdown-requested"})");
  }

  ToolResult captureScreenshot(std::string_view arguments) const
  {
    requireEmptyArguments(arguments, "nvpro_capture_screenshot");
    if(!app)
      return ToolResult::error(R"({"error":"application is not attached"})");
    if(app->isHeadless())
      return ToolResult::error(R"({"error":"generic screenshot capture requires a windowed swapchain"})");
    if(frameIndex == 0)
      return ToolResult::error(R"({"error":"no completed frame is available"})");

    std::vector<uint8_t> pngData;
    const VkResult       result = app->encodeScreenShotToPng(pngData);
    if(result != VK_SUCCESS)
      return ToolResult::error(mcp::json{{"error", "screenshot capture failed"}, {"vkResult", result}}.dump());

    const VkExtent2D size = app->getWindowSize();
    return ToolResult::success(
        mcp::json{
            {"status", "captured"},
            {"frameIndex", frameIndex},
            {"width", size.width},
            {"height", size.height},
            {"byteSize", pngData.size()},
        }
            .dump(2),
        {
            .data     = base64::encode(reinterpret_cast<const char*>(pngData.data()), pngData.size()),
            .mimeType = "image/png",
        });
  }

  ToolResult saveScreenshot(std::string_view arguments) const
  {
    if(!app)
      return ToolResult::error(R"({"error":"application is not attached"})");
    if(app->isHeadless())
      return ToolResult::error(R"({"error":"generic screenshot capture requires a windowed swapchain"})");
    if(frameIndex == 0)
      return ToolResult::error(R"({"error":"no completed frame is available"})");

    const mcp::json parsed = arguments.empty() ? mcp::json::object() : mcp::json::parse(arguments);
    if(!parsed.is_object() || parsed.size() != 1 || !parsed.contains("filename") || !parsed["filename"].is_string())
      throw std::invalid_argument("nvpro_save_screenshot requires only a string filename");

    std::filesystem::path filename = nvutils::pathFromUtf8(parsed["filename"].get<std::string>());
    if(filename.empty() || !nvutils::extensionMatches(filename, ".png"))
      throw std::invalid_argument("nvpro_save_screenshot filename must end in .png");
    filename                       = std::filesystem::absolute(filename).lexically_normal();
    const std::string filenameUtf8 = nvutils::utf8FromPath(filename);
    if(filenameUtf8.empty())
      throw std::invalid_argument("nvpro_save_screenshot filename is not valid UTF-8");

    const VkResult result = app->saveScreenShot(filename, 100);
    if(result != VK_SUCCESS)
      return ToolResult::error(mcp::json{{"error", "screenshot save failed"}, {"vkResult", result}}.dump());

    std::error_code fileError;
    const uintmax_t byteSize = std::filesystem::file_size(filename, fileError);
    if(fileError)
      return ToolResult::error(mcp::json{
          {"error", "saved screenshot metadata is unavailable"}, {"filename", filenameUtf8}, {"detail", fileError.message()}}
                                   .dump());

    const VkExtent2D size = app->getWindowSize();
    return ToolResult::success(mcp::json{
        {"status", "saved"},
        {"filename", filenameUtf8},
        {"frameIndex", frameIndex},
        {"width", size.width},
        {"height", size.height},
        {"byteSize", byteSize},
        {"mimeType", "image/png"},
    }
                                   .dump(2));
  }

  ToolResult profilerSnapshot(std::string_view arguments) const
  {
    requireEmptyArguments(arguments, "nvpro_get_profiler_snapshot");

    std::vector<nvutils::ProfilerTimeline::Snapshot> frameSnapshots;
    std::vector<nvutils::ProfilerTimeline::Snapshot> asyncSnapshots;
    profiler->getSnapshots(frameSnapshots, asyncSnapshots);

    mcp::json frames = mcp::json::array();
    for(size_t snapshotIndex = 0; snapshotIndex < frameSnapshots.size(); snapshotIndex++)
    {
      frames.push_back(timelineJson(frameSnapshots[snapshotIndex]));
    }

    mcp::json asyncs = mcp::json::array();
    for(size_t snapshotIndex = 0; snapshotIndex < asyncSnapshots.size(); snapshotIndex++)
    {
      asyncs.push_back(timelineJson(asyncSnapshots[snapshotIndex]));
    }

    return ToolResult::success(mcp::json{
        {"unit", "microseconds"},
        {"frameTimelines", std::move(frames)},
        {"asyncTimelines", std::move(asyncs)},
    }
                                   .dump(2));
  }

  ToolResult logs(std::string_view arguments)
  {
    const mcp::json parsed = arguments.empty() ? mcp::json::object() : mcp::json::parse(arguments);
    if(!parsed.is_object())
      throw std::invalid_argument("nvpro_get_logs arguments must be an object");
    for(mcp::json::const_iterator it = parsed.cbegin(); it != parsed.cend(); it++)
    {
      if(it.key() != "limit" && it.key() != "afterSequence" && it.key() != "minimumLevel")
        throw std::invalid_argument("unknown nvpro_get_logs argument: " + it.key());
    }

    size_t limit = 100;
    if(parsed.contains("limit"))
    {
      if(!parsed["limit"].is_number_unsigned() && !parsed["limit"].is_number_integer())
        throw std::invalid_argument("nvpro_get_logs limit must be an integer");
      const int64_t requestedLimit = parsed["limit"].get<int64_t>();
      if(requestedLimit < 1 || requestedLimit > static_cast<int64_t>(maxLogQueryEntries))
        throw std::out_of_range("nvpro_get_logs limit must be between 1 and 1000");
      limit = static_cast<size_t>(requestedLimit);
    }

    bool     hasAfterSequence{};
    uint64_t afterSequence{};
    if(parsed.contains("afterSequence"))
    {
      if(!parsed["afterSequence"].is_number_unsigned()
         && (!parsed["afterSequence"].is_number_integer() || parsed["afterSequence"].get<int64_t>() < 0))
      {
        throw std::invalid_argument("nvpro_get_logs afterSequence must be a non-negative integer");
      }
      afterSequence    = parsed["afterSequence"].get<uint64_t>();
      hasAfterSequence = true;
    }

    nvutils::Logger::LogLevel minimumLevel = nvutils::Logger::eDEBUG;
    if(parsed.contains("minimumLevel"))
    {
      if(!parsed["minimumLevel"].is_string())
        throw std::invalid_argument("nvpro_get_logs minimumLevel must be a string");
      minimumLevel = parseLogLevel(parsed["minimumLevel"].get<std::string>());
    }

    std::vector<LogEntry> selected;
    uint64_t              oldestSequence{};
    uint64_t              latestSequence{};
    uint64_t              nextAfterSequence = afterSequence;
    uint64_t              droppedEntries{};
    {
      std::lock_guard lock(logMutex);
      if(!logEntries.empty())
      {
        oldestSequence = logEntries.front().sequence;
        latestSequence = logEntries.back().sequence;
      }
      droppedEntries = droppedLogEntries;
      selected.reserve(std::min(limit, logEntries.size()));
      if(hasAfterSequence)
      {
        for(size_t entryIndex = 0; entryIndex < logEntries.size(); entryIndex++)
        {
          const LogEntry& entry = logEntries[entryIndex];
          if(entry.sequence <= afterSequence)
            continue;

          nextAfterSequence = entry.sequence;
          if(entry.level >= minimumLevel)
          {
            selected.push_back(entry);
            if(selected.size() == limit)
              break;
          }
        }
      }
      else
      {
        nextAfterSequence = latestSequence;
        for(size_t entryIndex = logEntries.size(); entryIndex > 0 && selected.size() < limit; entryIndex--)
        {
          const LogEntry& entry = logEntries[entryIndex - 1];
          if(entry.level >= minimumLevel)
            selected.push_back(entry);
        }
        std::reverse(selected.begin(), selected.end());
      }
    }

    mcp::json entries = mcp::json::array();
    for(size_t entryIndex = 0; entryIndex < selected.size(); entryIndex++)
    {
      const LogEntry& entry = selected[entryIndex];
      entries.push_back({
          {"sequence", entry.sequence},
          {"level", logLevelName(entry.level)},
          {"message", entry.message},
      });
    }

    const bool truncated = hasAfterSequence && oldestSequence > 0 && afterSequence < oldestSequence - 1;
    return ToolResult::success(mcp::json{
        {"entries", std::move(entries)},
        {"oldestSequence", oldestSequence},
        {"latestSequence", latestSequence},
        {"nextAfterSequence", nextAfterSequence},
        {"droppedEntries", droppedEntries},
        {"truncated", truncated},
    }
                                   .dump(2));
  }

  ElementCreateInfo        info;
  nvapp::Application*      app{};
  bool                     attached{false};
  bool                     accepting{false};
  uint64_t                 frameIndex{0};
  std::chrono::nanoseconds previousAcquireTimeout{std::chrono::nanoseconds::max()};
  std::chrono::nanoseconds appliedAcquireTimeout{std::chrono::nanoseconds::max()};

  std::unordered_map<std::string, Tool> tools;
  std::vector<std::string>              toolOrder;
  detail::ParameterList                 parameters;
  bool                                  parameterToolsRegistered{false};
#if defined(NVPRO2_TRACK_PARAMETER_REGISTRIES)
  bool                     automaticParameters{false};
  std::vector<std::string> warnedDuplicateParameters;
#endif
  const nvutils::ProfilerManager* profiler{};

  nvutils::Logger::LogObserver logObserver{};
  std::mutex                   logMutex;
  std::deque<LogEntry>         logEntries;
  size_t                       logBytes{};
  uint64_t                     nextLogSequence{1};
  uint64_t                     droppedLogEntries{};

  std::mutex                               queueMutex;
  std::deque<std::shared_ptr<PendingCall>> pendingCalls;
  std::deque<std::shared_ptr<PendingCall>> deferredScreenshotCalls;

  std::unique_ptr<mcp::server> server;
};

Element::Element(ElementCreateInfo info)
    : m_impl(std::make_unique<Impl>(std::move(info)))
{
  registerTool({
      .name        = "nvpro_get_application_state",
      .description = "Return generic live application mode, frame, window, viewport, and MCP server identity.",
      .handler     = [this](std::string_view arguments) { return m_impl->applicationState(arguments); },
      .annotations = {.readOnlyHint = true, .idempotentHint = true, .openWorldHint = false},
      .runOnApplicationThread = true,
  });
  registerTool({
      .name = "nvpro_get_logs",
      .description = "Return bounded nvutils logger-call fragments captured since this MCP element was created. Concatenate messages in sequence order to reconstruct console text. Omit afterSequence for the newest entries or pass nextAfterSequence to poll incrementally.",
      .inputSchema = R"({"type":"object","properties":{"limit":{"type":"integer","minimum":1,"maximum":1000,"default":100},"afterSequence":{"type":"integer","minimum":0},"minimumLevel":{"type":"string","enum":["debug","stats","ok","info","warning","error"],"default":"debug"}},"additionalProperties":false})",
      .handler     = [this](std::string_view arguments) { return m_impl->logs(arguments); },
      .annotations = {.readOnlyHint = true, .idempotentHint = true, .openWorldHint = false},
  });
  registerTool({
      .name                   = "nvpro_shutdown",
      .description            = "Request a clean application shutdown.",
      .handler                = [this](std::string_view arguments) { return m_impl->shutdown(arguments); },
      .annotations            = {.destructiveHint = true, .idempotentHint = true, .openWorldHint = false},
      .runOnApplicationThread = true,
  });
  registerTool({
      .name        = "nvpro_capture_screenshot",
      .description = "Return the next completed windowed frame as an in-memory PNG image without creating a file.",
      .handler     = [this](std::string_view arguments) { return m_impl->captureScreenshot(arguments); },
      .annotations = {.readOnlyHint = true, .destructiveHint = false, .openWorldHint = false},
      .runOnApplicationThread = true,
  });
  registerTool({
      .name        = "nvpro_save_screenshot",
      .description = "Save the next completed windowed frame to a PNG file and return its resolved metadata.",
      .inputSchema = R"({"type":"object","properties":{"filename":{"type":"string","description":"Destination path ending in .png"}},"required":["filename"],"additionalProperties":false})",
      .handler                = [this](std::string_view arguments) { return m_impl->saveScreenshot(arguments); },
      .annotations            = {.openWorldHint = false},
      .runOnApplicationThread = true,
  });
}

Element::~Element()
{
  assert(!m_impl->attached && "onDetach() was not called before destruction");
}

void Element::registerTool(Tool tool)
{
  m_impl->registerTool(std::move(tool));
}

void Element::registerParameters(const nvutils::ParameterRegistry& registry, std::initializer_list<std::string_view> names)
{
  m_impl->registerParameters(registry, names);
}

void Element::registerProfiler(const nvutils::ProfilerManager& profiler)
{
  m_impl->registerProfiler(profiler);
}

bool Element::running() const
{
  return m_impl->server && m_impl->server->is_running();
}

void Element::onAttach(nvapp::Application* app)
{
  if(m_impl->attached)
    return;

#if defined(NVPRO2_TRACK_PARAMETER_REGISTRIES)
  m_impl->registerAutomaticParameters();
#endif

  mcp::server::configuration config;
  config.host            = m_impl->info.host;
  config.port            = m_impl->info.port;
  config.name            = m_impl->info.serverName;
  config.version         = m_impl->info.serverVersion;
  config.mcp_endpoint    = m_impl->info.endpoint;
  config.session_timeout = m_impl->info.sessionTimeoutSeconds;

  m_impl->server = std::make_unique<mcp::server>(config);
  m_impl->server->set_server_info(m_impl->info.serverName, m_impl->info.serverVersion);
  m_impl->server->set_capabilities({{"tools", mcp::json::object()}});
  m_impl->server->set_instructions(m_impl->info.instructions);

  for(size_t toolIndex = 0; toolIndex < m_impl->toolOrder.size(); toolIndex++)
  {
    const std::string& name = m_impl->toolOrder[toolIndex];
    const Tool&        tool = m_impl->tools.at(name);
    mcp::tool          definition{
                 .name              = tool.name,
                 .description       = tool.description,
                 .parameters_schema = mcp::json::parse(tool.inputSchema),
                 .annotations       = annotationsJson(tool.annotations),
    };
    m_impl->server->register_tool(definition, [this, name](const mcp::json& arguments, const std::string&) {
      // cpp-mcp represents an omitted optional arguments field as an empty
      // array. Normalize that transport detail to the object MCP tools expect.
      const std::string argumentsJson = arguments.is_array() && arguments.empty() ? "{}" : arguments.dump();
      const Tool&       tool          = m_impl->tools.at(name);
      ToolResult        result        = tool.runOnApplicationThread ? m_impl->enqueueAndWait(name, argumentsJson) :
                                                                      m_impl->executeTool(name, argumentsJson);
      const mcp::json   payload       = mcp::json::parse(result.json);
      if(result.isError)
        throw std::runtime_error(payload.dump());
      mcp::json content = textContent(payload.dump(2));
      for(const ImageContent& image : result.images)
      {
        if(image.data.empty() || image.mimeType.empty())
          throw std::runtime_error("MCP image content requires encoded data and a MIME type");
        content.push_back({{"type", "image"}, {"data", image.data}, {"mimeType", image.mimeType}});
      }
      return content;
    });
  }

  m_impl->app       = app;
  m_impl->accepting = true;

  bool started{};
  try
  {
    started = m_impl->server->start(false);
  }
  catch(...)
  {
    m_impl->accepting = false;
    m_impl->app       = nullptr;
    m_impl->server.reset();
    throw;
  }

  if(!started)
  {
    m_impl->accepting = false;
    m_impl->app       = nullptr;
    m_impl->server.reset();
    LOGE("Failed to bind MCP server to http://%s:%u%s. Port %u is already in use or unavailable; select another MCP port.\n",
         m_impl->info.host.c_str(), m_impl->info.port, m_impl->info.endpoint.c_str(), m_impl->info.port);
    return;
  }

  // Application-thread tools are serviced by onUpdate(), so bound presentation
  // waits only after the endpoint is serving and external callers can need it.
  m_impl->previousAcquireTimeout                   = app->getSwapchainAcquireTimeout();
  const std::chrono::nanoseconds mcpAcquireTimeout = std::chrono::milliseconds(100);
  m_impl->appliedAcquireTimeout = m_impl->previousAcquireTimeout < mcpAcquireTimeout ? m_impl->previousAcquireTimeout : mcpAcquireTimeout;
  app->setSwapchainAcquireTimeout(m_impl->appliedAcquireTimeout);
  m_impl->frameIndex = 0;
  m_impl->attached   = true;
  LOGI("MCP server starting at http://%s:%u%s\n", m_impl->info.host.c_str(), m_impl->info.port, m_impl->info.endpoint.c_str());
}

void Element::onUpdate()
{
  m_impl->drain();
}

void Element::onPreRender()
{
  ++m_impl->frameIndex;
}

void Element::onPostRender()
{
  m_impl->completeDeferredScreenshots();
}

void Element::onDetach()
{
  if(!m_impl || !m_impl->attached)
    return;

  m_impl->failPending("application MCP endpoint stopped");
  if(m_impl->server)
  {
    m_impl->server->stop();
    m_impl->server.reset();
  }
  if(m_impl->app && m_impl->app->getSwapchainAcquireTimeout() == m_impl->appliedAcquireTimeout)
  {
    // Restore the application's prior policy unless another owner changed it
    // while MCP was attached.
    m_impl->app->setSwapchainAcquireTimeout(m_impl->previousAcquireTimeout);
  }
  m_impl->app      = nullptr;
  m_impl->attached = false;
}

}  // namespace nvmcp
