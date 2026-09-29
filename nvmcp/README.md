# nvmcp

`nvpro2::nvmcp` is an opt-in Streamable HTTP MCP endpoint for live
`nvapp::Application` control. It is disabled by default so samples that do not
use MCP do not download, build, or link the server.

## Design intent

`nvmcp` is development automation for agents working on samples, not a remote
control API for shipping applications. The initial surface is deliberately
small and composable:

- MCP is opt-in and loopback-only so the many samples using `nvpro_core2` do
  not gain a dependency, attack surface, or background server by default.
- Common tools cover lifecycle and explicitly selected runtime state. Samples
  register domain actions such as shader reload themselves, keeping
  sample-specific policy out of the common library.
- Tool handlers execute on MCP worker threads by default so expensive work does
  not interrupt rendering. Tools that access unsynchronized application state
  explicitly opt into short application-thread handlers, which `onUpdate()`
  keeps available when rendering is skipped. The MCP element opts into bounded
  presentation waits so applications without MCP retain their original
  blocking swapchain behavior.
- Parameters require an explicit allowlist by default. This makes the sample
  owner choose stable, meaningful agent controls instead of accidentally
  exposing every CLI option or internal variable.
- MCP controls backing state and named actions rather than ImGui widgets. The
  same controls therefore remain inspectable, deterministic, and usable in
  headless runs.

Enable and link the library before adding `nvpro_core2`:

```cmake
option(NVPRO2_ENABLE_nvmcp "Enable generic nvpro_core2 MCP control" ON)

target_link_libraries(${PROJECT_NAME} PRIVATE nvpro2::nvmcp)
```

Enabling `NVPRO2_ENABLE_nvmcp` fetches the pinned cpp-mcp dependency during
CMake configuration. It is not downloaded when the option remains disabled.

Trusted development builds can additionally enable
`NVPRO2_ENABLE_MCP_AUTO_PARAMETER_REGISTRY`. This exposes supported, uniquely
named parameters from every active `ParameterRegistry` without a per-sample
`registerParameters()` call:

```cmake
option(NVPRO2_ENABLE_nvmcp "Enable generic nvpro_core2 MCP control" ON)
option(NVPRO2_ENABLE_MCP_AUTO_PARAMETER_REGISTRY
       "Expose all supported parameters through MCP" ON)
```

The automatic mode deliberately replaces any explicit MCP parameter allowlist
and resolves the active registries for each tool call, so registries added after
the MCP element remain visible. Trigger and custom parameters remain excluded
because they represent actions. Duplicate names are ambiguous, so every
parameter with that name remains unexposed and a warning identifies the name.
The default build does not track registries, and samples must still create and
add the MCP element so server ownership and configuration remain explicit.
Parameter metadata does not currently distinguish runtime controls from values
consumed only during application or subsystem initialization, so automatic mode
may expose startup-only settings. Updating such a setting changes its registered
backing value but cannot make an initialized subsystem re-read it; use an
explicit allowlist or sample tool when that distinction is important.

Create the element, register sample tools, and add it to the application:

```cpp
#include <nvmcp/element_mcp.hpp>
#include <nvutils/parameter_registry.hpp>

nvutils::ParameterRegistry parameterRegistry;
float                      exposure{1.0F};
uint32_t                   maxSamples{64};
parameterRegistry.add({"exposure", "Exposure"}, &exposure);
parameterRegistry.add({"maxSamples", "Maximum samples"}, &maxSamples);

auto renderer = std::make_shared<MyRenderer>();
auto mcp = std::make_shared<nvmcp::Element>(
    nvmcp::ElementCreateInfo{.serverName = "my_sample"});

mcp->registerParameters(parameterRegistry, {"exposure", "maxSamples"});
mcp->registerTool({
    .name        = "my_sample_reload_shaders",
    .description = "Synchronously recompile this sample's shaders and return compiler diagnostics.",
    .handler     = [renderer](std::string_view argumentsJson) {
      const MyRenderer::ShaderReloadResult result = renderer->reloadShaders();
      const std::string response = serializeShaderReloadResult(result);
      return result.succeeded ? nvmcp::ToolResult::success(response) : nvmcp::ToolResult::error(response);
    },
    .annotations            = {.idempotentHint = true},
    .runOnApplicationThread = true,
});

app.addElement(renderer);
app.addElement(mcp);
```

Register tools before `app.addElement(mcp)`. Tool schemas and arguments cross
the public API as JSON text, while results use JSON text plus optional encoded
images, so sample translation units do not inherit `cpp-mcp`'s JSON dependency.

The shader-reload types above are illustrative: reload belongs to the sample,
and its tool must return the sample's actual completion status and compiler
diagnostics rather than assuming that a request succeeded.

Images are emitted as native MCP image content rather than base64 embedded in a
text result. Tool handlers run on MCP worker threads and may run concurrently;
samples are responsible for synchronizing state used from those handlers. Set
`runOnApplicationThread` for a short handler that must use state owned by the
application loop. Such calls are queued and executed once from `onUpdate()`,
including when a window is minimized or a frame is otherwise skipped. Do not
use it for work that can safely run in the background. Tool annotations use
MCP's conservative defaults; sample tools should explicitly narrow them only
when their behavior guarantees it is safe to do so. Tools without inputs accept
both an omitted MCP `arguments` field and an empty object, and reject other
input. Handlers with a custom input schema are responsible for validating its
semantics.

Windowed applications provide two screenshot paths. Calls wait for the next
submitted frame and complete before presentation takes ownership of its
swapchain image. `nvpro_capture_screenshot` returns an in-memory PNG as native
MCP image content and never creates a file. `nvpro_save_screenshot` requires a
`.png` destination and returns the resolved filename, frame index, dimensions,
byte size, and MIME type without returning image data. Windowed
`nvapp::Application` swapchains request transfer-source usage when they are
created. The generic tools reject headless applications because
`nvapp::Application` has no common headless output image; samples can register
a tool for their own render target. A minimized application completes a pending
capture after rendering resumes, subject to the configured MCP tool timeout.

`registerParameters()` adds `nvpro_get_parameters` and
`nvpro_set_parameters` for the explicitly named allowlist. Trigger and custom
parameters are intentionally excluded; register explicit tools for actions. It
must be called at most once and before the MCP element is attached. The
parameter registry and parameter destinations must outlive the MCP element.
Success callbacks follow the existing `ParameterParser` convention and must
not throw; their application-side effects are not transactional.

`nvpro_get_parameters` returns every exposed parameter when called without
arguments. Pass a non-empty `names` array to retrieve only specific parameters;
unknown or duplicate names are rejected so spelling mistakes are visible.

For settings already in `ParameterRegistry`, MCP exposure is one additional
`registerParameters()` call and reuses the existing name, help, type, bounds,
destination, and success callback. ImGui-only fields are not discovered
automatically: register their backing variables with `ParameterRegistry` when
they represent stable configuration, or use a named tool when the operation
has action semantics.

`registerProfiler()` adds `nvpro_get_profiler_snapshot`. It returns the latest
frame and asynchronous CPU/GPU timing snapshots already maintained by
`ProfilerManager`, in microseconds. Reusing the profiler's snapshots keeps the
MCP result consistent with the existing profiler UI and avoids a second timing
or averaging implementation. The profiler manager must outlive the MCP
element:

```cpp
mcp->registerProfiler(existingProfilerManager);
```

`nvpro_get_logs` returns a bounded history of calls emitted through
`nvutils::Logger` after the MCP element is created. A logger call may contain
multiple lines, and several calls without a trailing newline may form one
console line; concatenate `message` values in sequence order when console-style
text is needed. It observes the logger in addition to the application's existing
log callback, so enabling MCP does not replace an ImGui log window or other
consumer. Omit `afterSequence` to inspect the newest entries; for incremental
polling, pass the returned `nextAfterSequence` into the next call. Sequence
numbers are stable as new entries arrive. `minimumLevel` filters results and
`limit` bounds each response. The history is capped by entry count and memory,
and `truncated` reports when requested entries have already been evicted. The
logger's configured minimum level still applies, so messages filtered before
output cannot be recovered through MCP.

Each running application requires its own port. A busy or unavailable port is
reported during application startup instead of allowing clients to reach an
arbitrary process. Samples can expose `ElementCreateInfo::port` through their
existing command-line parameter registry; for example, `vk_gltf_renderer` uses
`--mcpPort 7672`.

For an MCP-controlled headless session, set `headlessFrameCount` to zero. The
application then renders until the common `nvpro_shutdown` tool or another
caller requests `Application::close()`.

## Security disclaimer

`nvmcp` is for local development on trusted machines. It uses unauthenticated
and unencrypted HTTP. SSL is disabled by default. The default endpoint is
`http://127.0.0.1:7671/mcp`. Any local process can invoke exposed tools.

## Connecting an agent

Start the MCP-enabled sample before connecting a client. Clients cache tool
lists, so restart or refresh the MCP connection after changing which tools a
sample registers.

### [Visual Studio Code](https://code.visualstudio.com/docs/agents/reference/mcp-configuration)

Create `.vscode/mcp.json` in the sample repository:

```json
{
  "servers": {
    "nvpro-core2": {
      "type": "http",
      "url": "http://127.0.0.1:7671/mcp"
    }
  }
}
```

Run `MCP: List Servers` from the Command Palette to inspect or restart the
server. Run `MCP: Reset Cached Tools` if a sample changes its registered tools.

### [Codex](https://developers.openai.com/codex/mcp/)

Create `.codex/config.toml` in the sample repository:

```toml
[mcp_servers.nvpro_core2]
url = "http://127.0.0.1:7671/mcp"
startup_timeout_sec = 10
tool_timeout_sec = 300
enabled = true
default_tools_approval_mode = "prompt"
```

Trust the project, then restart the Codex IDE extension or start a new Codex
session after changing the configuration. In the CLI, verify the connection
with:

```console
codex mcp list
```

The `/mcp` command shows MCP status in an interactive Codex session.

### [Cursor](https://docs.cursor.com/context/model-context-protocol)

Create `.cursor/mcp.json` in the sample repository:

```json
{
  "mcpServers": {
    "nvpro-core2": {
      "type": "http",
      "url": "http://127.0.0.1:7671/mcp"
    }
  }
}
```

Approve the project server when prompted. `cursor-agent mcp list` and
`cursor-agent mcp list-tools nvpro-core2` inspect the connection and tools.

### [Claude Code](https://code.claude.com/docs/en/mcp)

Add the running endpoint at project scope:

```console
claude mcp add --transport http --scope project nvpro-core2 http://127.0.0.1:7671/mcp
claude mcp list
```

Claude Code writes project-scoped servers to `.mcp.json` and asks for approval
before using configuration supplied by a repository. Use `/mcp` to inspect the
connection in an interactive session.

## Future directions

These ideas are intentionally recorded without fixing their exact tool schemas
before the required application behavior is designed.

### Deferred frame capture

The screenshot tools implement a narrow one-frame deferred operation. A shared
deferred-operation mechanism could generalize this behavior for other
multi-frame tools and add explicit status or cancellation if operations become
long-lived.

### Deterministic frame control

Agent workflows would benefit from pausing at a safe frame boundary, stepping
an exact number of frames, and waiting for a requested frame to finish before
inspecting state or capturing output. Windowed and headless behavior, GPU
completion versus CPU submission, and interaction with minimized windows must
be defined before selecting tools such as `pause`, `step`, or `wait_for_frame`.
Frame control must not block the application thread while waiting for the next
frame it is responsible for producing.

### ImGui exposure and interaction

Prefer exposing the stable backing configuration and named actions used by UI
widgets. This works headlessly and avoids coupling agents to labels, layout,
coordinates, focus, and transient widget IDs. Longer-term semantic ImGui
metadata could make controls discoverable, while input injection or direct
widget interaction should be reserved for testing behavior that cannot be
verified through backing state.
