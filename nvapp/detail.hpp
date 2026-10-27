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

#include <functional>
#include <string>
#include <vector>

#include <GLFW/glfw3.h>
#include <glm/vec2.hpp>
#include <imgui/imgui.h>

namespace nvgui {
class SettingsHandler;
}

/*-------------------------------------------------------------------------------------------------
# namespace nvapp::detail

> Implementation details shared by `nvapp::Application` and `nvapp::ApplicationBase`:
> the window, ImGui and frame-ring bookkeeping. Not part of the public API.

Free functions over the members each application class owns, so both keep
their own layout and the Vulkan class its public and protected surface.
-------------------------------------------------------------------------------------------------*/
namespace nvapp::detail {

// GLFW and the ImGui and ImPlot contexts, created once by the application constructor.
void initContexts();

// <exe>.ini as UTF-8, where ImGui keeps the window and layout settings.
std::string defaultIniFilename();

// Style, the "Application" settings handler for the window size and position,
// the .ini load and auto-save policy (persisting sessions only), and the fonts.
void initImGuiSettings(nvgui::SettingsHandler& handler, glm::uvec2& winSize, glm::ivec2& winPos, bool persistSettings, const std::string& iniFilename);

// Window size and position: `requested` when non-zero (then centered), else the
// saved values, else 80% of the primary monitor centered; 800x600 without a
// monitor or headless. Saved positions off every monitor are re-centered.
void chooseWindowSizeAndPos(const glm::uvec2& requested, bool headless, glm::uvec2& winSize, glm::ivec2& winPos);
void centerOnPrimaryMonitor(const glm::ivec2& winSize, glm::ivec2& winPos);
bool isWindowPosValid(const glm::ivec2& winPos);

// A DPI-aware window without a client API (the back end owns the swapchain),
// with the file-drop callback and `userPointer` installed. Null, with a log
// line, when there is no display.
GLFWwindow* createWindow(const char* title, const glm::uvec2& size, const glm::ivec2& pos, void* userPointer, GLFWdropfun dropCallback);

// Dockspace over the main viewport with "Viewport" as the central node;
// `dockSetup` lays out the other windows on first use, else "Settings" docks left.
void setupImguiDock(const std::function<void(ImGuiID)>& dockSetup);

// Rescales the ImGui fonts when the window's content scale moved away from
// `previousScale`; returns the current scale.
float updateDpiScale(GLFWwindow* window, float previousScale);

// Updates `size` with the framebuffer size when it is non-zero and differs;
// a minimized window reports (0,0), which keeps the last valid size.
bool framebufferResized(GLFWwindow* window, glm::uvec2& size);

// Size of the "Viewport" window when it exists (kept open, without padding),
// else `windowSize`.
glm::uvec2 viewportWindowSize(const glm::uvec2& windowSize);

// Deferred releases, one list per frame-ring slot: run when the slot comes round again.
using FreeQueue = std::vector<std::vector<std::function<void()>>>;
void freeQueueSubmit(FreeQueue& queue, uint32_t frame, std::function<void()>&& func);  // runs it when there is no ring
void freeQueueReset(FreeQueue& queue, uint32_t size);  // runs everything; the device must be idle
void freeQueueRun(FreeQueue& queue, uint32_t frame);

}  // namespace nvapp::detail
