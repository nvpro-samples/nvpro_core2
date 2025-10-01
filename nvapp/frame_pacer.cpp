/*
 * Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "frame_pacer.hpp"

#include <nvvk/check_error.hpp>

#include <GLFW/glfw3.h>
#include <volk/volk.h>
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#undef APIENTRY  // GLFW defines this but Windows tries to redefine it
#include <Windows.h>
#include <timeapi.h>
#endif

#include <cassert>
#include <cmath>
#include <limits>
#include <thread>

namespace nvapp {


double getMonitorsMinRefreshRate()
{
  // We need our target frame rate. We get this once per frame in case the
  // user changes their monitor's frame rate.
  // Ideally we'd get the exact composition rate for the current swapchain;
  // VK_EXT_present_timing will hopefully give us that when it's released.
  // Currently we use GLFW; this means we don't need anything
  // platform-specific, but means we only get an integer frame rate,
  // rounded down, across monitors. We take the minimum to avoid building up
  // frame latency.
  double refreshRate = std::numeric_limits<double>::infinity();
  {
    int           numMonitors = 0;
    GLFWmonitor** monitors    = glfwGetMonitors(&numMonitors);
    for(int i = 0; i < numMonitors; i++)
    {
      const GLFWvidmode* videoMode = glfwGetVideoMode(monitors[i]);
      if(videoMode)
      {
        refreshRate = std::min(refreshRate, static_cast<double>(videoMode->refreshRate));
      }
    }
  }
  // If we have no information about the frame rate or an impossible value,
  // use a default.
  if(std::isinf(refreshRate) || refreshRate <= 0.0)
  {
    refreshRate = 60.0;
  }

  return refreshRate;
}

void FramePacer::pace(double refreshRate)
{
  const double refreshInterval = 1.0 / refreshRate;

  // Pacing the CPU by enforcing at least `refreshInterval` seconds between
  // frames is all we need! If the GPU is fast things are OK; if the GPU is
  // slow then vkWaitSemaphores will take more time in the frame, which
  // will be counted in the CPU time.
  const double cpuTime   = m_cpuTimer.getSeconds();
  double       sleepTime = refreshInterval - cpuTime;
#ifdef _WIN32
  // On Windows, we know that 1ms is just about the right time to subtract;
  // it's just under the average amount that Windows adds to the sleep call.
  // On Linux the timers are accurate enough that we don't need this.
  sleepTime -= 1e-3;
#endif
  if(sleepTime > 0.0)
  {
    // Reuse the timer to measure how long sleeps actually take.
    m_cpuTimer.reset();
#ifdef _WIN32
    // On Windows, the default timer might quantize to 15.625 ms; see
    // https://randomascii.wordpress.com/2020/10/04/windows-timer-resolution-the-great-rule-change/ .
    // We use timeBeginPeriod to temporarily increase the resolution to 1 ms.
    timeBeginPeriod(1);
#endif
    std::this_thread::sleep_for(std::chrono::duration<double>(sleepTime));
#ifdef _WIN32
    timeEndPeriod(1);
#endif
  }

  // Reset the cpuTimer for the start of the frame.
  m_cpuTimer.reset();
}

}  // namespace nvapp
