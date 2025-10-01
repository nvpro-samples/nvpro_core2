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

#include <nvutils/timers.hpp>
namespace nvapp {

// Get the minimum refresh rate of the monitors
double getMonitorsMinRefreshRate();


// This class tries to reduce latency by working with the swapchain. The idea
// is that we want to delay the start of the frame (before input sampling) just
// long enough so that the GPU presents just before the compositor picks up
// the frame to composite.
//
// For now, we aim for an easier goal: submit a frame once VSync. Since the
// compositor consumes one frame per VSync, we should render at most one frame
// per VSync; any faster and we'd get swapchain backpressure and thus latency.
class FramePacer
{
public:
  // Call this just before glfwPollEvents() to sleep.
  void pace(double refreshRate = getMonitorsMinRefreshRate());

private:
  // System state
  nvutils::PerformanceTimer m_cpuTimer;
};

}  // namespace nvapp
