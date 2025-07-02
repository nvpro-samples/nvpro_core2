/*
 * Copyright (c) 2023-2025, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <string>

namespace nvutils {

// Generic utility class for measuring CPU time.
//
// Usage:
// ```
// PerformanceTimer timer;
// // ... do something...
// printf("Operation 1 took %f seconds\n", timer.getSeconds());
//
// timer.reset();
// // ... do something else...
// printf("Operation 2 took %f seconds\n", timer.getSeconds();
// ```
//
// On Windows and Unix systems, this timer should have precision within 100
// nanoseconds and ignore time when the computer is suspended (e.g. asleep or
// hibernating).
//
// On other systems, this falls back to std::chrono::steady_clock.
//
// Exact precision and dependency depends on the platform; Windows, for instance,
// will attempt to correct for innacuracies (https://learn.microsoft.com/en-us/windows/win32/sysinfo/acquiring-high-resolution-time-stamps#low-level-hardware-clock-characteristics),
// and on Unix we choose a method that's synced to Network Time Protocol (at
// the expense of a higher chance of non-monotonicity).
class PerformanceTimer
{
public:
  PerformanceTimer() { reset(); }

  // Starts or re-starts counting from the current time.
  void reset() { m_start = now(); }

  // Returns the number of seconds since the clock was initialized.
  // Always non-negative even if the underlying timer is non-monotonic.
  double getSeconds() const
  {
#ifdef __unix__
    const TimeValue t     = now();
    const double    delta = 1e-9 * static_cast<double>(t.nanoseconds - m_start.nanoseconds)  //
                         + static_cast<double>(t.seconds - m_start.seconds);
    return delta >= 0 ? delta : 0.;
#else
    const int64_t delta = now().ticks_100ns - m_start.ticks_100ns;
    return delta >= 0 ? static_cast<double>(delta) * 1e-7 : 0.;
#endif
  }

  // Convenience functions returning total time in different units
  double getMilliseconds() const { return getSeconds() * 1e3; }
  double getMicroseconds() const { return getSeconds() * 1e6; }

private:
  struct TimeValue
  {
#ifdef __unix__
    // On Unix platforms, store the full 128-bit time struct; this gets us
    // nanosecond precision and still avoids overflow issues.
    int64_t seconds{};
    int64_t nanoseconds{};
#else
    // Store the start time in ticks as a 64-bit signed integer, in units of
    // 100 nanoseconds (as this is what Windows uses).
    // Since on Windows we measure time since boot, rollover is implausible.
    // On other platforms, this will only roll over about 29226 years after the
    // platform's epoch.
    int64_t ticks_100ns{};
#endif
  };

  TimeValue m_start{};
  // Returns the current TimeValue.
  TimeValue now() const;
};


// Logging the time spent while alive in a scope.
// Usage: at beginning of a function:
//   auto stimer = ScopedTimer("Time for doing X");
// Nesting timers is handled, but since the time is printed when it goes out of
// scope, printing anything else will break the output formatting.
class ScopedTimer
{
public:
  ScopedTimer(const std::string& str);
  ScopedTimer(const char* fmt, ...);
  void init_(const std::string& str);
  ~ScopedTimer();
  static std::string indent()
  {
    std::string result(static_cast<size_t>(s_nesting * 2), ' ');
    for(int i = 0; i < s_nesting * 2; i += 2)
      result[i] = '|';
    return result;
  }

private:
  PerformanceTimer                m_timer;
  bool                            m_manualIndent = false;
  static inline thread_local int  s_nesting      = 0;
  static inline thread_local bool s_openNewline  = false;
};

// Can be used to measure time in a scope i.e. SCOPED_TIMER("Doing something"); will print "Doing something" and the time spent in the scope
// or SCOPED_TIMER(__FUNCTION__); will print the function name
#define SCOPED_TIMER(name) auto scopedTimer##__LINE__ = nvutils::ScopedTimer(name)

}  // namespace nvutils
