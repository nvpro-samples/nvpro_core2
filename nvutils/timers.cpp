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


#include <cstdarg>
#include <cassert>

#include "logger.hpp"
#include "timers.hpp"

// PerformanceTimer platform headers
#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <Windows.h>
#include <realtimeapiset.h>
#elif defined(__unix__)
#include <time.h>
#else
#include <chrono>
#endif

namespace nvutils {

//-------------------------------------------------------------------------------------------------
// PerformanceTimer

PerformanceTimer::TimeValue PerformanceTimer::now() const
{
#if defined(_WIN32)  // Windows implementation

  // On Windows, we use QueryUnbiasedInterruptTimePrecise, which
  // has good accuracy and ignores suspensions.
  // This is inspired by Calder White's article,
  // https://www.rippling.com/blog/rust-suspend-time .
  ULONGLONG uptime = 0;
  QueryUnbiasedInterruptTimePrecise(&uptime);
  // QueryUnbiasedInterruptTimePrecise returns values in 100ns intervals,
  // so we can return the value directly.
  return {.ticks_100ns = static_cast<int64_t>(uptime)};

#elif defined(__unix__)

  // On most Unix systems, we query CLOCK_MONOTONIC. We could do
  // CLOCK_MONOTONIC_RAW, but falling out-of-sync with real-world time is
  // probably worse than occasionally jumping backwards if the system's
  // oscillator is flawed.
  // On Linux, CLOCK_MONOTONIC does not include suspend time; see
  // https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/commit/?id=810fb07a9b504ac22b95899cf8b39d25a5f3e5c5 .
  // On Apple platforms, CLOCK_MONOTONIC includes suspend time
  // (according to https://www.manpagez.com/man/3/clock_gettime/), so
  // we use CLOCK_UPTIME_RAW instead.
#ifdef __APPLE__
  constexpr clockid_t clockID = CLOCK_UPTIME_RAW;
#else
  constexpr clockid_t clockID = CLOCK_MONOTONIC;
#endif
  static_assert(sizeof(time_t) >= 4);  // Make sure we aren't setting up for a Year 2038 bug
  timespec tv{};
  clock_gettime(clockID, &tv);
  return {.seconds = static_cast<int64_t>(tv.tv_sec), .nanoseconds = static_cast<int64_t>(tv.tv_nsec)};

#else  // Fallback implementation

  int64_t PerformanceTimer::now() const
  {
    const std::chrono::steady_clock::duration time = std::chrono::steady_clock::now().time_since_epoch();
    using ns100                                    = std::chrono::duration<int64_t, std::ratio<1i64, 10'000'000i64>>;
    return {.ticks_100ns = std::chrono::duration_cast<ns100>(time).count()};
  }

#endif
}

//-------------------------------------------------------------------------------------------------
// ScopedTimer

ScopedTimer::ScopedTimer(const char* fmt, ...)
{
  std::string str(256, '\0');  // initial guess. ideally the first try fits
  va_list     args1, args2;
  va_start(args1, fmt);
  va_copy(args2, args1);  // make a backup as vsnprintf may consume args1
  int rc = vsnprintf(str.data(), str.size(), fmt, args1);
  if(rc >= 0 && static_cast<size_t>(rc + 1) > str.size())
  {
    str.resize(rc + 1);  // include storage for '\0'
    rc = vsnprintf(str.data(), str.size(), fmt, args2);
  }
  va_end(args1);
  assert(rc >= 0 && "vsnprintf error");
  str.resize(rc >= 0 ? static_cast<size_t>(rc) : 0);
  init_(str);
}

ScopedTimer::ScopedTimer(const std::string& str)
{
  init_(str);
}

void ScopedTimer::init_(const std::string& str)
{
  // If nesting timers, break the newline of the previous one
  if(s_openNewline)
  {
    assert(s_nesting > 0);
    LOGI("\n");
  }

  m_manualIndent = !str.empty() && (str[0] == ' ' || str[0] == '-' || str[0] == '|');

  // Add indentation automatically if not already in str.
  if(s_nesting > 0 && !m_manualIndent)
  {
    LOGI("%s", indent().c_str());
  }

  LOGI("%s", str.c_str());
  s_openNewline = str.empty() || str[str.size() - 1] != '\n';
  ++s_nesting;
}

ScopedTimer::~ScopedTimer()
{
  --s_nesting;
  // If nesting timers and this is the second destructor in a row, indent and
  // print "Total" as it won't be on the same line.
  if(!s_openNewline && !m_manualIndent)
  {
    LOGI("%s|", indent().c_str());
  }
  else
  {
    LOGI(" ");
  }
  LOGI("-> %.3f ms\n", m_timer.getMilliseconds());
  s_openNewline = false;
}

}  // namespace nvutils
