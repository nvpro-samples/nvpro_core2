/*
* Copyright (c) 2014-2025, NVIDIA CORPORATION.  All rights reserved.
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
* SPDX-FileCopyrightText: Copyright (c) 2014-2025, NVIDIA CORPORATION.
* SPDX-License-Identifier: Apache-2.0
*/

#pragma once

#include <algorithm>
#include <cassert>
#include <limits>
#include <functional>
#include <memory>
#include <string>
#include <vector>
#include <mutex>
#include <list>
#include <array>

#include "timers.hpp"

namespace nvutils {


class ProfilerManager;

// The ProfilerTimeline allows to measure timed sections
// along a single timeline.
// There are two kinds of timed section operations:
// - per-frame operations start with the "frame" prefix.
//   Any frame section must be triggered within a `frameBegin`/`frameEnd` pairing.
// - single shot operations start with the "async" prefix.
//   They are thread-safe and can be called at any time.
//   Timer results using the same timer name are then overwritten.

class ProfilerTimeline
{
public:
  // on every `frameEnd` we query past frame timers
  // 0..FRAME_DELAY-1 must fit in FrameSectionID::subFrame
  static constexpr uint32_t MAX_FRAME_DELAY = 4;

  // Maximum number of frames when using a limited window for averaging.
  // It is possible to average all values as well.
  static constexpr uint32_t MAX_LAST_FRAMES = 128;  //JEM 32;

  struct CreateInfo
  {
    // for statistics and debugging
    std::string name;

    // if we detect a change in timers (api/name change we trigger a reset after that amount of frames)
    uint32_t frameConfigDelay = 8;

    // default internal array sizes for timers.
    // they grow automatically
    uint32_t defaultTimers = 128;

    // on every `frameEnd` we query past frame timers with this delay
    uint32_t frameDelay = MAX_FRAME_DELAY;

    // for per-frame timers
    // 0 means we average infinitely
    // N <= MAX_LAST_FRAMES means we average the last N frames
    uint32_t frameAveragingCount = MAX_LAST_FRAMES;
  };

  struct FrameSectionID
  {
    uint32_t id : 28;
    uint32_t subFrame : 4;
  };

  struct AsyncSectionID
  {
    uint32_t id;
  };

  // GPU times for a FrameSections are queried at "frameEnd" with the use of this function.
  // It returns true if the queried result was available, and writes the microseconds into gpuTime.
  typedef std::function<bool(FrameSectionID, double& gpuTime)> gpuFrameTimeProvider_fn;
  // GPU times for AsyncSectionID are queried at "
  typedef std::function<bool(AsyncSectionID, double& gpuTime)> gpuAsyncTimeProvider_fn;

  // This class can hold timer sections from different APIs (Vk, GL, CUDA...). We use the
  // below struct to serve as API agnostic interface.
  struct GpuTimeProvider
  {
    std::string             apiName;
    gpuFrameTimeProvider_fn frameFunction;
    gpuAsyncTimeProvider_fn asyncFunction;

    // Utility functions that help converting the FrameSectionID or AsyncSectionID to a linear array index.
    // Useful for GPU timer classes to manage resources as big arrays, or pools of arrays.

    // always 2 consecutive indices per timer, one for begin, one for end
    // frame timers use aup to MAX_FRAME_DELAY queries in-flight, hence the multiplication.
    static inline uint32_t getTimerBaseIdx(FrameSectionID slot)
    {
      return ((slot.id * MAX_FRAME_DELAY) + slot.subFrame) * 2;
    }

    // always 2 consecutive indices per timer, one for begin, one for end
    static inline uint32_t getTimerBaseIdx(AsyncSectionID slot) { return (slot.id * 2); }
  };

  //////////////////////////////////////////////////////////////////////////
  // per-frame timer operations
  // NOT thread-safe

  // move to next frame on this queue (closes previous frame and start new one)
  inline void frameAdvance()
  {
    if(m_inFrame)
    {
      frameEnd();
    }
    frameBegin();
  }

  // begin a timed per-frame section
  // must be called within begin/endFrame
  // The `gpuTimeProvider` pointer is copied and must be kept alive throughout for as long as this class is in use
  FrameSectionID frameBeginSection(const std::string& name, GpuTimeProvider* gpuTimeProvider = nullptr);
  // end a timed per-frame section
  void frameEndSection(FrameSectionID sec);

  // GPU timer implementations may want to use this function to reset the cpu time to exclude internal setup overhead
  void frameResetCpuBegin(FrameSectionID sec);

  // When a section is used within a loop (same nesting level), and the the same arguments for name and api are
  // passed, we normally average the results of those sections together when printing the stats or using the
  // getAveraged functions below.
  // Calling the splitter (outside of a section) means we insert a split point that the averaging will not
  // pass.
  // ```
  // // by default accumulates all "blubb" timers in the loop on the same nesting level
  // for (..) {
  //   auto timerSection = ProfilerTimeline.frameSection("blubb");
  // }
  //
  // // When timers with the same name and nesting level should be treated
  // // separately, then we need to inject the split.
  //
  // {
  //   auto timerSection = ProfilerTimeline.frameSection("Pass A");
  //   {
  //    auto timerSection = ProfilerTimeline.frameSection("blubb");
  //   }
  // }
  // // we want to differentiate results of "blubb" from "Pass A" and "Pass B"
  // ProfilerTimeline.frameAccumulationSplit();
  // {
  //   auto timerSection = ProfilerTimeline.frameSection("Pass B");
  //   {
  //     auto timerSection = ProfilerTimeline.frameSection("blubb");
  //   }
  // }
  //
  // ```
  void frameAccumulationSplit();

  //////////////////////////////////////////////////////////////////////////
  // async timer operations
  // thread-safe

  // begin an async timed section
  // The `gpuTimeProvider` pointer is copied and must be kept alive throughout for as long as this class is in use
  AsyncSectionID asyncBeginSection(const std::string& name, GpuTimeProvider* gpuTimeProvider = nullptr);
  // end an async timed timed section
  void asyncEndSection(AsyncSectionID sec);

  // GPU profilers may want to reset the cpu time to exclude internal setup overhead
  void asyncResetCpuBegin(AsyncSectionID sec);
  // can release timer names you never want to use again
  void asyncRemoveTimer(const std::string& name);

  //////////////////////////////////////////////////////////////////////////
  // getters

  struct TimerStats
  {
    // time in microseconds
    double   last        = 0;
    double   average     = 0;
    double   absMinValue = 0;
    double   absMaxValue = 0;
    uint32_t index       = 0;

    std::array<double, MAX_LAST_FRAMES> times = {};
  };

  struct TimerInfo
  {
    // number of averaged values
    // 0 means timer was unavailable
    uint32_t numAveraged = 0;

    // accumulation happens for example in loops:
    //   for (..) { auto scopeTimer = timeSection("blah"); ... }
    // then the reported values are the accumulated sum of all those timers.
    bool accumulated = false;
    bool async       = false;

    // nesting level for frame sections
    uint32_t level = 0;

    TimerStats cpu;
    TimerStats gpu;
  };

  // to allow thread-safe querying of results, all results
  // can be queried in bulk and are handed out as a copy
  // into this struct.
  class Snapshot
  {
  public:
    // name of the ProfilerTimeline from creation time
    std::string name;
    // ProfilerTimeline ID
    size_t id;

    // all arrays match in length

    // results for a given timer
    std::vector<TimerInfo> timerInfos;
    // name of the timer
    std::vector<std::string> timerNames;
    // name of the GPU api the timer used
    std::vector<std::string> timerApiNames;

    // If `full == true` appends all properties of a `TimerInfo`,
    // otherwise only the `level` and `averages` for GPU and CPU are added.
    void appendToString(std::string& stats, bool full) const;
  };

  // getters are thread-safe

  const std::string& getName() const { return m_info.name; }
  ProfilerManager*   getProfiler() const { return m_profiler; }

  void getAsyncSnapshot(Snapshot& snapShot) const;
  bool getAsyncTimerInfo(const std::string& name, TimerInfo& timerInfo, std::string& apiName) const;

  void getFrameSnapshot(Snapshot& snapShot) const;
  bool getFrameTimerInfo(const std::string& name, TimerInfo& info, std::string& apiName) const;

  //////////////////////////////////////////////////////////////////////////
  // configuration changes
  // thread-safe

  // clears all past frame timer results,
  // clears all async timers (including those in-flight)
  void clear();

  // resets recurring sections.
  // in case averaging should be reset after a few frames (warm-up cache, hide early heavier frames after
  // configuration changes)
  // implicitly resets are triggered if the frame's configuration of timer section changes compared to
  // previous frame.
  // delay == 0 maps to `CreateInfo::frameConfigDelay`
  void resetFrameSections(uint32_t delay = 0);

  // 0 means we average all values
  // otherwise num <= MAX_LAST_FRAMES to average in a cyclic window
  void setFrameAveragingCount(uint32_t num);

  //////////////////////////////////////////////////////////////////////////

  // utility class to call begin/end within local scope
  class FrameSection
  {
  public:
    FrameSection(ProfilerTimeline& ProfilerTimeline, FrameSectionID id)
        : m_ProfilerTimeline(ProfilerTimeline)
        , m_id(id) {};
    ~FrameSection() { m_ProfilerTimeline.frameEndSection(m_id); }

  private:
    ProfilerTimeline& m_ProfilerTimeline;
    FrameSectionID    m_id;
  };

  // must be called within frameBegin/frameEnd
  // not thread-safe
  FrameSection frameSection(const std::string& name) { return FrameSection(*this, frameBeginSection(name, nullptr)); }

  // utility class to call begin/end within local scope
  class AsyncSection
  {
  public:
    AsyncSection(ProfilerTimeline& ProfilerTimeline, AsyncSectionID id)
        : m_ProfilerTimeline(ProfilerTimeline)
        , m_id(id) {};
    ~AsyncSection() { m_ProfilerTimeline.asyncEndSection(m_id); }

  private:
    ProfilerTimeline& m_ProfilerTimeline;
    AsyncSectionID    m_id;
  };

  // thread-safe
  AsyncSection asyncSection(const std::string& name) { return AsyncSection(*this, asyncBeginSection(name, nullptr)); }

  //////////////////////////////////////////////////////////////////////////
  // internals
  ProfilerTimeline() = default;

protected:
  friend class ProfilerManager;

  ProfilerTimeline(ProfilerManager* profiler, const ProfilerTimeline::CreateInfo& createInfo)
  {
    assert(profiler);

    m_info     = createInfo;
    m_profiler = profiler;

    m_frame.averagingCount     = createInfo.frameAveragingCount;
    m_frame.averagingCountLast = createInfo.frameAveragingCount;

    grow(m_frame.sections, createInfo.defaultTimers, createInfo.frameAveragingCount);
    grow(m_async.sections, createInfo.defaultTimers, 0);

    frameBegin();
  }

  // start a frame on this queue
  void frameBegin();
  // end a frame on this queue
  void frameEnd();

  FrameSectionID frameGetSectionID();

  bool frameGetTimerInfo(uint32_t i, TimerInfo& info);
  void frameInternalSnapshot();

  bool asyncGetTimerInfo(uint32_t i, TimerInfo& info) const;

  static constexpr uint32_t LEVEL_SINGLESHOT = ~0;

  struct TimeValues
  {
    double valueLast   = 0;
    double valueTotal  = 0;
    double absMinValue = std::numeric_limits<double>::max();
    double absMaxValue = 0;

    uint32_t cycleIndex = 0;
    uint32_t cycleCount = MAX_LAST_FRAMES;
    uint32_t validCount = 0;

    std::array<double, MAX_LAST_FRAMES> times = {};

    // use non-zero for averaging over a window
    TimeValues(uint32_t averagedFrameCount_ = MAX_LAST_FRAMES) { init(averagedFrameCount_); }

    void init(uint32_t averagedFrameCount_)
    {
      cycleCount = std::min(averagedFrameCount_, MAX_LAST_FRAMES);
      reset();
    }

    void reset();

    void add(double time)
    {
      absMinValue = std::min(time, absMinValue);
      absMaxValue = std::max(time, absMaxValue);
      valueLast   = time;

      if(cycleCount)
      {
        // Averaging is performed over a window.
        // minus does remove the old value
        valueTotal += time - times[(MAX_LAST_FRAMES + cycleIndex - cycleCount) % MAX_LAST_FRAMES];

        validCount = std::min(validCount + 1, cycleCount);
      }
      else
      {
        // Averaging is done over all frames
        valueTotal += time;
        validCount++;
      }

      // store cycle so we can later remove it
      times[cycleIndex] = time;

      // advance cycle
      cycleIndex = (cycleIndex + 1) % MAX_LAST_FRAMES;
    }

    double getAveraged()
    {
      if(validCount)
      {
        return valueTotal / double(validCount);
      }
      else
      {
        return 0;
      }
    }
  };

  struct SectionData
  {
    std::string      name            = {};
    GpuTimeProvider* gpuTimeProvider = nullptr;

    uint32_t level    = 0;
    uint32_t subFrame = 0;

    std::array<double, MAX_FRAME_DELAY> cpuTimes = {};  // In microseconds
    std::array<double, MAX_FRAME_DELAY> gpuTimes = {};  // In microseconds

    // number of times summed since last reset
    uint32_t numTimes = 0;

    TimeValues gpuTime;
    TimeValues cpuTime;

    // splitter is used to prevent accumulated case below
    // when same depth level is used
    // {section("BLAH"); ... }
    // splitter
    // {section("BLAH"); ...}
    // now the result of "BLAH" is not accumulated

    bool splitter = false;

    // if the same timer name is used within a loop (same
    // depth level), e.g.:
    //
    // for () { section("BLAH"); ... }
    //
    // we accumulate the timing values of all of them

    bool accumulated = false;
  };

  struct FrameData
  {
    uint32_t averagingCount     = MAX_LAST_FRAMES;
    uint32_t averagingCountLast = MAX_LAST_FRAMES;

    // non zero value is decremented once per frame and
    // causes timers to be reset each reset frame.
    uint32_t resetDelay = 0;
    // frame count must be strictly monotonic
    uint32_t count = 0;
    // count of last frame a reset occured
    uint32_t countLastReset = 0;

    bool     hasSplitter = false;
    uint32_t level       = 0;

    uint32_t sectionsCount     = 0;
    uint32_t sectionsCountLast = 0;

    double     cpuCurrentTime = 0;  // In microseconds
    TimeValues cpuTime;
    TimeValues gpuTime;

    std::vector<SectionData> sections;
  };

  struct AsyncData
  {
    uint32_t                 sectionsCount = 0;
    std::vector<SectionData> sections;
  };

  void grow(std::vector<SectionData>& sections, size_t newSize, uint32_t averagingCount);

  bool             m_inFrame = false;
  ProfilerManager* m_profiler{};

  ProfilerTimeline::CreateInfo m_info;

  FrameData          m_frame;
  Snapshot           m_frameSnapshot;
  mutable std::mutex m_frameSnapshotMutex;


  AsyncData          m_async;
  mutable std::mutex m_asyncMutex;
};

class ProfilerManager
{
public:
  //////////////////////////////////////////////////////////////////////////

  ProfilerManager() = default;
  ~ProfilerManager();


  // all functions are thread-safe

  ProfilerTimeline* createTimeline(const ProfilerTimeline::CreateInfo& createInfo);
  void              destroyTimeline(ProfilerTimeline* timeline);

  inline double getMicroseconds() const { return m_timer.getMicroseconds(); }

  // calls Profiler::Timeline::setFrameAveragingCount for all timelines
  void setFrameAveragingCount(uint32_t num);
  // calls Profiler::Timeline::resetFrameSections for all timelines
  void resetFrameSections(uint32_t delayInFrames);

  // pretty print current timers
  // If `full == true` appends all properties of a `ProfilerTimeline::TimerInfo`,
  // otherwise only the `level` and `averages` for GPU and CPU are added.
  void appendPrint(std::string& statsFrames, std::string& statsAsyncs, bool full = false) const;

  // sets vectors to contain all snapshots
  void getSnapshots(std::vector<ProfilerTimeline::Snapshot>& frameSnapshots,
                    std::vector<ProfilerTimeline::Snapshot>& asyncSnapshots) const;

protected:
  std::list<std::unique_ptr<ProfilerTimeline>> m_timelines;
  mutable std::mutex                           m_mutex;
  PerformanceTimer                             m_timer;
};
}  // namespace nvutils
