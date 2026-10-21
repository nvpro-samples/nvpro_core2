/*
* Copyright (c) 2014-2026, NVIDIA CORPORATION.  All rights reserved.
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
* SPDX-FileCopyrightText: Copyright (c) 2014-2026, NVIDIA CORPORATION.  All rights reserved.
* SPDX-License-Identifier: Apache-2.0
*/

#include <algorithm>
#include <cassert>

#include <fmt/format.h>

#include "profiler.hpp"

namespace nvutils {

void ProfilerTimeline::Snapshot::appendToString(std::string& stats, bool full = false) const
{
  const uint32_t maxLevels = 8;
  const uint32_t maxLevel  = maxLevels - 1;
  const char*    spaces    = "        ";

  uint32_t foundMaxLevel = 0;
  for(size_t i = 0; i < timerInfos.size(); i++)
  {
    const TimerInfo& info = timerInfos[i];
    foundMaxLevel         = std::max(foundMaxLevel, info.level);
  }
  foundMaxLevel = std::min(foundMaxLevel, maxLevel);

  for(size_t i = 0; i < timerInfos.size(); i++)
  {
    const TimerInfo& info  = timerInfos[i];
    uint32_t         level = std::min(info.level, maxLevel);


    const char* apiName        = !timerApiNames[i].empty() ? timerApiNames[i].c_str() : "N/A";
    const char* timerName      = !timerNames[i].empty() ? timerNames[i].c_str() : "N/A";
    const char* indentSpaces   = &spaces[maxLevels - level];
    const char* indentSpacesOp = &spaces[maxLevels - (foundMaxLevel - level)];

    if(full)
    {
      stats += fmt::format("Timeline \"{}\"; level {}; Timer \"{}\"; GPU; avg {}; min {}; max {}; last {}; CPU; avg {}; min {}; max {}; last {}; samples {};\n",
                           name, int32_t(info.async ? -1 : info.level), timerNames[i], (uint32_t)(info.gpu.average),
                           (uint32_t)(info.gpu.absMinValue), (uint32_t)(info.gpu.absMaxValue),
                           (uint32_t)(info.gpu.last), (uint32_t)(info.cpu.average), (uint32_t)(info.cpu.absMinValue),
                           (uint32_t)(info.cpu.absMaxValue), (uint32_t)(info.cpu.last), info.numAveraged);
    }
    else
    {
      stats += fmt::format("{:12}; {:3};{}{:16}{}; GPU; avg {:6}; CPU; avg {:6}; microseconds;\n", name,
                           int32_t(info.async ? -1 : info.level), indentSpaces, timerNames[i], indentSpacesOp,
                           (uint32_t)(info.gpu.average), (uint32_t)(info.cpu.average));
    }
  }
}

//////////////////////////////////////////////////////////////////////////

void ProfilerTimeline::TimeValues::reset()
{
  valueTotal  = 0;
  valueLast   = 0;
  absMinValue = std::numeric_limits<double>::max();
  absMaxValue = 0;
  cycleIndex  = 0;
  validCount  = 0;
  times       = {};
}

//////////////////////////////////////////////////////////////////////////

void ProfilerTimeline::setFrameAveragingCount(uint32_t num)
{
  assert(num <= MAX_LAST_FRAMES);
  m_frame.averagingCount = num;
}

void ProfilerTimeline::clear()
{
  {
    std::lock_guard guard(m_asyncMutex);
    m_async.sections.clear();
    m_async.sectionsCount = 0;
  }

  {
    std::lock_guard guard(m_frameSnapshotMutex);
    m_frameSnapshot = {};
  }
}

void ProfilerTimeline::resetFrameSections(uint32_t delay)
{
  m_frame.resetDelay = delay ? delay : m_info.frameConfigDelay;
}

//////////////////////////////////////////////////////////////////////////

void ProfilerTimeline::frameAccumulationSplit()
{
  assert(m_inFrame);

  FrameSectionID sectionID = frameGetSectionID();

  m_frame.sections[sectionID.id].level    = m_frame.level;
  m_frame.sections[sectionID.id].splitter = true;

  m_frame.hasSplitter = true;
}

void ProfilerTimeline::frameBegin()
{
  m_frame.hasSplitter    = false;
  m_frame.level          = 1;
  m_frame.sectionsCount  = 0;
  // update sections keep their slots, only the nesting restarts
  m_update.level         = 1;
  m_update.openParent    = ~0u;
  m_frame.cpuCurrentTime = -m_profiler->getMicroseconds();
  m_inFrame              = true;
}

void ProfilerTimeline::frameEnd()
{
  assert(m_frame.level == 1);
  assert(m_inFrame);

  m_frame.cpuCurrentTime += m_profiler->getMicroseconds();

  if(m_frame.sectionsCount && m_frame.sectionsCount != m_frame.sectionsCountLast)
  {
    m_frame.sectionsCountLast = m_frame.sectionsCount;
    m_frame.resetDelay        = m_info.frameConfigDelay;
  }

  if(m_frame.resetDelay)
  {
    m_frame.resetDelay--;
    for(size_t i = 0; i < m_frame.sections.size(); i++)
    {
      SectionData& section = m_frame.sections[i];
      section.numTimes     = 0;
      section.cpuTime.reset();
      section.gpuTime.reset();
    }
    m_frame.cpuTime.reset();
    m_frame.gpuTime.reset();

    m_frame.countLastReset = m_frame.count;
  }

  if(m_frame.averagingCount != m_frame.averagingCountLast)
  {
    for(size_t i = 0; i < m_frame.sections.size(); i++)
    {
      m_frame.sections[i].cpuTime.init(m_frame.averagingCount);
      m_frame.sections[i].gpuTime.init(m_frame.averagingCount);
    }
    m_frame.cpuTime.init(m_frame.averagingCount);
    m_frame.gpuTime.init(m_frame.averagingCount);

    m_frame.averagingCountLast = m_frame.averagingCount;
  }

  // we have enough valid frames since last reset
  if((m_frame.count - m_frame.countLastReset) > m_info.frameDelay)
  {
    double   gpuTime      = 0;
    uint32_t gpuLastLevel = ~0;

    for(uint32_t i = 0; i < m_frame.sectionsCount; i++)
    {
      SectionData& section = m_frame.sections[i];

      if(section.splitter)
        continue;

      uint32_t       queryFrame = (m_frame.count + 1) % m_info.frameDelay;
      FrameSectionID sec;
      sec.id       = i;
      sec.subFrame = queryFrame;

      bool available = !section.gpuTimeProvider || section.gpuTimeProvider->frameFunction(sec, section.gpuTimes[queryFrame]);

      // reset gpu last level, if we start a new section on a lower level
      if(gpuLastLevel != ~0 && section.level < gpuLastLevel)
      {
        gpuLastLevel = ~0;
      }

      if(available)
      {
        section.cpuTime.add(section.cpuTimes[queryFrame]);
        section.gpuTime.add(section.gpuTimes[queryFrame]);
        section.numTimes++;

        if(gpuLastLevel == ~0 || gpuLastLevel == section.level)
        {
          gpuTime += section.gpuTimes[queryFrame];
          gpuLastLevel = section.level;
        }
      }
    }

    m_frame.gpuTime.add(gpuTime);
    m_frame.cpuTime.add(m_frame.cpuCurrentTime);
  }

  frameInternalSnapshot();

  // update sections use the same frameDelay as frame sections
  updateResolvePending();
  updateInternalSnapshot();

  m_frame.count++;
  m_inFrame = false;
}


void ProfilerTimeline::frameInternalSnapshot()
{
  std::lock_guard lock(m_frameSnapshotMutex);

  m_frameSnapshot.timerInfos.clear();
  m_frameSnapshot.timerNames.clear();
  m_frameSnapshot.timerApiNames.clear();
  m_frameSnapshot.name = m_info.name;
  m_frameSnapshot.id   = (size_t)this;

  TimerInfo info{};
  info.cpu.last        = m_frame.cpuTime.valueLast;
  info.cpu.average     = m_frame.cpuTime.getAveraged();
  info.cpu.absMaxValue = m_frame.cpuTime.absMaxValue;
  info.cpu.absMinValue = m_frame.cpuTime.absMinValue;
  info.cpu.times       = m_frame.cpuTime.times;
  info.cpu.index       = m_frame.cpuTime.cycleIndex;

  info.gpu.last        = m_frame.gpuTime.valueLast;
  info.gpu.average     = m_frame.gpuTime.getAveraged();
  info.gpu.absMaxValue = m_frame.gpuTime.absMaxValue;
  info.gpu.absMinValue = m_frame.gpuTime.absMinValue;
  info.gpu.times       = m_frame.gpuTime.times;
  info.gpu.index       = m_frame.gpuTime.cycleIndex;

  info.numAveraged = m_frame.cpuTime.validCount;

  if(m_frame.cpuTime.validCount)
  {
    m_frameSnapshot.timerInfos.push_back(info);
    m_frameSnapshot.timerNames.push_back("Frame");
    m_frameSnapshot.timerApiNames.push_back("GPU");
  }

  for(uint32_t i = 0; i < m_frame.sectionsCountLast; i++)
  {
    SectionData& section = m_frame.sections[i];

    section.accumulated = false;
  }

  for(uint32_t i = 0; i < m_frame.sectionsCountLast; i++)
  {
    SectionData& section = m_frame.sections[i];

    if(section.splitter)
      continue;

    if(frameGetTimerInfo(i, info))
    {
      m_frameSnapshot.timerInfos.push_back(info);
      m_frameSnapshot.timerNames.push_back(section.name);
      m_frameSnapshot.timerApiNames.push_back(section.gpuTimeProvider ? section.gpuTimeProvider->apiName : "");
    }
  }
}

ProfilerTimeline::FrameSectionID ProfilerTimeline::frameGetSectionID()
{
  uint32_t       numEntries = (uint32_t)m_frame.sections.size();
  FrameSectionID sec{};

  assert(m_inFrame);

  sec.id       = m_frame.sectionsCount++;
  sec.subFrame = m_frame.count % m_info.frameDelay;

  if(sec.id >= m_frame.sections.size())
  {
    grow(m_frame.sections, m_frame.sections.size() * 2, m_frame.averagingCountLast);
  }

  return sec;
}

ProfilerTimeline::FrameSectionID ProfilerTimeline::frameBeginSection(const std::string& name, GpuTimeProvider* gpuTimeProvider)
{
  FrameSectionID sectionID = frameGetSectionID();
  SectionData&   section   = m_frame.sections[sectionID.id];
  uint32_t       level     = m_frame.level++;

  if(section.name != name || section.gpuTimeProvider != gpuTimeProvider || section.level != level)
  {
    section.name            = name;
    section.gpuTimeProvider = gpuTimeProvider;

    m_frame.resetDelay = m_info.frameConfigDelay;
  }

  section.subFrame        = sectionID.subFrame;
  section.level           = level;
  section.splitter        = false;
  section.gpuTimeProvider = gpuTimeProvider;

  section.cpuTimes[sectionID.subFrame] = -m_profiler->getMicroseconds();
  section.gpuTimes[sectionID.subFrame] = 0;

  return sectionID;
}

void ProfilerTimeline::frameEndSection(FrameSectionID sectionID)
{
  SectionData& section = m_frame.sections[sectionID.id];
  section.cpuTimes[sectionID.subFrame] += m_profiler->getMicroseconds();
  m_frame.level--;
}

void ProfilerTimeline::frameResetCpuBegin(FrameSectionID sectionID)
{
  SectionData& section                 = m_frame.sections[sectionID.id];
  section.cpuTimes[sectionID.subFrame] = -m_profiler->getMicroseconds();
}

ProfilerTimeline::AsyncSectionID ProfilerTimeline::asyncBeginSection(const std::string& name, GpuTimeProvider* gpuTimeProvider)
{
  std::lock_guard lock(m_asyncMutex);

  AsyncSectionID sectionID = {};
  bool           found     = false;

  // find empty slot or with same name
  for(uint32_t i = 0; i < m_async.sectionsCount; i++)
  {
    SectionData& section = m_async.sections[i];
    if(section.name.empty() || section.name == name)
    {
      sectionID.id = i;
      found        = true;
      break;
    }
  }

  if(!found)
  {
    sectionID.id = m_async.sectionsCount++;
    if(sectionID.id >= m_async.sections.size())
    {
      grow(m_async.sections, m_async.sections.size() * 2, 0);
    }
  }

  SectionData& section    = m_async.sections[sectionID.id];
  section.name            = name;
  section.gpuTimeProvider = gpuTimeProvider;

  section.subFrame        = 0;
  section.level           = LEVEL_SINGLESHOT;
  section.splitter        = false;
  section.gpuTimeProvider = gpuTimeProvider;

  section.cpuTimes[0] = -m_profiler->getMicroseconds();
  section.gpuTimes[0] = 0;

  return sectionID;
}

void ProfilerTimeline::asyncEndSection(ProfilerTimeline::AsyncSectionID sectionID)
{
  double endTime = m_profiler->getMicroseconds();

  {
    std::lock_guard lock(m_asyncMutex);
    if(sectionID.id < m_async.sectionsCount)
    {
      SectionData& section = m_async.sections[sectionID.id];
      section.cpuTimes[0] += endTime;
      section.numTimes = 1;
      section.emitCount++;
    }
  }
}

void ProfilerTimeline::asyncResetCpuBegin(AsyncSectionID sectionID)
{
  std::lock_guard lock(m_asyncMutex);
  if(sectionID.id < m_async.sectionsCount)
  {
    SectionData& section = m_async.sections[sectionID.id];
    section.cpuTimes[0]  = -m_profiler->getMicroseconds();
  }
}

void ProfilerTimeline::asyncClearSections()
{
  std::lock_guard lock(m_asyncMutex);

  for(uint32_t i = 0; i < m_async.sectionsCount; i++)
  {
    m_async.sections[i] = SectionData{};
  }
  m_async.sectionsCount = 0;
}

void ProfilerTimeline::asyncRemoveTimer(const std::string& name)
{
  std::lock_guard lock(m_asyncMutex);

  for(uint32_t i = 0; i < m_async.sectionsCount; i++)
  {
    SectionData& section = m_async.sections[i];
    if(section.name == name)
    {
      section.name               = {};
      section.cpuTime.validCount = 0;
      if(i == m_async.sectionsCount - 1)
      {
        m_async.sectionsCount--;
      }

      return;
    }
  }
}


void ProfilerTimeline::grow(std::vector<SectionData>& sections, size_t newSize, uint32_t averagingCount)
{
  size_t oldSize = sections.size();

  if(oldSize == newSize)
  {
    return;
  }

  sections.resize(newSize);

  for(size_t i = oldSize; i < newSize; i++)
  {
    sections[i].cpuTime.init(averagingCount);
    sections[i].gpuTime.init(averagingCount);
  }
}

bool ProfilerTimeline::frameGetTimerInfo(uint32_t i, TimerInfo& info)
{
  SectionData& section = m_frame.sections[i];

  if(!section.numTimes || section.accumulated)
  {
    return false;
  }
  info.accumulated     = false;
  info.async           = false;
  info.level           = section.level;
  info.cpu.last        = section.cpuTime.valueLast;
  info.gpu.last        = section.gpuTime.valueLast;
  info.gpu.average     = section.gpuTime.getAveraged();
  info.cpu.average     = section.cpuTime.getAveraged();
  info.cpu.absMinValue = section.cpuTime.absMinValue;
  info.cpu.absMaxValue = section.cpuTime.absMaxValue;
  info.gpu.absMinValue = section.gpuTime.absMinValue;
  info.gpu.absMaxValue = section.gpuTime.absMaxValue;
  info.cpu.times       = section.cpuTime.times;
  info.gpu.times       = section.gpuTime.times;
  info.cpu.index       = section.cpuTime.cycleIndex;
  info.gpu.index       = section.gpuTime.cycleIndex;
  bool found           = false;
  if(section.level != LEVEL_SINGLESHOT && m_frame.hasSplitter)
  {
    for(uint32_t n = i + 1; n < m_frame.sectionsCountLast; n++)
    {
      SectionData& otherSection = m_frame.sections[n];
      if(otherSection.name == section.name && otherSection.level == section.level
         && otherSection.gpuTimeProvider == section.gpuTimeProvider && !otherSection.accumulated)
      {
        found = true;
        info.cpu.last += otherSection.cpuTime.valueLast;
        info.gpu.last += otherSection.gpuTime.valueLast;
        info.gpu.average += otherSection.gpuTime.getAveraged();
        info.cpu.average += otherSection.cpuTime.getAveraged();
        info.cpu.absMinValue += otherSection.cpuTime.absMinValue;
        info.cpu.absMaxValue += otherSection.cpuTime.absMaxValue;
        info.gpu.absMinValue += otherSection.gpuTime.absMinValue;
        info.gpu.absMaxValue += otherSection.gpuTime.absMaxValue;
        otherSection.accumulated = true;
      }

      if(otherSection.splitter && otherSection.level <= section.level)
        break;
    }
  }
  info.accumulated = found;
  info.numAveraged = section.cpuTime.validCount;

  return true;
}

bool ProfilerTimeline::asyncGetTimerInfo(uint32_t i, TimerInfo& info) const
{
  const SectionData& section = m_async.sections[i];

  AsyncSectionID sectionID;
  sectionID.id = i;

  // query
  double cpuTime   = section.cpuTimes[0];
  double gpuTime   = 0;
  bool   available = (!section.gpuTimeProvider || section.gpuTimeProvider->asyncFunction(sectionID, gpuTime));

  if(available)
  {
    info.accumulated = false;
    info.async       = true;
    info.numAveraged = 1;
    info.level       = 0;
    info.emitCount   = section.emitCount;

    info.cpu.absMaxValue = cpuTime;
    info.cpu.absMaxValue = cpuTime;
    info.cpu.average     = cpuTime;
    info.cpu.last        = cpuTime;

    info.gpu.absMaxValue = gpuTime;
    info.gpu.absMaxValue = gpuTime;
    info.gpu.average     = gpuTime;
    info.gpu.last        = gpuTime;

    return true;
  }

  return false;
}

//////////////////////////////////////////////////////////////////////////
// update sections

static inline uint64_t updateMakeKey(const std::string& name, uint32_t level, uint32_t parentId)
{
  const uint64_t nameHash = std::hash<std::string>{}(name);
  // the same name under two parents makes two rows
  return nameHash ^ (uint64_t(level) * 0x9E3779B97F4A7C15ull) ^ (uint64_t(parentId + 1) * 0xC2B2AE3D27D4EB4Full);
}

uint32_t ProfilerTimeline::updateFindOrCreateSlot(const std::string& name, uint32_t level, uint32_t parentId)
{
  const uint64_t key = updateMakeKey(name, level, parentId);

  auto it = m_update.slotByKey.find(key);
  if(it != m_update.slotByKey.end())
  {
    return it->second;
  }

  const uint32_t slot = m_update.sectionsCount++;
  if(slot >= m_update.sections.size())
  {
    grow(m_update.sections, m_update.sections.size() * 2, 0);
  }

  SectionData& section = m_update.sections[slot];
  section              = SectionData{};
  section.name         = name;
  section.level        = level;
  section.parentId     = parentId;

  m_update.slotByKey[key] = slot;

  // roots in creation order, a child at the end of its parent's subtree
  if(parentId == ~0u)
  {
    m_update.order.push_back(slot);
  }
  else
  {
    const uint32_t parentLevel = m_update.sections[parentId].level;

    size_t pos = 0;
    while(pos < m_update.order.size() && m_update.order[pos] != parentId)
      pos++;
    pos++;
    while(pos < m_update.order.size() && m_update.sections[m_update.order[pos]].level > parentLevel)
      pos++;
    m_update.order.insert(m_update.order.begin() + pos, slot);
  }

  return slot;
}

uint32_t ProfilerTimeline::updateFindOrCreateEntrySlot(uint32_t slot, uint32_t entry)
{
  if(entry == 0)
  {
    return slot;
  }

  while(m_update.sections[slot].entrySlots.size() < entry)
  {
    const uint32_t entrySlot = m_update.sectionsCount++;
    if(entrySlot >= m_update.sections.size())
    {
      grow(m_update.sections, m_update.sections.size() * 2, 0);
    }

    SectionData& section = m_update.sections[entrySlot];
    section              = SectionData{};
    section.level        = m_update.sections[slot].level;
    section.ownerId      = slot;

    m_update.sections[slot].entrySlots.push_back(entrySlot);
  }

  return m_update.sections[slot].entrySlots[entry - 1];
}

ProfilerTimeline::UpdateSectionID ProfilerTimeline::updateBeginSection(const std::string& name, GpuTimeProvider* gpuTimeProvider)
{
  assert(m_inFrame);

  const uint32_t level    = m_update.level++;
  const uint32_t slot     = updateFindOrCreateSlot(name, level, m_update.openParent);
  const uint32_t subFrame = m_frame.count % m_info.frameDelay;

  // a section entered again in the same frame is measured in an extra slot, with its own GPU
  // queries. updateResolvePending() sums the entries
  uint32_t entry;
  {
    SectionData& section          = m_update.sections[slot];
    const bool   reentered        = (section.lastEnterFrame == m_frame.count);
    section.lastEnterFrame        = m_frame.count;
    section.gpuTimeProvider       = gpuTimeProvider;
    section.subFrame              = subFrame;
    section.recordFrame[subFrame] = m_frame.count;
    section.entryCounts[subFrame] = reentered ? section.entryCounts[subFrame] + 1 : 1;
    entry                         = section.entryCounts[subFrame] - 1;
  }

  const uint32_t measuredSlot = updateFindOrCreateEntrySlot(slot, entry);

  SectionData& measured       = m_update.sections[measuredSlot];
  measured.gpuTimeProvider    = gpuTimeProvider;
  measured.cpuTimes[subFrame] = -m_profiler->getMicroseconds();
  measured.gpuTimes[subFrame] = 0;

  // children are keyed under the section's own slot
  m_update.openParent = slot;

  UpdateSectionID sectionID{};
  sectionID.id       = measuredSlot;
  sectionID.subFrame = subFrame;
  return sectionID;
}

void ProfilerTimeline::updateEndSection(UpdateSectionID sectionID)
{
  SectionData& measured = m_update.sections[sectionID.id];
  measured.cpuTimes[sectionID.subFrame] += m_profiler->getMicroseconds();

  // the section's own slot
  const uint32_t slot    = (measured.ownerId != ~0u) ? measured.ownerId : sectionID.id;
  SectionData&   section = m_update.sections[slot];

  section.pending[sectionID.subFrame] = true;

  m_update.level--;
  m_update.openParent = section.parentId;
}

void ProfilerTimeline::updateResetCpuBegin(UpdateSectionID sectionID)
{
  SectionData& section                 = m_update.sections[sectionID.id];
  section.cpuTimes[sectionID.subFrame] = -m_profiler->getMicroseconds();
}

void ProfilerTimeline::updateResolvePending()
{
  const uint32_t queryFrame = (m_frame.count + 1) % m_info.frameDelay;

  for(uint32_t slot : m_update.order)
  {
    SectionData& section = m_update.sections[slot];

    // a section that did not run keeps its last value
    if(!section.pending[queryFrame])
      continue;

    // sum the frame's entries
    double cpuTime   = 0;
    double gpuTime   = 0;
    bool   available = true;
    for(uint32_t entry = 0; available && entry < section.entryCounts[queryFrame]; entry++)
    {
      UpdateSectionID sec{};
      sec.id       = entry ? section.entrySlots[entry - 1] : slot;
      sec.subFrame = queryFrame;

      const SectionData& measured     = m_update.sections[sec.id];
      double             entryGpuTime = 0;
      // a provider without updateFunction gives no GPU time
      if(measured.gpuTimeProvider && measured.gpuTimeProvider->updateFunction)
      {
        available = measured.gpuTimeProvider->updateFunction(sec, entryGpuTime);
      }
      cpuTime += measured.cpuTimes[queryFrame];
      gpuTime += entryGpuTime;
    }
    if(!available)
    {
      // timestamps not available yet, keep it pending
      continue;
    }

    section.gpuTimes[queryFrame] = gpuTime;
    section.valueFrame           = section.recordFrame[queryFrame];
    section.cpuTime.add(cpuTime);
    section.gpuTime.add(gpuTime);
    section.numTimes++;
    section.emitCount++;
    section.lastEmitFrame       = m_frame.count;
    section.pending[queryFrame] = false;
  }
}

bool ProfilerTimeline::updateGetTimerInfo(uint32_t slot, TimerInfo& info, bool zeroed) const
{
  const SectionData& section = m_update.sections[slot];

  info             = TimerInfo{};
  info.accumulated = false;
  info.async       = false;
  info.level       = section.level;
  // 0 means no measurement yet
  info.numAveraged = section.emitCount ? 1 : 0;
  info.emitCount   = section.emitCount;

  // last value, not an average; min/max cover all measurements
  const double cpuValue = zeroed ? 0.0 : section.cpuTime.valueLast;
  const double gpuValue = zeroed ? 0.0 : section.gpuTime.valueLast;

  info.cpu.last        = cpuValue;
  info.cpu.average     = cpuValue;
  info.cpu.absMinValue = section.cpuTime.absMinValue;
  info.cpu.absMaxValue = section.cpuTime.absMaxValue;
  info.cpu.times       = section.cpuTime.times;
  info.cpu.index       = section.cpuTime.cycleIndex;

  info.gpu.last        = gpuValue;
  info.gpu.average     = gpuValue;
  info.gpu.absMinValue = section.gpuTime.absMinValue;
  info.gpu.absMaxValue = section.gpuTime.absMaxValue;
  info.gpu.times       = section.gpuTime.times;
  info.gpu.index       = section.gpuTime.cycleIndex;

  return true;
}

void ProfilerTimeline::updateInternalSnapshot()
{
  std::lock_guard lock(m_updateSnapshotMutex);

  m_updateSnapshot.timerInfos.clear();
  m_updateSnapshot.timerNames.clear();
  m_updateSnapshot.timerApiNames.clear();
  m_updateSnapshot.name = m_info.name;
  m_updateSnapshot.id   = (size_t)this;

  if(m_update.order.empty())
    return;

  // artificial parent, like the "Async" one
  m_updateSnapshot.timerInfos.push_back({});
  m_updateSnapshot.timerNames.push_back("Updates");
  m_updateSnapshot.timerApiNames.push_back("GPU");

  // rows reporting 0:
  // - with updateResetWhenIdle, those that did not run this frame, have nothing in flight and
  //   had no measurement resolve this frame
  // - otherwise, those not measured in the latest frame where a root ran, so the tree shows one
  //   pass and a parent is never smaller than its children
  uint32_t referenceFrame = 0;
  bool     haveReference  = false;
  if(!m_info.updateResetWhenIdle)
  {
    for(uint32_t slot : m_update.order)
    {
      const SectionData& section = m_update.sections[slot];
      if(section.parentId != ~0u || section.emitCount == 0)
        continue;
      if(!haveReference || section.valueFrame > referenceFrame)
      {
        referenceFrame = section.valueFrame;
        haveReference  = true;
      }
    }
  }

  for(uint32_t slot : m_update.order)
  {
    const SectionData& section = m_update.sections[slot];

    bool zeroed;
    if(m_info.updateResetWhenIdle)
    {
      bool pendingAny = false;
      for(uint32_t i = 0; i < m_info.frameDelay; i++)
        pendingAny = pendingAny || section.pending[i];

      zeroed = (section.lastEnterFrame != m_frame.count) && !pendingAny && (section.lastEmitFrame != m_frame.count);
    }
    else
    {
      zeroed = (section.emitCount == 0) || !haveReference || (section.valueFrame != referenceFrame);
    }

    TimerInfo timerInfo;
    updateGetTimerInfo(slot, timerInfo, zeroed);

    m_updateSnapshot.timerInfos.push_back(timerInfo);
    m_updateSnapshot.timerNames.push_back(section.name);
    m_updateSnapshot.timerApiNames.push_back(section.gpuTimeProvider ? section.gpuTimeProvider->apiName : "");
  }
}

void ProfilerTimeline::updateClearSections()
{
  m_update.sections.clear();
  m_update.slotByKey.clear();
  m_update.order.clear();
  m_update.sectionsCount = 0;
  m_update.level         = 1;
  m_update.openParent    = ~0u;

  grow(m_update.sections, m_info.defaultTimers, 0);

  std::lock_guard lock(m_updateSnapshotMutex);
  m_updateSnapshot.timerInfos.clear();
  m_updateSnapshot.timerNames.clear();
  m_updateSnapshot.timerApiNames.clear();
}

void ProfilerTimeline::getUpdateSnapshot(Snapshot& snapShot) const
{
  std::lock_guard lock(m_updateSnapshotMutex);

  snapShot = m_updateSnapshot;
}

void ProfilerTimeline::updateRemoveSlotRecursive(uint32_t slot)
{
  // children first, so an erase never invalidates a parent lookup
  for(uint32_t i = 0; i < m_update.sectionsCount; i++)
  {
    if(!m_update.sections[i].name.empty() && m_update.sections[i].parentId == slot)
    {
      updateRemoveSlotRecursive(i);
    }
  }

  SectionData& section = m_update.sections[slot];

  // and its extra entry slots
  for(uint32_t entrySlot : section.entrySlots)
  {
    m_update.sections[entrySlot] = SectionData{};
  }

  m_update.slotByKey.erase(updateMakeKey(section.name, section.level, section.parentId));
  m_update.order.erase(std::remove(m_update.order.begin(), m_update.order.end(), slot), m_update.order.end());

  // emptied but not reused, live UpdateSectionIDs may still refer to it
  section = SectionData{};
}

void ProfilerTimeline::updateRemoveTimer(const std::string& name)
{
  for(uint32_t i = 0; i < m_update.sectionsCount; i++)
  {
    if(m_update.sections[i].name == name)
    {
      updateRemoveSlotRecursive(i);
    }
  }
}

void ProfilerTimeline::getAsyncSnapshot(Snapshot& snapShot) const
{
  snapShot.name = m_info.name;
  snapShot.id   = (size_t)this;
  snapShot.timerInfos.clear();
  snapShot.timerNames.clear();
  snapShot.timerApiNames.clear();

  std::lock_guard lock(m_asyncMutex);

  //
  snapShot.timerInfos.push_back({});
  snapShot.timerNames.push_back("Async");
  snapShot.timerApiNames.push_back("GPU");

  // append all
  for(uint32_t i = 0; i < m_async.sectionsCount; i++)
  {
    const SectionData& section = m_async.sections[i];

    if(section.name.empty())
      continue;

    TimerInfo timerInfo;
    if(asyncGetTimerInfo(i, timerInfo))
    {
      timerInfo.level++;  // take the artificial Async parent into account
      snapShot.timerInfos.push_back(timerInfo);
      snapShot.timerNames.push_back(section.name);
      snapShot.timerApiNames.push_back(section.gpuTimeProvider ? section.gpuTimeProvider->apiName : "");
    }
  }

  // refit if nothing was added
  if(snapShot.timerInfos.size() == 1)
  {
    snapShot.timerInfos.clear();
    snapShot.timerNames.clear();
    snapShot.timerApiNames.clear();
  }
}

bool ProfilerTimeline::getAsyncTimerInfo(const std::string& name, TimerInfo& timerInfo, std::string& apiName) const
{
  std::lock_guard lock(m_asyncMutex);

  for(uint32_t i = 0; i < m_async.sectionsCount; i++)
  {
    const SectionData& section = m_async.sections[i];

    if(section.name != name)
      continue;

    if(section.gpuTimeProvider)
      apiName = section.gpuTimeProvider->apiName;

    return asyncGetTimerInfo(i, timerInfo);
  }

  return false;
}

void ProfilerTimeline::getFrameSnapshot(Snapshot& snapShot) const
{
  std::lock_guard lock(m_frameSnapshotMutex);

  snapShot = m_frameSnapshot;
}

bool ProfilerTimeline::getFrameTimerInfo(const std::string& name, TimerInfo& info, std::string& apiName) const
{
  std::lock_guard lock(m_frameSnapshotMutex);

  for(size_t i = 0; i < m_frameSnapshot.timerNames.size(); i++)
  {
    if(m_frameSnapshot.timerNames[i] == name)
    {
      apiName = m_frameSnapshot.timerApiNames[i];
      info    = m_frameSnapshot.timerInfos[i];
      return true;
    }
  }

  return false;
}

//////////////////////////////////////////////////////////////////////////

ProfilerManager::~ProfilerManager()
{
  assert(m_timelines.empty() && "forgot to destroy all timelines");
}

ProfilerTimeline* ProfilerManager::createTimeline(const ProfilerTimeline::CreateInfo& createInfo)
{
  std::lock_guard lock(m_mutex);

  return m_timelines.emplace_back(new ProfilerTimeline(this, createInfo)).get();
}

void ProfilerManager::destroyTimeline(ProfilerTimeline* timeline)
{
  std::lock_guard lock(m_mutex);

  for(auto it = m_timelines.begin(); it != m_timelines.end(); it++)
  {
    if(it->get() == timeline)
    {
      m_timelines.erase(it);
      return;
    }
  }

  assert(0 && "invalid timeline");
}

void ProfilerManager::setFrameAveragingCount(uint32_t num)
{
  std::lock_guard lock(m_mutex);
  for(auto& it : m_timelines)
  {
    it->setFrameAveragingCount(num);
  }
}

void ProfilerManager::resetFrameSections(uint32_t delayInFrames)
{
  std::lock_guard lock(m_mutex);
  for(auto& it : m_timelines)
  {
    it->resetFrameSections(delayInFrames);
  }
}

void ProfilerManager::appendPrint(std::string& statsFrames, std::string& statsAsyncs, bool full) const
{
  std::vector<ProfilerTimeline::Snapshot> frameSnapshots;
  std::vector<ProfilerTimeline::Snapshot> asyncSnapshots;

  getSnapshots(frameSnapshots, asyncSnapshots);

  for(auto& it : frameSnapshots)
  {
    it.appendToString(statsFrames, full);
  }
  for(auto& it : asyncSnapshots)
  {
    it.appendToString(statsAsyncs, full);
  }
}

void ProfilerManager::setUpdateResetWhenIdle(bool state)
{
  std::lock_guard lock(m_mutex);
  for(auto& it : m_timelines)
  {
    it->setUpdateResetWhenIdle(state);
  }
}

bool ProfilerManager::getUpdateResetWhenIdle() const
{
  std::lock_guard lock(m_mutex);
  for(auto& it : m_timelines)
  {
    return it->getUpdateResetWhenIdle();
  }
  return true;
}

void ProfilerManager::getSnapshots(std::vector<ProfilerTimeline::Snapshot>& frameSnapshots,
                                   std::vector<ProfilerTimeline::Snapshot>& asyncSnapshots) const
{
  std::lock_guard lock(m_mutex);
  frameSnapshots.resize(m_timelines.size());
  asyncSnapshots.resize(m_timelines.size());
  size_t i = 0;
  for(auto& it : m_timelines)
  {
    it->getFrameSnapshot(frameSnapshots[i]);
    it->getAsyncSnapshot(asyncSnapshots[i]);
    i++;
  }
}

void ProfilerManager::getSnapshots(std::vector<ProfilerTimeline::Snapshot>& frameSnapshots,
                                   std::vector<ProfilerTimeline::Snapshot>& asyncSnapshots,
                                   std::vector<ProfilerTimeline::Snapshot>& updateSnapshots) const
{
  std::lock_guard lock(m_mutex);
  frameSnapshots.resize(m_timelines.size());
  asyncSnapshots.resize(m_timelines.size());
  updateSnapshots.resize(m_timelines.size());
  size_t i = 0;
  for(auto& it : m_timelines)
  {
    it->getFrameSnapshot(frameSnapshots[i]);
    it->getAsyncSnapshot(asyncSnapshots[i]);
    it->getUpdateSnapshot(updateSnapshots[i]);
    i++;
  }
}

}  // namespace nvutils

//--------------------------------------------------------------------------------------------------
// Usage example
//--------------------------------------------------------------------------------------------------
[[maybe_unused]] static void usage_Profiler()
{
  nvutils::ProfilerManager profilerManager;

  // in a typical single-threaded main loop we have one timeline on which we submit commands.
  nvutils::ProfilerTimeline* profilerTimeline = profilerManager.createTimeline({"primary"});

  // A profiler timeline can measure async shot events
  {
    // async event timers are thread-safe

    auto profiledSection = profilerTimeline->asyncSection("preparation");

    // do some work
  }

  // or re-occurring per-frame events
  /* while(!glfwWindowShouldClose()) */
  {
    {
      // per-frame sections must be within frameBegin/frameEnd
      // and are NOT thread-safe with respect to the timeline.

      const auto profiledSection = profilerTimeline->frameSection("processing");

      // do some work

      // When `profiledSection` goes out of scope, it ends section timing
    }

    profilerTimeline->frameAdvance();
  }

  {
    // async event timers with identical names will be overwritten

    auto profiledSection = profilerTimeline->asyncSection("preparation");

    // do some work
  }

  // You can always query past results in a thread-safe manner.
  // Note that querying is always done in a non-blocking way if GPU timers are involved.
  // Therefore the results might be from past frames, or incomplete if they haven't finished yet.

  // Async event timers are queried at snapshot time, whilst frame event timers are
  // queried at end of frame.

  std::string myFrameStats;
  std::string myAsyncStats;

  profilerManager.appendPrint(myFrameStats, myAsyncStats);

  // output strings to log etc.
}
