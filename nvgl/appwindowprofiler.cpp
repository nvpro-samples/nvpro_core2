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

#include <algorithm>
#include <cassert>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdarg.h>
#include <cstdio>
#include <cfloat>
#include <filesystem>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

#include "fileoperations.hpp"
#include "appwindowprofiler.hpp"
#include <nvutils/file_operations.hpp>

namespace nvgl {

static void replace(std::string& str, const std::string& from, const std::string& to)
{
  size_t start_pos = 0;
  while((start_pos = str.find(from, start_pos)) != std::string::npos)
  {
    str.replace(start_pos, from.length(), to);
    start_pos += to.length();
  }
}

static void fixDeviceName(std::string& deviceName)
{
  replace(deviceName, "INTEL(R) ", "");
  replace(deviceName, "AMD ", "");
  replace(deviceName, "DRI ", "");
  replace(deviceName, "(TM) ", "");
  replace(deviceName, " Series", "");
  replace(deviceName, " Graphics", "");
  replace(deviceName, "/PCIe/SSE2", "");
  std::replace(deviceName.begin(), deviceName.end(), ' ', '_');

  deviceName.erase(std::remove(deviceName.begin(), deviceName.end(), '/'), deviceName.end());
  deviceName.erase(std::remove(deviceName.begin(), deviceName.end(), '\\'), deviceName.end());
  deviceName.erase(std::remove(deviceName.begin(), deviceName.end(), ':'), deviceName.end());
  deviceName.erase(std::remove(deviceName.begin(), deviceName.end(), '?'), deviceName.end());
  deviceName.erase(std::remove(deviceName.begin(), deviceName.end(), '*'), deviceName.end());
  deviceName.erase(std::remove(deviceName.begin(), deviceName.end(), '<'), deviceName.end());
  deviceName.erase(std::remove(deviceName.begin(), deviceName.end(), '>'), deviceName.end());
  deviceName.erase(std::remove(deviceName.begin(), deviceName.end(), '|'), deviceName.end());
  deviceName.erase(std::remove(deviceName.begin(), deviceName.end(), '"'), deviceName.end());
  deviceName.erase(std::remove(deviceName.begin(), deviceName.end(), ','), deviceName.end());
}


void AppWindowProfiler::parseConfigFile(const char* filename)
{
  nvutils::ParameterParser::Tokenized tokenized;

  if(!tokenized.initFromFile(filename))
  {
    LOGW("file not found: %s\n", filename);
    return;
  }

  std::string path = getFilePath(filename);

  m_parameterParser.parse(tokenized.getArgs(), false, path);
}


void AppWindowProfiler::setVsync(bool state)
{
  if(m_internal)
  {
    swapVsync(state);
    LOGI("vsync: %s\n", state ? "on" : "off");
  }
  m_config.vsyncstate = state;
  m_vsync             = state;
}

int AppWindowProfiler::run(const std::string& title, int argc, const char** argv, int width, int height)
{
  m_config.winsize[0] = m_config.winsize[0] ? m_config.winsize[0] : width;
  m_config.winsize[1] = m_config.winsize[1] ? m_config.winsize[1] : height;

  // skip first argument here (exe file)
  parseConfig(argc - 1, argv + 1, ".");
  if(!validateConfig())
  {
    return EXIT_FAILURE;
  }

  if(!open(m_config.winpos[0], m_config.winpos[1], m_config.winsize[0], m_config.winsize[1], title.c_str(), true))
  {
    LOGE("Could not create window\n");
    return EXIT_FAILURE;
  }
  m_windowState.m_winSize[0] = m_config.winsize[0];
  m_windowState.m_winSize[1] = m_config.winsize[1];

  postConfigPreContext();
  contextInit();
  m_activeContext = true;

  // hack to react on $DEVICE$ filename
  if(!m_config.logFilename.empty())
  {
    parameterCallback(m_paramLog);
  }

  if(contextGetDeviceName())
  {
    std::string deviceName = contextGetDeviceName();
    fixDeviceName(deviceName);
    LOGI("DEVICE: %s\n", deviceName.c_str());
  }

  initBenchmark();

  setVsync(m_config.vsyncstate);

  bool Run = begin();
  m_active = true;

  bool quickExit = m_config.quickexit;
  if(m_config.frameLimit)
  {
    m_profilerPrint = false;
    quickExit       = true;
  }

  double timeStart = getTime();
  double timeBegin = getTime();
  double frames    = 0;

  bool lastVsync = m_vsync;

  m_hadProfilerPrint = false;

  double lastProfilerPrintTime = 0;

  std::string timerFrame;
  std::string timerAsync;

  if(Run)
  {
    while(pollEvents())
    {
      bool wasClosed = false;
      while(!isOpen())
      {
        NVPSystem::waitEvents();
        wasClosed = true;
      }
      if(wasClosed)
      {
        continue;
      }

      if(m_windowState.onPress(KEY_V))
      {
        setVsync(!m_vsync);
      }

      std::string stats;
      {
        bool   benchmarkActive = m_benchmark.isActive && !m_benchmark.sequencer.isCompleted();
        double curTime         = getTime();
        double printInterval   = m_profilerPrint && !benchmarkActive ? float(m_config.intervalSeconds) : float(FLT_MAX);
        bool   printStats      = ((curTime - lastProfilerPrintTime) > printInterval);

        if(printStats)
        {
          lastProfilerPrintTime = curTime;
        }
        m_profilerTimeline->frameAdvance();

        {
          //const nvh::Profiler::Section profile(m_profiler, "App");
          think(getTime() - timeStart);
        }
        memset(m_windowState.m_keyToggled, 0, sizeof(m_windowState.m_keyToggled));
        swapBuffers();

        // Note: this will display stats of previous frame, not current frame.
        if(printStats)
        {
          timerFrame.clear();
          timerAsync.clear();
          m_profiler.appendPrint(timerFrame, timerAsync);
          LOGI("%s", timerFrame.c_str());
        }
      }

      m_hadProfilerPrint = false;

      if(m_profilerPrint && !stats.empty())
      {
        if(!m_config.timerLimit || m_config.timerLimit == 1)
        {
          LOGI("%s\n", stats.c_str());
          m_hadProfilerPrint = true;
        }
        if(m_config.timerLimit == 1)
        {
          m_config.frameLimit = 1;
        }
        if(m_config.timerLimit)
        {
          m_config.timerLimit--;
        }
      }

      advanceBenchmark();
      postProfiling();

      frames++;

      double timeCurrent = getTime();
      double timeDelta   = timeCurrent - timeBegin;
      if(timeDelta > double(m_config.intervalSeconds) || lastVsync != m_vsync || m_config.frameLimit == 1)
      {
        if(lastVsync != m_vsync)
        {
          timeDelta = 0;
        }

        if(m_timeInTitle)
        {
          std::ostringstream combined;
          combined << title << ": " << (timeDelta * 1000.0 / (frames)) << " [ms]"
                   << (m_vsync ? " (vsync on - V for toggle)" : "");
          setTitle(combined.str().c_str());
        }

        if(m_config.frameLimit == 1)
        {
          LOGI("frametime: %f ms\n", (timeDelta * 1000.0 / (frames)));
        }

        frames    = 0;
        timeBegin = timeCurrent;
        lastVsync = m_vsync;
      }

      if(m_windowState.m_keyPressed[KEY_ESCAPE] || m_config.frameLimit == 1)
        break;

      if(m_config.frameLimit)
        m_config.frameLimit--;
    }
  }
  contextSync();
  exitScreenshot();

  if(quickExit)
  {
    exit(EXIT_SUCCESS);
    return EXIT_SUCCESS;
  }

  end();
  m_active = false;
  contextDeinit();
  postEnd();

  return Run ? EXIT_SUCCESS : EXIT_FAILURE;
}

void AppWindowProfiler::leave()
{
  m_config.frameLimit = 1;
}

std::string AppWindowProfiler::specialStrings(const char* original)
{
  std::string str(original);

  if(strstr(original, "$DEVICE$"))
  {
    if(contextGetDeviceName())
    {
      std::string deviceName = contextGetDeviceName();
      fixDeviceName(deviceName);
      if(deviceName.empty())
      {
        // no proper device name available
        return std::string();
      }

      // replace $DEVICE$
      replace(str, "$DEVICE$", deviceName);
    }
    else
    {
      // no proper device name available
      return std::string();
    }
  }
  return str;
}

void AppWindowProfiler::parameterCallback(const nvutils::ParameterBase* param)
{
  if(param == m_paramLog)
  {
    const std::filesystem::path logfileName =
        nvutils::pathFromUtf8(specialStrings(nvutils::utf8FromPath(m_config.logFilename).c_str()));
    if(!logfileName.empty())
    {
      nvutils::Logger::getInstance().setOutputFile(logfileName.c_str());
    }
  }
  else if(param == m_paramCfg || param == m_paramBat)
  {
    parseConfigFile(m_config.configFilename.c_str());
  }
  else if(param == m_paramWinsize)
  {
    if(m_internal)
    {
      setWindowSize(m_config.winsize[0], m_config.winsize[1]);
    }
  }

  if(!m_active)
    return;

  if(param == m_paramVsync)
  {
    setVsync(m_config.vsyncstate);
  }
  else if(param == m_paramScreenshot)
  {
    std::string filename = specialStrings(m_config.screenshotFilename.c_str());
    if(!filename.empty())
    {
      screenshot(filename.c_str());
    }
  }
  else if(param == m_paramClear)
  {
    clear(m_config.clearColor[0], m_config.clearColor[1], m_config.clearColor[2]);
  }
}

void AppWindowProfiler::setupParameters()
{
  auto callback = [&](const nvutils::ParameterBase* param) { parameterCallback(param); };

  m_paramWinsize =
      m_parameterList.addArray(makeParamInfo("winsize|Set window size (width and height)", callback), 2, m_config.winsize);
  m_paramVsync = m_parameterList.add(makeParamInfo("vsync|Enable or disable vsync", callback), &m_config.vsyncstate);
  m_paramLog   = m_parameterList.add(makeParamInfo("logfile|set logfile", callback), &m_config.logFilename);

  m_parameterList.addArray(makeParamInfo("winpos|Set window position (x and y)"), 2, m_config.winpos);
  m_parameterList.add(makeParamInfo("frames|Set number of frames to render before exit"), &m_config.frameLimit);
  m_parameterList.add(makeParamInfo("timerprints|Set number of timerprints to do, before exit"), &m_config.timerLimit);
  m_parameterList.add(makeParamInfo("timerinterval|Set interval of timer prints in seconds"), &m_config.intervalSeconds);
  m_parameterList.add(makeParamInfo("bmpatexit|Set file to store a bitmap image of the last frame at exit"), &m_config.dumpatexitFilename);
  m_parameterList.add(makeParamInfo("benchmark|Set benchmark filename"), &m_benchmark.initInfo.scriptFilename);
  m_parameterList.add(makeParamInfo("quickexit|skips tear down"), &m_config.quickexit);
  m_paramScreenshot = m_parameterList.add(makeParamInfo("screenshot|makes a screenshot into this file", callback),
                                          &m_config.screenshotFilename);
  m_paramClear = m_parameterList.addArray(makeParamInfo("clear|clears window color (r,b,g in 0-255) using OS", callback),
                                          3, m_config.clearColor);

  m_parameterParser.add(m_parameterList);
}

void AppWindowProfiler::exitScreenshot()
{
  if(!m_config.dumpatexitFilename.empty() && !m_hadScreenshot)
  {
    screenshot(m_config.dumpatexitFilename.c_str());
    m_hadScreenshot = true;
  }
}

void AppWindowProfiler::initBenchmark()
{
  if(!m_benchmark.initInfo.hasScript())
    return;

  m_benchmark.initInfo.parameterParser   = &m_parameterParser;
  m_benchmark.initInfo.parameterRegistry = &m_parameterList;
  m_benchmark.initInfo.profilerManager   = &m_profiler;

  m_benchmark.isActive = m_benchmark.sequencer.init(m_benchmark.initInfo);
  if(m_benchmark.isActive)
  {
    // do first iteration at startup
    m_benchmark.sequencer.prepareFrame();

    m_profilerTimeline->resetFrameSections();

    m_profilerPrint = false;
  }
}

void AppWindowProfiler::advanceBenchmark()
{
  if(!m_benchmark.isActive)
    return;

  bool completed = m_benchmark.sequencer.prepareFrame();

  postBenchmarkAdvance();

  if(completed)
  {
    leave();
  }
}

void AppWindowProfiler::contextInit()
{
  // create OpenGL stuff at last
  m_contextWindow.init(&m_contextInfo, m_internal, m_windowName.c_str());
  // create other additional OpenGL tools
  m_profilerGL.init(m_profilerTimeline);

  m_windowState.m_swapSize[0] = getWidth();
  m_windowState.m_swapSize[1] = getHeight();
}

void AppWindowProfiler::contextDeinit()
{
  m_profilerGL.deinit();
  m_contextWindow.deinit();
}

nvutils::ParameterBase::Info makeParamInfo(const std::string& combined, nvutils::ParameterBase::CallbackSuccess callbackSuccess /*= nullptr*/)
{
  nvutils::ParameterBase::Info info;

  size_t start = 0;
  size_t end   = combined.find('|');
  if(end != std::string::npos)
  {
    info.shortName = info.name = combined.substr(start, end - start);
    info.help                  = combined.substr(end + 1);
  }
  else
  {
    info.name      = combined;
    info.shortName = combined;
  }

  info.callbackSuccess = std::move(callbackSuccess);

  return info;
}

static_assert(AppWindowProfiler::BUTTON_RELEASE == GLFW_RELEASE, "glfw/nvpwindow mismatch");
static_assert(AppWindowProfiler::BUTTON_PRESS == GLFW_PRESS, "glfw/nvpwindow mismatch");
static_assert(AppWindowProfiler::BUTTON_REPEAT == GLFW_REPEAT, "glfw/nvpwindow mismatch");

static_assert(AppWindowProfiler::MOUSE_BUTTON_LEFT == GLFW_MOUSE_BUTTON_LEFT, "glfw/nvpwindow mismatch");
static_assert(AppWindowProfiler::MOUSE_BUTTON_RIGHT == GLFW_MOUSE_BUTTON_RIGHT, "glfw/nvpwindow mismatch");
static_assert(AppWindowProfiler::MOUSE_BUTTON_MIDDLE == GLFW_MOUSE_BUTTON_MIDDLE, "glfw/nvpwindow mismatch");

static_assert(AppWindowProfiler::KMOD_SHIFT == GLFW_MOD_SHIFT, "glfw/nvpwindow mismatch");
static_assert(AppWindowProfiler::KMOD_CONTROL == GLFW_MOD_CONTROL, "glfw/nvpwindow mismatch");
static_assert(AppWindowProfiler::KMOD_ALT == GLFW_MOD_ALT, "glfw/nvpwindow mismatch");
static_assert(AppWindowProfiler::KMOD_SUPER == GLFW_MOD_SUPER, "glfw/nvpwindow mismatch");

/*
for key in keysheader:gmatch("#define ([%w_]+)") do
print("static_assert(AppWindowProfiler::"..key:sub(6,-1).." == "..key..", \"glfw/nvpwindow mismatch\");")
end
*/

static_assert(AppWindowProfiler::KEY_UNKNOWN == GLFW_KEY_UNKNOWN, "glfw/nvpwindow mismatch");
static_assert(AppWindowProfiler::KEY_SPACE == GLFW_KEY_SPACE, "glfw/nvpwindow mismatch");
static_assert(AppWindowProfiler::KEY_APOSTROPHE == GLFW_KEY_APOSTROPHE, "glfw/nvpwindow mismatch");
static_assert(AppWindowProfiler::KEY_COMMA == GLFW_KEY_COMMA, "glfw/nvpwindow mismatch");
static_assert(AppWindowProfiler::KEY_MINUS == GLFW_KEY_MINUS, "glfw/nvpwindow mismatch");
static_assert(AppWindowProfiler::KEY_PERIOD == GLFW_KEY_PERIOD, "glfw/nvpwindow mismatch");
static_assert(AppWindowProfiler::KEY_SLASH == GLFW_KEY_SLASH, "glfw/nvpwindow mismatch");
static_assert(AppWindowProfiler::KEY_0 == GLFW_KEY_0, "glfw/nvpwindow mismatch");
static_assert(AppWindowProfiler::KEY_1 == GLFW_KEY_1, "glfw/nvpwindow mismatch");
static_assert(AppWindowProfiler::KEY_2 == GLFW_KEY_2, "glfw/nvpwindow mismatch");
static_assert(AppWindowProfiler::KEY_3 == GLFW_KEY_3, "glfw/nvpwindow mismatch");
static_assert(AppWindowProfiler::KEY_4 == GLFW_KEY_4, "glfw/nvpwindow mismatch");
static_assert(AppWindowProfiler::KEY_5 == GLFW_KEY_5, "glfw/nvpwindow mismatch");
static_assert(AppWindowProfiler::KEY_6 == GLFW_KEY_6, "glfw/nvpwindow mismatch");
static_assert(AppWindowProfiler::KEY_7 == GLFW_KEY_7, "glfw/nvpwindow mismatch");
static_assert(AppWindowProfiler::KEY_8 == GLFW_KEY_8, "glfw/nvpwindow mismatch");
static_assert(AppWindowProfiler::KEY_9 == GLFW_KEY_9, "glfw/nvpwindow mismatch");
static_assert(AppWindowProfiler::KEY_SEMICOLON == GLFW_KEY_SEMICOLON, "glfw/nvpwindow mismatch");
static_assert(AppWindowProfiler::KEY_EQUAL == GLFW_KEY_EQUAL, "glfw/nvpwindow mismatch");
static_assert(AppWindowProfiler::KEY_A == GLFW_KEY_A, "glfw/nvpwindow mismatch");
static_assert(AppWindowProfiler::KEY_B == GLFW_KEY_B, "glfw/nvpwindow mismatch");
static_assert(AppWindowProfiler::KEY_C == GLFW_KEY_C, "glfw/nvpwindow mismatch");
static_assert(AppWindowProfiler::KEY_D == GLFW_KEY_D, "glfw/nvpwindow mismatch");
static_assert(AppWindowProfiler::KEY_E == GLFW_KEY_E, "glfw/nvpwindow mismatch");
static_assert(AppWindowProfiler::KEY_F == GLFW_KEY_F, "glfw/nvpwindow mismatch");
static_assert(AppWindowProfiler::KEY_G == GLFW_KEY_G, "glfw/nvpwindow mismatch");
static_assert(AppWindowProfiler::KEY_H == GLFW_KEY_H, "glfw/nvpwindow mismatch");
static_assert(AppWindowProfiler::KEY_I == GLFW_KEY_I, "glfw/nvpwindow mismatch");
static_assert(AppWindowProfiler::KEY_J == GLFW_KEY_J, "glfw/nvpwindow mismatch");
static_assert(AppWindowProfiler::KEY_K == GLFW_KEY_K, "glfw/nvpwindow mismatch");
static_assert(AppWindowProfiler::KEY_L == GLFW_KEY_L, "glfw/nvpwindow mismatch");
static_assert(AppWindowProfiler::KEY_M == GLFW_KEY_M, "glfw/nvpwindow mismatch");
static_assert(AppWindowProfiler::KEY_N == GLFW_KEY_N, "glfw/nvpwindow mismatch");
static_assert(AppWindowProfiler::KEY_O == GLFW_KEY_O, "glfw/nvpwindow mismatch");
static_assert(AppWindowProfiler::KEY_P == GLFW_KEY_P, "glfw/nvpwindow mismatch");
static_assert(AppWindowProfiler::KEY_Q == GLFW_KEY_Q, "glfw/nvpwindow mismatch");
static_assert(AppWindowProfiler::KEY_R == GLFW_KEY_R, "glfw/nvpwindow mismatch");
static_assert(AppWindowProfiler::KEY_S == GLFW_KEY_S, "glfw/nvpwindow mismatch");
static_assert(AppWindowProfiler::KEY_T == GLFW_KEY_T, "glfw/nvpwindow mismatch");
static_assert(AppWindowProfiler::KEY_U == GLFW_KEY_U, "glfw/nvpwindow mismatch");
static_assert(AppWindowProfiler::KEY_V == GLFW_KEY_V, "glfw/nvpwindow mismatch");
static_assert(AppWindowProfiler::KEY_W == GLFW_KEY_W, "glfw/nvpwindow mismatch");
static_assert(AppWindowProfiler::KEY_X == GLFW_KEY_X, "glfw/nvpwindow mismatch");
static_assert(AppWindowProfiler::KEY_Y == GLFW_KEY_Y, "glfw/nvpwindow mismatch");
static_assert(AppWindowProfiler::KEY_Z == GLFW_KEY_Z, "glfw/nvpwindow mismatch");
static_assert(AppWindowProfiler::KEY_LEFT_BRACKET == GLFW_KEY_LEFT_BRACKET, "glfw/nvpwindow mismatch");
static_assert(AppWindowProfiler::KEY_BACKSLASH == GLFW_KEY_BACKSLASH, "glfw/nvpwindow mismatch");
static_assert(AppWindowProfiler::KEY_RIGHT_BRACKET == GLFW_KEY_RIGHT_BRACKET, "glfw/nvpwindow mismatch");
static_assert(AppWindowProfiler::KEY_GRAVE_ACCENT == GLFW_KEY_GRAVE_ACCENT, "glfw/nvpwindow mismatch");
static_assert(AppWindowProfiler::KEY_WORLD_1 == GLFW_KEY_WORLD_1, "glfw/nvpwindow mismatch");
static_assert(AppWindowProfiler::KEY_WORLD_2 == GLFW_KEY_WORLD_2, "glfw/nvpwindow mismatch");
static_assert(AppWindowProfiler::KEY_ESCAPE == GLFW_KEY_ESCAPE, "glfw/nvpwindow mismatch");
static_assert(AppWindowProfiler::KEY_ENTER == GLFW_KEY_ENTER, "glfw/nvpwindow mismatch");
static_assert(AppWindowProfiler::KEY_TAB == GLFW_KEY_TAB, "glfw/nvpwindow mismatch");
static_assert(AppWindowProfiler::KEY_BACKSPACE == GLFW_KEY_BACKSPACE, "glfw/nvpwindow mismatch");
static_assert(AppWindowProfiler::KEY_INSERT == GLFW_KEY_INSERT, "glfw/nvpwindow mismatch");
static_assert(AppWindowProfiler::KEY_DELETE == GLFW_KEY_DELETE, "glfw/nvpwindow mismatch");
static_assert(AppWindowProfiler::KEY_RIGHT == GLFW_KEY_RIGHT, "glfw/nvpwindow mismatch");
static_assert(AppWindowProfiler::KEY_LEFT == GLFW_KEY_LEFT, "glfw/nvpwindow mismatch");
static_assert(AppWindowProfiler::KEY_DOWN == GLFW_KEY_DOWN, "glfw/nvpwindow mismatch");
static_assert(AppWindowProfiler::KEY_UP == GLFW_KEY_UP, "glfw/nvpwindow mismatch");
static_assert(AppWindowProfiler::KEY_PAGE_UP == GLFW_KEY_PAGE_UP, "glfw/nvpwindow mismatch");
static_assert(AppWindowProfiler::KEY_PAGE_DOWN == GLFW_KEY_PAGE_DOWN, "glfw/nvpwindow mismatch");
static_assert(AppWindowProfiler::KEY_HOME == GLFW_KEY_HOME, "glfw/nvpwindow mismatch");
static_assert(AppWindowProfiler::KEY_END == GLFW_KEY_END, "glfw/nvpwindow mismatch");
static_assert(AppWindowProfiler::KEY_CAPS_LOCK == GLFW_KEY_CAPS_LOCK, "glfw/nvpwindow mismatch");
static_assert(AppWindowProfiler::KEY_SCROLL_LOCK == GLFW_KEY_SCROLL_LOCK, "glfw/nvpwindow mismatch");
static_assert(AppWindowProfiler::KEY_NUM_LOCK == GLFW_KEY_NUM_LOCK, "glfw/nvpwindow mismatch");
static_assert(AppWindowProfiler::KEY_PRINT_SCREEN == GLFW_KEY_PRINT_SCREEN, "glfw/nvpwindow mismatch");
static_assert(AppWindowProfiler::KEY_PAUSE == GLFW_KEY_PAUSE, "glfw/nvpwindow mismatch");
static_assert(AppWindowProfiler::KEY_F1 == GLFW_KEY_F1, "glfw/nvpwindow mismatch");
static_assert(AppWindowProfiler::KEY_F2 == GLFW_KEY_F2, "glfw/nvpwindow mismatch");
static_assert(AppWindowProfiler::KEY_F3 == GLFW_KEY_F3, "glfw/nvpwindow mismatch");
static_assert(AppWindowProfiler::KEY_F4 == GLFW_KEY_F4, "glfw/nvpwindow mismatch");
static_assert(AppWindowProfiler::KEY_F5 == GLFW_KEY_F5, "glfw/nvpwindow mismatch");
static_assert(AppWindowProfiler::KEY_F6 == GLFW_KEY_F6, "glfw/nvpwindow mismatch");
static_assert(AppWindowProfiler::KEY_F7 == GLFW_KEY_F7, "glfw/nvpwindow mismatch");
static_assert(AppWindowProfiler::KEY_F8 == GLFW_KEY_F8, "glfw/nvpwindow mismatch");
static_assert(AppWindowProfiler::KEY_F9 == GLFW_KEY_F9, "glfw/nvpwindow mismatch");
static_assert(AppWindowProfiler::KEY_F10 == GLFW_KEY_F10, "glfw/nvpwindow mismatch");
static_assert(AppWindowProfiler::KEY_F11 == GLFW_KEY_F11, "glfw/nvpwindow mismatch");
static_assert(AppWindowProfiler::KEY_F12 == GLFW_KEY_F12, "glfw/nvpwindow mismatch");
static_assert(AppWindowProfiler::KEY_F13 == GLFW_KEY_F13, "glfw/nvpwindow mismatch");
static_assert(AppWindowProfiler::KEY_F14 == GLFW_KEY_F14, "glfw/nvpwindow mismatch");
static_assert(AppWindowProfiler::KEY_F15 == GLFW_KEY_F15, "glfw/nvpwindow mismatch");
static_assert(AppWindowProfiler::KEY_F16 == GLFW_KEY_F16, "glfw/nvpwindow mismatch");
static_assert(AppWindowProfiler::KEY_F17 == GLFW_KEY_F17, "glfw/nvpwindow mismatch");
static_assert(AppWindowProfiler::KEY_F18 == GLFW_KEY_F18, "glfw/nvpwindow mismatch");
static_assert(AppWindowProfiler::KEY_F19 == GLFW_KEY_F19, "glfw/nvpwindow mismatch");
static_assert(AppWindowProfiler::KEY_F20 == GLFW_KEY_F20, "glfw/nvpwindow mismatch");
static_assert(AppWindowProfiler::KEY_F21 == GLFW_KEY_F21, "glfw/nvpwindow mismatch");
static_assert(AppWindowProfiler::KEY_F22 == GLFW_KEY_F22, "glfw/nvpwindow mismatch");
static_assert(AppWindowProfiler::KEY_F23 == GLFW_KEY_F23, "glfw/nvpwindow mismatch");
static_assert(AppWindowProfiler::KEY_F24 == GLFW_KEY_F24, "glfw/nvpwindow mismatch");
static_assert(AppWindowProfiler::KEY_F25 == GLFW_KEY_F25, "glfw/nvpwindow mismatch");
static_assert(AppWindowProfiler::KEY_KP_0 == GLFW_KEY_KP_0, "glfw/nvpwindow mismatch");
static_assert(AppWindowProfiler::KEY_KP_1 == GLFW_KEY_KP_1, "glfw/nvpwindow mismatch");
static_assert(AppWindowProfiler::KEY_KP_2 == GLFW_KEY_KP_2, "glfw/nvpwindow mismatch");
static_assert(AppWindowProfiler::KEY_KP_3 == GLFW_KEY_KP_3, "glfw/nvpwindow mismatch");
static_assert(AppWindowProfiler::KEY_KP_4 == GLFW_KEY_KP_4, "glfw/nvpwindow mismatch");
static_assert(AppWindowProfiler::KEY_KP_5 == GLFW_KEY_KP_5, "glfw/nvpwindow mismatch");
static_assert(AppWindowProfiler::KEY_KP_6 == GLFW_KEY_KP_6, "glfw/nvpwindow mismatch");
static_assert(AppWindowProfiler::KEY_KP_7 == GLFW_KEY_KP_7, "glfw/nvpwindow mismatch");
static_assert(AppWindowProfiler::KEY_KP_8 == GLFW_KEY_KP_8, "glfw/nvpwindow mismatch");
static_assert(AppWindowProfiler::KEY_KP_9 == GLFW_KEY_KP_9, "glfw/nvpwindow mismatch");
static_assert(AppWindowProfiler::KEY_KP_DECIMAL == GLFW_KEY_KP_DECIMAL, "glfw/nvpwindow mismatch");
static_assert(AppWindowProfiler::KEY_KP_DIVIDE == GLFW_KEY_KP_DIVIDE, "glfw/nvpwindow mismatch");
static_assert(AppWindowProfiler::KEY_KP_MULTIPLY == GLFW_KEY_KP_MULTIPLY, "glfw/nvpwindow mismatch");
static_assert(AppWindowProfiler::KEY_KP_SUBTRACT == GLFW_KEY_KP_SUBTRACT, "glfw/nvpwindow mismatch");
static_assert(AppWindowProfiler::KEY_KP_ADD == GLFW_KEY_KP_ADD, "glfw/nvpwindow mismatch");
static_assert(AppWindowProfiler::KEY_KP_ENTER == GLFW_KEY_KP_ENTER, "glfw/nvpwindow mismatch");
static_assert(AppWindowProfiler::KEY_KP_EQUAL == GLFW_KEY_KP_EQUAL, "glfw/nvpwindow mismatch");
static_assert(AppWindowProfiler::KEY_LEFT_SHIFT == GLFW_KEY_LEFT_SHIFT, "glfw/nvpwindow mismatch");
static_assert(AppWindowProfiler::KEY_LEFT_CONTROL == GLFW_KEY_LEFT_CONTROL, "glfw/nvpwindow mismatch");
static_assert(AppWindowProfiler::KEY_LEFT_ALT == GLFW_KEY_LEFT_ALT, "glfw/nvpwindow mismatch");
static_assert(AppWindowProfiler::KEY_LEFT_SUPER == GLFW_KEY_LEFT_SUPER, "glfw/nvpwindow mismatch");
static_assert(AppWindowProfiler::KEY_RIGHT_SHIFT == GLFW_KEY_RIGHT_SHIFT, "glfw/nvpwindow mismatch");
static_assert(AppWindowProfiler::KEY_RIGHT_CONTROL == GLFW_KEY_RIGHT_CONTROL, "glfw/nvpwindow mismatch");
static_assert(AppWindowProfiler::KEY_RIGHT_ALT == GLFW_KEY_RIGHT_ALT, "glfw/nvpwindow mismatch");
static_assert(AppWindowProfiler::KEY_RIGHT_SUPER == GLFW_KEY_RIGHT_SUPER, "glfw/nvpwindow mismatch");
static_assert(AppWindowProfiler::KEY_MENU == GLFW_KEY_MENU, "glfw/nvpwindow mismatch");
static_assert(AppWindowProfiler::KEY_LAST == GLFW_KEY_LAST, "glfw/nvpwindow mismatch");


void AppWindowProfiler::onMouseMotion(int x, int y)
{
  AppWindowProfiler::WindowState& window = m_windowState;

  if(!window.m_mouseButtonFlags && mouse_pos(x, y))
    return;

  window.m_mouseCurrent[0] = x;
  window.m_mouseCurrent[1] = y;
}

void AppWindowProfiler::onMouseButton(MouseButton Button, ButtonAction Action, int mods, int x, int y)
{
  AppWindowProfiler::WindowState& window = m_windowState;
  m_profilerTimeline->resetFrameSections();

  if(mouse_button(Button, Action))
    return;

  switch(Action)
  {
    case BUTTON_PRESS: {
      switch(Button)
      {
        case MOUSE_BUTTON_LEFT: {
          window.m_mouseButtonFlags |= MOUSE_BUTTONFLAG_LEFT;
        }
        break;
        case MOUSE_BUTTON_MIDDLE: {
          window.m_mouseButtonFlags |= MOUSE_BUTTONFLAG_MIDDLE;
        }
        break;
        case MOUSE_BUTTON_RIGHT: {
          window.m_mouseButtonFlags |= MOUSE_BUTTONFLAG_RIGHT;
        }
        break;
      }
    }
    break;
    case BUTTON_RELEASE: {
      if(!window.m_mouseButtonFlags)
        break;

      switch(Button)
      {
        case MOUSE_BUTTON_LEFT: {
          window.m_mouseButtonFlags &= ~MOUSE_BUTTONFLAG_LEFT;
        }
        break;
        case MOUSE_BUTTON_MIDDLE: {
          window.m_mouseButtonFlags &= ~MOUSE_BUTTONFLAG_MIDDLE;
        }
        break;
        case MOUSE_BUTTON_RIGHT: {
          window.m_mouseButtonFlags &= ~MOUSE_BUTTONFLAG_RIGHT;
        }
        break;
      }
    }
    break;
  }
}

void AppWindowProfiler::onMouseWheel(int y)
{
  AppWindowProfiler::WindowState& window = m_windowState;
  m_profilerTimeline->resetFrameSections();

  if(mouse_wheel(y))
    return;

  window.m_mouseWheel += y;
}

void AppWindowProfiler::onKeyboard(KeyCode key, ButtonAction action, int mods, int x, int y)
{
  AppWindowProfiler::WindowState& window = m_windowState;
  m_profilerTimeline->resetFrameSections();

  if(key_button(key, action, mods))
    return;

  bool newState = false;

  switch(action)
  {
    case BUTTON_PRESS:
    case BUTTON_REPEAT: {
      newState = true;
      break;
    }
    case BUTTON_RELEASE: {
      newState = false;
      break;
    }
  }

  window.m_keyToggled[key] = window.m_keyPressed[key] != newState;
  window.m_keyPressed[key] = newState;
}

void AppWindowProfiler::onKeyboardChar(unsigned char key, int mods, int x, int y)
{
  m_profilerTimeline->resetFrameSections();

  if(key_char(key))
    return;
}


void AppWindowProfiler::onWindowClose()
{
  exitScreenshot();
}

void AppWindowProfiler::onWindowResize(int width, int height)
{
  m_profilerTimeline->resetFrameSections();

  if(width == 0 || height == 0)
  {
    return;
  }

  m_windowState.m_winSize[0] = width;
  m_windowState.m_winSize[1] = height;
  if(m_activeContext)
  {
    swapResize(width, height);
  }
  if(m_active)
  {
    resize(m_windowState.m_swapSize[0], m_windowState.m_swapSize[1]);
  }
}

void AppWindowProfiler::cb_windowsizefun(GLFWwindow* glfwwin, int w, int h)
{
  AppWindowProfiler* win = (AppWindowProfiler*)glfwGetWindowUserPointer(glfwwin);
  if(win->isClosing())
    return;
  win->m_windowSize[0] = w;
  win->m_windowSize[1] = h;
  win->onWindowResize(w, h);
}
void AppWindowProfiler::cb_windowclosefun(GLFWwindow* glfwwin)
{
  AppWindowProfiler* win = (AppWindowProfiler*)glfwGetWindowUserPointer(glfwwin);
  win->m_isClosing       = true;
  win->onWindowClose();
}

void AppWindowProfiler::cb_mousebuttonfun(GLFWwindow* glfwwin, int button, int action, int mods)
{
  double x, y;
  glfwGetCursorPos(glfwwin, &x, &y);

  AppWindowProfiler* win = (AppWindowProfiler*)glfwGetWindowUserPointer(glfwwin);
  if(win->isClosing())
    return;
  win->m_keyModifiers = mods;
  win->m_mouseX       = int(x);
  win->m_mouseY       = int(y);
  win->onMouseButton((AppWindowProfiler::MouseButton)button, (AppWindowProfiler::ButtonAction)action, mods,
                     win->m_mouseX, win->m_mouseY);
}
void AppWindowProfiler::cb_cursorposfun(GLFWwindow* glfwwin, double x, double y)
{
  AppWindowProfiler* win = (AppWindowProfiler*)glfwGetWindowUserPointer(glfwwin);
  if(win->isClosing())
    return;
  win->m_mouseX = int(x);
  win->m_mouseY = int(y);
  win->onMouseMotion(win->m_mouseX, win->m_mouseY);
}
void AppWindowProfiler::cb_scrollfun(GLFWwindow* glfwwin, double x, double y)
{
  AppWindowProfiler* win = (AppWindowProfiler*)glfwGetWindowUserPointer(glfwwin);
  if(win->isClosing())
    return;
  win->m_mouseWheel += int(y);
  win->onMouseWheel(int(y));
}
void AppWindowProfiler::cb_keyfun(GLFWwindow* glfwwin, int key, int scancode, int action, int mods)
{
  AppWindowProfiler* win = (AppWindowProfiler*)glfwGetWindowUserPointer(glfwwin);
  if(win->isClosing())
    return;
  win->m_keyModifiers = mods;
  win->onKeyboard((AppWindowProfiler::KeyCode)key, (AppWindowProfiler::ButtonAction)action, mods, win->m_mouseX, win->m_mouseY);
}
void AppWindowProfiler::cb_charfun(GLFWwindow* glfwwin, unsigned int codepoint)
{
  AppWindowProfiler* win = (AppWindowProfiler*)glfwGetWindowUserPointer(glfwwin);
  if(win->isClosing())
    return;
  win->onKeyboardChar(codepoint, win->m_keyModifiers, win->m_mouseX, win->m_mouseY);
}

void AppWindowProfiler::cb_dropfun(GLFWwindow* glfwwin, int count, const char** paths)
{
  AppWindowProfiler* win = (AppWindowProfiler*)glfwGetWindowUserPointer(glfwwin);
  if(win->isClosing())
    return;
  win->onDragDrop(count, paths);
}

bool AppWindowProfiler::isClosing() const
{
  return m_isClosing || glfwWindowShouldClose(m_internal);
}

bool AppWindowProfiler::isOpen() const
{
  return glfwGetWindowAttrib(m_internal, GLFW_VISIBLE) == GLFW_TRUE
         && glfwGetWindowAttrib(m_internal, GLFW_ICONIFIED) == GLFW_FALSE && !isClosing();
}

bool AppWindowProfiler::open(int posX, int posY, int width, int height, const char* title, bool requireGLContext)
{
  assert(NVPSystem::isInited() && "NVPSystem::Init not called");

  m_windowSize[0] = width;
  m_windowSize[1] = height;

  m_windowName = title ? title : "Sample";

#ifdef _WIN32
  (void)requireGLContext;
  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
#else
  if(!requireGLContext)
  {
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
  }
  else
  {
    glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_API);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
    // Some samples make use of compatibility profile features
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_COMPAT_PROFILE);
#ifndef NDEBUG
#ifdef GLFW_CONTEXT_DEBUG  // Since GLFW_CONTEXT_DEBUG is new in GLFW 3.4
    glfwWindowHint(GLFW_CONTEXT_DEBUG, 1);
#else
    glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, 1);
#endif
#endif
  }
#endif

  m_internal = glfwCreateWindow(width, height, m_windowName.c_str(), nullptr, nullptr);
  if(!m_internal)
  {
    return false;
  }

  if(posX != 0 || posY != 0)
  {
    glfwSetWindowPos(m_internal, posX, posY);
  }
  glfwSetWindowUserPointer(m_internal, this);
  glfwSetWindowCloseCallback(m_internal, cb_windowclosefun);
  glfwSetCursorPosCallback(m_internal, cb_cursorposfun);
  glfwSetMouseButtonCallback(m_internal, cb_mousebuttonfun);
  glfwSetKeyCallback(m_internal, cb_keyfun);
  glfwSetScrollCallback(m_internal, cb_scrollfun);
  glfwSetCharCallback(m_internal, cb_charfun);
  glfwSetWindowSizeCallback(m_internal, cb_windowsizefun);
  glfwSetDropCallback(m_internal, cb_dropfun);

  return true;
}

void AppWindowProfiler::deinit()
{
  glfwDestroyWindow(m_internal);
  m_internal      = nullptr;
  m_windowSize[0] = 0;
  m_windowSize[1] = 0;
  m_windowName    = std::string();
}

void AppWindowProfiler::close()
{
  glfwSetWindowShouldClose(m_internal, GLFW_TRUE);
}

void AppWindowProfiler::setTitle(const char* title)
{
  glfwSetWindowTitle(m_internal, title);
}

void AppWindowProfiler::maximize()
{
  glfwMaximizeWindow(m_internal);
}

void AppWindowProfiler::restore()
{
  glfwRestoreWindow(m_internal);
}

void AppWindowProfiler::minimize()
{
  glfwIconifyWindow(m_internal);
}

void AppWindowProfiler::setWindowPos(int x, int y)
{
  glfwSetWindowPos(m_internal, x, y);
}

void AppWindowProfiler::setWindowSize(int w, int h)
{
  glfwSetWindowSize(m_internal, w, h);
}

std::filesystem::path AppWindowProfiler::openFileDialog(const char* title, const char* exts)
{
  return NVPSystem::windowOpenFileDialog(m_internal, title, exts);
}
std::filesystem::path AppWindowProfiler::saveFileDialog(const char* title, const char* exts)
{
  return NVPSystem::windowSaveFileDialog(m_internal, title, exts);
}
void AppWindowProfiler::screenshot(const char* filename)
{
  NVPSystem::windowScreenshot(m_internal, filename);
}
void AppWindowProfiler::clear(uint32_t r, uint32_t g, uint32_t b)
{
  NVPSystem::windowClear(m_internal, r, g, b);
}

void AppWindowProfiler::setFullScreen(bool bYes)
{
  if(bYes == m_isFullScreen)
    return;

  GLFWmonitor*       monitor = glfwGetWindowMonitor(m_internal);
  const GLFWvidmode* mode    = glfwGetVideoMode(monitor);

  if(bYes)
  {
    glfwGetWindowPos(m_internal, &m_preFullScreenPos[0], &m_preFullScreenPos[1]);
    glfwGetWindowSize(m_internal, &m_preFullScreenSize[0], &m_preFullScreenSize[1]);
    glfwSetWindowMonitor(m_internal, monitor, 0, 0, mode->width, mode->height, mode->refreshRate);
    glfwSetWindowAttrib(m_internal, GLFW_RESIZABLE, GLFW_FALSE);
    glfwSetWindowAttrib(m_internal, GLFW_DECORATED, GLFW_FALSE);
  }
  else
  {
    glfwSetWindowMonitor(m_internal, nullptr, m_preFullScreenPos[0], m_preFullScreenPos[1], m_preFullScreenSize[0],
                         m_preFullScreenSize[1], 0);
    glfwSetWindowAttrib(m_internal, GLFW_RESIZABLE, GLFW_TRUE);
    glfwSetWindowAttrib(m_internal, GLFW_DECORATED, GLFW_TRUE);
  }

  m_isFullScreen = bYes;
}

}  // namespace nvgl
