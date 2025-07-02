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


#include <csignal>
#include <cstdarg>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>


#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <Windows.h>
#include <debugapi.h>
#elif defined(__unix__)
#include <signal.h>
#endif

#include <fmt/format.h>


#include "logger.hpp"
#include "file_operations.hpp"
#include "timers.hpp"

nvutils::Logger::~Logger()
{
  if(m_logFile.is_open())
  {
    m_logFile.close();
  }
}

void nvutils::Logger::setOutputFile(const std::filesystem::path& filename)
{
  std::lock_guard<std::mutex> lock(m_logMutex);
  if(m_logFile.is_open())
  {
    m_logFile.close();
  }
  m_logFile.open(filename, std::ios::out);
  if(!m_logFile)
  {
    std::cerr << "Failed to open log file: " << filename << std::endl;
    m_logToFile = false;
  }
  else
  {
    m_logToFile = true;
  }
}


void nvutils::Logger::setLogCallback(LogCallback callback)
{
  std::lock_guard<std::mutex> lock(m_logMutex);
  m_logCallback = std::move(callback);
}

void nvutils::Logger::log(LogLevel level,
#ifdef _MSC_VER
                          _Printf_format_string_
#endif
                          const char* format,
                          ...)
{
  if(level < m_minLogLevel)
    return;

  ensureLogFileIsOpen();

  // Format message
  va_list args;
  va_start(args, format);
  std::string message = formatString(format, args);
  va_end(args);

  std::ostringstream logStream;
  addPrefixes(logStream, level, message);
  outputToConsole(level, logStream.str());
  outputToFile(logStream.str());
  outputToCallback(level, logStream.str());

#ifdef _WIN32
  OutputDebugStringA((logStream.str()).c_str());
#endif

  breakOnErrors(level, logStream.str());
}

void nvutils::Logger::breakOnErrors(LogLevel level, const std::string& message)
{
  if(!m_breakOnError)
    return;
  if(level == LogLevel::eERROR)
  {
#ifdef _WIN32
    if(IsDebuggerPresent())
    {
      DebugBreak();
    }
#elif defined(__unix__)
    // On Linux, we can use a breakpoint to stop the debugger
    raise(SIGTRAP);
#endif
  }
}


void nvutils::Logger::ensureLogFileIsOpen()
{
  static bool firstLog = true;
  if(firstLog && m_logToFile && !m_logFile.is_open())
  {
    firstLog                      = false;
    std::filesystem::path exePath = nvutils::getExecutablePath();
    std::filesystem::path logName = "log_";
    logName += exePath.stem();
    logName += ".txt";
    const std::filesystem::path logPath = exePath.parent_path() / logName;
    setOutputFile(logPath);
  }
}

void nvutils::Logger::addPrefixes(std::ostringstream& logStream, LogLevel level, const std::string& message)
{
  static bool suppressPrefixes = false;
  if(!suppressPrefixes)
  {
    if(m_show & eSHOW_TIME)
      logStream << "[" << currentTime() << "] ";
    if(m_show & eSHOW_LEVEL)
      logStream << logLevelToString(level) << ": ";
  }
  logStream << message;
  suppressPrefixes = message.back() != '\n';
}

void nvutils::Logger::outputToFile(const std::string& message)
{
  if(m_logToFile && m_logFile.is_open())
  {
    std::lock_guard<std::mutex> lock(m_logMutex);
    m_logFile << message /* << std::endl */;
  }
}

void nvutils::Logger::outputToCallback(LogLevel level, const std::string& message)
{
  if(m_logCallback)
  {
    m_logCallback(level, message);
  }
}


std::string nvutils::Logger::formatString(const char* format, va_list args)
{
  // Initial buffer size
  int               bufferSize = 1024;
  std::vector<char> buffer(bufferSize);

  // Try to format the string into the buffer
  va_list argsCopy;
  va_copy(argsCopy, args);  // Copy args to reuse them for vsnprintf
  int requiredSize = vsnprintf(buffer.data(), bufferSize, format, argsCopy);
  va_end(argsCopy);

  // Check if the buffer was large enough
  if(requiredSize >= bufferSize)
  {
    bufferSize = requiredSize + 1;  // Increase buffer size as needed
    buffer.resize(bufferSize);
    vsnprintf(buffer.data(), bufferSize, format, args);  // Format again with correct size
  }

  return std::string(buffer.data());
}

const char* nvutils::Logger::logLevelToString(LogLevel level)
{
  switch(level)
  {
    case LogLevel::eDEBUG:
      return "DEBUG";
    case LogLevel::eSTATS:
      return "STATS";
    case LogLevel::eOK:
      return "OK";
    case LogLevel::eINFO:
      return "INFO";
    case LogLevel::eWARNING:
      return "WARNING";
    case LogLevel::eERROR:
      return "ERROR";
  }
    // This checks that we've handled all enum cases above; it can be replaced
    // with std::unreachable in C++23.
#ifdef _MSC_VER
  __assume(false);
#else
  __builtin_unreachable();
#endif
  return "";
}

void nvutils::Logger::outputToConsole(LogLevel level, const std::string& message)
{
#ifdef _WIN32
  HANDLE hConsole = GetStdHandle(STD_OUTPUT_HANDLE);
  if(level == LogLevel::eERROR)
  {
    SetConsoleTextAttribute(hConsole, FOREGROUND_RED | FOREGROUND_INTENSITY);
    std::cerr << message /* << std::endl */;
  }
  else if(level == LogLevel::eWARNING)
  {
    SetConsoleTextAttribute(hConsole, FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_INTENSITY);
    std::cout << message /* << std::endl */;
  }
  else
  {
    SetConsoleTextAttribute(hConsole, FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_BLUE);
    std::cout << message /* << std::endl */;
  }
  SetConsoleTextAttribute(hConsole, FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_BLUE);
#else
  if(level == LogLevel::eERROR)
  {
    std::cerr << "\033[1;31m" << message << "\033[0m" /* << std::endl */;
  }
  else if(level == LogLevel::eWARNING)
  {
    std::cout << "\033[1;33m" << message << "\033[0m" /* << std::endl */;
  }
  else
  {
    std::cout << message /* << std::endl */;
  }
#endif
}


std::string nvutils::Logger::currentTime()
{
  static nvutils::PerformanceTimer startTimer;

  // Get total milliseconds and extract hours, minutes, seconds
  uint64_t duration = static_cast<uint64_t>(startTimer.getMilliseconds());
  // clang-format off
  const uint64_t ms      = duration % 1000; duration /= 1000;
  const uint64_t seconds = duration % 60;   duration /= 60;
  const uint64_t minutes = duration % 60;   duration /= 60;
  const uint64_t hours   = duration;
  // clang-format on

  return fmt::format("{:02}:{:02}:{:02}.{:03}", hours, minutes, seconds, ms);
}
