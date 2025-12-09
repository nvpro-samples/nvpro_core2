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
  try  // Destructors in C++ are noexcept; catch just in case
  {
    if(m_logFile.is_open())
    {
      m_logFile.close();
    }
  }
  catch(const std::exception& e)
  {
    // Log it as best we can; not much we can do here
    std::cerr << "Caught exception while closing file: " << e.what() << "\n";
  }
}


void nvutils::Logger::setMinimumLogLevel(LogLevel level) noexcept
{
  std::lock_guard<std::recursive_mutex> lock(m_logMutex);
  m_minLogLevel = level;
}


void nvutils::Logger::setShowFlags(ShowFlags flags) noexcept
{
  std::lock_guard<std::recursive_mutex> lock(m_logMutex);
  m_show = flags;
}


void nvutils::Logger::setOutputFile(const std::filesystem::path& filename) noexcept
{
  std::lock_guard<std::recursive_mutex> lock(m_logMutex);
  try
  {
    if(m_logFile.is_open())
    {
      m_logFile.close();
    }
    m_logFile.open(filename, std::ios::out);
  }
  catch(const std::exception& /* unused */)
  {
    // m_logFile.operator bool() will be false and we'll handle the error below
  }

  if(!m_logFile)
  {
    std::cerr << "Failed to open log file: " << nvutils::utf8FromPath(filename) << std::endl;
    m_logToFile = false;
  }
  else
  {
    m_logToFile = true;
  }
}


void nvutils::Logger::enableFileOutput(bool enable) noexcept
{
  std::lock_guard<std::recursive_mutex> lock(m_logMutex);
  m_logToFile = enable;
}


void nvutils::Logger::setFileFlush(bool enable) noexcept
{
  std::lock_guard<std::recursive_mutex> lock(m_logMutex);
  m_fileFlush = enable;
}


void nvutils::Logger::setLogCallback(LogCallback&& callback) noexcept
{
  std::lock_guard<std::recursive_mutex> lock(m_logMutex);
  m_logCallback = std::move(callback);
}


void nvutils::Logger::log(LogLevel level,
#ifdef _MSC_VER
                          _Printf_format_string_
#endif
                          const char* format,
                          ...) noexcept
{
  std::lock_guard<std::recursive_mutex> lock(m_logMutex);
  if(level < m_minLogLevel)
    return;

  ensureLogFileIsOpen();

  // Format message
  std::string message;
  va_list     args;
  va_start(args, format);
  try
  {
    message = formatString(format, args);
    addPrefixes(level, message);
  }
  catch(const std::exception& e)
  {
    std::cerr << "Could not format string: " << e.what() << "\n";
    va_end(args);
    return;
  }
  va_end(args);

  outputToConsoles(level, message);
  outputToFile(message);
  outputToCallback(level, message);

  breakOnErrors(level, message);
}


void nvutils::Logger::breakOnError(bool enable) noexcept
{
  std::lock_guard<std::recursive_mutex> lock(m_logMutex);
  m_breakOnError = enable;
}


void nvutils::Logger::breakOnErrors(LogLevel level, const std::string& message) noexcept
{
  if(!m_breakOnError)
    return;
  if(level == LogLevel::eERROR)
  {
#ifdef _WIN32
    // Suppress a Coverity warning about debugger functions here -- this is
    // intentional.
    // coverity[dont_call]
    if(IsDebuggerPresent())
    {
      // If you've reached this breakpoint, your sample has just printed an
      // error. Look at the console or debug output to see what it is.
      // coverity[dont_call]
      DebugBreak();
    }
#elif defined(__unix__)
    // On Linux, we can use a breakpoint to stop the debugger
    raise(SIGTRAP);
#endif
  }
}


void nvutils::Logger::ensureLogFileIsOpen() noexcept
{
  static bool firstLog = true;
  if(firstLog && m_logToFile && !m_logFile.is_open())
  {
    firstLog = false;
    try
    {
      std::filesystem::path exePath = nvutils::getExecutablePath();
      std::filesystem::path logName = "log_";
      logName += exePath.stem();
      logName += ".txt";
      const std::filesystem::path logPath = exePath.parent_path() / logName;
      setOutputFile(logPath);
    }
    catch(const std::exception& e)
    {
      std::cerr << "Caught exception while opening log file: " << e.what() << "\n";
    }
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


static std::string currentTime()
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


static const char* logLevelToString(nvutils::Logger::LogLevel level)
{
  switch(level)
  {
    case nvutils::Logger::LogLevel::eDEBUG:
      return "DEBUG";
    case nvutils::Logger::LogLevel::eSTATS:
      return "STATS";
    case nvutils::Logger::LogLevel::eOK:
      return "OK";
    case nvutils::Logger::LogLevel::eINFO:
      return "INFO";
    case nvutils::Logger::LogLevel::eWARNING:
      return "WARNING";
    case nvutils::Logger::LogLevel::eERROR:
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


void nvutils::Logger::addPrefixes(LogLevel level, std::string& message)
{
  static bool suppressPrefixes = false;
  if(!suppressPrefixes && m_show != 0)
  {
    std::stringstream logStream;
    if(m_show & eSHOW_LEVEL)
      logStream << logLevelToString(level) << ": ";
    if(m_show & eSHOW_TIME)
      logStream << "[" << currentTime() << "] ";
    logStream << message;
    message = logStream.str();
  }
  suppressPrefixes = message.empty() || message.back() != '\n';
}


void nvutils::Logger::outputToConsoles(LogLevel level, const std::string& message) noexcept
{
  std::ostream* stdConsole = (level == LogLevel::eERROR ? &std::cerr : &std::cout);
#ifdef _WIN32
  // Convert our message to UTF-16, which is the same encoding as in
  // std::filesystem::path:
  const std::wstring utf16 = nvutils::pathFromUtf8(message).native();

  // Output to the debug console.
  // Coverity warns that this sends information to the debug console.
  // But that's exactly what we want to do, so it's a false positive.
  // coverity[dont_call]
  OutputDebugStringW(utf16.c_str());

  // Try printing to the console (which allows us to use UTF-8) with colors.
  // If that fails, the output has probably been redirected; use regular
  // std::cout and std::cerr.
  HANDLE hConsole       = GetStdHandle(level == LogLevel::eERROR ? STD_ERROR_HANDLE : STD_OUTPUT_HANDLE);
  BOOL   consoleWriteOk = TRUE;
  if(level == LogLevel::eERROR)
  {
    SetConsoleTextAttribute(hConsole, FOREGROUND_RED | FOREGROUND_INTENSITY);
    consoleWriteOk = WriteConsoleW(hConsole, utf16.c_str(), static_cast<DWORD>(utf16.size()), nullptr, nullptr);
    SetConsoleTextAttribute(hConsole, FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_BLUE);
  }
  else if(level == LogLevel::eWARNING)
  {
    SetConsoleTextAttribute(hConsole, FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_INTENSITY);
    consoleWriteOk = WriteConsoleW(hConsole, utf16.c_str(), static_cast<DWORD>(utf16.size()), nullptr, nullptr);
    SetConsoleTextAttribute(hConsole, FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_BLUE);
  }
  else
  {
    consoleWriteOk = WriteConsoleW(hConsole, utf16.c_str(), static_cast<DWORD>(utf16.size()), nullptr, nullptr);
  }

  if(!consoleWriteOk)
  {
    *stdConsole << message;
  }

#else
  const bool supportsColor = isatty(fileno(level == LogLevel::eERROR ? stderr : stdout));
  if(level == LogLevel::eERROR && supportsColor)
  {
    *stdConsole << "\033[1;31m" << message << "\033[0m";
  }
  else if(level == LogLevel::eWARNING && supportsColor)
  {
    *stdConsole << "\033[1;33m" << message << "\033[0m";
  }
  else
  {
    *stdConsole << message;
  }
#endif
}


void nvutils::Logger::outputToFile(const std::string& message) noexcept
{
  if(m_logToFile && m_logFile.is_open())
  {
    m_logFile << message;
    if(m_fileFlush)
    {
      m_logFile.flush();
    }
  }
}


void nvutils::Logger::outputToCallback(LogLevel level, const std::string& message) noexcept
{
  if(m_logCallback)
  {
    m_logCallback(level, message);
  }
}


[[maybe_unused]] static void usage_Logger()
{
  // Get the logger instance
  nvutils::Logger& logger = nvutils::Logger::getInstance();

  // Set the minimum log level
  logger.setMinimumLogLevel(nvutils::Logger::LogLevel::eINFO);

  // Set the information to show in the log
  logger.setShowFlags(nvutils::Logger::eSHOW_TIME | nvutils::Logger::eSHOW_LEVEL);

  // Set the output file : default is the name of the executable with .txt extension
  logger.setOutputFile("logfile.txt");

  // Enable or disable file output
  logger.enableFileOutput(true);

  // Set a custom log callback
  logger.setLogCallback([](nvutils::Logger::LogLevel level, const std::string& message) {
    std::cout << "Custom Log: " << message << std::endl;
  });

  // Log messages
  LOGD("This is a debug message.");
  LOGI("This is an info message.");
  LOGW("This is a warning message.");
  const int integerValue = 12345;
  LOGE("This is an error message with id: %d.", integerValue);
}
