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
#include <filesystem>
#include <fstream>
#include <functional>
#include <mutex>
#include <string>

/*

Logger class for reporting messages with different log levels.
It can print to multiple places at once, produce breakpoints on errors,
 
To use it, call one of the LOG functions as you would printf.
LOGI, LOGW, LOGE = info, warning, error, and so on.
Alternatively, you can use std::print-style syntax by defining
NVLOGGER_ENABLE_FMT before including this file and then using the PRINT macros.

For more examples, please see usage_Logger() in logger.cpp.

# Text encoding:
Printing to the Windows debug console is the only operation that assumes a
text encoding; the input is assumed to be UTF-8. In all other cases, strings
are copied into the output.

# Safety:
On error, all functions print an error message. None throw.

All functions are thread-safe.

`Printf`-style functions have annotations that should produce warnings at
compile-time or when performing static analysis. Their format strings may be
dynamic - but this can be bad if an adversary can choose the content of the
format string.

`std::print`-style functions are safer: they produce compile-time errors, and
their format strings must be compile-time constants. Dynamic formatting
should be performed outside of printing, like this:

```cpp
ImGui::InputText("Enter a format string: ", userFormat, sizeof(userFormat));
try
{
  std::string formatted = fmt::vformat(userFormat, ...);
}
catch (const std::exception& e)
{
  (error handling...)
}
PRINTI("{}", formatted);
```

*/


#ifndef LOGGER_HPP
#define LOGGER_HPP


namespace nvutils {

class Logger
{
public:
  // Log levels. Higher values are more severe.
  enum LogLevel
  {
    // Info only useful during sample development.
    eDEBUG = 0,
    // Performance statistics.
    eSTATS = 1,
    // An operation succeeded.
    eOK = 2,
    // General information.
    eINFO = 3,
    // Recoverable errors: "something is not good but I can return an error
    // code that the app will look at".
    eWARNING = 4,
    // Unrecoverable errors; coding errors; "should never happen" errors.
    // Breaks if m_breakOnError is set.
    eERROR = 5,
  };

  enum ShowBits : uint32_t
  {
    eSHOW_NONE  = 0,
    eSHOW_TIME  = 1 << 0,
    eSHOW_LEVEL = 1 << 1
  };
  using ShowFlags = uint32_t;

  using LogCallback = std::function<void(LogLevel, const std::string&)>;

  // Get the logger instance
  static Logger& getInstance() noexcept
  {
    static Logger instance;
    return instance;
  }

  // Set the minimum log level
  void setMinimumLogLevel(LogLevel level) noexcept;

  // Set the information to show in the log
  void setShowFlags(ShowFlags flags) noexcept;

  // Set the output file
  void setOutputFile(const std::filesystem::path& filename) noexcept;

  // Enable or disable file output
  void enableFileOutput(bool enable) noexcept;

  // Set whether to flush all prints to the log file. This can be useful for
  // debugging on OSes that buffer writes such as Linux.
  void setFileFlush(bool enable) noexcept;

  // Set a custom log callback
  void setLogCallback(LogCallback&& callback) noexcept;

  // Log a message
  void log(LogLevel level,
#ifdef _MSC_VER
           _Printf_format_string_  // Enable MSVC /analyze warnings about incorrect format strings
#endif
           const char* format,
           ...) noexcept
#if defined(__GNUC__)
      __attribute__((format(printf, 3, 4)))  // Enable GCC + clang warnings about incorrect format strings
#endif
      ;

  // Set whether to break on errors
  void breakOnError(bool enable) noexcept;

private:
#ifdef DEBUG
  LogLevel m_minLogLevel = LogLevel::eDEBUG;  // Messages with levels lower than this are omitted
#else
  LogLevel m_minLogLevel = LogLevel::eSTATS;  // Messages with levels lower than this are omitted
#endif
  std::ofstream        m_logFile;                    // Output file stream
  bool                 m_logToFile = true;           // Enable file output
  bool                 m_fileFlush = false;          // Whether to flush all prints to the log file
  std::recursive_mutex m_logMutex;                   // Mutex to protect member variables
  LogCallback          m_logCallback  = nullptr;     // Custom log callback
  ShowFlags            m_show         = eSHOW_NONE;  // Default shows no extra information
  bool                 m_breakOnError = true;        // Break on errors by default

  Logger() {}
  ~Logger();
  Logger(const Logger&)            = delete;
  Logger& operator=(const Logger&) = delete;

  void        ensureLogFileIsOpen() noexcept;
  std::string formatString(const char* format, va_list args);
  void        addPrefixes(LogLevel level, std::string& message);
  void        outputToConsoles(LogLevel level, const std::string& message) noexcept;
  void        outputToFile(const std::string& message) noexcept;
  void        outputToCallback(LogLevel level, const std::string& message) noexcept;
  void        breakOnErrors(LogLevel level, const std::string& message) noexcept;
};

}  // namespace nvutils

// Logging macros

#define LOGD(format, ...) nvutils::Logger::getInstance().log(nvutils::Logger::LogLevel::eDEBUG, format, ##__VA_ARGS__)
#define LOGSTATS(format, ...)                                                                                          \
  nvutils::Logger::getInstance().log(nvutils::Logger::LogLevel::eSTATS, format, ##__VA_ARGS__)
#define LOGOK(format, ...) nvutils::Logger::getInstance().log(nvutils::Logger::LogLevel::eOK, format, ##__VA_ARGS__)
#define LOGI(format, ...) nvutils::Logger::getInstance().log(nvutils::Logger::LogLevel::eINFO, format, ##__VA_ARGS__)
#define LOGW(format, ...) nvutils::Logger::getInstance().log(nvutils::Logger::LogLevel::eWARNING, format, ##__VA_ARGS__)
#define LOGE(format, ...) nvutils::Logger::getInstance().log(nvutils::Logger::LogLevel::eERROR, format, ##__VA_ARGS__)

// std::print-style macros and functions.
// These are not allowed in CUDA source files, because cudafe++ transforms
// Unicode code points to octal code units. (nvbug 4839128)
#if defined(NVLOGGER_ENABLE_FMT) && !defined(__CUDACC__)
#include <fmt/format.h>
// This macro catches exceptions from fmt::format. This gives us compile-time
// checking, while still making these functions have the same noexcept
// semantics as nvprintf.
#define PRINT_CATCH(lvl, fmtstr, ...)                                                                                   \
  {                                                                                                                     \
    try                                                                                                                 \
    {                                                                                                                   \
      nvutils::Logger::getInstance().log(lvl, "%s", fmt::format(fmtstr, __VA_ARGS__).c_str());                          \
    }                                                                                                                   \
    catch(const std::exception&)                                                                                        \
    {                                                                                                                   \
      nvutils::Logger::getInstance().log(nvutils::Logger::LogLevel::eERROR, "PRINT_CATCH: Could not format string.\n"); \
    }                                                                                                                   \
  }

#define PRINTD(fmtstr, ...) PRINT_CATCH(nvutils::Logger::LogLevel::eDEBUG, fmtstr, __VA_ARGS__)
#define PRINTSTATS(fmtstr, ...) PRINT_CATCH(nvutils::Logger::LogLevel::eSTATS, fmtstr, __VA_ARGS__)
#define PRINTOK(fmtstr, ...) PRINT_CATCH(nvutils::Logger::LogLevel::eOK, fmtstr, __VA_ARGS__)
#define PRINTI(fmtstr, ...) PRINT_CATCH(nvutils::Logger::LogLevel::eINFO, fmtstr, __VA_ARGS__)
#define PRINTW(fmtstr, ...) PRINT_CATCH(nvutils::Logger::LogLevel::eWARNING, fmtstr, __VA_ARGS__)
#define PRINTE(fmtstr, ...) PRINT_CATCH(nvutils::Logger::LogLevel::eERROR, fmtstr, __VA_ARGS__)
#endif  // defined(NVLOGGER_ENABLE_FMT) && !defined(__CUDACC__)


#endif  // LOGGER_HPP
