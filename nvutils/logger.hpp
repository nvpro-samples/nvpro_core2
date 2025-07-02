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
 * Logger class for handling logging with different log levels.
 * 
 * Usage:
 * 
 * // Get the logger instance
 * Logger& logger = Logger::getInstance();
 * 
 * // Set the log level
 * logger.setLogLevel(Logger::LogLevel::eINFO);
 * 
 * // Set the information to show in the log
 * logger.setShowFlags(Logger::eSHOW_TIME | Logger::eSHOW_LEVEL);
 * 
 * // Set the output file : default is the name of the executable with .txt extension
 * logger.setOutputFile("logfile.txt");
 * 
 * // Enable or disable file output
 * logger.enableFileOutput(true);
 * 
 * // Set a custom log callback
 * logger.setLogCallback([](Logger::LogLevel level, const std::string& message) {
 *     std::cout << "Custom Log: " << message << std::endl;
 * });
 * 
 * // Log messages
 * LOGD("This is a debug message.");
 * LOGI("This is an info message.");
 * LOGW("This is a warning message.");
 * LOGE("This is an error message with id: %d.", integerValue);
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

  enum ShowFlags
  {
    eSHOW_NONE  = 0,
    eSHOW_TIME  = 1 << 0,
    eSHOW_LEVEL = 1 << 1
  };

  using LogCallback = std::function<void(LogLevel, const std::string&)>;

  // Get the logger instance
  static Logger& getInstance()
  {
    static Logger instance;
    return instance;
  }

  // Set the minimum log level
  void setLogLevel(LogLevel level) { m_minLogLevel = level; }

  // Set the information to show in the log
  void setShowFlags(ShowFlags flags) { m_show = flags; }

  // Set the output file
  void setOutputFile(const std::filesystem::path& filename);

  // Enable or disable file output
  void enableFileOutput(bool enable) { m_logToFile = enable; }

  // Set a custom log callback
  void setLogCallback(LogCallback callback);

  // Log a message
  void log(LogLevel level,
#ifdef _MSC_VER
           _Printf_format_string_  // Enable MSVC /analyze warnings about incorrect format strings
#endif
           const char* format,
           ...)
#if defined(__GNUC__)
      __attribute__((format(printf, 3, 4)))  // Enable GCC + clang warnings about incorrect format strings
#endif
      ;

  void breakOnError(bool enable) { m_breakOnError = enable; }

private:
#ifdef DEBUG
  LogLevel m_minLogLevel = LogLevel::eDEBUG;  // Messages with levels lower than this are omitted
#else
  LogLevel m_minLogLevel = LogLevel::eSTATS;  // Messages with levels lower than this are omitted
#endif
  std::ofstream m_logFile;                    // Output file stream
  bool          m_logToFile = true;           // Enable file output
  std::mutex    m_logMutex;                   // Mutex to protect the log file and callback
  LogCallback   m_logCallback  = nullptr;     // Custom log callback
  ShowFlags     m_show         = eSHOW_NONE;  // Default shows no extra information
  bool          m_breakOnError = true;        // Break on errors by default

  Logger() {}
  ~Logger();
  Logger(const Logger&)            = delete;
  Logger& operator=(const Logger&) = delete;

  std::string formatString(const char* format, va_list args);
  const char* logLevelToString(LogLevel level);
  void        outputToConsole(LogLevel level, const std::string& message);
  std::string currentTime();
  void        ensureLogFileIsOpen();
  void        addPrefixes(std::ostringstream& logStream, LogLevel level, const std::string& message);
  void        outputToFile(const std::string& message);
  void        outputToCallback(LogLevel level, const std::string& message);
  void        breakOnErrors(LogLevel level, const std::string& message);
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


#endif  // LOGGER_HPP
