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

#include <algorithm>
#include <cstring>
#include <iostream>
#include <fstream>
#include <sstream>

#include <fmt/ranges.h>

#include "logger.hpp"
#include "parameter_parser.hpp"
#include <nvutils/file_operations.hpp>

namespace nvutils {

ParameterParser::ParameterParser(const std::string& description, const std::vector<std::string>& configFileExtensions)
{
  m_helpDescription = description;

  auto fnConfigLoad = [&](const ParameterBase* const, std::span<const char* const> args, const std::filesystem::path& filenameBasePath) {
    Tokenized tokenized;

    const std::filesystem::path configFile = getFilename(filenameBasePath, args[0]);

    if(tokenized.initFromFile(configFile))
    {
      if(m_verbose)
      {
        Logger::getInstance().log(Logger::eINFO, "parser: configfile %s - start\n", nvutils::utf8FromPath(configFile).c_str());
      }

      size_t maxArgs = tokenized.getArgs().size();
      size_t parsed  = parse(tokenized.getArgs(), false, tokenized.getFilenameBasePath());

      if(m_verbose)
      {
        Logger::getInstance().log(Logger::eINFO, "parser: configfile %s - completed %d of %d\n",
                                  nvutils::utf8FromPath(configFile).c_str(), uint32_t(parsed), uint32_t(maxArgs));
      }
      return true;
    }
    else
    {
      return false;
    }
  };

  auto fnHelp = [&](const ParameterBase* const, std::span<const char* const> args, const std::filesystem::path& filenameBasePath) {
    printHelp();
    exit(1);
    return true;
  };

  add(m_builtinRegistry.addCustom(
      {
          .name      = "configfile",
          .help      = "(string) - Parses provided config file. Relative filenames are allowed.",
          .shortName = "cf",
      },
      1, fnConfigLoad, configFileExtensions));
  add(m_builtinRegistry.addCustom(
      {
          .name      = "help",
          .help      = "() - Prints all known parameter options registered to the parser.",
          .shortName = "h",
      },
      0, fnHelp));
}

void ParameterParser::printHelp() const
{
  static constexpr int MAX_LINE_WIDTH = 60;

  // Print the general description.
  if(!m_helpDescription.empty())
    Logger::getInstance().log(Logger::eINFO, "%s\n", m_helpDescription.c_str());

  // Find the argument with the longest combined flag length (in order to align the help messages).
  uint32_t maxFlagLength = 0;

  for(const ParameterBase* parameter : m_parsedParameters)
  {
    uint32_t flagLength = static_cast<uint32_t>(parameter->info.name.size() + 2);  // +2 for "--"
    if(!parameter->info.shortName.empty())
      flagLength += +2 + static_cast<uint32_t>(parameter->info.shortName.size() + 1);  // 2 + for ", " and +1 for "-"

    maxFlagLength = std::max(maxFlagLength, flagLength);
  }

  // Now print each argument.
  for(const ParameterBase* parameter : m_parsedParameters)
  {

    std::string flags = "--" + parameter->info.name;
    if(!parameter->info.shortName.empty())
      flags += ", -" + parameter->info.shortName;

    std::stringstream sstr;
    sstr << std::left << std::setw(maxFlagLength) << flags;


    std::string help = parameter->getTypeString();
    if(parameter->extensions.size())
    {
      help += fmt::format("[{}]", parameter->extensions);
    }

    if(parameter->info.help.size())
    {
      help += ": " + parameter->info.help;
    }

    // Print the help for each argument. This is a bit more involved since we do line wrapping for long descriptions.
    size_t spacePos  = 0;
    size_t lineWidth = 0;
    while(spacePos != std::string::npos)
    {
      size_t nextspacePos = help.find_first_of(' ', spacePos + 1);
      sstr << help.substr(spacePos, nextspacePos - spacePos);
      lineWidth += nextspacePos - spacePos;
      spacePos = nextspacePos;

      if(lineWidth > MAX_LINE_WIDTH)
      {
        Logger::getInstance().log(Logger::eINFO, "%s\n", sstr.str().c_str());
        sstr = std::stringstream();
        if(maxFlagLength > 0)
        {
          sstr << std::left << std::setw(static_cast<std::streamsize>(maxFlagLength) - 1) << " ";
        }
        else
        {
          sstr << " ";
        }
        lineWidth = 0;
      }
    }
  }
}

void ParameterParser::add(const ParameterBase* parameter)
{
  if(m_parsedParameterSet.find(parameter) != m_parsedParameterSet.end())
  {
    return;
  }

  auto inserted = m_keywordMap.insert({"--" + parameter->info.name, parameter});
  assert(inserted.second);
  if(!parameter->info.shortName.empty())
  {
    inserted = m_keywordMap.insert({"-" + parameter->info.shortName, parameter});
    assert(inserted.second);
  }

  if(parameter->extensions.size())
  {
    m_parsedExtensions.push_back(parameter);
  }

  m_parsedParameters.push_back(parameter);
  m_parsedParameterSet.insert(parameter);
}

void ParameterParser::add(const ParameterRegistry& registry, uint32_t visibilityMask)
{
  auto parameters = registry.getParameters();
  for(const auto& param : parameters)
  {
    if(param->info.visibility & visibilityMask)
      add(param);
  }
}

// internal safe parsing of string to int, exit(1) on error
int ParameterParser::parseInt(const ParameterBase& parameter, const char* str, size_t a)
{
  try
  {
    int value = std::stoi(std::string(str));
    return value;
  }
  catch(const std::invalid_argument&)
  {
    Logger::getInstance().log(Logger::eERROR, "parser: %2d-%2d: --%s invalid parameter value \"%s\", not an integer \n",
                              uint32_t(a), uint32_t(a + parameter.argCount), parameter.info.name.c_str(), str);
  }
  catch(const std::out_of_range&)
  {
    std::cerr << "Input is out of range for int" << std::endl;
    Logger::getInstance().log(Logger::eERROR, "parser: %2d-%2d: --%s invalid parameter value \"%s\", out of range for int \n",
                              uint32_t(a), uint32_t(a + parameter.argCount), parameter.info.name.c_str(), str);
  }

  printHelp();
  exit(1);
}

// internal safe parsing of string to int, exit(1) on error
float ParameterParser::parseFloat(const ParameterBase& parameter, const char* str, size_t a)
{
  try
  {
    float value = std::stof(std::string(str));
    return value;
  }
  catch(const std::invalid_argument&)
  {
    Logger::getInstance().log(Logger::eERROR, "parser: %2d-%2d: --%s invalid parameter value \"%s\", not a float \n",
                              uint32_t(a), uint32_t(a + parameter.argCount), parameter.info.name.c_str(), str);
  }
  catch(const std::out_of_range&)
  {
    std::cerr << "Input is out of range for int" << std::endl;
    Logger::getInstance().log(Logger::eERROR, "parser: %2d-%2d: --%s invalid parameter value \"%s\", out of range for float \n",
                              uint32_t(a), uint32_t(a + parameter.argCount), parameter.info.name.c_str(), str);
  }

  printHelp();
  exit(1);
}

size_t ParameterParser::parse(std::span<const char* const> args,
                              bool                         skipExe,
                              const std::filesystem::path& filenameBasePathIn,
                              const std::string&           stopKeyword,
                              bool                         silentUnknown)
{
  std::filesystem::path filenameBasePath = filenameBasePathIn;

  if(filenameBasePath.has_extension())
  {
    filenameBasePath = filenameBasePath.parent_path();
  }

  for(size_t a = skipExe ? 1 : 0; a < args.size(); a++)
  {
    // inclusive
    size_t argsLeft = args.size() - a;

    std::string arg(args[a]);
    auto        it = m_keywordMap.find(arg);
    if(it != m_keywordMap.end())
    {
      const ParameterBase& parameter = *it->second;

      // argCount is exclusive of keyword
      if(argsLeft > parameter.argCount)
      {
        bool success = true;

        switch(parameter.type)
        {
          case ParameterBase::Type::BOOL8:
            parameter.destination.b8[0] = parseInt(parameter, args[1 + a], a) ? true : false;
            break;
          case ParameterBase::Type::BOOL8_TRIGGER:
            parameter.destination.b8[0] = parameter.minMaxValues[0].u32[0] ? true : false;
            break;
          case ParameterBase::Type::FLOAT32:
            for(uint32_t i = 0; i < parameter.argCount; i++)
            {
              parameter.destination.f32[i] =
                  std::max(std::min(parseFloat(parameter, args[i + 1 + a], a), parameter.minMaxValues[1].f32[i]),
                           parameter.minMaxValues[0].f32[i]);
            }
            break;
          case ParameterBase::Type::INT8:
            for(uint32_t i = 0; i < parameter.argCount; i++)
            {
              parameter.destination.i8[i] =
                  std::max(std::min(int8_t(parseInt(parameter, args[i + 1 + a], a)), parameter.minMaxValues[1].i8[i]),
                           parameter.minMaxValues[0].i8[i]);
            }
            break;
          case ParameterBase::Type::INT16:
            for(uint32_t i = 0; i < parameter.argCount; i++)
            {
              parameter.destination.i16[i] =
                  std::max(std::min(int16_t(parseInt(parameter, args[i + 1 + a], a)), parameter.minMaxValues[1].i16[i]),
                           parameter.minMaxValues[0].i16[i]);
            }
            break;
          case ParameterBase::Type::INT32:
            for(uint32_t i = 0; i < parameter.argCount; i++)
            {
              parameter.destination.i32[i] =
                  std::max(std::min(parseInt(parameter, args[i + 1 + a], a), parameter.minMaxValues[1].i32[i]),
                           parameter.minMaxValues[0].i32[i]);
            }
            break;
          case ParameterBase::Type::UINT8:
            for(uint32_t i = 0; i < parameter.argCount; i++)
            {
              parameter.destination.u8[i] =
                  std::max(std::min(uint8_t(parseInt(parameter, args[i + 1 + a], a)), parameter.minMaxValues[1].u8[i]),
                           parameter.minMaxValues[0].u8[i]);
            }
            break;
          case ParameterBase::Type::UINT16:
            for(uint32_t i = 0; i < parameter.argCount; i++)
            {
              parameter.destination.u16[i] =
                  std::max(std::min(uint16_t(parseInt(parameter, args[i + 1 + a], a)), parameter.minMaxValues[1].u16[i]),
                           parameter.minMaxValues[0].u16[i]);
            }
            break;
          case ParameterBase::Type::UINT32:
            for(uint32_t i = 0; i < parameter.argCount; i++)
            {
              parameter.destination.u32[i] =
                  std::max(std::min(uint32_t(parseInt(parameter, args[i + 1 + a], a)), parameter.minMaxValues[1].u32[i]),
                           parameter.minMaxValues[0].u32[i]);
            }
            break;
          case ParameterBase::Type::STRING:
            *parameter.destination.string = args[1 + a];
            break;
          case ParameterBase::Type::FILENAME:
            *parameter.destination.filename = getFilename(filenameBasePath, args[1 + a]);
            break;
          case ParameterBase::Type::CUSTOM:
            if(parameter.argCount)
            {
              success = parameter.callbackCustom(&parameter, std::span<const char* const>(&args[1 + a], parameter.argCount),
                                                 filenameBasePath);
            }
            else
            {
              success = parameter.callbackCustom(&parameter, {}, filenameBasePath);
            }
            break;
          default:
            assert(false && "invalid parameter type");
            break;
        }

        if(success && parameter.info.callbackSuccess)
        {
          parameter.info.callbackSuccess(&parameter);
        }

        if(m_verbose && success)
        {
          Logger::getInstance().log(Logger::eINFO, "parser: %2d-%2d: --%s", uint32_t(a),
                                    uint32_t(a + parameter.argCount), parameter.info.name.c_str());
          for(uint32_t i = 0; i < parameter.argCount; i++)
          {
            Logger::getInstance().log(Logger::eINFO, " %s", args[i + 1 + a]);
          }
          Logger::getInstance().log(Logger::eINFO, "\n");
        }

        a += parameter.argCount;
      }
      else
      {
        Logger::getInstance().log(Logger::eERROR, "parser: %d - %d: %s - not enough arguments left\n", uint32_t(a),
                                  uint32_t(a + parameter.argCount), parameter.info.name.c_str());
        printHelp();
        exit(1);

        return args.size();
      }
    }
    else if(arg == stopKeyword)
    {
      return a + 1;
    }
    else
    {
      const ParameterBase* parameter = findViaExtension(arg);
      if(parameter)
      {
        bool success = true;

        if(parameter->type == ParameterBase::Type::FILENAME)
        {
          *parameter->destination.filename = getFilename(filenameBasePath, arg);
        }
        else if(parameter->type == ParameterBase::Type::CUSTOM)
        {
          success = parameter->callbackCustom(parameter, std::span<const char* const>(&args[a], 1), filenameBasePath);
        }
        else
        {
          assert(0 && "invalid parameter type for extension case");
        }

        if(success && parameter->info.callbackSuccess)
        {
          parameter->info.callbackSuccess(parameter);
        }
        if(m_verbose)
        {
          if(success)
          {
            Logger::getInstance().log(Logger::eINFO, "parser: %2d-%2d: --%s", uint32_t(a), uint32_t(a),
                                      parameter->info.name.c_str());
          }
          else
          {
            Logger::getInstance().log(Logger::eERROR, "parser: %2d-%2d: --%s failed\n", uint32_t(a), uint32_t(a),
                                      parameter->info.name.c_str());
            printHelp();
            exit(1);
          }
        }
      }
      else if(!silentUnknown)
      {
        Logger::getInstance().log(Logger::eERROR, "parser: %d: %s - unknown parameter\n", uint32_t(a), arg.c_str());
        printHelp();
        exit(1);
      }
    }
  }

  return args.size();
}

std::filesystem::path ParameterParser::getFilename(const std::filesystem::path& filenameBasePath, const std::filesystem::path& arg)
{
  if(arg.is_relative())
  {
    return filenameBasePath / arg;
  }
  else
  {
    return arg;
  }
}

static bool endsWith(const std::string& str, const std::string& suffix)
{
  return str.size() >= suffix.size() && str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0;
}

const nvutils::ParameterBase* ParameterParser::findViaExtension(const std::string& argin) const
{
  std::string arg = argin;

  std::transform(arg.begin(), arg.end(), arg.begin(), [](unsigned char c) { return std::tolower(c); });

  for(const ParameterBase* parameter : m_parsedExtensions)
  {
    for(const std::string& ext : parameter->extensions)
    {
      if(endsWith(arg, ext))
      {
        return parameter;
      }
    }
  }

  return nullptr;
}

void ParameterParser::Tokenized::initFromString(const std::string& content, const std::filesystem::path& filenameBasePath)
{
  m_args             = {};
  m_content          = content;
  m_filenameBasePath = filenameBasePath;

  processContent();
}

bool ParameterParser::Tokenized::initFromFile(const std::filesystem::path& filename)
{
  m_content = {};
  m_args    = {};

  std::ifstream fileStream(filename);
  if(!fileStream)
  {
    LOGW("Parameter parser could not open file %s", nvutils::utf8FromPath(filename).c_str());
    return false;
  }

  if(filename.has_parent_path())
  {
    m_filenameBasePath = filename.parent_path();
  }

  std::stringstream buffer;
  buffer << fileStream.rdbuf();

  m_content = buffer.str();

  // Remove carriage-return
  std::erase_if(m_content, [](std::string::value_type x) { return x == '\r'; });

  processContent();

  return true;
}

void ParameterParser::Tokenized::processContent()
{
  bool wasSpace  = true;
  bool inQuotes  = false;
  bool inComment = false;
  bool wasQuote  = false;
  bool wasEscape = false;

  for(size_t i = 0; i < m_content.size(); i++)
  {
    char* token     = &m_content[i];
    char  current   = m_content[i];
    bool  isEndline = current == '\n';
    bool  isSpace   = (current == ' ' || current == '\t' || current == '\n');
    bool  isQuote   = current == '"' || current == '\'';
    bool  isComment = current == '#';
    bool  isEscape  = current == '\\';

    if(isEndline && inComment)
    {
      inComment = false;
    }
    if(isComment && !inQuotes)
    {
      m_content[i] = 0;
      inComment    = true;
    }

    if(inComment)
      continue;

    if(inQuotes)
    {
      if(wasEscape && (current == 'n' || current == 't'))
      {
        m_content[i]     = current == 'n' ? '\n' : '\t';
        m_content[i - 1] = ' ';
      }
    }

    if(isQuote)
    {
      inQuotes = !inQuotes;
      // treat as space
      m_content[i] = 0;
      isSpace      = true;
    }
    else if(isSpace)
    {
      // turn space to a terminator
      if(!inQuotes)
      {
        m_content[i] = 0;
      }
    }
    else if(wasSpace && (!inQuotes || wasQuote))
    {
      // start a new arg unless comment
      m_args.push_back(token);
    }

    wasSpace  = isSpace;
    wasQuote  = isQuote;
    wasEscape = isEscape;
  }
}

}  // namespace nvutils

//--------------------------------------------------------------------------------------------------
// Usage example
//--------------------------------------------------------------------------------------------------
static void usage_ParameterParser()
{
  // create registry
  nvutils::ParameterRegistry registry;

  bool     blubb = false;
  uint32_t blah  = 123;

  // register some parameters
  registry.add({"blubb", "triggering this parameter enables blubb"}, &blubb, true);
  registry.add({"blah", "modifies blah, clamped to [0,10]"}, &blah, 0, 10);


  // create parser
  nvutils::ParameterParser parser("my test");

  // add all parameters from the registry.
  parser.add(registry);

  // one can also add parameters individually, from other registries etc.
  nvutils::ParameterRegistry otherRegistry;
  std::filesystem::path      filename;
  const nvutils::Parameter<std::filesystem::path>* filenameParameter = otherRegistry.add({"filename", "loads file"}, &filename);

  // filenames that are relative will be automatically made relative to the `filenameBasePath` provided to the parsing
  // function (default is none, so working directory) or indirectly provided when loaded from a configfile.
  parser.add(filenameParameter);

  // typically parses command line
  {
    // get from main...
    int    argc = 0;
    char** argv = nullptr;

    parser.parse(argc, argv);
  }

  // but can also parse a string through a helper class
  // the --help and --configfile options always exist
  std::string example = "--help --blubb --blah 12 --filename test.jpg";

  nvutils::ParameterParser::Tokenized tokenized;
  tokenized.initFromString(example);

  std::filesystem::path filenameBasePath = "/somedirectory";

  parser.parse(tokenized.getArgs(), false, filenameBasePath);

  // blah would be clamped to 10
  // filename would be set to "/somedirectory/test.jpg"
}
