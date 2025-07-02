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

#pragma once

#include <cassert>
#include <span>
#include <filesystem>
#include <unordered_map>
#include <unordered_set>

#include "parameter_registry.hpp"

namespace nvutils {

// This class parses arguments for parameters provided from the
// ParameterRegistry and updates the destination pointers accordingly.
//
// The primary use-case is parsing the command line arguments,
// however, it is possible to load parameters from a config file as well
//
// sample config file
//
//    ```
//    # use `#` as first character in a line to use as comment
//    # parameters from the registry are prefixed with `--` using `ParameterBase::name`
//    --myparameter 123
//    # and `-` prefix using `ParameterBase::shortName`
//    -mp 345
//    # filename parameters support relative filenames relative to the calling
//    # config file, or the base path that is provided to the top level parsing function.
//
//    --configfile "blubb.cfg"
//
//    # also possible to put multiple parameters in a single line
//    --foo test this --bar 1.337
//
//    ```

class ParameterParser
{
public:
  // There are always two internal parameters provided by the parser:
  // --configfile -cf "filename": loads parameters from another file, filename can be relative.
  //   Optionally associate the configfile with extensions, then it can be triggered without lead parameter name.
  // --help -h: prints the description and then all registered parameters
  ParameterParser(const std::string& helpDescription = {}, const std::vector<std::string>& configFileExtensions = {});

  void setHelpDescription(const std::string& helpDescription) { m_helpDescription = helpDescription; }

  // Prints successfully parsed parameter details or errors via `Logger::eINFO` and `Logger::eERROR`
  void setVerbose(bool verbose) { m_verbose = verbose; }

  // Prints the help string to `Logger::eINFO`
  void printHelp() const;

  // Add a parameter from a registry to be included in the parsing.
  // Pointer must be kept alive
  // Silently ignores adding the same pointer again.
  void add(const ParameterBase* parameter);

  // Add all parameters from a registry which pass `(parmeter.info.visibility & visibilityMask) != 0`
  // Pointer must be kept alive
  // Silently ignores adding the same pointer again.
  void add(const ParameterRegistry& registry, uint32_t visibilityMask = ~0u);

  // Parses inputs and writes parameter destination values, returns how many arguments were processed.
  // Terminates early when hitting the `stopKeyword` and then returns the next index after stopKeyword.
  // `filenameBasePath` is prepended to filename parameters that contained relative file names.
  // If `silentUnknown == true` then no errors are printed for unknown arguments.
  size_t parse(std::span<const char* const> args,
               bool                         skipExe,
               const std::filesystem::path& filenameBasePath = {},
               const std::string&           stopKeyword      = {},
               bool                         silentUnknown    = false);

  int parse(int argc, const char** argv, bool skipExe = true, const std::filesystem::path& filenameBasePath = {})
  {
    return int(parse(std::span(argv, argc), skipExe, filenameBasePath));
  }

  int parse(int argc, char** argv, bool skipExe = true, const std::filesystem::path& filenameBasePath = {})
  {
    return int(parse(std::span((const char**)argv, argc), skipExe, filenameBasePath));
  }

  static std::filesystem::path getFilename(const std::filesystem::path& filenameBasePath, const std::filesystem::path& arg);

  // Utility class to load a text file into a tokenized list of arguments that can be parsed.
  // Can also tokenize a provided string. It allows usage of '#' to skip lines when parsing
  // as described for the main class.
  class Tokenized
  {
  public:
    void initFromString(const std::string& content, const std::filesystem::path& filenameBasePath = {});

    bool initFromFile(const std::filesystem::path& filename);

    std::span<const char* const> getArgs(size_t offset = 0) const
    {
      assert(offset < m_args.size());
      return std::span(m_args.data() + offset, m_args.size() - offset);
    }
    std::filesystem::path getFilenameBasePath() const { return m_filenameBasePath; }

  private:
    void processContent();

    std::filesystem::path    m_filenameBasePath;
    std::string              m_content;
    std::vector<const char*> m_args;
  };

private:
  int   parseInt(const ParameterBase& parameter, const char* str, size_t a);
  float parseFloat(const ParameterBase& parameter, const char* str, size_t a);

  const ParameterBase* findViaExtension(const std::string& arg) const;

  // verbose logging
  bool m_verbose{};
  // map with keywords from parameters
  std::unordered_map<std::string, const ParameterBase*> m_keywordMap;

  // vector of FILENAME_EXTENSION parameters
  std::vector<const ParameterBase*> m_parsedExtensions;

  // unique set of pointers added for parsing
  std::unordered_set<const ParameterBase*> m_parsedParameterSet;
  // linear list of added parameters used for printing the help in order
  std::vector<const ParameterBase*> m_parsedParameters;

  // used for the built-in parameters (configfile,help)
  ParameterRegistry m_builtinRegistry;

  // used when printing help
  std::string m_helpDescription;
};

}  // namespace nvutils
