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

#include "parameter_parser.hpp"
#include "profiler.hpp"

#include <filesystem>

namespace nvutils {

// The ParameterSequencer class allows parsing a parameter file
// in sequences. Each sequence starts with the "SEQUENCE" keyword
// followed by a string or file end
//
// Example script file
//
//   ```
//   # during `ParameterSequencer::prepareFrame` we evaluate all settings
//   # until the next SEQUENCE keyword
//   SEQUENCE "blubb"
//   --modelfile "somefile.gltf"
//   --mysetting 1
//
//   SEQUENCE "blah"
//   --mysetting 0
//
//   ```
//
// Each sequence is measured for a length of `sequenceFrameCount` many frames,
// and the profiler uses a window of `profilerAverageCount` many frames for averaging.
// After each sequence a report is generated from the profiler and logged via `LogLevel::eSTATS`.

class ParameterSequencer
{
public:
  // Information passed to InitInfo::postCallbacks about the sequence that
  // just ran.
  struct State
  {
    uint32_t    index = 0;    // Sequence index within script
    std::string description;  // Sequence description within script
  };

  class InitInfo
  {
  public:
    // the parameters sequence is provided either as content string or as filename.
    std::string           scriptContent;   // parameter: "sequencestring"
    std::filesystem::path scriptFilename;  // parameter: "sequencefile"

    // registers the above using this
    void registerScriptParameters(ParameterRegistry& registry, ParameterParser& parser);

    bool hasScript() const { return !scriptFilename.empty() || !scriptContent.empty(); }

    // internal parameters that are allowed to change per sequence
    // these are registered and added to the `parameterParser` and `parameterRegistry`
    // at `ParameterSequencer::init` time.

    // how many frames each sequence is running
    uint32_t sequenceFrameCount = 128;  // parameter: "sequenceframes"
    // how many frames to delay measuring frames in profiler
    uint32_t profilerResetFrameCount = 8;  // parameter: "sequenceresetframes"
    // how many last N frames to average (0 averages entire sequence)
    uint32_t profilerAverageCount = ProfilerTimeline::MAX_LAST_FRAMES;  // parameter: "sequenceaverages"

    // mandatory, the scripts are parsed using this parser
    ParameterParser* parameterParser{};
    // mandatory, the internal parameters are registered here
    ParameterRegistry* parameterRegistry{};

    // optional, after each sequence we print the results provided from this manager
    ProfilerManager* profilerManager{};

    // To get called after a new benchmark setting.
    // The input to each function is the description of the previous benchmark.
    std::vector<std::function<void(const State&)>> postCallbacks;
  };

  // The script is parsed using the provided `parameterParser` (must be kept alive).
  // The sequence profiling results are queried from the provided `profilerManager` (must be kept alive).
  // It is possible to run without a provided profilerManager and then simply execute the script.
  //
  // Pointers must be kept alive.
  //
  // Returns `true` if the script content or file was provided and was loaded successfully and the sequence can be run.
  bool init(const InitInfo& initInfo);

  // The user must continue to generate frames until this is true
  bool isCompleted() const { return m_completed; }

  // The main function to call every frame while the sequencer wasn't completed.
  // This function triggers the parameter parsing of the next sequence within the sequence
  // script.
  // When a previous sequence completed it will query the profiler for a string of the statistics in
  // `full` detail mode and the log them via `LogLevel::eSTATS
  //
  // Returns `true` if the sequences were completed and no more frames are required.
  bool prepareFrame();

protected:
  bool     m_completed = true;
  InitInfo m_info;

  // tokenized version of the script file
  ParameterParser::Tokenized m_tokenizedScript;

  // start argument for the next sequence within m_tokenizedScript
  size_t m_currentArgument = 0;

  // current frame count
  uint32_t m_frameCount = 0;

  // Info about the current sequence
  State m_sequenceState = {};
};
}  // namespace nvutils
