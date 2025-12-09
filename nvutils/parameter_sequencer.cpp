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

#include "logger.hpp"
#include "parameter_sequencer.hpp"

namespace nvutils {


void ParameterSequencer::InitInfo::registerScriptParameters(ParameterRegistry& registry, ParameterParser& parser)
{
  parser.add(registry.add({.name = "sequencefile", .help = "filename for text file containing sequences of parameters to be set."},
                          &scriptFilename));
  parser.add(registry.add({.name = "sequencestring", .help = "string containing sequences of parameters to be set."}, &scriptContent));
}

bool ParameterSequencer::init(const InitInfo& info)
{
  m_info = info;

  if(!m_info.scriptContent.empty())
  {
    assert(m_info.scriptFilename.empty());

    m_tokenizedScript.initFromString(m_info.scriptContent, {});
  }
  else if(!m_info.scriptFilename.empty())
  {
    if(!m_tokenizedScript.initFromFile(m_info.scriptFilename))
    {
      return false;
    }
  }
  else
  {
    return false;
  }

  if(strcmp(m_tokenizedScript.getArgs()[0], "SEQUENCE") != 0)
  {
    return false;
  }
  // skip first SEQUENCE
  m_currentArgument = 1;

  assert(m_info.parameterParser && "Parameter parser must be specified");
  assert(m_info.parameterRegistry && "Parameter registry must be specified");
  ParameterParser&   parser   = *m_info.parameterParser;
  ParameterRegistry& registry = *m_info.parameterRegistry;
  parser.add(registry.add({.name = "sequenceframes", .help = "number of frames to run each parameter sequence"},
                          &m_info.sequenceFrameCount));
  parser.add(registry.add({.name = "sequenceaverages", .help = "number of last frames to use for averaging in the profiler. 0 averages all"},
                          &m_info.profilerAverageCount, 0, nvutils::ProfilerTimeline::MAX_LAST_FRAMES));
  parser.add(registry.add({.name = "sequenceresetframes", .help = "number of frames to delay the reset of the profiler per sequence"},
                          &m_info.profilerResetFrameCount, 0, 8));


  m_frameCount = 0;
  m_completed  = false;

  return true;
}

bool ParameterSequencer::prepareFrame()
{
  if(m_completed)
    return true;

  if((m_frameCount % m_info.sequenceFrameCount) == 0)
  {
    // print old
    if(m_currentArgument > 2)
    {
      std::string statsFrame;
      std::string statsSingle;
      if(m_info.profilerManager)
      {
        m_info.profilerManager->appendPrint(statsFrame, statsSingle, true);
        // print old stats
        Logger::getInstance().log(Logger::eSTATS, "ParameterSequence %d \"%s\" = {\n%s\n%s}\n", m_sequenceState.index,
                                  m_sequenceState.description.c_str(), statsFrame.c_str(), statsSingle.c_str());
      }

      // Callback all registered functions
      for(auto& func : m_info.postCallbacks)
        func(m_sequenceState);

      m_sequenceState.index++;
    }

    // test if done
    m_completed = (m_currentArgument >= m_tokenizedScript.getArgs().size());

    if(!m_completed)
    {
      m_sequenceState.description = m_tokenizedScript.getArgs(m_currentArgument)[0];
      m_currentArgument++;

      auto   args       = m_tokenizedScript.getArgs(m_currentArgument);
      size_t stopOffset = m_info.parameterParser->parse(args, false, m_tokenizedScript.getFilenameBasePath(), "SEQUENCE");

      if(m_info.profilerManager)
      {
        m_info.profilerManager->setFrameAveragingCount(m_info.profilerAverageCount);
        m_info.profilerManager->resetFrameSections(m_info.profilerResetFrameCount);
      }

      m_currentArgument = m_currentArgument + stopOffset;
    }
  }

  m_frameCount++;

  return m_completed;
}


}  // namespace nvutils

//--------------------------------------------------------------------------------------------------
// Usage example
//--------------------------------------------------------------------------------------------------
[[maybe_unused]] static void usage_ParameterSequencer()
{
  // create registry & parser
  nvutils::ParameterRegistry registry;
  nvutils::ParameterParser   parser("my test");

  uint32_t blah = 123;

  // register some parameters
  registry.add({"blah", "modifies blah, clamped to [0,10]"}, &blah, 0, 10);

  nvutils::ParameterSequencer::InitInfo sequencerInfo;
  sequencerInfo.registerScriptParameters(registry, parser);

  // imagine we parse command line to get settings
  {
    // get from main...
    int    argc = 0;
    char** argv = nullptr;

    parser.parse(argc, argv);
  }


  // Here we just hardcode two sequences.
  // Each sequence starts with the SEQUENCE keyword which is followed by a single string argument.
  // That single string is then embedded into the automatic printing of results that are queried from the provided profiler.
  sequencerInfo.scriptContent = "SEQUENCE \"first sequence\" --blah 7 SEQUENCE \"second sequence\" --blah 9";

  // profiler should average over all frames not just last N
  sequencerInfo.profilerAverageCount = 0;
  // one sequence should run for 128 frames
  sequencerInfo.sequenceFrameCount = 128;

  // always need a parser
  sequencerInfo.parameterParser = &parser;
  // and registry of internal parameters
  sequencerInfo.parameterRegistry = &registry;

  // want to log profiling results with the sequencer
  nvutils::ProfilerManager   profilerManager;
  nvutils::ProfilerTimeline* timeline = profilerManager.createTimeline({"primary"});

  sequencerInfo.profilerManager = &profilerManager;

  // optionally, add a function called once each sequence is finished:
  sequencerInfo.postCallbacks.push_back([](const nvutils::ParameterSequencer::State& sequence) {
    LOGI("Finished sequence %d: %s\n", sequence.index, sequence.description.c_str());
  });

  // initialize the sequencer
  nvutils::ParameterSequencer sequencer;
  bool                        doSequences = sequencer.init(sequencerInfo);

  bool renderLoop = true;
  while(renderLoop)
  {
    // could be in onPreRender when using an Element
    timeline->frameAdvance();

    if(doSequences)
    {
      // The `prepareFrame` call will change sequence settings every `128` frames (as defined in the settings above)
      // and will print profiler results from the previous sequence to the global logger with full timer details.
      if(sequencer.prepareFrame())
      {
        // returns true if there is left in the script
        break;
      }
    }

    // handle parameter changes triggered by sequencer preparation or other events
    // mysetup(blah);

    // do render or other processing logic
  }

  profilerManager.destroyTimeline(timeline);
}
