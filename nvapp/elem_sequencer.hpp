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

#include <nvutils/parameter_sequencer.hpp>

#include "application.hpp"

namespace nvapp {

// Element that contains a `ParameterSequencer` and advances it
// if applicable.

class ElementSequencer : public nvapp::IAppElement
{
public:
  ElementSequencer(const nvutils::ParameterSequencer::InitInfo& sequencerInfo)
      : m_sequencerInfo(sequencerInfo)
  {
  }
  virtual void onAttach(nvapp::Application* app) override
  {
    m_app         = app;
    m_doSequences = m_sequencer.init(m_sequencerInfo);
  }
  virtual void onPreRender() override
  {
    if(m_doSequences)
    {
      bool finished = m_sequencer.prepareFrame();
      if(finished)
      {
        m_app->close();
      }
    }
  }

private:
  nvutils::ParameterSequencer::InitInfo m_sequencerInfo;
  nvutils::ParameterSequencer           m_sequencer;
  nvapp::Application*                   m_app         = nullptr;
  bool                                  m_doSequences = false;
};
}  // namespace nvapp
