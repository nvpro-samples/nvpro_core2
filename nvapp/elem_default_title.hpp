/*
 * Copyright (c) 2022-2025, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

// Various Application utilities
// - Display a menu with File/Quit
// - Display basic information in the window title

#pragma once

#include "application.hpp"

// Use:
//  include this file at the end of all other includes,
//  and add engines
//
// Ex:
//   app->addEngine(std::make_shared<nvappDefaultMenu>());
//

namespace nvapp {

/*-------------------------------------------------------------------------------------------------
# class nvapp::ElementDefaultWindowTitle

>  This class is an element of the application that is responsible for the default window title of the application. It is using the `GLFW` library to set the window title with the application name, the size of the window and the frame rate.

To use this class, you need to add it to the `nvapp::Application` using the `addElement` method.

-------------------------------------------------------------------------------------------------*/

class ElementDefaultWindowTitle : public nvapp::IAppElement
{
public:
  ElementDefaultWindowTitle(const std::string& prefix = "", const std::string& suffix = "");

  void onAttach(nvapp::Application* app) override;

  void onUIRender() override;
  void setPrefix(const std::string& str);
  void setSuffix(const std::string& str);

private:
  nvapp::Application* m_app{nullptr};
  float               m_dirtyTimer{0.0F};
  std::string         m_prefix;
  std::string         m_suffix;
};

}  // namespace nvapp
