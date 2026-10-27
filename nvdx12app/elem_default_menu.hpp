/*
 * Copyright (c) 2022-2026, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.  All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Default menu of the application: File/Exit and View/V-Sync

#pragma once


#include "application.hpp"

// Ex:
//   app->addElement(std::make_shared<nvdx12app::ElementDefaultMenu>());
//

namespace nvdx12app {


/*-------------------------------------------------------------------------------------------------
# class nvdx12app::ElementDefaultMenu

>  This class is an element of the application that is responsible for the default menu of the application. It is using the `ImGui` library to create a menu with File/Exit and View/V-Sync.

Mirrors nvapp::ElementDefaultMenu, for nvdx12app::Application.

To use this class, you need to add it to the `nvdx12app::Application` using the `addElement` method.

-------------------------------------------------------------------------------------------------*/

class ElementDefaultMenu : public nvdx12app::IAppElement
{
public:
  void onAttach(nvdx12app::Application* app) override;
  void onUIMenu() override;

private:
  nvdx12app::Application* m_app{nullptr};
};

}  // namespace nvdx12app
