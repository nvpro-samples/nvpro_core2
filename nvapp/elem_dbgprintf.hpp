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

#pragma once

#include "application.hpp"


namespace nvapp {

/*-------------------------------------------------------------------------------------------------
# class nvvkhl::ElementDbgPrintf

>  This class is an element of the application that is responsible for the debug printf in the shader. It is using the `VK_EXT_debug_printf` extension to print information from the shader.

To use this class, you need to add it to the `nvapp::Application` using the `addElement` method.

  
  Add to main
  - Before creating the Vulkan context 
    
        nvvk::ValidationSettings validation{};
        validation.setPreset(nvvk::ValidationSettings::LayerPresets::eDebugPrintf);
        vkSetup.instanceCreateInfoExt = validation.buildPNextChain();

  - Add the Element to the Application
        
        app->addElement(g_dbgPrintf);

  - In the target application, push the mouse coordinated
    
        m_pushConst.mouseCoord = ElementDbgPrintf::getMouseCoord();

  In the Shader, do:

  - Add the extension
    #extension GL_EXT_debug_printf : enable
  
  - Where to get the information
  
        ivec2 fragCoord = ivec2(floor(gl_FragCoord.xy));
        if(fragCoord == ivec2(pushC.mouseCoord))
            debugPrintfEXT("Value: %f\n", myVal);

-------------------------------------------------------------------------------------------------*/


class ElementDbgPrintf : public nvapp::IAppElement
{
public:
  ElementDbgPrintf() = default;

  // Return the relative mouse coordinates in the window named "Viewport"
  static glm::vec2 getMouseCoord();

  void onAttach(Application* app) override;
  void onDetach() override;

private:
  VkInstance               m_instance     = {};
  VkDebugUtilsMessengerEXT m_dbgMessenger = {};
};

}  // namespace nvapp
