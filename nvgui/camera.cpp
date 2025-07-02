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


#include <fstream>
#include <sstream>
#include <fmt/format.h>

#include <imgui/imgui.h>
#include <tinygltf/json.hpp>
#include <nvutils/camera_manipulator.hpp>
#include <nvutils/logger.hpp>
#include <nvutils/file_operations.hpp>
#include <nvgui/property_editor.hpp>
#include <nvgui/tooltip.hpp>

#include "camera.hpp"

#ifdef _MSC_VER
#define SAFE_SSCANF sscanf_s
#else
#define SAFE_SSCANF sscanf
#endif


using nlohmann::json;

namespace PE = nvgui::PropertyEditor;

//--------------------------------------------------------------------------------------------------
// Holds all saved cameras in a vector of Cameras
// - The first camera in the list is the HOME camera, the one that was set before this is called.
// - The update function will check if something has changed and will save the JSON to disk, only
//  once in a while.
// - Adding a camera will be added only if it is different from all other saved cameras
// - load/save Setting will load next to the executable, the "jsonFilename" + ".json"
//
struct CameraPresetManager
{
  CameraPresetManager() {}
  ~CameraPresetManager() {};

  static CameraPresetManager& getInstance()
  {
    static CameraPresetManager instance;
    return instance;
  }

  // update setting, load or save
  void update(std::shared_ptr<nvutils::CameraManipulator> cameraManip)
  {
    // Push the HOME camera and load default setting
    if(m_cameras.empty())
    {
      m_cameras.emplace_back(cameraManip->getCamera());
    }
    if(m_doLoadSetting)
      loadSetting(cameraManip);

    // Save settings (with a delay after the last modification, so we don't spam disk too much)
    auto& IO = ImGui::GetIO();
    if(m_settingsDirtyTimer > 0.0f)
    {
      m_settingsDirtyTimer -= IO.DeltaTime;
      if(m_settingsDirtyTimer <= 0.0f)
      {
        saveSetting(cameraManip);
        m_settingsDirtyTimer = 0.0f;
      }
    }
  }

  // Clear all cameras except the HOME
  void removedSavedCameras()
  {
    if(m_cameras.size() > 1)
      m_cameras.erase(m_cameras.begin() + 1, m_cameras.end());
  }

  void setCameraJsonFile(const std::filesystem::path& filename)
  {
    std::filesystem::path jsonFile = nvutils::getExecutablePath().parent_path() / filename.filename();
    jsonFile.replace_extension(".json");
    m_jsonFilename  = std::move(jsonFile);
    m_doLoadSetting = true;
    removedSavedCameras();
  }


  void setHomeCamera(const nvutils::CameraManipulator::Camera& camera)
  {
    if(m_cameras.empty())
      m_cameras.resize(1);
    m_cameras[0] = camera;
  }

  // Adding a camera only if it different from all the saved ones
  void addCamera(const nvutils::CameraManipulator::Camera& camera)
  {
    bool unique = true;
    for(const auto& c : m_cameras)
    {
      if(c == camera)
      {
        unique = false;
        break;
      }
    }
    if(unique)
    {
      m_cameras.emplace_back(camera);
      markIniSettingsDirty();
    }
  }

  // Removing a camera
  void removeCamera(int delete_item)
  {
    m_cameras.erase(m_cameras.begin() + delete_item);
    markIniSettingsDirty();
  }

  void markIniSettingsDirty()
  {
    if(m_settingsDirtyTimer <= 0.0f)
      m_settingsDirtyTimer = 0.1f;
  }

  template <typename T>
  bool getJsonValue(const json& j, const std::string& name, T& value)
  {
    auto fieldIt = j.find(name);
    if(fieldIt != j.end())
    {
      value = (*fieldIt);
      return true;
    }
    LOGW("Could not find JSON field %s", name.c_str());
    return false;
  }

  template <typename T>
  bool getJsonArray(const json& j, const std::string& name, T& value)
  {
    auto fieldIt = j.find(name);
    if(fieldIt != j.end())
    {
      value = T((*fieldIt).begin(), (*fieldIt).end());
      return true;
    }
    LOGW("Could not find JSON field %s", name.c_str());
    return false;
  }


  void loadSetting(std::shared_ptr<nvutils::CameraManipulator> cameraM)
  {
    if(m_jsonFilename.empty())
    {
      // Default name
      m_jsonFilename = nvutils::getExecutablePath().replace_extension(".json");
    }

    if(m_cameras.empty() || m_doLoadSetting == false)
      return;

    try
    {
      m_doLoadSetting = false;

      std::ifstream i(m_jsonFilename);
      if(!i.is_open())
        return;

      // Parsing the file
      json j;
      i >> j;

      // Temp
      int                iVal;
      float              fVal;
      std::vector<float> vfVal;

      // Settings
      if(getJsonValue(j, "mode", iVal))
        cameraM->setMode(static_cast<nvutils::CameraManipulator::Modes>(iVal));
      if(getJsonValue(j, "speed", fVal))
        cameraM->setSpeed(fVal);
      if(getJsonValue(j, "anim_duration", fVal))
        cameraM->setAnimationDuration(fVal);

      // All cameras
      std::vector<json> cc;
      getJsonArray(j, "cameras", cc);
      for(auto& c : cc)
      {
        nvutils::CameraManipulator::Camera camera;
        if(getJsonArray(c, "eye", vfVal))
          camera.eye = {vfVal[0], vfVal[1], vfVal[2]};
        if(getJsonArray(c, "ctr", vfVal))
          camera.ctr = {vfVal[0], vfVal[1], vfVal[2]};
        if(getJsonArray(c, "up", vfVal))
          camera.up = {vfVal[0], vfVal[1], vfVal[2]};
        if(getJsonValue(c, "fov", fVal))
          camera.fov = fVal;
        m_cameras.emplace_back(camera);
      }
      i.close();
    }
    catch(...)
    {
      return;
    }
  }

  void saveSetting(std::shared_ptr<nvutils::CameraManipulator>& cameraManip)
  {
    if(m_jsonFilename.empty())
      return;

    try
    {
      json j;
      j["mode"]          = cameraManip->getMode();
      j["speed"]         = cameraManip->getSpeed();
      j["anim_duration"] = cameraManip->getAnimationDuration();

      // Save all extra cameras
      json cc = json::array();
      for(size_t n = 1; n < m_cameras.size(); n++)
      {
        auto& c   = m_cameras[n];
        json  jo  = json::object();
        jo["eye"] = std::vector<float>{c.eye.x, c.eye.y, c.eye.z};
        jo["up"]  = std::vector<float>{c.up.x, c.up.y, c.up.z};
        jo["ctr"] = std::vector<float>{c.ctr.x, c.ctr.y, c.ctr.z};
        jo["fov"] = c.fov;
        cc.push_back(jo);
      }
      j["cameras"] = cc;

      std::ofstream o(m_jsonFilename);
      if(o.is_open())
      {
        o << j.dump(2) << std::endl;
        o.close();
      }
    }
    catch(const std::exception& e)
    {
      LOGE("Could not save camera settings to %s: %s\n", nvutils::utf8FromPath(m_jsonFilename).c_str(), e.what());
    }
  }

  // Holds all cameras. [0] == HOME
  std::vector<nvutils::CameraManipulator::Camera> m_cameras{};
  float                                           m_settingsDirtyTimer{0};
  std::filesystem::path                           m_jsonFilename{};
  bool                                            m_doLoadSetting{true};
};


//--------------------------------------------------------------------------------------------------
// Display the values of the current camera: position, center, up and FOV
//
static void CurrentCameraTab(std::shared_ptr<nvutils::CameraManipulator> cameraM,
                             nvutils::CameraManipulator::Camera&         camera,
                             bool&                                       changed,
                             bool&                                       instantSet)
{
  bool y_is_up = camera.up.y == 1;

  if(PE::begin())
  {
    PE::InputFloat3("Eye", &camera.eye.x, "%.5f", 0, "Position of the Camera");
    changed |= ImGui::IsItemDeactivatedAfterEdit();
    PE::InputFloat3("Center", &camera.ctr.x, "%.5f", 0, "Center of camera interest");
    changed |= ImGui::IsItemDeactivatedAfterEdit();
    PE::InputFloat3("Up", &camera.up.x, "%.5f", 0, "Up vector interest");
    changed |= ImGui::IsItemDeactivatedAfterEdit();
    if(PE::entry("Y is UP", [&] { return ImGui::Checkbox("##Y", &y_is_up); }, "Is Y pointing up or Z?"))
    {
      camera.up = y_is_up ? glm::vec3(0, 1, 0) : glm::vec3(0, 0, 1);
      changed   = true;
    }
    if(glm::length(camera.up) < 0.0001f)
    {
      camera.up = y_is_up ? glm::vec3(0, 1, 0) : glm::vec3(0, 0, 1);
      changed   = true;
    }
    if(PE::SliderFloat("FOV", &camera.fov, 1.F, 179.F, "%.1f deg", ImGuiSliderFlags_Logarithmic, "Field of view in degrees"))
    {
      instantSet = true;
      changed    = true;
    }

    if(PE::treeNode("Clip planes"))
    {
      glm::vec2 clip = cameraM->getClipPlanes();
      PE::InputFloat("Near", &clip.x);
      changed |= ImGui::IsItemDeactivatedAfterEdit();
      PE::InputFloat("Far", &clip.y);
      changed |= ImGui::IsItemDeactivatedAfterEdit();
      PE::treePop();
      cameraM->setClipPlanes(clip);
    }

    if(cameraM->isAnimated())
    {
      // Ignoring any changes while the camera is moving to the goal.
      // The camera has to be in the new position before setting a new value.
      changed = false;
    }

    ImGui::TableNextRow();
    ImGui::TableNextColumn();

    ImGui::TextDisabled("(?)");
    nvgui::tooltip(cameraM->getHelp().c_str(), false, 0.0f);
    ImGui::TableNextColumn();
    if(ImGui::SmallButton("Copy"))
    {
      std::string text = fmt::format("{{{:.5f}, {:.5f}, {:.5f}}}, {{{:.5f}, {:.5f}, {:.5f}}}, {{{:.5f}, {:.5f}, {:.5f}}}",
                                     camera.eye.x, camera.eye.y, camera.eye.z, camera.ctr.x, camera.ctr.y, camera.ctr.z,
                                     camera.up.x, camera.up.y, camera.up.z);
      ImGui::SetClipboardText(text.c_str());
    }
    nvgui::tooltip("Copy to the clipboard the current camera: {eye}, {ctr}, {up}");
    ImGui::SameLine();
    const char* pPastedString;
    if(ImGui::SmallButton("Paste") && (pPastedString = ImGui::GetClipboardText()))
    {
      float val[9]{};
      int   result = SAFE_SSCANF(pPastedString, "{%f, %f, %f}, {%f, %f, %f}, {%f, %f, %f}", &val[0], &val[1], &val[2],
                                 &val[3], &val[4], &val[5], &val[6], &val[7], &val[8]);
      if(result == 9)  // 9 value properly scanned
      {
        camera.eye = glm::vec3{val[0], val[1], val[2]};
        camera.ctr = glm::vec3{val[3], val[4], val[5]};
        camera.up  = glm::vec3{val[6], val[7], val[8]};
        changed    = true;
      }
    }
    nvgui::tooltip("Paste from the clipboard the current camera: {eye}, {ctr}, {up}");
    PE::end();
  }
}

//--------------------------------------------------------------------------------------------------
// Display buttons for all saved cameras. Allow to create and delete saved cameras
//
static void SavedCameraTab(std::shared_ptr<nvutils::CameraManipulator> cameraM, nvutils::CameraManipulator::Camera& camera, bool& changed)
{
  // Dummy
  ImVec2      button_sz(50, 30);
  ImGuiStyle& style             = ImGui::GetStyle();
  int         buttons_count     = (int)CameraPresetManager ::getInstance().m_cameras.size();
  float       window_visible_x2 = ImGui::GetWindowPos().x + ImGui::GetWindowContentRegionMax().x;

  // The HOME camera button, different from the other ones
  if(ImGui::Button("Home", ImVec2(ImGui::GetWindowContentRegionMax().x, 50)))
  {
    camera  = CameraPresetManager ::getInstance().m_cameras[0];
    changed = true;
  }
  nvgui::tooltip("Reset the camera to its origin");

  // Display all the saved camera in an array of buttons
  int delete_item = -1;
  for(int n = 1; n < buttons_count; n++)
  {
    ImGui::PushID(n);
    {
      std::string label = fmt::format("# {}", n);
      if(ImGui::Button(label.c_str(), button_sz))
      {
        camera  = CameraPresetManager ::getInstance().m_cameras[n];
        changed = true;
      }
    }

    // Middle click to delete a camera
    if(ImGui::IsItemHovered() && ImGui::GetIO().MouseClicked[ImGuiMouseButton_Middle])
      delete_item = n;

    // Displaying the position of the camera when hovering the button
    {
      std::string label =
          fmt::format("Pos: {:.5f}, {:.5f}, {:.5f}", CameraPresetManager ::getInstance().m_cameras[n].eye.x,
                      CameraPresetManager ::getInstance().m_cameras[n].eye.y,
                      CameraPresetManager ::getInstance().m_cameras[n].eye.z);
      nvgui::tooltip(label.c_str());
    }

    // Wrapping all buttons (see ImGUI Demo)
    float last_button_x2 = ImGui::GetItemRectMax().x;
    float next_button_x2 = last_button_x2 + style.ItemSpacing.x + button_sz.x;  // Expected position if next button was on same line
    if(n + 1 < buttons_count && next_button_x2 < window_visible_x2)
      ImGui::SameLine();

    ImGui::PopID();
  }

  // Adding a camera button
  if(ImGui::Button("+"))
  {
    CameraPresetManager ::getInstance().addCamera(cameraM->getCamera());
  }
  nvgui::tooltip("Add a new saved camera");
  ImGui::SameLine();
  ImGui::TextDisabled("(?)");
  nvgui::tooltip("Middle-click a camera to delete it", false, 0.0f);

  // Remove element
  if(delete_item > 0)
  {
    CameraPresetManager ::getInstance().removeCamera(delete_item);
  }
}

//--------------------------------------------------------------------------------------------------
// This holds all camera setting, like the speed, the movement mode, transition duration
//
static void CameraExtraTab(std::shared_ptr<nvutils::CameraManipulator> cameraM, bool& changed)
{
  // Navigation Mode
  if(PE::begin())
  {
    auto mode     = cameraM->getMode();
    auto speed    = cameraM->getSpeed();
    auto duration = static_cast<float>(cameraM->getAnimationDuration());

    changed |= PE::entry(
        "Navigation",
        [&] {
          int rmode = static_cast<int>(mode);
          changed |= ImGui::RadioButton("Examine", &rmode, nvutils::CameraManipulator::Examine);
          nvgui::tooltip("The camera orbit around a point of interest");
          changed |= ImGui::RadioButton("Fly", &rmode, nvutils::CameraManipulator::Fly);
          nvgui::tooltip("The camera is free and move toward the looking direction");
          changed |= ImGui::RadioButton("Walk", &rmode, nvutils::CameraManipulator::Walk);
          nvgui::tooltip("The camera is free but stay on a plane");
          cameraM->setMode(static_cast<nvutils::CameraManipulator::Modes>(rmode));
          return changed;
        },
        "Camera Navigation Mode");

    changed |= PE::SliderFloat("Speed", &speed, 0.01F, 10.0F, "%.3f", 0, "Changing the default speed movement");
    changed |= PE::SliderFloat("Transition", &duration, 0.0F, 2.0F, "%.3f", 0, "Nb seconds to move to new position");

    cameraM->setSpeed(speed);
    cameraM->setAnimationDuration(duration);

    if(changed)
    {
      CameraPresetManager ::getInstance().markIniSettingsDirty();
    }

    PE::end();
  }
}

//--------------------------------------------------------------------------------------------------
// Display the camera eye and center of interest position of the camera (nvutils::CameraManipulator)
// Allow also to modify the field-of-view (FOV)
// And basic control information is displayed
bool nvgui::CameraWidget(std::shared_ptr<nvutils::CameraManipulator> cameraManip)
{
  assert(cameraManip && "CameraManipulator is not set");
  bool changed{false};
  bool instantSet{false};
  auto camera = cameraManip->getCamera();

  // Updating the camera manager
  CameraPresetManager::getInstance().update(cameraManip);

  // Starting UI
  if(ImGui::BeginTabBar("CameraManipulator"))
  {
    if(ImGui::BeginTabItem("Current"))
    {
      CurrentCameraTab(cameraManip, camera, changed, instantSet);
      ImGui::EndTabItem();
    }

    if(ImGui::BeginTabItem("Cameras"))
    {
      SavedCameraTab(cameraManip, camera, changed);
      ImGui::EndTabItem();
    }

    if(ImGui::BeginTabItem("Extra"))
    {
      CameraExtraTab(cameraManip, changed);
      ImGui::EndTabItem();
    }
    ImGui::EndTabBar();
  }

  // Apply the change back to the camera
  if(changed)
  {
    cameraManip->setCamera(camera, instantSet);
  }

  return changed;
}

void nvgui::SetCameraJsonFile(const std::filesystem::path& filename)
{
  CameraPresetManager::getInstance().setCameraJsonFile(filename);
}

void nvgui::SetHomeCamera(const nvutils::CameraManipulator::Camera& camera)
{
  CameraPresetManager::getInstance().setHomeCamera(camera);
}

void nvgui::AddCamera(const nvutils::CameraManipulator::Camera& camera)
{
  CameraPresetManager::getInstance().addCamera(camera);
}
