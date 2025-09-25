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
#include <nvgui/fonts.hpp>

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
    for(const nvutils::CameraManipulator::Camera& c : m_cameras)
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
      markJsonSettingsDirty();
    }
  }

  // Removing a camera
  void removeCamera(int delete_item)
  {
    m_cameras.erase(m_cameras.begin() + delete_item);
    markJsonSettingsDirty();
  }

  void markJsonSettingsDirty()
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

    const glm::vec2& currentClipPlanes = cameraM->getClipPlanes();
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
        if(getJsonArray(c, "clip", vfVal))
          camera.clip = {vfVal[0], vfVal[1]};
        else
          camera.clip = currentClipPlanes;  // For old JSON files that didn't have clip planes saved
        addCamera(camera);
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
        auto& c    = m_cameras[n];
        json  jo   = json::object();
        jo["eye"]  = std::vector<float>{c.eye.x, c.eye.y, c.eye.z};
        jo["up"]   = std::vector<float>{c.up.x, c.up.y, c.up.z};
        jo["ctr"]  = std::vector<float>{c.ctr.x, c.ctr.y, c.ctr.z};
        jo["fov"]  = c.fov;
        jo["clip"] = std::vector<float>{c.clip.x, c.clip.y};
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


// Create a more compact button bar with better spacing
static float s_buttonSpacing = 4.0f;

// Calls PropertyEditor::begin() and sets the second column to auto-stretch.
static bool PeBeginAutostretch(const char* label)
{
  if(!PE::begin(label, ImGuiTableFlags_SizingFixedFit))
    return false;
  ImGui::TableSetupColumn("Property");
  ImGui::TableSetupColumn("Value", ImGuiTableColumnFlags_WidthStretch);
  return true;
}

//--------------------------------------------------------------------------------------------------
// Quick Actions Bar with icon buttons
//
static bool QuickActionsBar(std::shared_ptr<nvutils::CameraManipulator> cameraM, nvutils::CameraManipulator::Camera& camera)
{
  bool changed = false;

  // We make the default button color match the background here so that it
  // looks the same as in NavigationSettingsSection.
  ImGui::PushStyleColor(ImGuiCol_Button, ImGui::GetStyle().Colors[ImGuiCol_ChildBg]);

  // Home button
  if(ImGui::Button(ICON_MS_HOME))
  {
    camera  = CameraPresetManager::getInstance().m_cameras[0];
    changed = true;
  }
  nvgui::tooltip("Reset to home camera position");

  // Add/Save camera button
  ImGui::SameLine(0, s_buttonSpacing);
  if(ImGui::Button(ICON_MS_ADD_A_PHOTO))
  {
    CameraPresetManager::getInstance().addCamera(cameraM->getCamera());
  }
  nvgui::tooltip("Save current camera position");

  // Copy button
  ImGui::SameLine(0, s_buttonSpacing);
  if(ImGui::Button(ICON_MS_CONTENT_COPY))
  {
    std::string text = camera.getString();
    ImGui::SetClipboardText(text.c_str());
  }
  nvgui::tooltip("Copy camera state to clipboard");


  // Paste button
  const char* pPastedString;
  ImGui::SameLine(0, s_buttonSpacing);
  if(ImGui::Button(ICON_MS_CONTENT_PASTE) && (pPastedString = ImGui::GetClipboardText()))
  {
    std::string text(pPastedString);
    changed = camera.setFromString(text);
  }
  nvgui::tooltip("Paste camera state from clipboard");

  // Help button, right-aligned
  const float button_size = ImGui::CalcTextSize(ICON_MS_HELP).x + ImGui::GetStyle().FramePadding.x * 2.f;
  ImGui::SameLine(ImGui::GetContentRegionMax().x - button_size, 0.0f);
  if(ImGui::Button(ICON_MS_HELP))
  {
    ImGui::OpenPopup("Camera Help");
  }
  nvgui::tooltip("Show camera controls help");

  ImGui::PopStyleColor();

  // Help popup
  if(ImGui::BeginPopupModal("Camera Help", nullptr, ImGuiWindowFlags_AlwaysAutoResize))
  {
    ImGui::Text("Camera Controls:");
    ImGui::BulletText("Left Mouse: Orbit/Pan/Dolly (depends on mode)");
    ImGui::BulletText("Right Mouse: Look around");
    ImGui::BulletText("Middle Mouse: Pan");
    ImGui::BulletText("Mouse Wheel: Zoom (change FOV)");
    ImGui::BulletText("WASD: Move camera");
    ImGui::BulletText("Q/E: Roll camera");
    ImGui::Spacing();
    ImGui::Text("Navigation Modes:");
    ImGui::BulletText("Examine: Orbit around center point");
    ImGui::BulletText("Fly: Free movement in 3D space");
    ImGui::BulletText("Walk: Movement constrained to horizontal plane");

    if(ImGui::Button("Close", ImVec2(120, 0)))
      ImGui::CloseCurrentPopup();
    ImGui::EndPopup();
  }

  return changed;
}

//--------------------------------------------------------------------------------------------------
// Camera Presets Grid with icons
//
static bool PresetsSection(std::shared_ptr<nvutils::CameraManipulator> cameraM, nvutils::CameraManipulator::Camera& camera)
{
  bool changed = false;

  auto& presetManager   = CameraPresetManager::getInstance();
  int   buttonsCount    = (int)presetManager.m_cameras.size();
  float windowVisibleX2 = ImGui::GetWindowPos().x + ImGui::GetWindowContentRegionMax().x;

  if(buttonsCount == 1)
    ImGui::TextDisabled(" - No saved cameras");

  // Display saved cameras
  int         delete_item = -1;
  std::string thisLabel   = "#1";
  std::string nextLabel;
  for(int n = 1; n < buttonsCount; n++)
  {
    nextLabel = fmt::format("#{}", n + 1);
    ImGui::PushID(n);
    if(ImGui::Button(thisLabel.c_str()))
    {
      camera  = presetManager.m_cameras[n];
      changed = true;
    }

    // Middle click to delete
    if(ImGui::IsItemHovered() && ImGui::GetIO().MouseClicked[ImGuiMouseButton_Middle])
      delete_item = n;

    // Hover tooltip with position info and deletion instruction
    const auto& cam = presetManager.m_cameras[n];
    std::string tooltip =
        fmt::format("Camera #{}\n({:.1f}, {:.1f}, {:.1f})\nMiddle click to delete", n, cam.eye.x, cam.eye.y, cam.eye.z);
    nvgui::tooltip(tooltip.c_str());

    // Auto-wrap buttons
    float last_button_x2 = ImGui::GetItemRectMax().x;
    float next_button_x2 =
        last_button_x2 + s_buttonSpacing + ImGui::CalcTextSize(nextLabel.c_str()).x + ImGui::GetStyle().FramePadding.x * 2.f;
    if(n + 1 < buttonsCount && next_button_x2 < windowVisibleX2)
      ImGui::SameLine(0, s_buttonSpacing);


    thisLabel = std::move(nextLabel);

    ImGui::PopID();
  }

  // Delete camera if requested
  if(delete_item > 0)
  {
    presetManager.removeCamera(delete_item);
  }

  return changed;
}

//--------------------------------------------------------------------------------------------------
// Navigation Settings Section: Mode (examine, fly, walk), Speed
//
static bool NavigationSettingsSection(std::shared_ptr<nvutils::CameraManipulator> cameraM)
{
  bool changed = false;

  ImGui::Separator();
  // Dear ImGui in v1.92 has a FIXME where it doesn't add 1px of spacing after separators
  ImGui::SetCursorPosY(ImGui::GetCursorPosY() + 1.f);

  auto mode     = cameraM->getMode();
  auto speed    = cameraM->getSpeed();
  auto duration = static_cast<float>(cameraM->getAnimationDuration());

  // Change the button color to show the one that's currently active, and to
  // make the other ones match the color of the background.
  ImGui::PushStyleColor(ImGuiCol_Button, ImGui::GetStyle().Colors[ImGuiCol_Button]);

  auto setColor = [&](bool selected) {
    ImGui::GetStyle().Colors[ImGuiCol_Button] =
        selected ? ImGui::GetStyle().Colors[ImGuiCol_ButtonActive] : ImGui::GetStyle().Colors[ImGuiCol_ChildBg];
  };

  // Left-aligned navigation buttons
  setColor(mode == nvutils::CameraManipulator::Examine);
  if(ImGui::Button(ICON_MS_ORBIT))
  {
    cameraM->setMode(nvutils::CameraManipulator::Examine);
    changed = true;
  }
  nvgui::tooltip("Orbit around a point of interest");
  ImGui::SameLine(0, s_buttonSpacing);
  setColor(mode == nvutils::CameraManipulator::Fly);
  if(ImGui::Button(ICON_MS_FLIGHT))
  {
    cameraM->setMode(nvutils::CameraManipulator::Fly);
    changed = true;
  }
  nvgui::tooltip("Fly: Free camera movement");
  ImGui::SameLine(0, s_buttonSpacing);
  setColor(mode == nvutils::CameraManipulator::Walk);
  if(ImGui::Button(ICON_MS_DIRECTIONS_WALK))
  {
    cameraM->setMode(nvutils::CameraManipulator::Walk);
    changed = true;
  }
  nvgui::tooltip("Walk: Stay on a horizontal plane");

  ImGui::PopStyleColor();
  const bool showSettings = (mode == nvutils::CameraManipulator::Fly || mode == nvutils::CameraManipulator::Walk);
  // Speed and transition controls (only shown when fly or walk is selected)
  if(showSettings)
  {
    if(PeBeginAutostretch("##Speed"))
    {
      // ImGuiSliderFlags_Logarithmic requires a value range for its scaling to work.
      const float speedMin = 1e-3f;
      const float speedMax = 1e+3f;
      changed |= PE::DragFloat("Speed", &speed, 2e-4f * (speedMax - speedMin), speedMin, speedMax, "%.2f",
                               ImGuiSliderFlags_Logarithmic, "Speed of camera movement");
      cameraM->setSpeed(speed);
      PE::end();
    }
  }

  return false;
}

//--------------------------------------------------------------------------------------------------
// Camera Position Section : Eye, Center, Up vectors
//
static bool PositionSection(std::shared_ptr<nvutils::CameraManipulator> cameraM,
                            nvutils::CameraManipulator::Camera&         camera,
                            ImGuiTreeNodeFlags                          flag = ImGuiTreeNodeFlags_None)
{
  bool changed = false;

  // We'll ignore changes during animation (but don't want to ignore other
  // changes), so we track changes locally and decide whether to commit them
  // at the end:
  bool myChanged = false;

  if(ImGui::TreeNodeEx("Position", flag))
  {
    if(PeBeginAutostretch("##Position"))
    {
      myChanged |= PE::InputFloat3("Eye", &camera.eye.x);
      myChanged |= PE::InputFloat3("Center", &camera.ctr.x);
      myChanged |= PE::InputFloat3("Up", &camera.up.x);
      PE::end();
    }
    ImGui::TreePop();
  }

  if(!cameraM->isAnimated())  // Ignore changes during animation
  {
    changed |= myChanged;
  }

  return changed;
}

//--------------------------------------------------------------------------------------------------
// Projection Settings Section: field of view, Z-clip planes
//
static bool ProjectionSettingsSection(std::shared_ptr<nvutils::CameraManipulator> cameraManip,
                                      nvutils::CameraManipulator::Camera&         camera,
                                      ImGuiTreeNodeFlags                          flag = ImGuiTreeNodeFlags_None)
{
  bool changed = false;
  if(ImGui::TreeNodeEx("Projection", flag))
  {
    if(PeBeginAutostretch("##Projection"))
    {
      changed |= PE::SliderFloat("FOV", &camera.fov, 1.F, 179.F, "%.1f°", ImGuiSliderFlags_Logarithmic,
                                 "Field of view of the camera (degrees)");

      // ImGuiSliderFlags_Logarithmic requires a value range for its scaling to work.
      const float minClip = 1e-5f;
      const float maxClip = 1e+9f;
      changed |= PE::DragFloat2("Z-Clip", &camera.clip.x, 2e-5f * (maxClip - minClip), minClip, maxClip, "%.6f",
                                ImGuiSliderFlags_Logarithmic, "Near/Far clip planes for depth buffer");

      PE::end();
    }
    ImGui::TreePop();
  }

  return changed;
}


//--------------------------------------------------------------------------------------------------
// Advanced Settings Section : Up vector (Y-up, Z-up), animation transition time
//
static bool OtherSettingsSection(std::shared_ptr<nvutils::CameraManipulator> cameraM,
                                 nvutils::CameraManipulator::Camera&         camera,
                                 ImGuiTreeNodeFlags                          flag = ImGuiTreeNodeFlags_None)
{
  bool changed = false;
  if(ImGui::TreeNodeEx("Other", flag))
  {
    if(PeBeginAutostretch("##Other"))
    {
      PE::entry("Up vector", [&] {
        const bool yIsUp = camera.up.y == 1;
        if(ImGui::RadioButton("Y-up", yIsUp))
        {
          camera.up = glm::vec3(0, 1, 0);
          changed   = true;
        }
        ImGui::SameLine();
        if(ImGui::RadioButton("Z-up", !yIsUp))
        {
          camera.up = glm::vec3(0, 0, 1);
          changed   = true;
        }
        if(glm::length(camera.up) < 0.0001f)
        {
          camera.up = yIsUp ? glm::vec3(0, 1, 0) : glm::vec3(0, 0, 1);
          changed   = true;
        }
        return changed;
      });

      auto duration = static_cast<float>(cameraM->getAnimationDuration());
      changed |= PE::SliderFloat("Transition", &duration, 0.0F, 2.0F, "%.2fs", ImGuiSliderFlags_None,
                                 "Transition duration of camera movement");
      cameraM->setAnimationDuration(duration);

      PE::end();
    }
    ImGui::TreePop();
  }

  return changed;
}

//--------------------------------------------------------------------------------------------------
// Unified camera widget: position, presets, navigation settings
//
bool nvgui::CameraWidget(std::shared_ptr<nvutils::CameraManipulator> cameraManip, bool embed, CameraWidgetSections openSections)
{
  assert(cameraManip && "CameraManipulator is not set");

  bool changed{false};
  bool instantChanged{false};

  nvutils::CameraManipulator::Camera camera = cameraManip->getCamera();

  // Updating the camera manager
  CameraPresetManager::getInstance().update(cameraManip);

  if(embed)
  {
    ImGui::Text("Camera Settings");
    if(!ImGui::BeginChild("CameraPanel", ImVec2(0, 0), ImGuiChildFlags_Borders | ImGuiChildFlags_AutoResizeY))
    {
      return false;
    }
  }

  // Main camera panel with modern design
  {
    changed |= QuickActionsBar(cameraManip, camera);

    changed |= PresetsSection(cameraManip, camera);

    changed |= NavigationSettingsSection(cameraManip);

    ImGui::Separator();

    // Clip planes / FOV section
    instantChanged |= ProjectionSettingsSection(cameraManip, camera,
                                                (openSections & CameraSection_Projection) ? ImGuiTreeNodeFlags_DefaultOpen :
                                                                                            ImGuiTreeNodeFlags_None);

    changed |= PositionSection(cameraManip, camera,
                               (openSections & CameraSection_Position) ? ImGuiTreeNodeFlags_DefaultOpen : ImGuiTreeNodeFlags_None);

    // Up vector / Animation duration section
    changed |= OtherSettingsSection(cameraManip, camera,
                                    (openSections & CameraSection_Other) ? ImGuiTreeNodeFlags_DefaultOpen : ImGuiTreeNodeFlags_None);

    if(embed)
    {
      ImGui::EndChild();
    }
  }

  // Apply the change back to the camera
  if(changed || instantChanged)
  {
    CameraPresetManager::getInstance().markJsonSettingsDirty();
    cameraManip->setCamera(camera, instantChanged);
  }

  return changed || instantChanged;
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
