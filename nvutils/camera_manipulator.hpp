/*
 * Copyright (c) 2018-2026, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
//--------------------------------------------------------------------

#pragma once

#include <array>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace nvutils {
namespace CameraConstants {
// Distance thresholds
constexpr double EPSILON      = 1e-6;
constexpr double MIN_DISTANCE = 0.000001;

// FOV limits (degrees)
constexpr double MIN_FOV = 0.01;
constexpr double MAX_FOV = 179.0;

// Orthographic limits
constexpr double MIN_ORTHOGRAPHIC_SIZE = 0.01;

// Input scaling
constexpr double WHEEL_ZOOM_RATE        = 0.1;   // 10% per wheel step
constexpr double MAX_DOLLY_DISPLACEMENT = 0.99;  // Don't cross center

// Animation
constexpr double DEFAULT_ANIMATION_DURATION = 0.5;  // seconds

// Aspect ratio safety
constexpr double MIN_ASPECT_RATIO = EPSILON;
}  // namespace CameraConstants

/*-------------------------------------------------------------------------------------------------
  nvutils::CameraManipulator implements camera movement controls.
  It gives a simple way to:
  - Orbit        (LMB)
  - Pan          (LMB + CTRL  | MMB)
  - Dolly        (LMB + SHIFT | RMB)
  - Look Around  (LMB + ALT   | LMB + CTRL + SHIFT)

  In various ways:
  - examine (orbit around object)
  - walk (look up or down but stay on a plane)
  - fly (go toward the interest point)

  To use the camera manipulator, you need to do the following:
  - Call setWindowSize() at creation of the application and when the window size change
  - Call setLookat() at creation to initialize the camera look position
  - Call setMousePosition() on application mouse down
  - Call mouseMove() on application mouse move

  Retrieve the camera matrix by calling getMatrix().

  For example usage, see usage_CameraManipulator() at the bottom of camera_manipulator.cpp.
  Often samples will use this via nvgui::CameraWidget (preset manager and UI)
  or nvapp::ElementCamera (updates the camera every frame for you).

  Coordinate system and behavior:
  - Right-handed coordinate system
  - Default up vector: +Y (0, 1, 0)
  - Camera looks down -Z axis in local space
  - Screen space: origin top-left, +X right, +Y down
  - Displacement is normalized by window size
  - Orbit: horizontal around world up, vertical around camera right

-------------------------------------------------------------------------------------------------*/
class CameraManipulator
{
public:
  CameraManipulator();

  enum Modes
  {
    Examine,
    Fly,
    Walk
  };

  enum class Actions
  {
    NoAction,
    Orbit,
    Dolly,
    Pan,
    LookAround
  };

  enum ProjectionType
  {
    Perspective,
    Orthographic
  };

  struct Inputs
  {
    bool lmb   = false;
    bool mmb   = false;
    bool rmb   = false;
    bool shift = false;
    bool ctrl  = false;
    bool alt   = false;
  };

  struct Camera
  {
    glm::dvec3     eye            = glm::dvec3(10, 10, 10);
    glm::dvec3     ctr            = glm::dvec3(0, 0, 0);
    glm::dvec3     up             = glm::dvec3(0, 1, 0);
    double         fov            = 60.0;
    glm::dvec2     nearFar        = {0.001, 100000.0};
    glm::dvec2     orthMag        = {5.0, 5.0};  // Orthographic half-width/height (glTF xmag, ymag)
    ProjectionType projectionType = ProjectionType::Perspective;

    bool operator!=(const Camera& rhr) const
    {
      return (eye != rhr.eye) || (ctr != rhr.ctr) || (up != rhr.up) || (fov != rhr.fov) || (nearFar != rhr.nearFar)
             || (projectionType != rhr.projectionType) || (orthMag != rhr.orthMag);
    }
    bool operator==(const Camera& rhr) const
    {
      return (eye == rhr.eye) && (ctr == rhr.ctr) && (up == rhr.up) && (fov == rhr.fov) && (nearFar == rhr.nearFar)
             && (projectionType == rhr.projectionType) && (orthMag == rhr.orthMag);
    }

    // basic serialization, mostly for copy/paste
    std::string getString() const;
    bool        setFromString(const std::string& text);
  };

public:
  // === Interaction ===
  // Handle mouse movement with current input state.
  // screenDisplacement: current cursor position in screen space.
  // inputs: mouse button and modifier state.
  // returns: the action that was applied, if any.
  Actions mouseMove(glm::dvec2 screenDisplacement, const Inputs& inputs);

  // === Camera State ===
  // Set the camera to look at the interest point.
  // eye: camera position in world space.
  // center: point of interest.
  // up: up vector (normalized internally).
  // instantSet: if true, jump immediately; if false, animate smoothly.
  void setLookat(const glm::dvec3& eye, const glm::dvec3& center, const glm::dvec3& up, bool instantSet = true);

  // === Animation ===
  // Animation occurs when `setCamera()` was called with interpolation.
  // It's different than when camera plugins override
  // Animation state machine:
  //   Idle -> (setCamera/setLookat with instantSet=false) -> Animating
  //   Animating -> (t >= 1.0) -> Idle
  //   Any user interaction cancels animation immediately and calls onUserInterrupt().

  // Update the camera animation if active (which happens when `setCamera()`
  // was called with interpolation).
  // This should be called every frame as it also calls plugin update methods.
  // currentTimeMs: optional external time source (milliseconds). Pass < 0 to use system time.
  void updateAnim(double currentTimeMs = -1.0);
  // Get the progress of the current animation as a unorm from 0.0 == start to 1.0 == finish.
  double getAnimationProgress() const;
  double getAnimationDuration() const { return m_duration; }
  void   setAnimationDuration(double val);
  bool   isAnimated() const { return m_isAnimating; }

  // === Configuration ===
  // To call when the size of the window change. This allows to do nicer movement according to the window size.
  void setWindowSize(glm::uvec2 winSize);

  const Camera& getCamera() const { return m_current; }
  // Set the new camera as a goal
  // instantSet = true will not interpolate to the new position
  void setCamera(Camera camera, bool instantSet = true);

  // Retrieve the position, interest and up vector of the camera
  void       getLookat(glm::dvec3& eye, glm::dvec3& center, glm::dvec3& up) const;
  glm::dvec3 getEye() const { return m_current.eye; }
  glm::dvec3 getCenter() const { return m_current.ctr; }
  glm::dvec3 getUp() const { return m_current.up; }
  glm::dvec3 getViewDirection() const { return glm::normalize(m_current.ctr - m_current.eye); }
  double     getDistanceToCenter() const { return glm::length(m_current.ctr - m_current.eye); }

  // Set the manipulator mode, from Examiner, to walk, to fly, ...
  void setMode(Modes mode) { m_mode = mode; }

  // Retrieve the current manipulator mode
  Modes getMode() const { return m_mode; }

  // Retrieving the transformation matrix of the camera
  const glm::dmat4& getViewMatrix() const { return m_matrix; }

  const glm::dmat4 getPerspectiveMatrix() const
  {
    glm::dmat4 projMatrix;
    if(m_current.projectionType == ProjectionType::Orthographic)
    {
      double halfWidth  = m_current.orthMag.x;
      double halfHeight = m_current.orthMag.y;
      projMatrix =
          glm::orthoRH_ZO(-halfWidth, halfWidth, -halfHeight, halfHeight, m_current.nearFar.x, m_current.nearFar.y);
    }
    else
    {
      projMatrix = glm::perspectiveRH_ZO(getRadFov(), getAspectRatio(), m_current.nearFar.x, m_current.nearFar.y);
    }
    projMatrix[1][1] *= -1;  // Flip the Y axis for Vulkan
    return projMatrix;
  }

  // Set the position and point of interest from the given view matrix.
  // instantSet = true will not interpolate to the new position.
  // centerDistance is the distance of the center from the eye.
  void setMatrix(const glm::dmat4& mat_, bool instantSet = true, double centerDistance = 1.0);

  // Changing the default speed movement
  void setSpeed(double speed);

  // Retrieving the current speed
  double getSpeed() const { return m_speed; }

  // Mouse position
  void       setMousePosition(const glm::dvec2& pos) { m_mouse = pos; }
  glm::dvec2 getMousePosition() const { return m_mouse; }

  // Apply a camera motion derived from screen displacement.
  // screenDisplacement: current cursor position in screen space.
  // action: the camera action to apply.
  void motion(const glm::dvec2& screenDisplacement, Actions action = Actions::NoAction);

  // Apply camera movement from keyboard input (e.g., WASD).
  // delta: movement deltas in screen space.
  // action: the camera action to apply (Dolly or Pan).
  void keyMotion(glm::dvec2 delta, Actions action);

  // Handle mouse wheel input (dolly or zoom).
  // value: wheel delta.
  // inputs: modifier state (shift changes zoom/FOV).
  void wheel(double value, const Inputs& inputs);

  // Retrieve the screen dimension
  glm::uvec2 getWindowSize() const { return m_windowSize; }
  double getAspectRatio() const { return static_cast<double>(m_windowSize.x) / static_cast<double>(m_windowSize.y); }

  // Adjust the orthographic camera's aspect ratio to match the current viewport.
  // More specifically, sets the orthographic width (xmag) to be
  // the height (ymag) multiplied by the aspect ratio.
  void adjustOrthographicAspect();

  // Field of view in degrees; clamped between MIN_FOV and MAX_FOV.
  void   setFov(double fovDegree);
  double getFov() const { return m_current.fov; }
  double getRadFov() const { return glm::radians(m_current.fov); }

  // Clip planes
  void              setClipPlanes(glm::dvec2 nearFar);
  const glm::dvec2& getClipPlanes() const { return m_current.nearFar; }

  // Projection type
  void           setProjectionType(ProjectionType type) { m_current.projectionType = type; }
  ProjectionType getProjectionType() const { return m_current.projectionType; }

  // Convert between projection types while preserving the view at center point
  void convertToPerspective();
  void convertToOrthographic();

  // Orthographic size
  void       setOrthographicMagnitudes(const glm::dvec2& mag);
  glm::dvec2 getOrthographicMagnitudes() const { return m_current.orthMag; }
  double     getOrthographicXmag() const { return m_current.orthMag.x; }
  double     getOrthographicYmag() const { return m_current.orthMag.y; }

  // Returns a string that can be included in help dialogs.
  const std::string& getHelp();

  // Fit the camera to fully view a bounding box.
  // boxMin: lower corner of the box.
  // boxMax: upper corner of the box.
  // instantFit: if true, jump immediately; if false, animate to new position.
  // tight: if true, fit tightly to corners; otherwise fit to bounding sphere.
  // aspect: aspect ratio of the window.
  void fit(const glm::dvec3& boxMin, const glm::dvec3& boxMax, bool instantFit = true, bool tight = false, double aspect = 1.0);

  // Convenience setters
  void setEye(const glm::dvec3& eye, bool instantSet = true)
  {
    setLookat(eye, m_current.ctr, m_current.up, instantSet);
  }
  void setCenter(const glm::dvec3& center, bool instantSet = true)
  {
    setLookat(m_current.eye, center, m_current.up, instantSet);
  }
  void setUp(const glm::dvec3& up, bool instantSet = true) { setLookat(m_current.eye, m_current.ctr, up, instantSet); }

  // Returns an increasing time value in milliseconds.
  double getTimeMs() const;

private:
  struct ViewDimensions
  {
    double width;
    double height;
  };

  struct CameraFrame
  {
    glm::dvec3 forward{};
    glm::dvec3 right{};
    glm::dvec3 up{};
  };

  // Update the internal matrix.
  void updateLookatMatrix();

  // Apply immediate camera update and refresh the view matrix.
  void applyCameraInstant(const Camera& camera);
  // Start an animation toward the target camera.
  void startAnimationTo(const Camera& camera);
  // User input cancels animation; optionally refresh the view matrix.
  void applyUserChange(bool updateMatrix = true);

  // Helpers
  ViewDimensions getViewDimensions() const;
  CameraFrame    computeCameraFrame() const;
  glm::dvec3     projectToGroundPlane(const glm::dvec3& vec) const;
  void           zoomOrthographic(double factor);

  // Do panning: movement parallel to the screen
  // displacement: normalized screen displacement [0,1]
  void pan(glm::dvec2 displacement);
  // Do orbiting: rotation around the center of interest. If `invert` is true,
  // then the camera stays in place and interest orbits around the camera position.
  // displacement: normalized screen displacement [0,1]
  void orbit(glm::dvec2 displacement, bool invert = false);
  // Do dolly: movement toward the interest. In orthographic mode, this zooms the view volume.
  // displacement: normalized screen displacement [0,1]
  void dolly(glm::dvec2 displacement, bool keepCenterFixed = false);

  void       baseAnimation(double elapsedTime);
  glm::dvec3 computeBezier(double t, const glm::dvec3& p0, const glm::dvec3& p1, const glm::dvec3& p2) const;
  void       findBezierPoints();

  static bool isValidPosition(const glm::dvec3& pos);
  static bool isValidDirection(const glm::dvec3& dir);
  bool        validateCamera(const Camera& cam) const;

protected:
  glm::dmat4 m_matrix = glm::dmat4(1);

  Camera                m_current;   // Current camera position
  Camera                m_goal;      // Wish camera position
  std::optional<Camera> m_snapshot;  // Current camera the moment a set look-at is done

  // Animation
  std::array<glm::dvec3, 3> m_bezier         = {};
  double                    m_animDollyZoom0 = 0.0;
  double                    m_animDollyZoom1 = 0.0;
  bool                      m_vertigoEffect  = false;
  double                    m_startTime      = 0;
  double                    m_duration       = CameraConstants::DEFAULT_ANIMATION_DURATION;
  bool                      m_isAnimating    = false;

  // Window size
  glm::uvec2 m_windowSize = glm::uvec2(1, 1);

  // Other
  double     m_speed = 3.0;
  glm::dvec2 m_mouse = glm::dvec2(0.0, 0.0);

  Modes m_mode = Modes::Examine;
};

// Plugin interface for the camera system.
// This allows you to override camera motions and do things like animate
// cameras along paths while also adjusting other scene parameters.
// Different samples might want different ways of creating camera paths -- e.g.
// some might allow you to set your own keypoints for flying through scenes,
// while others might automatically generate camera trajectories -- so we
// make this fairly general instead of mandating a particular system.
struct CameraPlugin
{
  // Returns a unique name for this plugin so it doesn't clash with others
  // that are serialized.
  virtual const char* getName() = 0;
  // Called every time CameraManipulator::updateAnim() is called.
  // Use this to control the camera's position.
  virtual void onUpdateAnim(CameraManipulator& cameraManip) {}
  // Called when the user interacts with the camera. If the plugin's
  // controlling the camera, it should stop and return control to the user.
  virtual void onUserInterrupt() {}
  // Called to render the ImGui UI for the plugin. This will be called from
  // inside an ImGui window or child. Returns whether any settings were changed.
  virtual bool onUIRender(CameraManipulator& cameraManip) { return false; }
  // Called when a camera preset (in nvutils/camera.hpp) is added.
  // (It'll always be the last one in nvgui::GetCameras().)
  virtual void onPresetAdd() {}
  // Called when camera preset `removedIndex` is removed.
  virtual void onPresetRemove(size_t removedIndex) {}
  // Serializes this plugin's state to a string.
  // This is called when camera presets are saved.
  virtual std::string onSerialize() { return ""; }
  // Deserializes this plugin's state from a string.
  // This is called when camera presets are loaded.
  virtual void onDeserialize(const std::string& serialized) {}

  // Since this has virtual methods:
  virtual ~CameraPlugin() = default;
};

std::vector<std::shared_ptr<CameraPlugin>>& cameraPlugins();

}  // namespace nvutils
