/*
 * Copyright (c) 2018--2026, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2018--2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
//--------------------------------------------------------------------

#pragma once

#include <array>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <optional>
#include <string>

namespace nvutils {
namespace CameraConstants {
// Distance thresholds
constexpr float EPSILON      = 1e-6f;
constexpr float MIN_DISTANCE = 0.000001f;

// FOV limits (degrees)
constexpr float MIN_FOV = 0.01f;
constexpr float MAX_FOV = 179.0f;

// Orthographic limits
constexpr float MIN_ORTHOGRAPHIC_SIZE = 0.01f;

// Input scaling
constexpr float WHEEL_ZOOM_RATE        = 0.1f;   // 10% per wheel step
constexpr float MAX_DOLLY_DISPLACEMENT = 0.99f;  // Don't cross center

// Animation
constexpr double DEFAULT_ANIMATION_DURATION = 0.5;  // seconds

// Aspect ratio safety
constexpr float MIN_ASPECT_RATIO = EPSILON;
}  // namespace CameraConstants

/*-------------------------------------------------------------------------------------------------
  # class nvutils::CameraManipulator

  nvutils::CameraManipulator is a camera manipulator help class
  It allow to simply do
  - Orbit        (LMB)
  - Pan          (LMB + CTRL  | MMB)
  - Dolly        (LMB + SHIFT | RMB)
  - Look Around  (LMB + ALT   | LMB + CTRL + SHIFT)

  In a various ways:
  - examiner(orbit around object)
  - walk (look up or down but stays on a plane)
  - fly ( go toward the interest point)

  Do use the camera manipulator, you need to do the following
  - Call setWindowSize() at creation of the application and when the window size change
  - Call setLookat() at creation to initialize the camera look position
  - Call setMousePosition() on application mouse down
  - Call mouseMove() on application mouse move

  Retrieve the camera matrix by calling getMatrix()

  See: appbase_vkpp.hpp

  ```cpp
  // Retrieve/set camera information
  CameraManip.getLookat(eye, center, up);
  CameraManip.setLookat(eye, center, glm::vec3(m_upVector == 0, m_upVector == 1, m_upVector == 2));
  CameraManip.getFov();
  CameraManip.setSpeed(navSpeed);
  CameraManip.setMode(navMode == 0 ? nvutils::CameraManipulator::Modes::Examine : nvutils::CameraManipulator::Modes::Fly);
  // On mouse down, keep mouse coordinates
  CameraManip.setMousePosition(x, y);
  // On mouse move and mouse button down
  if(m_inputs.lmb || m_inputs.rmb || m_inputs.mmb)
  {
  CameraManip.mouseMove(x, y, m_inputs);
  }
  // Wheel changes the FOV
  CameraManip.wheel(delta > 0 ? 1 : -1, m_inputs);
  // Retrieve the matrix to push to the shader
  m_ubo.view = CameraManip.getMatrix();	
  ````

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
    glm::vec3      eye            = glm::vec3(10, 10, 10);
    glm::vec3      ctr            = glm::vec3(0, 0, 0);
    glm::vec3      up             = glm::vec3(0, 1, 0);
    float          fov            = 60.0f;
    glm::vec2      nearFar        = {0.001f, 100000.0f};
    glm::vec2      orthMag        = {5.0f, 5.0f};  // Orthographic half-width/height (glTF xmag, ymag)
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
  Actions mouseMove(glm::vec2 screenDisplacement, const Inputs& inputs);

  // === Camera State ===
  // Set the camera to look at the interest point.
  // eye: camera position in world space.
  // center: point of interest.
  // up: up vector (normalized internally).
  // instantSet: if true, jump immediately; if false, animate smoothly.
  void setLookat(const glm::vec3& eye, const glm::vec3& center, const glm::vec3& up, bool instantSet = true);

  // === Animation ===
  // Update the camera animation if active.
  // currentTimeMs: optional external time source (milliseconds). Pass < 0 to use system time.
  void updateAnim(double currentTimeMs = -1.0);

  // === Configuration ===
  // To call when the size of the window change. This allows to do nicer movement according to the window size.
  void setWindowSize(glm::uvec2 winSize);

  Camera getCamera() const { return m_current; }
  void   setCamera(Camera camera, bool instantSet = true);

  // Retrieve the position, interest and up vector of the camera
  void      getLookat(glm::vec3& eye, glm::vec3& center, glm::vec3& up) const;
  glm::vec3 getEye() const { return m_current.eye; }
  glm::vec3 getCenter() const { return m_current.ctr; }
  glm::vec3 getUp() const { return m_current.up; }
  glm::vec3 getViewDirection() const { return glm::normalize(m_current.ctr - m_current.eye); }
  float     getDistanceToCenter() const { return glm::length(m_current.ctr - m_current.eye); }
  float     getAnimationProgress() const;

  // Set the manipulator mode, from Examiner, to walk, to fly, ...
  void setMode(Modes mode) { m_mode = mode; }

  // Retrieve the current manipulator mode
  Modes getMode() const { return m_mode; }

  // Retrieving the transformation matrix of the camera
  const glm::mat4& getViewMatrix() const { return m_matrix; }

  const glm::mat4 getPerspectiveMatrix() const
  {
    glm::mat4 projMatrix;
    if(m_current.projectionType == ProjectionType::Orthographic)
    {
      float halfWidth  = m_current.orthMag.x;
      float halfHeight = m_current.orthMag.y;
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

  // Set the position, interest from the matrix.
  // instantSet = true will not interpolate to the new position
  // centerDistance is the distance of the center from the eye
  void setMatrix(const glm::mat4& mat_, bool instantSet = true, float centerDistance = 1.f);

  // Changing the default speed movement
  void setSpeed(float speed);

  // Retrieving the current speed
  float getSpeed() const { return m_speed; }

  // Mouse position
  void      setMousePosition(const glm::vec2& pos) { m_mouse = pos; }
  glm::vec2 getMousePosition() const { return m_mouse; }

  // Apply a camera motion derived from screen displacement.
  // screenDisplacement: current cursor position in screen space.
  // action: the camera action to apply.
  void motion(const glm::vec2& screenDisplacement, Actions action = Actions::NoAction);

  // Apply camera movement from keyboard input (e.g., WASD).
  // delta: movement deltas in screen space.
  // action: the camera action to apply (Dolly or Pan).
  void keyMotion(glm::vec2 delta, Actions action);

  // Handle mouse wheel input (dolly or zoom).
  // value: wheel delta.
  // inputs: modifier state (shift changes zoom/FOV).
  void wheel(float value, const Inputs& inputs);

  // Retrieve the screen dimension
  glm::uvec2 getWindowSize() const { return m_windowSize; }
  float      getAspectRatio() const { return static_cast<float>(m_windowSize.x) / static_cast<float>(m_windowSize.y); }

  // Adjust the orthographic projection to maintain the correct aspect ratio
  void adjustOrthographicAspect();

  // Field of view in degrees
  void  setFov(float fovDegree);
  float getFov() const { return m_current.fov; }
  float getRadFov() const { return glm::radians(m_current.fov); }

  // Clip planes
  void             setClipPlanes(glm::vec2 nearFar);
  const glm::vec2& getClipPlanes() const { return m_current.nearFar; }

  // Projection type
  void           setProjectionType(ProjectionType type) { m_current.projectionType = type; }
  ProjectionType getProjectionType() const { return m_current.projectionType; }

  // Convert between projection types while preserving the view at center point
  void convertToPerspective();
  void convertToOrthographic();

  // Orthographic size
  void      setOrthographicMagnitudes(const glm::vec2& mag);
  glm::vec2 getOrthographicMagnitudes() const { return m_current.orthMag; }
  float     getOrthographicXmag() const { return m_current.orthMag.x; }
  float     getOrthographicYmag() const { return m_current.orthMag.y; }

  // Animation duration
  double getAnimationDuration() const { return m_duration; }
  void   setAnimationDuration(double val);
  bool   isAnimated() const { return m_isAnimating; }

  // Animation state machine:
  //   Idle -> (setCamera/setLookat with instantSet=false) -> Animating
  //   Animating -> (t >= 1.0) -> Idle
  //   Any user interaction cancels animation immediately.

  // Returning a default help string
  const std::string& getHelp();

  // Fit the camera to fully view a bounding box.
  // boxMin: lower corner of the box.
  // boxMax: upper corner of the box.
  // instantFit: if true, jump immediately; if false, animate.
  // tight: if true, fit tightly to corners; otherwise fit to bounding sphere.
  // aspect: aspect ratio for the fit.
  void fit(const glm::vec3& boxMin, const glm::vec3& boxMax, bool instantFit = true, bool tight = false, float aspect = 1.0f);

  // Convenience setters
  void setEye(const glm::vec3& eye, bool instantSet = true) { setLookat(eye, m_current.ctr, m_current.up, instantSet); }
  void setCenter(const glm::vec3& center, bool instantSet = true)
  {
    setLookat(m_current.eye, center, m_current.up, instantSet);
  }
  void setUp(const glm::vec3& up, bool instantSet = true) { setLookat(m_current.eye, m_current.ctr, up, instantSet); }

private:
  struct ViewDimensions
  {
    float width;
    float height;
  };

  struct CameraFrame
  {
    glm::vec3 forward{};
    glm::vec3 right{};
    glm::vec3 up{};
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
  glm::vec3      projectToGroundPlane(const glm::vec3& vec) const;
  void           zoomOrthographic(float factor);

  // Do panning: movement parallels to the screen
  // displacement: normalized screen displacement [0,1]
  void pan(glm::vec2 displacement);
  // Do orbiting: rotation around the center of interest. If invert, the interest orbits around the camera position
  // displacement: normalized screen displacement [0,1]
  void orbit(glm::vec2 displacement, bool invert = false);
  // Do dolly: movement toward the interest. In orthographic mode, this zooms the view volume.
  // displacement: normalized screen displacement [0,1]
  void dolly(glm::vec2 displacement, bool keepCenterFixed = false);

  double getSystemTime() const;

  glm::vec3 computeBezier(float t, const glm::vec3& p0, const glm::vec3& p1, const glm::vec3& p2) const;
  void      findBezierPoints();

  static bool isValidPosition(const glm::vec3& pos);
  static bool isValidDirection(const glm::vec3& dir);
  bool        validateCamera(const Camera& cam) const;

protected:
  glm::mat4 m_matrix = glm::mat4(1);

  Camera                m_current;   // Current camera position
  Camera                m_goal;      // Wish camera position
  std::optional<Camera> m_snapshot;  // Current camera the moment a set look-at is done

  // Animation
  std::array<glm::vec3, 3> m_bezier         = {};
  float                    m_animDollyZoom0 = 0.0f;
  float                    m_animDollyZoom1 = 0.0f;
  double                   m_startTime      = 0;
  double                   m_duration       = CameraConstants::DEFAULT_ANIMATION_DURATION;
  bool                     m_isAnimating    = false;

  // Window size
  glm::uvec2 m_windowSize = glm::uvec2(1, 1);

  // Other
  float     m_speed = 3.f;
  glm::vec2 m_mouse = glm::vec2(0.f, 0.f);

  Modes m_mode = Modes::Examine;
};

// Global Manipulator

}  // namespace nvutils
