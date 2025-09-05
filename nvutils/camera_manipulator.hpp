/*
 * Copyright (c) 2018-2025, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2018-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
//--------------------------------------------------------------------

#pragma once

#include <array>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <string>

namespace nvutils {
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
  CameraManip.setMode(navMode == 0 ? nvutils::CameraManipulator::Examine : nvutils::CameraManipulator::Fly);
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

-------------------------------------------------------------------------------------------------*/

class CameraManipulator
{
public:
  CameraManipulator();

  // clang-format off
  enum Modes { Examine, Fly, Walk};
  enum Actions { NoAction, Orbit, Dolly, Pan, LookAround };
  struct Inputs {bool lmb=false; bool mmb=false; bool rmb=false; 
                 bool shift=false; bool ctrl=false; bool alt=false;};
  // clang-format on

  struct Camera
  {
    glm::vec3 eye  = glm::vec3(10, 10, 10);
    glm::vec3 ctr  = glm::vec3(0, 0, 0);
    glm::vec3 up   = glm::vec3(0, 1, 0);
    float     fov  = 60.0f;
    glm::vec2 clip = {0.001f, 100000.0f};

    bool operator!=(const Camera& rhr) const
    {
      return (eye != rhr.eye) || (ctr != rhr.ctr) || (up != rhr.up) || (fov != rhr.fov) || (clip != rhr.clip);
    }
    bool operator==(const Camera& rhr) const
    {
      return (eye == rhr.eye) && (ctr == rhr.ctr) && (up == rhr.up) && (fov == rhr.fov) && (clip == rhr.clip);
    }

    // basic serialization, mostly for copy/paste
    std::string getString() const;
    bool        setFromString(const std::string& text);
  };

public:
  // Main function to call from the application
  // On application mouse move, call this function with the current mouse position, mouse
  // button presses and keyboard modifier. The camera matrix will be updated and
  // can be retrieved calling getMatrix
  Actions mouseMove(glm::vec2 screenDisplacement, const Inputs& inputs);

  // Set the camera to look at the interest point
  // instantSet = true will not interpolate to the new position
  void setLookat(const glm::vec3& eye, const glm::vec3& center, const glm::vec3& up, bool instantSet = true);

  // This should be called in an application loop to update the camera matrix if this one is animated: new position, key movement
  void updateAnim();

  // To call when the size of the window change.  This allows to do nicer movement according to the window size.
  void setWindowSize(glm::uvec2 winSize) { m_windowSize = winSize; }

  Camera getCamera() const { return m_current; }
  void   setCamera(Camera camera, bool instantSet = true);

  // Retrieve the position, interest and up vector of the camera
  void      getLookat(glm::vec3& eye, glm::vec3& center, glm::vec3& up) const;
  glm::vec3 getEye() const { return m_current.eye; }
  glm::vec3 getCenter() const { return m_current.ctr; }
  glm::vec3 getUp() const { return m_current.up; }

  // Set the manipulator mode, from Examiner, to walk, to fly, ...
  void setMode(Modes mode) { m_mode = mode; }

  // Retrieve the current manipulator mode
  Modes getMode() const { return m_mode; }

  // Retrieving the transformation matrix of the camera
  const glm::mat4& getViewMatrix() const { return m_matrix; }

  const glm::mat4 getPerspectiveMatrix() const
  {
    glm::mat4 projMatrix = glm::perspectiveRH_ZO(getRadFov(), getAspectRatio(), m_current.clip.x, m_current.clip.y);
    projMatrix[1][1] *= -1;  // Flip the Y axis
    return projMatrix;
  }

  // Set the position, interest from the matrix.
  // instantSet = true will not interpolate to the new position
  // centerDistance is the distance of the center from the eye
  void setMatrix(const glm::mat4& mat_, bool instantSet = true, float centerDistance = 1.f);

  // Changing the default speed movement
  void setSpeed(float speed) { m_speed = speed; }

  // Retrieving the current speed
  float getSpeed() const { return m_speed; }

  // Mouse position
  void      setMousePosition(const glm::vec2& pos) { m_mouse = pos; }
  glm::vec2 getMousePosition() const { return m_mouse; }

  // Main function which is called to apply a camera motion.
  // It is preferable to
  void motion(const glm::vec2& screenDisplacement, Actions action = NoAction);

  // This is called when moving with keys (ex. WASD)
  void keyMotion(glm::vec2 delta, Actions action);

  // To call when the mouse wheel change
  void wheel(float value, const Inputs& inputs);

  // Retrieve the screen dimension
  glm::uvec2 getWindowSize() const { return m_windowSize; }
  float      getAspectRatio() const { return static_cast<float>(m_windowSize.x) / static_cast<float>(m_windowSize.y); }

  // Field of view in degrees
  void  setFov(float fovDegree);
  float getFov() const { return m_current.fov; }
  float getRadFov() const { return glm::radians(m_current.fov); }

  // Clip planes
  void             setClipPlanes(glm::vec2 clip) { m_current.clip = clip; }
  const glm::vec2& getClipPlanes() const { return m_current.clip; }

  // Animation duration
  double getAnimationDuration() const { return m_duration; }
  void   setAnimationDuration(double val) { m_duration = val; }
  bool   isAnimated() const { return m_animDone == false; }

  // Returning a default help string
  const std::string& getHelp();

  // Fitting the camera position and interest to see the bounding box
  void fit(const glm::vec3& boxMin, const glm::vec3& boxMax, bool instantFit = true, bool tight = false, float aspect = 1.0f);

private:
  // Update the internal matrix.
  void updateLookatMatrix() { m_matrix = glm::lookAt(m_current.eye, m_current.ctr, m_current.up); }

  // Do panning: movement parallels to the screen
  void pan(glm::vec2 displacement);
  // Do orbiting: rotation around the center of interest. If invert, the interest orbit around the camera position
  void orbit(glm::vec2 displacement, bool invert = false);
  // Do dolly: movement toward the interest.
  void dolly(glm::vec2 displacement, bool keepCenterFixed = false);


  double getSystemTime();

  glm::vec3 computeBezier(float t, glm::vec3& p0, glm::vec3& p1, glm::vec3& p2);
  void      findBezierPoints();

protected:
  glm::mat4 m_matrix = glm::mat4(1);

  Camera m_current;   // Current camera position
  Camera m_goal;      // Wish camera position
  Camera m_snapshot;  // Current camera the moment a set look-at is done

  // Animation
  std::array<glm::vec3, 3> m_bezier    = {};
  double                   m_startTime = 0;
  double                   m_duration  = 0.5;
  bool                     m_animDone  = true;

  // Window size
  glm::uvec2 m_windowSize = glm::uvec2(1, 1);

  // Other
  float     m_speed = 3.f;
  glm::vec2 m_mouse = glm::vec2(0.f, 0.f);

  Modes m_mode = Examine;
};

// Global Manipulator

}  // namespace nvutils
