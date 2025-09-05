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

#include <algorithm>
#include <chrono>
#include <iostream>
#include <cmath>
#include <fmt/format.h>

#include "camera_manipulator.hpp"

#ifdef _MSC_VER
#define SAFE_SSCANF sscanf_s
#else
#define SAFE_SSCANF sscanf
#endif

namespace nvutils {

//--------------------------------------------------------------------------------------------------
CameraManipulator::CameraManipulator()
{
  updateLookatMatrix();
}

//--------------------------------------------------------------------------------------------------
// Set the new camera as a goal
// instantSet = true will not interpolate to the new position
void CameraManipulator::setCamera(Camera camera, bool instantSet /*=true*/)
{
  m_animDone = true;

  if(instantSet || m_duration == 0.0)
  {
    m_current = camera;
    updateLookatMatrix();
  }
  else if(camera != m_current)
  {
    m_goal      = camera;
    m_snapshot  = m_current;
    m_animDone  = false;
    m_startTime = getSystemTime();
    findBezierPoints();
  }
}


//--------------------------------------------------------------------------------------------------
// Creates a viewing matrix derived from an eye point, a reference point indicating the center of
// the scene, and an up vector
//
void CameraManipulator::setLookat(const glm::vec3& eye, const glm::vec3& center, const glm::vec3& up, bool instantSet)
{
  setCamera({eye, center, up, m_current.fov, m_current.clip}, instantSet);
}

//-----------------------------------------------------------------------------
// Get the current camera's look-at parameters.
void CameraManipulator::getLookat(glm::vec3& eye, glm::vec3& center, glm::vec3& up) const
{
  eye    = m_current.eye;
  center = m_current.ctr;
  up     = m_current.up;
}

//--------------------------------------------------------------------------------------------------
// Pan the camera perpendicularly to the light of sight.
void CameraManipulator::pan(glm::vec2 displacement)
{
  if(m_mode == Fly)
  {
    displacement *= -1.f;
  }

  glm::vec3 viewDirection(m_current.eye - m_current.ctr);
  float     viewDistance = static_cast<float>(glm::length(viewDirection)) / 0.785f;  // 45 degrees
  viewDirection          = glm::normalize(viewDirection);
  glm::vec3 rightVector  = glm::cross(m_current.up, viewDirection);
  glm::vec3 upVector     = glm::cross(viewDirection, rightVector);
  rightVector            = glm::normalize(rightVector);
  upVector               = glm::normalize(upVector);

  glm::vec3 panOffset = (-displacement.x * rightVector + displacement.y * upVector) * viewDistance;
  m_current.eye += panOffset;
  m_current.ctr += panOffset;
}

//--------------------------------------------------------------------------------------------------
// Orbit the camera around the center of interest. If 'invert' is true,
// then the camera stays in place and the interest orbit around the camera.
//
void CameraManipulator::orbit(glm::vec2 displacement, bool invert /*= false*/)
{
  if(displacement == glm::vec2(0.f, 0.f))
    return;

  // Full width will do a full turn
  displacement *= glm::two_pi<float>();

  // Get the camera
  glm::vec3 origin(invert ? m_current.eye : m_current.ctr);
  glm::vec3 position(invert ? m_current.ctr : m_current.eye);

  // Get the length of sight
  glm::vec3 centerToEye(position - origin);
  float     radius = glm::length(centerToEye);
  centerToEye      = glm::normalize(centerToEye);
  glm::vec3 axeZ   = centerToEye;

  // Find the rotation around the UP axis (Y)
  glm::mat4 rotY = glm::rotate(glm::mat4(1), -displacement.x, m_current.up);

  // Apply the (Y) rotation to the eye-center vector
  centerToEye = rotY * glm::vec4(centerToEye, 0);

  // Find the rotation around the X vector: cross between eye-center and up (X)
  glm::vec3 axeX = glm::normalize(glm::cross(m_current.up, axeZ));
  glm::mat4 rotX = glm::rotate(glm::mat4(1), -displacement.y, axeX);

  // Apply the (X) rotation to the eye-center vector
  glm::vec3 rotationVec = rotX * glm::vec4(centerToEye, 0);

  if(glm::sign(rotationVec.x) == glm::sign(centerToEye.x))
    centerToEye = rotationVec;

  // Make the vector as long as it was originally
  centerToEye *= radius;

  // Finding the new position
  glm::vec3 newPosition = centerToEye + origin;

  if(!invert)
  {
    m_current.eye = newPosition;  // Normal: change the position of the camera
  }
  else
  {
    m_current.ctr = newPosition;  // Inverted: change the interest point
  }
}

//--------------------------------------------------------------------------------------------------
// Move the camera toward the interest point, but don't cross it
//
void CameraManipulator::dolly(glm::vec2 displacement, bool keepCenterFixed /*= false*/)
{
  glm::vec3 directionVec = m_current.ctr - m_current.eye;
  float     length       = static_cast<float>(glm::length(directionVec));

  // We are at the point of interest, do nothing!
  if(length < 0.000001f)
    return;

  // Use the larger movement.
  float largerDisplacement = fabs(displacement.x) > fabs(displacement.y) ? displacement.x : -displacement.y;

  // Don't move over the point of interest.
  if(largerDisplacement >= 1.0f)
    return;

  directionVec *= largerDisplacement;

  // Not going up
  if(m_mode == Walk)
  {
    if(m_current.up.y > m_current.up.z)
      directionVec.y = 0;
    else
      directionVec.z = 0;
  }

  m_current.eye += directionVec;

  // In fly mode, the interest moves with us.
  if((m_mode == Fly || m_mode == Walk) && !keepCenterFixed)
  {
    m_current.ctr += directionVec;
  }
}

//--------------------------------------------------------------------------------------------------
// Modify the position of the camera over time
// - The camera can be updated through keys. A key set a direction which is added to both
//   eye and center, until the key is released
// - A new position of the camera is defined and the camera will reach that position
//   over time.
void CameraManipulator::updateAnim()
{
  auto elapse = static_cast<float>(getSystemTime() - m_startTime) / 1000.f;

  // Camera moving to new position
  if(m_animDone)
    return;

  float t = std::min(elapse / float(m_duration), 1.0f);
  // Evaluate polynomial (smoother step from Perlin)
  t = t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f);
  if(t >= 1.0f)
  {
    m_current  = m_goal;
    m_animDone = true;
    updateLookatMatrix();
    return;
  }

  // Interpolate camera position and interest
  // The distance of the camera between the interest is preserved to create a nicer interpolation
  m_current.ctr  = glm::mix(m_snapshot.ctr, m_goal.ctr, t);
  m_current.up   = glm::mix(m_snapshot.up, m_goal.up, t);
  m_current.eye  = computeBezier(t, m_bezier[0], m_bezier[1], m_bezier[2]);
  m_current.fov  = glm::mix(m_snapshot.fov, m_goal.fov, t);
  m_current.clip = glm::mix(m_snapshot.clip, m_goal.clip, t);

  updateLookatMatrix();
}

//--------------------------------------------------------------------------------------------------
//
void CameraManipulator::setMatrix(const glm::mat4& matrix, bool instantSet, float centerDistance)
{
  Camera camera;
  camera.eye = matrix[3];

  auto rotMat = glm::mat3(matrix);
  camera.ctr  = {0, 0, -centerDistance};
  camera.ctr  = camera.eye + (rotMat * camera.ctr);
  camera.up   = {0, 1, 0};
  camera.fov  = m_current.fov;

  m_animDone = instantSet;

  if(instantSet)
  {
    m_current = camera;
  }
  else
  {
    m_goal      = camera;
    m_snapshot  = m_current;
    m_startTime = getSystemTime();
    findBezierPoints();
  }
  updateLookatMatrix();
}

//--------------------------------------------------------------------------------------------------
// Low level function for when the camera move.
void CameraManipulator::motion(const glm::vec2& screenDisplacement, Actions action /*= 0*/)
{
  glm::vec2 displacement = {
      float(screenDisplacement.x - m_mouse[0]) / float(m_windowSize.x),
      float(screenDisplacement.y - m_mouse[1]) / float(m_windowSize.y),
  };

  switch(action)
  {
    case Orbit:
      orbit(displacement, false);
      break;
    case CameraManipulator::Dolly:
      dolly(displacement);
      break;
    case CameraManipulator::Pan:
      pan(displacement);
      break;
    case CameraManipulator::LookAround:
      orbit({displacement.x, -displacement.y}, true);
      break;
  }

  // Resetting animation and update the camera
  m_animDone = true;
  updateLookatMatrix();

  m_mouse = screenDisplacement;
}

//--------------------------------------------------------------------------------------------------
// Function for when the camera move with keys (ex. WASD).
// Note: dx and dy are the speed of the camera movement.
void CameraManipulator::keyMotion(glm::vec2 delta, Actions action)
{
  float movementSpeed = m_speed;

  auto directionVector = glm::normalize(m_current.ctr - m_current.eye);  // Vector from eye to center
  delta *= movementSpeed;

  glm::vec3 keyboardMovementVector{0, 0, 0};
  if(action == Dolly)
  {
    keyboardMovementVector = directionVector * delta.x;
    if(m_mode == Walk)
    {
      if(m_current.up.y > m_current.up.z)
        keyboardMovementVector.y = 0;
      else
        keyboardMovementVector.z = 0;
    }
  }
  else if(action == Pan)
  {
    auto rightVector       = glm::cross(directionVector, m_current.up);
    keyboardMovementVector = rightVector * delta.x + m_current.up * delta.y;
  }

  m_current.eye += keyboardMovementVector;
  m_current.ctr += keyboardMovementVector;

  // Resetting animation and update the camera
  m_animDone = true;
  updateLookatMatrix();
}

//--------------------------------------------------------------------------------------------------
// To call when the mouse is moving
// It find the appropriate camera operator, based on the mouse button pressed and the
// keyboard modifiers (shift, ctrl, alt)
//
// Returns the action that was activated
//
CameraManipulator::Actions CameraManipulator::mouseMove(glm::vec2 screenDisplacement, const Inputs& inputs)
{
  if(!inputs.lmb && !inputs.rmb && !inputs.mmb)
  {
    setMousePosition(screenDisplacement);
    return NoAction;  // no mouse button pressed
  }

  Actions curAction = NoAction;
  if(inputs.lmb)
  {
    if(((inputs.ctrl) && (inputs.shift)) || inputs.alt)
      curAction = m_mode == Examine ? LookAround : Orbit;
    else if(inputs.shift)
      curAction = Dolly;
    else if(inputs.ctrl)
      curAction = Pan;
    else
      curAction = m_mode == Examine ? Orbit : LookAround;
  }
  else if(inputs.mmb)
    curAction = Pan;
  else if(inputs.rmb)
    curAction = Dolly;

  if(curAction != NoAction)
    motion(screenDisplacement, curAction);

  return curAction;
}

//--------------------------------------------------------------------------------------------------
// Trigger a dolly when the wheel change, or change the FOV if the shift key was pressed
//
void CameraManipulator::wheel(float value, const Inputs& inputs)
{
  float deltaX = (value * fabsf(value)) / static_cast<float>(m_windowSize.x);

  if(inputs.shift)
  {
    setFov(m_current.fov + value);
  }
  else
  {
    // Dolly in or out. CTRL key keeps center fixed, which has for side effect to adjust the speed for fly/walk mode
    dolly(glm::vec2(deltaX), inputs.ctrl);
    updateLookatMatrix();
  }
}

// Set and clamp FOV between 0.01 and 179 degrees
void CameraManipulator::setFov(float fovDegree)
{
  m_current.fov = std::min(std::max(fovDegree, 0.01f), 179.0f);
}

glm::vec3 CameraManipulator::computeBezier(float t, glm::vec3& p0, glm::vec3& p1, glm::vec3& p2)
{
  float u  = 1.f - t;
  float tt = t * t;
  float uu = u * u;

  glm::vec3 p = uu * p0;  // first term
  p += 2 * u * t * p1;    // second term
  p += tt * p2;           // third term

  return p;
}

void CameraManipulator::findBezierPoints()
{
  glm::vec3 p0 = m_current.eye;
  glm::vec3 p2 = m_goal.eye;
  glm::vec3 p1, pc;

  // point of interest
  glm::vec3 pi = (m_goal.ctr + m_current.ctr) * 0.5f;

  glm::vec3 p02    = (p0 + p2) * 0.5f;                            // mid p0-p2
  float     radius = (length(p0 - pi) + length(p2 - pi)) * 0.5f;  // Radius for p1
  glm::vec3 p02pi(p02 - pi);                                      // Vector from interest to mid point
  p02pi = glm::normalize(p02pi);
  p02pi *= radius;
  pc   = pi + p02pi;                        // Calculated point to go through
  p1   = 2.f * pc - p0 * 0.5f - p2 * 0.5f;  // Computing p1 for t=0.5
  p1.y = p02.y;                             // Clamping the P1 to be in the same height as p0-p2

  m_bezier[0] = p0;
  m_bezier[1] = p1;
  m_bezier[2] = p2;
}

//--------------------------------------------------------------------------------------------------
// Return the time in fraction of milliseconds
//
double CameraManipulator::getSystemTime()
{
  auto now(std::chrono::system_clock::now());
  auto duration = now.time_since_epoch();
  return std::chrono::duration_cast<std::chrono::microseconds>(duration).count() / 1000.0;
}

//--------------------------------------------------------------------------------------------------
// Return a string which can be included in help dialogs
//
const std::string& CameraManipulator::getHelp()
{
  static std::string helpText =
      "LMB: rotate around the target\n"
      "RMB: Dolly in/out\n"
      "MMB: Pan along view plane\n"
      "LMB + Shift: Dolly in/out\n"
      "LMB + Ctrl: Pan\n"
      "LMB + Alt: Look aroundPan\n"
      "Mouse wheel: Dolly in/out\n"
      "Mouse wheel + Shift: Zoom in/out\n";
  return helpText;
}

//--------------------------------------------------------------------------------------------------
// Move the camera closer or further from the center of the the bounding box, to see it completely
//
// boxMin - lower corner of the bounding box
// boxMax - upper corner of the bounding box
// instantFit - true: set the new position, false: will animate to new position.
// tight - true: fit exactly the corner, false: fit to radius (larger view, will not get closer or further away)
// aspect - aspect ratio of the window.
//
void CameraManipulator::fit(const glm::vec3& boxMin, const glm::vec3& boxMax, bool instantFit /*= true*/, bool tightFit /*=false*/, float aspect /*=1.0f*/)
{
  // Calculate the half extents of the bounding box
  const glm::vec3 boxHalfSize = 0.5f * (boxMax - boxMin);

  // Calculate the center of the bounding box
  const glm::vec3 boxCenter = 0.5f * (boxMin + boxMax);

  const float yfov = tan(glm::radians(m_current.fov * 0.5f));
  const float xfov = yfov * aspect;

  // Calculate the ideal distance for a tight fit or fit to radius
  float idealDistance = 0;

  if(tightFit)
  {
    // Get only the rotation matrix
    glm::mat3 mView = glm::lookAt(m_current.eye, boxCenter, m_current.up);

    // Check each 8 corner of the cube
    for(int i = 0; i < 8; i++)
    {
      // Rotate the bounding box in the camera view
      glm::vec3 vct(i & 1 ? boxHalfSize.x : -boxHalfSize.x,   //
                    i & 2 ? boxHalfSize.y : -boxHalfSize.y,   //
                    i & 4 ? boxHalfSize.z : -boxHalfSize.z);  //
      vct = mView * vct;

      if(vct.z < 0)  // Take only points in front of the center
      {
        // Keep the largest offset to see that vertex
        idealDistance = std::max(fabsf(vct.y) / yfov + fabsf(vct.z), idealDistance);
        idealDistance = std::max(fabsf(vct.x) / xfov + fabsf(vct.z), idealDistance);
      }
    }
  }
  else  // Using the bounding sphere
  {
    const float radius = glm::length(boxHalfSize);
    idealDistance      = std::max(radius / xfov, radius / yfov);
  }

  // Calculate the new camera position based on the ideal distance
  const glm::vec3 newEye = boxCenter - idealDistance * glm::normalize(boxCenter - m_current.eye);

  // Set the new camera position and interest point
  setLookat(newEye, boxCenter, m_current.up, instantFit);
}

std::string CameraManipulator::Camera::getString() const
{
  return fmt::format("{{{}, {}, {}}}, {{{}, {}, {}}}, {{{}, {}, {}}}, {{{}}}, {{{}, {}}}",  //
                     eye.x, eye.y, eye.z,                                                   //
                     ctr.x, ctr.y, ctr.z,                                                   //
                     up.x, up.y, up.z,                                                      //
                     fov, clip.x, clip.y);
}

bool CameraManipulator::Camera::setFromString(const std::string& text)
{
  if(text.empty())
    return false;

  std::array<float, 12> val{};
  int result = SAFE_SSCANF(text.c_str(), "{%f, %f, %f}, {%f, %f, %f}, {%f, %f, %f}, {%f}, {%f, %f}", &val[0], &val[1],
                           &val[2], &val[3], &val[4], &val[5], &val[6], &val[7], &val[8], &val[9], &val[10], &val[11]);
  if(result >= 9)  // Before 2025-09-03, this format didn't include the FOV at the end
  {
    eye = glm::vec3{val[0], val[1], val[2]};
    ctr = glm::vec3{val[3], val[4], val[5]};
    up  = glm::vec3{val[6], val[7], val[8]};
    if(result >= 10)
      fov = val[9];
    if(result >= 11)
      clip = glm::vec2{val[10], val[11]};

    return true;
  }
  return false;
}

}  // namespace nvutils
