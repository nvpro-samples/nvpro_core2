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

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fmt/format.h>

#include "camera_manipulator.hpp"
#include "logger.hpp"

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
  if(!validateCamera(camera))
  {
    LOGW("CameraManipulator::setCamera: Invalid camera parameters\n");
    return;
  }

  camera.up     = glm::normalize(camera.up);
  m_isAnimating = false;

  // Force instant transition if projection type changes
  if(camera.projectionType != m_current.projectionType)
  {
    instantSet = true;
  }

  if(instantSet || m_duration == 0.0)
  {
    applyCameraInstant(camera);
  }
  else if(camera != m_current)
  {
    startAnimationTo(camera);
  }
}


//--------------------------------------------------------------------------------------------------
// Creates a viewing matrix derived from an eye point, a reference point indicating the center of
// the scene, and an up vector
//
void CameraManipulator::setLookat(const glm::vec3& eye, const glm::vec3& center, const glm::vec3& up, bool instantSet)
{
  Camera cam = m_current;  // preserve projection, clip, orthographic size, etc.
  cam.eye    = eye;
  cam.ctr    = center;
  cam.up     = up;
  if(!validateCamera(cam))
  {
    LOGW("CameraManipulator::setLookat: Invalid camera parameters\n");
    return;
  }
  setCamera(cam, instantSet);
}

//-----------------------------------------------------------------------------
// Get the current camera's look-at parameters.
void CameraManipulator::getLookat(glm::vec3& eye, glm::vec3& center, glm::vec3& up) const
{
  eye    = m_current.eye;
  center = m_current.ctr;
  up     = m_current.up;
}

float CameraManipulator::getAnimationProgress() const
{
  if(!m_isAnimating)
    return 1.0f;

  const float elapsed = static_cast<float>((getSystemTime() - m_startTime) / 1000.0);
  return std::min(elapsed / static_cast<float>(m_duration), 1.0f);
}

void CameraManipulator::setWindowSize(glm::uvec2 winSize)
{
  if(winSize.x == 0 || winSize.y == 0)
  {
    LOGW("CameraManipulator::setWindowSize: Invalid window size\n");
    return;
  }
  m_windowSize = winSize;
}

void CameraManipulator::setSpeed(float speed)
{
  m_speed = speed;
}

void CameraManipulator::setClipPlanes(glm::vec2 nearFar)
{
  if(nearFar.x <= 0.0f || nearFar.y <= nearFar.x)
  {
    LOGW("CameraManipulator::setClipPlanes: Invalid clip planes\n");
    return;
  }
  m_current.nearFar = nearFar;
}

void CameraManipulator::setOrthographicMagnitudes(const glm::vec2& mag)
{
  if(mag.x <= 0.0f || mag.y <= 0.0f)
  {
    LOGW("CameraManipulator::setOrthographicMagnitudes: Magnitudes must be positive\n");
    return;
  }
  m_current.orthMag = mag;
}

void CameraManipulator::setAnimationDuration(double val)
{
  if(val < 0.0)
  {
    LOGW("CameraManipulator::setAnimationDuration: Duration must be non-negative\n");
    return;
  }
  m_duration = val;
}

CameraManipulator::ViewDimensions CameraManipulator::getViewDimensions() const
{
  if(m_current.projectionType == ProjectionType::Orthographic)
  {
    return {m_current.orthMag.x * 2.0f, m_current.orthMag.y * 2.0f};
  }

  const float distance   = glm::length(m_current.eye - m_current.ctr);
  const float halfHeight = distance * std::tan(getRadFov() * 0.5f);
  const float viewHeight = 2.0f * halfHeight;
  const float viewWidth  = viewHeight * std::max(getAspectRatio(), CameraConstants::MIN_ASPECT_RATIO);
  return {viewWidth, viewHeight};
}

CameraManipulator::CameraFrame CameraManipulator::computeCameraFrame() const
{
  CameraFrame     frame;
  const glm::vec3 viewDelta = m_current.ctr - m_current.eye;
  if(glm::length(viewDelta) < CameraConstants::EPSILON)
  {
    frame.forward = glm::vec3(0, 0, -1);
    frame.right   = glm::vec3(1, 0, 0);
    frame.up      = glm::vec3(0, 1, 0);
    return frame;
  }

  frame.forward   = glm::normalize(viewDelta);
  glm::vec3 right = glm::cross(frame.forward, m_current.up);
  if(glm::dot(right, right) < CameraConstants::EPSILON)
  {
    const glm::vec3 fallbackUp = std::abs(frame.forward.y) < 0.99f ? glm::vec3(0, 1, 0) : glm::vec3(1, 0, 0);
    right                      = glm::cross(frame.forward, fallbackUp);
  }
  frame.right = glm::normalize(right);
  frame.up    = glm::cross(frame.right, frame.forward);
  return frame;
}

glm::vec3 CameraManipulator::projectToGroundPlane(const glm::vec3& vec) const
{
  const float upLen2 = glm::dot(m_current.up, m_current.up);
  if(upLen2 < CameraConstants::EPSILON)
    return vec;

  const float projection = glm::dot(vec, m_current.up) / upLen2;
  return vec - projection * m_current.up;
}

void CameraManipulator::zoomOrthographic(float factor)
{
  m_current.orthMag.x = std::max(m_current.orthMag.x * factor, CameraConstants::MIN_ORTHOGRAPHIC_SIZE);
  m_current.orthMag.y = std::max(m_current.orthMag.y * factor, CameraConstants::MIN_ORTHOGRAPHIC_SIZE);
}

void CameraManipulator::updateLookatMatrix()
{
  m_matrix = glm::lookAt(m_current.eye, m_current.ctr, m_current.up);
}

void CameraManipulator::applyCameraInstant(const Camera& camera)
{
  m_current = camera;
  m_snapshot.reset();
  m_isAnimating = false;
  updateLookatMatrix();
}

void CameraManipulator::startAnimationTo(const Camera& camera)
{
  m_goal        = camera;
  m_snapshot    = m_current;
  m_startTime   = getSystemTime();
  m_isAnimating = true;
  findBezierPoints();

  // Calculate the Dolly-zoom style FOV: keep apparent size consistent from start to end. (Vertigo effect)
  const float d0   = glm::length(m_snapshot->eye - m_snapshot->ctr);
  const float d1   = glm::length(m_goal.eye - m_goal.ctr);
  m_animDollyZoom0 = d0 * std::tan(glm::radians(m_snapshot->fov * 0.5f));
  m_animDollyZoom1 = d1 * std::tan(glm::radians(m_goal.fov * 0.5f));
}

void CameraManipulator::applyUserChange(bool updateMatrix)
{
  m_isAnimating = false;
  if(updateMatrix)
  {
    updateLookatMatrix();
  }
}

//--------------------------------------------------------------------------------------------------
// Pan the camera perpendicularly to the light of sight.
void CameraManipulator::pan(glm::vec2 displacement)
{
  if(displacement == glm::vec2(0.f, 0.f))
    return;

  if(m_mode == Modes::Fly)
  {
    displacement *= -1.f;
  }

  const CameraFrame    frame = computeCameraFrame();
  const ViewDimensions view  = getViewDimensions();

  glm::vec3 panOffset = (-displacement.x * frame.right * view.width) + (displacement.y * frame.up * view.height);
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
  if(radius < CameraConstants::EPSILON)
    return;
  centerToEye = glm::normalize(centerToEye);

  // Find the rotation around the UP axis (Y)
  glm::mat4 rotY = glm::rotate(glm::mat4(1), -displacement.x, m_current.up);

  // Apply the (Y) rotation to the eye-center vector
  centerToEye = rotY * glm::vec4(centerToEye, 0);

  // Find the rotation around the X vector: cross between eye-center and up (X)
  glm::vec3 axeX = glm::cross(m_current.up, centerToEye);
  if(glm::dot(axeX, axeX) < CameraConstants::EPSILON)
    return;
  axeX           = glm::normalize(axeX);
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
// For orthographic cameras, this adjusts the orthographic size (zoom) instead
//
void CameraManipulator::dolly(glm::vec2 displacement, bool keepCenterFixed /*= false*/)
{
  // Use the larger movement.
  float largerDisplacement = (fabs(displacement.x) > fabs(displacement.y)) ? displacement.x : -displacement.y;

  // For orthographic cameras, adjust the size (zoom)
  if(m_current.projectionType == ProjectionType::Orthographic)
  {
    float zoomFactor = 1.0f - largerDisplacement;
    zoomOrthographic(zoomFactor);
    return;
  }

  // Perspective camera: move camera position
  glm::vec3 directionVec = m_current.ctr - m_current.eye;
  float     length       = static_cast<float>(glm::length(directionVec));

  // We are at the point of interest, do nothing!
  if(length < CameraConstants::MIN_DISTANCE)
    return;

  // Don't move over the point of interest.
  if(largerDisplacement >= CameraConstants::MAX_DOLLY_DISPLACEMENT)
    return;

  directionVec *= largerDisplacement;

  // Not going up
  if(m_mode == Modes::Walk)
  {
    directionVec = projectToGroundPlane(directionVec);
  }

  m_current.eye += directionVec;

  // In fly mode, the interest moves with us.
  if((m_mode == Modes::Fly || m_mode == Modes::Walk) && !keepCenterFixed)
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
void CameraManipulator::updateAnim(double currentTimeMs)
{
  if(currentTimeMs < 0.0)
    currentTimeMs = getSystemTime();

  auto elapse = static_cast<float>(currentTimeMs - m_startTime) / 1000.f;

  // Camera moving to new position
  if(!m_isAnimating)
    return;

  float t = std::min(elapse / float(m_duration), 1.0f);
  // Evaluate polynomial (smoother step from Perlin)
  t = t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f);
  if(t >= 1.0f)
  {
    m_current     = m_goal;
    m_isAnimating = false;
    m_snapshot.reset();
    updateLookatMatrix();
    return;
  }

  // Interpolate camera position and interest
  // The distance of the camera between the interest is preserved to create a nicer interpolation
  if(!m_snapshot)
  {
    m_isAnimating = false;
    return;
  }

  m_current.ctr = glm::mix(m_snapshot->ctr, m_goal.ctr, t);
  m_current.up  = glm::mix(m_snapshot->up, m_goal.up, t);
  m_current.eye = computeBezier(t, m_bezier[0], m_bezier[1], m_bezier[2]);

  // Dolly-zoom style FOV: keep apparent size consistent from start to end. (Vertigo effect)
  const float distance = glm::length(m_current.eye - m_current.ctr);
  const float k        = glm::mix(m_animDollyZoom0, m_animDollyZoom1, t);
  if(distance > CameraConstants::EPSILON && k > 0.0f)
  {
    m_current.fov = glm::degrees(2.0f * std::atan(k / distance));
    m_current.fov = glm::clamp(m_current.fov, CameraConstants::MIN_FOV, CameraConstants::MAX_FOV);
  }
  else
  {
    m_current.fov = glm::mix(m_snapshot->fov, m_goal.fov, t);
  }
  m_current.nearFar = glm::mix(m_snapshot->nearFar, m_goal.nearFar, t);
  m_current.orthMag = glm::mix(m_snapshot->orthMag, m_goal.orthMag, t);

  updateLookatMatrix();
}

//--------------------------------------------------------------------------------------------------
//
void CameraManipulator::setMatrix(const glm::mat4& matrix, bool instantSet, float centerDistance)
{
  Camera camera = m_current;

  auto      rotMat        = glm::mat3(matrix);
  glm::vec3 forwardVector = rotMat * glm::vec3(0, 0, -centerDistance);

  camera.eye = matrix[3];
  camera.ctr = camera.eye + forwardVector;
  camera.up  = {0, 1, 0};

  if(!validateCamera(camera))
  {
    LOGW("CameraManipulator::setMatrix: Invalid camera parameters\n");
    return;
  }

  if(instantSet)
  {
    applyCameraInstant(camera);
  }
  else
  {
    startAnimationTo(camera);
  }
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
    case Actions::Orbit:
      orbit(displacement, false);
      break;
    case Actions::Dolly:
      dolly(displacement);
      break;
    case Actions::Pan:
      pan(displacement);
      break;
    case Actions::LookAround:
      orbit({displacement.x, -displacement.y}, true);
      break;
    default:
      break;
  }

  // Resetting animation and update the camera
  applyUserChange();

  m_mouse = screenDisplacement;
}

//--------------------------------------------------------------------------------------------------
// Function for when the camera move with keys (ex. WASD).
// Note: dx and dy are the speed of the camera movement.
void CameraManipulator::keyMotion(glm::vec2 delta, Actions action)
{
  if(delta == glm::vec2(0.f, 0.f))
    return;

  float movementSpeed = m_speed;

  const CameraFrame frame = computeCameraFrame();
  delta *= movementSpeed;

  glm::vec3 keyboardMovementVector{0, 0, 0};
  if(action == Actions::Dolly)
  {
    keyboardMovementVector = frame.forward * delta.x;
    if(m_mode == Modes::Walk)
    {
      keyboardMovementVector = projectToGroundPlane(keyboardMovementVector);
    }
  }
  else if(action == Actions::Pan)
  {
    keyboardMovementVector = frame.right * delta.x + frame.up * delta.y;
  }

  m_current.eye += keyboardMovementVector;
  m_current.ctr += keyboardMovementVector;

  // Resetting animation and update the camera
  applyUserChange();
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
    return Actions::NoAction;  // no mouse button pressed
  }

  Actions curAction = Actions::NoAction;
  if(inputs.lmb)
  {
    if(((inputs.ctrl) && (inputs.shift)) || inputs.alt)
      curAction = m_mode == Modes::Examine ? Actions::LookAround : Actions::Orbit;
    else if(inputs.shift)
      curAction = Actions::Dolly;
    else if(inputs.ctrl)
      curAction = Actions::Pan;
    else
      curAction = m_mode == Modes::Examine ? Actions::Orbit : Actions::LookAround;
  }
  else if(inputs.mmb)
    curAction = Actions::Pan;
  else if(inputs.rmb)
    curAction = Actions::Dolly;

  if(curAction != Actions::NoAction)
    motion(screenDisplacement, curAction);

  return curAction;
}

//--------------------------------------------------------------------------------------------------
// Trigger a dolly when the wheel change, or change the FOV/ortho-size if the shift key was pressed
//
void CameraManipulator::wheel(float value, const Inputs& inputs)
{
  if(value == 0.0f)
    return;

  float deltaX = (value * fabsf(value)) / static_cast<float>(m_windowSize.x);

  if(inputs.shift)
  {
    if(m_current.projectionType == ProjectionType::Orthographic)
    {

      float zoomFactor = 1.0f + deltaX;
      zoomOrthographic(zoomFactor);
      applyUserChange();
    }
    else
    {
      // For perspective cameras, adjust FOV
      setFov(m_current.fov + value);
      applyUserChange(false);
    }
  }
  else
  {
    // Dolly in or out. CTRL key keeps center fixed, which has for side effect to adjust the speed for fly/walk mode
    dolly(glm::vec2(deltaX), inputs.ctrl);
    applyUserChange();
  }
}

//--------------------------------------------------------------------------------------------------
// Adjust the orthographic camera's aspect ratio to match the current viewport.
// Sets the orthographic width (xmag) to be the height (ymag) multiplied by the aspect ratio.
void CameraManipulator::adjustOrthographicAspect()
{
  if(m_current.projectionType != ProjectionType::Orthographic)
    return;

  const float aspect = getAspectRatio();
  if(aspect <= 0.0f)
    return;

  const float height = m_current.orthMag.y;
  const float width  = height * aspect;
  if(width <= 0.0f)
    return;

  if(std::abs(width - m_current.orthMag.x) > CameraConstants::EPSILON)
  {
    m_current.orthMag.x = width;
    m_current.orthMag.y = height;
  }
}

// Set and clamp FOV between 0.01 and 179 degrees
void CameraManipulator::setFov(float fovDegree)
{
  m_current.fov = std::min(std::max(fovDegree, CameraConstants::MIN_FOV), CameraConstants::MAX_FOV);
}

//--------------------------------------------------------------------------------------------------
// Convert from orthographic to perspective projection, preserving the view at center point
//
void CameraManipulator::convertToPerspective()
{
  if(m_current.projectionType == ProjectionType::Perspective)
    return;  // Already perspective

  // Calculate FOV based on the orthographic viewport and distance to center
  const float distance = glm::length(m_current.eye - m_current.ctr);
  if(distance > 0.0f && m_current.orthMag.y > 0.0f)
  {
    // FOV = 2 * atan(ymag / distance)
    m_current.fov = glm::degrees(2.0f * std::atan(m_current.orthMag.y / distance));
    // Clamp to reasonable range
    m_current.fov = glm::clamp(m_current.fov, CameraConstants::MIN_FOV, CameraConstants::MAX_FOV);
  }

  m_current.projectionType = ProjectionType::Perspective;
}

//--------------------------------------------------------------------------------------------------
// Convert from perspective to orthographic projection, preserving the view at center point
//
void CameraManipulator::convertToOrthographic()
{
  if(m_current.projectionType == ProjectionType::Orthographic)
    return;  // Already orthographic

  // Calculate orthographic viewport based on FOV and distance to center
  const float distance = glm::length(m_current.eye - m_current.ctr);
  if(distance > 0.0f)
  {
    // visibleHeight = 2 * distance * tan(fov/2)
    // ymag = visibleHeight / 2 = distance * tan(fov/2)
    const float halfFovRad = glm::radians(m_current.fov * 0.5f);
    m_current.orthMag.y    = distance * std::tan(halfFovRad);
    m_current.orthMag.x    = m_current.orthMag.y * getAspectRatio();
  }

  m_current.projectionType = ProjectionType::Orthographic;
}

// Quadratic Bezier curve: B(t) = (1-t)^2*p0 + 2*(1-t)*t*p1 + t^2*p2
glm::vec3 CameraManipulator::computeBezier(float t, const glm::vec3& p0, const glm::vec3& p1, const glm::vec3& p2) const
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
  if(!m_snapshot)
    return;

  // Compute a smooth arc in view space between current and goal positions.
  const glm::vec3 p0 = m_current.eye;
  const glm::vec3 p2 = m_goal.eye;
  // Point of interest (center)
  const glm::vec3 pi = (m_goal.ctr + m_snapshot->ctr) * 0.5f;
  // Midpoint between endpoints
  const glm::vec3 mid = (p0 + p2) * 0.5f;

  // Radius based on average distance to interest
  const float radius = 0.5f * (glm::length(p0 - pi) + glm::length(p2 - pi));
  // Vector from interest to the midpoint
  glm::vec3 toMid = mid - pi;
  if(glm::dot(toMid, toMid) < CameraConstants::EPSILON)
  {
    toMid = glm::vec3(0, 0, 1);
  }
  // Calculated point to pass through
  const glm::vec3 pc = pi + radius * glm::normalize(toMid);
  // Compute control point so curve goes through pc at t=0.5
  glm::vec3 p1 = 2.0f * pc - 0.5f * (p0 + p2);

  // Project onto plane perpendicular to average up vector to avoid Y-up assumptions
  const glm::vec3 avgUp      = glm::normalize(m_snapshot->up + m_goal.up);
  const float     projection = glm::dot(mid - p1, avgUp);
  p1 += projection * avgUp;

  m_bezier[0] = p0;
  m_bezier[1] = p1;
  m_bezier[2] = p2;
}

//--------------------------------------------------------------------------------------------------
// Return the time in fraction of milliseconds
//
double CameraManipulator::getSystemTime() const
{
  auto now(std::chrono::system_clock::now());
  auto duration = now.time_since_epoch();
  return std::chrono::duration_cast<std::chrono::microseconds>(duration).count() / 1000.0;
}

bool CameraManipulator::isValidPosition(const glm::vec3& pos)
{
  return !std::isnan(pos.x) && !std::isnan(pos.y) && !std::isnan(pos.z) && !std::isinf(pos.x) && !std::isinf(pos.y)
         && !std::isinf(pos.z);
}

bool CameraManipulator::isValidDirection(const glm::vec3& dir)
{
  float len = glm::length(dir);
  return isValidPosition(dir) && len > CameraConstants::EPSILON;
}

bool CameraManipulator::validateCamera(const Camera& cam) const
{
  if(!isValidPosition(cam.eye) || !isValidPosition(cam.ctr) || !isValidDirection(cam.up))
    return false;

  if(glm::distance(cam.eye, cam.ctr) < CameraConstants::MIN_DISTANCE)
    return false;

  if(cam.fov < CameraConstants::MIN_FOV || cam.fov > CameraConstants::MAX_FOV)
    return false;

  return true;
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
      "LMB + Alt: Look around\n"
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
  return fmt::format("{{{}, {}, {}}}, {{{}, {}, {}}}, {{{}, {}, {}}}, {{{}}}, {{{}, {}}}, {{{}, {}}}, {{{}}}",  //
                     eye.x, eye.y, eye.z,                                                                       //
                     ctr.x, ctr.y, ctr.z,                                                                       //
                     up.x, up.y, up.z,                                                                          //
                     fov,                                                                                       //
                     nearFar.x, nearFar.y,                                                                      //
                     orthMag.x, orthMag.y,                                                                      //
                     static_cast<int>(projectionType));
}

bool CameraManipulator::Camera::setFromString(const std::string& text)
{
  if(text.empty())
    return false;

  std::array<float, 16> val{};
  int result = SAFE_SSCANF(text.c_str(), "{%f, %f, %f}, {%f, %f, %f}, {%f, %f, %f}, {%f}, {%f, %f}, {%f, %f}, {%f}",
                           &val[0], &val[1], &val[2], &val[3], &val[4], &val[5], &val[6], &val[7], &val[8], &val[9],
                           &val[10], &val[11], &val[12], &val[13], &val[14]);
  if(result >= 9)  // Before 2025-09-03, this format didn't include the FOV at the end
  {
    eye = glm::vec3{val[0], val[1], val[2]};
    ctr = glm::vec3{val[3], val[4], val[5]};
    up  = glm::vec3{val[6], val[7], val[8]};
    if(result >= 10)
      fov = val[9];
    if(result >= 12)
      nearFar = glm::vec2{val[10], val[11]};
    if(result >= 14)
      orthMag = glm::vec2{val[12], val[13]};
    if(result >= 15)
      projectionType = static_cast<ProjectionType>(static_cast<int>(val[14]));

    return true;
  }
  return false;
}

}  // namespace nvutils
