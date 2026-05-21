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

namespace {

// Camera plugins are shared across all CameraManipulator instances
// (though usually there's only one).
std::vector<std::shared_ptr<CameraPlugin>> s_cameraPlugins;

// Returns 'up' if it is not parallel to the view direction, otherwise a world axis that is.
// glm::lookAt and any cross(forward, up) basis construction would otherwise produce NaN.
glm::dvec3 effectiveUp(const glm::dvec3& eye, const glm::dvec3& ctr, const glm::dvec3& up)
{
  const glm::dvec3 right = glm::cross(ctr - eye, up);
  if(glm::dot(right, right) > CameraConstants::EPSILON)
    return up;
  return std::abs(up.y) < 0.9 ? glm::dvec3(0, 1, 0) : glm::dvec3(1, 0, 0);
}
}  // namespace

CameraManipulator::CameraManipulator()
{
  updateLookatMatrix();
}

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

void CameraManipulator::setLookat(const glm::dvec3& eye, const glm::dvec3& center, const glm::dvec3& up, bool instantSet)
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

void CameraManipulator::getLookat(glm::dvec3& eye, glm::dvec3& center, glm::dvec3& up) const
{
  eye    = m_current.eye;
  center = m_current.ctr;
  up     = m_current.up;
}

double CameraManipulator::getAnimationProgress() const
{
  if(!m_isAnimating)
    return 1.0;

  const double elapsed = (getTimeMs() - m_startTime) * 1e-3;
  return std::min(elapsed / m_duration, 1.0);
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

void CameraManipulator::setSpeed(double speed)
{
  m_speed = speed;
}

void CameraManipulator::setClipPlanes(glm::dvec2 nearFar)
{
  if(nearFar.x <= 0.0 || nearFar.y <= nearFar.x)
  {
    LOGW("CameraManipulator::setClipPlanes: Invalid clip planes\n");
    return;
  }
  m_current.nearFar = nearFar;
}

void CameraManipulator::setOrthographicMagnitudes(const glm::dvec2& mag)
{
  if(mag.x <= 0.0 || mag.y <= 0.0)
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
    return {m_current.orthMag.x * 2.0, m_current.orthMag.y * 2.0};
  }

  const double distance   = glm::length(m_current.eye - m_current.ctr);
  const double halfHeight = distance * std::tan(getRadFov() * 0.5);
  const double viewHeight = 2.0 * halfHeight;
  const double viewWidth  = viewHeight * std::max(getAspectRatio(), CameraConstants::MIN_ASPECT_RATIO);
  return {viewWidth, viewHeight};
}

CameraManipulator::CameraFrame CameraManipulator::computeCameraFrame() const
{
  CameraFrame      frame;
  const glm::dvec3 viewDelta = m_current.ctr - m_current.eye;
  if(glm::length(viewDelta) < CameraConstants::EPSILON)
  {
    frame.forward = glm::dvec3(0, 0, -1);
    frame.right   = glm::dvec3(1, 0, 0);
    frame.up      = glm::dvec3(0, 1, 0);
    return frame;
  }

  frame.forward = glm::normalize(viewDelta);
  frame.right   = glm::normalize(glm::cross(frame.forward, effectiveUp(m_current.eye, m_current.ctr, m_current.up)));
  frame.up      = glm::cross(frame.right, frame.forward);
  return frame;
}

glm::dvec3 CameraManipulator::projectToGroundPlane(const glm::dvec3& vec) const
{
  const double upLen2 = glm::dot(m_current.up, m_current.up);
  if(upLen2 < CameraConstants::EPSILON)
    return vec;

  const double projection = glm::dot(vec, m_current.up) / upLen2;
  return vec - projection * m_current.up;
}

void CameraManipulator::zoomOrthographic(double factor)
{
  m_current.orthMag.x = std::max(m_current.orthMag.x * factor, CameraConstants::MIN_ORTHOGRAPHIC_SIZE);
  m_current.orthMag.y = std::max(m_current.orthMag.y * factor, CameraConstants::MIN_ORTHOGRAPHIC_SIZE);
}

void CameraManipulator::updateLookatMatrix()
{
  m_matrix = glm::lookAt(m_current.eye, m_current.ctr, effectiveUp(m_current.eye, m_current.ctr, m_current.up));
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
  m_startTime   = getTimeMs();
  m_isAnimating = true;
  findBezierPoints();

  // Dolly-zoom (vertigo effect): adjust FOV during a move to keep apparent size consistent.
  // Only enabled when FOV actually changes; otherwise the dolly-zoom formula fights the
  // Bezier eye path and causes FOV oscillations.
  const double d0  = glm::length(m_snapshot->eye - m_snapshot->ctr);
  const double d1  = glm::length(m_goal.eye - m_goal.ctr);
  m_animDollyZoom0 = d0 * std::tan(glm::radians(m_snapshot->fov * 0.5));
  m_animDollyZoom1 = d1 * std::tan(glm::radians(m_goal.fov * 0.5));
  m_vertigoEffect  = std::abs(m_snapshot->fov - m_goal.fov) > 0.001;
}

void CameraManipulator::applyUserChange(bool updateMatrix)
{
  m_isAnimating = false;
  if(updateMatrix)
  {
    updateLookatMatrix();
  }

  for(size_t pluginIdx = 0; pluginIdx < s_cameraPlugins.size(); pluginIdx++)
  {
    s_cameraPlugins[pluginIdx]->onUserInterrupt();
  }
}

void CameraManipulator::pan(glm::dvec2 displacement)
{
  if(displacement == glm::dvec2(0.0, 0.0))
    return;

  if(m_mode == Modes::Fly)
  {
    displacement *= -1.0;
  }

  const CameraFrame    frame = computeCameraFrame();
  const ViewDimensions view  = getViewDimensions();

  glm::dvec3 panOffset = (-displacement.x * frame.right * view.width) + (displacement.y * frame.up * view.height);
  m_current.eye += panOffset;
  m_current.ctr += panOffset;
}

void CameraManipulator::orbit(glm::dvec2 displacement, bool invert /*= false*/)
{
  if(displacement == glm::dvec2(0.0, 0.0))
    return;

  // Full width will do a full turn
  displacement *= glm::two_pi<double>();

  // Get the camera
  glm::dvec3 origin(invert ? m_current.eye : m_current.ctr);
  glm::dvec3 position(invert ? m_current.ctr : m_current.eye);

  // Get the length of sight
  glm::dvec3 centerToEye(position - origin);
  double     radius = glm::length(centerToEye);
  if(radius < CameraConstants::EPSILON)
    return;
  centerToEye = glm::normalize(centerToEye);

  // Decompose centerToEye into an elevation around 'up' and a horizontal direction:
  //   centerToEye = cos(elev) * up + sin(elev) * horizontal
  // Yaw rotates 'horizontal' around 'up', pitch changes 'elev'.
  constexpr double kPolePad   = 1e-3;
  const double     cosElev    = glm::dot(centerToEye, m_current.up);
  glm::dvec3       horizontal = centerToEye - cosElev * m_current.up;
  const double     sinElev    = glm::length(horizontal);
  const double     elev       = std::atan2(sinElev, cosElev);

  if(sinElev < CameraConstants::EPSILON)
  {
    // At the pole, 'horizontal' is undefined. Pick any direction perpendicular to 'up'.
    const glm::dvec3 ref = (std::abs(m_current.up.x) < 0.9) ? glm::dvec3(1.0, 0.0, 0.0) : glm::dvec3(0.0, 0.0, 1.0);
    horizontal           = glm::normalize(ref - glm::dot(ref, m_current.up) * m_current.up);
  }
  else
  {
    horizontal /= sinElev;
  }

  // Yaw around 'up'
  const double yawC = std::cos(-displacement.x);
  const double yawS = std::sin(-displacement.x);
  horizontal        = yawC * horizontal + yawS * glm::cross(m_current.up, horizontal);

  // Pitch, clamped to keep centerToEye off the poles
  const double newElev = glm::clamp(elev - displacement.y, kPolePad, glm::pi<double>() - kPolePad);

  centerToEye = (std::cos(newElev) * m_current.up + std::sin(newElev) * horizontal) * radius;

  // Finding the new position
  glm::dvec3 newPosition = centerToEye + origin;

  if(!invert)
  {
    m_current.eye = newPosition;  // Normal: change the position of the camera
  }
  else
  {
    m_current.ctr = newPosition;  // Inverted: change the interest point
  }
}

void CameraManipulator::dolly(glm::dvec2 displacement, bool keepCenterFixed /*= false*/)
{
  // Use the larger movement.
  double largerDisplacement = (std::abs(displacement.x) > std::abs(displacement.y)) ? displacement.x : -displacement.y;

  // For orthographic cameras, adjust the size (zoom)
  if(m_current.projectionType == ProjectionType::Orthographic)
  {
    double zoomFactor = 1.0 - largerDisplacement;
    zoomOrthographic(zoomFactor);
    return;
  }

  // Perspective camera: move camera position
  glm::dvec3 directionVec = m_current.ctr - m_current.eye;
  double     length       = glm::length(directionVec);

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

void CameraManipulator::updateAnim(double currentTimeMs)
{
  // Camera moving to new position
  if(m_isAnimating)
  {
    if(currentTimeMs < 0.0)
      currentTimeMs = getTimeMs();

    const double elapsedTime = (currentTimeMs - m_startTime) * 1e-3;
    baseAnimation(elapsedTime);
  }

  for(size_t pluginIdx = 0; pluginIdx < s_cameraPlugins.size(); pluginIdx++)
  {
    s_cameraPlugins[pluginIdx]->onUpdateAnim(*this);
  }

  // If something might have changed:
  if(m_isAnimating || s_cameraPlugins.size() > 0)
  {
    updateLookatMatrix();
  }
}

void CameraManipulator::baseAnimation(double elapsedTime)
{
  double t = std::min(elapsedTime / m_duration, 1.0);
  // Evaluate polynomial (smoother step from Perlin)
  t = t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
  if(t >= 1.0)
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

  // Dolly-zoom style FOV: keep apparent size consistent from start to end (vertigo effect).
  const double distance = glm::length(m_current.eye - m_current.ctr);
  if(m_vertigoEffect && distance > CameraConstants::EPSILON)
  {
    const double k = glm::mix(m_animDollyZoom0, m_animDollyZoom1, t);
    m_current.fov  = glm::degrees(2.0 * std::atan(k / distance));
    m_current.fov  = glm::clamp(m_current.fov, CameraConstants::MIN_FOV, CameraConstants::MAX_FOV);
  }
  else
  {
    m_current.fov = glm::mix(m_snapshot->fov, m_goal.fov, t);
  }
  m_current.nearFar = glm::mix(m_snapshot->nearFar, m_goal.nearFar, t);
  m_current.orthMag = glm::mix(m_snapshot->orthMag, m_goal.orthMag, t);
}

void CameraManipulator::setMatrix(const glm::dmat4& matrix, bool instantSet, double centerDistance)
{
  Camera camera = m_current;

  auto       rotMat        = glm::dmat3(matrix);
  glm::dvec3 forwardVector = rotMat * glm::dvec3(0, 0, -centerDistance);

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

void CameraManipulator::motion(const glm::dvec2& screenDisplacement, Actions action /*= 0*/)
{
  glm::dvec2 displacement = {
      (screenDisplacement.x - m_mouse[0]) / static_cast<double>(m_windowSize.x),
      (screenDisplacement.y - m_mouse[1]) / static_cast<double>(m_windowSize.y),
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

void CameraManipulator::keyMotion(glm::dvec2 delta, Actions action)
{
  if(delta == glm::dvec2(0.0, 0.0))
    return;

  double movementSpeed = m_speed;

  const CameraFrame frame = computeCameraFrame();
  delta *= movementSpeed;

  glm::dvec3 keyboardMovementVector{0, 0, 0};
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

CameraManipulator::Actions CameraManipulator::mouseMove(glm::dvec2 screenDisplacement, const Inputs& inputs)
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

void CameraManipulator::wheel(double value, const Inputs& inputs)
{
  if(value == 0.0)
    return;

  double deltaX = (value * std::abs(value)) / static_cast<double>(m_windowSize.x);

  if(inputs.shift)
  {
    if(m_current.projectionType == ProjectionType::Orthographic)
    {

      double zoomFactor = 1.0 + deltaX;
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
    dolly(glm::dvec2(deltaX), inputs.ctrl);
    applyUserChange();
  }
}

void CameraManipulator::adjustOrthographicAspect()
{
  if(m_current.projectionType != ProjectionType::Orthographic)
    return;

  const double aspect = getAspectRatio();
  if(aspect <= 0.0)
    return;

  const double height = m_current.orthMag.y;
  const double width  = height * aspect;
  if(width <= 0.0)
    return;

  if(std::abs(width - m_current.orthMag.x) > CameraConstants::EPSILON)
  {
    m_current.orthMag.x = width;
    m_current.orthMag.y = height;
  }
}

void CameraManipulator::setFov(double fovDegree)
{
  m_current.fov = std::min(std::max(fovDegree, CameraConstants::MIN_FOV), CameraConstants::MAX_FOV);
}

void CameraManipulator::convertToPerspective()
{
  if(m_current.projectionType == ProjectionType::Perspective)
    return;  // Already perspective

  // Calculate FOV based on the orthographic viewport and distance to center
  const double distance = glm::length(m_current.eye - m_current.ctr);
  if(distance > 0.0 && m_current.orthMag.y > 0.0)
  {
    // FOV = 2 * atan(ymag / distance)
    m_current.fov = glm::degrees(2.0 * std::atan(m_current.orthMag.y / distance));
    // Clamp to reasonable range
    m_current.fov = glm::clamp(m_current.fov, CameraConstants::MIN_FOV, CameraConstants::MAX_FOV);
  }

  m_current.projectionType = ProjectionType::Perspective;
}

void CameraManipulator::convertToOrthographic()
{
  if(m_current.projectionType == ProjectionType::Orthographic)
    return;  // Already orthographic

  // Calculate orthographic viewport based on FOV and distance to center
  const double distance = glm::length(m_current.eye - m_current.ctr);
  if(distance > 0.0)
  {
    // visibleHeight = 2 * distance * tan(fov/2)
    // ymag = visibleHeight / 2 = distance * tan(fov/2)
    const double halfFovRad = glm::radians(m_current.fov * 0.5);
    m_current.orthMag.y     = distance * std::tan(halfFovRad);
    m_current.orthMag.x     = m_current.orthMag.y * getAspectRatio();
  }

  m_current.projectionType = ProjectionType::Orthographic;
}

// Quadratic Bezier curve: B(t) = (1-t)^2*p0 + 2*(1-t)*t*p1 + t^2*p2
glm::dvec3 CameraManipulator::computeBezier(double t, const glm::dvec3& p0, const glm::dvec3& p1, const glm::dvec3& p2) const
{
  double u  = 1.0 - t;
  double tt = t * t;
  double uu = u * u;

  glm::dvec3 p = uu * p0;  // first term
  p += 2 * u * t * p1;     // second term
  p += tt * p2;            // third term

  return p;
}

void CameraManipulator::findBezierPoints()
{
  if(!m_snapshot)
    return;

  // Compute a smooth arc in view space between current and goal positions.
  const glm::dvec3 p0 = m_current.eye;
  const glm::dvec3 p2 = m_goal.eye;
  // Point of interest (center)
  const glm::dvec3 pi = (m_goal.ctr + m_snapshot->ctr) * 0.5;
  // Midpoint between endpoints
  const glm::dvec3 mid = (p0 + p2) * 0.5;

  // Radius based on average distance to interest
  const double radius = 0.5 * (glm::length(p0 - pi) + glm::length(p2 - pi));
  // Vector from interest to the midpoint
  glm::dvec3 toMid = mid - pi;
  if(glm::dot(toMid, toMid) < CameraConstants::EPSILON)
  {
    toMid = glm::dvec3(0, 0, 1);
  }
  // Calculated point to pass through
  const glm::dvec3 pc = pi + radius * glm::normalize(toMid);
  // Compute control point so curve goes through pc at t=0.5
  glm::dvec3 p1 = 2.0 * pc - 0.5 * (p0 + p2);

  // Project onto plane perpendicular to average up vector to avoid Y-up assumptions
  const glm::dvec3 avgUp      = glm::normalize(m_snapshot->up + m_goal.up);
  const double     projection = glm::dot(mid - p1, avgUp);
  p1 += projection * avgUp;

  m_bezier[0] = p0;
  m_bezier[1] = p1;
  m_bezier[2] = p2;
}

//--------------------------------------------------------------------------------------------------

double CameraManipulator::getTimeMs() const
{
  auto now(std::chrono::steady_clock::now());
  auto duration = now.time_since_epoch();
  return std::chrono::duration_cast<std::chrono::duration<double>>(duration).count() * 1000.0;
}

bool CameraManipulator::isValidPosition(const glm::dvec3& pos)
{
  return !std::isnan(pos.x) && !std::isnan(pos.y) && !std::isnan(pos.z) && !std::isinf(pos.x) && !std::isinf(pos.y)
         && !std::isinf(pos.z);
}

bool CameraManipulator::isValidDirection(const glm::dvec3& dir)
{
  double len = glm::length(dir);
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

void CameraManipulator::fit(const glm::dvec3& boxMin, const glm::dvec3& boxMax, bool instantFit /*= true*/, bool tightFit /*=false*/, double aspect /*=1.0*/)
{
  // Calculate the half extents of the bounding box
  const glm::dvec3 boxHalfSize = 0.5 * (boxMax - boxMin);

  // Calculate the center of the bounding box
  const glm::dvec3 boxCenter = 0.5 * (boxMin + boxMax);

  const double yfov = std::tan(glm::radians(m_current.fov * 0.5));
  const double xfov = yfov * aspect;

  // Calculate the ideal distance for a tight fit or fit to radius
  double idealDistance = 0;

  if(tightFit)
  {
    // Get only the rotation matrix
    glm::dmat3 mView = glm::lookAt(m_current.eye, boxCenter, effectiveUp(m_current.eye, boxCenter, m_current.up));

    // Check each 8 corner of the cube
    for(int i = 0; i < 8; i++)
    {
      // Rotate the bounding box in the camera view
      glm::dvec3 vct(i & 1 ? boxHalfSize.x : -boxHalfSize.x,   //
                     i & 2 ? boxHalfSize.y : -boxHalfSize.y,   //
                     i & 4 ? boxHalfSize.z : -boxHalfSize.z);  //
      vct = mView * vct;

      if(vct.z < 0)  // Take only points in front of the center
      {
        // Keep the largest offset to see that vertex
        idealDistance = std::max(std::abs(vct.y) / yfov + std::abs(vct.z), idealDistance);
        idealDistance = std::max(std::abs(vct.x) / xfov + std::abs(vct.z), idealDistance);
      }
    }
  }
  else  // Using the bounding sphere
  {
    const double radius = glm::length(boxHalfSize);
    idealDistance       = std::max(radius / xfov, radius / yfov);
  }

  // Calculate the new camera position based on the ideal distance
  const glm::dvec3 newEye = boxCenter - idealDistance * glm::normalize(boxCenter - m_current.eye);

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

  std::array<double, 16> val{};
  int result = SAFE_SSCANF(text.c_str(), "{%lf, %lf, %lf}, {%lf, %lf, %lf}, {%lf, %lf, %lf}, {%lf}, {%lf, %lf}, {%lf, %lf}, {%lf}",
                           &val[0], &val[1], &val[2], &val[3], &val[4], &val[5], &val[6], &val[7], &val[8], &val[9],
                           &val[10], &val[11], &val[12], &val[13], &val[14]);
  if(result >= 9)  // Before 2025-09-03, this format didn't include the FOV at the end
  {
    eye = glm::dvec3{val[0], val[1], val[2]};
    ctr = glm::dvec3{val[3], val[4], val[5]};
    up  = glm::dvec3{val[6], val[7], val[8]};
    if(result >= 10)
      fov = val[9];
    if(result >= 12)
      nearFar = glm::dvec2{val[10], val[11]};
    if(result >= 14)
      orthMag = glm::dvec2{val[12], val[13]};
    if(result >= 15)
      projectionType = static_cast<ProjectionType>(static_cast<int>(val[14]));

    return true;
  }
  return false;
}

std::vector<std::shared_ptr<CameraPlugin>>& cameraPlugins()
{
  return s_cameraPlugins;
}

}  // namespace nvutils

[[maybe_unused]] static void usage_CameraManipulator()
{
  nvutils::CameraManipulator camera;

  // Set camera information
  camera.setLookat(glm::dvec3(0., 0., -5.),                // eye
                   glm::dvec3(0., 0., 0.),                 // center
                   glm::dvec3(0., 1., 0.));                // up
  camera.setSpeed(10.);                                    // movement speed
  camera.setMode(nvutils::CameraManipulator::Modes::Fly);  // movement mode

  // Retrieve camera information
  glm::dvec3 eye, center, up;
  camera.getLookat(eye, center, up);
  const double fov = camera.getFov();

  // In the frame loop...
  while(true)
  {
    // Get mouse and keyboard inputs and move the camera.
    // Typically, you'll have both keyboard and mouse movement.
    // nvapp::ElementCamera implements all this for you, but
    // nvutils::CameraManipulator gives you lower-level access so you can
    // move it however you want.
    nvutils::CameraManipulator::Inputs inputs{
        .lmb   = true,   // ImGui::IsMouseDown(ImGuiMouseButton_Left)
        .mmb   = false,  // ImGui::IsMouseDown(ImGuiMouseButton_Middle)
        .rmb   = false,  // ImGui::IsMouseDown(ImGuiMouseButton_Right)
        .shift = false,  // ImGui::IsKeyDown(ImGuiKey_LeftShift) || ImGui::IsKeyDown(ImGuiKey_RightShift)
        .ctrl  = false,  // ImGui::IsKeyDown(ImGuiKey_LeftCtrl)  || ImGui::IsKeyDown(ImGuiKey_RightCtrl)
        .alt   = false,  // ImGui::IsKeyDown(ImGuiKey_LeftAlt)   || ImGui::IsKeyDown(ImGuiKey_RightAlt)
    };
    const glm::dvec2 mousePos = {512., 384.};  // in pixels

    // Keyboard movement
    if(/* ImGui::IsKeyDown(ImGuiKey_UpArrow) */ false)
    {
      camera.keyMotion({0.0, 1.0}, nvutils::CameraManipulator::Actions::Pan);
    }

    // Mouse movement
    camera.mouseMove(mousePos, inputs);

    // Scroll wheel
    camera.wheel(/* ImGui::GetIO().MouseWheel*/ 0.0, inputs);

    // Retrieve the matrix to push to the shader.
    // A good practice here if you need precision is to use double-precision
    // when composing matrices on the CPU (so that you don't lose so much if,
    // say, the world and view matrices cancel out), then upload to the GPU
    // in single-precision (while GPUs can handle double-precision, it's
    // usually only fast on professional-grade GPUs.)
    const glm::dmat4 world             = {};  // from model
    const glm::dmat4 view              = camera.getViewMatrix();
    const glm::dmat4 proj              = camera.getPerspectiveMatrix();
    const glm::mat4  vertexToClipSpace = proj * view * world;
  }
}
