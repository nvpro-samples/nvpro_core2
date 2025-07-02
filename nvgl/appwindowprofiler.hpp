/*
 * Copyright (c) 2014-2025, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2014-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */


#pragma once

#include <cstring>  // for memset
#include <filesystem>

#include <nvutils/parameter_sequencer.hpp>
#include <nvutils/profiler.hpp>

#include "nvpsystem.hpp"
#include "contextwindow.hpp"
#include "profiler_gl.hpp"

namespace nvgl {

// combined description is a single string separated "name|help string", and registers name both as name and shortName
static nvutils::ParameterBase::Info makeParamInfo(const std::string&                      combined,
                                                  nvutils::ParameterBase::CallbackSuccess callbackSuccess = nullptr);


/*-------------------------------------------------------------------------------------------------
    # class nvh::AppWindowProfiler
    nvh::AppWindowProfiler provides an alternative utility wrapper class around NVPWindow.
    It is useful to derive single-window applications from and is used by some
    but not all nvpro-samples.

    Further functionality is provided :
    - built-in profiler/timer reporting to console
    - command-line argument parsing as well as config file parsing using the ParameterTools
      see AppWindowProfiler::setupParameters() for built-in commands
    - benchmark/automation mode using ParameterTools
    - screenshot creation
    - logfile based on devicename (depends on context)
    - optional context/swapchain interface
      the derived classes nvvk/appwindowprofiler_vk and nvgl/appwindowprofiler_gl make use of this
-------------------------------------------------------------------------------------------------*/

#define NV_CONCAT(a, b) a##b

#define NV_PROFILE_BASE_SECTION(name) auto NV_CONCAT(frameSection, __COUNTER__) = m_profilerTimeline->frameSection(name)
#define NV_PROFILE_BASE_SPLIT() m_profilerTimeline->frameAccumulationSplit()

#define NV_PROFILE_GL_SECTION(name) auto NV_CONCAT(frameSection, __COUNTER__) = m_profilerGL.frameSection(name)
#define NV_PROFILE_GL_SPLIT() m_profilerGL.frameAccumulationSplit()

class AppWindowProfiler
{
public:
  // these are taken from GLFW3 and must be kept in a matching state
  enum ButtonAction
  {
    BUTTON_RELEASE = 0,
    BUTTON_PRESS   = 1,
    BUTTON_REPEAT  = 2,
  };

  enum MouseButton
  {
    MOUSE_BUTTON_LEFT   = 0,
    MOUSE_BUTTON_RIGHT  = 1,
    MOUSE_BUTTON_MIDDLE = 2,
    NUM_MOUSE_BUTTONIDX,
  };

  enum MouseButtonFlag
  {
    MOUSE_BUTTONFLAG_NONE   = 0,
    MOUSE_BUTTONFLAG_LEFT   = (1 << MOUSE_BUTTON_LEFT),
    MOUSE_BUTTONFLAG_RIGHT  = (1 << MOUSE_BUTTON_RIGHT),
    MOUSE_BUTTONFLAG_MIDDLE = (1 << MOUSE_BUTTON_MIDDLE)
  };

  enum KeyCode
  {
    KEY_UNKNOWN           = -1,
    KEY_SPACE             = 32,
    KEY_APOSTROPHE        = 39, /* ' */
    KEY_LEFT_PARENTHESIS  = 40, /* ( */
    KEY_RIGHT_PARENTHESIS = 41, /* ) */
    KEY_ASTERISK          = 42, /* * */
    KEY_PLUS              = 43, /* + */
    KEY_COMMA             = 44, /* , */
    KEY_MINUS             = 45, /* - */
    KEY_PERIOD            = 46, /* . */
    KEY_SLASH             = 47, /* / */
    KEY_0                 = 48,
    KEY_1                 = 49,
    KEY_2                 = 50,
    KEY_3                 = 51,
    KEY_4                 = 52,
    KEY_5                 = 53,
    KEY_6                 = 54,
    KEY_7                 = 55,
    KEY_8                 = 56,
    KEY_9                 = 57,
    KEY_SEMICOLON         = 59, /* ; */
    KEY_EQUAL             = 61, /* = */
    KEY_A                 = 65,
    KEY_B                 = 66,
    KEY_C                 = 67,
    KEY_D                 = 68,
    KEY_E                 = 69,
    KEY_F                 = 70,
    KEY_G                 = 71,
    KEY_H                 = 72,
    KEY_I                 = 73,
    KEY_J                 = 74,
    KEY_K                 = 75,
    KEY_L                 = 76,
    KEY_M                 = 77,
    KEY_N                 = 78,
    KEY_O                 = 79,
    KEY_P                 = 80,
    KEY_Q                 = 81,
    KEY_R                 = 82,
    KEY_S                 = 83,
    KEY_T                 = 84,
    KEY_U                 = 85,
    KEY_V                 = 86,
    KEY_W                 = 87,
    KEY_X                 = 88,
    KEY_Y                 = 89,
    KEY_Z                 = 90,
    KEY_LEFT_BRACKET      = 91,  /* [ */
    KEY_BACKSLASH         = 92,  /* \ */
    KEY_RIGHT_BRACKET     = 93,  /* ] */
    KEY_GRAVE_ACCENT      = 96,  /* ` */
    KEY_WORLD_1           = 161, /* non-US #1 */
    KEY_WORLD_2           = 162, /* non-US #2 */

    /* Function keys */
    KEY_ESCAPE        = 256,
    KEY_ENTER         = 257,
    KEY_TAB           = 258,
    KEY_BACKSPACE     = 259,
    KEY_INSERT        = 260,
    KEY_DELETE        = 261,
    KEY_RIGHT         = 262,
    KEY_LEFT          = 263,
    KEY_DOWN          = 264,
    KEY_UP            = 265,
    KEY_PAGE_UP       = 266,
    KEY_PAGE_DOWN     = 267,
    KEY_HOME          = 268,
    KEY_END           = 269,
    KEY_CAPS_LOCK     = 280,
    KEY_SCROLL_LOCK   = 281,
    KEY_NUM_LOCK      = 282,
    KEY_PRINT_SCREEN  = 283,
    KEY_PAUSE         = 284,
    KEY_F1            = 290,
    KEY_F2            = 291,
    KEY_F3            = 292,
    KEY_F4            = 293,
    KEY_F5            = 294,
    KEY_F6            = 295,
    KEY_F7            = 296,
    KEY_F8            = 297,
    KEY_F9            = 298,
    KEY_F10           = 299,
    KEY_F11           = 300,
    KEY_F12           = 301,
    KEY_F13           = 302,
    KEY_F14           = 303,
    KEY_F15           = 304,
    KEY_F16           = 305,
    KEY_F17           = 306,
    KEY_F18           = 307,
    KEY_F19           = 308,
    KEY_F20           = 309,
    KEY_F21           = 310,
    KEY_F22           = 311,
    KEY_F23           = 312,
    KEY_F24           = 313,
    KEY_F25           = 314,
    KEY_KP_0          = 320,
    KEY_KP_1          = 321,
    KEY_KP_2          = 322,
    KEY_KP_3          = 323,
    KEY_KP_4          = 324,
    KEY_KP_5          = 325,
    KEY_KP_6          = 326,
    KEY_KP_7          = 327,
    KEY_KP_8          = 328,
    KEY_KP_9          = 329,
    KEY_KP_DECIMAL    = 330,
    KEY_KP_DIVIDE     = 331,
    KEY_KP_MULTIPLY   = 332,
    KEY_KP_SUBTRACT   = 333,
    KEY_KP_ADD        = 334,
    KEY_KP_ENTER      = 335,
    KEY_KP_EQUAL      = 336,
    KEY_LEFT_SHIFT    = 340,
    KEY_LEFT_CONTROL  = 341,
    KEY_LEFT_ALT      = 342,
    KEY_LEFT_SUPER    = 343,
    KEY_RIGHT_SHIFT   = 344,
    KEY_RIGHT_CONTROL = 345,
    KEY_RIGHT_ALT     = 346,
    KEY_RIGHT_SUPER   = 347,
    KEY_MENU          = 348,
    KEY_LAST          = KEY_MENU,
  };

  enum KeyModifiers
  {
    KMOD_SHIFT   = 1,
    KMOD_CONTROL = 2,
    KMOD_ALT     = 4,
    KMOD_SUPER   = 8,
  };

  class WindowState
  {
  public:
    WindowState() {}

    int m_winSize[2]{};
    int m_swapSize[2]{};
    int m_mouseCurrent[2]{};
    int m_mouseButtonFlags = 0;
    int m_mouseWheel       = 0;

    bool m_keyPressed[KEY_LAST + 1]{};
    bool m_keyToggled[KEY_LAST + 1]{};

    bool onPress(int key) { return m_keyPressed[key] && m_keyToggled[key]; }
  };

  //////////////////////////////////////////////////////////////////////////

  struct GLFWwindow* m_internal = nullptr;  ///< internal delegate to GLFWwindow
  std::string        m_windowName;

  //////////////////////////////////////////////////////////////////////////

  WindowState                m_windowState;
  nvutils::ProfilerManager   m_profiler;
  nvutils::ProfilerTimeline* m_profilerTimeline = nullptr;
  bool                       m_profilerPrint;
  bool                       m_hadProfilerPrint;
  bool                       m_timeInTitle;

  nvutils::ParameterRegistry m_parameterList;
  nvutils::ParameterParser   m_parameterParser;

  nvgl::ContextWindowCreateInfo m_contextInfo;
  ContextWindow                 m_contextWindow;

  nvgl::ProfilerGpuTimer m_profilerGL;


  AppWindowProfiler()
      : m_profilerPrint(true)
      , m_vsync(false)
      , m_active(false)
      , m_timeInTitle(true)
      , m_parameterParser("project", {".cfg"})
      , m_hadScreenshot(false)
  {
    m_contextInfo.robust = false;
    m_contextInfo.core   = false;
#ifdef NDEBUG
    m_contextInfo.debug = false;
#else
    m_contextInfo.debug = true;
#endif
    m_contextInfo.share = NULL;
    m_contextInfo.major = 4;
    m_contextInfo.minor = 5;

    m_profilerTimeline = m_profiler.createTimeline({"Primary"});
    setupParameters();
  }

  ~AppWindowProfiler() { m_profiler.destroyTimeline(m_profilerTimeline); }

  // Sample Related
  //////////////////////////////////////////////////////////////////////////

  // setup sample (this is executed after window/context creation)
  virtual bool begin() { return false; }
  // tear down sample (triggered by ESC/window close)
  virtual void end() {}
  // do primary logic/drawing etc. here
  virtual void think(double time) {}
  // react on swapchain resizes here
  // may be different to winWidth/winHeight!
  virtual void resize(int swapWidth, int swapHeight) {}

  // return true to prevent m_window state updates
  virtual bool mouse_pos(int x, int y) { return false; }
  virtual bool mouse_button(int button, int action) { return false; }
  virtual bool mouse_wheel(int wheel) { return false; }
  virtual bool key_button(int button, int action, int modifier) { return false; }
  virtual bool key_char(int button) { return false; }

  virtual void onDragDrop(int num, const char** paths) {}

  virtual void parseConfig(int argc, const char** argv, const std::string& path)
  {
    // if you want to handle parameters not represented in
    // m_parameterList then override this function accordingly.
    m_parameterParser.parse(argc, argv, false, path);

    // This function is called before "begin" and provided with the commandline used in "run".
    // It can also be called by the benchmarking system, and parseConfigFile.
  }
  virtual bool validateConfig()
  {
    // override if you want to test the state of app after parsing configs
    // returning false terminates app
    return true;
  }

  // additional special-purpose callbacks

  virtual void postProfiling() {}
  virtual void postEnd() {}
  virtual void postBenchmarkAdvance() {}
  virtual void postConfigPreContext() {};

  //////////////////////////////////////////////////////////////////////////

  // initial kickoff (typically called from main)
  int  run(const std::string& name, int argc, const char** argv, int width, int height);
  void leave();

  void parseConfigFile(const char* filename);

  // handles special strings (returns empty string if
  // could not do the replacement properly)
  // known specials:
  // $DEVICE$
  std::string specialStrings(const char* original);


  void setVsync(bool state);
  bool getVsync() const { return m_vsync; }


  //////////////////////////////////////////////////////////////////////////
  // NVPWindow

  inline bool pollEvents()  // returns false on exit, can do while(pollEvents()){ ... }
  {
    NVPSystem::pollEvents();
    return !isClosing();
  }
  inline void        waitEvents() { NVPSystem::waitEvents(); }
  inline double      getTime() { return NVPSystem::getTime(); }
  inline std::string exePath() { return NVPSystem::exePath(); }

  // Accessors
  inline int getWidth() const { return m_windowSize[0]; }
  inline int getHeight() const { return m_windowSize[1]; }
  inline int getMouseWheel() const { return m_mouseWheel; }
  inline int getKeyModifiers() const { return m_keyModifiers; }
  inline int getMouseX() const { return m_mouseX; }
  inline int getMouseY() const { return m_mouseY; }

  void        setTitle(const char* title);
  void        setFullScreen(bool bYes);
  void        setWindowPos(int x, int y);
  void        setWindowSize(int w, int h);
  inline void setKeyModifiers(int m) { m_keyModifiers = m; }
  inline void setMouse(int x, int y)
  {
    m_mouseX = x;
    m_mouseY = y;
  }

  inline bool isFullScreen() const { return m_isFullScreen; }
  bool        isClosing() const;
  bool        isOpen() const;

  virtual bool open(int posX, int posY, int width, int height, const char* title, bool requireGLContext);  ///< creates internal window and opens it
  void deinit();  ///< destroys internal window

  void close();  ///<  triggers closing event, still needs deinit for final cleanup
  void maximize();
  void restore();
  void minimize();

  // uses operating system specific code for sake of debugging/automated testing
  void screenshot(const char* filename);
  void clear(uint32_t r, uint32_t g, uint32_t b);

  /// \defgroup dialog
  /// simple modal file dialog, uses OS basic api
  /// the exts string must be a | separated list that has two items per possible extension
  /// "extension descripton|*.ext"
  /// @{
  std::filesystem::path openFileDialog(const char* title, const char* exts);
  std::filesystem::path saveFileDialog(const char* title, const char* exts);
  /// @}


private:
  struct Benchmark
  {
    bool                                  isActive = false;
    nvutils::ParameterSequencer::InitInfo initInfo;
    nvutils::ParameterSequencer           sequencer;
  };

  struct Config
  {
    int32_t               winpos[2];
    int32_t               winsize[2];
    bool                  vsyncstate      = true;
    bool                  quickexit       = false;
    uint32_t              intervalSeconds = 2;
    uint32_t              frameLimit      = 0;
    uint32_t              timerLimit      = 0;
    std::string           dumpatexitFilename;
    std::string           screenshotFilename;
    std::filesystem::path logFilename;
    std::string           configFilename;
    uint32_t              clearColor[3] = {127, 0, 0};

    Config()
    {
      winpos[0]  = 50;
      winpos[1]  = 50;
      winsize[0] = 0;
      winsize[1] = 0;
    }
  };

  int  m_mouseX               = 0;
  int  m_mouseY               = 0;
  int  m_mouseWheel           = 0;
  int  m_windowSize[2]        = {0, 0};
  int  m_keyModifiers         = 0;
  bool m_isFullScreen         = false;
  bool m_isClosing            = false;
  int  m_preFullScreenPos[2]  = {0, 0};
  int  m_preFullScreenSize[2] = {0, 0};

  bool      m_activeContext = false;
  bool      m_active        = false;
  bool      m_vsync         = false;
  bool      m_hadScreenshot = false;
  Config    m_config;
  Benchmark m_benchmark;

  const nvutils::ParameterBase* m_paramWinsize{};
  const nvutils::ParameterBase* m_paramVsync{};
  const nvutils::ParameterBase* m_paramScreenshot{};
  const nvutils::ParameterBase* m_paramLog{};
  const nvutils::ParameterBase* m_paramCfg{};
  const nvutils::ParameterBase* m_paramBat{};
  const nvutils::ParameterBase* m_paramClear{};

  void parameterCallback(const nvutils::ParameterBase* param);

  void setupParameters();
  void exitScreenshot();

  void initBenchmark();
  void advanceBenchmark();

  void        contextInit();
  void        contextDeinit();
  void        contextSync() { glFinish(); }
  const char* contextGetDeviceName() { return m_contextWindow.m_deviceName.c_str(); }

  void swapResize(int winWidth, int winHeight)
  {
    m_windowState.m_swapSize[0] = winWidth;
    m_windowState.m_swapSize[1] = winHeight;
  }
  void swapBuffers() { m_contextWindow.swapBuffers(); }
  void swapVsync(bool state) { m_contextWindow.swapInterval(state ? 1 : 0); }

  void onWindowClose();
  void onWindowResize(int w, int h);
  void onMouseMotion(int x, int y);
  void onMouseWheel(int delta);
  void onMouseButton(MouseButton button, ButtonAction action, int mods, int x, int y);
  void onKeyboard(KeyCode key, ButtonAction action, int mods, int x, int y);
  void onKeyboardChar(unsigned char key, int mods, int x, int y);

  static void cb_windowsizefun(GLFWwindow* glfwwin, int w, int h);
  static void cb_windowclosefun(GLFWwindow* glfwwin);
  static void cb_mousebuttonfun(GLFWwindow* glfwwin, int button, int action, int mods);
  static void cb_cursorposfun(GLFWwindow* glfwwin, double x, double y);
  static void cb_scrollfun(GLFWwindow* glfwwin, double x, double y);
  static void cb_keyfun(GLFWwindow* glfwwin, int key, int scancode, int action, int mods);
  static void cb_charfun(GLFWwindow* glfwwin, unsigned int codepoint);
  static void cb_dropfun(GLFWwindow* glfwwin, int count, const char** paths);
};
}  // namespace nvgl
