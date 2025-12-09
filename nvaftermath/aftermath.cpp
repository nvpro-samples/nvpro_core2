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

#ifdef AFTERMATH_AVAILABLE
#include <chrono>
#include <filesystem>
#include <nvutils/file_operations.hpp>
#include <nvutils/logger.hpp>
#endif

#include "aftermath.hpp"


AftermathCrashTracker& AftermathCrashTracker::getInstance()
{
  static AftermathCrashTracker instance;
  return instance;
}

void AftermathCrashTracker::initialize()
{
  initAftermath();
}

#ifdef AFTERMATH_AVAILABLE
GpuCrashTracker& AftermathCrashTracker::getGpuCrashTracker()
{
  return *m_tracker;
}
#endif

void AftermathCrashTracker::addShaderBinary(const std::span<const uint32_t>& data)
{
#ifdef AFTERMATH_AVAILABLE
  m_tracker->addShaderBinary(data);
#endif
}

void AftermathCrashTracker::errorCallback(VkResult result)
{
#ifdef AFTERMATH_AVAILABLE
  if(result == VK_ERROR_DEVICE_LOST)
  {
    // Device lost notification is asynchronous to the NVIDIA display
    // driver's GPU crash handling. Give the Nsight Aftermath GPU crash dump
    // thread some time to do its work before terminating the process.
    auto tdr_termination_timeout = std::chrono::seconds(5);
    auto t_start                 = std::chrono::steady_clock::now();
    auto t_elapsed               = std::chrono::milliseconds::zero();

    GFSDK_Aftermath_CrashDump_Status status = GFSDK_Aftermath_CrashDump_Status_Unknown;
    AFTERMATH_CHECK_ERROR(GFSDK_Aftermath_GetCrashDumpStatus(&status));

    while(status != GFSDK_Aftermath_CrashDump_Status_CollectingDataFailed
          && status != GFSDK_Aftermath_CrashDump_Status_Finished && t_elapsed < tdr_termination_timeout)
    {
      // Sleep 50ms and poll the status again until timeout or Aftermath finished processing the crash dump.
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
      AFTERMATH_CHECK_ERROR(GFSDK_Aftermath_GetCrashDumpStatus(&status));

      auto t_end = std::chrono::steady_clock::now();
      t_elapsed  = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start);
    }

    if(status != GFSDK_Aftermath_CrashDump_Status_Finished)
    {
      std::stringstream err_msg;
      LOGE("Unexpected crash dump status: %d", status);
      // ERR_EXIT(err_msg.str().c_str(), "Aftermath Error");
    }

    LOGOK("\n\nAftermath file dumped under:\n\t%s", nvutils::utf8FromPath(std::filesystem::current_path()).c_str());

    // Terminate on failure
    //#ifdef _WIN32
    //      err_msg << "\n\n\nSave path to clipboard?";
    //      int ret = MessageBox(nullptr, err_msg.str().c_str(), "Nsight Aftermath", MB_YESNO | MB_ICONEXCLAMATION);
    //      if(ret == IDYES)
    //      {
    //        // ImGui::SetClipboardText(nvutils::utf8FromPath(std::filesystem::current_path()).c_str());
    //      }
    //#else
    //      printf("%s\n", err_msg.str().c_str());
    //#endif

    exit(1);
  }
#endif
}

AftermathCrashTracker::AftermathCrashTracker()
{
#ifdef AFTERMATH_AVAILABLE
  m_tracker = std::make_unique<::GpuCrashTracker>(m_marker);
#endif  // #ifdef AFTERMATH_AVAILABLE
}

void AftermathCrashTracker::initAftermath()
{
#ifdef AFTERMATH_AVAILABLE
  m_tracker->Initialize();
#endif
}


//--------------------------------------------------------------------------------------------------
// Usage example
//--------------------------------------------------------------------------------------------------
#ifdef AFTERMATH_DEMO_ONLY
[[maybe_unused]] static void usage_AftermathCrashTracker()
{

  // Initialize the AftermathCrashTracker
  auto& aftermath = AftermathCrashTracker::getInstance();
  aftermath.initialize();
  aftermath.addExtensions(vkSetup.deviceExtensions);  // Add the Aftermath extension to the device extensions

  // The callback function is called when a validation error is triggered.
  // This will wait to give time to dump the GPU crash.
  //---
  nvvk::CheckError::getInstance().setCallbackFunction([&](VkResult result) { aftermath.errorCallback(result); });


  // Adding the shader binaries to the Aftermath
  //---
#if defined(AFTERMATH_AVAILABLE)
  // This aftermath callback is used to report the shader hash (Spirv) to the Aftermath library.
  m_slangCompiler.setCompileCallback([&](const std::filesystem::path& sourceFile, const uint32_t* spirvCode, size_t spirvSize) {
    std::span<const uint32_t> data(spirvCode, spirvSize / sizeof(uint32_t));
    AftermathCrashTracker::getInstance().addShaderBinary(data);
  });
#endif
}
#endif