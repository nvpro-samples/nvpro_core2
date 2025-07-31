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


//--------------------------------------------------------------------------------------------------
// The following is to enable the use of the Vulkan validation layers
// Usage:
//  nvvk::ValidationSettings vvlInfo{};   // Set defaults
//  vvlInfo.fine_grained_locking = false; // Change the value
//  vkSetup.instanceCreateInfoExt = vvlInfo.buildPNextChain();  // Adding the validation layer settings
// https://vulkan.lunarg.com/doc/sdk/1.3.296.0/windows/khronos_validation_layer.html
#pragma once

#include <vector>
#include <cstring>  // for memset
#include <vulkan/vulkan.h>

namespace nvvk {

struct ValidationSettings
{
  enum class LayerPresets
  {
    eStandard,                       // Good default validation setup that balances validation coverage and performance
    eBestPractices,                  // Provides warnings on valid API usage that is potential API misuse
    eSynchronization,                // Identify resource access conflicts due to missing or incorrect synchronization
    eGpuAssisted,                    // Check for API usage errors at shader execution time
    eGpuAssistedReserveBindingSlot,  // GPU-assisted validation with dedicated binding slot
    eDebugPrintf,                    // Enable debug printf features
  };

  // Core Validation Settings
  VkBool32 validate_core{VK_TRUE};
  VkBool32 check_image_layout{VK_TRUE};
  VkBool32 check_command_buffer{VK_TRUE};
  VkBool32 check_object_in_use{VK_TRUE};
  VkBool32 check_query{VK_TRUE};
  VkBool32 check_shaders{VK_TRUE};
  VkBool32 check_shaders_caching{VK_TRUE};
  VkBool32 unique_handles{VK_TRUE};
  VkBool32 object_lifetime{VK_TRUE};
  VkBool32 stateless_param{VK_TRUE};
  VkBool32 thread_safety{VK_TRUE};

  // Synchronization Settings
  VkBool32 validate_sync{VK_FALSE};
  VkBool32 syncval_submit_time_validation{VK_TRUE};
  VkBool32 syncval_shader_accesses_heuristic{VK_FALSE};

  // GPU Validation Settings
  VkBool32 validate_gpu_based{VK_FALSE};
  VkBool32 gpuav_shader_instrumentation{VK_TRUE};
  VkBool32 gpuav_descriptor_checks{VK_TRUE};
  VkBool32 gpuav_warn_on_robust_oob{VK_TRUE};
  VkBool32 gpuav_buffer_address_oob{VK_TRUE};
  uint32_t gpuav_max_buffer_device_addresses{10000};
  VkBool32 gpuav_validate_ray_query{VK_TRUE};
  VkBool32 gpuav_select_instrumented_shaders{VK_FALSE};
  VkBool32 gpuav_buffers_validation{VK_TRUE};
  VkBool32 gpuav_indirect_draws_buffers{VK_FALSE};
  VkBool32 gpuav_indirect_dispatches_buffers{VK_FALSE};
  VkBool32 gpuav_indirect_trace_rays_buffers{VK_FALSE};
  VkBool32 gpuav_buffer_copies{VK_TRUE};
  VkBool32 gpuav_index_buffers{VK_TRUE};
  VkBool32 gpuav_reserve_binding_slot{VK_TRUE};
  VkBool32 gpuav_vma_linear_output{VK_TRUE};

  // Debug Printf Settings
  VkBool32 printf_to_stdout{VK_TRUE};
  VkBool32 printf_verbose{VK_FALSE};
  uint32_t printf_buffer_size{1024};

  // Best Practices Settings
  VkBool32 validate_best_practices{VK_FALSE};
  VkBool32 validate_best_practices_arm{VK_FALSE};
  VkBool32 validate_best_practices_amd{VK_FALSE};
  VkBool32 validate_best_practices_img{VK_FALSE};
  VkBool32 validate_best_practices_nvidia{VK_FALSE};

  // Message and Debug Settings
  std::vector<const char*> debug_action{"VK_DBG_LAYER_ACTION_LOG_MSG"};
  std::vector<const char*> report_flags{"error"};
  std::vector<const char*> message_id_filter{};  // Filter validation messages by message IDs
  VkBool32                 enable_message_limit{VK_TRUE};
  uint32_t                 duplicate_message_limit{3};
  VkBool32                 message_format_display_application_name{VK_FALSE};

  // General Settings
  VkBool32    fine_grained_locking{VK_TRUE};
  const char* layerEnables{""};

  VkBaseInStructure* buildPNextChain()
  {
    updateSettings();
    return reinterpret_cast<VkBaseInStructure*>(&m_layerSettingsCreateInfo);
  }

  void setPreset(LayerPresets preset)
  {
    // First reset all settings to their default disabled state
    memset(this, 0, sizeof(*this));

    // Restore default buffer size
    printf_buffer_size = 1024;

    // Restore default vectors
    debug_action = {"VK_DBG_LAYER_ACTION_LOG_MSG"};
    report_flags = {"error"};

    switch(preset)
    {
      case LayerPresets::eStandard:
        // Core Validation Settings
        validate_core         = VK_TRUE;
        check_image_layout    = VK_TRUE;
        check_command_buffer  = VK_TRUE;
        check_object_in_use   = VK_TRUE;
        check_query           = VK_TRUE;
        check_shaders         = VK_TRUE;
        check_shaders_caching = VK_TRUE;
        unique_handles        = VK_TRUE;
        object_lifetime       = VK_TRUE;
        stateless_param       = VK_TRUE;
        thread_safety         = VK_TRUE;
        fine_grained_locking  = VK_TRUE;
        layerEnables          = "";
        break;

      case LayerPresets::eBestPractices:
        validate_best_practices = VK_TRUE;
        layerEnables            = "VK_VALIDATION_FEATURE_ENABLE_BEST_PRACTICES_EXT";
        break;

      case LayerPresets::eSynchronization:
        validate_sync                     = VK_TRUE;
        syncval_submit_time_validation    = VK_TRUE;
        syncval_shader_accesses_heuristic = VK_TRUE;
        thread_safety                     = VK_TRUE;
        unique_handles                    = VK_TRUE;
        layerEnables                      = "VK_VALIDATION_FEATURE_ENABLE_SYNCHRONIZATION_VALIDATION_EXT";
        break;

      case LayerPresets::eGpuAssisted:
        validate_gpu_based           = VK_TRUE;
        gpuav_shader_instrumentation = VK_TRUE;
        gpuav_descriptor_checks      = VK_TRUE;
        gpuav_buffer_address_oob     = VK_TRUE;
        gpuav_validate_ray_query     = VK_TRUE;
        gpuav_buffers_validation     = VK_TRUE;
        gpuav_buffer_copies          = VK_TRUE;
        gpuav_index_buffers          = VK_TRUE;
        gpuav_reserve_binding_slot   = VK_TRUE;
        layerEnables                 = "VK_VALIDATION_FEATURE_ENABLE_GPU_ASSISTED_EXT";
        break;

      case LayerPresets::eGpuAssistedReserveBindingSlot:
        validate_gpu_based           = VK_TRUE;
        gpuav_shader_instrumentation = VK_TRUE;
        gpuav_descriptor_checks      = VK_TRUE;
        gpuav_buffer_address_oob     = VK_TRUE;
        gpuav_validate_ray_query     = VK_TRUE;
        gpuav_buffers_validation     = VK_TRUE;
        gpuav_buffer_copies          = VK_TRUE;
        gpuav_index_buffers          = VK_TRUE;
        gpuav_reserve_binding_slot   = VK_TRUE;
        layerEnables                 = "VK_VALIDATION_FEATURE_ENABLE_GPU_ASSISTED_RESERVE_BINDING_SLOT_EXT";
        break;

      case LayerPresets::eDebugPrintf:
        debug_action       = {"VK_DBG_LAYER_ACTION_IGNORE"};  // Explicitly ignore extra validation messages
        printf_to_stdout   = VK_FALSE;                        // Changed from VK_TRUE to allow debug callback usage
        printf_buffer_size = 1024;
        layerEnables       = "VK_VALIDATION_FEATURE_ENABLE_DEBUG_PRINTF_EXT";
        break;
    }
  }

  void updateSettings()
  {
    // clang-format off
    m_settings = {
        // Core Validation Settings
        {m_layer_name, "fine_grained_locking", VK_LAYER_SETTING_TYPE_BOOL32_EXT, 1, &fine_grained_locking},
        {m_layer_name, "validate_core", VK_LAYER_SETTING_TYPE_BOOL32_EXT, 1, &validate_core},
        {m_layer_name, "check_image_layout", VK_LAYER_SETTING_TYPE_BOOL32_EXT, 1, &check_image_layout},
        {m_layer_name, "check_command_buffer", VK_LAYER_SETTING_TYPE_BOOL32_EXT, 1, &check_command_buffer},
        {m_layer_name, "check_object_in_use", VK_LAYER_SETTING_TYPE_BOOL32_EXT, 1, &check_object_in_use},
        {m_layer_name, "check_query", VK_LAYER_SETTING_TYPE_BOOL32_EXT, 1, &check_query},
        {m_layer_name, "check_shaders", VK_LAYER_SETTING_TYPE_BOOL32_EXT, 1, &check_shaders},
        {m_layer_name, "check_shaders_caching", VK_LAYER_SETTING_TYPE_BOOL32_EXT, 1, &check_shaders_caching},
        {m_layer_name, "unique_handles", VK_LAYER_SETTING_TYPE_BOOL32_EXT, 1, &unique_handles},
        {m_layer_name, "object_lifetime", VK_LAYER_SETTING_TYPE_BOOL32_EXT, 1, &object_lifetime},
        {m_layer_name, "stateless_param", VK_LAYER_SETTING_TYPE_BOOL32_EXT, 1, &stateless_param},
        {m_layer_name, "thread_safety", VK_LAYER_SETTING_TYPE_BOOL32_EXT, 1, &thread_safety},

        // Synchronization Settings
        {m_layer_name, "validate_sync", VK_LAYER_SETTING_TYPE_BOOL32_EXT, 1, &validate_sync},
        {m_layer_name, "syncval_submit_time_validation", VK_LAYER_SETTING_TYPE_BOOL32_EXT, 1, &syncval_submit_time_validation},
        {m_layer_name, "syncval_shader_accesses_heuristic", VK_LAYER_SETTING_TYPE_BOOL32_EXT, 1, &syncval_shader_accesses_heuristic},

        // GPU Validation Settings
        {m_layer_name, "validate_gpu_based", VK_LAYER_SETTING_TYPE_BOOL32_EXT, 1, &validate_gpu_based},
        {m_layer_name, "gpuav_shader_instrumentation", VK_LAYER_SETTING_TYPE_BOOL32_EXT, 1, &gpuav_shader_instrumentation},
        {m_layer_name, "gpuav_descriptor_checks", VK_LAYER_SETTING_TYPE_BOOL32_EXT, 1, &gpuav_descriptor_checks},
        {m_layer_name, "gpuav_warn_on_robust_oob", VK_LAYER_SETTING_TYPE_BOOL32_EXT, 1, &gpuav_warn_on_robust_oob},
        {m_layer_name, "gpuav_buffer_address_oob", VK_LAYER_SETTING_TYPE_BOOL32_EXT, 1, &gpuav_buffer_address_oob},
        {m_layer_name, "gpuav_max_buffer_device_addresses", VK_LAYER_SETTING_TYPE_UINT32_EXT, 1, &gpuav_max_buffer_device_addresses},
        {m_layer_name, "gpuav_validate_ray_query", VK_LAYER_SETTING_TYPE_BOOL32_EXT, 1, &gpuav_validate_ray_query},
        {m_layer_name, "gpuav_select_instrumented_shaders", VK_LAYER_SETTING_TYPE_BOOL32_EXT, 1, &gpuav_select_instrumented_shaders},
        {m_layer_name, "gpuav_buffers_validation", VK_LAYER_SETTING_TYPE_BOOL32_EXT, 1, &gpuav_buffers_validation},
        {m_layer_name, "gpuav_indirect_draws_buffers", VK_LAYER_SETTING_TYPE_BOOL32_EXT, 1, &gpuav_indirect_draws_buffers},
        {m_layer_name, "gpuav_indirect_dispatches_buffers", VK_LAYER_SETTING_TYPE_BOOL32_EXT, 1, &gpuav_indirect_dispatches_buffers},
        {m_layer_name, "gpuav_indirect_trace_rays_buffers", VK_LAYER_SETTING_TYPE_BOOL32_EXT, 1, &gpuav_indirect_trace_rays_buffers},
        {m_layer_name, "gpuav_buffer_copies", VK_LAYER_SETTING_TYPE_BOOL32_EXT, 1, &gpuav_buffer_copies},
        {m_layer_name, "gpuav_index_buffers", VK_LAYER_SETTING_TYPE_BOOL32_EXT, 1, &gpuav_index_buffers},
        {m_layer_name, "gpuav_reserve_binding_slot", VK_LAYER_SETTING_TYPE_BOOL32_EXT, 1, &gpuav_reserve_binding_slot},
        {m_layer_name, "gpuav_vma_linear_output", VK_LAYER_SETTING_TYPE_BOOL32_EXT, 1, &gpuav_vma_linear_output},

        // Debug Printf Settings
        {m_layer_name, "printf_to_stdout", VK_LAYER_SETTING_TYPE_BOOL32_EXT, 1, &printf_to_stdout},
        {m_layer_name, "printf_verbose", VK_LAYER_SETTING_TYPE_BOOL32_EXT, 1, &printf_verbose},
        {m_layer_name, "printf_buffer_size", VK_LAYER_SETTING_TYPE_UINT32_EXT, 1, &printf_buffer_size},

        // Best Practices Settings
        {m_layer_name, "validate_best_practices", VK_LAYER_SETTING_TYPE_BOOL32_EXT, 1, &validate_best_practices},
        {m_layer_name, "validate_best_practices_arm", VK_LAYER_SETTING_TYPE_BOOL32_EXT, 1, &validate_best_practices_arm},
        {m_layer_name, "validate_best_practices_amd", VK_LAYER_SETTING_TYPE_BOOL32_EXT, 1, &validate_best_practices_amd},
        {m_layer_name, "validate_best_practices_img", VK_LAYER_SETTING_TYPE_BOOL32_EXT, 1, &validate_best_practices_img},
        {m_layer_name, "validate_best_practices_nvidia", VK_LAYER_SETTING_TYPE_BOOL32_EXT, 1, &validate_best_practices_nvidia},

        // Message and Debug Settings
        {m_layer_name, "debug_action", VK_LAYER_SETTING_TYPE_STRING_EXT, uint32_t(debug_action.size()), debug_action.data()},
        {m_layer_name, "report_flags", VK_LAYER_SETTING_TYPE_STRING_EXT, uint32_t(report_flags.size()), report_flags.data()},
        {m_layer_name, "message_id_filter", VK_LAYER_SETTING_TYPE_STRING_EXT, uint32_t(message_id_filter.size()), message_id_filter.data()},
        {m_layer_name, "enable_message_limit", VK_LAYER_SETTING_TYPE_BOOL32_EXT, 1, &enable_message_limit},
        {m_layer_name, "duplicate_message_limit", VK_LAYER_SETTING_TYPE_UINT32_EXT, 1, &duplicate_message_limit},
        {m_layer_name, "message_format_display_application_name", VK_LAYER_SETTING_TYPE_BOOL32_EXT, 1, &message_format_display_application_name},

        // Layer Enables
        {m_layer_name, "enables", VK_LAYER_SETTING_TYPE_STRING_EXT, 1, &layerEnables},
    };
    // clang-format on

    m_layerSettingsCreateInfo.sType        = VK_STRUCTURE_TYPE_LAYER_SETTINGS_CREATE_INFO_EXT;
    m_layerSettingsCreateInfo.settingCount = static_cast<uint32_t>(m_settings.size());
    m_layerSettingsCreateInfo.pSettings    = m_settings.data();
  }

  VkLayerSettingsCreateInfoEXT   m_layerSettingsCreateInfo{};
  std::vector<VkLayerSettingEXT> m_settings;
  static constexpr const char*   m_layer_name{"VK_LAYER_KHRONOS_validation"};
};

}  // namespace nvvk
