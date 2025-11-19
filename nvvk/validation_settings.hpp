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
//
// Usage:
//  nvvk::ValidationSettings vvlInfo{};
//  vvlInfo.fine_grained_locking = false;  // Customize individual settings as needed
//  vkSetup.instanceCreateInfoExt = vvlInfo.buildPNextChain();
//
//  // Explicit preset selection:
//  nvvk::ValidationSettings vvlInfo{};
//  vvlInfo.setPreset(nvvk::ValidationSettings::LayerPresets::eDebugPrintf);
//  vvlInfo.printf_buffer_size = 4096;   // Customize individual settings as needed
//  vvlInfo.printf_to_stdout = VK_FALSE; // Allow capture
//  vkSetup.instanceCreateInfoExt = vvlInfo.buildPNextChain();
//
// Note: buildPNextChain() should be called only once. The settings are applied
//       when the Vulkan instance is created.
//
// Note: recommended, use nvvk::ValidationSettings::LayerPresets::eStandard
//
// Official Documentation:
// https://vulkan.lunarg.com/doc/view/latest/windows/khronos_validation_layer.html
#pragma once

#include <vector>
#include <cstring>  // for memset
#include <vulkan/vulkan.h>

namespace nvvk {

struct ValidationSettings
{
  enum class LayerPresets
  {
    eDefault,          // Default validation layer settings
    eStandard,         // Good default validation setup that balances validation coverage and performance
    eReducedOverhead,  // Disables some checks in the interest of better performance
    eBestPractices,    // Provides warnings on valid API usage that is potential API misuse
    eSynchronization,  // Identify resource access conflicts due to missing or incorrect synchronization
    eGpuAssisted,      // Check for API usage errors at shader execution time
    eDebugPrintf,      // Enable debug printf features
  };

  // Core Validation Settings
  VkBool32 validate_core{VK_TRUE};          // Core
  VkBool32 check_image_layout{VK_TRUE};     // Image Layout
  VkBool32 check_command_buffer{VK_TRUE};   // Command Buffer State
  VkBool32 check_object_in_use{VK_TRUE};    // Object in Use
  VkBool32 check_query{VK_TRUE};            // Query
  VkBool32 check_shaders{VK_TRUE};          // Shader
  VkBool32 check_shaders_caching{VK_TRUE};  // Caching
  // VkBool32 debug_disable_spirv_val{VK_FALSE};  // Disable spirv-val (allows normal shader validation to run, but removes just spirv-val for performance reasons)
  VkBool32 unique_handles{VK_TRUE};   // Handle Wrapping
  VkBool32 object_lifetime{VK_TRUE};  // Object Lifetime
  VkBool32 stateless_param{VK_TRUE};  // Stateless Parameter
  VkBool32 thread_safety{VK_TRUE};    // Thread Safety

  // Synchronization Settings
  VkBool32 validate_sync{VK_FALSE};                      // Synchronization
  VkBool32 syncval_submit_time_validation{VK_TRUE};      // Submit time validation
  VkBool32 syncval_shader_accesses_heuristic{VK_FALSE};  // Shader accesses heuristic
  VkBool32 syncval_message_extra_properties{VK_FALSE};   // Extra properties

  // GPU Validation Settings
  VkBool32                 gpuav_enable{VK_FALSE};                       // GPU Assisted Validation
  VkBool32                 gpuav_safe_mode{VK_FALSE};                    // Safe Mode
  VkBool32                 gpuav_force_on_robustness{VK_FALSE};          // Force on robustness features
  VkBool32                 gpuav_shader_instrumentation{VK_TRUE};        // Shader instrumentation
  VkBool32                 gpuav_select_instrumented_shaders{VK_FALSE};  // Enable instrumenting shaders selectively
  std::vector<const char*> gpuav_shaders_to_instrument{};                // Shader/pipeline name regexes

  // GPU-AV Shader Instrumentation Settings
  VkBool32 gpuav_descriptor_checks{VK_TRUE};                 // Descriptors indexing
  VkBool32 gpuav_post_process_descriptor_indexing{VK_TRUE};  // Post process descriptor indexing
  VkBool32 gpuav_buffer_address_oob{VK_TRUE};                // Out of bounds buffer device addresses
  VkBool32 gpuav_validate_ray_query{VK_TRUE};                // RayQuery SPIR-V instructions
  VkBool32 gpuav_vertex_attribute_fetch_oob{VK_TRUE};        // Out of bounds vertex attribute fetching

  // GPU-AV Buffer Validation Settings
  VkBool32 gpuav_buffers_validation{VK_TRUE};           // Buffer content validation
  VkBool32 gpuav_indirect_draws_buffers{VK_TRUE};       // Indirect draws parameters
  VkBool32 gpuav_indirect_dispatches_buffers{VK_TRUE};  // Indirect dispatches parameters
  VkBool32 gpuav_indirect_trace_rays_buffers{VK_TRUE};  // Indirect trace rays parameters
  VkBool32 gpuav_buffer_copies{VK_TRUE};                // Buffer copies
  VkBool32 gpuav_index_buffers{VK_TRUE};                // Index buffers

  // GPU-AV Debug Settings (for layer developers)
  // VkBool32 gpuav_debug_disable_all{VK_FALSE};                    // Disable all of GPU-AV
  // VkBool32 gpuav_debug_validate_instrumented_shaders{VK_FALSE};  // Validate instrumented shaders
  // VkBool32 gpuav_debug_dump_instrumented_shaders{VK_FALSE};      // Dump instrumented shaders
  // uint32_t gpuav_debug_max_instrumentations_count{0};            // Limit how many time a pass can instrument the SPIR-V
  // VkBool32 gpuav_debug_print_instrumentation_info{VK_FALSE};     // Print SPIR-V instrumentation info

  // Debug Printf Settings
  VkBool32 printf_only_preset{VK_FALSE};  // Debug Printf only preset (a single, quick setting to turn on only DebugPrintf and turn off everything else)
  VkBool32 printf_to_stdout{VK_TRUE};  // Redirect Printf messages to stdout
  VkBool32 printf_verbose{VK_FALSE};   // Printf verbose
  VkBool32 printf_enable{VK_FALSE};    // Debug Printf
  uint32_t printf_buffer_size{1024};   // Printf buffer size

  // Best Practices Settings
  VkBool32 validate_best_practices{VK_FALSE};         // Best Practices
  VkBool32 validate_best_practices_arm{VK_FALSE};     // ARM-specific best practices
  VkBool32 validate_best_practices_amd{VK_FALSE};     // AMD-specific best practices
  VkBool32 validate_best_practices_img{VK_FALSE};     // IMG-specific best practices
  VkBool32 validate_best_practices_nvidia{VK_FALSE};  // NVIDIA-specific best practices

  // Message and Debug Settings
  std::vector<const char*> debug_action{"VK_DBG_LAYER_ACTION_LOG_MSG"};        // Debug Action
  const char*              log_filename{"stdout"};                             // Log Filename
  std::vector<const char*> report_flags{"error"};                              // Message Severity
  VkBool32                 enable_message_limit{VK_TRUE};                      // Limit Duplicated Messages
  uint32_t                 duplicate_message_limit{10};                        // Max Duplicated Messages
  std::vector<const char*> message_id_filter{};                                // Mute Message VUIDs
  VkBool32                 message_format_json{VK_FALSE};                      // JSON
  VkBool32                 message_format_display_application_name{VK_FALSE};  // Display Application Name

  // General Settings
  VkBool32 fine_grained_locking{VK_TRUE};  // Fine Grained Locking

  VkBaseInStructure* buildPNextChain()
  {
    updateSettings();
    return reinterpret_cast<VkBaseInStructure*>(&m_layerSettingsCreateInfo);
  }

  void setPreset(LayerPresets preset)
  {
    switch(preset)
    {
      case LayerPresets::eDefault:
        // Already at defaults from member initializers, nothing to change
        break;

      case LayerPresets::eStandard:
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
        thread_safety         = VK_FALSE;
        report_flags          = {"error", "warn"};
        enable_message_limit  = VK_TRUE;
        break;

      case LayerPresets::eReducedOverhead:
        // Core Validation Settings
        validate_core           = VK_TRUE;
        check_image_layout      = VK_FALSE;
        check_command_buffer    = VK_FALSE;
        check_object_in_use     = VK_FALSE;
        check_query             = VK_FALSE;
        check_shaders           = VK_TRUE;
        check_shaders_caching   = VK_TRUE;
        unique_handles          = VK_FALSE;
        object_lifetime         = VK_TRUE;
        stateless_param         = VK_TRUE;
        thread_safety           = VK_FALSE;
        validate_sync           = VK_FALSE;
        gpuav_enable            = VK_FALSE;
        printf_enable           = VK_FALSE;
        validate_best_practices = VK_FALSE;
        report_flags            = {"error"};
        enable_message_limit    = VK_TRUE;
        break;

      case LayerPresets::eBestPractices:
        validate_core           = VK_FALSE;
        check_image_layout      = VK_FALSE;
        check_command_buffer    = VK_FALSE;
        check_object_in_use     = VK_FALSE;
        check_query             = VK_FALSE;
        check_shaders           = VK_FALSE;
        check_shaders_caching   = VK_FALSE;
        unique_handles          = VK_FALSE;
        object_lifetime         = VK_FALSE;
        stateless_param         = VK_FALSE;
        thread_safety           = VK_FALSE;
        validate_sync           = VK_FALSE;
        gpuav_enable            = VK_FALSE;
        printf_enable           = VK_FALSE;
        validate_best_practices = VK_TRUE;
        debug_action            = {"VK_DBG_LAYER_ACTION_LOG_MSG"};
        report_flags            = {"error", "warn", "perf"};
        enable_message_limit    = VK_TRUE;

        break;

      case LayerPresets::eSynchronization:
        validate_core           = VK_FALSE;
        check_image_layout      = VK_FALSE;
        check_command_buffer    = VK_FALSE;
        check_object_in_use     = VK_FALSE;
        check_query             = VK_FALSE;
        check_shaders           = VK_FALSE;
        check_shaders_caching   = VK_FALSE;
        unique_handles          = VK_TRUE;
        object_lifetime         = VK_FALSE;
        stateless_param         = VK_FALSE;
        thread_safety           = VK_TRUE;
        validate_sync           = VK_TRUE;
        gpuav_enable            = VK_FALSE;
        printf_enable           = VK_FALSE;
        validate_best_practices = VK_FALSE;
        debug_action            = {"VK_DBG_LAYER_ACTION_LOG_MSG"};
        report_flags            = {"error"};
        enable_message_limit    = VK_TRUE;
        break;

      case LayerPresets::eGpuAssisted:
        validate_core                     = VK_FALSE;
        check_image_layout                = VK_FALSE;
        check_command_buffer              = VK_FALSE;
        check_object_in_use               = VK_FALSE;
        check_query                       = VK_FALSE;
        check_shaders                     = VK_FALSE;
        check_shaders_caching             = VK_FALSE;
        unique_handles                    = VK_FALSE;
        object_lifetime                   = VK_FALSE;
        stateless_param                   = VK_FALSE;
        thread_safety                     = VK_FALSE;
        validate_sync                     = VK_FALSE;
        gpuav_enable                      = VK_TRUE;
        gpuav_shader_instrumentation      = VK_TRUE;
        gpuav_select_instrumented_shaders = VK_FALSE;
        gpuav_buffers_validation          = VK_TRUE;
        printf_enable                     = VK_FALSE;
        validate_best_practices           = VK_FALSE;
        debug_action                      = {"VK_DBG_LAYER_ACTION_LOG_MSG"};
        report_flags                      = {"error"};
        enable_message_limit              = VK_TRUE;
        break;

      case LayerPresets::eDebugPrintf:
        validate_core           = VK_FALSE;
        check_image_layout      = VK_FALSE;
        check_command_buffer    = VK_FALSE;
        check_object_in_use     = VK_FALSE;
        check_query             = VK_FALSE;
        check_shaders           = VK_FALSE;
        check_shaders_caching   = VK_FALSE;
        unique_handles          = VK_FALSE;
        object_lifetime         = VK_FALSE;
        stateless_param         = VK_FALSE;
        thread_safety           = VK_FALSE;
        validate_sync           = VK_FALSE;
        gpuav_enable            = VK_FALSE;
        printf_enable           = VK_TRUE;
        validate_best_practices = VK_FALSE;
        debug_action            = {};
        report_flags            = {"error", "info"};
        enable_message_limit    = VK_FALSE;
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
        // {m_layer_name, "debug_disable_spirv_val", VK_LAYER_SETTING_TYPE_BOOL32_EXT, 1, &debug_disable_spirv_val},
        {m_layer_name, "unique_handles", VK_LAYER_SETTING_TYPE_BOOL32_EXT, 1, &unique_handles},
        {m_layer_name, "object_lifetime", VK_LAYER_SETTING_TYPE_BOOL32_EXT, 1, &object_lifetime},
        {m_layer_name, "stateless_param", VK_LAYER_SETTING_TYPE_BOOL32_EXT, 1, &stateless_param},
        {m_layer_name, "thread_safety", VK_LAYER_SETTING_TYPE_BOOL32_EXT, 1, &thread_safety},

        // Synchronization Settings
        {m_layer_name, "validate_sync", VK_LAYER_SETTING_TYPE_BOOL32_EXT, 1, &validate_sync},
        {m_layer_name, "syncval_submit_time_validation", VK_LAYER_SETTING_TYPE_BOOL32_EXT, 1, &syncval_submit_time_validation},
        {m_layer_name, "syncval_shader_accesses_heuristic", VK_LAYER_SETTING_TYPE_BOOL32_EXT, 1, &syncval_shader_accesses_heuristic},
        {m_layer_name, "syncval_message_extra_properties", VK_LAYER_SETTING_TYPE_BOOL32_EXT, 1, &syncval_message_extra_properties},

        // GPU Validation Settings (Official)
        {m_layer_name, "gpuav_enable", VK_LAYER_SETTING_TYPE_BOOL32_EXT, 1, &gpuav_enable},
        {m_layer_name, "gpuav_safe_mode", VK_LAYER_SETTING_TYPE_BOOL32_EXT, 1, &gpuav_safe_mode},
        {m_layer_name, "gpuav_force_on_robustness", VK_LAYER_SETTING_TYPE_BOOL32_EXT, 1, &gpuav_force_on_robustness},
        {m_layer_name, "gpuav_shader_instrumentation", VK_LAYER_SETTING_TYPE_BOOL32_EXT, 1, &gpuav_shader_instrumentation},
        {m_layer_name, "gpuav_select_instrumented_shaders", VK_LAYER_SETTING_TYPE_BOOL32_EXT, 1, &gpuav_select_instrumented_shaders},
        {m_layer_name, "gpuav_shaders_to_instrument", VK_LAYER_SETTING_TYPE_STRING_EXT, uint32_t(gpuav_shaders_to_instrument.size()), gpuav_shaders_to_instrument.data()},
        
        // GPU-AV Shader Instrumentation Settings
        {m_layer_name, "gpuav_descriptor_checks", VK_LAYER_SETTING_TYPE_BOOL32_EXT, 1, &gpuav_descriptor_checks},
        {m_layer_name, "gpuav_post_process_descriptor_indexing", VK_LAYER_SETTING_TYPE_BOOL32_EXT, 1, &gpuav_post_process_descriptor_indexing},
        {m_layer_name, "gpuav_buffer_address_oob", VK_LAYER_SETTING_TYPE_BOOL32_EXT, 1, &gpuav_buffer_address_oob},
        {m_layer_name, "gpuav_validate_ray_query", VK_LAYER_SETTING_TYPE_BOOL32_EXT, 1, &gpuav_validate_ray_query},
        {m_layer_name, "gpuav_vertex_attribute_fetch_oob", VK_LAYER_SETTING_TYPE_BOOL32_EXT, 1, &gpuav_vertex_attribute_fetch_oob},
        
        // GPU-AV Buffer Validation Settings
        {m_layer_name, "gpuav_buffers_validation", VK_LAYER_SETTING_TYPE_BOOL32_EXT, 1, &gpuav_buffers_validation},
        {m_layer_name, "gpuav_indirect_draws_buffers", VK_LAYER_SETTING_TYPE_BOOL32_EXT, 1, &gpuav_indirect_draws_buffers},
        {m_layer_name, "gpuav_indirect_dispatches_buffers", VK_LAYER_SETTING_TYPE_BOOL32_EXT, 1, &gpuav_indirect_dispatches_buffers},
        {m_layer_name, "gpuav_indirect_trace_rays_buffers", VK_LAYER_SETTING_TYPE_BOOL32_EXT, 1, &gpuav_indirect_trace_rays_buffers},
        {m_layer_name, "gpuav_buffer_copies", VK_LAYER_SETTING_TYPE_BOOL32_EXT, 1, &gpuav_buffer_copies},
        {m_layer_name, "gpuav_index_buffers", VK_LAYER_SETTING_TYPE_BOOL32_EXT, 1, &gpuav_index_buffers},
        
        // GPU-AV Debug Settings
        // {m_layer_name, "gpuav_debug_disable_all", VK_LAYER_SETTING_TYPE_BOOL32_EXT, 1, &gpuav_debug_disable_all},
        // {m_layer_name, "gpuav_debug_validate_instrumented_shaders", VK_LAYER_SETTING_TYPE_BOOL32_EXT, 1, &gpuav_debug_validate_instrumented_shaders},
        // {m_layer_name, "gpuav_debug_dump_instrumented_shaders", VK_LAYER_SETTING_TYPE_BOOL32_EXT, 1, &gpuav_debug_dump_instrumented_shaders},
        // {m_layer_name, "gpuav_debug_max_instrumentations_count", VK_LAYER_SETTING_TYPE_UINT32_EXT, 1, &gpuav_debug_max_instrumentations_count},
        // {m_layer_name, "gpuav_debug_print_instrumentation_info", VK_LAYER_SETTING_TYPE_BOOL32_EXT, 1, &gpuav_debug_print_instrumentation_info},

        // Debug Printf Settings
        {m_layer_name, "printf_only_preset", VK_LAYER_SETTING_TYPE_BOOL32_EXT, 1, &printf_only_preset},
        {m_layer_name, "printf_to_stdout", VK_LAYER_SETTING_TYPE_BOOL32_EXT, 1, &printf_to_stdout},
        {m_layer_name, "printf_verbose", VK_LAYER_SETTING_TYPE_BOOL32_EXT, 1, &printf_verbose},
        {m_layer_name, "printf_buffer_size", VK_LAYER_SETTING_TYPE_UINT32_EXT, 1, &printf_buffer_size},
        {m_layer_name, "printf_enable", VK_LAYER_SETTING_TYPE_BOOL32_EXT, 1, &printf_enable},

        // Best Practices Settings
        {m_layer_name, "validate_best_practices", VK_LAYER_SETTING_TYPE_BOOL32_EXT, 1, &validate_best_practices},
        {m_layer_name, "validate_best_practices_arm", VK_LAYER_SETTING_TYPE_BOOL32_EXT, 1, &validate_best_practices_arm},
        {m_layer_name, "validate_best_practices_amd", VK_LAYER_SETTING_TYPE_BOOL32_EXT, 1, &validate_best_practices_amd},
        {m_layer_name, "validate_best_practices_img", VK_LAYER_SETTING_TYPE_BOOL32_EXT, 1, &validate_best_practices_img},
        {m_layer_name, "validate_best_practices_nvidia", VK_LAYER_SETTING_TYPE_BOOL32_EXT, 1, &validate_best_practices_nvidia},

        // Message and Debug Settings
        {m_layer_name, "debug_action", VK_LAYER_SETTING_TYPE_STRING_EXT, uint32_t(debug_action.size()), debug_action.data()},
        {m_layer_name, "log_filename", VK_LAYER_SETTING_TYPE_STRING_EXT, 1, &log_filename},
        {m_layer_name, "report_flags", VK_LAYER_SETTING_TYPE_STRING_EXT, uint32_t(report_flags.size()), report_flags.data()},
        {m_layer_name, "enable_message_limit", VK_LAYER_SETTING_TYPE_BOOL32_EXT, 1, &enable_message_limit},
        {m_layer_name, "duplicate_message_limit", VK_LAYER_SETTING_TYPE_UINT32_EXT, 1, &duplicate_message_limit},
        {m_layer_name, "message_id_filter", VK_LAYER_SETTING_TYPE_STRING_EXT, uint32_t(message_id_filter.size()), message_id_filter.data()},
        {m_layer_name, "message_format_json", VK_LAYER_SETTING_TYPE_BOOL32_EXT, 1, &message_format_json},
        {m_layer_name, "message_format_display_application_name", VK_LAYER_SETTING_TYPE_BOOL32_EXT, 1, &message_format_display_application_name},
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
