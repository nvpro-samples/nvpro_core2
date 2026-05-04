/*
 * Copyright (c) 2023-2026, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cassert>
#include <vector>

#include "resources.hpp"

namespace nvvk {

//--- Swapchain ------------------------------------------------------------------------------------------------------------
/*--
 * Swapchain: presents rendered images to the screen.
 *
 * There are two distinct counts; do not mix them up.
 *   - imageCount      : swapchain images (presentation parallelism, default 3).
 *   - framesInFlight  : CPU frame slots recorded ahead of the GPU (default 2).
 *
 * `presentSemaphore` is per-image (acquire can return images out of order, notably under MAILBOX)
 * `acquireSemaphore` is per in-flight slot (consumed by acquire).
 *
 * On window resize or surface loss, `needRebuild` is set and the caller must
 * call `reinitResources()`.
-*/
class Swapchain
{
public:
  Swapchain() = default;
  ~Swapchain() { assert(m_swapChain == VK_NULL_HANDLE && "Missing deinit()"); }

  void               requestRebuild() { m_needRebuild = true; }
  bool               needRebuilding() const { return m_needRebuild; }
  void               setPreferredSurfaceFormat(VkSurfaceFormatKHR format) { m_preferredSurfaceFormat = format; }
  VkSurfaceFormatKHR getPreferredSurfaceFormat() const { return m_preferredSurfaceFormat; }
  VkImage            getImage() const { return m_images[m_frameImageIndex].image; }
  VkImageView        getImageView() const { return m_images[m_frameImageIndex].imageView; }
  VkFormat           getImageFormat() const { return m_surfaceFormat.surfaceFormat.format; }

  // Number of swapchain images (presentation parallelism).
  uint32_t getImageCount() const { return m_imageCount; }

  // Number of CPU frame slots (= concurrent frames in flight).
  uint32_t getFramesInFlight() const { return m_framesInFlight; }

  // acquireSemaphore is per in-flight slot (consumed by acquire).
  VkSemaphore getAcquireSemaphore() const { return m_frameResources[m_frameResourceIndex].acquireSemaphore; }

  // presentSemaphore is per *image* (consumed by present, must follow
  // the image because acquire can return images out of order).
  VkSemaphore         getPresentSemaphore() const { return m_images[m_frameImageIndex].presentSemaphore; }
  VkSurfaceFormat2KHR getSurfaceFormat() const { return m_surfaceFormat; }

  struct InitInfo
  {
    VkPhysicalDevice physicalDevice{};
    VkDevice         device{};
    QueueInfo        queue{};
    VkSurfaceKHR     surface{};
    VkCommandPool    cmdPool{};
    VkPresentModeKHR preferredVsyncOffMode = VK_PRESENT_MODE_IMMEDIATE_KHR;
    VkPresentModeKHR preferredVsyncOnMode  = VK_PRESENT_MODE_FIFO_KHR;
    // If provided , use this surface format. Make sure to select one of the formats returned by getAvailableFormats().
    VkSurfaceFormat2KHR preferredFormat = {.sType         = VK_STRUCTURE_TYPE_SURFACE_FORMAT_2_KHR,
                                           .surfaceFormat = {VK_FORMAT_UNDEFINED, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR}};

    // Preferred swapchain image count (presentation parallelism). Clamped at
    // runtime to the surface's [minImageCount, maxImageCount] (with a floor of
    // 2 so ImGui's MinImageCount invariant holds); the driver may also return
    // more images than requested (we honour whatever the swapchain reports).
    // 3 is a good default for low-latency triple buffering.
    uint32_t preferredImageCount = 3;

    // Preferred number of CPU frame slots recorded ahead of the GPU. Clamped
    // at runtime to `[2, actualImageCount]`.
    uint32_t preferredFramesInFlight = 2;
  };

  // Initialize the swapchain with the provided context and surface, then we can create and re-create it
  VkResult init(const InitInfo& info);

  // Destroy internal resources and reset its initial state
  void deinit();

  /*--
   * Create the swapchain using the provided context, surface, and vSync option. The actual window size is returned.
   * Queries the GPU capabilities, selects the best surface format and present mode, and creates the swapchain accordingly.
  -*/
  VkResult initResources(VkExtent2D& outWindowSize, bool vSync = true);

  /*--
   * Recreate the swapchain, typically after a window resize or when it becomes invalid.
   * This waits for all rendering to be finished before destroying the old swapchain and creating a new one.
  -*/
  VkResult reinitResources(VkExtent2D& outWindowSize, bool vSync = true);

  /*--
   * Destroy the swapchain and its associated resources.
   * This function is also called when the swapchain needs to be recreated.
  -*/
  void deinitResources();

  /*--
   * Prepares the command buffer for recording rendering commands.
   * This function handles synchronization with the previous frame and acquires the next image from the swapchain.
   * The command buffer is reset, ready for new rendering commands.
  -*/
  VkResult acquireNextImage(VkDevice device);

  /*--
   * Presents the rendered image to the screen.
   * The semaphore ensures that the image is presented only after rendering is complete.
   * Advances to the next frame in the cycle.
  -*/
  void presentFrame(VkQueue queue);


  /*--
   * Retrieve all image formats the given surface can support.
  -*/
  static std::vector<VkSurfaceFormat2KHR> getAvailableFormats(VkPhysicalDevice physicalDevice, VkSurfaceKHR surface);

  /*--
   * Retrieve all image formats the swapchain's surface can support.
  -*/
  const std::vector<VkSurfaceFormat2KHR>& getAvailableFormats() const { return m_availableFormats; }

  /*--
   * Set a new preferred swapchain format. This will not immediately rebuild the swapchain with the new format.
   * Instead, needRebuilding() will now return true and the next call to reinitResources() will actually
   * rebuild the swapchain with the new format.
   * Returns false if 'format' is not supported.
  -*/
  bool setPreferredSwapchainFormat(const VkSurfaceFormat2KHR& preferred);


private:
  /*-- Per-swapchain-image resources -------------------------------------------
   * One entry per image returned by vkGetSwapchainImagesKHR. The
   * presentSemaphore lives here (not on the in-flight slot) because
   * vkQueuePresentKHR consumes it for a specific image, and the presentation
   * engine may hand images back out of order.
  -*/
  struct Image
  {
    VkImage     image{};             // Swapchain image (owned by the swapchain)
    VkImageView imageView{};         // 2D view of the image
    VkSemaphore presentSemaphore{};  // Binary semaphore: signaled when rendering done, waited on by present
  };
  /*-- Per-in-flight-slot resources --------------------------------------------
   * One entry per "frame in flight" -- typically 2, regardless of the image
   * count. Holds resources tied to the CPU's submission cadence rather than
   * the displayed image. The acquireSemaphore is recycled here.
  -*/
  struct FrameResources
  {
    VkSemaphore acquireSemaphore{};  // Binary semaphore signaled by vkAcquireNextImageKHR
  };

  // We choose the format that is the most common, and that is supported by* the physical device.
  VkSurfaceFormat2KHR selectSwapSurfaceFormat(const std::vector<VkSurfaceFormat2KHR>& availableFormats) const;

  /*--
   * The present mode is chosen based on the vSync option
   * The `preferredVsyncOnMode` is used when vSync is enabled and the mode is supported.
   * The `preferredVsyncOffMode` is used when vSync is disabled and the mode is supported.
   * Otherwise, from most preferred to least:
   *   1. IMMEDIATE mode, when vSync is disabled (tearing allowed), since it's lowest-latency.
   *   2. MAILBOX mode, since it's the lowest-latency mode without tearing. Note that frame pacing is needed
          when vSync is on.
   *   3. FIFO mode, since all swapchains must support it.
  -*/
  VkPresentModeKHR selectSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes, bool vSync = true) const;

private:
  Swapchain(Swapchain&) = delete;                    //
  Swapchain& operator=(const Swapchain&) = default;  // Allow copy only for internal use in deinit() to restore pristine state

  VkPhysicalDevice    m_physicalDevice{};  // The physical device (GPU)
  VkDevice            m_device{};          // The logical device (interface to the physical device)
  QueueInfo           m_queue{};           // The queue used to submit command buffers to the GPU
  VkSwapchainKHR      m_swapChain{};       // The swapchain
  VkSurfaceFormat2KHR m_surfaceFormat;     // the surface format used by the swapchain
  VkSurfaceKHR        m_surface{};         // The surface to present images to
  VkCommandPool       m_cmdPool{};         // The command pool for the swapchain

  std::vector<Image>          m_images{};                // The swapchain images and their views
  std::vector<FrameResources> m_frameResources{};        // Synchronization primitives for each frame
  uint32_t                    m_frameResourceIndex = 0;  // Current frame index, cycles through frame resources
  uint32_t                    m_frameImageIndex    = 0;  // Index of the swapchain image we're currently rendering to
  bool                        m_needRebuild        = false;  // Flag indicating if the swapchain needs to be rebuilt

  VkPresentModeKHR                 m_preferredVsyncOffMode = VK_PRESENT_MODE_IMMEDIATE_KHR;  // used if available
  VkPresentModeKHR                 m_preferredVsyncOnMode  = VK_PRESENT_MODE_FIFO_KHR;       // used if available
  std::vector<VkSurfaceFormat2KHR> m_availableFormats{};  // List of formats available for this surface
  VkSurfaceFormat2KHR              m_preferredFormat        = {.sType = VK_STRUCTURE_TYPE_SURFACE_FORMAT_2_KHR,
                                                               .surfaceFormat{VK_FORMAT_UNDEFINED, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR}};  // Optional preferred format
  VkSurfaceFormatKHR               m_preferredSurfaceFormat = {VK_FORMAT_UNDEFINED, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR};

  // Triple buffering allows us to pipeline CPU and GPU work, which gives us
  // good throughput if their sum takes more than a frame.
  // But if we're using VK_PRESENT_MODE_FIFO_KHR without frame pacing and
  // workloads are < 1 frame, then work can be waiting for multiple frames for
  // the swapchain image to be available, increasing latency. For this reason,
  // it's good to use a frame pacer with the swapchain.

  // User-requested preferences (set by init(), sticky across reinitResources).
  uint32_t m_preferredImageCount     = 0;
  uint32_t m_preferredFramesInFlight = 0;

  // Actual runtime values, derived in initResources() from the preferences
  // clamped to the surface's capabilities and the swapchain's actual image count.
  uint32_t m_imageCount     = 0;  // From vkGetSwapchainImagesKHR
  uint32_t m_framesInFlight = 0;  // <= m_imageCount
};


}  // namespace nvvk
