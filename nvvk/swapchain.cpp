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


#include "nvutils/logger.hpp"

#include "commands.hpp"
#include "barriers.hpp"
#include "check_error.hpp"
#include "debug_util.hpp"
#include "swapchain.hpp"

VkResult nvvk::Swapchain::init(const InitInfo& info)
{
  m_physicalDevice = info.physicalDevice;
  m_device         = info.device;
  m_queue          = info.queue;
  m_surface        = info.surface;
  m_cmdPool        = info.cmdPool;
  if(info.preferredVsyncOffMode != VK_PRESENT_MODE_MAX_ENUM_KHR)
    m_preferredVsyncOffMode = info.preferredVsyncOffMode;
  if(info.preferredVsyncOnMode != VK_PRESENT_MODE_MAX_ENUM_KHR)
    m_preferredVsyncOnMode = info.preferredVsyncOnMode;

  VkBool32 supportsPresent = VK_FALSE;
  NVVK_FAIL_RETURN(vkGetPhysicalDeviceSurfaceSupportKHR(info.physicalDevice, info.queue.familyIndex, info.surface, &supportsPresent));

  if(supportsPresent != VK_TRUE)
  {
    LOGW("Selected queue family %d cannot present on surface %px. Swapchain creation failed.\n", info.queue.familyIndex, info.surface);
    return VK_ERROR_INITIALIZATION_FAILED;
  }

  return VK_SUCCESS;
}

void nvvk::Swapchain::deinit()
{
  if(m_device)
  {
    deinitResources();
  }
  *this = {};
}

VkResult nvvk::Swapchain::initResources(VkExtent2D& outWindowSize, bool vSync)
{
  // Query the physical device's capabilities for the given surface.
  const VkPhysicalDeviceSurfaceInfo2KHR surfaceInfo2{.sType   = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SURFACE_INFO_2_KHR,
                                                     .surface = m_surface};
  VkSurfaceCapabilities2KHR             capabilities2{.sType = VK_STRUCTURE_TYPE_SURFACE_CAPABILITIES_2_KHR};
  NVVK_FAIL_RETURN(vkGetPhysicalDeviceSurfaceCapabilities2KHR(m_physicalDevice, &surfaceInfo2, &capabilities2));

  uint32_t formatCount;
  NVVK_FAIL_RETURN(vkGetPhysicalDeviceSurfaceFormats2KHR(m_physicalDevice, &surfaceInfo2, &formatCount, nullptr));
  std::vector<VkSurfaceFormat2KHR> formats(formatCount, {.sType = VK_STRUCTURE_TYPE_SURFACE_FORMAT_2_KHR});
  NVVK_FAIL_RETURN(vkGetPhysicalDeviceSurfaceFormats2KHR(m_physicalDevice, &surfaceInfo2, &formatCount, formats.data()));

  uint32_t presentModeCount;
  NVVK_FAIL_RETURN(vkGetPhysicalDeviceSurfacePresentModesKHR(m_physicalDevice, m_surface, &presentModeCount, nullptr));
  std::vector<VkPresentModeKHR> presentModes(presentModeCount);
  NVVK_FAIL_RETURN(
      vkGetPhysicalDeviceSurfacePresentModesKHR(m_physicalDevice, m_surface, &presentModeCount, presentModes.data()));

  // Choose the best available surface format and present mode
  const VkSurfaceFormat2KHR surfaceFormat2 = selectSwapSurfaceFormat(formats);
  const VkPresentModeKHR    presentMode    = selectSwapPresentMode(presentModes, vSync);
  // Set the window size according to the surface's current extent
  outWindowSize = capabilities2.surfaceCapabilities.currentExtent;
  // Set the number of images in flight, respecting the GPU's maxImageCount limit.
  // If maxImageCount is equal to 0, then there is no limit other than memory,
  // so don't change m_maxFramesInFlight.
  if(capabilities2.surfaceCapabilities.maxImageCount > 0)
  {
    m_maxFramesInFlight = std::min(m_maxFramesInFlight, capabilities2.surfaceCapabilities.maxImageCount);
  }
  // Store the chosen image format
  m_imageFormat = surfaceFormat2.surfaceFormat.format;

  // Create the swapchain itself
  const VkSwapchainCreateInfoKHR swapchainCreateInfo{
      .sType            = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
      .surface          = m_surface,
      .minImageCount    = m_maxFramesInFlight,
      .imageFormat      = surfaceFormat2.surfaceFormat.format,
      .imageColorSpace  = surfaceFormat2.surfaceFormat.colorSpace,
      .imageExtent      = capabilities2.surfaceCapabilities.currentExtent,
      .imageArrayLayers = 1,
      .imageUsage       = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
      .imageSharingMode = VK_SHARING_MODE_EXCLUSIVE,
      .preTransform     = capabilities2.surfaceCapabilities.currentTransform,
      .compositeAlpha   = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
      .presentMode      = presentMode,
      .clipped          = VK_TRUE,
  };
  NVVK_FAIL_RETURN(vkCreateSwapchainKHR(m_device, &swapchainCreateInfo, nullptr, &m_swapChain));
  NVVK_DBG_NAME(m_swapChain);

  // Retrieve the swapchain images
  uint32_t imageCount;
  NVVK_FAIL_RETURN(vkGetSwapchainImagesKHR(m_device, m_swapChain, &imageCount, nullptr));
  // On llvmpipe for instance, we can get more images than the minimum requested.
  // We still need to get a handle for each image in the swapchain
  // (because vkAcquireNextImageKHR can return an index to each image),
  // so adjust m_maxFramesInFlight.
  assert((m_maxFramesInFlight <= imageCount) && "Wrong swapchain setup");
  m_maxFramesInFlight = imageCount;
  std::vector<VkImage> swapImages(m_maxFramesInFlight);
  NVVK_FAIL_RETURN(vkGetSwapchainImagesKHR(m_device, m_swapChain, &m_maxFramesInFlight, swapImages.data()));

  // Store the swapchain images and create views for them
  m_images.resize(m_maxFramesInFlight);
  VkImageViewCreateInfo imageViewCreateInfo{
      .sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
      .viewType = VK_IMAGE_VIEW_TYPE_2D,
      .format   = m_imageFormat,
      .components = {.r = VK_COMPONENT_SWIZZLE_IDENTITY, .g = VK_COMPONENT_SWIZZLE_IDENTITY, .b = VK_COMPONENT_SWIZZLE_IDENTITY, .a = VK_COMPONENT_SWIZZLE_IDENTITY},
      .subresourceRange = {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .baseMipLevel = 0, .levelCount = 1, .baseArrayLayer = 0, .layerCount = 1},
  };
  for(uint32_t i = 0; i < m_maxFramesInFlight; i++)
  {
    m_images[i].image = swapImages[i];
    NVVK_DBG_NAME(m_images[i].image);
    imageViewCreateInfo.image = m_images[i].image;
    NVVK_FAIL_RETURN(vkCreateImageView(m_device, &imageViewCreateInfo, nullptr, &m_images[i].imageView));
    NVVK_DBG_NAME(m_images[i].imageView);
  }

  // Initialize frame resources for each swapchain image
  m_frameResources.resize(m_maxFramesInFlight);
  for(size_t i = 0; i < m_maxFramesInFlight; ++i)
  {
    /*--
       * The sync objects are used to synchronize the rendering with the presentation.
       * The image available semaphore is signaled when the image is available to render.
       * The render finished semaphore is signaled when the rendering is finished.
       * The in flight fence is signaled when the frame is in flight.
      -*/
    const VkSemaphoreCreateInfo semaphoreCreateInfo{.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    NVVK_FAIL_RETURN(vkCreateSemaphore(m_device, &semaphoreCreateInfo, nullptr, &m_frameResources[i].imageAvailableSemaphore));
    NVVK_DBG_NAME(m_frameResources[i].imageAvailableSemaphore);
    NVVK_FAIL_RETURN(vkCreateSemaphore(m_device, &semaphoreCreateInfo, nullptr, &m_frameResources[i].renderFinishedSemaphore));
    NVVK_DBG_NAME(m_frameResources[i].renderFinishedSemaphore);
  }

  // Transition images to present layout
  {
    VkCommandBuffer cmd{};
    NVVK_FAIL_RETURN(nvvk::beginSingleTimeCommands(cmd, m_device, m_cmdPool));
    for(uint32_t i = 0; i < m_maxFramesInFlight; i++)
    {
      cmdImageMemoryBarrier(cmd, {m_images[i].image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR});
    }
    NVVK_FAIL_RETURN(nvvk::endSingleTimeCommands(cmd, m_device, m_cmdPool, m_queue.queue));
  }

  return VK_SUCCESS;
}

VkResult nvvk::Swapchain::reinitResources(VkExtent2D& outWindowSize, bool vSync)
{
  // Wait for all frames to finish rendering before recreating the swapchain
  vkQueueWaitIdle(m_queue.queue);

  m_frameResourceIndex = 0;
  m_needRebuild        = false;
  deinitResources();
  return initResources(outWindowSize, vSync);
}

void nvvk::Swapchain::deinitResources()
{
  vkDestroySwapchainKHR(m_device, m_swapChain, nullptr);
  for(auto& frameRes : m_frameResources)
  {
    vkDestroySemaphore(m_device, frameRes.imageAvailableSemaphore, nullptr);
    vkDestroySemaphore(m_device, frameRes.renderFinishedSemaphore, nullptr);
  }
  m_frameResources.clear();
  for(auto& image : m_images)
  {
    vkDestroyImageView(m_device, image.imageView, nullptr);
  }
  m_images.clear();
}

VkResult nvvk::Swapchain::acquireNextImage(VkDevice device)
{
  assert((m_needRebuild == false) && "Swapbuffer need to call reinitResources()");

  // Get the frame resources for the current frame
  // We use m_currentFrame here because we want to ensure we don't overwrite resources
  // that are still in use by previous frames
  auto& frame = m_frameResources[m_frameResourceIndex];

  // Acquire the next image from the swapchain
  // This will signal frame.imageAvailableSemaphore when the image is ready
  // and store the index of the acquired image in m_nextImageIndex
  VkResult result = vkAcquireNextImageKHR(device, m_swapChain, std::numeric_limits<uint64_t>::max(),
                                          frame.imageAvailableSemaphore, VK_NULL_HANDLE, &m_frameImageIndex);

  switch(result)
  {
    case VK_SUCCESS:
    case VK_SUBOPTIMAL_KHR:  // Still valid for presentation
      return result;

    case VK_ERROR_OUT_OF_DATE_KHR:  // The swapchain is no longer compatible with the surface and needs to be recreated
      m_needRebuild = true;
      return result;

    default:
      LOGW("Failed to acquire swapchain image: %d\n", result);
      return result;
  }
}

void nvvk::Swapchain::presentFrame(VkQueue queue)
{
  // Get the frame resources for the current image
  // We use m_nextImageIndex here because we want to signal the semaphore
  // associated with the image we just finished rendering
  auto& frame = m_frameResources[m_frameImageIndex];

  // Setup the presentation info, linking the swapchain and the image index
  const VkPresentInfoKHR presentInfo{
      .sType              = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
      .waitSemaphoreCount = 1,                               // Wait for rendering to finish
      .pWaitSemaphores    = &frame.renderFinishedSemaphore,  // Synchronize presentation
      .swapchainCount     = 1,                               // Swapchain to present the image
      .pSwapchains        = &m_swapChain,                    // Pointer to the swapchain
      .pImageIndices      = &m_frameImageIndex,              // Index of the image to present
  };

  // Present the image and handle potential resizing issues
  const VkResult result = vkQueuePresentKHR(queue, &presentInfo);
  // If the swapchain is out of date (e.g., window resized), it needs to be rebuilt
  if(result == VK_ERROR_OUT_OF_DATE_KHR)
  {
    m_needRebuild = true;
  }
  else
  {
    assert((result == VK_SUCCESS || result == VK_SUBOPTIMAL_KHR) && "Couldn't present swapchain image");
  }

  // Advance to the next frame in the swapchain
  m_frameResourceIndex = (m_frameResourceIndex + 1) % m_maxFramesInFlight;
}

VkSurfaceFormat2KHR nvvk::Swapchain::selectSwapSurfaceFormat(const std::vector<VkSurfaceFormat2KHR>& availableFormats) const
{
  // If there's only one available format and it's undefined, return a default format.
  if(availableFormats.size() == 1 && availableFormats[0].surfaceFormat.format == VK_FORMAT_UNDEFINED)
  {
    VkSurfaceFormat2KHR result{.sType         = VK_STRUCTURE_TYPE_SURFACE_FORMAT_2_KHR,
                               .surfaceFormat = {VK_FORMAT_B8G8R8A8_UNORM, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR}};
    return result;
  }

  const std::vector<VkSurfaceFormat2KHR> preferredFormats = {
      VkSurfaceFormat2KHR{.sType         = VK_STRUCTURE_TYPE_SURFACE_FORMAT_2_KHR,
                          .surfaceFormat = {VK_FORMAT_B8G8R8A8_UNORM, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR}},
      VkSurfaceFormat2KHR{.sType         = VK_STRUCTURE_TYPE_SURFACE_FORMAT_2_KHR,
                          .surfaceFormat = {VK_FORMAT_R8G8B8A8_UNORM, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR}}};

  // Check available formats against the preferred formats.
  for(const auto& preferredFormat : preferredFormats)
  {
    for(const auto& availableFormat : availableFormats)
    {
      if(availableFormat.surfaceFormat.format == preferredFormat.surfaceFormat.format
         && availableFormat.surfaceFormat.colorSpace == preferredFormat.surfaceFormat.colorSpace)
      {
        return availableFormat;  // Return the first matching preferred format.
      }
    }
  }

  // If none of the preferred formats are available, return the first available format.
  return availableFormats[0];
}

VkPresentModeKHR nvvk::Swapchain::selectSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes, bool vSync) const
{
  bool mailboxSupported = false, immediateSupported = false;

  for(VkPresentModeKHR mode : availablePresentModes)
  {
    if(vSync && (mode == m_preferredVsyncOnMode))
      return mode;

    if(!vSync && (mode == m_preferredVsyncOffMode))
      return mode;

    if(mode == VK_PRESENT_MODE_MAILBOX_KHR)
      mailboxSupported = true;
    if(mode == VK_PRESENT_MODE_IMMEDIATE_KHR)
      immediateSupported = true;
  }

  if(!vSync && immediateSupported)
  {
    return VK_PRESENT_MODE_IMMEDIATE_KHR;  // Best mode for low latency
  }

  if(mailboxSupported)
  {
    return VK_PRESENT_MODE_MAILBOX_KHR;
  }

  return VK_PRESENT_MODE_FIFO_KHR;
}
