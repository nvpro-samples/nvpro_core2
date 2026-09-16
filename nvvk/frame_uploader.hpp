/*
* Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
* SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
* SPDX-License-Identifier: Apache-2.0
*/

#pragma once

#include <nvvk/resources.hpp>
#include <nvvk/semaphore.hpp>
#include <nvvk/staging.hpp>
#include <nvvk/buffer_circular_allocator.hpp>
#include <nvvk/command_pools.hpp>

#include <nvvk/uploader_interface.hpp>

namespace nvvk {

// This class is meant to handle smaller data transfers that
// are done every frame.
//
// The intent of this class and `nvvk::AsyncUploader` is to replace the
// old `nvvk::StagingUploader` which we want to deprecate.
class FrameUploader : public CmdUploaderInterface
{
public:
  FrameUploader() = default;
  ~FrameUploader();
  FrameUploader(const FrameUploader&)            = delete;
  FrameUploader& operator=(const FrameUploader&) = delete;

  struct InitInfo
  {
    // resource allocator
    nvvk::ResourceAllocator* allocator{};
    // we always keep at least 1 block around
    VkDeviceSize blockSize = 4 * 1024 * 1024;
    // debug prefix for vulkan resources
    std::string debugName{};

    // if true, staging memory is guaranteed to have VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
    // if false, non-coherent host memory may be used and is flushed before GPU copy
    bool forceCoherentMapping = true;
  };

  struct Stats
  {
    size_t copyCount{};
    size_t copyBytes{};
  };

  VkResult init(const InitInfo& info);
  void     deinit();

  void updateFrameSemaphoreState(const nvvk::SemaphoreState& semaphoreState);
  void releaseCompletedAllocations(bool forceAll = false);

  //////////////////////////////////////////////////////////////////////////
  // UploaderInterface

  VkResult acquireMapping(size_t dataSize, nvvk::BufferRange& mappingSpace, MappingHandle& mappingHandle) override;
  void     commitMapping(MappingHandle& mappingHandle) override;
  void     releaseMapping(MappingHandle& mappingHandle) override;

  VkResult appendBufferRange(const nvvk::BufferRange& buffer, const void* data, nvvk::SemaphoreState* outSemaphoreState) override;
  VkResult appendBufferRangeMappings(size_t                    mappingCount,
                                     const MappedBufferRanges* ranges,
                                     MappingHandle&            mappingHandle,
                                     bool                      commitMapping_    = true,
                                     nvvk::SemaphoreState*     outSemaphoreState = nullptr) override;

  VkResult appendImageSub(nvvk::Image&                    image,
                          const VkOffset3D&               offset,
                          const VkExtent3D&               extent,
                          const VkImageSubresourceLayers& subresource,
                          size_t                          dataSize,
                          const void*                     data,
                          VkImageLayout                   newLayout         = VK_IMAGE_LAYOUT_UNDEFINED,
                          nvvk::SemaphoreState*           outSemaphoreState = nullptr) override;
  VkResult appendImageSubMappings(nvvk::Image&           image,
                                  size_t                 imageSubCount,
                                  const MappedImageSubs* imageSubs,
                                  MappingHandle&         mappingHandle,
                                  bool                   commitMapping_    = true,
                                  VkImageLayout          newLayout         = VK_IMAGE_LAYOUT_UNDEFINED,
                                  nvvk::SemaphoreState*  outSemaphoreState = nullptr) override;

  //////////////////////////////////////////////////////////////////////////
  // CmdUploaderInterface

  void cmdUploadAppended(VkCommandBuffer cmd) override;


protected:
  union MappingHandleDetail
  {
    uint64_t handle;
    struct
    {
      uint8_t  valid;
      uint8_t  used;
      uint16_t blockIndex;
      uint32_t frameIndex;
    };
  };

  VkResult acquireStagingSpace(nvvk::BufferCircularAllocation& subAllocation, nvvk::BufferRange& stagingSpace, size_t dataSize, const void* data);

  void markBlockForFlush(uint32_t blockIndex);
  void flushBatchBlocks();

  inline void getOutSemaphoreState(nvvk::SemaphoreState* out)
  {
    if(out != nullptr)
    {
      *out = m_semaphoreState;
    }
  }

  struct StagingAllocation
  {
    nvvk::BufferCircularAllocation subAllocation;
    nvvk::SemaphoreState           semaphoreState;
  };

  static constexpr uint32_t MAX_FLUSH_BLOCKS = 64;

  uint32_t                       m_frameIndex{0};
  bool                           m_forceCoherentMapping = true;
  nvvk::ResourceAllocator*       m_resourceAllocator    = nullptr;
  nvvk::StagingCopyBatch         m_batch;
  nvvk::BufferCircularAllocator  m_bufferCircularAllocator;
  Stats                          m_stats;
  uint64_t                       m_batchFlushBlockMask = 0;
  nvvk::SemaphoreState           m_semaphoreState;
  std::vector<StagingAllocation> m_stagingAllocations;
};

}  // namespace nvvk