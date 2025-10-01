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
#include <cassert>
#include <string>
#include <sstream>
#include <queue>
#include <vector>

#include <glm/glm.hpp>
#include <volk.h>

#include "resource_allocator.hpp"
#include "staging.hpp"


namespace nvvk {

/*-------------------------------------------------------------------------------------------------

# `nvvk::accelerationStructureBarrier` Function
This function sets up a memory barrier specifically for acceleration structure operations in Vulkan, ensuring proper data synchronization during the build or update phases. It operates between pipeline stages that deal with acceleration structure building.

# `nvvk::toTransformMatrixKHR` Function
This function converts a `glm::mat4` matrix to the matrix format required by acceleration structures in Vulkan.

# `nvvk::AccelerationStructureGeometryInfo` Structure
- **Purpose**: Holds information about acceleration structure geometry, including the geometry structure and build range information.

# `nvvk::AccelerationStructureBuildData` Structure
- **Purpose**: Manages the building of Vulkan acceleration structures of a specified type.
- **Key Functions**:
  - `addGeometry`: Adds a geometry with build range information to the acceleration structure.
  - `finalizeGeometry`: Configures the build information and calculates the necessary size information.
  - `makeCreateInfo`: Creates an acceleration structure based on the current build and size information.
  - `cmdBuildAccelerationStructure`: Builds the acceleration structure in a Vulkan command buffer.
  - `cmdUpdateAccelerationStructure`: Updates the acceleration structure in a Vulkan command buffer.
  - `hasCompactFlag`: Checks if the compact flag is set for the build.
- **Usage**:
  - For each BLAS, 
    - Add geometry using `addGeometry`.
    - Finalize the geometry and get the size requirements using `finalizeGeometry`.
    - Keep the max scratch buffer size in mind when creating the scratch buffer.
  - Create Scratch Buffer using the information returned by finalizeGeometry.
  - For each BLAS, 
    - Create the acceleration structure using `makeCreateInfo`.

* -------------------------------------------------------------------------------------------------*/


// Helper function to insert a memory barrier for acceleration structures
inline void accelerationStructureBarrier(VkCommandBuffer cmd, VkAccessFlags src, VkAccessFlags dst)
{
  assert(src == VK_ACCESS_TRANSFER_WRITE_BIT || src == VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR);
  assert(dst == VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR || dst == VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR
         || dst == (VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR | VK_ACCESS_2_SHADER_READ_BIT)
         || dst == (VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR | VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR));

  VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
  barrier.srcAccessMask = src;
  barrier.dstAccessMask = dst;
  vkCmdPipelineBarrier(cmd, src == VK_ACCESS_TRANSFER_WRITE_BIT ? VK_PIPELINE_STAGE_TRANSFER_BIT : VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                       VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR, 0, 1, &barrier, 0, nullptr, 0, nullptr);
}

// Convert a Mat4x4 to the matrix required by acceleration structures
inline VkTransformMatrixKHR toTransformMatrixKHR(glm::mat4 matrix)
{
  // VkTransformMatrixKHR uses a row-major memory layout, while glm::mat4
  // uses a column-major memory layout. We transpose the matrix so we can
  // memcpy the matrix's data directly.
  glm::mat4            temp = glm::transpose(matrix);
  VkTransformMatrixKHR out_matrix;
  memcpy(&out_matrix, &temp, sizeof(VkTransformMatrixKHR));
  return out_matrix;
}

// Single Geometry information, multiple can be used in a single BLAS
struct AccelerationStructureGeometryInfo
{
  VkAccelerationStructureGeometryKHR       geometry{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
  VkAccelerationStructureBuildRangeInfoKHR rangeInfo{};
};

// Template for building Vulkan Acceleration Structures of a specified type.
struct AccelerationStructureBuildData
{
  VkAccelerationStructureTypeKHR asType = VK_ACCELERATION_STRUCTURE_TYPE_MAX_ENUM_KHR;  // Mandatory to set

  // Collection of geometries for the acceleration structure.
  std::vector<VkAccelerationStructureGeometryKHR> asGeometry;
  // Build range information corresponding to each geometry.
  std::vector<VkAccelerationStructureBuildRangeInfoKHR> asBuildRangeInfo;
  // Build information required for acceleration structure.
  VkAccelerationStructureBuildGeometryInfoKHR buildInfo{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
  // Size information for acceleration structure build resources.
  VkAccelerationStructureBuildSizesInfoKHR sizeInfo{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};

  // Adds a geometry with its build range information to the acceleration structure.
  void addGeometry(const VkAccelerationStructureGeometryKHR& asGeom, const VkAccelerationStructureBuildRangeInfoKHR& offset);
  void addGeometry(const AccelerationStructureGeometryInfo& asGeom);

  AccelerationStructureGeometryInfo makeInstanceGeometry(size_t numInstances, VkDeviceAddress instanceBufferAddr);

  // Configures the build information and calculates the necessary size information.
  VkAccelerationStructureBuildSizesInfoKHR finalizeGeometry(VkDevice device, VkBuildAccelerationStructureFlagsKHR flags);

  // Creates an acceleration structure based on the current build and size info.
  VkAccelerationStructureCreateInfoKHR makeCreateInfo() const;

  // Commands to build the acceleration structure in a Vulkan command buffer.
  void cmdBuildAccelerationStructure(VkCommandBuffer cmd, VkAccelerationStructureKHR accelerationStructure, VkDeviceAddress scratchAddress);

  // Commands to update the acceleration structure in a Vulkan command buffer.
  void cmdUpdateAccelerationStructure(VkCommandBuffer cmd, VkAccelerationStructureKHR accelerationStructure, VkDeviceAddress scratchAddress);

  // Checks if the compact flag is set for the build.
  bool hasCompactFlag() const { return buildInfo.flags & VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_COMPACTION_BIT_KHR; }
};

// Get the maximum scratch buffer size required for the acceleration structure build
VkDeviceSize getMaxScratchSize(const std::vector<AccelerationStructureBuildData>& asBuildData);

/*-------------------------------------------------------------------------------------------------
 
 Manages the construction and optimization of bottom-level acceleration structures (BLAS)
 
 This class is designed to facilitate the construction of BLAS based on provided build information and queries.
  - Compacts BLAS for efficient memory usage and cleans up resources.
  - It ensures that operations are performed within a specified memory budget if possible.
  - It also provides statistical data on the compaction results.
 Note: works on a single vector of `AccelerationStructureBuildData` at a time.

 Usage:
 - Initialize a BlasBuilder object with a resource allocator via `init`
 - Query the scratch buffer size via `getScratchSize` (providing a reasonable budget allows to run more builds in parallel)
 - Create scratch buffer

 Within a loop
   - Call `cmdCreateBlas` to create all or a range of BLAS that are provided BlasBuildData
   - Submit command buffer and wait
   - Call `cmdCompactBlas` to compact the BLAS that have been built thus far.
   - Call `destroyNonCompactedBlas` to destroy the original BLAS that were compacted.
   - Continue loop if `cmdCreateBlas` returned `VK_INCOMPLETE`
 - Call `getStatistics` to get statistics about the compacted BLAS.
 - Call `deinit` to clean up all resources. 
 
 --------------------------------------------------------------------------------------------------- */
// AsBuilder ->  AccelerationStructurePack

// BlasBuilder -> AccelerationStructureBuilder

class AccelerationStructureBuilder
{
public:
  AccelerationStructureBuilder()                                    = default;
  AccelerationStructureBuilder(const AccelerationStructureBuilder&) = delete;

  ~AccelerationStructureBuilder() { assert(m_queryPool == VK_NULL_HANDLE); }  // Missing deinit() call

  void init(nvvk::ResourceAllocator* allocator);
  void deinit();

  struct Stats
  {
    VkDeviceSize totalOriginalSize = 0;
    VkDeviceSize totalCompactSize  = 0;

    std::string toString() const;
  };


  // Create the BLAS from the vector of BlasBuildData in multiple iterations.
  // The advantage of this function is that it will try to build as many BLAS as possible in parallel
  // as the `scratchSize` and `hintMaxBudget` allows. Higher `scratchSize` allows less barriers, and
  // higher `hintMaxBudget` means less need to call this function multiple times.
  //
  // Returns VK_SUCCESS if the entire input vector was processed,
  // returns VK_INCOMPLETE if this function needs to be called again until it returns VK_SUCCESS
  // Other result codes are to be treated as error.
  VkResult cmdCreateBlas(VkCommandBuffer                            cmd,
                         std::span<AccelerationStructureBuildData>& blasBuildData,  // List of the BLAS to build */
                         std::span<nvvk::AccelerationStructure>&    blasAccel,  // List of the acceleration structure
                         VkDeviceAddress                            scratchAddress,  //  Address of the scratch buffer
                         VkDeviceSize                               scratchSize,     //  Size of the scratch buffer
                         VkDeviceSize hintMaxBudget = 512'000'000  // Ceiling for sum of acceleration structure sizes built
  );

  // Compact the BLAS that have been built
  // Synchronization must be done by the application between the build and the compact
  VkResult cmdCompactBlas(VkCommandBuffer                            cmd,
                          std::span<AccelerationStructureBuildData>& blasBuildData,
                          std::span<nvvk::AccelerationStructure>&    blasAccel);

  // Destroy the original BLAS that was compacted
  void destroyNonCompactedBlas();


  // Return the statistics about the compacted BLAS
  Stats getStatistics() const { return m_stats; };

  // Scratch size strategy:
  // Loop over all BLAS to find the maximum and accumulated size.
  // - If the accumulated size is smaller or equal the budget, return it. This will allow to build all BLAS in one iteration.
  // - If the maximum size is greater than the budget, return it, so building the largest BLAS is guaranteed.
  // - Else return budget.
  //
  // Usage
  // - Call this function to get the optimal size needed for the scratch buffer
  // - User allocate the scratch buffer with the size returned by this function
  // - Provide size to cmdCreateBlas function.
  VkDeviceSize getScratchSize(VkDeviceSize hintMaxBudget, const std::span<nvvk::AccelerationStructureBuildData>& buildData) const;

  // Get the minimum offset alignment of the scratch buffer
  VkDeviceSize getScratchAlignment() const { return m_scratchAlignment; }

private:
  AccelerationStructureBuilder& operator=(const AccelerationStructureBuilder&) = default;

  void destroy();
  void destroyQueryPool();
  VkResult initializeQueryPoolIfNeeded(VkCommandBuffer cmd, const std::span<AccelerationStructureBuildData>& blasBuildData);
  VkResult cmdBuildAccelerationStructures(VkCommandBuffer cmd,
                                          VkDeviceSize&   budgetUsed,  // in/out: budget used so far
                                          std::span<AccelerationStructureBuildData>& blasBuildData,
                                          std::span<nvvk::AccelerationStructure>&    blasAccel,
                                          VkDeviceAddress                            scratchAddress,
                                          VkDeviceAddress                            scratchAddressEnd,
                                          VkDeviceSize                               hintMaxBudget,
                                          uint32_t&                                  currentQueryIdx,
                                          VkQueryPool                                queryPool);

  struct CompactBatchInfo
  {
    uint32_t    startIdx{};
    uint32_t    endIdx{};
    VkQueryPool queryPool{};
  };

  nvvk::ResourceAllocator*     m_alloc{};             // Allocator for the creation of acceleration structures
  VkDevice                     m_device{};            // Vulkan device
  VkQueryPool                  m_queryPool{};         // Query pool for BLAS compaction
  uint32_t                     m_currentBlasIdx{};    // Index to the current BLAS being processed
  uint32_t                     m_currentQueryIdx{};   // Index to the current query being processed
  uint32_t                     m_scratchAlignment{};  // Alignment of the scratch buffer
  std::queue<CompactBatchInfo> m_batches;             // Queue of compact batches to be processed

  std::vector<nvvk::AccelerationStructure> m_cleanupBlasAccel;  // List of BLAS to be cleaned up

  // Stats
  Stats m_stats;  // Statistics about the compacted BLAS
};

// Helper class for building both Bottom-Level Acceleration Structures (BLAS) and
// Top-Level Acceleration Structures (TLAS). This utility
// abstracts the complexity of acceleration structure generation while allowing
// compacting, updating, and managing buffers.
// For more advanced control one shall use nvvk::BlasBuilder and
// nvvk::AccelerationStructureBuildData.
class AccelerationStructureHelper
{
public:
  ~AccelerationStructureHelper() { assert(!m_transientPool && "deinit missing"); }

  void init(nvvk::ResourceAllocator* alloc,
            nvvk::StagingUploader*   uploader,
            nvvk::QueueInfo          queueInfo,
            VkDeviceSize             hintMaxAccelerationStructureSize = 512'000'000,
            VkDeviceSize             hintMaxScratchStructureSize      = 128'000'000);

  // Destroys all BLAS and TLAS resources, buffers, and clears internal state.
  // Must be called before rebuilding acceleration structures to avoid memory
  // leaks or double allocations.
  void deinit(void);

  // free both the TLAS and the BLAS set that were created using build methods,
  // some new builds and wait can be invoked afterward
  void deinitAccelerationStructures(void);

public:
  // BLAS related

  std::vector<nvvk::AccelerationStructureBuildData> blasBuildData;
  std::vector<nvvk::AccelerationStructure>          blasSet{};  // Bottom-level AS set
  AccelerationStructureBuilder::Stats               blasBuildStatistics;
  nvvk::Buffer                                      blasScratchBuffer{};

  // Builds a set of Bottom Level Acceleration Structures(BLAS) from a
  // Vector of geometry descriptors used for each BLAS
  // Same buildFlags will apply to each BLAS generated from asGeoInfoSet.
  void blasSubmitBuildAndWait(const std::vector<nvvk::AccelerationStructureGeometryInfo>& asGeoInfoSet,
                              VkBuildAccelerationStructureFlagsKHR                        buildFlags);

public:
  // TLAS related

  nvvk::AccelerationStructureBuildData tlasBuildData{};
  nvvk::AccelerationStructure          tlas{};  // Top-level AS
  nvvk::Buffer                         tlasInstancesBuffer{};
  nvvk::Buffer                         tlasScratchBuffer{};
  size_t                               tlasSize{0};

  // Builds the Top-Level Acceleration Structure (TLAS) from a list of instances
  // add VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR to buildFlags if you intend to use tlasUpdate
  void tlasSubmitBuildAndWait(const std::vector<VkAccelerationStructureInstanceKHR>& tlasInstances,
                              VkBuildAccelerationStructureFlagsKHR                   buildFlags);

  // Updates an existing TLAS with an updated list of instances.
  // If instance count differs from original, a rebuild is performed instead of an update.
  // TLAS must have been built with the VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR flag.
  void tlasSubmitUpdateAndWait(const std::vector<VkAccelerationStructureInstanceKHR>& tlasInstances);

private:
  nvvk::QueueInfo                                    m_queueInfo;
  nvvk::ResourceAllocator*                           m_alloc{nullptr};
  nvvk::StagingUploader*                             m_uploader{};
  VkPhysicalDeviceAccelerationStructurePropertiesKHR m_accelStructProps{};
  VkDeviceSize                                       m_blasAccelerationStructureBudget{};
  VkDeviceSize                                       m_blasScratchBudget{};
  VkCommandPool                                      m_transientPool{};
};

}  // namespace nvvk
