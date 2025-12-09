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


#include <assert.h>

#include <nvutils/alignment.hpp>
#include <nvvk/check_error.hpp>
#include <fmt/format.h>

#include "acceleration_structures.hpp"
#include "debug_util.hpp"
#include "commands.hpp"

namespace nvvk {

void AccelerationStructureBuildData::addGeometry(const VkAccelerationStructureGeometryKHR&       asGeom,
                                                 const VkAccelerationStructureBuildRangeInfoKHR& offset)
{
  asGeometry.push_back(asGeom);
  asBuildRangeInfo.push_back(offset);
}


void AccelerationStructureBuildData::addGeometry(const AccelerationStructureGeometryInfo& asGeom)
{
  asGeometry.push_back(asGeom.geometry);
  asBuildRangeInfo.push_back(asGeom.rangeInfo);
}


VkAccelerationStructureBuildSizesInfoKHR AccelerationStructureBuildData::finalizeGeometry(VkDevice device, VkBuildAccelerationStructureFlagsKHR flags)
{
  assert(asGeometry.size() > 0 && "No geometry added to Build Structure");
  assert(asType != VK_ACCELERATION_STRUCTURE_TYPE_MAX_ENUM_KHR && "Acceleration Structure Type not set");

  buildInfo.sType                     = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
  buildInfo.type                      = asType;
  buildInfo.flags                     = flags;
  buildInfo.mode                      = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
  buildInfo.srcAccelerationStructure  = VK_NULL_HANDLE;
  buildInfo.dstAccelerationStructure  = VK_NULL_HANDLE;
  buildInfo.geometryCount             = static_cast<uint32_t>(asGeometry.size());
  buildInfo.pGeometries               = asGeometry.data();
  buildInfo.ppGeometries              = nullptr;
  buildInfo.scratchData.deviceAddress = 0;

  std::vector<uint32_t> maxPrimCount(asBuildRangeInfo.size());
  for(size_t i = 0; i < asBuildRangeInfo.size(); ++i)
  {
    maxPrimCount[i] = asBuildRangeInfo[i].primitiveCount;
  }

  vkGetAccelerationStructureBuildSizesKHR(device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &buildInfo,
                                          maxPrimCount.data(), &sizeInfo);

  return sizeInfo;
}


VkAccelerationStructureCreateInfoKHR AccelerationStructureBuildData::makeCreateInfo() const
{
  assert(asType != VK_ACCELERATION_STRUCTURE_TYPE_MAX_ENUM_KHR && "Acceleration Structure Type not set");
  assert(sizeInfo.accelerationStructureSize > 0 && "Acceleration Structure Size not set");

  VkAccelerationStructureCreateInfoKHR createInfo{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR};
  createInfo.type = asType;
  createInfo.size = sizeInfo.accelerationStructureSize;

  return createInfo;
}


AccelerationStructureGeometryInfo AccelerationStructureBuildData::makeInstanceGeometry(size_t numInstances, VkDeviceAddress instanceBufferAddr)
{
  assert(asType == VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR && "Instance geometry can only be used with TLAS");

  // Describes instance data in the acceleration structure.
  VkAccelerationStructureGeometryInstancesDataKHR geometryInstances{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR};
  geometryInstances.data.deviceAddress = instanceBufferAddr;

  // Set up the geometry to use instance data.
  VkAccelerationStructureGeometryKHR geometry{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
  geometry.geometryType       = VK_GEOMETRY_TYPE_INSTANCES_KHR;
  geometry.geometry.instances = geometryInstances;

  // Specifies the number of primitives (instances in this case).
  VkAccelerationStructureBuildRangeInfoKHR rangeInfo{};
  rangeInfo.primitiveCount = static_cast<uint32_t>(numInstances);

  // Prepare and return geometry information.
  AccelerationStructureGeometryInfo result;
  result.geometry  = geometry;
  result.rangeInfo = rangeInfo;

  return result;
}


void AccelerationStructureBuildData::cmdBuildAccelerationStructure(VkCommandBuffer            cmd,
                                                                   VkAccelerationStructureKHR accelerationStructure,
                                                                   VkDeviceAddress            scratchAddress)
{
  assert(asGeometry.size() == asBuildRangeInfo.size() && "asGeometry.size() != asBuildRangeInfo.size()");
  assert(accelerationStructure != VK_NULL_HANDLE && "Acceleration Structure not created, first call createAccelerationStructure");

  const VkAccelerationStructureBuildRangeInfoKHR* rangeInfo = asBuildRangeInfo.data();

  // Build the acceleration structure
  buildInfo.mode                      = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
  buildInfo.srcAccelerationStructure  = VK_NULL_HANDLE;
  buildInfo.dstAccelerationStructure  = accelerationStructure;
  buildInfo.scratchData.deviceAddress = scratchAddress;
  buildInfo.pGeometries   = asGeometry.data();  // In case the structure was copied, we need to update the pointer
  buildInfo.geometryCount = static_cast<uint32_t>(asGeometry.size());

  vkCmdBuildAccelerationStructuresKHR(cmd, 1, &buildInfo, &rangeInfo);

  // Since the scratch buffer is reused across builds, we need a barrier to ensure one build
  // is finished before starting the next one.
  accelerationStructureBarrier(cmd, VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR, VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR);
}


void AccelerationStructureBuildData::cmdUpdateAccelerationStructure(VkCommandBuffer            cmd,
                                                                    VkAccelerationStructureKHR accelerationStructure,
                                                                    VkDeviceAddress            scratchAddress)
{
  assert(asGeometry.size() == asBuildRangeInfo.size() && "asGeometry.size() != asBuildRangeInfo.size()");
  assert(accelerationStructure != VK_NULL_HANDLE && "Acceleration Structure not created, first call createAccelerationStructure");

  const VkAccelerationStructureBuildRangeInfoKHR* rangeInfo = asBuildRangeInfo.data();

  // Build the acceleration structure
  buildInfo.mode                      = VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR;
  buildInfo.srcAccelerationStructure  = accelerationStructure;
  buildInfo.dstAccelerationStructure  = accelerationStructure;
  buildInfo.scratchData.deviceAddress = scratchAddress;
  buildInfo.pGeometries               = asGeometry.data();
  vkCmdBuildAccelerationStructuresKHR(cmd, 1, &buildInfo, &rangeInfo);

  // Since the scratch buffer is reused across builds, we need a barrier to ensure one build
  // is finished before starting the next one.
  accelerationStructureBarrier(cmd, VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR, VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR);
}

//////////////////////////////////////////////////////////////////////////

void AccelerationStructureBuilder::init(ResourceAllocator* allocator)
{
  m_alloc           = allocator;
  m_device          = allocator->getDevice();
  m_currentBlasIdx  = 0;
  m_currentQueryIdx = 0;

  VkPhysicalDeviceProperties2                        props = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
  VkPhysicalDeviceAccelerationStructurePropertiesKHR rayProps = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_PROPERTIES_KHR};
  props.pNext = &rayProps;

  vkGetPhysicalDeviceProperties2(allocator->getPhysicalDevice(), &props);

  m_scratchAlignment = rayProps.minAccelerationStructureScratchOffsetAlignment;
}

void AccelerationStructureBuilder::deinit()
{
  if(m_device != VK_NULL_HANDLE)
  {
    destroy();
  }

  *this = {};
}


VkResult AccelerationStructureBuilder::cmdCreateBlas(VkCommandBuffer                            cmd,
                                                     std::span<AccelerationStructureBuildData>& blasBuildData,
                                                     std::span<AccelerationStructure>&          blasAccel,
                                                     VkDeviceAddress scratchAddress,  //  Address of the scratch buffer
                                                     VkDeviceSize    scratchSize,     //  Size of the scratch buffer
                                                     VkDeviceSize    hintMaxBudget)
{
  // Create a new query pool for this batch
  NVVK_FAIL_RETURN(initializeQueryPoolIfNeeded(cmd, blasBuildData));

  // Track the starting BLAS index for this batch
  uint32_t batchStartIdx   = m_currentBlasIdx;
  uint32_t currentQueryIdx = batchStartIdx;  // Local query index for this batch

  VkDeviceSize scratchAddressEnd = scratchAddress + scratchSize;  // The end address of the scratch buffer.
  VkDeviceSize budgetUsed        = 0;  // Initialize the total budget used in this function call

  // Process each BLAS in the data vector while staying under the memory budget.
  while(m_currentBlasIdx < blasBuildData.size() && budgetUsed < hintMaxBudget)
  {
    // Build acceleration structures and accumulate the total memory used.
    NVVK_FAIL_RETURN(cmdBuildAccelerationStructures(cmd, budgetUsed, blasBuildData, blasAccel, scratchAddress,
                                                    scratchAddressEnd, hintMaxBudget, currentQueryIdx, m_queryPool));
  }

  // Store batch information for compaction
  if(m_currentBlasIdx > batchStartIdx && m_queryPool != VK_NULL_HANDLE)
  {
    CompactBatchInfo batch;
    batch.startIdx  = batchStartIdx;
    batch.endIdx    = m_currentBlasIdx;
    batch.queryPool = m_queryPool;
    m_batches.push(batch);
  }

  // Check if all BLAS have been built.
  VkResult status = (m_currentBlasIdx < blasBuildData.size()) ? VK_INCOMPLETE : VK_SUCCESS;

  return status;
}


// Initializes a query pool for recording acceleration structure properties if necessary.
// This function ensures a query pool is available if any BLAS in the build data is flagged for compaction.
VkResult AccelerationStructureBuilder::initializeQueryPoolIfNeeded(VkCommandBuffer cmd,
                                                                   const std::span<AccelerationStructureBuildData>& blasBuildData)
{
  // Check if any BLAS in this potential batch needs compaction
  bool needsCompaction = false;
  for(size_t i = m_currentBlasIdx; i < blasBuildData.size(); i++)
  {
    if(blasBuildData[i].hasCompactFlag())
    {
      needsCompaction = true;
      break;
    }
  }

  if(needsCompaction)
  {
    // Estimate max number of BLAS in this batch
    // Calculate how many BLAS we can potentially build in this batch based on budget
    uint32_t maxBatchSize = static_cast<uint32_t>(blasBuildData.size() - m_currentBlasIdx);

    // Create a query pool for this batch
    if(m_queryPool == VK_NULL_HANDLE)
    {
      VkQueryPoolCreateInfo qpci = {VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
      qpci.queryType             = VK_QUERY_TYPE_ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR;
      qpci.queryCount            = maxBatchSize;
      NVVK_FAIL_RETURN(vkCreateQueryPool(m_device, &qpci, nullptr, &m_queryPool));
      NVVK_DBG_NAME(m_queryPool);
    }

    // Reset the newly created query pool
    vkCmdResetQueryPool(cmd, m_queryPool, m_currentBlasIdx, maxBatchSize);
  }

  return VK_SUCCESS;
}


// Builds multiple Bottom-Level Acceleration Structures (BLAS) for a Vulkan ray tracing pipeline.
// This function manages memory budgets and submits the necessary commands to the specified command buffer.
//
// Parameters:
//   cmd               - Command buffer where acceleration structure commands are recorded.
//   budgetUsed        - Running value of the budget used.
//   blasBuildData     - Vector of data structures containing the geometry and other build-related information for each BLAS.
//   blasAccel         - Vector where the function will store the created acceleration structures.
//   scratchAddress    - The starting scratchAddress
//   scratchAddressEnd - The end address for the available scratch space
//   hintMaxBudget     - A hint for the maximum budget allowed for building acceleration structures.
//   currentQueryIdx   - Reference to the current index for queries, updated during execution.
//
// Returns:
//   The total device size used for building the acceleration structures during this function call.
VkResult AccelerationStructureBuilder::cmdBuildAccelerationStructures(VkCommandBuffer cmd,
                                                                      VkDeviceSize&   budgetUsed,
                                                                      std::span<AccelerationStructureBuildData>& blasBuildData,
                                                                      std::span<AccelerationStructure>& blasAccel,
                                                                      VkDeviceAddress                   scratchAddress,
                                                                      VkDeviceAddress scratchAddressEnd,
                                                                      VkDeviceSize    hintMaxBudget,
                                                                      uint32_t&       currentQueryIdx,
                                                                      VkQueryPool     queryPool)
{
  // Temporary vectors for storing build-related data
  std::vector<VkAccelerationStructureBuildGeometryInfoKHR> collectedBuildInfo;
  std::vector<VkAccelerationStructureKHR>                  collectedAccel;
  std::vector<VkAccelerationStructureBuildRangeInfoKHR*>   collectedRangeInfo;

  // Pre-allocate memory based on the number of BLAS to be built
  collectedBuildInfo.reserve(blasBuildData.size());
  collectedAccel.reserve(blasBuildData.size());
  collectedRangeInfo.reserve(blasBuildData.size());

  // Loop through BLAS data while there is scratch address space and budget available
  while(scratchAddress < scratchAddressEnd && budgetUsed < hintMaxBudget && m_currentBlasIdx < blasBuildData.size())
  {
    auto&                                data       = blasBuildData[m_currentBlasIdx];
    VkAccelerationStructureCreateInfoKHR createInfo = data.makeCreateInfo();

    if(scratchAddress + data.sizeInfo.buildScratchSize > scratchAddressEnd)
      break;

    // Create and store acceleration structure
    NVVK_FAIL_RETURN(m_alloc->createAcceleration(blasAccel[m_currentBlasIdx], createInfo));
    NVVK_DBG_NAME(blasAccel[m_currentBlasIdx].accel);
    collectedAccel.push_back(blasAccel[m_currentBlasIdx].accel);

    // Setup build information for the current BLAS
    data.buildInfo.mode                      = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    data.buildInfo.srcAccelerationStructure  = VK_NULL_HANDLE;
    data.buildInfo.dstAccelerationStructure  = blasAccel[m_currentBlasIdx].accel;
    data.buildInfo.scratchData.deviceAddress = scratchAddress;
    data.buildInfo.pGeometries               = data.asGeometry.data();
    collectedBuildInfo.push_back(data.buildInfo);
    collectedRangeInfo.push_back(data.asBuildRangeInfo.data());

    // Update the used budget with the size of the current structure
    budgetUsed += data.sizeInfo.accelerationStructureSize;

    // Update scratch address, keep it aligned
    VkDeviceSize scratchSize = nvutils::align_up(data.sizeInfo.buildScratchSize, m_scratchAlignment);
    scratchAddress += scratchSize;

    m_currentBlasIdx++;
  }

  // Command to build the acceleration structures on the GPU
  vkCmdBuildAccelerationStructuresKHR(cmd, static_cast<uint32_t>(collectedBuildInfo.size()), collectedBuildInfo.data(),
                                      collectedRangeInfo.data());

  // Barrier to ensure proper synchronization after building
  accelerationStructureBarrier(cmd, VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
                               VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR | VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR);

  // If a query pool is available, record the properties of the built acceleration structures
  if(queryPool != VK_NULL_HANDLE)
  {
    uint32_t numQueries = static_cast<uint32_t>(collectedAccel.size());
    vkCmdWriteAccelerationStructuresPropertiesKHR(cmd, numQueries, collectedAccel.data(),
                                                  VK_QUERY_TYPE_ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR, queryPool,
                                                  currentQueryIdx);
    currentQueryIdx += numQueries;
  }

  // Return the total budget used in this operation
  return VK_SUCCESS;
}


// Compacts the Bottom-Level Acceleration Structures (BLAS) that have been built, reducing their memory footprint.
// This function uses the results from previously performed queries to determine the compacted sizes and then
// creates new, smaller acceleration structures. It also handles copying from the original to the compacted structures.
//
// Notes:
//   It assumes that a query has been performed earlier to determine the possible compacted sizes of the acceleration structures.
//   This function may need to be called multiple times to compact all BLAS in the batch. Check for the return value VK_INCOMPLETE or VK_SUCCESS.
//
VkResult AccelerationStructureBuilder::cmdCompactBlas(VkCommandBuffer                            cmd,
                                                      std::span<AccelerationStructureBuildData>& blasBuildData,
                                                      std::span<AccelerationStructure>&          blasAccel)
{
  if(m_batches.empty())
  {
    return VK_SUCCESS;
  }

  // Process the first batch of BLAS for compaction
  auto& batch = m_batches.front();
  m_batches.pop();
  {
    if(batch.queryPool == VK_NULL_HANDLE)
    {
      return m_batches.empty() ? VK_SUCCESS : VK_INCOMPLETE;  // Skip batches that don't need compaction
    }

    uint32_t                  batchSize = batch.endIdx - batch.startIdx;
    std::vector<VkDeviceSize> compactSizes(batchSize);

    // Get query results for this batch
    VkResult result = vkGetQueryPoolResults(m_device, batch.queryPool, batch.startIdx, batchSize,
                                            batchSize * sizeof(VkDeviceSize), compactSizes.data(), sizeof(VkDeviceSize),
                                            VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);

    if(result != VK_SUCCESS)
    {
      // If we can't get the results, skip this batch
      return m_batches.empty() ? VK_SUCCESS : VK_INCOMPLETE;
    }

    // Process compaction for this batch
    for(uint32_t i = 0; i < batchSize; i++)
    {
      uint32_t blasIdx = batch.startIdx + i;
      if(blasIdx >= blasBuildData.size())
      {
        // We've processed all available BLAS
        break;
      }

      VkDeviceSize compactSize = compactSizes[i];

      if(compactSize > 0)
      {
        // Update statistical tracking of sizes before and after compaction.
        m_stats.totalCompactSize += compactSize;
        m_stats.totalOriginalSize += blasBuildData[blasIdx].sizeInfo.accelerationStructureSize;
        blasBuildData[blasIdx].sizeInfo.accelerationStructureSize = compactSize;
        m_cleanupBlasAccel.push_back(blasAccel[blasIdx]);  // Schedule old BLAS for cleanup.

        // Create a new acceleration structure for the compacted BLAS.
        VkAccelerationStructureCreateInfoKHR asCreateInfo{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR};
        asCreateInfo.size = compactSize;
        asCreateInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
        NVVK_FAIL_RETURN(m_alloc->createAcceleration(blasAccel[blasIdx], asCreateInfo));
        NVVK_DBG_NAME(blasAccel[blasIdx].accel);

        // Command to copy the original BLAS to the newly created compacted version.
        VkCopyAccelerationStructureInfoKHR copyInfo{VK_STRUCTURE_TYPE_COPY_ACCELERATION_STRUCTURE_INFO_KHR};
        copyInfo.src  = blasBuildData[blasIdx].buildInfo.dstAccelerationStructure;
        copyInfo.dst  = blasAccel[blasIdx].accel;
        copyInfo.mode = VK_COPY_ACCELERATION_STRUCTURE_MODE_COMPACT_KHR;
        vkCmdCopyAccelerationStructureKHR(cmd, &copyInfo);

        // Update the build data to reflect the new destination of the BLAS.
        blasBuildData[blasIdx].buildInfo.dstAccelerationStructure = blasAccel[blasIdx].accel;
      }
    }
  }

  return m_batches.empty() ? VK_SUCCESS : VK_INCOMPLETE;
}


void AccelerationStructureBuilder::destroyNonCompactedBlas()
{
  for(auto& blas : m_cleanupBlasAccel)
  {
    m_alloc->destroyAcceleration(blas);
  }
  m_cleanupBlasAccel.clear();
}

void AccelerationStructureBuilder::destroyQueryPool()
{
  // Clean up query pools
  vkDestroyQueryPool(m_device, m_queryPool, nullptr);
  m_queryPool = {};
  m_batches   = {};
}

void AccelerationStructureBuilder::destroy()
{
  destroyQueryPool();
  destroyNonCompactedBlas();
}

// Find a scratch size that is within the budget, or at least as big as the largest required scratch.
VkDeviceSize AccelerationStructureBuilder::getScratchSize(VkDeviceSize hintMaxBudget,
                                                          const std::span<AccelerationStructureBuildData>& buildData) const
{
  VkDeviceSize maxScratch{0};
  VkDeviceSize totalScratch{0};

  for(auto& buildInfo : buildData)
  {
    VkDeviceSize alignedSize = nvutils::align_up(buildInfo.sizeInfo.buildScratchSize, m_scratchAlignment);
    maxScratch               = std::max(maxScratch, alignedSize);
    totalScratch += alignedSize;
  }


  if(totalScratch <= hintMaxBudget)
  {
    // can be smaller than hintMaxBudget if all fit
    return totalScratch;
  }
  else
  {
    // must at least be maximum required, otherwise exhaust budget
    return std::max(maxScratch, hintMaxBudget);
  }
}

// Generates a formatted string summarizing the statistics of BLAS compaction results.
// The output includes the original and compacted sizes in megabytes (MB), the amount of memory saved,
// and the percentage reduction in size. This method is intended to provide a quick, human-readable
// summary of the compaction efficiency.
//
// Returns:
//   A string containing the formatted summary of the BLAS compaction statistics.
std::string AccelerationStructureBuilder::Stats::toString() const
{
  const VkDeviceSize savedSize = totalOriginalSize - totalCompactSize;
  const float fractionSmaller  = (totalOriginalSize == 0) ? 0.0f : savedSize / static_cast<float>(totalOriginalSize);

  const std::string output = fmt::format("BLAS Compaction: {} bytes -> {} bytes ({} bytes saved, {:.2f}% smaller)",
                                         totalOriginalSize, totalCompactSize, savedSize, fractionSmaller * 100.0f);

  return output;
}

//////////////////////////////////////////////////////////////////////////

// Returns the maximum scratch buffer size needed for building all provided acceleration structures.
// This function iterates through a vector of AccelerationStructureBuildData, comparing the scratch
// size required for each structure and returns the largest value found.
//
// Returns:
//   The maximum scratch size needed as a VkDeviceSize.
VkDeviceSize getMaxScratchSize(const std::vector<AccelerationStructureBuildData>& asBuildData)
{
  VkDeviceSize maxScratchSize = 0;
  for(const auto& blas : asBuildData)
  {
    maxScratchSize = std::max(maxScratchSize, blas.sizeInfo.buildScratchSize);
  }
  return maxScratchSize;
}


void AccelerationStructureHelper::init(ResourceAllocator* alloc,
                                       StagingUploader*   uploader,
                                       QueueInfo          queueInfo,
                                       VkDeviceSize       hintMaxAccelerationStructureSize /*= 512'000'000*/,
                                       VkDeviceSize       hintMaxScratchStructureSize /*= 128'000'000*/)
{
  assert(!m_transientPool && "init() called multiple times");

  m_queueInfo                       = queueInfo;
  m_alloc                           = alloc;
  m_uploader                        = uploader;
  m_blasAccelerationStructureBudget = hintMaxAccelerationStructureSize;
  m_blasScratchBudget               = hintMaxScratchStructureSize;
  m_transientPool                   = createTransientCommandPool(alloc->getDevice(), queueInfo.familyIndex);

  m_accelStructProps                 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_PROPERTIES_KHR};
  VkPhysicalDeviceProperties2 props2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
  props2.pNext                       = &m_accelStructProps;
  vkGetPhysicalDeviceProperties2(alloc->getPhysicalDevice(), &props2);
}

void AccelerationStructureHelper::deinit(void)
{
  if(m_transientPool)
    vkDestroyCommandPool(m_alloc->getDevice(), m_transientPool, nullptr);
  m_transientPool    = VK_NULL_HANDLE;
  m_alloc            = nullptr;
  m_queueInfo        = {};
  m_uploader         = nullptr;
  m_accelStructProps = {};
}

void AccelerationStructureHelper::deinitAccelerationStructures(void)
{
  // BLAS related
  for(auto& b : blasSet)
  {
    if(b.accel)
      m_alloc->destroyAcceleration(b);
  }
  blasSet.clear();
  blasBuildData.clear();
  blasBuildStatistics = {};
  if(blasScratchBuffer.buffer)
    m_alloc->destroyBuffer(blasScratchBuffer);
  blasScratchBuffer = {};

  // TLAS related5
  if(tlas.accel)
    m_alloc->destroyAcceleration(tlas);
  if(tlasInstancesBuffer.buffer)
    m_alloc->destroyBuffer(tlasInstancesBuffer);
  if(tlasScratchBuffer.buffer)
    m_alloc->destroyBuffer(tlasScratchBuffer);

  tlas                = {};
  tlasInstancesBuffer = {};
  tlasScratchBuffer   = {};
  tlasBuildData       = {};
  tlasSize            = {};
}

void AccelerationStructureHelper::blasSubmitBuildAndWait(const std::vector<AccelerationStructureGeometryInfo>& asGeoInfoSet,
                                                         VkBuildAccelerationStructureFlagsKHR buildFlags)
{
  VkDevice device = m_alloc->getDevice();

  assert(blasSet.empty() && "we must not invoke build if already built. use deinit before.");

  // Prepare the BLAS build data
  blasBuildData.reserve(asGeoInfoSet.size());

  for(const auto& asGeoInfo : asGeoInfoSet)
  {
    AccelerationStructureBuildData buildData{VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR};

    buildData.addGeometry(asGeoInfo);

    buildData.finalizeGeometry(m_alloc->getDevice(), buildFlags);

    blasBuildData.emplace_back(buildData);
  }

  // build the set of BLAS
  blasSet.resize(blasBuildData.size());

  // Find the most optimal size for our scratch buffer, and get the addresses of the scratch buffers
  // to allow a maximum of BLAS to be built in parallel, within the budget
  AccelerationStructureBuilder blasBuilder;
  blasBuilder.init(m_alloc);
  VkDeviceSize hintScratchBudget = m_blasScratchBudget;  // Limiting the size of the scratch buffer to 2MB
  VkDeviceSize scratchSize       = blasBuilder.getScratchSize(hintScratchBudget, blasBuildData);

  NVVK_CHECK(m_alloc->createBuffer(blasScratchBuffer, scratchSize,
                                   VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT
                                       | VK_BUFFER_USAGE_2_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR,
                                   VMA_MEMORY_USAGE_AUTO, {}, m_accelStructProps.minAccelerationStructureScratchOffsetAlignment));

  // Start the build and compaction of the BLAS
  VkDeviceSize hintBuildBudget = m_blasAccelerationStructureBudget;  // Limiting the size of the scratch buffer to 2MB
  bool         finished        = false;

  std::span<AccelerationStructureBuildData> buildDataSpan = blasBuildData;
  std::span<AccelerationStructure>          blasSpan      = blasSet;

  do
  {
    {
      VkCommandBuffer cmd = createSingleTimeCommands(device, m_transientPool);

      VkResult result = blasBuilder.cmdCreateBlas(cmd, buildDataSpan, blasSpan, blasScratchBuffer.address,
                                                  blasScratchBuffer.bufferSize, hintBuildBudget);
      if(result == VK_SUCCESS)
      {
        finished = true;
      }
      else if(result != VK_INCOMPLETE)
      {
        // Any result other than VK_SUCCESS or VK_INCOMPLETE is an error
        assert(0 && "Error building BLAS");
      }
      NVVK_CHECK(endSingleTimeCommands(cmd, device, m_transientPool, m_queueInfo.queue));
    }
    // compact BLAS if needed
    if(buildFlags & VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_COMPACTION_BIT_KHR)
    {
      // Compacting the BLAS, and destroy the previous ones
      VkCommandBuffer cmd = createSingleTimeCommands(device, m_transientPool);
      blasBuilder.cmdCompactBlas(cmd, buildDataSpan, blasSpan);
      NVVK_CHECK(endSingleTimeCommands(cmd, device, m_transientPool, m_queueInfo.queue));
      blasBuilder.destroyNonCompactedBlas();
    }
  } while(!finished);

  blasBuildStatistics = blasBuilder.getStatistics();

  // Giving a name to the BLAS
  for(size_t i = 0; i < blasSet.size(); i++)
  {
    NVVK_DBG_NAME(blasSet[i].accel);
  }

  // Cleanup
  blasBuilder.deinit();
}

void AccelerationStructureHelper::tlasSubmitBuildAndWait(const std::vector<VkAccelerationStructureInstanceKHR>& tlasInstances,
                                                         VkBuildAccelerationStructureFlagsKHR buildFlags)
{
  VkDevice device = m_alloc->getDevice();

  // we must not invoke build if already built. use update.
  assert((tlasInstancesBuffer.buffer == VK_NULL_HANDLE)
         && "Do not invoke build if already built. build with VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR, then use tlasUpdate");

  VkCommandBuffer cmd = createSingleTimeCommands(device, m_transientPool);

  // Create the buffer of instances.
  // Instance buffer device addresses must be aligned to 16 bytes according to
  // https://vulkan.lunarg.com/doc/view/1.4.328.1/windows/antora/spec/latest/chapters/accelstructures.html#VUID-vkCmdBuildAccelerationStructuresKHR-pInfos-03717 .
  constexpr VmaAllocationCreateFlags instanceAllocFlags   = 0;
  constexpr VkDeviceSize             instanceMinAlignment = 16;
  NVVK_CHECK(m_alloc->createBuffer(
      tlasInstancesBuffer, std::span<VkAccelerationStructureInstanceKHR const>(tlasInstances).size_bytes(),
      VK_BUFFER_USAGE_2_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR | VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT,
      VMA_MEMORY_USAGE_AUTO, instanceAllocFlags, instanceAllocFlags));
  NVVK_CHECK(m_uploader->appendBuffer(tlasInstancesBuffer, 0, std::span<VkAccelerationStructureInstanceKHR const>(tlasInstances)));
  NVVK_DBG_NAME(tlasInstancesBuffer.buffer);
  m_uploader->cmdUploadAppended(cmd);
  // Barrier to ensure transfer write completes before acceleration structure build.
  // VK_ACCESS_2_SHADER_READ_BIT is required because the acceleration structure build
  // operation reads the instance data from the buffer (not just writes to the AS).
  // Without this flag, validation layers report READ_AFTER_WRITE hazards.
  accelerationStructureBarrier(cmd, VK_ACCESS_TRANSFER_WRITE_BIT,
                               VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR | VK_ACCESS_2_SHADER_READ_BIT);


  tlasBuildData = {VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR};
  AccelerationStructureGeometryInfo geometryInfo =
      tlasBuildData.makeInstanceGeometry(tlasInstances.size(), tlasInstancesBuffer.address);
  tlasBuildData.addGeometry(geometryInfo);

  // Get the size of the TLAS
  auto sizeInfo = tlasBuildData.finalizeGeometry(m_alloc->getDevice(), buildFlags);

  // Create the scratch buffer
  NVVK_CHECK(m_alloc->createBuffer(tlasScratchBuffer, sizeInfo.buildScratchSize,
                                   VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT
                                       | VK_BUFFER_USAGE_2_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR,
                                   VMA_MEMORY_USAGE_AUTO, {}, m_accelStructProps.minAccelerationStructureScratchOffsetAlignment));
  NVVK_DBG_NAME(tlasScratchBuffer.buffer);

  // Create the TLAS
  NVVK_CHECK(m_alloc->createAcceleration(tlas, tlasBuildData.makeCreateInfo()));
  NVVK_DBG_NAME(tlas.accel);
  tlasBuildData.cmdBuildAccelerationStructure(cmd, tlas.accel, tlasScratchBuffer.address);

  tlasSize = tlasInstances.size();

  NVVK_CHECK(endSingleTimeCommands(cmd, device, m_transientPool, m_queueInfo.queue));

  m_uploader->releaseStaging();
}

void AccelerationStructureHelper::tlasSubmitUpdateAndWait(const std::vector<VkAccelerationStructureInstanceKHR>& tlasInstances)
{
  VkDevice device = m_alloc->getDevice();

  bool sizeChanged = (tlasInstances.size() != tlasSize);

  VkCommandBuffer cmd = createSingleTimeCommands(device, m_transientPool);

  // Update the instance buffer
  m_uploader->appendBuffer(tlasInstancesBuffer, 0, std::span(tlasInstances));
  m_uploader->cmdUploadAppended(cmd);

  // Make sure the copy of the instance buffer are copied before triggering the acceleration structure build
  accelerationStructureBarrier(cmd, VK_ACCESS_TRANSFER_WRITE_BIT,
                               VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR | VK_ACCESS_2_SHADER_READ_BIT);

  if(tlasScratchBuffer.buffer == VK_NULL_HANDLE)
  {
    NVVK_CHECK(m_alloc->createBuffer(tlasScratchBuffer, tlasBuildData.sizeInfo.buildScratchSize,
                                     VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT,
                                     VMA_MEMORY_USAGE_AUTO, {}, m_accelStructProps.minAccelerationStructureScratchOffsetAlignment));
    NVVK_DBG_NAME(tlasScratchBuffer.buffer);
  }

  // Building or updating the top-level acceleration structure
  if(sizeChanged)
  {
    tlasBuildData.cmdBuildAccelerationStructure(cmd, tlas.accel, tlasScratchBuffer.address);
    tlasSize = tlasInstances.size();
  }
  else
  {
    tlasBuildData.cmdUpdateAccelerationStructure(cmd, tlas.accel, tlasScratchBuffer.address);
  }

  // Make sure to have the TLAS ready before using it
  NVVK_CHECK(endSingleTimeCommands(cmd, device, m_transientPool, m_queueInfo.queue));

  m_uploader->releaseStaging();
}

}  // namespace nvvk
