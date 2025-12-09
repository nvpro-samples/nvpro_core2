/*
* Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
* SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION
* SPDX-License-Identifier: Apache-2.0
*/
#pragma once

#include <cassert>
#include <span>
#include <vector>

#include <volk.h>

#include "resources.hpp"

namespace nvvk {

// Descriptor Bindings
// Helps you build descriptor set layouts by storing information about each
// binding's type, number of descriptors, stages, and other properties.
//
// Usage:
//   see usage_DescriptorBindings in descriptors.cpp
class DescriptorBindings
{
public:
  // Adds a binding at the given `binding` index for `descriptorCount`
  // descriptors of type `descriptorType`. The resources pointed to may be
  // accessed via the given stages.
  //
  // `pImmutableSamplers` can be set to an array of `descriptorCount` samplers
  // to permanently bind them to the set layout; see
  // `VkDescriptorSetLayoutBinding::pImmutableSamplers` for more info.
  //
  // `bindingFlags` will be passed to `VkDescriptorSetLayoutBindingFlagsCreateInfo`;
  // it can be used, for instance, to let a descriptor be updated after it's bound.
  void addBinding(uint32_t                 binding,
                  VkDescriptorType         descriptorType,
                  uint32_t                 descriptorCount,
                  VkShaderStageFlags       stageFlags,
                  const VkSampler*         pImmutableSamplers = nullptr,
                  VkDescriptorBindingFlags bindingFlags       = 0);

  void addBinding(const VkDescriptorSetLayoutBinding& layoutBinding, VkDescriptorBindingFlags bindingFlags = 0);
  void addBindings(std::span<const VkDescriptorSetLayoutBinding> layoutBindings, VkDescriptorBindingFlags bindingFlags = 0);
  void addBindings(std::initializer_list<const VkDescriptorSetLayoutBinding> layoutBindings,
                   VkDescriptorBindingFlags                                  bindingFlags = 0);

  // Fills a `VkWriteDescriptorSet` struct for `descriptorCounts` descriptors,
  // starting at element `dstArrayElement`.
  //
  // If `dstArrayElement == ~0`, then the `descriptorCount` will be set to the
  // original binding's count and `dstArrayElement` to 0 -- i.e. it'll span the
  // entire binding.
  //
  // Note: the returned object is not ready to be used, as it doesn't contain
  // the information of the actual resources. You'll want to fill the image,
  // buffer, or texel buffer view info, or pass this to
  // `WriteSetContainer::append()`.
  //
  // If no entry exists for the given `binding`, returns a `VkWriteDescriptorSet`
  // with .descriptorType set to `VK_DESCRIPTOR_TYPE_MAX_ENUM`.
  VkWriteDescriptorSet getWriteSet(uint32_t        binding,
                                   VkDescriptorSet dstSet          = VK_NULL_HANDLE,
                                   uint32_t        dstArrayElement = ~0,
                                   uint32_t        descriptorCount = 1) const;

  void clear()
  {
    m_bindings.clear();
    m_bindingFlags.clear();
  }

  // Once the bindings have been added, this generates the descriptor layout corresponding to the
  // bound resources.
  VkResult createDescriptorSetLayout(VkDevice device, VkDescriptorSetLayoutCreateFlags flags, VkDescriptorSetLayout* pDescriptorSetLayout) const;

  // Appends the required pool sizes for `numSets` many sets.
  // If `totalVariableCount` is non zero, it will be used to override the total requirement for the variable binding.
  // Otherwise the regular `binding.descriptorCount * numSets` is used.
  void appendPoolSizes(std::vector<VkDescriptorPoolSize>& poolSizes, uint32_t numSets = 1, uint32_t totalVariableCount = 0) const;

  // Returns the required pool sizes for `numSets` many sets.
  // If `totalVariableCount` is non zero, it will be used to override the total requirement for the variable binding.
  // Otherwise the regular `binding.descriptorCount * numSets` is used.
  std::vector<VkDescriptorPoolSize> calculatePoolSizes(uint32_t numSets = 1, uint32_t totalVariableCount = 0) const;

  // Returns the bindings that were added
  const std::vector<VkDescriptorSetLayoutBinding>& getBindings() const { return m_bindings; }

private:
  std::vector<VkDescriptorSetLayoutBinding> m_bindings;
  std::vector<VkDescriptorBindingFlags>     m_bindingFlags;
  // Map from `VkDescriptorSetLayoutBinding::binding` to an index in the above arrays.
  // Vulkan recommends using as compact a maximum binding number as possible, so a linear array should be OK.
  std::vector<uint32_t> m_bindingToIndex;
};


//////////////////////////////////////////////////////////////////////////

// Helper container for the most common descriptor set use case -- bindings
// used to create a single layout and `numSets` descriptor sets allocated using
// that layout.
// It manages its own pool storage; all descriptor sets can be freed at once
// by resetting the pool.
//
// Usage:
//   see usage_DescriptorBindings() in descriptors.cpp
class DescriptorPack
{
public:
  DescriptorPack() = default;
  DescriptorPack(DescriptorPack&& other) noexcept;
  DescriptorPack& operator=(DescriptorPack&& other) noexcept;

  ~DescriptorPack();


  // Call this function to initialize `layout`, `pool`, and `sets`.
  //
  // If `numSets` is 0, this only creates the layout.
  //
  // If `totalVariableCount` is non zero, it will be used to override the total requirement for the variable binding,
  // and `descriptorVariableCounts` must be non null and the length of `numSets`.
  // Otherwise the regular `binding.descriptorCount * numSets` is used.
  VkResult init(const DescriptorBindings&        bindings,
                VkDevice                         device,
                uint32_t                         numSets                  = 1,
                VkDescriptorSetLayoutCreateFlags layoutFlags              = 0,
                VkDescriptorPoolCreateFlags      poolFlags                = 0,
                uint32_t                         totalVariableCount       = 0,
                const uint32_t*                  descriptorVariableCounts = nullptr);
  void     deinit();

  VkDescriptorSetLayout               getLayout() const { return m_layout; }
  const VkDescriptorSetLayout*        getLayoutPtr() const { return &m_layout; }
  VkDescriptorPool                    getPool() const { return m_pool; };
  const std::vector<VkDescriptorSet>& getSets() const { return m_sets; }
  VkDescriptorSet                     getSet(uint32_t setIndex) const { return m_sets[setIndex]; }
  const VkDescriptorSet*              getSetPtr(uint32_t setIndex = 0) const { return &m_sets[setIndex]; }

  // Wrapper to get a `VkWriteDescriptorSet` for a descriptor set stored in `sets` if it's not empty.
  // Empty `sets` usage is legal in the push descriptor use-case.
  // see `DescriptorBindings::getWriteSet` for more details
  VkWriteDescriptorSet makeWrite(uint32_t binding, uint32_t setIndex = 0, uint32_t dstArrayElement = ~0, uint32_t descriptorCount = 1) const
  {
    return m_bindings.getWriteSet(binding, m_sets.empty() ? VK_NULL_HANDLE : m_sets[setIndex], dstArrayElement, descriptorCount);
  }

private:
  DescriptorPack(const DescriptorPack&)            = delete;
  DescriptorPack& operator=(const DescriptorPack&) = delete;

  DescriptorBindings           m_bindings;
  VkDescriptorSetLayout        m_layout = VK_NULL_HANDLE;
  VkDescriptorPool             m_pool   = VK_NULL_HANDLE;
  std::vector<VkDescriptorSet> m_sets;

  VkDevice m_device = nullptr;
};

//////////////////////////////////////////////////////////////////////////

// Helper function to create a pipeline layout.
inline VkResult createPipelineLayout(VkDevice                               device,
                                     VkPipelineLayout*                      pPipelineLayout,
                                     std::span<const VkDescriptorSetLayout> layouts            = {},
                                     std::span<const VkPushConstantRange>   pushConstantRanges = {})
{
  const VkPipelineLayoutCreateInfo info{.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                                        .setLayoutCount         = static_cast<uint32_t>(layouts.size()),
                                        .pSetLayouts            = layouts.data(),
                                        .pushConstantRangeCount = static_cast<uint32_t>(pushConstantRanges.size()),
                                        .pPushConstantRanges    = pushConstantRanges.data()};
  return vkCreatePipelineLayout(device, &info, nullptr, pPipelineLayout);
}

// Overload so you can write {layout}, {pushConstantRange}
inline VkResult createPipelineLayout(VkDevice                                           device,
                                     VkPipelineLayout*                                  pPipelineLayout,
                                     std::initializer_list<const VkDescriptorSetLayout> layouts            = {},
                                     std::initializer_list<const VkPushConstantRange>   pushConstantRanges = {})
{
  return createPipelineLayout(device, pPipelineLayout, std::span<const VkDescriptorSetLayout>(layouts.begin(), layouts.size()),
                              std::span<const VkPushConstantRange>(pushConstantRanges.begin(), pushConstantRanges.size()));
}

//////////////////////////////////////////////////////////////////////////

// Storage class for write set containers with their payload
// Can be used to drive `vkUpdateDescriptorSets` as well
// as `vkCmdPushDescriptorSet`
class WriteSetContainer
{
public:
  // single element (writeSet.descriptorCount must be 1)
  void append(const VkWriteDescriptorSet& writeSet, const nvvk::Buffer& buffer, VkDeviceSize offset = 0, VkDeviceSize range = VK_WHOLE_SIZE);
  void append(const VkWriteDescriptorSet& writeSet, const nvvk::AccelerationStructure& accel);
  void append(const VkWriteDescriptorSet& writeSet, const nvvk::Image& image);

  void append(const VkWriteDescriptorSet& writeSet, VkBuffer buffer, VkDeviceSize offset = 0, VkDeviceSize range = VK_WHOLE_SIZE);
  void append(const VkWriteDescriptorSet& writeSet, VkBufferView bufferView);
  void append(const VkWriteDescriptorSet& writeSet, const VkDescriptorBufferInfo& bufferInfo);
  void append(const VkWriteDescriptorSet& writeSet, VkImageView imageView, VkImageLayout imageLayout, VkSampler sampler = VK_NULL_HANDLE);
  void append(const VkWriteDescriptorSet& writeSet, const VkDescriptorImageInfo& imageInfo);
  void append(const VkWriteDescriptorSet& writeSet, VkAccelerationStructureKHR accel);

  // writeSet.descriptorCount many elements
  void append(const VkWriteDescriptorSet& writeSet, const nvvk::Buffer* buffers);  // offset 0 and VK_WHOLE_SIZE
  void append(const VkWriteDescriptorSet& writeSet, const nvvk::AccelerationStructure* accels);
  void append(const VkWriteDescriptorSet& writeSet, const nvvk::Image* images);

  void append(const VkWriteDescriptorSet& writeSet, const VkDescriptorBufferInfo* bufferInfos);
  void append(const VkWriteDescriptorSet& writeSet, const VkDescriptorImageInfo* imageInfos);
  void append(const VkWriteDescriptorSet& writeSet, const VkAccelerationStructureKHR* accels);
  void append(const VkWriteDescriptorSet& writeSet, const VkBufferView* bufferViews);

  void clear();

  void reserve(uint32_t count)
  {
    m_writeSets.reserve(count);
    m_bufferOrImageDatas.reserve(count);
    m_accelOrViewDatas.reserve(count);
  }

  uint32_t size() const { return uint32_t(m_writeSets.size()); }

  // not a const function, as it will update the internal pointers if necessary
  const VkWriteDescriptorSet* data();

private:
  union BufferOrImageData
  {
    VkDescriptorBufferInfo buffer;
    VkDescriptorImageInfo  image;
  };
  static_assert(sizeof(VkDescriptorBufferInfo) == sizeof(VkDescriptorImageInfo));
  union AccelOrViewData
  {
    VkBufferView               texelBufferView;
    VkAccelerationStructureKHR accel;
  };
  static_assert(sizeof(VkBufferView) == sizeof(VkAccelerationStructureKHR));

  std::vector<VkWriteDescriptorSet>                         m_writeSets;
  std::vector<VkWriteDescriptorSetAccelerationStructureKHR> m_writeAccels;
  std::vector<BufferOrImageData>                            m_bufferOrImageDatas;
  std::vector<AccelOrViewData>                              m_accelOrViewDatas;
  bool                                                      m_needPointerUpdate = true;
};

//////////////////////////////////////////////////////////////////////////


}  // namespace nvvk
