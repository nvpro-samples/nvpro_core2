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

#include "descriptors.hpp"

#include <nvvk/check_error.hpp>

namespace nvvk {

constexpr uint32_t NO_BINDING_INDEX = ~0u;

void DescriptorBindings::addBinding(uint32_t                 binding,
                                    VkDescriptorType         descriptorType,
                                    uint32_t                 descriptorCount,
                                    VkShaderStageFlags       stageFlags,
                                    const VkSampler*         pImmutableSamplers,
                                    VkDescriptorBindingFlags bindingFlags)
{
  addBinding(VkDescriptorSetLayoutBinding{.binding            = binding,
                                          .descriptorType     = descriptorType,
                                          .descriptorCount    = descriptorCount,
                                          .stageFlags         = stageFlags,
                                          .pImmutableSamplers = pImmutableSamplers},
             bindingFlags);
}

void DescriptorBindings::addBinding(const VkDescriptorSetLayoutBinding& layoutBinding, VkDescriptorBindingFlags bindingFlags)
{
  // Update m_bindingToIndex.
  if(m_bindingToIndex.size() <= layoutBinding.binding)
  {
    m_bindingToIndex.resize(layoutBinding.binding + 1, NO_BINDING_INDEX);
  }
  m_bindingToIndex[layoutBinding.binding] = static_cast<uint32_t>(m_bindings.size());

  m_bindings.push_back(layoutBinding);
  m_bindingFlags.push_back(bindingFlags);
}

void DescriptorBindings::addBindings(std::span<const VkDescriptorSetLayoutBinding> layoutBindings, VkDescriptorBindingFlags bindingFlags)
{
  for(auto& b : layoutBindings)
  {
    addBinding(b, bindingFlags);
  }
}

void DescriptorBindings::addBindings(std::initializer_list<const VkDescriptorSetLayoutBinding> layoutBindings,
                                     VkDescriptorBindingFlags                                  bindingFlags)
{
  for(auto& b : layoutBindings)
  {
    addBinding(b, bindingFlags);
  }
}

VkWriteDescriptorSet DescriptorBindings::getWriteSet(uint32_t binding, VkDescriptorSet dstSet, uint32_t dstArrayElement, uint32_t descriptorCount) const
{
  VkWriteDescriptorSet writeSet = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
  writeSet.descriptorType       = VK_DESCRIPTOR_TYPE_MAX_ENUM;

  if(binding >= m_bindingToIndex.size())
  {
    assert(!"`binding` was out of range!");
    return writeSet;
  }
  const uint32_t i = m_bindingToIndex[binding];
  if(i == NO_BINDING_INDEX)
  {
    assert(!"`binding` was never added!");
    return writeSet;
  }
  const VkDescriptorSetLayoutBinding& b = m_bindings[i];

  writeSet.descriptorCount = dstArrayElement == ~0 ? b.descriptorCount : descriptorCount;
  writeSet.descriptorType  = b.descriptorType;
  writeSet.dstBinding      = binding;
  writeSet.dstSet          = dstSet;
  writeSet.dstArrayElement = dstArrayElement == ~0 ? 0 : dstArrayElement;

  assert(writeSet.dstArrayElement + writeSet.descriptorCount <= b.descriptorCount);

  return writeSet;
}

VkResult DescriptorBindings::createDescriptorSetLayout(VkDevice                         device,
                                                       VkDescriptorSetLayoutCreateFlags flags,
                                                       VkDescriptorSetLayout*           pDescriptorSetLayout) const
{
  VkResult result;
  VkDescriptorSetLayoutBindingFlagsCreateInfo bindingsInfo = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO};

  bindingsInfo.bindingCount  = uint32_t(m_bindingFlags.size());
  bindingsInfo.pBindingFlags = m_bindingFlags.data();

  VkDescriptorSetLayoutCreateInfo createInfo = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
  createInfo.bindingCount                    = uint32_t(m_bindings.size());
  createInfo.pBindings                       = m_bindings.data();
  createInfo.flags                           = flags;
  createInfo.pNext                           = &bindingsInfo;

  result = vkCreateDescriptorSetLayout(device, &createInfo, nullptr, pDescriptorSetLayout);

  return result;
}

void DescriptorBindings::appendPoolSizes(std::vector<VkDescriptorPoolSize>& poolSizes, uint32_t numSets, uint32_t totalVariableCount) const
{

  for(size_t i = 0; i < m_bindings.size(); i++)
  {
    const VkDescriptorSetLayoutBinding& it           = m_bindings[i];
    const VkDescriptorBindingFlags      bindingFlags = m_bindingFlags[i];

    // Bindings can have a zero descriptor count, used for the layout, but don't reserve storage for them.
    if(it.descriptorCount == 0)
    {
      continue;
    }

    uint32_t count = it.descriptorCount * numSets;
    if(totalVariableCount && bindingFlags & VK_DESCRIPTOR_BINDING_VARIABLE_DESCRIPTOR_COUNT_BIT)
    {
      count = totalVariableCount;
    }

    bool found = false;
    for(VkDescriptorPoolSize& itpool : poolSizes)
    {
      if(itpool.type == it.descriptorType)
      {
        itpool.descriptorCount += count;
        found = true;
        break;
      }
    }
    if(!found)
    {
      VkDescriptorPoolSize poolSize{};
      poolSize.type            = it.descriptorType;
      poolSize.descriptorCount = count;
      poolSizes.push_back(poolSize);
    }
  }
}

std::vector<VkDescriptorPoolSize> DescriptorBindings::calculatePoolSizes(uint32_t numSets, uint32_t totalVariableCount) const
{
  std::vector<VkDescriptorPoolSize> poolSizes;
  appendPoolSizes(poolSizes, numSets, totalVariableCount);
  return poolSizes;
}

//////////////////////////////////////////////////////////////////////////

VkResult DescriptorPack::init(const DescriptorBindings&        bindings,
                              VkDevice                         device,
                              uint32_t                         numSets,
                              VkDescriptorSetLayoutCreateFlags layoutFlags,
                              VkDescriptorPoolCreateFlags      poolFlags,
                              uint32_t                         totalVariableCount,
                              const uint32_t*                  descriptorVariableCounts)
{
  assert(nullptr == m_device && "initFromBindings must not be called twice in a row!");
  m_device = device;

  m_bindings = bindings;

  NVVK_FAIL_RETURN(bindings.createDescriptorSetLayout(device, layoutFlags, &m_layout));

  if(numSets > 0)
  {
    // Pool
    const std::vector<VkDescriptorPoolSize> poolSizes = bindings.calculatePoolSizes(numSets, totalVariableCount);
    const VkDescriptorPoolCreateInfo        poolCreateInfo{.sType   = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                                                           .flags   = poolFlags,
                                                           .maxSets = numSets,
                                                           .poolSizeCount = static_cast<uint32_t>(poolSizes.size()),
                                                           .pPoolSizes    = poolSizes.data()};
    NVVK_FAIL_RETURN(vkCreateDescriptorPool(device, &poolCreateInfo, nullptr, &m_pool));

    // Sets
    m_sets.resize(numSets);
    std::vector<VkDescriptorSetLayout> allocInfoLayouts(numSets, m_layout);
    VkDescriptorSetAllocateInfo        allocInfo{.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                                                 .descriptorPool     = m_pool,
                                                 .descriptorSetCount = numSets,
                                                 .pSetLayouts        = allocInfoLayouts.data()};
    // Optional variable descriptor counts
    VkDescriptorSetVariableDescriptorCountAllocateInfo varInfo{
        .sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_VARIABLE_DESCRIPTOR_COUNT_ALLOCATE_INFO,
        .descriptorSetCount = numSets,
        .pDescriptorCounts  = descriptorVariableCounts,
    };
    if(totalVariableCount > 0 && descriptorVariableCounts)
    {
      allocInfo.pNext = &varInfo;
    }

    NVVK_FAIL_RETURN(vkAllocateDescriptorSets(device, &allocInfo, m_sets.data()));
  }

  return VK_SUCCESS;
}

void DescriptorPack::deinit()
{
  m_bindings.clear();
  m_sets.clear();

  if(m_device)  // Only run if ever initialized
  {
    vkDestroyDescriptorSetLayout(m_device, m_layout, nullptr);
    m_layout = VK_NULL_HANDLE;

    vkDestroyDescriptorPool(m_device, m_pool, nullptr);
    m_pool = VK_NULL_HANDLE;

    m_device = nullptr;
  }
}


DescriptorPack::DescriptorPack(DescriptorPack&& other) noexcept
{
  this->operator=(std::move(other));
}

DescriptorPack& DescriptorPack::operator=(DescriptorPack&& other) noexcept
{
  assert(!m_device && "can't move into non-empty object");

  m_sets = std::move(other.m_sets);

  m_pool   = other.m_pool;
  m_layout = other.m_layout;
  m_device = other.m_device;

  other.m_pool   = VK_NULL_HANDLE;
  other.m_layout = VK_NULL_HANDLE;
  other.m_device = VK_NULL_HANDLE;

  return *this;
}

DescriptorPack::~DescriptorPack()
{
  assert(!m_device && "deinit() missing");
}

//////////////////////////////////////////////////////////////////////////

void WriteSetContainer::append(const VkWriteDescriptorSet& writeSet, VkBuffer buffer, VkDeviceSize offset, VkDeviceSize range)
{
  assert(writeSet.pImageInfo == nullptr);
  assert(writeSet.pTexelBufferView == nullptr);
  assert(writeSet.descriptorCount == 1);

  m_writeSets.emplace_back(writeSet).pBufferInfo = (const VkDescriptorBufferInfo*)1;

  BufferOrImageData basics;
  basics.buffer.buffer = buffer;
  basics.buffer.offset = offset;
  basics.buffer.range  = range;
  m_bufferOrImageDatas.emplace_back(basics);

  m_needPointerUpdate = true;
}

void WriteSetContainer::append(const VkWriteDescriptorSet& writeSet, const VkDescriptorBufferInfo& bufferInfo)
{
  assert(writeSet.pImageInfo == nullptr);
  assert(writeSet.pTexelBufferView == nullptr);
  assert(writeSet.descriptorCount == 1);

  m_writeSets.emplace_back(writeSet).pBufferInfo = (const VkDescriptorBufferInfo*)1;

  BufferOrImageData basics;
  basics.buffer = bufferInfo;
  m_bufferOrImageDatas.emplace_back(basics);

  m_needPointerUpdate = true;
}

void WriteSetContainer::append(const VkWriteDescriptorSet& writeSet, const nvvk::Buffer& buffer, VkDeviceSize offset, VkDeviceSize range)
{
  assert(writeSet.pImageInfo == nullptr);
  assert(writeSet.pTexelBufferView == nullptr);
  assert(writeSet.descriptorCount == 1);

  m_writeSets.emplace_back(writeSet).pBufferInfo = (const VkDescriptorBufferInfo*)1;

  BufferOrImageData basics;
  basics.buffer.buffer = buffer.buffer;
  basics.buffer.offset = offset;
  basics.buffer.range  = range;
  m_bufferOrImageDatas.emplace_back(basics);

  m_needPointerUpdate = true;
}

void WriteSetContainer::append(const VkWriteDescriptorSet& writeSet, const nvvk::Buffer* buffers)
{
  assert(writeSet.pImageInfo == nullptr);
  assert(writeSet.pTexelBufferView == nullptr);

  m_writeSets.emplace_back(writeSet).pBufferInfo = (const VkDescriptorBufferInfo*)1;

  for(uint32_t i = 0; i < writeSet.descriptorCount; i++)
  {
    BufferOrImageData basics;
    basics.buffer.buffer = buffers[i].buffer;
    basics.buffer.offset = 0;
    basics.buffer.range  = VK_WHOLE_SIZE;
    m_bufferOrImageDatas.emplace_back(basics);
  }

  m_needPointerUpdate = true;
}

void WriteSetContainer::append(const VkWriteDescriptorSet& writeSet, const VkDescriptorBufferInfo* bufferInfos)
{
  assert(writeSet.pImageInfo == nullptr);
  assert(writeSet.pTexelBufferView == nullptr);

  m_writeSets.emplace_back(writeSet).pBufferInfo = (const VkDescriptorBufferInfo*)1;

  for(uint32_t i = 0; i < writeSet.descriptorCount; i++)
  {
    BufferOrImageData basics;
    basics.buffer = bufferInfos[i];
    m_bufferOrImageDatas.emplace_back(basics);
  }

  m_needPointerUpdate = true;
}

void WriteSetContainer::append(const VkWriteDescriptorSet& writeSet, VkBufferView bufferView)
{
  assert(writeSet.pImageInfo == nullptr);
  assert(writeSet.pBufferInfo == nullptr);
  assert(writeSet.descriptorCount == 1);

  m_writeSets.emplace_back(writeSet).pTexelBufferView = (const VkBufferView*)1;

  AccelOrViewData basics;
  basics.texelBufferView = bufferView;
  m_accelOrViewDatas.emplace_back(basics);

  m_needPointerUpdate = true;
}

void WriteSetContainer::append(const VkWriteDescriptorSet& writeSet, const VkBufferView* bufferViews)
{
  assert(writeSet.pImageInfo == nullptr);
  assert(writeSet.pBufferInfo == nullptr);

  m_writeSets.emplace_back(writeSet).pTexelBufferView = (const VkBufferView*)1;

  for(uint32_t i = 0; i < writeSet.descriptorCount; i++)
  {
    AccelOrViewData basics;
    basics.texelBufferView = bufferViews[i];
    m_accelOrViewDatas.emplace_back(basics);
  }

  m_needPointerUpdate = true;
}

void WriteSetContainer::append(const VkWriteDescriptorSet& writeSet, const VkDescriptorImageInfo& imageInfo)
{
  assert(writeSet.pTexelBufferView == nullptr);
  assert(writeSet.pBufferInfo == nullptr);
  assert(writeSet.descriptorCount == 1);

  m_writeSets.emplace_back(writeSet).pImageInfo = (const VkDescriptorImageInfo*)1;

  BufferOrImageData basics;
  basics.image = imageInfo;
  m_bufferOrImageDatas.emplace_back(basics);

  m_needPointerUpdate = true;
}

void WriteSetContainer::append(const VkWriteDescriptorSet& writeSet, const nvvk::Image& image)
{
  assert(writeSet.pTexelBufferView == nullptr);
  assert(writeSet.pBufferInfo == nullptr);
  assert(writeSet.descriptorCount == 1);
  assert(image.descriptor.imageView);

  m_writeSets.emplace_back(writeSet).pImageInfo = (const VkDescriptorImageInfo*)1;

  BufferOrImageData basics;
  basics.image = image.descriptor;
  m_bufferOrImageDatas.emplace_back(basics);

  m_needPointerUpdate = true;
}

void WriteSetContainer::append(const VkWriteDescriptorSet& writeSet, VkImageView imageView, VkImageLayout imageLayout, VkSampler sampler)
{
  assert(writeSet.pTexelBufferView == nullptr);
  assert(writeSet.pBufferInfo == nullptr);
  assert(writeSet.descriptorCount == 1);

  m_writeSets.emplace_back(writeSet).pImageInfo = (const VkDescriptorImageInfo*)1;

  BufferOrImageData basics;
  basics.image.imageView   = imageView;
  basics.image.imageLayout = imageLayout;
  basics.image.sampler     = sampler;
  m_bufferOrImageDatas.emplace_back(basics);

  m_needPointerUpdate = true;
}

void WriteSetContainer::append(const VkWriteDescriptorSet& writeSet, const nvvk::Image* images)
{
  assert(writeSet.pTexelBufferView == nullptr);
  assert(writeSet.pBufferInfo == nullptr);

  m_writeSets.emplace_back(writeSet).pImageInfo = (const VkDescriptorImageInfo*)1;

  for(uint32_t i = 0; i < writeSet.descriptorCount; i++)
  {
    assert(images[i].descriptor.imageView);
    BufferOrImageData basics;
    basics.image = images[i].descriptor;
    m_bufferOrImageDatas.emplace_back(basics);
  }

  m_needPointerUpdate = true;
}
void WriteSetContainer::append(const VkWriteDescriptorSet& writeSet, const VkDescriptorImageInfo* imageInfos)
{
  assert(writeSet.pTexelBufferView == nullptr);
  assert(writeSet.pBufferInfo == nullptr);

  m_writeSets.emplace_back(writeSet).pImageInfo = (const VkDescriptorImageInfo*)1;

  for(uint32_t i = 0; i < writeSet.descriptorCount; i++)
  {
    BufferOrImageData basics;
    basics.image = imageInfos[i];
    m_bufferOrImageDatas.emplace_back(basics);
  }

  m_needPointerUpdate = true;
}

void WriteSetContainer::append(const VkWriteDescriptorSet& writeSet, VkAccelerationStructureKHR accel)
{
  assert(writeSet.pImageInfo == nullptr);
  assert(writeSet.pTexelBufferView == nullptr);
  assert(writeSet.pBufferInfo == nullptr);
  assert(writeSet.descriptorCount == 1);

  m_writeSets.emplace_back(writeSet);

  AccelOrViewData basics;
  basics.accel = accel;
  m_accelOrViewDatas.emplace_back(basics);
  m_writeAccels.push_back(
      {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR, .accelerationStructureCount = 1});

  m_needPointerUpdate = true;
}

void WriteSetContainer::append(const VkWriteDescriptorSet& writeSet, const nvvk::AccelerationStructure& accel)
{
  assert(writeSet.pImageInfo == nullptr);
  assert(writeSet.pTexelBufferView == nullptr);
  assert(writeSet.pBufferInfo == nullptr);
  assert(writeSet.descriptorCount == 1);

  m_writeSets.emplace_back(writeSet);

  AccelOrViewData basics;
  basics.accel = accel.accel;
  m_accelOrViewDatas.emplace_back(basics);
  m_writeAccels.push_back(
      {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR, .accelerationStructureCount = 1});

  m_needPointerUpdate = true;
}

void WriteSetContainer::append(const VkWriteDescriptorSet& writeSet, const nvvk::AccelerationStructure* accels)
{
  assert(writeSet.pImageInfo == nullptr);
  assert(writeSet.pTexelBufferView == nullptr);
  assert(writeSet.pBufferInfo == nullptr);

  m_writeSets.emplace_back(writeSet);

  for(uint32_t i = 0; i < writeSet.descriptorCount; i++)
  {
    AccelOrViewData basics;
    basics.accel = accels[i].accel;
    m_accelOrViewDatas.emplace_back(basics);
  }

  m_writeAccels.push_back({.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR,
                           .accelerationStructureCount = writeSet.descriptorCount});

  m_needPointerUpdate = true;
}

void WriteSetContainer::append(const VkWriteDescriptorSet& writeSet, const VkAccelerationStructureKHR* accels)
{

  assert(writeSet.pImageInfo == nullptr);
  assert(writeSet.pTexelBufferView == nullptr);
  assert(writeSet.pBufferInfo == nullptr);

  m_writeSets.emplace_back(writeSet);

  for(uint32_t i = 0; i < writeSet.descriptorCount; i++)
  {
    AccelOrViewData basics;
    basics.accel = accels[i];
    m_accelOrViewDatas.emplace_back(basics);
  }

  m_writeAccels.push_back({.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR,
                           .accelerationStructureCount = writeSet.descriptorCount});

  m_needPointerUpdate = true;
}

void WriteSetContainer::clear()
{
  m_writeSets.clear();
  m_writeAccels.clear();
  m_bufferOrImageDatas.clear();
  m_accelOrViewDatas.clear();

  m_needPointerUpdate = true;
}

const VkWriteDescriptorSet* WriteSetContainer::data()
{
  if(m_needPointerUpdate)
  {
    size_t accelWriteIndex    = 0;
    size_t bufferOrImageIndex = 0;
    size_t accelOrViewIndex   = 0;

    for(size_t i = 0; i < m_writeSets.size(); i++)
    {
      if(m_writeSets[i].pBufferInfo)
      {
        m_writeSets[i].pBufferInfo = &m_bufferOrImageDatas[bufferOrImageIndex].buffer;
        bufferOrImageIndex += m_writeSets[i].descriptorCount;
      }
      else if(m_writeSets[i].pImageInfo)
      {
        m_writeSets[i].pImageInfo = &m_bufferOrImageDatas[bufferOrImageIndex].image;
        bufferOrImageIndex += m_writeSets[i].descriptorCount;
      }
      else if(m_writeSets[i].pTexelBufferView)
      {
        m_writeSets[i].pTexelBufferView = &m_accelOrViewDatas[accelOrViewIndex].texelBufferView;
        accelOrViewIndex += m_writeSets[i].descriptorCount;
      }
      else if(m_writeSets[i].descriptorType == VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR)
      {
        m_writeAccels[accelWriteIndex].pAccelerationStructures = &m_accelOrViewDatas[accelOrViewIndex].accel;
        accelOrViewIndex += m_writeSets[i].descriptorCount;

        m_writeSets[i].pNext = &m_writeAccels[accelWriteIndex];
        accelWriteIndex++;
      }
    }
  }

  return m_writeSets.empty() ? nullptr : m_writeSets.data();
}
}  // namespace nvvk


//--------------------------------------------------------------------------------------------------
// Usage example
//--------------------------------------------------------------------------------------------------
[[maybe_unused]] static void usage_DescriptorBindings()
{
  VkDevice     device = nullptr;
  nvvk::Buffer myBufferA;
  nvvk::Buffer myBufferB;
  uint32_t     SHADERIO_BINDING = 0;
  struct PushConstants
  {
    float iResolution[2];
  };

  constexpr uint32_t NUM_SETS = 2;

  // Manually create layout and pool
  {
    // Create bindings.
    nvvk::DescriptorBindings bindings;
    // Binding `SHADERIO_BINDING` is 1 uniform buffer accessible to all stages,
    // that can be updated after binding when not in use.
    bindings.addBinding(SHADERIO_BINDING, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_ALL, nullptr,
                        VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT | VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT);

    // To manually create a layout and a pool and to allocate NUM_SETS sets:
    {
      VkDescriptorSetLayout dlayout = VK_NULL_HANDLE;
      bindings.createDescriptorSetLayout(device, 0, &dlayout);

      std::vector<VkDescriptorPoolSize> poolSizes = bindings.calculatePoolSizes(NUM_SETS);
      // Or if you have multiple descriptor layouts you'd like to allocate from a
      // single pool, you can use bindings.appendPoolSizes().

      VkDescriptorPool                 dpool = VK_NULL_HANDLE;
      const VkDescriptorPoolCreateInfo dpoolInfo{.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                                                 .maxSets       = NUM_SETS,
                                                 .poolSizeCount = uint32_t(poolSizes.size()),
                                                 .pPoolSizes    = poolSizes.data()};
      NVVK_CHECK(vkCreateDescriptorPool(device, &dpoolInfo, nullptr, &dpool));

      std::vector<VkDescriptorSet>       sets(NUM_SETS);
      std::vector<VkDescriptorSetLayout> allocInfoLayouts(NUM_SETS, dlayout);
      const VkDescriptorSetAllocateInfo  allocInfo{.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                                                   .descriptorPool     = dpool,
                                                   .descriptorSetCount = NUM_SETS,
                                                   .pSetLayouts        = allocInfoLayouts.data()};
      NVVK_CHECK(vkAllocateDescriptorSets(device, &allocInfo, sets.data()));

      // Cleanup
      vkDestroyDescriptorPool(device, dpool, nullptr);
    }
  }

  // Or have DescriptorPack simplify the above:
  nvvk::DescriptorPack dpack;
  {
    nvvk::DescriptorBindings bindings;
    bindings.addBinding(SHADERIO_BINDING, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_ALL, nullptr,
                        VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT | VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT);
    // The second argument, NUM_SETS here, can be left 0 when the intend is to use push descriptors, it defaults to `1`
    NVVK_CHECK(dpack.init(bindings, device, NUM_SETS));
  }

  // To update DescriptorSets:
  {
    // std::vectors inside for VkWriteDescriptorSets as well as the corresponding payloads.
    nvvk::WriteSetContainer writeContainer;
    // makeDescriptorWrite returns a VkWriteDescriptorSet without actual binding information,
    // the append function takes care of that.

    // when preparing push descriptors, `dpack.sets[0]` would be omitted / nullptr
    writeContainer.append(dpack.makeWrite(SHADERIO_BINDING, 0), myBufferA);

    // shortcut to provide `dpack.sets[1]` (also works when dpack.sets.empty() for push descriptors)
    writeContainer.append(dpack.makeWrite(SHADERIO_BINDING, 1), myBufferB);

    vkUpdateDescriptorSets(device, writeContainer.size(), writeContainer.data(), 0, nullptr);

    // the writeContainer can also be used for push descriptors, when the VkDescriptorSet provided were nullptr
    // vkCmdPushDescriptorSet(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0,  writeContainer.size(), writeContainer.data());
  }

  // To create a pipeline layout with an additional push constant range:
  VkPushConstantRange pushRange{.stageFlags = VK_SHADER_STAGE_ALL, .offset = 0, .size = sizeof(PushConstants)};
  VkPipelineLayout    pipelineLayout = VK_NULL_HANDLE;
  NVVK_CHECK(nvvk::createPipelineLayout(device, &pipelineLayout, {dpack.getLayout()}, {pushRange}));

  // Cleanup
  vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
  dpack.deinit();
}
