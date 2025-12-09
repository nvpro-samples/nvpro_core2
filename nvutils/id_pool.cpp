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
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cassert>
#include <cstring>

#include "id_pool.hpp"

namespace nvutils {

IDPool::IDPool(IDPool&& other) noexcept
    : m_ranges(other.m_ranges)
    , m_count(other.m_count)
    , m_capacity(other.m_capacity)
    , m_maxID(other.m_maxID)
    , m_usedIDs(other.m_usedIDs)
{
  other.m_ranges   = nullptr;
  other.m_count    = 0;
  other.m_capacity = 0;
  other.m_maxID    = 0;
  other.m_usedIDs  = 0;
}

nvutils::IDPool& IDPool::operator=(IDPool&& other) noexcept
{
  if(this == &other)
    return *this;  // Handle self-assignment

  deinit();
  m_ranges         = other.m_ranges;
  m_count          = other.m_count;
  m_capacity       = other.m_capacity;
  m_maxID          = other.m_maxID;
  m_usedIDs        = other.m_usedIDs;
  other.m_ranges   = nullptr;
  other.m_count    = 0;
  other.m_capacity = 0;
  other.m_maxID    = 0;
  other.m_usedIDs  = 0;
  return *this;
}

//////////////////////////////////////////////////////////////////////////
// most of the following code is taken from Emil Persson's MakeID
// http://www.humus.name/3D/MakeID.h (v1.02)

void IDPool::init(const uint32_t poolSize)
{
  assert(!m_ranges && "init called multiple times");
  assert(poolSize);

  uint32_t maxID = poolSize - 1;

  // Start with a single range, from 0 to max allowed ID (specified)
  m_ranges = static_cast<Range*>(::malloc(sizeof(Range)));
  assert(m_ranges != nullptr);  // Make sure allocation succeeded
  m_ranges[0].first = 0;
  m_ranges[0].last  = maxID;
  m_count           = 1;
  m_capacity        = 1;
  m_maxID           = maxID;
  m_usedIDs         = 0;
}

void IDPool::destroyAll()
{
  uint32_t poolSize = m_maxID + 1;
  m_usedIDs         = 0;
  deinit();
  init(poolSize);
}

void IDPool::deinit()
{
  assert(!m_usedIDs && "not all IDs were destroyed");

  if(m_ranges)
  {
    ::free(m_ranges);
    m_ranges   = nullptr;
    m_count    = 0;
    m_capacity = 0;
    m_maxID    = 0;
    m_usedIDs  = 0;
  }
}

bool IDPool::createID(uint32_t& id)
{
  if(m_ranges[0].first <= m_ranges[0].last)
  {
    id = m_ranges[0].first;

    // If current range is full and there is another one, that will become the new current range
    if(m_ranges[0].first == m_ranges[0].last && m_count > 1)
    {
      destroyRange(0);
    }
    else
    {
      ++m_ranges[0].first;
    }

    m_usedIDs++;
    return true;
  }

  // No availble ID left
  return false;
}

bool IDPool::createRangeID(uint32_t& id, const uint32_t count)
{
  uint32_t i = 0;
  do
  {
    const uint32_t range_count = 1 + m_ranges[i].last - m_ranges[i].first;
    if(count <= range_count)
    {
      id = m_ranges[i].first;

      // If current range is full and there is another one, that will become the new current range
      if(count == range_count && i + 1 < m_count)
      {
        destroyRange(i);
      }
      else
      {
        m_ranges[i].first += count;
      }

      m_usedIDs += count;
      return true;
    }
    ++i;
  } while(i < m_count);

  // No range of free IDs was large enough to create the requested continuous ID sequence
  return false;
}

bool IDPool::destroyRangeID(const uint32_t id, const uint32_t count)
{
  const uint32_t end_id = id + count;

  assert(end_id <= m_maxID + 1);

  // Binary search of the range list
  uint32_t i0 = 0;
  uint32_t i1 = m_count - 1;

  for(;;)
  {
    const uint32_t i = (i0 + i1) / 2;

    if(id < m_ranges[i].first)
    {
      // Before current range, check if neighboring
      if(end_id >= m_ranges[i].first)
      {
        if(end_id != m_ranges[i].first)
          return false;  // Overlaps a range of free IDs, thus (at least partially) invalid IDs

        // Neighbor id, check if neighboring previous range too
        if(i > i0 && id - 1 == m_ranges[i - 1].last)
        {
          // Merge with previous range
          m_ranges[i - 1].last = m_ranges[i].last;
          destroyRange(i);
        }
        else
        {
          // Just grow range
          m_ranges[i].first = id;
        }

        m_usedIDs -= count;
        return true;
      }
      else
      {
        // Non-neighbor id
        if(i != i0)
        {
          // Cull upper half of list
          i1 = i - 1;
        }
        else
        {
          // Found our position in the list, insert the deleted range here
          insertRange(i);
          m_ranges[i].first = id;
          m_ranges[i].last  = end_id - 1;

          m_usedIDs -= count;
          return true;
        }
      }
    }
    else if(id > m_ranges[i].last)
    {
      // After current range, check if neighboring
      if(id - 1 == m_ranges[i].last)
      {
        // Neighbor id, check if neighboring next range too
        if(i < i1 && end_id == m_ranges[i + 1].first)
        {
          // Merge with next range
          m_ranges[i].last = m_ranges[i + 1].last;
          destroyRange(i + 1);
        }
        else
        {
          // Just grow range
          m_ranges[i].last += count;
        }
        m_usedIDs -= count;
        return true;
      }
      else
      {
        // Non-neighbor id
        if(i != i1)
        {
          // Cull bottom half of list
          i0 = i + 1;
        }
        else
        {
          // Found our position in the list, insert the deleted range here
          insertRange(i + 1);
          m_ranges[i + 1].first = id;
          m_ranges[i + 1].last  = end_id - 1;

          m_usedIDs -= count;
          return true;
        }
      }
    }
    else
    {
      // Inside a free block, not a valid ID
      return false;
    }
  }
}

bool IDPool::isRangeAvailable(uint32_t searchCount) const
{
  uint32_t i = 0;
  do
  {
    uint32_t count = m_ranges[i].last - m_ranges[i].first + 1;
    if(count >= searchCount)
      return true;

    ++i;
  } while(i < m_count);

  return false;
}

void IDPool::printRanges() const
{
  uint32_t i = 0;
  for(;;)
  {
    if(m_ranges[i].first < m_ranges[i].last)
      printf("%u-%u", m_ranges[i].first, m_ranges[i].last);
    else if(m_ranges[i].first == m_ranges[i].last)
      printf("%u", m_ranges[i].first);
    else
      printf("-");

    ++i;
    if(i >= m_count)
    {
      printf("\n");
      return;
    }

    printf(", ");
  }
}

void IDPool::checkRanges() const
{
  for(uint32_t i = 0; i < m_count; i++)
  {
    assert(m_ranges[i].last <= m_maxID);

    if(m_ranges[i].first == m_ranges[i].last + 1)
    {
      continue;
    }
    assert(m_ranges[i].first <= m_ranges[i].last);
    assert(m_ranges[i].first <= m_maxID);
  }
}

void IDPool::insertRange(const uint32_t index)
{
  if(m_count >= m_capacity)
  {
    m_capacity += m_capacity;
    m_ranges = (Range*)realloc(m_ranges, m_capacity * sizeof(Range));
    assert(m_ranges);  // Make sure reallocation succeeded
  }

  ::memmove(m_ranges + index + 1, m_ranges + index, (m_count - index) * sizeof(Range));
  ++m_count;
}

void IDPool::destroyRange(const uint32_t index)
{
  --m_count;
  ::memmove(m_ranges + index, m_ranges + index + 1, (m_count - index) * sizeof(Range));
}

}  // namespace nvutils

[[maybe_unused]] static void usage_IDPool()
{
  // let's allow up to 16-bit worth of textures
  nvutils::IDPool idGen(1 << 16);

  uint32_t bindlessTextureID;
  idGen.createID(bindlessTextureID);

  // use bindlessTextureID to fill a descriptor array element

  // when the texture is deleted, return the ID

  idGen.destroyID(bindlessTextureID);
}
