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

#include "bit_array.hpp"

namespace nvutils {

/** \brief Create a new BitVector with all bits set to false
      \param size Number of Bits in the Array
  **/
BitArray::BitArray(size_t size, int defaultValue)
    : m_size(size)
    , m_bits(new BitStorageType[determineNumberOfElements()])
{
  if(defaultValue == 0)
  {
    clear();
  }
  else if(defaultValue > 0)
  {
    fill();
  }
}

BitArray::BitArray(const BitArray& rhs)
    : m_size(rhs.m_size)
    , m_bits(new BitStorageType[determineNumberOfElements()])
{
  std::copy(rhs.m_bits.get(), rhs.m_bits.get() + determineNumberOfElements(), m_bits.get());
}

BitArray::BitArray(BitArray&& other)
{
  m_bits = std::move(other.m_bits);
  std::swap(m_size, other.m_size);
}

BitArray& BitArray::operator=(BitArray&& other)
{
  if(this != &other)
  {
    m_bits = std::move(other.m_bits);
    std::swap(m_size, other.m_size);
  }

  return *this;
}

void BitArray::resize(size_t newSize, int defaultValue)
{
  // if the default value for the new bits is true enabled the unused bits in the last element
  // before we do the resize
  if(defaultValue > 0)
  {
    setUnusedBits();
  }

  size_t oldNumberOfElements = determineNumberOfElements();
  m_size                     = newSize;
  size_t newNumberOfElements = determineNumberOfElements();

  // the number of elements has changed, reallocate array
  if(oldNumberOfElements != newNumberOfElements)
  {
    std::unique_ptr<BitStorageType[]> newBits = std::make_unique<BitStorageType[]>(determineNumberOfElements());
    if(newNumberOfElements < oldNumberOfElements)
    {
      std::copy(m_bits.get(), m_bits.get() + newNumberOfElements, newBits.get());
    }
    else
    {
      std::copy(m_bits.get(), m_bits.get() + oldNumberOfElements, newBits.get());

      if(defaultValue >= 0)
      {
        std::fill(newBits.get() + oldNumberOfElements, newBits.get() + newNumberOfElements,
                  defaultValue > 0 ? ~BitStorageType(0) : BitStorageType(0));
      }
    }
    m_bits = std::move(newBits);
  }

  // always clear unused bits after resizing etc.
  clearUnusedBits();
}

BitArray& BitArray::operator=(const BitArray& rhs)
{
  if(m_size != rhs.m_size)
  {
    m_size = rhs.m_size;
    m_bits = std::make_unique<BitStorageType[]>(determineNumberOfElements());
  }
  std::copy(rhs.m_bits.get(), rhs.m_bits.get() + determineNumberOfElements(), m_bits.get());

  return *this;
}

bool BitArray::operator==(const BitArray& rhs) const
{
  return (m_size == rhs.m_size) ? std::equal(m_bits.get(), m_bits.get() + determineNumberOfElements(), rhs.m_bits.get()) : false;
}

BitArray BitArray::operator^(BitArray const& rhs)
{
  assert(size() == rhs.size());

  BitArray result(size(), -1);
  for(size_t index = 0; index < determineNumberOfElements(); ++index)
  {
    result.m_bits[index] = m_bits[index] ^ rhs.m_bits[index];
  }
  clearUnusedBits();

  return result;
}

BitArray BitArray::operator|(BitArray const& rhs)
{
  assert(size() == rhs.size());

  BitArray result(size(), -1);
  for(size_t index = 0; index < determineNumberOfElements(); ++index)
  {
    result.m_bits[index] = m_bits[index] | rhs.m_bits[index];
  }
  clearUnusedBits();

  return result;
}

BitArray BitArray::operator&(BitArray const& rhs)
{
  assert(size() == rhs.size());

  BitArray result(size(), -1);
  for(size_t index = 0; index < determineNumberOfElements(); ++index)
  {
    result.m_bits[index] = m_bits[index] & rhs.m_bits[index];
  }
  clearUnusedBits();

  return result;
}

BitArray& BitArray::operator^=(BitArray const& rhs)
{
  assert(size() == rhs.size());

  for(size_t index = 0; index < determineNumberOfElements(); ++index)
  {
    m_bits[index] ^= rhs.m_bits[index];
  }
  clearUnusedBits();

  return *this;
}

BitArray& BitArray::operator|=(BitArray const& rhs)
{
  assert(size() == rhs.size());

  for(size_t index = 0; index < determineNumberOfElements(); ++index)
  {
    m_bits[index] |= rhs.m_bits[index];
  }

  return *this;
}

BitArray& BitArray::operator&=(BitArray const& rhs)
{
  assert(size() == rhs.size());

  for(size_t index = 0; index < determineNumberOfElements(); ++index)
  {
    m_bits[index] &= rhs.m_bits[index];
  }

  return *this;
}

BitArray BitArray::operator~() const
{
  BitArray result(size(), false);

  for(size_t index = 0; index < determineNumberOfElements(); ++index)
  {
    result.m_bits[index] = ~m_bits[index];
  }
  result.clearUnusedBits();

  return result;
}

void BitArray::clear()
{
  std::fill(m_bits.get(), m_bits.get() + determineNumberOfElements(), 0);
}

void BitArray::fill()
{
  if(determineNumberOfElements())
  {
    std::fill(m_bits.get(), m_bits.get() + determineNumberOfElements(), ~0);

    clearUnusedBits();
  }
}

size_t BitArray::countLeadingZeroes() const
{
  size_t index = 0;

  // first count
  while(index < determineNumberOfElements() && !m_bits[index])
  {
    ++index;
  }

  size_t leadingZeroes = index * StorageBitsPerElement;
  if(index < determineNumberOfElements())
  {
    leadingZeroes += std::countr_zero(m_bits[index]);
  }

  return std::min(leadingZeroes, size());
}

size_t BitArray::countSetBits() const
{
  size_t count = 0;
  for(size_t i = 0; i < determineNumberOfElements(); i++)
  {
    count += std::popcount(m_bits[i]);
  }

  return count;
}

}  // namespace nvutils


//--------------------------------------------------------------------------------------------------
// Usage example
//--------------------------------------------------------------------------------------------------
[[maybe_unused]] static void usage_BitArray()
{
  nvutils::BitArray modifiedObjects(1024);

  // set some bits
  modifiedObjects.setBit(24, true);
  modifiedObjects.setBit(37, true);


  // let's say we want to update some collection of objects
  struct Object
  {
    uint32_t foo = 0;
    void     update() { foo++; }
  };

  Object myObjects[1024];

  // iterate over all set bits using the built-in traversal mechanism
  auto fnMyVisitor = [&](size_t index) { myObjects[index].update(); };

  modifiedObjects.traverseBits(fnMyVisitor);

  // supports some bitwise operations
  nvutils::BitArray notModified = ~modifiedObjects;
}
