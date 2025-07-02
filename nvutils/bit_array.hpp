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

#include <algorithm>
#include <cassert>
#include <memory>

#include "bit_operations.hpp"

namespace nvutils {

// This class provides a container for an array of bits.
// It provides utility functions for bitwise operations on all bits,
// as well as means to traverse all set bits.
class BitArray
{
public:
  typedef uint64_t          BitStorageType;
  static constexpr uint64_t StorageBitsPerElement = sizeof(BitStorageType) * 8;

  BitArray() = default;

  // `defaultValue` > 0 true, == 0 false, < 0 uninitialized
  BitArray(size_t size, int defaultValue = 0);
  BitArray(const BitArray& rhs);
  BitArray(BitArray&& other);

  BitArray& operator=(const BitArray& rhs);
  BitArray& operator=(BitArray&& other);
  bool      operator==(const BitArray& rhs) const;
  BitArray  operator^(BitArray const& rhs);
  BitArray  operator&(BitArray const& rhs);
  BitArray  operator|(BitArray const& rhs);
  BitArray& operator^=(BitArray const& rhs);
  BitArray& operator&=(BitArray const& rhs);
  BitArray& operator|=(BitArray const& rhs);
  BitArray  operator~() const;

  void clear();
  void fill();

  // Change the number of bits in this array. The state of remaining bits is being kept.
  // New bits will be initialized to false.
  // `newSize` New number of bits in this array
  // `defaultValue` > 0 true, == 0 false, < 0 uninitialized
  void resize(size_t newSize, int defaultValue = 0);

  size_t size() const { return m_size; }

  // inline functions
  void enableBit(size_t index);
  void disableBit(size_t index);
  void setBit(size_t index, bool value);
  bool getBit(size_t index) const;

  BitStorageType const* data() const;

  template <typename Visitor>
  void traverseBits(Visitor visitor) const;

  // `begin` and `count` must be multiple of StorageBitsPerElement
  template <typename Visitor>
  void traverseBits(Visitor visitor, size_t begin, size_t count) const;

  size_t countLeadingZeroes() const;
  size_t countSetBits() const;

private:
  size_t                            m_size = 0;
  std::unique_ptr<BitStorageType[]> m_bits;

  void   determineBitPosition(size_t index, size_t& element, size_t& bit) const;
  size_t determineNumberOfElements() const;

  // Clear the last unused bits in the last element.
  // Clear bits whose number is >= m_size. those are traversed unconditional and would produce invalid results.
  // restrict shifting range to 0 to StorageBitsPerElement - 1 to handle the case usedBitsInLastElement==0
  // which would result in shifting StorageBitsPerElement which is undefined by the standard and not the desired operation.
  void clearUnusedBits();

  // Set the last unused bits in the last element.
  // Set bits whose number is >= m_size. This is required when expanding the vector with the bits set to true.
  void setUnusedBits();
};

//////////////////////////////////////////////////////////////////////////
// inlined functions

//  Determine the element / bit for the given index
inline void BitArray::determineBitPosition(size_t index, size_t& element, size_t& bit) const
{
  assert(index < m_size);
  element = index / StorageBitsPerElement;
  bit     = index % StorageBitsPerElement;
}

inline size_t BitArray::determineNumberOfElements() const
{
  return (m_size + StorageBitsPerElement - 1) / StorageBitsPerElement;
}

inline void BitArray::enableBit(size_t index)
{
  size_t element;
  size_t bit;
  determineBitPosition(index, element, bit);
  m_bits[element] |= BitStorageType(1) << bit;
}

inline void BitArray::disableBit(size_t index)
{
  size_t element;
  size_t bit;
  determineBitPosition(index, element, bit);
  m_bits[element] &= ~(BitStorageType(1) << bit);
}

inline void BitArray::setBit(size_t index, bool value)
{
  size_t element;
  size_t bit;
  determineBitPosition(index, element, bit);

  if(value)
  {
    m_bits[element] |= BitStorageType(1) << bit;
  }
  else
  {
    m_bits[element] &= ~(BitStorageType(1) << bit);
  }
}

inline BitArray::BitStorageType const* BitArray::data() const
{
  return m_bits.get();
}

inline bool BitArray::getBit(size_t index) const
{
  size_t element;
  size_t bit;
  determineBitPosition(index, element, bit);
  return !!(m_bits[element] & (BitStorageType(1) << bit));
}

// call Visitor( size_t index ) on all bits which are set.
template <typename Visitor>
inline void BitArray::traverseBits(Visitor visitor) const
{
  bitTraverse(m_bits.get(), determineNumberOfElements(), visitor);
}

template <typename Visitor>
void BitArray::traverseBits(Visitor visitor, size_t begin, size_t count) const
{
  assert((begin % StorageBitsPerElement) == 0 && (count % StorageBitsPerElement) == 0);
  size_t elementStart = begin / StorageBitsPerElement;
  size_t elementCount = count / StorageBitsPerElement;

  if(elementCount)
  {
    assert((elementStart + elementCount) <= determineNumberOfElements());
    bitTraverse(m_bits.get() + elementStart, elementCount, visitor, begin);
  }
}

inline void BitArray::clearUnusedBits()
{
  if(m_size)
  {
    size_t usedBitsInLastElement = m_size % StorageBitsPerElement;
    m_bits[determineNumberOfElements() - 1] &=
        ~BitStorageType(0) >> ((StorageBitsPerElement - usedBitsInLastElement) & (StorageBitsPerElement - 1));
  }
}

inline void BitArray::setUnusedBits()
{
  if(m_size)
  {
    size_t usedBitsInLastElement = m_size % StorageBitsPerElement;
    m_bits[determineNumberOfElements() - 1] |= ~BitStorageType(0) << usedBitsInLastElement;
  }
}
}  // namespace nvutils
