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


#pragma once

#include <cstdint>
#include <cassert>
#include <bit>


namespace nvutils {

// Utility functions when working on bits

//  Call visitor(index + offset) for each bit set within `bits`
template <typename BitType, typename Visitor>
inline void bitTraverse(BitType bits, Visitor& visitor, size_t offset = 0)
{
  while(bits)
  {
    uint32_t localIndex = std::countr_zero(bits);
    visitor(offset + localIndex);
    bits ^= BitType(1) << localIndex;  // clear the current bit so that the next one is being found by the bitscan
  }
}

//  Call visitor(index + offset) for each bit set within `elements[]`
template <typename BitType, typename Visitor>
inline void bitTraverse(const BitType* elements, size_t numberOfElements, Visitor& visitor, size_t offset = 0)
{
  size_t baseIndex = 0;
  for(size_t elementIndex = 0; elementIndex < numberOfElements; ++elementIndex)
  {
    bitTraverse(elements[elementIndex], visitor, baseIndex + offset);
    baseIndex += sizeof(BitType) * 8;
  }
}
}  // namespace nvutils
