/*
 * Copyright (c) 2019-2026, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>

namespace nvutils {

// This class provides a way to create unique IDs out of a maximum pool.
// Useful to implement bindless texture index or similar allocators.
// It is possible to allocate IDs from the front, favoring the lowest available
// id values, or from the back, favoring the highest available values.
// With this dual region approach one could manage static and dynamic resources
// in a single flat buffer without requiring a fixed limit split.
class IDPool
{
  // Based on Emil Persson's MakeID
  // http://www.humus.name/3D/MakeID.h (v1.02)

public:
  IDPool() = default;

  // number of elements in pool
  // poolSize must be >= 1
  // highest id is `poolSize-1`
  IDPool(uint32_t poolSize) { init(poolSize); }

  IDPool(const IDPool& other)            = delete;
  IDPool& operator=(const IDPool& other) = delete;

  IDPool(IDPool&& other) noexcept;
  IDPool& operator=(IDPool&& other) noexcept;

  ~IDPool() { deinit(); }

  // number of elements in pool
  // poolSize must be >= 1
  // highest id is `poolSize-1`
  void init(const uint32_t poolSize);
  void deinit();

  // operations return true on success

  // single ID from the front of the first available range
  bool createID(uint32_t& id);

  // single ID from the back of the last available range
  bool createIDFromBack(uint32_t& id);

  // consecutive IDs starting at returned id, preferring low id values
  // count must be >= 1
  bool createRangeID(uint32_t& id, const uint32_t count);

  // consecutive IDs starting at returned id, preferring high id values
  // count must be >= 1
  bool createRangeIDFromBack(uint32_t& id, const uint32_t count);

  bool destroyID(const uint32_t id) { return destroyRangeID(id, 1); }
  // count must be >= 1
  bool destroyRangeID(const uint32_t id, const uint32_t count);
  void destroyAll();

  bool isRangeAvailable(uint32_t searchCount) const;

  void printRanges() const;
  void checkRanges() const;

private:
  struct Range
  {
    uint32_t first;
    uint32_t last;
  };

  Range*   m_ranges   = nullptr;  // Sorted array of ranges of free IDs
  uint32_t m_count    = 0;        // Number of ranges in list
  uint32_t m_capacity = 0;        // Total capacity of range list
  uint32_t m_maxID    = 0;        // Highest ID value
  uint32_t m_usedIDs  = 0;        // Number of IDs in use

  void insertRange(const uint32_t index);
  void destroyRange(const uint32_t index);
};

}  // namespace nvutils
