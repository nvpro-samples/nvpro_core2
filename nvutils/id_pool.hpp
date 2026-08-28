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
//
// When to use this vs OffsetAllocator:
//   * IDPool: fixed slot-per-id workloads (bindless indices, descriptor slots,
//     mesh id ranges). Guarantees lowest-id / highest-id preference, provides
//     getUsedBounds for the dual-region use case, and uses ~1-2 bits per id
//     regardless of fill/fragmentation.
//   * OffsetAllocator (third_party/offsetallocator): general-purpose variable-
//     size offset allocator. Faster on pure range create/destroy at larger counts,
//     but stores a ~28-byte node per live allocation and does not guarantee
//     lowest-id.
//
// Implementation: a free-bitmask (1 == free, 64-bit words) plus a "longest free
// run" segment tree, using the "occupancy bitmask + bit scan" idea from
// OffsetAllocator extended to answer run-length queries in O(log). Earlier
// revisions were based on Emil Persson's MakeID (http://www.humus.name/3D/MakeID.h);
// the public behavior is preserved.
//
// Single-ID ops (count == 1) are specialized and never touch the run tree, which
// is allocated lazily on the first range op -- pure single-ID pools stay at
// ~1 bit/id.
//
// Per-id memory (poolSize a multiple of the tree-leaf size):
//   free bitmask       : 1     bit / id                (always allocated)
//   occupancy summaries: ~0.02 bit / id                (always allocated)
//   run-length tree    : 0.75  bit / id                (lazy: only after a range op)
//   -> total, single-ID only pool : ~1.02 bit / id     (tree never built)
//   -> total, range pool          : ~1.77 bit / id
class IDPool
{
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
  bool createID(uint32_t& id) { return createRangeID(id, 1); }

  // single ID from the back of the last available range
  bool createIDFromBack(uint32_t& id) { return createRangeIDFromBack(id, 1); }

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

  // may lazily build / sync the segment tree.
  bool isRangeAvailable(uint32_t searchCount);

  // number of elements in pool (highest id is `poolSize-1`)
  uint32_t getPoolSize() const { return m_poolSize; }
  // number of IDs currently in use
  uint32_t getUsedCount() const { return m_usedIDs; }

  // Splits the id space into a low ("front") region and a high ("back") region,
  // separated by the largest contiguous run of free IDs. Intended for the
  // dual-region use case where front and back allocators grow toward each other.
  //
  // The bounds reflect live extent, not origin: if a large gap opens within the
  // back region, the split point may land inside back-allocated space, causing
  // frontUsedEnd to cover some back-origin IDs or vice versa. This is expected.
  //
  // On return:
  //   frontUsedEnd  - exclusive upper bound of the front region: every in-use ID
  //                   below the gap is < frontUsedEnd.
  //   backUsedBegin - inclusive lower bound of the back region: every in-use ID
  //                   above the gap is >= backUsedBegin.
  // The half-open range [frontUsedEnd, backUsedBegin) is always free. An empty
  // pool yields (0, poolSize); a full pool yields (poolSize, poolSize).
  // May lazily sync the segment tree.
  void getUsedBounds(uint32_t& frontUsedEnd, uint32_t& backUsedBegin);

  void printRanges() const;
  void checkRanges();

private:
  // Segment-tree leaf granularity. Each leaf summarizes WORDS_PER_LEAF bitmask
  // words (WORDS_PER_LEAF*64 ids). 4 hits the speed/memory sweet spot across the
  // pool sizes and count ranges we care about (see id_pool.cpp for details).
  static constexpr uint32_t WORDS_PER_LEAF = 4;
  static constexpr uint32_t LEAF_IDS       = WORDS_PER_LEAF * 64;

  static constexpr uint32_t MAX_OCC_LEVELS = 7;  // 64^6 words > 2^32 ids

  // Longest-free-run summary for a span of ids (`len` is implicit from depth):
  //   pref = free run length at the low edge, suf = at the high edge,
  //   best = longest free run anywhere in the span.
  struct Node
  {
    uint32_t pref;
    uint32_t suf;
    uint32_t best;
  };

  uint64_t* m_free     = nullptr;  // m_capWords words, 1 == free (padding words are 0)
  uint32_t  m_cap      = 0;        // number of tree leaves (a power of two)
  uint32_t  m_capWords = 0;        // allocated bitmask words == m_cap * WORDS_PER_LEAF
  uint32_t  m_numWords = 0;        // number of words actually covering the pool
  uint32_t  m_poolSize = 0;        // number of ids
  uint32_t  m_usedIDs  = 0;        // number of ids in use

  // Occupancy hierarchy: level 0 is m_free; level k+1 has one bit per level-k word,
  // set iff that word holds a free id. Always current; used to find the lowest /
  // highest free id (count == 1) in O(levels) without the segment tree.
  uint64_t* m_occStore                 = nullptr;  // storage for summary levels 1..
  uint64_t* m_occ[MAX_OCC_LEVELS]      = {};       // m_occ[0] == m_free, m_occ[1..] are summaries
  uint32_t  m_occWords[MAX_OCC_LEVELS] = {};       // words at each level
  uint32_t  m_occLevels                = 0;

  // The segment tree is a lazily-maintained (and lazily-allocated) cache over
  // m_free: single-ID ops dirty words without updating it, and a range query
  // resyncs before reading it. Because these three fields mutate on read-shaped
  // queries (getUsedBounds, isRangeAvailable), those methods are non-const.
  Node*    m_tree    = nullptr;  // 2*m_cap nodes; index 1 == root, leaves at [m_cap, 2*m_cap)
  uint32_t m_dirtyLo = 0;        // dirtied word range [m_dirtyLo, m_dirtyHi];
  uint32_t m_dirtyHi = 0;        // clean when m_dirtyLo > m_dirtyHi

  static Node computeLeaf(const uint64_t* leafWords);  // summarize WORDS_PER_LEAF words
  static Node mergeNodes(const Node& a, const Node& b, uint32_t childLen);

  void     moveFrom(IDPool& other) noexcept;
  void     resetBits();
  void     buildTree();         // allocate + build the (lazy) segment tree
  void     rebuildTreeNodes();  // recompute all tree nodes from the bitmask
  void     bitmaskWrite(uint32_t word, uint64_t newVal);
  void     occPropagate(uint32_t word, bool becameNonEmpty);
  uint32_t occLowestFreeID() const;
  uint32_t occHighestFreeID() const;
  uint32_t childLenOf(uint32_t nodeIndex) const;
  void     markDirty(uint32_t firstWord, uint32_t lastWord);
  void     syncTree();
  void     updateWords(uint32_t firstWord, uint32_t lastWord);
  void     setBits(uint32_t id, uint32_t count, bool setFree);
  bool     anyFreeInRange(uint32_t id, uint32_t count) const;
  uint32_t descendLeftmost(uint32_t count) const;
  uint32_t descendRightmost(uint32_t count) const;
  uint32_t scanLeafLowest(uint32_t leaf, uint32_t count) const;   // lowest run>=count in a leaf's words
  uint32_t scanLeafHighest(uint32_t leaf, uint32_t count) const;  // highest run>=count in a leaf's words
};

}  // namespace nvutils
