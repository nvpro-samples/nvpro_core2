/*
 * Copyright (c) 2025-2026, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

// WORDS_PER_LEAF choice: measured on MSVC /O2, front-alloc churn with peak-
// skewed counts up to 256, pool sizes 64 KiB -> 16 Mi ids. The tradeoff is
// monotonic (smaller = faster per range op, larger = less tree memory); the
// value 4 lands ~15-25% faster than 8 for a modest memory hit and matches or
// beats every larger K across the range we care about. Count-1 ops never touch
// the tree so this choice is invisible for the bindless-index use case.

#include <bit>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "id_pool.hpp"

namespace nvutils {

//////////////////////////////////////////////////////////////////////////
// Local bit helpers operating on 64-bit "1 == free" words.

static constexpr uint64_t FULL = ~uint64_t(0);
static constexpr uint32_t NONE = 0xFFFFFFFFu;

// length of the longest run of set bits in v
static inline uint32_t longestRun(uint64_t v)
{
  uint32_t best = 0;
  while(v)
  {
    v >>= std::countr_zero(v);  // skip a run of used bits (v != 0, so shift < 64)
    uint32_t r = (uint32_t)std::countr_one(v);
    if(r > best)
      best = r;
    if(r >= 64)
      break;
    v >>= r;  // skip the run of free bits
  }
  return best;
}

// bit i set iff bits [i, i+count-1] are all set in v (count in [1,64])
static inline uint64_t consecutiveOnesMask(uint64_t v, uint32_t count)
{
  while(count > 1)
  {
    uint32_t s = count >> 1;
    v &= v >> s;
    count -= s;
  }
  return v;
}

static inline uint64_t maskFrom(uint32_t a)
{
  return FULL << a;  // bits [a,63]
}
static inline uint64_t maskTo(uint32_t b)
{
  return b == 63 ? FULL : ((uint64_t(1) << (b + 1)) - 1);  // bits [0,b]
}
static inline uint64_t maskRange(uint32_t a, uint32_t b)
{
  return maskFrom(a) & maskTo(b);
}

//////////////////////////////////////////////////////////////////////////

IDPool::Node IDPool::computeLeaf(const uint64_t* leafWords)
{
  auto ofWord = [](uint64_t v) {
    return IDPool::Node{(uint32_t)std::countr_one(v), (uint32_t)std::countl_one(v), longestRun(v)};
  };
  Node     leaf = ofWord(leafWords[0]);
  uint32_t len  = 64;
  for(uint32_t i = 1; i < WORDS_PER_LEAF; i++)  // sequentially merge the remaining words
  {
    Node     w     = ofWord(leafWords[i]);
    uint32_t cross = leaf.suf + w.pref;
    uint32_t m     = leaf.best > w.best ? leaf.best : w.best;
    Node     r;
    r.pref = (leaf.pref == len) ? len + w.pref : leaf.pref;  // whole left span free -> extend prefix
    r.suf  = (w.suf == 64) ? 64 + leaf.suf : w.suf;          // whole right word free -> extend suffix
    r.best = m > cross ? m : cross;
    leaf   = r;
    len += 64;
  }
  return leaf;
}

// combine two sibling summaries; childLen = ids covered by each child
IDPool::Node IDPool::mergeNodes(const Node& a, const Node& b, uint32_t childLen)
{
  Node r;
  r.pref         = (a.pref == childLen) ? childLen + b.pref : a.pref;
  r.suf          = (b.suf == childLen) ? childLen + a.suf : b.suf;
  uint32_t cross = a.suf + b.pref;
  uint32_t m     = a.best > b.best ? a.best : b.best;
  r.best         = m > cross ? m : cross;
  return r;
}

IDPool::IDPool(IDPool&& other) noexcept
{
  moveFrom(other);
}

IDPool& IDPool::operator=(IDPool&& other) noexcept
{
  if(this == &other)
    return *this;  // Handle self-assignment
  deinit();
  moveFrom(other);
  return *this;
}

void IDPool::moveFrom(IDPool& other) noexcept
{
  m_free      = other.m_free;
  m_tree      = other.m_tree;
  m_occStore  = other.m_occStore;
  m_cap       = other.m_cap;
  m_capWords  = other.m_capWords;
  m_numWords  = other.m_numWords;
  m_poolSize  = other.m_poolSize;
  m_usedIDs   = other.m_usedIDs;
  m_occLevels = other.m_occLevels;
  m_dirtyLo   = other.m_dirtyLo;
  m_dirtyHi   = other.m_dirtyHi;
  for(uint32_t k = 0; k < MAX_OCC_LEVELS; k++)
  {
    m_occ[k]      = other.m_occ[k];
    m_occWords[k] = other.m_occWords[k];
  }
  m_occ[0] = m_free;  // level 0 aliases our (moved) bitmask

  other.m_free      = nullptr;
  other.m_tree      = nullptr;
  other.m_occStore  = nullptr;
  other.m_cap       = 0;
  other.m_capWords  = 0;
  other.m_numWords  = 0;
  other.m_poolSize  = 0;
  other.m_usedIDs   = 0;
  other.m_occLevels = 0;
  other.m_dirtyLo   = 0;
  other.m_dirtyHi   = 0;
  for(uint32_t k = 0; k < MAX_OCC_LEVELS; k++)
  {
    other.m_occ[k]      = nullptr;
    other.m_occWords[k] = 0;
  }
}

void IDPool::init(const uint32_t poolSize)
{
  assert(!m_free && "init called multiple times");
  assert(poolSize);

  m_poolSize         = poolSize;
  m_usedIDs          = 0;
  m_numWords         = (poolSize + 63) >> 6;
  uint32_t numLeaves = (m_numWords + WORDS_PER_LEAF - 1) / WORDS_PER_LEAF;
  m_cap              = (uint32_t)std::bit_ceil(numLeaves);  // tree leaves (power of two)
  m_capWords         = m_cap * WORDS_PER_LEAF;              // allocated bitmask words

  m_free = static_cast<uint64_t*>(::malloc(size_t(m_capWords) * sizeof(uint64_t)));
  assert(m_free);  // Make sure allocation succeeded

  // The run-length segment tree is only needed for range operations (count > 1).
  // It is allocated lazily on the first createRangeID/...FromBack, so a pool used
  // purely for single IDs (plus getUsedBounds / isRangeAvailable, which scan the
  // bitmask directly) never pays for it -- ~1 bit/id instead of ~1.77 bit/id.
  m_tree = nullptr;

  // Occupancy hierarchy word counts: level 0 is m_free, each higher level packs
  // one bit per word of the level below until a single word remains.
  m_occWords[0] = m_capWords;
  m_occLevels   = 1;
  while(m_occWords[m_occLevels - 1] > 1)
  {
    m_occWords[m_occLevels] = (m_occWords[m_occLevels - 1] + 63) >> 6;
    m_occLevels++;
  }
  uint32_t occTotal = 0;
  for(uint32_t k = 1; k < m_occLevels; k++)
    occTotal += m_occWords[k];
  m_occStore   = occTotal ? static_cast<uint64_t*>(::malloc(size_t(occTotal) * sizeof(uint64_t))) : nullptr;
  m_occ[0]     = m_free;
  uint64_t* op = m_occStore;
  for(uint32_t k = 1; k < m_occLevels; k++)
  {
    m_occ[k] = op;
    op += m_occWords[k];
  }

  resetBits();
}

void IDPool::deinit()
{
  assert(!m_usedIDs && "not all IDs were destroyed");

  if(m_free)
  {
    ::free(m_free);
    ::free(m_tree);
    ::free(m_occStore);
    m_free      = nullptr;
    m_tree      = nullptr;
    m_occStore  = nullptr;
    m_cap       = 0;
    m_capWords  = 0;
    m_numWords  = 0;
    m_poolSize  = 0;
    m_usedIDs   = 0;
    m_occLevels = 0;
    for(uint32_t k = 0; k < MAX_OCC_LEVELS; k++)
    {
      m_occ[k]      = nullptr;
      m_occWords[k] = 0;
    }
  }
}

void IDPool::destroyAll()
{
  m_usedIDs = 0;
  resetBits();
}

// set every id in the pool to "free" and rebuild the summaries
void IDPool::resetBits()
{
  ::memset(m_free, 0, size_t(m_capWords) * sizeof(uint64_t));

  uint32_t fullWords = m_poolSize >> 6;
  for(uint32_t i = 0; i < fullWords; i++)
    m_free[i] = FULL;
  uint32_t rem = m_poolSize & 63;
  if(rem)
    m_free[fullWords] = (uint64_t(1) << rem) - 1;

  // Rebuild the occupancy summaries bottom-up (a bit set == that word has a free id).
  for(uint32_t k = 1; k < m_occLevels; k++)
  {
    ::memset(m_occ[k], 0, size_t(m_occWords[k]) * sizeof(uint64_t));
    for(uint32_t c = 0; c < m_occWords[k - 1]; c++)
      if(m_occ[k - 1][c] != 0)
        m_occ[k][c >> 6] |= uint64_t(1) << (c & 63);
  }

  m_dirtyLo = 0xFFFFFFFFu;  // clean (lo > hi)
  m_dirtyHi = 0;
  if(m_tree)
    rebuildTreeNodes();  // only if the tree has been allocated (a range was used)
}

// (re)compute every segment-tree node from the current bitmask; tree must exist
void IDPool::rebuildTreeNodes()
{
  for(uint32_t p = 0; p < m_cap; p++)
    m_tree[m_cap + p] = computeLeaf(&m_free[p * WORDS_PER_LEAF]);
  for(uint32_t i = m_cap - 1; i >= 1; i--)
    m_tree[i] = mergeNodes(m_tree[2 * i], m_tree[2 * i + 1], childLenOf(i));
  m_dirtyLo = 0xFFFFFFFFu;
  m_dirtyHi = 0;
}

// allocate the segment tree on first range use and build it from the bitmask
void IDPool::buildTree()
{
  m_tree = static_cast<Node*>(::malloc(size_t(2) * m_cap * sizeof(Node)));
  assert(m_tree);
  rebuildTreeNodes();
}

// write a word of the bitmask and keep the occupancy hierarchy in sync
void IDPool::bitmaskWrite(uint32_t word, uint64_t newVal)
{
  uint64_t old = m_free[word];
  if(newVal == old)
    return;
  m_free[word]  = newVal;
  bool wasEmpty = (old == 0);
  bool nowEmpty = (newVal == 0);
  if(wasEmpty != nowEmpty)
    occPropagate(word, wasEmpty /* became non-empty */);
}

// propagate a word's empty<->non-empty transition up the occupancy summaries
void IDPool::occPropagate(uint32_t word, bool becameNonEmpty)
{
  uint32_t idx = word;
  for(uint32_t k = 1; k < m_occLevels; k++)
  {
    uint32_t wk = idx >> 6;
    uint64_t b  = uint64_t(1) << (idx & 63);
    if(becameNonEmpty)
    {
      bool parentWasEmpty = (m_occ[k][wk] == 0);
      m_occ[k][wk] |= b;
      if(!parentWasEmpty)
        return;  // parent already flagged non-empty
    }
    else
    {
      m_occ[k][wk] &= ~b;
      if(m_occ[k][wk] != 0)
        return;  // parent still has other non-empty children
    }
    idx = wk;
  }
}

// lowest free id via the occupancy hierarchy, or NONE if the pool is full
uint32_t IDPool::occLowestFreeID() const
{
  uint32_t k = m_occLevels - 1;
  uint64_t v = m_occ[k][0];
  if(v == 0)
    return NONE;
  uint32_t word = (uint32_t)std::countr_zero(v);
  while(k > 0)
  {
    k--;
    v    = m_occ[k][word];
    word = word * 64 + (uint32_t)std::countr_zero(v);
  }
  return word;  // at level 0 this is the id
}

// highest free id via the occupancy hierarchy, or NONE if the pool is full
uint32_t IDPool::occHighestFreeID() const
{
  uint32_t k = m_occLevels - 1;
  uint64_t v = m_occ[k][0];
  if(v == 0)
    return NONE;
  uint32_t word = 63 - (uint32_t)std::countl_zero(v);
  while(k > 0)
  {
    k--;
    v    = m_occ[k][word];
    word = word * 64 + (63 - (uint32_t)std::countl_zero(v));
  }
  return word;
}

// remember that words [firstWord,lastWord] no longer agree with the lazy tree
void IDPool::markDirty(uint32_t firstWord, uint32_t lastWord)
{
  if(firstWord < m_dirtyLo)
    m_dirtyLo = firstWord;
  if(lastWord > m_dirtyHi)
    m_dirtyHi = lastWord;
}

// bring the lazy tree back in sync with the bitmask (no-op if nothing is dirty)
void IDPool::syncTree()
{
  if(m_dirtyLo > m_dirtyHi)
    return;
  updateWords(m_dirtyLo, m_dirtyHi);
  m_dirtyLo = 0xFFFFFFFFu;
  m_dirtyHi = 0;
}

// number of ids covered by each child of internal node `nodeIndex`
uint32_t IDPool::childLenOf(uint32_t nodeIndex) const
{
  return (m_cap / (uint32_t)std::bit_floor(nodeIndex) / 2) * LEAF_IDS;
}

// Refresh the leaves covering [firstWord,lastWord] and their ancestors. Each
// affected tree level is rebuilt exactly once.
void IDPool::updateWords(uint32_t firstWord, uint32_t lastWord)
{
  uint32_t firstLeaf = firstWord / WORDS_PER_LEAF;
  uint32_t lastLeaf  = lastWord / WORDS_PER_LEAF;

  // Common case (single leaf touched): walk the path to the root but stop as
  // soon as a node is unchanged.
  if(firstLeaf == lastLeaf)
  {
    uint32_t n = m_cap + firstLeaf;
    m_tree[n]  = computeLeaf(&m_free[firstLeaf * WORDS_PER_LEAF]);
    for(n >>= 1; n >= 1; n >>= 1)
    {
      Node  merged = mergeNodes(m_tree[2 * n], m_tree[2 * n + 1], childLenOf(n));
      Node& cur    = m_tree[n];
      if(merged.pref == cur.pref && merged.suf == cur.suf && merged.best == cur.best)
        break;
      cur = merged;
    }
    return;
  }

  for(uint32_t leaf = firstLeaf; leaf <= lastLeaf; leaf++)
    m_tree[m_cap + leaf] = computeLeaf(&m_free[leaf * WORDS_PER_LEAF]);

  uint32_t a = (m_cap + firstLeaf) >> 1;
  uint32_t b = (m_cap + lastLeaf) >> 1;
  while(a >= 1)
  {
    uint32_t childLen = childLenOf(a);  // constant across a whole tree level
    for(uint32_t i = a; i <= b; i++)
      m_tree[i] = mergeNodes(m_tree[2 * i], m_tree[2 * i + 1], childLen);
    if(a == 1)
      break;
    a >>= 1;
    b >>= 1;
  }
}

// mark [id, id+count) as used (setFree == false) or free (setFree == true).
// Used by the range (count > 1) paths, which update the tree eagerly so that
// range-heavy churn never accumulates a wide dirty span.
void IDPool::setBits(uint32_t id, uint32_t count, bool setFree)
{
  uint32_t first = id;
  uint32_t last  = id + count - 1;
  uint32_t fw    = first >> 6;
  uint32_t lw    = last >> 6;

  if(fw == lw)
  {
    uint64_t mask = maskRange(first & 63, last & 63);
    bitmaskWrite(fw, setFree ? (m_free[fw] | mask) : (m_free[fw] & ~mask));
    if(m_tree)
      updateWords(fw, fw);
    return;
  }

  uint64_t maskF = maskFrom(first & 63);
  uint64_t maskL = maskTo(last & 63);
  uint64_t mid   = setFree ? FULL : uint64_t(0);
  bitmaskWrite(fw, setFree ? (m_free[fw] | maskF) : (m_free[fw] & ~maskF));
  for(uint32_t w = fw + 1; w < lw; w++)
    bitmaskWrite(w, mid);
  bitmaskWrite(lw, setFree ? (m_free[lw] | maskL) : (m_free[lw] & ~maskL));
  if(m_tree)
    updateWords(fw, lw);
}

bool IDPool::anyFreeInRange(uint32_t id, uint32_t count) const
{
  uint32_t first = id, last = id + count - 1;
  uint32_t fw = first >> 6, lw = last >> 6;
  if(fw == lw)
    return (m_free[fw] & maskRange(first & 63, last & 63)) != 0;
  if(m_free[fw] & maskFrom(first & 63))
    return true;
  for(uint32_t w = fw + 1; w < lw; w++)
    if(m_free[w])
      return true;
  return (m_free[lw] & maskTo(last & 63)) != 0;
}

// lowest start of `count` consecutive free bits within a leaf's WORDS_PER_LEAF words
uint32_t IDPool::scanLeafLowest(uint32_t leaf, uint32_t count) const
{
  uint32_t base       = leaf * WORDS_PER_LEAF;
  uint32_t carry      = 0;  // free bits ending exactly at the end of the previous word
  uint32_t carryStart = 0;
  for(uint32_t w = base; w < base + WORDS_PER_LEAF; w++)
  {
    uint64_t v   = m_free[w];
    uint32_t low = (uint32_t)std::countr_one(v);
    if(carry && carry + low >= count)  // completes a run started in a previous word
      return carryStart;
    if(count <= 64)
    {
      uint64_t runs = consecutiveOnesMask(v, count);
      if(runs)
        return w * 64 + (uint32_t)std::countr_zero(runs);
    }
    uint32_t high = (uint32_t)std::countl_one(v);
    if(high == 64)
    {
      if(carry == 0)
      {
        carry      = 64;
        carryStart = w * 64;
      }
      else
        carry += 64;
    }
    else
    {
      carry      = high;
      carryStart = w * 64 + 64 - high;
    }
    if(carry >= count)
      return carryStart;
  }
  return NONE;  // unreachable when the leaf really holds a run >= count
}

// highest start of `count` consecutive free bits within a leaf's WORDS_PER_LEAF words
uint32_t IDPool::scanLeafHighest(uint32_t leaf, uint32_t count) const
{
  uint32_t base     = leaf * WORDS_PER_LEAF;
  uint32_t carry    = 0;  // free bits starting exactly at the start of the next word
  uint32_t carryEnd = 0;
  for(uint32_t w = base + WORDS_PER_LEAF; w-- > base;)
  {
    uint64_t v    = m_free[w];
    uint32_t high = (uint32_t)std::countl_one(v);
    if(carry && high + carry >= count)
      return carryEnd - count;
    if(count <= 64)
    {
      uint64_t runs = consecutiveOnesMask(v, count);
      if(runs)
        return w * 64 + (63 - (uint32_t)std::countl_zero(runs));
    }
    uint32_t low = (uint32_t)std::countr_one(v);
    if(low == 64)
    {
      if(carry == 0)
      {
        carry    = 64;
        carryEnd = w * 64 + 64;
      }
      else
        carry += 64;
    }
    else
    {
      carry    = low;
      carryEnd = w * 64 + low;
    }
    if(carry >= count)
      return carryEnd - count;
  }
  return NONE;
}

// lowest start of `count` consecutive free bits; caller guarantees root.best >= count
uint32_t IDPool::descendLeftmost(uint32_t count) const
{
  uint32_t node = 1, lo = 0, hi = m_cap - 1;  // leaf indices
  while(lo < hi)                              // not yet a leaf
  {
    uint32_t mid = (lo + hi) >> 1;
    uint32_t L = 2 * node, R = 2 * node + 1;
    if(m_tree[L].best >= count)  // lowest run lives entirely in the left child
    {
      node = L;
      hi   = mid;
      continue;
    }
    if(m_tree[L].suf + m_tree[R].pref >= count)  // it straddles the boundary
      return (mid + 1) * LEAF_IDS - m_tree[L].suf;
    node = R;  // otherwise it is in the right child
    lo   = mid + 1;
  }
  return scanLeafLowest(lo, count);  // exact position within the chosen leaf
}

// highest start of `count` consecutive free bits; caller guarantees root.best >= count
uint32_t IDPool::descendRightmost(uint32_t count) const
{
  uint32_t node = 1, lo = 0, hi = m_cap - 1;
  while(lo < hi)
  {
    uint32_t mid = (lo + hi) >> 1;
    uint32_t L = 2 * node, R = 2 * node + 1;
    if(m_tree[R].best >= count)  // highest run lives entirely in the right child
    {
      node = R;
      lo   = mid + 1;
      continue;
    }
    if(m_tree[L].suf + m_tree[R].pref >= count)  // it straddles the boundary
      return (mid + 1) * LEAF_IDS + m_tree[R].pref - count;
    node = L;
    hi   = mid;
  }
  return scanLeafHighest(lo, count);
}

bool IDPool::createRangeID(uint32_t& id, const uint32_t count)
{
  // Fast path: a single ID from the front. Occupancy hierarchy finds the lowest
  // free id; the segment tree is only dirtied, not updated.
  if(count == 1)
  {
    uint32_t lid = occLowestFreeID();
    if(lid == NONE)
      return false;  // pool full
    uint32_t w = lid >> 6;
    bitmaskWrite(w, m_free[w] & ~(uint64_t(1) << (lid & 63)));
    if(m_tree)
      markDirty(w, w);  // only track dirt when a tree exists to resync
    m_usedIDs++;
    id = lid;
    return true;
  }

  if(count == 0)
    return false;
  if(m_tree)
    syncTree();  // flush single-ID dirt into the existing tree
  else
    buildTree();  // first range use: allocate + build the tree
  if(m_tree[1].best < count)
    return false;  // No range of free IDs was large enough

  uint32_t start = descendLeftmost(count);
  setBits(start, count, false);
  m_usedIDs += count;
  id = start;
  return true;
}

bool IDPool::createRangeIDFromBack(uint32_t& id, const uint32_t count)
{
  // Fast path: a single ID from the back. Occupancy hierarchy finds the highest
  // free id; the segment tree is only dirtied, not updated.
  if(count == 1)
  {
    uint32_t hid = occHighestFreeID();
    if(hid == NONE)
      return false;  // pool full
    uint32_t w = hid >> 6;
    bitmaskWrite(w, m_free[w] & ~(uint64_t(1) << (hid & 63)));
    if(m_tree)
      markDirty(w, w);
    m_usedIDs++;
    id = hid;
    return true;
  }

  if(count == 0)
    return false;
  if(m_tree)
    syncTree();
  else
    buildTree();
  if(m_tree[1].best < count)
    return false;

  uint32_t start = descendRightmost(count);
  setBits(start, count, false);
  m_usedIDs += count;
  id = start;
  return true;
}

bool IDPool::destroyRangeID(const uint32_t id, const uint32_t count)
{
  if(count == 0)
    return false;

  assert(uint64_t(id) + count <= m_poolSize);
  if(uint64_t(id) + count > m_poolSize)
    return false;

  // Fast path: freeing a single ID is a bitmask bit set (with occupancy update);
  // the segment tree is only dirtied (destroy never needs to read it).
  if(count == 1)
  {
    uint32_t w   = id >> 6;
    uint64_t bit = uint64_t(1) << (id & 63);
    if(m_free[w] & bit)
      return false;  // already free -> invalid / double free
    bitmaskWrite(w, m_free[w] | bit);
    if(m_tree)
      markDirty(w, w);
    m_usedIDs--;
    return true;
  }

  if(anyFreeInRange(id, count))
    return false;  // Overlaps a range of free IDs, thus (at least partially) invalid IDs

  setBits(id, count, true);
  m_usedIDs -= count;
  return true;
}

bool IDPool::isRangeAvailable(uint32_t searchCount)
{
  if(searchCount <= 1)
    return m_poolSize - m_usedIDs >= searchCount;  // any free id satisfies a run of 0 or 1

  if(m_tree)
  {
    syncTree();
    return m_tree[1].best >= searchCount;
  }

  // No tree (single-ID pool): scan the bitmask for a run of searchCount free bits.
  uint32_t run = 0;
  for(uint32_t w = 0; w < m_numWords; w++)
  {
    uint64_t v = m_free[w];
    if(v == FULL)
    {
      run += 64;
      if(run >= searchCount)
        return true;
      continue;
    }
    if(v == 0)
    {
      run = 0;
      continue;
    }
    if(run + (uint32_t)std::countr_one(v) >= searchCount)  // continues the run before this word
      return true;
    if(longestRun(v) >= searchCount)  // a run wholly inside this word
      return true;
    run = (uint32_t)std::countl_one(v);  // trailing free bits continue into the next word
    if(run >= searchCount)
      return true;
  }
  return false;
}

void IDPool::getUsedBounds(uint32_t& frontUsedEnd, uint32_t& backUsedBegin)
{
  // No IDs in use: the whole pool is a single free gap.
  if(m_usedIDs == 0)
  {
    frontUsedEnd  = 0;
    backUsedBegin = m_poolSize;
    return;
  }

  // The widest free run is the divider between the front- and back-allocated
  // regions. Any smaller free holes remain inside one of the regions and are
  // simply scanned as empty slots by the caller.
  uint32_t bestLen = 0;
  uint32_t start   = 0;

  if(m_tree)
  {
    syncTree();
    bestLen = m_tree[1].best;
    if(bestLen)
      start = descendLeftmost(bestLen);  // lowest widest run, matching the tie-breaking below
  }
  else
  {
    // No tree (single-ID pool): find the widest free run by scanning the bitmask.
    // O(pool/64); intended for a handful of calls per frame, not per-op.
    uint32_t run = 0, runStart = 0;
    for(uint32_t w = 0; w < m_numWords; w++)
    {
      uint64_t v = m_free[w];
      if(v == 0)
      {
        run = 0;
        continue;
      }
      if(v == FULL)
      {
        if(run == 0)
          runStart = w << 6;
        run += 64;
        if(run > bestLen)
        {
          bestLen = run;
          start   = runStart;
        }
        continue;
      }
      for(uint32_t bit = 0; bit < 64; bit++)
      {
        uint32_t curId = (w << 6) + bit;
        if(curId >= m_poolSize)
          break;
        if((v >> bit) & 1)
        {
          if(run == 0)
            runStart = curId;
          run++;
          if(run > bestLen)
          {
            bestLen = run;
            start   = runStart;
          }
        }
        else
          run = 0;
      }
    }
  }

  // Pool full: no free gap. Treat the entire pool as the front region.
  if(bestLen == 0)
  {
    frontUsedEnd  = m_poolSize;
    backUsedBegin = m_poolSize;
    return;
  }

  frontUsedEnd  = start;
  backUsedBegin = start + bestLen;
}

void IDPool::printRanges() const
{
  bool     first    = true;
  uint32_t runStart = 0;
  bool     inRun    = false;

  auto flush = [&](uint32_t last) {
    if(!first)
      printf(", ");
    first = false;
    if(runStart == last)
      printf("%u", runStart);
    else
      printf("%u-%u", runStart, last);
  };

  for(uint32_t id = 0; id < m_poolSize; id++)
  {
    bool freeBit = (m_free[id >> 6] >> (id & 63)) & 1;
    if(freeBit)
    {
      if(!inRun)
      {
        inRun    = true;
        runStart = id;
      }
    }
    else if(inRun)
    {
      flush(id - 1);
      inRun = false;
    }
  }
  if(inRun)
    flush(m_poolSize - 1);
  if(first)
    printf("-");  // no free ids
  printf("\n");
}

void IDPool::checkRanges()
{
  // free bit count must match usedIDs, and the tree root's longest-run summary
  // must agree with a direct scan of the bitmask.
  uint32_t freeCount = 0;
  for(uint32_t w = 0; w < m_numWords; w++)
    freeCount += (uint32_t)std::popcount(m_free[w]);
  assert(m_poolSize - freeCount == m_usedIDs);

  uint32_t run = 0, best = 0;
  for(uint32_t id = 0; id < m_poolSize; id++)
  {
    if((m_free[id >> 6] >> (id & 63)) & 1)
    {
      run++;
      if(run > best)
        best = run;
    }
    else
      run = 0;
  }
  if(m_tree)
  {
    syncTree();
    assert(m_tree[1].best == best);
  }
}

}  // namespace nvutils

[[maybe_unused]] static void usage_IDPool()
{
  // let's allow up to 16-bit worth of textures
  nvutils::IDPool idGenTextures(1 << 16);

  uint32_t bindlessTextureID;
  idGenTextures.createID(bindlessTextureID);

  // use bindlessTextureID to fill a descriptor array element

  // when the texture is deleted, return the ID

  idGenTextures.destroyID(bindlessTextureID);


  // Imagine a scenario where we organize meshes in data-driven design in a flat
  // buffer. We may want to put all data that is static at the end, so we can
  // improve our buffer upload behavior. Now dynamic data is in the front of the buffer
  // and we will typically upload a smaller portion of the buffer per-frame.
  // If static and dynamic was mixed, we would have to use more sophisticated update
  // mechanisms that are optimized for scattering, otherwise we risk uploading a lot of data.
  nvutils::IDPool idGenMeshes(1 << 16);

  uint32_t meshBaseID;
  uint32_t meshCount    = 5;
  bool     meshIsStatic = true;

  if(meshIsStatic)
  {
    idGenMeshes.createRangeIDFromBack(meshBaseID, meshCount);
  }
  else
  {
    idGenMeshes.createRangeID(meshBaseID, meshCount);
  }
}
