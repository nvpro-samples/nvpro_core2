/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*-------------------------------------------------------------------------------------------------
# nvdx12/shaders/bindless.h

> The HLSL side of the nvdx12 shader contract: the descriptor heap, push
> constants and push data of `nvdx12::createBindlessRootSignature`, and the
> accessors over nvdx12 shader addresses.

```hlsl
#include <nvdx12/shaders/bindless.h>
```

The compiler needs `-HV 2021`, Shader Model 6.6 or newer, and the
nvpro_core2 root (or a folder holding a copy of nvdx12/shaders) on its
include path: `nvdx12::ShaderCompiler` takes the application's search
paths, `BuiltinKernels` adds the nvpro_core2 root and the executable folder.

Contents:
- `HEAP_SRV(slot)` / `HEAP_UAV(slot)`: the two views of an image pair,
  `slot` being `nvdx12::ImageSlot::srv` (HEAP_UAV reads `srv + 1`);
  `HEAP_SAMPLER(slot)`: a sampler slot. Single-slot views
  (writeStorageImageDescriptor, buffer UAVs) are `ResourceDescriptorHeap[slot]`.
- `PUSH_CONSTANTS(T, name)`: `name` is the struct pushed with
  `nvdx12::cmdPushConstants`, the root constants `b0` of
  `nvdx12::createBindlessRootSignature` (up to 240 bytes).
- `PUSH_DATA(T, name)`: `name` is the pushed struct, read from the root SRV
  `t0` of the same root signature (`nvdx12::UploadRing::cmdPushData`
  writes it), for data beyond the root constants.
- Shader addresses. `nvdx12::Buffer::address` encodes
  `(heapSlotOfRawUav << 32) | byteOffset`. `addrSlot` and `addrOffset` split
  it; `addrLoad`, `addrStore`, `addrLoadAt`, `addrStoreAt`, `addrLoadU8` and
  the `addrAtomic*` functions run on the raw UAV the address names.
  - A buffer above 4 GiB has one raw UAV per 4 GiB window on consecutive
    slots: 64-bit address arithmetic (`addrElem`, `Buffer::addressAt`)
    carries into the next window; the 32-bit offsets the accessors add to
    an address stay in its window.
  - `ADDR_GLOBALLY_COHERENT`: defined before the include, every accessor
    uses a `globallycoherent` raw UAV, for shaders that spin on values
    other thread groups write.
- `ADDR_TLAS(handle)`: the acceleration structure behind
  `nvdx12::AccelerationStructure::shaderHandle()`.
-------------------------------------------------------------------------------------------------*/

#ifndef NVDX12_BINDLESS_H
#define NVDX12_BINDLESS_H

// ---------------------------------------------------------------------------
// Descriptor heap access
// ---------------------------------------------------------------------------
// An image on the heap is a pair (nvdx12::ImageSlot): SRV at `slot`, UAV at
// `slot + 1`; shaders receive the SRV index. The indices are tagged
// non-uniform: a uniform index is a subset in spec terms and the tag is
// typically cheap where the index is uniform, which saves per-call-site
// bookkeeping.
#define HEAP_SRV(_slot) ResourceDescriptorHeap[NonUniformResourceIndex(_slot)]
#define HEAP_UAV(_slot) ResourceDescriptorHeap[NonUniformResourceIndex((_slot) + 1u)]
#define HEAP_SAMPLER(_slot) SamplerDescriptorHeap[NonUniformResourceIndex(_slot)]

// ---------------------------------------------------------------------------
// Push constants (root constants b0 of the bindless root signature)
// ---------------------------------------------------------------------------
// Declares `name` as the struct pushed with nvdx12::cmdPushConstants.
// Constant-buffer packing applies: scalars, vectors and 64-bit addresses
// match the C++ layout; arrays are strided to 16 bytes and a vector may not
// straddle a 16-byte boundary. PUSH_DATA below reads natural layout and has
// no size limit beyond the ring.
#define PUSH_CONSTANTS(T, name) ConstantBuffer<T> name : register(b0, space0)

// ---------------------------------------------------------------------------
// Push data (root SRV t0 of the bindless root signature)
// ---------------------------------------------------------------------------
ByteAddressBuffer g_pushData : register(t0, space0);

// Declares `name` as the pushed struct. Static const globals are
// initialized at shader start, before any entry point code runs. In
// ray-tracing stages the copy stays live across every TraceRay: there,
// `#define name g_pushData.Load<T>(0)` loads at each use instead.
#define PUSH_DATA(T, name) static const T name = g_pushData.Load<T>(0)

// ---------------------------------------------------------------------------
// Shader addresses: (heapSlot << 32) | byteOffset over raw UAVs
// ---------------------------------------------------------------------------
// Heap slot of the raw UAV behind `addr`.
uint addrSlot(uint64_t addr)
{
  return uint(addr >> 32);
}
uint addrOffset(uint64_t addr)
{
  return uint(addr);
}

// Raw buffer type of every accessor; ADDR_GLOBALLY_COHERENT makes the loads
// bypass the non-coherent caches.
#ifdef ADDR_GLOBALLY_COHERENT
#define ADDR_RWBUF globallycoherent RWByteAddressBuffer
#else
#define ADDR_RWBUF RWByteAddressBuffer
#endif

// Runs the statements on the buffer `b` the address names.
#define ADDR_ON_BUFFER(_addr, ...)                                                                                     \
  {                                                                                                                    \
    ADDR_RWBUF b = ResourceDescriptorHeap[NonUniformResourceIndex(addrSlot(_addr))];                                   \
    __VA_ARGS__                                                                                                        \
  }

// Element `idx` of an array of T starting at `addr`.
template <typename T>
T addrLoad(uint64_t addr, uint idx)
{
  const uint off = addrOffset(addr) + idx * sizeof(T);
  T          r;
  ADDR_ON_BUFFER(addr, r = b.Load<T>(off);)
  return r;
}
template <typename T>
T addrLoad(uint64_t addr)
{
  const uint off = addrOffset(addr);
  T          r;
  ADDR_ON_BUFFER(addr, r = b.Load<T>(off);)
  return r;
}
// Byte-offset variant for struct members: addr + offsetof(member).
template <typename T>
T addrLoadAt(uint64_t addr, uint byteOffset)
{
  const uint off = addrOffset(addr) + byteOffset;
  T          r;
  ADDR_ON_BUFFER(addr, r = b.Load<T>(off);)
  return r;
}
template <typename T>
void addrStore(uint64_t addr, uint idx, T value)
{
  const uint off = addrOffset(addr) + idx * sizeof(T);
  ADDR_ON_BUFFER(addr, b.Store<T>(off, value);)
}
template <typename T>
void addrStoreAt(uint64_t addr, uint byteOffset, T value)
{
  const uint off = addrOffset(addr) + byteOffset;
  ADDR_ON_BUFFER(addr, b.Store<T>(off, value);)
}
// Address of element `idx` of an array of T starting at `_addr`, in 64 bits so
// that it carries into the next 4 GiB window.
#define addrElem(T, _addr, _idx) ((_addr) + uint64_t(_idx) * sizeof(T))

// Byte `idx` of the array starting at `addr`.
uint addrLoadU8(uint64_t addr, uint idx)
{
  const uint byteAddr = addrOffset(addr) + idx;
  uint       word;
  ADDR_ON_BUFFER(addr, word = b.Load(byteAddr & ~3u);)
  return (word >> ((byteAddr & 3u) * 8u)) & 0xFFu;
}

// Atomics at a byte offset relative to `addr`; each returns the previous value.
uint addrAtomicAdd(uint64_t addr, uint byteOffset, uint value)
{
  const uint off = addrOffset(addr) + byteOffset;
  uint       orig;
  ADDR_ON_BUFFER(addr, b.InterlockedAdd(off, value, orig);)
  return orig;
}
int addrAtomicAdd(uint64_t addr, uint byteOffset, int value)
{
  const uint off = addrOffset(addr) + byteOffset;
  int        orig;
  ADDR_ON_BUFFER(addr, b.InterlockedAdd(off, value, orig);)
  return orig;
}
uint addrAtomicMax(uint64_t addr, uint byteOffset, uint value)
{
  const uint off = addrOffset(addr) + byteOffset;
  uint       orig;
  ADDR_ON_BUFFER(addr, b.InterlockedMax(off, value, orig);)
  return orig;
}
uint addrAtomicOr(uint64_t addr, uint byteOffset, uint value)
{
  const uint off = addrOffset(addr) + byteOffset;
  uint       orig;
  ADDR_ON_BUFFER(addr, b.InterlockedOr(off, value, orig);)
  return orig;
}
uint addrAtomicXor(uint64_t addr, uint byteOffset, uint value)
{
  const uint off = addrOffset(addr) + byteOffset;
  uint       orig;
  ADDR_ON_BUFFER(addr, b.InterlockedXor(off, value, orig);)
  return orig;
}
uint addrAtomicExchange(uint64_t addr, uint byteOffset, uint value)
{
  const uint off = addrOffset(addr) + byteOffset;
  uint       orig;
  ADDR_ON_BUFFER(addr, b.InterlockedExchange(off, value, orig);)
  return orig;
}
uint addrAtomicCompSwap(uint64_t addr, uint byteOffset, uint compare, uint value)
{
  const uint off = addrOffset(addr) + byteOffset;
  uint       orig;
  ADDR_ON_BUFFER(addr, b.InterlockedCompareExchange(off, compare, value, orig);)
  return orig;
}
uint64_t addrAtomicAdd64(uint64_t addr, uint byteOffset, uint64_t value)
{
  const uint off = addrOffset(addr) + byteOffset;
  uint64_t   orig;
  ADDR_ON_BUFFER(addr, b.InterlockedAdd64(off, value, orig);)
  return orig;
}
uint64_t addrAtomicCompSwap64(uint64_t addr, uint byteOffset, uint64_t compare, uint64_t value)
{
  const uint off = addrOffset(addr) + byteOffset;
  uint64_t   orig;
  ADDR_ON_BUFFER(addr, b.InterlockedCompareExchange64(off, compare, value, orig);)
  return orig;
}
// Float atomic add through a compare-exchange loop.
float addrAtomicAddFloat(uint64_t addr, uint byteOffset, float value)
{
  const uint off = addrOffset(addr) + byteOffset;
  uint       orig;
  ADDR_ON_BUFFER(addr, orig = b.Load(off);)
  [loop] for(;;)
  {
    const uint desired = asuint(asfloat(orig) + value);
    uint       seen;
    ADDR_ON_BUFFER(addr, b.InterlockedCompareExchange(off, orig, desired, seen);)
    if(seen == orig)
      return asfloat(orig);
    orig = seen;
  }
}

// Acceleration structure behind a shader handle (TLAS / PTLAS SRV slot << 32).
#define ADDR_TLAS(_handle)                                                                                             \
  ((RaytracingAccelerationStructure)ResourceDescriptorHeap[NonUniformResourceIndex(uint((_handle) >> 32))])

#endif  // NVDX12_BINDLESS_H
