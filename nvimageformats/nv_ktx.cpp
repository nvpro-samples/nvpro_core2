/*
 * Copyright (c) 2021-2026, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "nv_ktx.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>  // Some functions produce assertion errors to assist with debugging when NDEBUG is false.
#include <fstream>
#include <limits>
#include <mutex>
#include <sstream>
#include <string.h>  // memcpy
#include <vulkan/vulkan_core.h>
#ifdef NVP_SUPPORTS_ZSTD
#include <zstd.h>
#endif
#ifdef NVP_SUPPORTS_GZLIB
#include <zlib.h>
#endif
#ifdef NVP_SUPPORTS_BASISU
#include <basisu_comp.h>
#include <basisu_transcoder.h>
#if BASISD_LIB_VERSION < 160
#error When nv_ktx is compiled with Basis Universal support, it requires Basis Universal version 1.60 or higher, due to a change in the variable names for ASTC 4x4 quality.
#endif
#endif

#include "third_party/khr_df/khr_df.h"
#include "texture_formats.h"

namespace nv_ktx {

// Some sources for this code:
// KTX 1 specification at https://github.com/KhronosGroup/KTX-Specification/releases/tag/1.0-final
// KTX 2 specification at https://github.khronos.org/KTX-Specification/
// Khronos Data Format Specification 1.3.1 at https://www.khronos.org/registry/DataFormat/specs/1.3/dataformat.1.3.html
//   (especially chapters 3-5)

//-----------------------------------------------------------------------------
// SHARED KTX1 + KTX2 FUNCTIONS
//-----------------------------------------------------------------------------

const size_t  IDENTIFIER_LEN                 = 12;
const uint8_t ktx1Identifier[IDENTIFIER_LEN] = {0xAB, 0x4B, 0x54, 0x58, 0x20, 0x31, 0x31, 0xBB, 0x0D, 0x0A, 0x1A, 0x0A};
const uint8_t ktx2Identifier[IDENTIFIER_LEN] = {0xAB, 0x4B, 0x54, 0x58, 0x20, 0x32, 0x30, 0xBB, 0x0D, 0x0A, 0x1A, 0x0A};

namespace {
// Resizing a vector can produce an exception if the allocation fails, but
// there's no real way to validate this ahead of time. So we use
// a try/catch block.
template <class T>
ErrorWithText resizeVectorOrError(std::vector<T>& vec, size_t newSize)
{
  try
  {
    vec.resize(newSize);
  }
  catch(...)
  {
    return "Allocating " + std::to_string(newSize) + " bytes of data failed.";
  }
  return {};
}

// Multiplies three values, returning false if the calculation would overflow,
// interpreting each value as 1 if it would be 0.
bool getNumSubresources(size_t a, size_t b, size_t c, size_t& out)
{
  return checked_math::mul3(std::max(a, size_t(1)), std::max(b, size_t(1)), std::max(c, size_t(1)), out);
}

// Supports an std::istream interface that operates on an in-memory array.
// Same as in nv_dds.cpp.
class MemoryStreamBuffer : public std::basic_streambuf<char>
{
  char*           m_data        = nullptr;
  std::streamoff  m_nextIndex   = 0;  // Always in [0, m_sizeInBytes].
  std::streamsize m_sizeInBytes = 0;

public:
  MemoryStreamBuffer(char* data, std::streamsize sizeInBytes)
      : m_data(data)
      , m_sizeInBytes(sizeInBytes)
  {
  }
  pos_type seekoff(off_type off, std::ios_base::seekdir dir, std::ios_base::openmode which = std::ios_base::in | std::ios_base::out) override
  {
    const pos_type indicatesError = pos_type(static_cast<off_type>(-1));
    switch(dir)
    {
      case std::ios_base::beg:
        if(off < 0 || off > m_sizeInBytes)
          return indicatesError;
        m_nextIndex = off;
        break;
      case std::ios_base::cur:
        if(((off < 0) && m_nextIndex + off < 0) || (off >= 0 && off > m_sizeInBytes - m_nextIndex))
          return indicatesError;
        m_nextIndex += off;
        break;
      case std::ios_base::end:
        if(off > 0 || off < -m_sizeInBytes)
          return indicatesError;
        m_nextIndex = m_sizeInBytes + off;
        break;
      default:
        return indicatesError;
    }
    return m_nextIndex;
  }
  pos_type seekpos(pos_type pos, std::ios_base::openmode which = std::ios_base::in | std::ios_base::out) override
  {
    return seekoff(static_cast<off_type>(pos), std::ios_base::beg, which);
  }
  // Gets the number of characters certainly available.
  std::streamsize showmanyc() override { return m_sizeInBytes - m_nextIndex; }
  // Gets the next character advancing the read pointer; EOF on error.
  int_type uflow() override
  {
    if(m_nextIndex < m_sizeInBytes)
    {
      return traits_type::to_int_type(m_data[m_nextIndex++]);
    }
    return traits_type::eof();
  }
  // Gets the next character without advancing the read pointer; EOF on error.
  int_type underflow() override
  {
    if(m_nextIndex < m_sizeInBytes)
    {
      return traits_type::to_int_type(m_data[m_nextIndex]);
    }
    return traits_type::eof();
  }
  // Reads a given number of characters and stores them into s' character array.
  // Returns the number of characters successfully read.
  std::streamsize xsgetn(char_type* s, std::streamsize count) override
  {
    if(count < 0 || s == nullptr || m_nextIndex >= m_sizeInBytes)
    {
      return 0;
    }
    const std::streamsize readableChars = std::min(count, m_sizeInBytes - m_nextIndex);
    memcpy(s, m_data + m_nextIndex, readableChars);
    m_nextIndex += readableChars;
    return readableChars;
  }
};

// An std::istream interface for a constant, in-memory array.
// Note that Clang emits a Wreorder-ctor warning unless the memory buffer
// members are listed before the istream, so we use multiple inheritance
// here to put them in the right order.
class MemoryStream : private MemoryStreamBuffer, public std::istream
{
public:
  MemoryStream(const char* data, std::streamsize sizeInBytes)
      : MemoryStreamBuffer(const_cast<char*>(data), sizeInBytes)
      , std::istream(this)
  {
    rdbuf(this);
  }
};
}  // namespace

ErrorWithText Image::allocate(uint32_t _numMips, uint32_t _numLayers, uint32_t _numFaces)
{
  clear();

  numMips            = _numMips;
  numLayersPossibly0 = _numLayers;
  numFaces           = _numFaces;

  size_t numSubresources = 0;
  if(!getNumSubresources(numMips, numLayersPossibly0, numFaces, numSubresources))
  {
    return "Computing the required number of subresources overflowed a size_t!";
  }
  return resizeVectorOrError(m_data, numSubresources);
}

void Image::clear()
{
  m_data.clear();
}

std::vector<char>& Image::subresource(uint32_t mip, uint32_t layer, uint32_t face)
{
  const uint32_t numMipsClamped   = std::max(numMips, 1U);
  const uint32_t numLayersClamped = std::max(numLayersPossibly0, 1U);
  if(mip >= numMipsClamped || layer >= numLayersClamped || face >= numFaces)
  {
    throw std::out_of_range("Image::subresource values were out of range");
  }

  // Here's the layout for data that we use. Note that we store the lowest mips
  // (mip 0) first, while the KTX format stores the highest mips first.
  return m_data[(size_t(mip) * size_t(numLayersClamped) + size_t(layer)) * size_t(numFaces) + size_t(face)];
}

VkImageType Image::getImageType() const
{
  if(mip0Width == 0 || mip0Height == 0)
  {
    return VK_IMAGE_TYPE_1D;
  }
  else if(mip0Depth == 0)
  {
    return VK_IMAGE_TYPE_2D;
  }
  else
  {
    return VK_IMAGE_TYPE_3D;
  }
}

VkImageViewType Image::getImageViewType() const
{
  const bool isArray = (numLayersPossibly0 > 0);
  if(mip0Width == 0 || mip0Height == 0)  // 1D
  {
    return isArray ? VK_IMAGE_VIEW_TYPE_1D_ARRAY : VK_IMAGE_VIEW_TYPE_1D;
  }
  else if(mip0Depth == 0)  // 2D
  {
    if(numFaces > 1)
    {
      return isArray ? VK_IMAGE_VIEW_TYPE_CUBE_ARRAY : VK_IMAGE_VIEW_TYPE_CUBE;
    }
    else
    {
      return isArray ? VK_IMAGE_VIEW_TYPE_2D_ARRAY : VK_IMAGE_VIEW_TYPE_2D;
    }
  }
  else  // 3D
  {
    return VK_IMAGE_VIEW_TYPE_3D;
  }
}

bool Image::requiresComplexDecoding() const
{
  return m_fileInfo.ktx1NeedsEndianSwap                            // Requires endian swapping
         || m_fileInfo.ktx2SupercompressionScheme != 0             // Requires inflation
         || (m_fileInfo.ktx2ColorModel == KHR_DF_MODEL_ETC1S       //
             || m_fileInfo.ktx2ColorModel == KHR_DF_MODEL_UASTC);  // Requires transcoding
}

const SubresourceLayout& Image::getSubresourceLayout(uint32_t mip, uint32_t layer, uint32_t face) const
{
  return const_cast<Image*>(this)->subresourceLayout(mip, layer, face);
}

SubresourceLayout& Image::subresourceLayout(uint32_t mip, uint32_t layer, uint32_t face)
{
  return m_subresourceLayouts[(size_t(mip) * std::max(1u, numLayersPossibly0) + layer) * numFaces + face];
}

size_t Image::getSubresourceByteSizeSum(const SubresourceRange& range) const
{
  size_t       sum                = 0;
  const size_t subresourcesPerMip = size_t(range.numLayers) * range.numFaces;
  for(uint32_t mip = range.firstMip; mip < range.firstMip + range.numMips; mip++)
  {
    // All subresources for a given mip have the same size, so we can do this:
    sum += getSubresourceByteSize(mip) * subresourcesPerMip;
  }
  return sum;
}

size_t Image::getMipByteSizeSum(uint32_t mip) const
{
  return size_t(getSubresourceByteSize(mip)) * std::max(1u, numLayersPossibly0) * numFaces;
}

// Macro for "read this variable from the istream; if it fails, return an error message"
#define READ_OR_ERROR(input, variable, error_message)                                                                  \
  if(!(input).read(reinterpret_cast<char*>(&(variable)), sizeof(variable)))                                            \
  {                                                                                                                    \
    return (error_message);                                                                                            \
  }

// Macro for "If this returned an error, propagate that error"
#define UNWRAP_ERROR(expr_returning_error_with_text)                                                                   \
  if(ErrorWithText unwrap_error_tmp = (expr_returning_error_with_text))                                                \
  {                                                                                                                    \
    return unwrap_error_tmp;                                                                                           \
  }

namespace {
size_t roundUp(size_t value, size_t multiplier)
{
  const size_t mod = value % multiplier;
  if(mod == 0)
    return value;
  return value + (multiplier - mod);
}

// Basic Data Format Descriptor from the Khronos Data Format,
// without sample information.
struct BasicDataFormatDescriptor
{
  uint32_t vendorId : 17;
  uint32_t descriptorType : 15;
  uint16_t versionNumber : 16;
  uint16_t descriptorBlockSize : 16;
  uint8_t  colorModel : 8;
  uint8_t  colorPrimaries : 8;
  uint8_t  transferFunction : 8;
  uint8_t  flags : 8;
  uint8_t  texelBlockDimension0 : 8;
  uint8_t  texelBlockDimension1 : 8;
  uint8_t  texelBlockDimension2 : 8;
  uint8_t  texelBlockDimension3 : 8;
  uint8_t  bytesPlane0 : 8;
  uint8_t  bytesPlane1 : 8;
  uint8_t  bytesPlane2 : 8;
  uint8_t  bytesPlane3 : 8;
  uint8_t  bytesPlane4 : 8;
  uint8_t  bytesPlane5 : 8;
  uint8_t  bytesPlane6 : 8;
  uint8_t  bytesPlane7 : 8;
};
static_assert(sizeof(BasicDataFormatDescriptor) == 24, "Basic data format descriptor size must match the KDF spec!");

struct DFSample
{
  uint16_t bitOffset;
  uint8_t  bitLength;
  uint8_t  channelType;
  uint8_t  samplePosition0;
  uint8_t  samplePosition1;
  uint8_t  samplePosition2;
  uint8_t  samplePosition3;
  uint32_t lower;
  uint32_t upper;
};
static_assert(sizeof(DFSample) == 16, "Basic data format descriptor sample type size must match Khronos Data Format spec!");

// Interprets T as an array of uint32_ts and swaps the endianness of each element.
template <class T>
void swapEndian32(T& data)
{
  static_assert((sizeof(T) % 4) == 0, "T must be interpretable as an array of 32-bit words.");
  size_t    numInts           = sizeof(T) / 4;
  uint32_t* reinterpretedData = reinterpret_cast<uint32_t*>(&data);
  for(size_t i = 0; i < numInts; i++)
  {
    uint32_t value = reinterpretedData[i];
    value          = ((value & 0x000000FFu) << 24)  //
            | ((value & 0x0000FF00u) << 8)          //
            | ((value & 0x00FF0000u) >> 8)          //
            | ((value & 0xFF000000u) >> 24);
    reinterpretedData[i] = value;
  }
}

// Interprets data, an array dataSizeBytes long, as an array of elements of size
// typeSizeBytes. Then swaps the endianness of each element.
void swapEndianGeneral(size_t dataSizeBytes, void* data, uint32_t typeSizeBytes)
{
  if(typeSizeBytes == 0 || typeSizeBytes == 1)
  {
    return;  // Nothing to do
  }
  assert(dataSizeBytes % typeSizeBytes == 0);  // Otherwise we'll swap all but the last element
  uint8_t*     dataAsBytes = reinterpret_cast<uint8_t*>(data);
  const size_t typeSize64  = size_t(typeSizeBytes);
  const size_t numElements = dataSizeBytes / typeSizeBytes;  // e.g. 5/3 -> 1 element
  const size_t numSwaps    = typeSize64 / 2;                 // e.g. 3 bytes -> 1 swap

  for(size_t eltIdx = 0; eltIdx < numElements; eltIdx++)
  {
    for(size_t swapIdx = 0; swapIdx < numSwaps; swapIdx++)
    {
      std::swap(dataAsBytes[eltIdx * typeSize64 + swapIdx], dataAsBytes[eltIdx * typeSize64 + typeSize64 - 1 - swapIdx]);
    }
  }
}

static_assert(CHAR_BIT == 8, "Things will probably go wrong in nv_ktx code with istream reads if chars aren't 8 bits.");

ErrorWithText readKeyValueData(std::istream&                             input,
                               uint32_t                                  kvdByteLength,
                               bool                                      srcIsBigEndian,
                               std::map<std::string, std::vector<char>>& outKeyValueData)
{
  std::vector<char> kvBlock;
  UNWRAP_ERROR(resizeVectorOrError(kvBlock, kvdByteLength));
  if(!input.read(kvBlock.data(), size_t(kvdByteLength)))
  {
    return "Unable to read " + std::to_string(kvdByteLength) + " bytes of KTX2 key/value data.";
  }

  size_t byteIndex = 0;
  while(byteIndex < kvdByteLength)
  {
    // Read keyAndValueByteLength
    uint32_t keyAndValueByteLength = 0;
    // Check to make sure we don't read out of bounds
    if(byteIndex + sizeof(keyAndValueByteLength) >= kvBlock.size())
    {
      return "Key/value data starting at byte " + std::to_string(byteIndex)
             + "of the key/value data block did not have enough space to contain the 32-bit key/value size. Is the key/value data truncated?";
    }
    memcpy(&keyAndValueByteLength, &kvBlock[byteIndex], sizeof(keyAndValueByteLength));
    if(srcIsBigEndian)
    {
      swapEndian32<uint32_t>(keyAndValueByteLength);
    }
    byteIndex += sizeof(keyAndValueByteLength);

    // If byteIndex + keyAndValueByteLength > the length of kvBlock, we read
    // byteIndex incorrectly; we'll treat this as a non-fatal error.
    if(byteIndex + keyAndValueByteLength > kvBlock.size())
    {
      assert("Key/value data had byte length that was too long!");
      return {};
    }

    // Parse the key and value according to 3.11.2 "keyAndValue".
    // First, find the index of the NUL character terminating the key.
    size_t keyLength = 0;
    while(keyLength < size_t(keyAndValueByteLength))
    {
      if(kvBlock[byteIndex + keyLength] == '\0')
      {
        break;  // Found the NUL character!
      }
      keyLength++;  // Next character
    }

    // If we somehow reached the end without finding the NUL character, this
    // key is invalid - we could read it correctly, but others might not,
    // and we have to copy the keys to the output anyways.
    if(keyLength == size_t(keyAndValueByteLength))
    {
      return "Key starting at byte " + std::to_string(byteIndex) + " of the key/value data did not have a NUL character.";
    }

    // Construct the key and value from ranges.
    std::string key(&kvBlock[byteIndex], keyLength);  // Don't include null character
    std::vector<char> value(kvBlock.begin() + (byteIndex + keyLength + 1), kvBlock.begin() + (byteIndex + keyAndValueByteLength));
    // Handle duplicate keys gracefully by using later keys. Note that KTX
    // requires that keys not be duplicated.
    outKeyValueData.insert_or_assign(key, value);

    // Skip directly to the next key, including padding.
    byteIndex += roundUp(size_t(keyAndValueByteLength), 4);
  }

  return {};
}

// Computes the size of a subresource of size `width` x `height` x `depth`, encoded
// using ASTC blocks of size `blockWidth` x `blockHeight` x `blockDepth`. Returns false
// if the calculation would overflow, and returns true and stores the result in
// `out` otherwise.
bool astcSize(size_t blockWidth, size_t blockHeight, size_t blockDepth, size_t width, size_t height, size_t depth, size_t& out)
{
  return checked_math::mul4(((width + blockWidth - 1) / blockWidth),     // # of ASTC blocks along the x axis
                            ((height + blockHeight - 1) / blockHeight),  // # of ASTC blocks along the y axis
                            ((depth + blockDepth - 1) / blockDepth),     // # of ASTC blocks along the z axis
                            16,                                          // Each ASTC block size is 128 bits = 16 bytes
                            out);
}

// Returns the size of a width x height x depth image of the given VkFormat.
// Returns an error if the given image sizes are strictly invalid.
ErrorWithText exportSize(size_t width, size_t height, size_t depth, VkFormat format, size_t& outSize)
{
  const char* overflowErrorMessage = "Invalid file: One of the subresources had a size that would require more than 2^64-1 bytes of data!";
  switch(format)
  {
    case VK_FORMAT_R4G4_UNORM_PACK8:
    case VK_FORMAT_R8_UNORM:
    case VK_FORMAT_R8_SNORM:
    case VK_FORMAT_R8_USCALED:
    case VK_FORMAT_R8_SSCALED:
    case VK_FORMAT_R8_UINT:
    case VK_FORMAT_R8_SINT:
    case VK_FORMAT_R8_SRGB:
    case VK_FORMAT_S8_UINT:
      if(!checked_math::mul4(width, height, depth, 8 / 8, outSize))  // 8 bits per pixel
        return overflowErrorMessage;
      return {};
    case VK_FORMAT_R4G4B4A4_UNORM_PACK16:
    case VK_FORMAT_B4G4R4A4_UNORM_PACK16:
    case VK_FORMAT_R5G6B5_UNORM_PACK16:
    case VK_FORMAT_B5G6R5_UNORM_PACK16:
    case VK_FORMAT_R5G5B5A1_UNORM_PACK16:
    case VK_FORMAT_B5G5R5A1_UNORM_PACK16:
    case VK_FORMAT_A1R5G5B5_UNORM_PACK16:
    case VK_FORMAT_R8G8_UNORM:
    case VK_FORMAT_R8G8_SNORM:
    case VK_FORMAT_R8G8_USCALED:
    case VK_FORMAT_R8G8_SSCALED:
    case VK_FORMAT_R8G8_UINT:
    case VK_FORMAT_R8G8_SINT:
    case VK_FORMAT_R8G8_SRGB:
    case VK_FORMAT_R16_UNORM:
    case VK_FORMAT_R16_SNORM:
    case VK_FORMAT_R16_USCALED:
    case VK_FORMAT_R16_SSCALED:
    case VK_FORMAT_R16_UINT:
    case VK_FORMAT_R16_SINT:
    case VK_FORMAT_R16_SFLOAT:
    case VK_FORMAT_D16_UNORM:
    case VK_FORMAT_D16_UNORM_S8_UINT:
      if(!checked_math::mul4(width, height, depth, 16 / 8, outSize))  // 16 bits per pixel
        return overflowErrorMessage;
      return {};
    case VK_FORMAT_R8G8B8_UNORM:
    case VK_FORMAT_R8G8B8_SNORM:
    case VK_FORMAT_R8G8B8_USCALED:
    case VK_FORMAT_R8G8B8_SSCALED:
    case VK_FORMAT_R8G8B8_UINT:
    case VK_FORMAT_R8G8B8_SINT:
    case VK_FORMAT_R8G8B8_SRGB:
    case VK_FORMAT_B8G8R8_UNORM:
    case VK_FORMAT_B8G8R8_SNORM:
    case VK_FORMAT_B8G8R8_USCALED:
    case VK_FORMAT_B8G8R8_SSCALED:
    case VK_FORMAT_B8G8R8_UINT:
    case VK_FORMAT_B8G8R8_SINT:
    case VK_FORMAT_B8G8R8_SRGB:
      if(!checked_math::mul4(width, height, depth, 24 / 8, outSize))  // 24 bits per pixel
        return overflowErrorMessage;
      return {};
    case VK_FORMAT_R8G8B8A8_UNORM:
    case VK_FORMAT_R8G8B8A8_SNORM:
    case VK_FORMAT_R8G8B8A8_USCALED:
    case VK_FORMAT_R8G8B8A8_SSCALED:
    case VK_FORMAT_R8G8B8A8_UINT:
    case VK_FORMAT_R8G8B8A8_SINT:
    case VK_FORMAT_R8G8B8A8_SRGB:
    case VK_FORMAT_B8G8R8A8_UNORM:
    case VK_FORMAT_B8G8R8A8_SNORM:
    case VK_FORMAT_B8G8R8A8_USCALED:
    case VK_FORMAT_B8G8R8A8_SSCALED:
    case VK_FORMAT_B8G8R8A8_UINT:
    case VK_FORMAT_B8G8R8A8_SINT:
    case VK_FORMAT_B8G8R8A8_SRGB:
    case VK_FORMAT_A8B8G8R8_UNORM_PACK32:
    case VK_FORMAT_A8B8G8R8_SNORM_PACK32:
    case VK_FORMAT_A8B8G8R8_USCALED_PACK32:
    case VK_FORMAT_A8B8G8R8_SSCALED_PACK32:
    case VK_FORMAT_A8B8G8R8_UINT_PACK32:
    case VK_FORMAT_A8B8G8R8_SINT_PACK32:
    case VK_FORMAT_A8B8G8R8_SRGB_PACK32:
    case VK_FORMAT_A2R10G10B10_UNORM_PACK32:
    case VK_FORMAT_A2R10G10B10_SNORM_PACK32:
    case VK_FORMAT_A2R10G10B10_USCALED_PACK32:
    case VK_FORMAT_A2R10G10B10_SSCALED_PACK32:
    case VK_FORMAT_A2R10G10B10_UINT_PACK32:
    case VK_FORMAT_A2R10G10B10_SINT_PACK32:
    case VK_FORMAT_A2B10G10R10_UNORM_PACK32:
    case VK_FORMAT_A2B10G10R10_SNORM_PACK32:
    case VK_FORMAT_A2B10G10R10_USCALED_PACK32:
    case VK_FORMAT_A2B10G10R10_SSCALED_PACK32:
    case VK_FORMAT_A2B10G10R10_UINT_PACK32:
    case VK_FORMAT_A2B10G10R10_SINT_PACK32:
    case VK_FORMAT_R16G16_UNORM:
    case VK_FORMAT_R16G16_SNORM:
    case VK_FORMAT_R16G16_USCALED:
    case VK_FORMAT_R16G16_SSCALED:
    case VK_FORMAT_R16G16_UINT:
    case VK_FORMAT_R16G16_SINT:
    case VK_FORMAT_R16G16_SFLOAT:
    case VK_FORMAT_R32_UINT:
    case VK_FORMAT_R32_SINT:
    case VK_FORMAT_R32_SFLOAT:
    case VK_FORMAT_B10G11R11_UFLOAT_PACK32:
    case VK_FORMAT_E5B9G9R9_UFLOAT_PACK32:
    case VK_FORMAT_X8_D24_UNORM_PACK32:
    case VK_FORMAT_D32_SFLOAT:
    case VK_FORMAT_D24_UNORM_S8_UINT:
      if(!checked_math::mul4(width, height, depth, 32 / 8, outSize))  // 32 bits per pixel
        return overflowErrorMessage;
      return {};
    case VK_FORMAT_R16G16B16_UNORM:
    case VK_FORMAT_R16G16B16_SNORM:
    case VK_FORMAT_R16G16B16_USCALED:
    case VK_FORMAT_R16G16B16_SSCALED:
    case VK_FORMAT_R16G16B16_UINT:
    case VK_FORMAT_R16G16B16_SINT:
    case VK_FORMAT_R16G16B16_SFLOAT:
      if(!checked_math::mul4(width, height, depth, 48 / 8, outSize))  // 48 bits per pixel
        return overflowErrorMessage;
      return {};
    case VK_FORMAT_R16G16B16A16_UNORM:
    case VK_FORMAT_R16G16B16A16_SNORM:
    case VK_FORMAT_R16G16B16A16_USCALED:
    case VK_FORMAT_R16G16B16A16_SSCALED:
    case VK_FORMAT_R16G16B16A16_UINT:
    case VK_FORMAT_R16G16B16A16_SINT:
    case VK_FORMAT_R16G16B16A16_SFLOAT:
    case VK_FORMAT_R32G32_UINT:
    case VK_FORMAT_R32G32_SINT:
    case VK_FORMAT_R32G32_SFLOAT:
    case VK_FORMAT_R64_UINT:
    case VK_FORMAT_R64_SINT:
    case VK_FORMAT_R64_SFLOAT:
      // Technically 40 or 64, but we choose the latter to make earlier special cases work:
    case VK_FORMAT_D32_SFLOAT_S8_UINT:
      if(!checked_math::mul4(width, height, depth, 64 / 8, outSize))  // 64 bits per pixel
        return overflowErrorMessage;
      return {};
    case VK_FORMAT_R32G32B32_UINT:
    case VK_FORMAT_R32G32B32_SINT:
    case VK_FORMAT_R32G32B32_SFLOAT:
      if(!checked_math::mul4(width, height, depth, 96 / 8, outSize))  // 96 bits per pixel
        return overflowErrorMessage;
      return {};
    case VK_FORMAT_R32G32B32A32_UINT:
    case VK_FORMAT_R32G32B32A32_SINT:
    case VK_FORMAT_R32G32B32A32_SFLOAT:
    case VK_FORMAT_R64G64_UINT:
    case VK_FORMAT_R64G64_SINT:
    case VK_FORMAT_R64G64_SFLOAT:
      if(!checked_math::mul4(width, height, depth, 128 / 8, outSize))  // 128 bits per pixel
        return overflowErrorMessage;
      return {};
    case VK_FORMAT_R64G64B64_UINT:
    case VK_FORMAT_R64G64B64_SINT:
    case VK_FORMAT_R64G64B64_SFLOAT:
      if(!checked_math::mul4(width, height, depth, 196 / 8, outSize))  // 196 bits per pixel
        return overflowErrorMessage;
      return {};
    case VK_FORMAT_R64G64B64A64_UINT:
    case VK_FORMAT_R64G64B64A64_SINT:
    case VK_FORMAT_R64G64B64A64_SFLOAT:
      if(!checked_math::mul4(width, height, depth, 256 / 8, outSize))  // 256 bits per pixel
        return overflowErrorMessage;
      return {};
    case VK_FORMAT_BC1_RGB_UNORM_BLOCK:
    case VK_FORMAT_BC1_RGB_SRGB_BLOCK:
    case VK_FORMAT_BC1_RGBA_UNORM_BLOCK:
    case VK_FORMAT_BC1_RGBA_SRGB_BLOCK:
    case VK_FORMAT_BC4_UNORM_BLOCK:
    case VK_FORMAT_BC4_SNORM_BLOCK:
      if(!checked_math::mul4((width + 3) / 4, (height + 3) / 4, depth, 8, outSize))  // 8 bytes per block
        return overflowErrorMessage;
      return {};
    case VK_FORMAT_BC2_UNORM_BLOCK:
    case VK_FORMAT_BC2_SRGB_BLOCK:
    case VK_FORMAT_BC3_UNORM_BLOCK:
    case VK_FORMAT_BC3_SRGB_BLOCK:
    case VK_FORMAT_BC5_UNORM_BLOCK:
    case VK_FORMAT_BC5_SNORM_BLOCK:
    case VK_FORMAT_BC6H_UFLOAT_BLOCK:
    case VK_FORMAT_BC6H_SFLOAT_BLOCK:
    case VK_FORMAT_BC7_UNORM_BLOCK:
    case VK_FORMAT_BC7_SRGB_BLOCK:
      if(!checked_math::mul4((width + 3) / 4, (height + 3) / 4, depth, 16, outSize))  // 16 bytes per block
        return overflowErrorMessage;
      return {};
    case VK_FORMAT_ASTC_4x4_UNORM_BLOCK:
    case VK_FORMAT_ASTC_4x4_SRGB_BLOCK:
      if(!astcSize(4, 4, 1, width, height, depth, outSize))
        return overflowErrorMessage;
      return {};
    case VK_FORMAT_ASTC_5x4_UNORM_BLOCK:
    case VK_FORMAT_ASTC_5x4_SRGB_BLOCK:
      if(!astcSize(5, 4, 1, width, height, depth, outSize))
        return overflowErrorMessage;
      return {};
    case VK_FORMAT_ASTC_5x5_UNORM_BLOCK:
    case VK_FORMAT_ASTC_5x5_SRGB_BLOCK:
      if(!astcSize(5, 5, 1, width, height, depth, outSize))
        return overflowErrorMessage;
      return {};
    case VK_FORMAT_ASTC_6x5_UNORM_BLOCK:
    case VK_FORMAT_ASTC_6x5_SRGB_BLOCK:
      if(!astcSize(6, 5, 1, width, height, depth, outSize))
        return overflowErrorMessage;
      return {};
    case VK_FORMAT_ASTC_6x6_UNORM_BLOCK:
    case VK_FORMAT_ASTC_6x6_SRGB_BLOCK:
      if(!astcSize(6, 6, 1, width, height, depth, outSize))
        return overflowErrorMessage;
      return {};
    case VK_FORMAT_ASTC_8x5_UNORM_BLOCK:
    case VK_FORMAT_ASTC_8x5_SRGB_BLOCK:
      if(!astcSize(8, 5, 1, width, height, depth, outSize))
        return overflowErrorMessage;
      return {};
    case VK_FORMAT_ASTC_8x6_UNORM_BLOCK:
    case VK_FORMAT_ASTC_8x6_SRGB_BLOCK:
      if(!astcSize(8, 6, 1, width, height, depth, outSize))
        return overflowErrorMessage;
      return {};
    case VK_FORMAT_ASTC_8x8_UNORM_BLOCK:
    case VK_FORMAT_ASTC_8x8_SRGB_BLOCK:
      if(!astcSize(8, 8, 1, width, height, depth, outSize))
        return overflowErrorMessage;
      return {};
    case VK_FORMAT_ASTC_10x5_UNORM_BLOCK:
    case VK_FORMAT_ASTC_10x5_SRGB_BLOCK:
      if(!astcSize(10, 5, 1, width, height, depth, outSize))
        return overflowErrorMessage;
      return {};
    case VK_FORMAT_ASTC_10x6_UNORM_BLOCK:
    case VK_FORMAT_ASTC_10x6_SRGB_BLOCK:
      if(!astcSize(10, 6, 1, width, height, depth, outSize))
        return overflowErrorMessage;
      return {};
    case VK_FORMAT_ASTC_10x8_UNORM_BLOCK:
    case VK_FORMAT_ASTC_10x8_SRGB_BLOCK:
      if(!astcSize(10, 8, 1, width, height, depth, outSize))
        return overflowErrorMessage;
      return {};
    case VK_FORMAT_ASTC_10x10_UNORM_BLOCK:
    case VK_FORMAT_ASTC_10x10_SRGB_BLOCK:
      if(!astcSize(10, 10, 1, width, height, depth, outSize))
        return overflowErrorMessage;
      return {};
    case VK_FORMAT_ASTC_12x10_UNORM_BLOCK:
    case VK_FORMAT_ASTC_12x10_SRGB_BLOCK:
      if(!astcSize(12, 10, 1, width, height, depth, outSize))
        return overflowErrorMessage;
      return {};
    case VK_FORMAT_ASTC_12x12_UNORM_BLOCK:
    case VK_FORMAT_ASTC_12x12_SRGB_BLOCK:
      if(!astcSize(12, 12, 1, width, height, depth, outSize))
        return overflowErrorMessage;
      return {};
    default:
      return "Tried to find size of unrecognized VkFormat " + std::to_string(format) + ".";
  }
}

ErrorWithText exportSizeExtended(size_t width, size_t height, size_t depth, VkFormat format, size_t& outSize, CustomExportSizeFuncPtr extraCallback)
{
  const ErrorWithText builtinResult = exportSize(width, height, depth, format, outSize);
  if(!builtinResult.has_value() || extraCallback == nullptr)
  {
    return builtinResult;
  }

  // We didn't recognize the format using the built-in exportSize, so try the
  // provided reader.
  return extraCallback(width, height, depth, format, outSize);
}

//-----------------------------------------------------------------------------
// KTX1 READER
//-----------------------------------------------------------------------------

// Used in the KTX1 reader to determine the default transfer function for a
// VkFormat, since KTX1 doesn't have the Data Format Descriptor.
bool isKtx1FormatSrgbByDefault(VkFormat format)
{
  switch(format)
  {
      // Formats that are definitely always sRGB
    case VK_FORMAT_R8_SRGB:
    case VK_FORMAT_R8G8_SRGB:
    case VK_FORMAT_R8G8B8_SRGB:
    case VK_FORMAT_B8G8R8_SRGB:
    case VK_FORMAT_R8G8B8A8_SRGB:
    case VK_FORMAT_B8G8R8A8_SRGB:
    case VK_FORMAT_A8B8G8R8_SRGB_PACK32:
    case VK_FORMAT_BC1_RGB_SRGB_BLOCK:
    case VK_FORMAT_BC1_RGBA_SRGB_BLOCK:
    case VK_FORMAT_BC2_SRGB_BLOCK:
    case VK_FORMAT_BC3_SRGB_BLOCK:
    case VK_FORMAT_BC7_SRGB_BLOCK:
    case VK_FORMAT_ETC2_R8G8B8_SRGB_BLOCK:
    case VK_FORMAT_ETC2_R8G8B8A1_SRGB_BLOCK:
    case VK_FORMAT_ETC2_R8G8B8A8_SRGB_BLOCK:
    case VK_FORMAT_ASTC_4x4_SRGB_BLOCK:
    case VK_FORMAT_ASTC_5x4_SRGB_BLOCK:
    case VK_FORMAT_ASTC_5x5_SRGB_BLOCK:
    case VK_FORMAT_ASTC_6x5_SRGB_BLOCK:
    case VK_FORMAT_ASTC_6x6_SRGB_BLOCK:
    case VK_FORMAT_ASTC_8x5_SRGB_BLOCK:
    case VK_FORMAT_ASTC_8x6_SRGB_BLOCK:
    case VK_FORMAT_ASTC_8x8_SRGB_BLOCK:
    case VK_FORMAT_ASTC_10x5_SRGB_BLOCK:
    case VK_FORMAT_ASTC_10x6_SRGB_BLOCK:
    case VK_FORMAT_ASTC_10x8_SRGB_BLOCK:
    case VK_FORMAT_ASTC_10x10_SRGB_BLOCK:
    case VK_FORMAT_ASTC_12x10_SRGB_BLOCK:
    case VK_FORMAT_ASTC_12x12_SRGB_BLOCK:
    case VK_FORMAT_PVRTC1_2BPP_SRGB_BLOCK_IMG:
    case VK_FORMAT_PVRTC1_4BPP_SRGB_BLOCK_IMG:
    case VK_FORMAT_PVRTC2_2BPP_SRGB_BLOCK_IMG:
    case VK_FORMAT_PVRTC2_4BPP_SRGB_BLOCK_IMG:
      return true;
      // Formats that are definitely always linear
    case VK_FORMAT_R32G32B32A32_SFLOAT:
    case VK_FORMAT_R32G32B32_SFLOAT:
    case VK_FORMAT_R16G16B16A16_SFLOAT:
    case VK_FORMAT_R32G32_SFLOAT:
    case VK_FORMAT_R16G16_SFLOAT:
    case VK_FORMAT_D32_SFLOAT:
    case VK_FORMAT_R32_SFLOAT:
    case VK_FORMAT_R16_SFLOAT:
    case VK_FORMAT_E5B9G9R9_UFLOAT_PACK32:
    case VK_FORMAT_BC6H_SFLOAT_BLOCK:
    case VK_FORMAT_BC6H_UFLOAT_BLOCK:
      // According to toktx, otherwise the correct choice is to interpret it as linear:
    default:
      return false;
  }
}

// The values from UInt32 endianness through UInt32 bytesOfKeyValueData in the KTX1 header.
struct KTX1TopLevelHeader
{
  uint32_t endianness;
  uint32_t glType;
  uint32_t glTypeSize;
  uint32_t glFormat;
  uint32_t glInternalFormat;
  uint32_t glBaseInternalFormat;
  uint32_t pixelWidth;
  uint32_t pixelHeight;
  uint32_t pixelDepth;
  uint32_t numberOfArrayElements;
  uint32_t numberOfFaces;
  uint32_t numberOfMipmapLevels;
  uint32_t bytesOfKeyValueData;
};
}  // namespace

// Reads the header of a KTX 1.0 file, *starting after the 12-byte identifier*.
ErrorWithText Image::readHeaderFromKTX1Stream(std::istream& input, const ReadSettings& readSettings)
{
  // The start of the KTX 1.0 file.
  const std::streampos startPos = input.tellg() - std::streamoff(IDENTIFIER_LEN);

  // Record size of the input for validation, if that's enabled.
  size_t validationInputSize = 0;
  if(readSettings.validateInputSize)
  {
    input.seekg(0, std::ios_base::end);
    const std::streampos endPos = input.tellg();
    validationInputSize         = static_cast<size_t>(endPos - startPos);
    input.seekg(startPos + std::streamoff(IDENTIFIER_LEN), std::ios_base::beg);
  }

  KTX1TopLevelHeader header{};
  READ_OR_ERROR(input, header, "Failed to read KTX1 header.");
  // Determine if the file needs to be swapped from big-endian to little-endian.
  // (We assume the machine is little-endian.)
  // If this is true, then when making the file's data usable by the GPU,
  // we need to swap every UInt32 in the KTX1 File Structure as well as
  // each element in uncompressed texture data.
  m_fileInfo.ktx1NeedsEndianSwap = false;
  if(header.endianness == 0x01020304u)
  {
    m_fileInfo.ktx1NeedsEndianSwap = true;
  }
  else if(header.endianness != 0x04030201u)
  {
    std::stringstream str;
    str << "KXT1 endianness (0x" << std::hex << header.endianness << ") did not match either big-endian or little-endian formats.";
    return str.str();
  }

  if(m_fileInfo.ktx1NeedsEndianSwap)
  {
    swapEndian32<KTX1TopLevelHeader>(header);
  }

  m_fileInfo.ktx1GlTypeSize = header.glTypeSize;

  // Set the dimensions in the structure to indicate the size and type of the
  // texture. Then replace some fields with 1 if they were 0 to make the rest
  // of the importer less complex.
  mip0Width          = header.pixelWidth;
  mip0Height         = header.pixelHeight;
  mip0Depth          = header.pixelDepth;
  numLayersPossibly0 = header.numberOfArrayElements;
  numFaces           = header.numberOfFaces;

  appShouldGenerateMips = (header.numberOfMipmapLevels == 0);
  if(appShouldGenerateMips)
  {
    header.numberOfMipmapLevels = 1;
  }
  numMips = header.numberOfMipmapLevels;

  // Keep track of the special case where we have padding with non-array cubemap textures:
  const bool isArray = (header.numberOfArrayElements != 0);
  if(!isArray)
    header.numberOfArrayElements = 1;
  if(header.pixelDepth == 0)
    header.pixelDepth = 1;
  if(header.pixelHeight == 0)
    header.pixelHeight = 1;

  // Validate pixel width
  if(header.pixelWidth == 0)
  {
    return "KTX1 image had a width of 0 pixels!";
  }
  // Validate number of faces
  if(header.numberOfFaces == 0)
  {
    return "KTX1 image had no faces!";
  }
  if(header.numberOfFaces > 6)
  {
    return "KTX1 image had too many faces!";
  }
  // The maximum image dimensions are 2^32-1 by 2^32-1, so there can be at most
  // 31 mips.
  if(header.numberOfMipmapLevels > 31)
  {
    return "KTX1 image had more than 31 mips!";
  }

  size_t numSubresources = 0;
  if(!getNumSubresources(numMips, numLayersPossibly0, numFaces, numSubresources))
  {
    return "Computing the number of mips times layers times faces in the file overflowed!";
  }

  if(readSettings.validateInputSize)
  {
    if(numSubresources > validationInputSize)
    {
      return "The KTX1 input had a likely invalid header - it listed " + std::to_string(numMips) + " mips (or 0), "
             + std::to_string(numLayersPossibly0) + " layers (or 0), and " + std::to_string(numFaces)
             + " faces - but the input was only " + std::to_string(validationInputSize) + " bytes long!";
    }
    if(header.bytesOfKeyValueData > validationInputSize)
    {
      return "The KTX1 input had an invalid header - it listed " + std::to_string(header.bytesOfKeyValueData)
             + " bytes of key/value data, but the input was only " + std::to_string(validationInputSize) + " bytes long!";
    }
  }

  //---------------------------------------------------------------------------
  // Read key-value data.
  UNWRAP_ERROR(readKeyValueData(input, header.bytesOfKeyValueData, m_fileInfo.ktx1NeedsEndianSwap, keyValueData));

  // KTX1 doesn't have ktxSwizzle, so:
  swizzle = {Swizzle::R, Swizzle::G, Swizzle::B, Swizzle::A};

  //---------------------------------------------------------------------------
  // Calculate formats and fields used for decompression.

  // Determine a corresponding VkFormat for the given GL format.
  format = texture_formats::openGLToVulkan({header.glInternalFormat, header.glFormat, header.glType});
  if(VK_FORMAT_UNDEFINED == format)
  {
    return "Could not determine a corresponding Vulkan format for GL internal format " + std::to_string(header.glInternalFormat)
           + ", GL format " + std::to_string(header.glFormat) + ", and GL type " + std::to_string(header.glType) + ".";
  }

  // Guess if this format is sRGB.
  isSrgb = isKtx1FormatSrgbByDefault(format);

  // Always set this to false, since DXT2 and DXT4 aren't supported in
  // EXT_texture_compression_s3tc.
  isPremultiplied = false;

  // Allocate and fill out subresource layouts while validating the rest of the file.
  UNWRAP_ERROR(resizeVectorOrError(m_subresourceLayouts, numSubresources));
  size_t remainingAllowedUncompressedBytes = readSettings.maxSizeInBytes;

  for(uint32_t mip = 0; mip < header.numberOfMipmapLevels; mip++)
  {
    // Read the image size. We use this for mip padding later on, and rely on
    // exportSize for individual subresources.
    uint32_t imageSize = 0;
    READ_OR_ERROR(input, imageSize, "Failed to read KTX1 imageSize for mip " + std::to_string(mip) + ".");
    if(m_fileInfo.ktx1NeedsEndianSwap)
    {
      swapEndian32(imageSize);
    }

    const size_t mipWidth  = std::max(1u, header.pixelWidth >> mip);
    const size_t mipHeight = std::max(1u, header.pixelHeight >> mip);
    const size_t mipDepth  = std::max(1u, header.pixelDepth >> mip);

    // Compute the size of a face in bytes
    size_t faceSizeBytes = 0;
    UNWRAP_ERROR(exportSizeExtended(mipWidth, mipHeight, mipDepth, format, faceSizeBytes, readSettings.customSizeCallback));
    // Validate it
    {
      size_t maxUncompressedMipSize = 0;
      if(!checked_math::mul3(faceSizeBytes, header.numberOfArrayElements, header.numberOfFaces, maxUncompressedMipSize))
      {
        return "The number of uncompressed bytes to store decompressed mip " + std::to_string(mip) + " would have overflowed a size_t.";
      }

      if(remainingAllowedUncompressedBytes < maxUncompressedMipSize)
      {
        return "This file would require more than the limit of maxSizeInBytes = "
               + std::to_string(readSettings.maxSizeInBytes) + " bytes without supercompression.";
      }
      remainingAllowedUncompressedBytes -= maxUncompressedMipSize;
    }

    if(readSettings.validateInputSize)
    {
      if(((validationInputSize / size_t(header.numberOfArrayElements)) / size_t(header.numberOfFaces)) < faceSizeBytes)
      {
        return "The KTX1 file said it contained " + std::to_string(header.numberOfArrayElements)
               + " array elements and " + std::to_string(header.numberOfFaces) + " faces in mip " + std::to_string(mip)
               + ", but the input was too short (" + std::to_string(validationInputSize) + " bytes) to contain that!";
      }
    }

    for(uint32_t arrayElement = 0; arrayElement < header.numberOfArrayElements; arrayElement++)
    {
      for(uint32_t face = 0; face < header.numberOfFaces; face++)
      {
        subresourceLayout(mip, arrayElement, face) = SubresourceLayout{.fileOffset   = size_t(input.tellg() - startPos),
                                                                       .fileByteSize = faceSizeBytes,
                                                                       .uncompressedByteSize = faceSizeBytes};

        if(!input.seekg(static_cast<std::streamoff>(faceSizeBytes), std::ios_base::cur))
        {
          return "Seeking past mip " + std::to_string(mip) + " layer " + std::to_string(arrayElement) + " face "
                 + std::to_string(face) + " failed (is the file truncated)?";
        }

        // Handle cubePadding
        if((!isArray) && (header.numberOfFaces == 6))
        {
          // faceSizeBytes mod 4: 0 1 2 3
          // cubePadding        : 0 3 2 1
          const std::streamoff cubePaddingBytes = 3 - ((static_cast<std::streamoff>(faceSizeBytes) + 3) % 4);
          if(!input.seekg(cubePaddingBytes, std::ios_base::cur))
          {
            return "Seeking past KTX1 cube padding failed. Is the input truncated?";
          }
        }
      }
    }

    // Handle mip padding. The spec says that this is always 3 - ((imageSize + 3)%4) bytes,
    // and we assume bytes are the same size as chars.
    const std::streamoff mipPaddingBytes = 3 - ((static_cast<std::streamoff>(imageSize) + 3) % 4);
    if(!input.seekg(mipPaddingBytes, std::ios_base::cur))
    {
      return "Seeking past KTX1 mip padding failed. Is the input truncated?";
    }
  }

  return {};
}

// readSubresourcesFromStream() backend for a KTX1 file.
ErrorWithText Image::readSubresourcesFromKTX1Stream(std::istream& input, const SubresourceRange& range, SubresourceTarget* outSubresources)
{
  const std::streamoff startPos = input.tellg();

  for(uint32_t dMip = 0; dMip < range.numMips; dMip++)
  {
    for(uint32_t dLayer = 0; dLayer < range.numLayers; dLayer++)
    {
      for(uint32_t dFace = 0; dFace < range.numFaces; dFace++)
      {
        const uint32_t mip   = range.firstMip + dMip;
        const uint32_t layer = range.firstLayer + dLayer;
        const uint32_t face  = range.firstFace + dFace;

        const SubresourceLayout& subresourceLayout = getSubresourceLayout(mip, layer, face);

        SubresourceTarget& target = outSubresources[(dMip * range.numLayers + dLayer) * range.numFaces + dFace];

        if(!input.seekg(startPos + subresourceLayout.fileOffset, std::ios_base::beg))
        {
          return "Seeking to the data for mip " + std::to_string(mip) + " layer " + std::to_string(layer) + " face "
                 + std::to_string(face) + " failed. Is the input truncated?";
        }

        if(!input.read(reinterpret_cast<char*>(target.data), subresourceLayout.fileByteSize))
        {
          return "Reading the data for mip " + std::to_string(mip) + " layer " + std::to_string(layer) + " face "
                 + std::to_string(face) + " failed. Is the input truncated?";
        }

        // Apply endianness swapping
        if(m_fileInfo.ktx1NeedsEndianSwap)
        {
          swapEndianGeneral(subresourceLayout.fileByteSize, target.data, m_fileInfo.ktx1GlTypeSize);
        }
      }
    }
  }

  return {};
}

//-----------------------------------------------------------------------------
// KTX2 READER + WRITER
//-----------------------------------------------------------------------------

// Also specific to Basis ETC1S+BasisLZ - Basis includes support for texture
// array animation encoding using P-frames and I-frames, and it marks which
// frames are which using flags in its table of image descriptions.
// This is the "isPFrame" flag in the KTX2 specification.
const uint32_t KTX2_IMAGE_IS_P_FRAME = 2;

static_assert(sizeof(SubresourceLayout) == 3 * sizeof(uint64_t), "SubresourceLayout size must match KTX2 spec!");

#ifdef NVP_SUPPORTS_ZSTD
// A Zstandard decompression context that is automatically freed when it goes out of scope.
struct ScopedZstdDContext
{
  ScopedZstdDContext() {}
  ~ScopedZstdDContext() { Free(); }
  void Init()
  {
    Free();
    pCtx = ZSTD_createDCtx();
  }
  void Free()
  {
    if(pCtx != nullptr)
    {
      ZSTD_freeDCtx(pCtx);
      pCtx = nullptr;
    }
  }
  ZSTD_DCtx* pCtx = nullptr;
};

// A Zstandard compression context that is automatically freed when it goes out of scope.
struct ScopedZstdCContext
{
  ScopedZstdCContext() {}
  ~ScopedZstdCContext() { Free(); }
  void Init()
  {
    Free();
    pCtx = ZSTD_createCCtx();
  }
  void Free()
  {
    if(pCtx != nullptr)
    {
      ZSTD_freeCCtx(pCtx);
      pCtx = nullptr;
    }
  }
  ZSTD_CCtx* pCtx = nullptr;
};
#endif

#ifdef NVP_SUPPORTS_GZLIB
// A Zlib inflation stream that is automatically deinitialized when it goes out of scope.
struct ScopedZlibDStream
{
  ScopedZlibDStream() {}
  ~ScopedZlibDStream() { Free(); }
  int Init()
  {
    Free();
    stream.zalloc   = Z_NULL;
    stream.zfree    = Z_NULL;
    stream.opaque   = Z_NULL;
    stream.avail_in = 0;
    stream.next_in  = Z_NULL;
    return inflateInit(&stream);
  }
  void     Free() { inflateEnd(&stream); }
  z_stream stream{};
};
#endif

#ifdef NVP_SUPPORTS_BASISU

// Stores data per-image that is required to transcode a BasisLZ+ETC1S image.
struct BasisLZDecompressionObjects
{
  // Stores and knows how to decode the BasisLZ + ETC1S stream
  basist::basisu_lowlevel_etc1s_transcoder* etc1sTranscoder = nullptr;
  // Additional information from the global data block not included in the transcoder
  std::vector<basist::ktx2_etc1s_image_desc> etc1sImageDescs;
  basist::ktx2_transcoder_state              ktx2TranscoderState;

  ~BasisLZDecompressionObjects()
  {
    if(etc1sTranscoder)
    {
      delete etc1sTranscoder;
      etc1sTranscoder = nullptr;
    }
  }
};

// Basis Universal makes use of some global data, and prints developer error
// messages if we attempt to initialize this more than once.
// This Meyers singleton keeps track of said global data.
struct BasisUSingleton
{
  static BasisUSingleton& GetInstance()
  {
    static BasisUSingleton s;
    return s;
  }

  // No copying
  BasisUSingleton(const BasisUSingleton&)            = delete;
  BasisUSingleton& operator=(const BasisUSingleton&) = delete;

  void TranscodeUastcToBc7OrAstc44(char* output, const char* inData, size_t width, size_t height, size_t depth, bool toAstc)
  {
    if(!Initialize())
      return;

    const basist::uastc_block* buf        = reinterpret_cast<const basist::uastc_block*>(inData);
    const size_t               numBlocksX = (width + 3) / 4;
    const size_t               numBlocksY = (height + 3) / 4;
    const size_t               numBlocksZ = (depth + 3) / 4;
    const size_t               numBlocks  = numBlocksX * numBlocksY * numBlocksZ;
    if(numBlocks > INT64_MAX)
      return;  // Won't fit in an OpenMP range

    const int64_t numBlocksI = static_cast<int64_t>(numBlocks);
    if(toAstc)
    {
#if defined(_OPENMP)
#pragma omp parallel for
#endif
      for(int64_t blockIdx = 0; blockIdx < numBlocksI; blockIdx++)
      {
        const basist::uastc_block& block = buf[blockIdx];
        char*                      dst   = output + size_t(blockIdx) * 16;  // 16 bytes per ASTC block
        basist::transcode_uastc_to_astc(block, dst);
      }
    }
    else
    {
      // To BC7
#if defined(_OPENMP)
#pragma omp parallel for
#endif
      for(int64_t blockIdx = 0; blockIdx < numBlocksI; blockIdx++)
      {
        const basist::uastc_block& block = buf[blockIdx];
        char*                      dst   = output + size_t(blockIdx) * 16;  // 16 bytes per BC7 block
        basist::transcode_uastc_to_bc7(block, dst);
      }
    }
  }

  ErrorWithText PrepareBasisLZObjects(BasisLZDecompressionObjects& outObjects,
                                      const std::vector<uint8_t>&  ktxSGD,
                                      const uint32_t               numMips,
                                      const uint32_t               numLayers,
                                      const uint32_t               numFaces)
  {
    if(!Initialize())
    {
      return "Initializing BasisU failed!";
    }
    // For reference, see Basis Universal's ktx2_transcoder::decompress_etc1s_global_data().

    // Compute the image count, erroring if we would overflow:
    uint32_t imageCount = numMips;
    {
      if(numMips == 0 || numLayers == 0 || numFaces == 0)
      {
        return "The number of mips, layers, or faces was 0!";
      }
      if(imageCount >= (UINT_MAX / numLayers))
      {
        return "The number of images was over 2^32-1!";
      }
      imageCount *= numLayers;
      if(imageCount >= (UINT_MAX / numFaces))
      {
        return "The number of images was over 2^32-1!";
      }
      imageCount *= numFaces;
    }

    if(ktxSGD.size() < sizeof(basist::ktx2_etc1s_global_data_header))
    {
      return "The data included ETC1S+BasisLZ compression, but the length of the supercompression global data was too "
             "short to contain the required information!";
    }

    size_t offsetInSGD = 0;

    basist::ktx2_etc1s_global_data_header etc1sHeader;
    memcpy(&etc1sHeader, ktxSGD.data() + offsetInSGD, sizeof(etc1sHeader));
    offsetInSGD += sizeof(etc1sHeader);

    // Check the ETC1S header.
    if((!etc1sHeader.m_endpoints_byte_length) || (!etc1sHeader.m_selectors_byte_length) || (!etc1sHeader.m_tables_byte_length))
    {
      return "The data included ETC1S+BasisLZ compression, but the supercompression global data had invalid byte "
             "length "
             "properties!";
    }

    if((!etc1sHeader.m_endpoint_count) || (!etc1sHeader.m_selector_count))
    {
      return "The data included ETC1S+BasisLZ compression, but the endpoint or selector count is 0, which is invalid!";
    }

    if((sizeof(basist::ktx2_etc1s_global_data_header) + sizeof(basist::ktx2_etc1s_image_desc) * imageCount + etc1sHeader.m_endpoints_byte_length
        + etc1sHeader.m_selectors_byte_length + etc1sHeader.m_tables_byte_length + etc1sHeader.m_extended_byte_length)
       > ktxSGD.size())
    {
      return "The data included ETC1S+BasisLZ compression, but the supercompression global data was invalid: it was "
             "too "
             "small to contain the data its header said it contains!";
    }

    // Read the image descriptions
    UNWRAP_ERROR(resizeVectorOrError(outObjects.etc1sImageDescs, imageCount));
    memcpy(outObjects.etc1sImageDescs.data(), ktxSGD.data() + offsetInSGD, sizeof(basist::ktx2_etc1s_image_desc) * imageCount);
    offsetInSGD += sizeof(basist::ktx2_etc1s_image_desc) * imageCount;

    // We'll verify that slice byte lengths are nonzero later.

    // Initialize the ETC1S transcoder.
    if(outObjects.etc1sTranscoder)
      delete outObjects.etc1sTranscoder;
    outObjects.etc1sTranscoder = new basist::basisu_lowlevel_etc1s_transcoder();
    outObjects.etc1sTranscoder->clear();

    // Decode tables and palettes.
    const uint8_t* endpointData = ktxSGD.data() + offsetInSGD;
    const uint8_t* selectorData = endpointData + uint32_t(etc1sHeader.m_endpoints_byte_length);
    const uint8_t* tablesData   = selectorData + uint32_t(etc1sHeader.m_selectors_byte_length);

    if(!outObjects.etc1sTranscoder->decode_tables(tablesData, etc1sHeader.m_tables_byte_length))
    {
      return "Failed to decode tables from the BasisLZ supercompression data";
    }

    if(!outObjects.etc1sTranscoder->decode_palettes(etc1sHeader.m_endpoint_count, endpointData, etc1sHeader.m_endpoints_byte_length,  //
                                                    etc1sHeader.m_selector_count, selectorData, etc1sHeader.m_selectors_byte_length))
    {
      return "Failed to decode palettes from the BasisLZ supercompression data";
    }

    outObjects.ktx2TranscoderState.clear();

    return {};
  }

  // Attempts to initialize the Basis encoder and decoder, and returns true if that
  // succeeded or if it was already initialized (this can be called
  // multiple times)
  bool Initialize()
  {
    if(!m_initialized.load())
    {
      std::lock_guard<std::mutex> lock(m_modificationMutex);
      basist::basisu_transcoder_init();
      basisu::basisu_encoder_init();
      m_initialized.store(true);
    }
    return true;
  }

private:
  BasisUSingleton() {};
  // Frees objects if loaded
  ~BasisUSingleton()
  {
    std::lock_guard<std::mutex> lock(m_modificationMutex);
    m_initialized.store(false);
  }

private:
  std::mutex        m_modificationMutex;
  std::atomic<bool> m_initialized = false;
};
#endif

#pragma pack(push, 1)
struct KTX2TopLevelHeader
{
  VkFormat vkFormat;
  uint32_t typeSize;
  uint32_t pixelWidth;
  uint32_t pixelHeight;
  uint32_t pixelDepth;
  uint32_t layerCount;  // Num array elements
  uint32_t faceCount;
  uint32_t levelCount;  // Num mips
  uint32_t supercompressionScheme;

  // Index (1)
  uint32_t dfdByteOffset;
  uint32_t dfdByteLength;
  uint32_t kvdByteOffset;
  uint32_t kvdByteLength;
  uint64_t sgdByteOffset;
  uint64_t sgdByteLength;
};
#pragma pack(pop)

static_assert(sizeof(VkFormat) == sizeof(uint32_t), "VkFormat size must match KTX2 spec!");
static_assert(sizeof(KTX2TopLevelHeader) == 68, "KTX2 top-level header size must match spec! Padding issue?");

// Reads a KTX 2.0 file, *starting after the 12-byte identifier*.
ErrorWithText Image::readHeaderFromKTX2Stream(std::istream& input, const ReadSettings& readSettings)
{
  // Get the position of the start of the file in the stream so that we can add
  // padding correctly later.
  const std::streampos startPos = input.tellg() - std::streampos(IDENTIFIER_LEN);  // Since we start after the identifier
  size_t validationInputSize = 0;
  if(readSettings.validateInputSize)
  {
    input.seekg(0, std::ios_base::end);
    const std::streampos endPos = input.tellg();
    validationInputSize         = static_cast<size_t>(endPos - startPos);
    input.seekg(startPos + std::streampos(IDENTIFIER_LEN), std::ios_base::beg);
  }

  //---------------------------------------------------------------------------
  // Read sections 0 and 1 of the file structure.
  KTX2TopLevelHeader header{};
  READ_OR_ERROR(input, header, "Failed to read KTX2 header and section 1.");

  // Copy the dimensions into the structure so that we can determine the
  // texture type later.
  // `format` is the inflated VkFormat; we may swap it out as we read more
  // supercompression info due to transcoding.
  format             = header.vkFormat;
  mip0Width          = header.pixelWidth;
  mip0Height         = header.pixelHeight;
  mip0Depth          = header.pixelDepth;
  numLayersPossibly0 = header.layerCount;
  // numFaces cannot be 0, on the other hand, so we set it below!

  // If the image width is 0, we can't read it (and the file is invalid).
  if(header.pixelWidth == 0)
  {
    return "KTX2 image width was 0 (i.e. the file contains no pixels).";
  }

  // These dimensions are 0 only to indicate the type of the texture
  // (see section 4.1). To make the rest of the reader simpler and to avoid
  // errors later on, we can change each of these to 1 if they're 0.
  if(header.pixelHeight == 0)
    header.pixelHeight = 1;
  if(header.pixelDepth == 0)
    header.pixelDepth = 1;
  if(header.layerCount == 0)
    header.layerCount = 1;
  if(header.faceCount == 0)
    header.faceCount = 1;
  // KTX files also only use levelCount == 0 to indicate that loaders should
  // generate other levels if needed (section 3.7 "levelCount"). Since the user
  // ultimately controls this, we change a 0 to a 1 here as well.
  // We have a special case where we need max(1, the original number of levels)
  // for Basis ETC1S unpacking.
  appShouldGenerateMips = (header.levelCount == 0);
  if(appShouldGenerateMips)
  {
    header.levelCount = 1;
  }
  numMips  = header.levelCount;
  numFaces = header.faceCount;

  // Validate the data format descriptor byte length. KDF 1.3 assumes the Data
  // Format Descriptor works as a series of 32-bit words.
  if((header.dfdByteLength % 4) != 0)
  {
    return "KTX2 Data Format Descriptor byte length was not a multiple of 4, so is not valid.";
  }

  // Validate the level count because we allocate memory based off it. It can't
  // be larger than 31 - if it were, then pixelWidth and pixelHeight wouldn't
  // fit in UInt32 types.
  if(numMips > 31)
  {
    std::stringstream str;
    str << "KTX2 levelCount was too large (" << numMips << ") - the "
        << "maximum number of mips possible in a KTX2 file is 31.";
    return str.str();
  }

  size_t numSubresources = 0;
  if(!getNumSubresources(numMips, numLayersPossibly0, numFaces, numSubresources))
  {
    return "Computing the number of mips times layers times faces in the file overflowed!";
  }

  if(readSettings.validateInputSize)
  {
    if(numSubresources > validationInputSize)
    {
      return "The KTX2 input had a likely invalid header - it listed " + std::to_string(numMips) + " mips (or 0), "
             + std::to_string(numLayersPossibly0) + " layers (or 0), and " + std::to_string(numFaces)
             + " faces - but the input was only " + std::to_string(validationInputSize) + " bytes long!";
    }
    if(header.dfdByteLength > validationInputSize)
    {
      return "The KTX2 input had an invalid header - it said its Data Format Descriptor was " + std::to_string(header.dfdByteLength)
             + " bytes long, but the input was only " + std::to_string(validationInputSize) + " bytes long!";
    }
    if(header.kvdByteLength > validationInputSize)
    {
      return "The KTX2 input had an invalid header - it listed " + std::to_string(header.kvdByteLength)
             + " bytes of key/value data, but the input was only " + std::to_string(validationInputSize) + " bytes long!";
    }
  }

  //---------------------------------------------------------------------------
  // Load the level indices (section 2)
  UNWRAP_ERROR(resizeVectorOrError(m_levelIndices, numMips));
  if(!input.read(reinterpret_cast<char*>(m_levelIndices.data()), sizeof(SubresourceLayout) * numMips))
  {
    return "Unable to read Level Index from KTX2 file.";
  }

  //---------------------------------------------------------------------------
  // Load the Data Format Descriptor. We read this as a uint32_t array and then
  // interpret it later.
  std::vector<uint32_t> dfd;
  UNWRAP_ERROR(resizeVectorOrError(dfd, header.dfdByteLength / 4));
  if(!input.read(reinterpret_cast<char*>(dfd.data()), header.dfdByteLength))
  {
    return "Unable to read Data Format Descriptor from KTX2 file.";
  }

  // Get some information from the Basic Data Format Descriptor.
  uint32_t                  dfdTotalSize = 0;
  BasicDataFormatDescriptor basicDFD{};
  std::vector<DFSample>     dfdSamples;
  bool                      basicDFDExists = false;
  if(header.dfdByteLength >= sizeof(dfdTotalSize) + sizeof(basicDFD))
  {
    dfdTotalSize = dfd[0];
    if(dfdTotalSize != header.dfdByteLength)
    {
      return "KTX2 Data Format Descriptor was invalid (data format descriptor's dfdTotalSize didn't match the index's "
             "dfdByteLength)";
    }

    memcpy(&basicDFD, &dfd[1], sizeof(BasicDataFormatDescriptor));
    basicDFDExists = true;

    // Attempt to read the sample data from the Data Format Descriptor.
    if(size_t(basicDFD.descriptorBlockSize) + sizeof(dfdTotalSize) > header.dfdByteLength)
    {
      return "KTX2 Data Format Descriptor was invalid (the basic data format descriptor descriptorBlockSize was "
             + std::to_string(basicDFD.descriptorBlockSize)
             + " - that plus 4 bytes for totalSize was larger than the length of the whole data format descriptor, "
             + std::to_string(dfdTotalSize) + ")";
    }
    if(size_t(basicDFD.descriptorBlockSize) < sizeof(BasicDataFormatDescriptor))
    {
      return "KTX2 Data Format Descriptor was invalid (the basic data format descriptor descriptorBlockSize was "
             + std::to_string(basicDFD.descriptorBlockSize) + ", which was shorter than the size of the basic DFD itself, "
             + std::to_string(sizeof(BasicDataFormatDescriptor)) + ")";
    }

    const size_t numDFDSamples = (size_t(basicDFD.descriptorBlockSize) - sizeof(BasicDataFormatDescriptor)) / sizeof(DFSample);
    UNWRAP_ERROR(resizeVectorOrError(dfdSamples, numDFDSamples));
    // If numDFDSamples is 0, dfdSamples.data() can be nullptr, and passing nullptr to memcpy() is undefined behavior.
    if(numDFDSamples != 0)
    {
      memcpy(dfdSamples.data(), reinterpret_cast<char*>(dfd.data()) + sizeof(dfdTotalSize) + sizeof(BasicDataFormatDescriptor),
             numDFDSamples * sizeof(DFSample));
    }
  }

  isPremultiplied = false;
  isSrgb          = true;
  if(basicDFDExists)
  {
    if((basicDFD.flags & KHR_DF_FLAG_ALPHA_PREMULTIPLIED) != 0)
    {
      isPremultiplied = true;
    }

    if(basicDFD.transferFunction == KHR_DF_TRANSFER_SRGB)
    {
      isSrgb = true;
    }
    else if(basicDFD.transferFunction == KHR_DF_TRANSFER_LINEAR)
    {
      isSrgb = false;
    }
    else
    {
      return "KTX2 Data Format Descriptor had an unhandled transferFunction (" + std::to_string(basicDFD.transferFunction) + ")";
    }

    m_fileInfo.ktx2ColorModel = basicDFD.colorModel;

    if(basicDFD.colorModel == KHR_DF_MODEL_UASTC)
    {
#ifdef NVP_SUPPORTS_BASISU
      if(readSettings.deviceSupportsAstc)
      {
        // Prefer ASTC, since then transcoding is lossless:
        format = isSrgb ? VK_FORMAT_ASTC_4x4_SRGB_BLOCK : VK_FORMAT_ASTC_4x4_UNORM_BLOCK;
      }
      else
      {
        // Otherwise, BC7 is preferred:
        format = isSrgb ? VK_FORMAT_BC7_SRGB_BLOCK : VK_FORMAT_BC7_UNORM_BLOCK;
      }
#else
      return "KTX2 color model was Basis UASTC, but NVP_SUPPORTS_BASISU was not defined.";
#endif
    }
    else if(basicDFD.colorModel == KHR_DF_MODEL_ETC1S)
    {
#ifdef NVP_SUPPORTS_BASISU
      // There are four ETC1S channel possibilities. The final format is
      // BC4 for RRR, BC5 for RRR+GGG, and BC7 for RGB and RGB+AAA.
      m_fileInfo.ktx2BasisEtc1sNumSlices = dfdSamples.size();
      if(dfdSamples.size() == 1)
      {
        // Must be RRR or RGB
        if(dfdSamples[0].channelType == KHR_DF_CHANNEL_ETC1S_RRR)
        {
          m_fileInfo.ktx2BasisEtc1sCombination = Etc1sCombination::R;
          format                               = VK_FORMAT_BC4_UNORM_BLOCK;
        }
        else if(dfdSamples[0].channelType == KHR_DF_CHANNEL_ETC1S_RGB)
        {
          m_fileInfo.ktx2BasisEtc1sCombination = Etc1sCombination::RGB;
          format                               = isSrgb ? VK_FORMAT_BC7_SRGB_BLOCK : VK_FORMAT_BC7_UNORM_BLOCK;
        }
        else
        {
          return "KTX2 color model was Basis ETC1S, but there was one slice of an unknown channel type ("
                 + std::to_string(dfdSamples[0].channelType) + ")";
        }
      }
      else if(dfdSamples.size() == 2)
      {
        // Must be RRR+GGG or RGB+AAA; we disallow listing these channels
        // in a different order.
        if(dfdSamples[0].channelType == KHR_DF_CHANNEL_ETC1S_RRR && dfdSamples[1].channelType == KHR_DF_CHANNEL_ETC1S_GGG)
        {
          m_fileInfo.ktx2BasisEtc1sCombination = Etc1sCombination::RG;
          format                               = VK_FORMAT_BC5_UNORM_BLOCK;
        }
        else if(dfdSamples[0].channelType == KHR_DF_CHANNEL_ETC1S_RGB && dfdSamples[1].channelType == KHR_DF_CHANNEL_ETC1S_AAA)
        {
          m_fileInfo.ktx2BasisEtc1sCombination = Etc1sCombination::RGBA;
          format                               = isSrgb ? VK_FORMAT_BC7_SRGB_BLOCK : VK_FORMAT_BC7_UNORM_BLOCK;
        }
        else
        {
          return "KTX2 color model was Basis ETC1S and there were two slices, but the channel types ("
                 + std::to_string(dfdSamples[0].channelType) + " and " + std::to_string(dfdSamples[1].channelType) + " were unknown";
        }
      }
      else
      {
        return "KTX2 color model was Basis ETC1S, but there were an unusual number of slices ("
               + std::to_string(dfdSamples.size()) + ", should be 1 or 2)";
      }
#else
      return "KTX2 color model was Basis ETC1S, but NVP_SUPPORTS_BASISU was not defined.";
#endif
    }
    else if(format == VK_FORMAT_UNDEFINED)
    {
      return "KTX2 VkFormat was VK_FORMAT_UNDEFINED, but the Data Format Descriptor block had unrecognized "
             "colorModel number "
             + std::to_string(basicDFD.colorModel) + ".";
    }
  }

  // Perform additional validation to rule out invalid Basis+format+supercompression
  // combinations. Not doing these checks can lead to surprising behavior!
  // Basis ETC1S must only appear with supercompression mode 1, and vice versa.
  if((basicDFD.colorModel == KHR_DF_MODEL_ETC1S) != (header.supercompressionScheme == 1))
  {
    return "KTX2 file was invalid - the Basis ETC1S flag didn't match whether supercompression scheme 1 (BasisLZ) was used.";
  }

  //---------------------------------------------------------------------------
  // Read section 4, key/value data. To do this, we can read kvdByteLength
  // bytes, then extract keys and values from that.
  UNWRAP_ERROR(readKeyValueData(input, header.kvdByteLength, false, keyValueData));

  // Parse the ktxSwizzle value if it exists.
  {
    swizzle          = {Swizzle::R, Swizzle::G, Swizzle::B, Swizzle::A};
    const auto kvpIt = keyValueData.find("KTXswizzle");
    if(kvpIt != keyValueData.end())
    {
      // Read up to 4 characters (slightly less constrained than the spec)
      const std::vector<char> value = kvpIt->second;
      // Value should end with a NULL character, but if it doesn't that's OK
      const size_t charsToRead = std::min(size_t(4), value.size());
      for(size_t i = 0; i < charsToRead; i++)
      {
        switch(value[i])
        {
          case 'r':
            swizzle[i] = Swizzle::R;
            break;
          case 'g':
            swizzle[i] = Swizzle::G;
            break;
          case 'b':
            swizzle[i] = Swizzle::B;
            break;
          case 'a':
            swizzle[i] = Swizzle::A;
            break;
          case '0':
            swizzle[i] = Swizzle::ZERO;
            break;
          case '1':
            swizzle[i] = Swizzle::ONE;
            break;
          default:
            break;
        }
      }
    }
  }

  //---------------------------------------------------------------------------
  // Section 6, supercompression global data.
  // First check the sgdByteLength because we allocate memory based off it.
  // Values that are really large are allowed by the KTX2 spec, but someone could
  // use this to cause an out-of-memory error.
  if(header.sgdByteLength > readSettings.maxSizeInBytes)
  {
    return "Supercompression global data length (sgdByteLength) was over the maximum size specified in the "
           "ReadSettings object! The file is either invalid, or if this is intentional, "
           "ReadSettings::maxSizeInBytes should be set to a larger value.";
  }

  m_fileInfo.ktx2SupercompressionScheme = header.supercompressionScheme;
  m_fileInfo.ktx2GlobalDataOffset       = header.sgdByteOffset;
  m_fileInfo.ktx2GlobalDataByteSize     = header.sgdByteLength;

  //---------------------------------------------------------------------------
  // Section 7, Mip Level Array.
  // Here we set up subresource layouts.
  UNWRAP_ERROR(resizeVectorOrError(m_subresourceLayouts, numSubresources));
  size_t remainingAllowedUncompressedBytes = readSettings.maxSizeInBytes;

  for(uint32_t mip = 0; mip < numMips; ++mip)
  {
    const SubresourceLayout& levelIndex = m_levelIndices[mip];
    const size_t             mipWidth   = std::max(1u, header.pixelWidth >> mip);
    const size_t             mipHeight  = std::max(1u, header.pixelHeight >> mip);
    const size_t             mipDepth   = std::max(1u, header.pixelDepth >> mip);

    size_t finalFaceSize = 0;
    UNWRAP_ERROR(exportSizeExtended(mipWidth, mipHeight, mipDepth, format, finalFaceSize, readSettings.customSizeCallback));
    // Check that the amount of data we'll allocate doesn't go past
    // max_uncompressed_size_in_bytes:
    {
      size_t maxUncompressedMipSize = 0;
      if(!checked_math::mul3(finalFaceSize, header.layerCount, header.faceCount, maxUncompressedMipSize))
      {
        return "The number of uncompressed bytes to store decompressed mip " + std::to_string(mip) + " would have overflowed a size_t.";
      }
      maxUncompressedMipSize = std::max(maxUncompressedMipSize, levelIndex.uncompressedByteSize);

      if(remainingAllowedUncompressedBytes < maxUncompressedMipSize)
      {
        return "This file would require more than the limit of maxSizeInBytes = "
               + std::to_string(readSettings.maxSizeInBytes) + " bytes without supercompression.";
      }
      remainingAllowedUncompressedBytes -= maxUncompressedMipSize;
    }

    // Validate sizes
    if(readSettings.validateInputSize)
    {
      // Level-wide constraint on read data
      if(levelIndex.fileByteSize > validationInputSize)
      {
        return "The KTX2 file said that level " + std::to_string(mip) + " contained " + std::to_string(levelIndex.fileByteSize)
               + " bytes of supercompressed data, but the file was only " + std::to_string(validationInputSize) + " bytes long!";
      }

      if(header.supercompressionScheme == 0)
      {
        // Level-wide constraint on read data
        if(levelIndex.uncompressedByteSize > validationInputSize)
        {
          return "The KTX2 file said no supercompression was used and that level " + std::to_string(mip) + " contained "
                 + std::to_string(levelIndex.uncompressedByteSize) + " bytes of data, but the file was only "
                 + std::to_string(validationInputSize) + " bytes long!";
        }

        // Per-face more specific constraint, making use of how non-supercompressed
        // UASTC and ASTC (the transcoded-to format) are both 128 bits/block.
        if((validationInputSize / size_t(header.layerCount)) / size_t(header.faceCount) < finalFaceSize)
        {
          return "The KTX2 file said it contained " + std::to_string(header.layerCount) + " array elements and "
                 + std::to_string(header.faceCount) + " faces in mip " + std::to_string(mip)
                 + ", but the input was too short (" + std::to_string(validationInputSize) + " bytes) to contain that!";
        }
      }
    }

    size_t fileOffset = levelIndex.fileOffset;
    for(uint32_t layer = 0; layer < header.layerCount; layer++)
    {
      for(uint32_t face = 0; face < header.faceCount; face++)
      {
        SubresourceLayout& layout = subresourceLayout(mip, layer, face);

        // If we're decompressing per-mip:
        if(m_fileInfo.ktx2SupercompressionScheme == uint32_t(SupercompressionScheme::eZstd)
           || m_fileInfo.ktx2SupercompressionScheme == uint32_t(SupercompressionScheme::eZlib))
        {
          layout = levelIndex;
        }
        else
        {
          layout.fileOffset   = fileOffset;
          layout.fileByteSize = finalFaceSize;
        }

        layout.uncompressedByteSize = finalFaceSize;
        if(layout.fileOffset > std::numeric_limits<size_t>::max() - finalFaceSize)
        {
          return "Computing file offsets would have overflowed a size_t!";
        }
        fileOffset += finalFaceSize;
      }
    }
  }

  return {};
}

// readSubresourcesFromStream() backend for a KTX2 file.
ErrorWithText Image::readSubresourcesFromKTX2Stream(std::istream& input, const SubresourceRange& range, SubresourceTarget* outSubresources)
{
  const std::streampos startPos = input.tellg();

  // First set up global decompression objects:
  std::vector<uint8_t> supercompressionGlobalData;
  if(m_fileInfo.ktx2GlobalDataByteSize > 0)
  {
    UNWRAP_ERROR(resizeVectorOrError(supercompressionGlobalData, m_fileInfo.ktx2GlobalDataByteSize));
    if(!input.seekg(m_fileInfo.ktx2GlobalDataOffset + startPos, std::ios::beg))
    {
      return "Seeking to supercompression global data failed.";
    }
    if(!input.read(reinterpret_cast<char*>(supercompressionGlobalData.data()), m_fileInfo.ktx2GlobalDataByteSize))
    {
      return "Reading supercompressionGlobalData failed.";
    }
  }

#ifdef NVP_SUPPORTS_ZSTD
  ScopedZstdDContext zstdDCtx;
#endif
#ifdef NVP_SUPPORTS_BASISU
  BasisLZDecompressionObjects basisLZDCtx;
  // Basis ETC1S supports a sort of video format, where there are I-frames
  // and P-frames and frames correspond to array elements. The Basis code
  // currently allows this if there's a KTXanimData key, or if there are
  // P-frames indicated in the supercompression image descriptions.
  // We diverge slightly from Basis here and require videos to be 2D; Basis
  // technically allows cubemap arrays with KTXanimData set to be interpreted
  // as videos, I think.
  bool isVideo = false;
#endif
  if(m_fileInfo.ktx2SupercompressionScheme == uint32_t(SupercompressionScheme::eBasisLZ))
  {
#ifdef NVP_SUPPORTS_BASISU
    // Initialize supercompression global data
    UNWRAP_ERROR(BasisUSingleton::GetInstance().PrepareBasisLZObjects(basisLZDCtx, supercompressionGlobalData, numMips,
                                                                      std::max(1u, numLayersPossibly0), numFaces));
    // Video criterion; don't permit 1-frame videos following Basis here
    if(numFaces == 1 && numLayersPossibly0 > 1)
    {
      isVideo = (keyValueData.find("KTXanimData") != keyValueData.end());
      if(!isVideo)
      {
        for(const basist::ktx2_etc1s_image_desc& id : basisLZDCtx.etc1sImageDescs)
        {
          if(id.m_image_flags & KTX2_IMAGE_IS_P_FRAME)
          {
            isVideo = true;
            break;
          }
        }
      }
    }

    // Check validity of slice sizes
    for(const basist::ktx2_etc1s_image_desc& id : basisLZDCtx.etc1sImageDescs)
    {
      if(id.m_rgb_slice_byte_length == 0)
      {
        return "KTX2 stream was incorrectly formatted: a BasisLZ+ETC1S RGB slice had byte length 0.";
      }
      if((m_fileInfo.ktx2BasisEtc1sNumSlices == 2) && (id.m_alpha_slice_byte_length == 0))
      {
        return "KTX2 stream was incorrectly formatted: a BasisLZ+ETC1S alpha slice had byte length 0.";
      }
    }
#else
    return "KTX2 header specified BasisLZ supercompression, but NVP_SUPPORTS_BASISU was not defined.";
#endif
  }
  else if(m_fileInfo.ktx2SupercompressionScheme == uint32_t(SupercompressionScheme::eZstd))
  {
    // Set up Zstandard
#ifdef NVP_SUPPORTS_ZSTD
    zstdDCtx.Init();
    if(zstdDCtx.pCtx == nullptr)
    {
      return "Initializing Zstandard context failed.";
    }
#else
    return "KTX2 stream uses Zstandard supercompression, but nv_ktx was built without Zstd.";
#endif
  }
  else if(m_fileInfo.ktx2SupercompressionScheme == uint32_t(SupercompressionScheme::eZlib))
  {
// Nothing to do for Zlib, but check to ensure it's supported
#ifndef NVP_SUPPORTS_GZLIB
    return "KTX2 stream uses Zlib supercompression, but nv_ktx was built without Zlib.";
#endif
  }
  else if(m_fileInfo.ktx2SupercompressionScheme != uint32_t(SupercompressionScheme::eNone))
  {
    return "Does not know about supercompression scheme " + std::to_string(m_fileInfo.ktx2SupercompressionScheme) + ".";
  }

  // Read, inflate, and decompress each image in turn.
  //
  // For each mip:
  //   If uncompressed:
  //     Copy each subresource
  //   Else if Zstd:
  //     Zstd decompress the mip data
  //     Copy each subresource
  //   Else if Zlib:
  //     Zlib decompress the mip data
  //     Copy each subresource
  //   Else if Basis ETC1S+BasisLZ:
  //     BasisLZ decompress the mip data to 1 or 2 ETC1S slices.
  //     For each subresource
  //       Transcode it from ETC1S to the inflated VkFormat
  //
  //   Then: if UASTC:
  //     For each subresource
  //       Transcode it from UASTC to the inflated VkFormat

  // Temporary buffers used for supercompressed data.
  std::vector<char> supercompressedData;
  std::vector<char> inflatedData;
  // Traverse mips in reverse order following the spec
  // so that we move forwards through the file:
  for(int mip = int(range.firstMip + range.numMips) - 1; mip >= int(range.firstMip); mip--)
  {
    // FAST PATH - if no supercompression and no UASTC, we can read directly:
    if(!requiresComplexDecoding())
    {
      for(uint32_t layer = range.firstLayer; layer < range.firstLayer + range.numLayers; layer++)
      {
        for(uint32_t face = range.firstFace; face < range.firstFace + range.numFaces; face++)
        {
          const SubresourceLayout& source = getSubresourceLayout(mip, layer, face);
          if(!input.seekg(source.fileOffset + startPos, std::ios::beg))
          {
            return "Failed to seek to the data for mip " + std::to_string(mip) + " layer " + std::to_string(layer)
                   + " face " + std::to_string(face) + ". Is the stream truncated?";
          }

          const size_t targetIdx = ((mip - range.firstMip) * range.numLayers + (layer - range.firstLayer)) * range.numFaces
                                   + (face - range.firstFace);
          SubresourceTarget& target = outSubresources[targetIdx];

          if(!input.read(reinterpret_cast<char*>(target.data), source.fileByteSize))
          {
            return "Reading data for mip " + std::to_string(mip) + " layer " + std::to_string(layer) + " face "
                   + std::to_string(face) + " from the stream failed. Is the stream truncated?";
          }
        }
      }
    }
    else
    {
      // Seek to the start of that mip's data and read it. Note that this skips
      // over mipPadding.
      const SubresourceLayout& levelIndex = m_levelIndices[mip];
      if(!input.seekg(levelIndex.fileOffset + startPos, std::ios::beg))
      {
        return "Failed to seek to KTX2 mip " + std::to_string(mip) + " data!";
      }

      const size_t mipWidth         = std::max(1u, mip0Width >> mip);
      const size_t mipHeight        = std::max(1u, mip0Height >> mip);
      const size_t mipDepth         = std::max(1u, mip0Depth >> mip);
      const size_t inflatedFaceSize = getSubresourceLayout(mip, 0, 0).uncompressedByteSize;

      //               decompression     transcoding
      // supercompressedData -> inflatedData -> subresource
      //                             ^
      //                             |
      //                             in the ETC1S + UASTC case, we load file data into here directly
      //                             (it turns out ETC1S doesn't do anything per-level)
      if(m_fileInfo.ktx2SupercompressionScheme == uint32_t(SupercompressionScheme::eNone))
      {
        // UASTC, ETC1S: Load file data into inflatedData directly
        UNWRAP_ERROR(resizeVectorOrError(inflatedData, levelIndex.uncompressedByteSize));
        if(!input.read(inflatedData.data(), levelIndex.uncompressedByteSize))
        {
          return "Reading mip " + std::to_string(mip) + "'s data failed.";
        }
      }
      else if(m_fileInfo.ktx2SupercompressionScheme == uint32_t(SupercompressionScheme::eBasisLZ))
      {
        // ETC1S files often have uncompressedByteLength set to 0 for some reason.
        // In any case, we want to read the compressed byte length.
        // NOTE(nbickford): I think this can be combined with the above branch,
        // but will need to check to be 100% sure.
        UNWRAP_ERROR(resizeVectorOrError(inflatedData, levelIndex.fileByteSize));
        if(!input.read(inflatedData.data(), levelIndex.fileByteSize))
        {
          return "Reading mip " + std::to_string(mip) + "'s data failed.";
        }
      }
      else
      {
        // Read into supercompressedData
        UNWRAP_ERROR(resizeVectorOrError(supercompressedData, levelIndex.fileByteSize));
        if(!input.read(supercompressedData.data(), levelIndex.fileByteSize))
        {
          return "Reading mip " + std::to_string(mip) + "'s supercompressed data failed.";
        }

        // Inflate the supercompressed data. We must use another buffer for this.
        UNWRAP_ERROR(resizeVectorOrError(inflatedData, levelIndex.uncompressedByteSize));

        if(m_fileInfo.ktx2SupercompressionScheme == uint32_t(SupercompressionScheme::eZstd))
        {
          // Zstandard
          // NOTE(nbickford): Currently we decompress the entire mip the way
          // nv_ktx v1 did. But if less than the full mip was requested,
          // we don't need to do this! We could use the Zstandard streaming
          // API to skip over the bytes we don't need, read the bytes we need,
          // and skip the rest.
#ifdef NVP_SUPPORTS_ZSTD
          size_t zstdError = ZSTD_decompress(inflatedData.data(), inflatedData.size(),  //
                                             supercompressedData.data(), supercompressedData.size());
          if(ZSTD_isError(zstdError))
          {
            const char* zstdErrorName = ZSTD_getErrorName(zstdError);
            return "Mip " + std::to_string(mip) + " Zstandard inflation failed with the message '"
                   + std::string(zstdErrorName) + "' (code " + std::to_string(zstdError) + ").";
          }
#else
          assert(!"nv_ktx was compiled without Zstandard support, but the KTX stream was not rejected! This should never happen.");
#endif
        }
        else if(m_fileInfo.ktx2SupercompressionScheme == uint32_t(SupercompressionScheme::eZlib))
        {
          // Zlib
          // NOTE(nbickford): Same note as for Zstd about the streaming API.
#ifdef NVP_SUPPORTS_GZLIB
          ScopedZlibDStream zlibStream;
          int               zlibError = zlibStream.Init();
          if(zlibError != Z_OK)
          {
            return "Zlib initialization failed (error code " + std::to_string(zlibError) + ").";
          }
          if(supercompressedData.size() > UINT_MAX || inflatedData.size() > UINT_MAX)
          {
            return "Zlib compressed or decompressed data for mip " + std::to_string(mip) + " was larger than 4 GB.";
          }
          zlibStream.stream.next_in   = reinterpret_cast<Bytef*>(supercompressedData.data());
          zlibStream.stream.avail_in  = static_cast<uInt>(supercompressedData.size());
          zlibStream.stream.next_out  = reinterpret_cast<Bytef*>(inflatedData.data());
          zlibStream.stream.avail_out = static_cast<uInt>(inflatedData.size());
          zlibError                   = inflate(&zlibStream.stream, Z_NO_FLUSH);
          if(zlibError != Z_OK)
          {
            return "Zlib inflation failed (error code " + std::to_string(zlibError) + ").";
          }
          zlibStream.Free();
#else
          assert(!"nv_ktx was compiled without Zlib support, but the KTX stream was not rejected! This should never happen.");
#endif
        }
      }

      // Check size ahead of time to ensure we don't read out of bounds.
      // This would otherwise result in an access violation on
      // invalid_face_count_and_padding.ktx2, or on an otherwise truncated file.
      // This doesn't apply to ETC1S, because it does inflation and transcoding
      // all at once.
      if(m_fileInfo.ktx2SupercompressionScheme != uint32_t(SupercompressionScheme::eBasisLZ))
      {
        const size_t inflatedDataSize       = inflatedData.size();
        const size_t expectedBytesInThisMip = inflatedFaceSize * std::max(1u, numLayersPossibly0) * numFaces;
        if(expectedBytesInThisMip > inflatedDataSize)
        {
          return "Expected " + std::to_string(expectedBytesInThisMip) + " bytes in mip " + std::to_string(mip)
                 + ", but the inflated data was only " + std::to_string(inflatedDataSize) + " bytes long.";
        }
      }

      // Write into each subresource, possibly transcoding from the source
      // format to this->format (for UASTC and ETC1S).
      for(uint32_t layer = range.firstLayer; layer < range.firstLayer + range.numLayers; layer++)
      {
        for(uint32_t face = range.firstFace; face < range.firstFace + range.numFaces; face++)
        {
          // Read position in `inflatedData`
          const size_t inflatedDataPos = (layer * numFaces + face) * inflatedFaceSize;

          // Prepare the output buffer.
          const size_t targetIdx = ((mip - range.firstMip) * range.numLayers + (layer - range.firstLayer)) * range.numFaces
                                   + (face - range.firstFace);
          SubresourceTarget& target = outSubresources[targetIdx];

          if(m_fileInfo.ktx2ColorModel == KHR_DF_MODEL_UASTC)
          {
#ifdef NVP_SUPPORTS_BASISU
            const bool to_astc = (format == VK_FORMAT_ASTC_4x4_SRGB_BLOCK) || (format == VK_FORMAT_ASTC_4x4_UNORM_BLOCK);
            BasisUSingleton::GetInstance().TranscodeUastcToBc7OrAstc44(reinterpret_cast<char*>(target.data),
                                                                       &inflatedData[inflatedDataPos], mipWidth,
                                                                       mipHeight, mipDepth, to_astc);
#else
            assert(!"nv_ktx was compiled without Basis support, but the KTX stream was not rejected! This should never happen.");
#endif
          }
          else if(m_fileInfo.ktx2ColorModel == KHR_DF_MODEL_ETC1S)
          {
#ifdef NVP_SUPPORTS_BASISU
            // Get the inflated VkFormat in an enum Basis uses
            basist::transcoder_texture_format basisDstFmt{};
            switch(format)
            {
              case VK_FORMAT_BC4_UNORM_BLOCK:
                basisDstFmt = basist::transcoder_texture_format::cTFBC4;
                break;
              case VK_FORMAT_BC5_UNORM_BLOCK:
                basisDstFmt = basist::transcoder_texture_format::cTFBC5;
                break;
              case VK_FORMAT_BC7_SRGB_BLOCK:
              case VK_FORMAT_BC7_UNORM_BLOCK:
                basisDstFmt = basist::transcoder_texture_format::cTFBC7_RGBA;
                break;
              default:
                return "No Basis ETC1S transcoder_texture_format was specified for the destination transcode format! This should never happen.";
            }

            // Get the ETC1S image description
            const size_t etc1sImageIdx =
                (std::max(1u, numLayersPossibly0) * size_t(mip) + size_t(layer)) * size_t(numFaces) + size_t(face);
            const basist::ktx2_etc1s_image_desc imageDesc  = basisLZDCtx.etc1sImageDescs[etc1sImageIdx];
            const size_t                        numBlocksX = (mipWidth + 3) / 4;
            const size_t                        numBlocksY = (mipHeight + 3) / 4;

            if(!basisLZDCtx.etc1sTranscoder->transcode_image(
                   basisDstFmt,                                      // Basis destination format
                   target.data,                                      // Output data
                   uint32_t(numBlocksX * numBlocksY),                // Number of blocks in the output
                   reinterpret_cast<uint8_t*>(inflatedData.data()),  // Compressed data for this level
                   uint32_t(inflatedData.size()),                    // Compressed data length
                   uint32_t(numBlocksX), uint32_t(numBlocksY),       // Block dimensions
                   uint32_t(mipWidth), uint32_t(mipHeight),          // Pixel dimensions
                   uint32_t(mip),                                    // Mip number
                   imageDesc.m_rgb_slice_byte_offset, imageDesc.m_rgb_slice_byte_length,  // Range of first slice from the start of the compressed data
                   imageDesc.m_alpha_slice_byte_offset, imageDesc.m_alpha_slice_byte_length,  // Range of second slice from the start of the compressed data
                   0,                                                    // No need for nonstandard decoder flags here
                   (m_fileInfo.ktx2BasisEtc1sNumSlices == 2),            // Whether it has 2 slices or only 1
                   isVideo,                                              // Whether this is ETC1S video
                   0,                                                    // Output row pitch in blocks, or 0
                   &basisLZDCtx.ktx2TranscoderState.m_transcoder_state,  // Persistent transcoder state
                   false))                                               // Output in blocks, not pixels
            {
              return "Failed to decompress BasisLZ+ETC1S mip " + std::to_string(mip) + ", layer "
                     + std::to_string(layer) + ", face " + std::to_string(face) + "!";
            }
#else
            assert(!"nv_ktx was compiled without Basis support, but the KTX stream was not rejected! This should never happen.");
#endif
          }
          else
          {
            // Not UASTC or ETC1S, no transcoding needed
            // We've checked to make sure this is okay above, but double-check
            // here in case the behavior above changes in future versions of
            // the code.
            if(m_fileInfo.ktx2SupercompressionScheme == uint32_t(SupercompressionScheme::eBasisLZ))
            {
              return "Failed to read KTX2 file: BasisLZ supercompression was enabled, but control reached the non-BasisLZ copy. This should never happen.";
            }
            if(inflatedDataPos + inflatedFaceSize > inflatedData.size())
            {
              return "Failed to read KTX2 file: the size of the inflated data didn't match the expected size.";
            }
            memcpy(target.data, &inflatedData[inflatedDataPos], inflatedFaceSize);
          }
        }
      }
    }
  }

  return {};
}

//-----------------------------------------------------------------------------
// Writing functions

namespace {
// Sets the Data Format Descriptor information for ASTC LDR formats given the
// size of the block. Doesn't change the transfer function and flags.
void setAstcFlags(uint8_t xSize, uint8_t ySize, BasicDataFormatDescriptor& descriptor, std::vector<DFSample>& samples)
{
  descriptor.colorModel     = KHR_DF_MODEL_ASTC;
  descriptor.colorPrimaries = KHR_DF_PRIMARIES_BT709;
  // Don't override transferFunction and flags.
  assert(xSize > 0 && ySize > 0);
  descriptor.texelBlockDimension0 = xSize - 1;
  descriptor.texelBlockDimension1 = ySize - 1;
  descriptor.bytesPlane0          = 16;

  samples.resize(1);
  samples[0].bitLength   = 127;
  samples[0].channelType = KHR_DF_CHANNEL_ASTC_DATA;
  samples[0].lower       = 0;
  samples[0].upper       = UINT32_MAX;
}

template <class T>
size_t vectorByteSize(const std::vector<T>& vec)
{
  return vec.size() * sizeof(T);
}

std::vector<char> stringToCharVector(const std::string& str)
{
  const char*  cString        = str.c_str();
  const size_t strlenWithZero = str.size() + 1;
  return std::vector<char>(cString, cString + strlenWithZero);
}

// Returns the least common multiple of n and 4.
size_t lcm4(size_t n)
{
  if(n % 4 == 0)
  {
    return n;
  }
  else if(n % 2 == 0)
  {
    return n * 2;
  }
  else
  {
    return n * 4;
  }
}
}  // namespace


ErrorWithText Image::writeKTX2Stream(std::ostream& output, const WriteSettings& writeSettings)
{
  // This function is more difficult than DDS writing, since the header
  // contains offsets into the rest of the file.
  // The way this works is that we'll initially write a blank identifier and
  // top-level header, then write the rest of the file keeping track of the
  // different offsets. Then we'll rewind back to the start and write the
  // header and offsets.

  // For Basis formats, we actually use the Basis Universal KTX2 writer entirely.
  // The reason is that basisu::basis_compressor doesn't have a level of
  // abstraction between "compress a block of data" and
  // "write a .basis or .ktx2 file" - the functions to do so are declared as
  // private. KTX-Software gets around this by writing to a .basis stream
  // in-memory, then reading the .basis stream and getting the needed data,
  // which takes a few hundred lines of code. It feels safest and least cursed
  // to me to rely on the built-in Basis implementation here.

  //---------------------------------------------------------------------------
  // Common things for both writers.
  // Some dimension fields can be 0 to indicate the texture type; create copies
  // of them where these 0s have been changed to 1s for indexing.
  if(numMips < 1)
  {
    return "Failed to write KTX2 file: numMips (" + std::to_string(numMips) + ") was less than 1.";
  }
  const uint32_t numLayersOr1 = std::max(numLayersPossibly0, 1u);

  // First, apply modifications to the KTXswizzle information early so that
  // they're handled by both writers.
  {
    std::stringstream ktxSwizzle;
    for(int c = 0; c < 4; c++)
    {
      switch(swizzle[c])
      {
        case Swizzle::R:
          ktxSwizzle << "r";
          break;
        case Swizzle::G:
          ktxSwizzle << "g";
          break;
        case Swizzle::B:
          ktxSwizzle << "b";
          break;
        case Swizzle::A:
          ktxSwizzle << "a";
          break;
        case Swizzle::ZERO:
          ktxSwizzle << "0";
          break;
        case Swizzle::ONE:
          ktxSwizzle << "1";
          break;
        default:
          assert(!"Unknown Swizzle!");
          return "Internal error: Unknown Swizzle.";
          break;
      }
    }
    keyValueData["KTXswizzle"] = stringToCharVector(ktxSwizzle.str());
  }

  if(writeSettings.encodeRgba8ToFormat != EncodeRgba8ToFormat::NO)
  {
#ifndef NVP_SUPPORTS_BASISU
    return "Failed to write KTX2 file: encoding to a Basis format was specified, but NVP_SUPPORTS_BASISU was "
           "not defined.";
#else
    // Validate format
    if((format != VK_FORMAT_B8G8R8A8_SRGB) && (format != VK_FORMAT_B8G8R8A8_UNORM))
    {
      return "Failed to write KTX2 file: encoding RGBA8 data to a Basis format was specified, but the Vulkan format of "
             "the input data, "
             + std::to_string(format) + " wasn't VK_FORMAT_B8G8R8A8_UNORM or VK_FORMAT_B8G8R8A8_SRGB.";
    }

    // Basisu doesn't support volume textures fully yet, I think
    if(mip0Depth > 1)
    {
      return "Failed to write KTX2 file: Volumes with Basis compression aren't supported yet.";
    }

    BasisUSingleton::GetInstance().Initialize();
    basisu::basis_compressor_params params;
    params.m_perceptual                       = isSrgb;
    params.m_ktx2_srgb_transfer_func          = isSrgb;
    params.m_mip_gen                          = false;
    params.m_read_source_images               = false;
    params.m_write_output_basis_or_ktx2_files = false;
    params.m_status_output                    = false;
    params.m_debug                            = false;
    params.m_validate_etc1s                   = false;
    params.m_compression_level                = writeSettings.etc1sEncodingLevel;
    params.m_check_for_alpha                  = false;
    params.m_multithreading                   = true;
    params.m_create_ktx2_file                 = true;
    params.m_etc1s_quality_level              = 128;

    // Determine the texture type
    if((mip0Depth == 0) || (mip0Depth == 1))  // Avoid volumes for now
    {
      // [2D or cubemap] *
      if(numFaces == 6)
      {
        // Cubemap
        params.m_tex_type = basist::cBASISTexTypeCubemapArray;
      }
      else
      {
        if(numLayersPossibly0 == 0)
        {
          // 2D non-array
          params.m_tex_type = basist::cBASISTexType2D;
        }
        else
        {
          // 2D array
          params.m_tex_type = basist::cBASISTexType2DArray;
        }
      }
    }
    else
    {
      // Volumes; reject cubemaps
      if(numFaces == 6)
      {
        return "Cubemaps where each face is a volume are not supported in KTX2 according to section 4.1, Texture Type.";
      }
      else
      {
        params.m_tex_type = basist::cBASISTexTypeVolume;
      }
    }

    // UASTC vs. ETC1S
    if(writeSettings.encodeRgba8ToFormat == EncodeRgba8ToFormat::UASTC)
    {
      params.m_uastc                    = true;
      params.m_pack_uastc_ldr_4x4_flags = static_cast<uint32_t>(writeSettings.uastcEncodingQuality);
      params.m_force_alpha              = true;
      if(writeSettings.supercompression == SupercompressionScheme::eZstd)
      {
        params.m_rdo_uastc_ldr_4x4                = true;
        params.m_rdo_uastc_ldr_4x4_quality_scalar = writeSettings.rdoLambda;
        params.m_rdo_uastc_ldr_4x4_multithreading = true;
        params.m_ktx2_uastc_supercompression      = basist::ktx2_supercompression::KTX2_SS_ZSTANDARD;
        params.m_ktx2_zstd_supercompression_level = writeSettings.supercompressionLevel;
      }
    }
    else
    {
      params.m_force_alpha         = (writeSettings.encodeRgba8ToFormat == EncodeRgba8ToFormat::ETC1S_RGBA);
      params.m_etc1s_quality_level = std::max(0, std::min((writeSettings.etc1sEncodingLevel * 255) / 6, 255));
      //params.m_global_sel_pal = true; // Enabling this seems to make things very slow
      params.m_uastc = false;
      // KTX-Software currently sets these to true if the input is a normal map.
      params.m_no_endpoint_rdo = !writeSettings.rdoEtc1s;
      params.m_no_selector_rdo = !writeSettings.rdoEtc1s;
      if(!writeSettings.rdoEtc1s)
      {
        params.m_compression_level = 0;
      }
    }

    // Create a job pool for multithreading
    basisu::job_pool jobPool(std::thread::hardware_concurrency());
    params.m_pJob_pool = &jobPool;

    // Copy key/value data, except for KTXwriter, since basisu will make its
    // own key for that.
    for(const auto& kvp : keyValueData)
    {
      if(kvp.first == "KTXwriter")
        continue;
      params.m_ktx2_key_values.push_back(basist::ktx2_transcoder::key_value());
      basist::ktx2_transcoder::key_value& outKVP = params.m_ktx2_key_values.back();
      if((kvp.first.size() > (UINT32_MAX - 1)) || (kvp.second.size() > UINT32_MAX))
      {
        return "A key/value pair was too large for Basis' KTX2 writer.";
      }
      outKVP.m_key.append(reinterpret_cast<const uint8_t*>(kvp.first.data()), static_cast<uint32_t>(kvp.first.size()));
      outKVP.m_key.push_back(0);
      outKVP.m_value.append(reinterpret_cast<const uint8_t*>(kvp.second.data()), static_cast<uint32_t>(kvp.second.size()));
    }

    // Copy image data. I believe these are in KTX2 order.
    params.m_source_images.reserve(size_t(numLayersOr1) * size_t(numFaces));
    params.m_source_mipmap_images.reserve(size_t(numLayersOr1) * size_t(numFaces));
    for(uint32_t layer = 0; layer < numLayersOr1; layer++)
    {
      for(uint32_t face = 0; face < numFaces; face++)
      {
        if(numMips > 1)
        {
          params.m_source_mipmap_images.push_back(basisu::vector<basisu::image>());
          params.m_source_mipmap_images.back().reserve(size_t(numMips) - 1);
        }
        for(uint32_t mip = 0; mip < numMips; mip++)
        {
          std::vector<char>& thisSubresource = subresource(mip, layer, face);
          const uint32_t     width           = std::max(1u, mip0Width >> mip);
          const uint32_t     height          = std::max(1u, mip0Height >> mip);
          const size_t       widthS          = static_cast<size_t>(width);
          const size_t       heightS         = static_cast<size_t>(height);
          // Mip 0 images go in m_source_images, while higher mips go in m_source_mipmap_images.
          if(mip == 0)
          {
            params.m_source_images.push_back(basisu::image(width, height));
          }
          else
          {
            params.m_source_mipmap_images.back().push_back(basisu::image(width, height));
          }
          basisu::image& out_image = (mip == 0 ? params.m_source_images.back() : params.m_source_mipmap_images.back().back());

          for(size_t y = 0; y < heightS; y++)
          {
            for(size_t x = 0; x < widthS; x++)
            {
              // Workaround for an issue where basisu ignores m_check_for_alpha set to false if user-supplied mips are provided
              const uint8_t a = params.m_force_alpha ? static_cast<uint8_t>(thisSubresource[(widthS * y + x) * 4 + 3]) : 255;
              out_image(uint32_t(x), uint32_t(y))
                  .set(static_cast<uint8_t>(thisSubresource[(widthS * y + x) * 4 + 2]),  // R from B
                       static_cast<uint8_t>(thisSubresource[(widthS * y + x) * 4 + 1]),  // G from G
                       static_cast<uint8_t>(thisSubresource[(widthS * y + x) * 4 + 0]),  // B from R
                       a);                                                               // A from A
            }
          }
        }
      }
    }

    // Create the KTX2 data!
    basisu::basis_compressor basisCompressor;
    // basisu::enable_debug_printf(true); // Uncomment this to print out status messages

    if(!basisCompressor.init(params))
    {
      return "Failed to initialize Basis Universal compressor.";
    }

    const basisu::basis_compressor::error_code basisResult = basisCompressor.process();
    switch(basisResult)
    {
      case basisu::basis_compressor::cECSuccess:
        break;
      case basisu::basis_compressor::cECFailedReadingSourceImages:
        return "Basis Universal compressor failed to read source images.";
      case basisu::basis_compressor::cECFailedValidating:
        return "Basis Universal compressor input failed validation (most likely an error in the Texture Tools "
               "Exporter).";
      case basisu::basis_compressor::cECFailedEncodeUASTC:
        return "Basis Universal compressor failed to encode to UASTC.";
      case basisu::basis_compressor::cECFailedFrontEnd:
        return "Basis Universal compressor failed in the ETC1S frontend.";
      case basisu::basis_compressor::cECFailedFontendExtract:
        return "Basis Universal compressor failed to extract data from the ETC1S frontend.";
      case basisu::basis_compressor::cECFailedBackend:
        return "Basis Universal compressor failed during BasisLZ compression.";
      case basisu::basis_compressor::cECFailedCreateBasisFile:
        return "Basis Universal compressor failed when creating Basis-formatted output.";
      case basisu::basis_compressor::cECFailedWritingOutput:
        return "Basis Universal compressor failed when writing the output file.";
      case basisu::basis_compressor::cECFailedUASTCRDOPostProcess:
        return "Basis Universal compressor failed when performing the UASTC Rate-Distortion Optimization post-process.";
      case basisu::basis_compressor::cECFailedCreateKTX2File:
        return "Basis Universal compressor to create a KTX2 file.";
      case basisu::basis_compressor::cECFailedInitializing:
        return "Basis Universal failed initializing.";
      default:
        return "Basis Universal error.";
    }

    // Write it out to the stream!
    if(!output.write(reinterpret_cast<const char*>(basisCompressor.get_output_ktx2_file().data()),
                     basisCompressor.get_output_ktx2_file().size()))
    {
      return "Basis Universal compressor succeeded, but the I/O operation of writing the compressed data to a stream "
             "failed! Is the file in use or the location requires administrator permissions?";
    }

    return {};
#endif
  }

  //---------------------------------------------------------------------------
  // Normal KTX2 writing
  const std::streampos startPos = output.tellp();

  // Allocate the header and Level Index.
  KTX2TopLevelHeader             header{};
  std::vector<SubresourceLayout> levelIndex(numMips);  // "mip offsets"

  // Write the header (we'll write it again), then zeros up to the Data Format Descriptor.
  if(!output.write(reinterpret_cast<const char*>(ktx2Identifier), IDENTIFIER_LEN))
  {
    return "Failed to write identifier the first time. Is the stream writable?";
  }
  if(!output.write(reinterpret_cast<char*>(&header), sizeof(header)))
  {
    return "Failed to write zeros for header.";
  }
  if(!output.write(reinterpret_cast<char*>(levelIndex.data()), vectorByteSize(levelIndex)))
  {
    return "Failed to write zeros for level index.";
  }

  //---------------------------------------------------------------------------
  // Write the Data Format Descriptor.
  // First, we know header.dfdByteOffset now.
  header.dfdByteOffset = static_cast<uint32_t>(output.tellp() - startPos);
  // Since the Khronos Data Format specifies
  // data format descriptors for most of the formats we support, we base
  // things off that.
  // Unfortunately, BC2, BC3, and BC5 use two samples, so we need to use a vector here:
  std::vector<DFSample>     dfSamples(1);  // Can be resized
  BasicDataFormatDescriptor dfdBlock{};
  // Set some initial fields
  dfdBlock.descriptorType = 0;
  dfdBlock.vendorId       = 0;  // Khronos
  dfdBlock.versionNumber  = 2;
  dfdBlock.colorPrimaries = KHR_DF_PRIMARIES_BT709;
  // Note that we don't use the transferFunctions in the examples, since we
  // specify a transfer function for each format.
  dfdBlock.transferFunction = isSrgb ? KHR_DF_TRANSFER_SRGB : KHR_DF_TRANSFER_LINEAR;
  // Similarly with premultiplied alpha:
  if(isPremultiplied)
  {
    dfdBlock.flags |= KHR_DF_FLAG_ALPHA_PREMULTIPLIED;
  }
  else
  {
    dfdBlock.flags |= KHR_DF_FLAG_ALPHA_STRAIGHT;
  }

  // Switch over formats
  // The only case where VkFormat isn't enough is BGRX vs. BGRA, where we have
  // to do something special for the BGRX case.
  switch(format)
  {
    case VK_FORMAT_BC7_UNORM_BLOCK:
    case VK_FORMAT_BC7_SRGB_BLOCK:
      // BC7
      dfdBlock.colorModel           = KHR_DF_MODEL_BC7;
      dfdBlock.colorPrimaries       = KHR_DF_PRIMARIES_BT709;
      dfdBlock.texelBlockDimension0 = 3;
      dfdBlock.texelBlockDimension1 = 3;
      dfdBlock.bytesPlane0          = 16;
      dfSamples[0].bitLength        = 127;
      dfSamples[0].channelType      = KHR_DF_CHANNEL_BC7_COLOR;
      dfSamples[0].upper            = UINT32_MAX;
      break;
    case VK_FORMAT_BC6H_SFLOAT_BLOCK:
      // BC6H signed
      dfdBlock.colorModel           = KHR_DF_MODEL_BC6H;
      dfdBlock.colorPrimaries       = KHR_DF_PRIMARIES_BT709;
      dfdBlock.texelBlockDimension0 = 3;
      dfdBlock.texelBlockDimension1 = 3;
      dfdBlock.bytesPlane0          = 16;
      dfSamples[0].bitLength        = 127;
      dfSamples[0].channelType =
          uint8_t(KHR_DF_CHANNEL_BC6H_COLOR) | uint8_t(KHR_DF_SAMPLE_DATATYPE_FLOAT | KHR_DF_SAMPLE_DATATYPE_SIGNED);
      dfSamples[0].lower = 0xBF800000u;  // -1.0f
      dfSamples[0].upper = 0x3F800000u;  // 1.0f
      break;
    case VK_FORMAT_BC6H_UFLOAT_BLOCK:
      // BC6H unsigned
      dfdBlock.colorModel           = KHR_DF_MODEL_BC6H;
      dfdBlock.colorPrimaries       = KHR_DF_PRIMARIES_BT709;
      dfdBlock.texelBlockDimension0 = 3;
      dfdBlock.texelBlockDimension1 = 3;
      dfdBlock.bytesPlane0          = 16;
      dfSamples[0].bitLength        = 127;
      dfSamples[0].channelType      = uint8_t(KHR_DF_CHANNEL_BC6H_COLOR) | uint8_t(KHR_DF_SAMPLE_DATATYPE_FLOAT);
      dfSamples[0].lower            = 0;            // 0.0f
      dfSamples[0].upper            = 0x3F800000u;  // 1.0f
      break;
    case VK_FORMAT_ASTC_4x4_UNORM_BLOCK:
    case VK_FORMAT_ASTC_4x4_SRGB_BLOCK:
      setAstcFlags(4, 4, dfdBlock, dfSamples);
      break;
    case VK_FORMAT_ASTC_5x4_UNORM_BLOCK:
    case VK_FORMAT_ASTC_5x4_SRGB_BLOCK:
      setAstcFlags(5, 4, dfdBlock, dfSamples);
      break;
    case VK_FORMAT_ASTC_5x5_UNORM_BLOCK:
    case VK_FORMAT_ASTC_5x5_SRGB_BLOCK:
      setAstcFlags(5, 5, dfdBlock, dfSamples);
      break;
    case VK_FORMAT_ASTC_6x5_UNORM_BLOCK:
    case VK_FORMAT_ASTC_6x5_SRGB_BLOCK:
      setAstcFlags(6, 5, dfdBlock, dfSamples);
      break;
    case VK_FORMAT_ASTC_6x6_UNORM_BLOCK:
    case VK_FORMAT_ASTC_6x6_SRGB_BLOCK:
      setAstcFlags(6, 6, dfdBlock, dfSamples);
      break;
    case VK_FORMAT_ASTC_8x5_UNORM_BLOCK:
    case VK_FORMAT_ASTC_8x5_SRGB_BLOCK:
      setAstcFlags(8, 5, dfdBlock, dfSamples);
      break;
    case VK_FORMAT_ASTC_8x6_UNORM_BLOCK:
    case VK_FORMAT_ASTC_8x6_SRGB_BLOCK:
      setAstcFlags(8, 6, dfdBlock, dfSamples);
      break;
    case VK_FORMAT_ASTC_8x8_UNORM_BLOCK:
    case VK_FORMAT_ASTC_8x8_SRGB_BLOCK:
      setAstcFlags(8, 8, dfdBlock, dfSamples);
      break;
    case VK_FORMAT_ASTC_10x5_UNORM_BLOCK:
    case VK_FORMAT_ASTC_10x5_SRGB_BLOCK:
      setAstcFlags(10, 5, dfdBlock, dfSamples);
      break;
    case VK_FORMAT_ASTC_10x6_UNORM_BLOCK:
    case VK_FORMAT_ASTC_10x6_SRGB_BLOCK:
      setAstcFlags(10, 6, dfdBlock, dfSamples);
      break;
    case VK_FORMAT_ASTC_10x8_UNORM_BLOCK:
    case VK_FORMAT_ASTC_10x8_SRGB_BLOCK:
      setAstcFlags(10, 8, dfdBlock, dfSamples);
      break;
    case VK_FORMAT_ASTC_10x10_UNORM_BLOCK:
    case VK_FORMAT_ASTC_10x10_SRGB_BLOCK:
      setAstcFlags(10, 10, dfdBlock, dfSamples);
      break;
    case VK_FORMAT_ASTC_12x10_UNORM_BLOCK:
    case VK_FORMAT_ASTC_12x10_SRGB_BLOCK:
      setAstcFlags(12, 10, dfdBlock, dfSamples);
      break;
    case VK_FORMAT_ASTC_12x12_UNORM_BLOCK:
    case VK_FORMAT_ASTC_12x12_SRGB_BLOCK:
      setAstcFlags(12, 12, dfdBlock, dfSamples);
      break;
    case VK_FORMAT_BC5_UNORM_BLOCK:
      dfdBlock.colorModel           = KHR_DF_MODEL_BC5;
      dfdBlock.colorPrimaries       = KHR_DF_PRIMARIES_BT709;
      dfdBlock.texelBlockDimension0 = 3;
      dfdBlock.texelBlockDimension1 = 3;
      dfdBlock.bytesPlane0          = 16;
      dfSamples.resize(2);
      dfSamples[0].bitLength   = 63;
      dfSamples[0].channelType = KHR_DF_CHANNEL_BC5_RED;
      dfSamples[0].upper       = UINT32_MAX;
      dfSamples[1].bitLength   = 63;
      dfSamples[1].bitOffset   = 64;
      dfSamples[1].channelType = KHR_DF_CHANNEL_BC5_GREEN;
      dfSamples[1].upper       = UINT32_MAX;
      break;
    case VK_FORMAT_BC4_UNORM_BLOCK:
      dfdBlock.colorModel           = KHR_DF_MODEL_BC4;
      dfdBlock.colorPrimaries       = KHR_DF_PRIMARIES_BT709;
      dfdBlock.texelBlockDimension0 = 3;
      dfdBlock.texelBlockDimension1 = 3;
      dfdBlock.bytesPlane0          = 8;
      dfSamples[0].bitLength        = 63;
      dfSamples[0].channelType      = KHR_DF_CHANNEL_BC4_DATA;
      dfSamples[0].upper            = UINT32_MAX;
      break;
    case VK_FORMAT_BC3_UNORM_BLOCK:
    case VK_FORMAT_BC3_SRGB_BLOCK:
      // We actually switch between DXT4 and DXT5 here based on
      // premultiplication, following the examples in the spec.
      if(isPremultiplied)
      {
        dfdBlock.colorModel = KHR_DF_MODEL_DXT4;
      }
      else
      {
        dfdBlock.colorModel = KHR_DF_MODEL_DXT5;
      }
      dfdBlock.colorPrimaries       = KHR_DF_PRIMARIES_BT709;
      dfdBlock.texelBlockDimension0 = 3;
      dfdBlock.texelBlockDimension1 = 3;
      dfdBlock.bytesPlane0          = 16;
      dfSamples.resize(2);
      dfSamples[0].bitLength   = 63;
      dfSamples[0].channelType = uint8_t(KHR_DF_CHANNEL_BC3_ALPHA) | uint8_t(KHR_DF_SAMPLE_DATATYPE_LINEAR);
      dfSamples[0].upper       = UINT32_MAX;
      dfSamples[1].bitOffset   = 64;
      dfSamples[1].bitLength   = 63;
      dfSamples[1].channelType = KHR_DF_CHANNEL_BC3_COLOR;
      dfSamples[1].upper       = UINT32_MAX;
      break;
    case VK_FORMAT_BC2_UNORM_BLOCK:
    case VK_FORMAT_BC2_SRGB_BLOCK:
      // Same premultiplication situation here as BC3
      if(isPremultiplied)
      {
        dfdBlock.colorModel = KHR_DF_MODEL_DXT2;
      }
      else
      {
        dfdBlock.colorModel = KHR_DF_MODEL_DXT3;
      }
      dfdBlock.colorPrimaries       = KHR_DF_PRIMARIES_BT709;
      dfdBlock.texelBlockDimension0 = 3;
      dfdBlock.texelBlockDimension1 = 3;
      dfdBlock.bytesPlane0          = 16;
      dfSamples.resize(2);
      dfSamples[0].bitLength = 63;
      // The alpha channel must always be linear!
      dfSamples[0].channelType = uint8_t(KHR_DF_CHANNEL_BC2_ALPHA) | uint8_t(KHR_DF_SAMPLE_DATATYPE_LINEAR);
      dfSamples[0].upper       = UINT32_MAX;
      dfSamples[1].bitOffset   = 64;
      dfSamples[1].bitLength   = 63;
      dfSamples[1].channelType = KHR_DF_CHANNEL_BC2_COLOR;
      dfSamples[1].upper       = UINT32_MAX;
      break;
    case VK_FORMAT_BC1_RGBA_UNORM_BLOCK:
    case VK_FORMAT_BC1_RGBA_SRGB_BLOCK:
      // BC1a
      dfdBlock.colorModel           = KHR_DF_MODEL_DXT1A;
      dfdBlock.colorPrimaries       = KHR_DF_PRIMARIES_BT709;
      dfdBlock.texelBlockDimension0 = 3;
      dfdBlock.texelBlockDimension1 = 3;
      dfdBlock.bytesPlane0          = 8;
      dfSamples[0].bitLength        = 63;
      dfSamples[0].channelType      = KHR_DF_CHANNEL_BC1A_ALPHAPRESENT;
      dfSamples[0].upper            = UINT32_MAX;
      break;
    case VK_FORMAT_BC1_RGB_UNORM_BLOCK:
    case VK_FORMAT_BC1_RGB_SRGB_BLOCK:
      // BC1
      dfdBlock.colorModel           = KHR_DF_MODEL_DXT1A;
      dfdBlock.colorPrimaries       = KHR_DF_PRIMARIES_BT709;
      dfdBlock.texelBlockDimension0 = 3;
      dfdBlock.texelBlockDimension1 = 3;
      dfdBlock.bytesPlane0          = 8;
      dfSamples[0].bitLength        = 63;
      dfSamples[0].channelType      = KHR_DF_CHANNEL_BC1A_COLOR;
      dfSamples[0].upper            = UINT32_MAX;
      break;
    case VK_FORMAT_R8_UNORM:
    case VK_FORMAT_R8_SRGB:
      // R8
      dfdBlock.colorModel           = KHR_DF_MODEL_RGBSDA;
      dfdBlock.colorPrimaries       = KHR_DF_PRIMARIES_BT709;
      dfdBlock.texelBlockDimension0 = 0;
      dfdBlock.texelBlockDimension1 = 0;
      dfdBlock.bytesPlane0          = 1;

      dfSamples[0].bitOffset   = 0;
      dfSamples[0].bitLength   = 7;  // "8"
      dfSamples[0].channelType = KHR_DF_CHANNEL_RGBSDA_RED;
      dfSamples[0].upper       = 255;
      break;
    case VK_FORMAT_B8G8R8_UNORM:
    case VK_FORMAT_B8G8R8_SRGB:
      // B in byte 0, G in byte 1, R in byte 2
      dfdBlock.colorModel           = KHR_DF_MODEL_RGBSDA;
      dfdBlock.colorPrimaries       = KHR_DF_PRIMARIES_BT709;
      dfdBlock.texelBlockDimension0 = 0;
      dfdBlock.texelBlockDimension1 = 0;
      dfdBlock.bytesPlane0          = 3;

      dfSamples.resize(3);

      dfSamples[0].bitOffset   = 0;
      dfSamples[0].bitLength   = 7;  // "8"
      dfSamples[0].channelType = KHR_DF_CHANNEL_RGBSDA_BLUE;
      dfSamples[0].upper       = 255;

      dfSamples[1].bitOffset   = 8;
      dfSamples[1].bitLength   = 7;  // "8"
      dfSamples[1].channelType = KHR_DF_CHANNEL_RGBSDA_GREEN;
      dfSamples[1].upper       = 255;

      dfSamples[2].bitOffset   = 16;
      dfSamples[2].bitLength   = 7;  // "8"
      dfSamples[2].channelType = KHR_DF_CHANNEL_RGBSDA_RED;
      dfSamples[2].upper       = 255;
      break;
    case VK_FORMAT_B8G8R8A8_UNORM:
    case VK_FORMAT_B8G8R8A8_SRGB:
      dfdBlock.colorModel           = KHR_DF_MODEL_RGBSDA;
      dfdBlock.colorPrimaries       = KHR_DF_PRIMARIES_BT709;
      dfdBlock.texelBlockDimension0 = 0;
      dfdBlock.texelBlockDimension1 = 0;
      dfdBlock.bytesPlane0          = 4;

      // B in byte 0, G in byte 1, R in byte 2, A in byte 3
      dfSamples.resize(4);

      dfSamples[0].bitOffset   = 0;
      dfSamples[0].bitLength   = 7;  // "8"
      dfSamples[0].channelType = KHR_DF_CHANNEL_RGBSDA_BLUE;
      dfSamples[0].upper       = 255;

      dfSamples[1].bitOffset   = 8;
      dfSamples[1].bitLength   = 7;  // "8"
      dfSamples[1].channelType = KHR_DF_CHANNEL_RGBSDA_GREEN;
      dfSamples[1].upper       = 255;

      dfSamples[2].bitOffset   = 16;
      dfSamples[2].bitLength   = 7;  // "8"
      dfSamples[2].channelType = KHR_DF_CHANNEL_RGBSDA_RED;
      dfSamples[2].upper       = 255;

      dfSamples[3].bitOffset   = 24;
      dfSamples[3].bitLength   = 7;  // "8"
      dfSamples[3].channelType = uint8_t(KHR_DF_CHANNEL_RGBSDA_ALPHA) | uint8_t(KHR_DF_SAMPLE_DATATYPE_LINEAR);
      dfSamples[3].upper       = 255;
      break;
    case VK_FORMAT_R8G8B8_UNORM:
    case VK_FORMAT_R8G8B8_SRGB:
      // R in byte 0, G in byte 1, B in byte 2
      dfdBlock.colorModel           = KHR_DF_MODEL_RGBSDA;
      dfdBlock.colorPrimaries       = KHR_DF_PRIMARIES_BT709;
      dfdBlock.texelBlockDimension0 = 0;
      dfdBlock.texelBlockDimension1 = 0;
      dfdBlock.bytesPlane0          = 3;

      dfSamples.resize(3);

      dfSamples[0].bitOffset   = 0;
      dfSamples[0].bitLength   = 7;  // "8"
      dfSamples[0].channelType = KHR_DF_CHANNEL_RGBSDA_RED;
      dfSamples[0].upper       = 255;

      dfSamples[1].bitOffset   = 8;
      dfSamples[1].bitLength   = 7;  // "8"
      dfSamples[1].channelType = KHR_DF_CHANNEL_RGBSDA_GREEN;
      dfSamples[1].upper       = 255;

      dfSamples[2].bitOffset   = 16;
      dfSamples[2].bitLength   = 7;  // "8"
      dfSamples[2].channelType = KHR_DF_CHANNEL_RGBSDA_BLUE;
      dfSamples[2].upper       = 255;
      break;
    case VK_FORMAT_R8G8B8A8_UNORM:
    case VK_FORMAT_R8G8B8A8_SRGB:
      dfdBlock.colorModel           = KHR_DF_MODEL_RGBSDA;
      dfdBlock.colorPrimaries       = KHR_DF_PRIMARIES_BT709;
      dfdBlock.texelBlockDimension0 = 0;
      dfdBlock.texelBlockDimension1 = 0;
      dfdBlock.bytesPlane0          = 4;

      // B in byte 0, G in byte 1, R in byte 2, A in byte 3
      dfSamples.resize(4);

      dfSamples[0].bitOffset   = 0;
      dfSamples[0].bitLength   = 7;  // "8"
      dfSamples[0].channelType = KHR_DF_CHANNEL_RGBSDA_RED;
      dfSamples[0].upper       = 255;

      dfSamples[1].bitOffset   = 8;
      dfSamples[1].bitLength   = 7;  // "8"
      dfSamples[1].channelType = KHR_DF_CHANNEL_RGBSDA_GREEN;
      dfSamples[1].upper       = 255;

      dfSamples[2].bitOffset   = 16;
      dfSamples[2].bitLength   = 7;  // "8"
      dfSamples[2].channelType = KHR_DF_CHANNEL_RGBSDA_BLUE;
      dfSamples[2].upper       = 255;

      dfSamples[3].bitOffset   = 24;
      dfSamples[3].bitLength   = 7;  // "8"
      dfSamples[3].channelType = uint8_t(KHR_DF_CHANNEL_RGBSDA_ALPHA) | uint8_t(KHR_DF_SAMPLE_DATATYPE_LINEAR);
      dfSamples[3].upper       = 255;
      break;
    case VK_FORMAT_R16_SFLOAT:
      dfdBlock.colorModel           = KHR_DF_MODEL_RGBSDA;
      dfdBlock.colorPrimaries       = KHR_DF_PRIMARIES_BT709;
      dfdBlock.texelBlockDimension0 = 0;
      dfdBlock.texelBlockDimension1 = 0;
      dfdBlock.bytesPlane0          = sizeof(uint16_t);

      dfSamples[0].bitOffset = 0;
      dfSamples[0].bitLength = 15;  // "16"
      dfSamples[0].channelType =
          uint8_t(KHR_DF_CHANNEL_RGBSDA_RED) | uint8_t(KHR_DF_SAMPLE_DATATYPE_SIGNED | KHR_DF_SAMPLE_DATATYPE_FLOAT);
      // Yes, these are 32-bit floats, not 16-bit floats! From the spec
      dfSamples[0].lower = 0xBF800000u;  // -1.0f
      dfSamples[0].upper = 0x3F800000u;  // 1.0f
      break;
    case VK_FORMAT_R16G16_SFLOAT:
      dfdBlock.colorModel           = KHR_DF_MODEL_RGBSDA;
      dfdBlock.colorPrimaries       = KHR_DF_PRIMARIES_BT709;
      dfdBlock.texelBlockDimension0 = 0;
      dfdBlock.texelBlockDimension1 = 0;
      dfdBlock.bytesPlane0          = sizeof(uint16_t) * 2;

      dfSamples.resize(2);
      for(uint32_t c = 0; c < 2; c++)
      {
        dfSamples[c].bitOffset = 16 * c;
        dfSamples[c].bitLength = 15;           // "16"
        dfSamples[c].lower     = 0xBF800000u;  // -1.0f
        dfSamples[c].upper     = 0x3F800000u;  // 1.0f
      }

      dfSamples[0].channelType =
          uint8_t(KHR_DF_CHANNEL_RGBSDA_RED) | uint8_t(KHR_DF_SAMPLE_DATATYPE_SIGNED | KHR_DF_SAMPLE_DATATYPE_FLOAT);
      dfSamples[1].channelType =
          uint8_t(KHR_DF_CHANNEL_RGBSDA_GREEN) | uint8_t(KHR_DF_SAMPLE_DATATYPE_SIGNED | KHR_DF_SAMPLE_DATATYPE_FLOAT);
      break;
    case VK_FORMAT_R16G16B16A16_SFLOAT:
      dfdBlock.colorModel           = KHR_DF_MODEL_RGBSDA;
      dfdBlock.colorPrimaries       = KHR_DF_PRIMARIES_BT709;
      dfdBlock.texelBlockDimension0 = 0;
      dfdBlock.texelBlockDimension1 = 0;
      dfdBlock.bytesPlane0          = sizeof(uint16_t) * 4;

      dfSamples.resize(4);
      for(uint32_t c = 0; c < 4; c++)
      {
        dfSamples[c].bitOffset = 16 * c;
        dfSamples[c].bitLength = 15;           // "16"
        dfSamples[c].lower     = 0xBF800000u;  // -1.0f
        dfSamples[c].upper     = 0x3F800000u;  // 1.0f
      }

      dfSamples[0].channelType =
          uint8_t(KHR_DF_CHANNEL_RGBSDA_RED) | uint8_t(KHR_DF_SAMPLE_DATATYPE_SIGNED | KHR_DF_SAMPLE_DATATYPE_FLOAT);
      dfSamples[1].channelType =
          uint8_t(KHR_DF_CHANNEL_RGBSDA_GREEN) | uint8_t(KHR_DF_SAMPLE_DATATYPE_SIGNED | KHR_DF_SAMPLE_DATATYPE_FLOAT);
      dfSamples[2].channelType =
          uint8_t(KHR_DF_CHANNEL_RGBSDA_BLUE) | uint8_t(KHR_DF_SAMPLE_DATATYPE_SIGNED | KHR_DF_SAMPLE_DATATYPE_FLOAT);
      dfSamples[3].channelType =
          uint8_t(KHR_DF_CHANNEL_RGBSDA_ALPHA) | uint8_t(KHR_DF_SAMPLE_DATATYPE_SIGNED | KHR_DF_SAMPLE_DATATYPE_FLOAT);
      break;
    case VK_FORMAT_R32_SFLOAT:
      dfdBlock.colorModel           = KHR_DF_MODEL_RGBSDA;
      dfdBlock.colorPrimaries       = KHR_DF_PRIMARIES_BT709;
      dfdBlock.texelBlockDimension0 = 0;
      dfdBlock.texelBlockDimension1 = 0;
      dfdBlock.bytesPlane0          = sizeof(uint32_t);

      dfSamples[0].bitOffset = 0;
      dfSamples[0].bitLength = 31;  // "32"
      dfSamples[0].channelType =
          uint8_t(KHR_DF_CHANNEL_RGBSDA_RED) | uint8_t(KHR_DF_SAMPLE_DATATYPE_SIGNED | KHR_DF_SAMPLE_DATATYPE_FLOAT);
      dfSamples[0].lower = 0xBF800000u;  // -1.0f
      dfSamples[0].upper = 0x3F800000u;  // 1.0f
      break;
    case VK_FORMAT_R32G32_SFLOAT:
      dfdBlock.colorModel           = KHR_DF_MODEL_RGBSDA;
      dfdBlock.colorPrimaries       = KHR_DF_PRIMARIES_BT709;
      dfdBlock.texelBlockDimension0 = 0;
      dfdBlock.texelBlockDimension1 = 0;
      dfdBlock.bytesPlane0          = sizeof(uint32_t) * 2;

      dfSamples.resize(2);
      for(uint32_t c = 0; c < 2; c++)
      {
        dfSamples[c].bitOffset = 32 * c;
        dfSamples[c].bitLength = 31;           // "32"
        dfSamples[c].lower     = 0xBF800000u;  // -1.0f
        dfSamples[c].upper     = 0x3F800000u;  // 1.0f
      }

      dfSamples[0].channelType =
          uint8_t(KHR_DF_CHANNEL_RGBSDA_RED) | uint8_t(KHR_DF_SAMPLE_DATATYPE_SIGNED | KHR_DF_SAMPLE_DATATYPE_FLOAT);
      dfSamples[1].channelType =
          uint8_t(KHR_DF_CHANNEL_RGBSDA_GREEN) | uint8_t(KHR_DF_SAMPLE_DATATYPE_SIGNED | KHR_DF_SAMPLE_DATATYPE_FLOAT);
      break;
    case VK_FORMAT_R32G32B32A32_SFLOAT:
      dfdBlock.colorModel           = KHR_DF_MODEL_RGBSDA;
      dfdBlock.colorPrimaries       = KHR_DF_PRIMARIES_BT709;
      dfdBlock.texelBlockDimension0 = 0;
      dfdBlock.texelBlockDimension1 = 0;
      dfdBlock.bytesPlane0          = sizeof(uint32_t) * 4;

      dfSamples.resize(4);
      for(uint32_t c = 0; c < 4; c++)
      {
        dfSamples[c].bitOffset = 32 * c;
        dfSamples[c].bitLength = 31;           // "32"
        dfSamples[c].lower     = 0xBF800000u;  // -1.0f
        dfSamples[c].upper     = 0x3F800000u;  // 1.0f
      }

      dfSamples[0].channelType =
          uint8_t(KHR_DF_CHANNEL_RGBSDA_RED) | uint8_t(KHR_DF_SAMPLE_DATATYPE_SIGNED | KHR_DF_SAMPLE_DATATYPE_FLOAT);
      dfSamples[1].channelType =
          uint8_t(KHR_DF_CHANNEL_RGBSDA_GREEN) | uint8_t(KHR_DF_SAMPLE_DATATYPE_SIGNED | KHR_DF_SAMPLE_DATATYPE_FLOAT);
      dfSamples[2].channelType =
          uint8_t(KHR_DF_CHANNEL_RGBSDA_BLUE) | uint8_t(KHR_DF_SAMPLE_DATATYPE_SIGNED | KHR_DF_SAMPLE_DATATYPE_FLOAT);
      dfSamples[3].channelType =
          uint8_t(KHR_DF_CHANNEL_RGBSDA_ALPHA) | uint8_t(KHR_DF_SAMPLE_DATATYPE_SIGNED | KHR_DF_SAMPLE_DATATYPE_FLOAT);
      break;
    default:
      return "The writer has no method to write the Data Format Descriptor for VkFormat " + std::to_string(format) + ".";
  }

  // In the case of supercompressed formats, we have to set all bytesPlane
  // fields to 0 per the "DFD for Supercompressed Data" section.
  if(writeSettings.supercompression != SupercompressionScheme::eNone)
  {
    dfdBlock.bytesPlane0 = 0;
    dfdBlock.bytesPlane1 = 0;
    dfdBlock.bytesPlane2 = 0;
    dfdBlock.bytesPlane3 = 0;
    dfdBlock.bytesPlane4 = 0;
    dfdBlock.bytesPlane5 = 0;
    dfdBlock.bytesPlane6 = 0;
    dfdBlock.bytesPlane7 = 0;
  }

  // Compute sizes
  dfdBlock.descriptorBlockSize = sizeof(dfdBlock) + vectorByteSize(dfSamples);
  const uint32_t dfdTotalSize  = dfdBlock.descriptorBlockSize + sizeof(uint32_t);

  // Write the Data Format Descriptor
  if(!output.write(reinterpret_cast<const char*>(&dfdTotalSize), sizeof(dfdTotalSize)))
  {
    return "Writing dfdTotalSize failed.";
  }

  if(!output.write(reinterpret_cast<char*>(&dfdBlock), sizeof(dfdBlock)))
  {
    return "Writing the Basic Data Format Descriptor failed.";
  }

  if(!output.write(reinterpret_cast<char*>(dfSamples.data()), vectorByteSize(dfSamples)))
  {
    return "Writing the samples of the Basic Data Format Descriptor failed.";
  }

  // Also set dfdByteLength now.
  header.dfdByteLength = dfdTotalSize;

  //---------------------------------------------------------------------------
  // Key/Value Data

  // Fill in the KTXwriter field if it's not already assigned.
  if(keyValueData.find("KTXwriter") == keyValueData.end())
  {
    keyValueData["KTXwriter"] = stringToCharVector("nvpro-samples' nv_ktx version 3.0.0");
  }

  // We now know the offset of the key/value data.
  header.kvdByteOffset = static_cast<uint32_t>(output.tellp() - startPos);
  header.kvdByteLength = 0;
  // Write the key/value data.
  for(const auto& kvp : keyValueData)
  {
    // Include the null character on the key, but note that the values already
    // include it (and may not be strings!)
    const uint32_t keySizeWithNull = static_cast<uint32_t>(kvp.first.size() + 1);
    const uint32_t valueSize       = static_cast<uint32_t>(kvp.second.size());
    // Write key and value byte length. Does not include padding!
    const uint32_t keyAndValueByteLength = keySizeWithNull + valueSize;
    if(!output.write(reinterpret_cast<const char*>(&keyAndValueByteLength), sizeof(uint32_t)))
    {
      return "Writing keyAndValueByteLength failed.";
    }
    if(!output.write(kvp.first.c_str(), keySizeWithNull))
    {
      return "Writing a key-value pair failed.";
    }
    if(!output.write(reinterpret_cast<const char*>(kvp.second.data()), valueSize))
    {
      return "Writing a key-value pair failed.";
    }

    // Write up to 4 null characters
    std::array<char, 4> nulls            = {'\0', '\0', '\0', '\0'};
    const size_t        valuePaddingSize = roundUp(keyAndValueByteLength, 4) - keyAndValueByteLength;
    assert(valuePaddingSize < 4);
    if(!output.write(nulls.data(), valuePaddingSize))
    {
      return "Writing value padding failed.";
    }

    header.kvdByteLength +=
        static_cast<uint32_t>(sizeof(uint32_t)) + keyAndValueByteLength + static_cast<uint32_t>(valuePaddingSize);
  }

  //---------------------------------------------------------------------------
  // Section 6
  // Currently no Basis supercompression, so no need for alignment here.
  assert(header.sgdByteLength == 0);

  //---------------------------------------------------------------------------
  // Section 7, the Mip Level Array
  // This is also where we perform Zstd supercompression!

  // First get the texel block size.
  size_t        texelBlockSize;
  ErrorWithText maybeError = exportSizeExtended(1, 1, 1, format, texelBlockSize, writeSettings.customSizeCallback);
  if(maybeError.has_value())
  {
    return "Error getting the texel block size for VkFormat " + std::to_string(format) + ": " + maybeError.value();
  }

  if(texelBlockSize == 0)
  {
    return "The texel block size for VkFormat " + std::to_string(format)
           + " was 0, which should never happen; likely this is an error "
        "in a custom size callback.";
  }

// Zstandard supercompression context
#ifdef NVP_SUPPORTS_ZSTD
  ScopedZstdCContext zstdContext;
  int                zstdClampedSupercompressionLevel = writeSettings.supercompressionLevel;
#endif
  if(writeSettings.supercompression == SupercompressionScheme::eZstd)
  {
#ifdef NVP_SUPPORTS_ZSTD
    zstdContext.Init();
    if(zstdContext.pCtx == nullptr)
    {
      return "Initializing the Zstandard context for supercompression failed!";
    }

    // Clamp the compression level to Zstandard's min and max
    const int zstdMinLevel = ZSTD_minCLevel();
    const int zstdMaxLevel = ZSTD_maxCLevel();
    assert(zstdMaxLevel >= zstdMinLevel);
    if(zstdClampedSupercompressionLevel < zstdMinLevel)
      zstdClampedSupercompressionLevel = zstdMinLevel;
    if(zstdClampedSupercompressionLevel > zstdMaxLevel)
      zstdClampedSupercompressionLevel = zstdMaxLevel;
#else
    return "Zstandard supercompression was selected for KTX2 writing, but nv_ktx was built without Zstd!";
#endif
  }

  // Write mips from smallest to largest.
  for(int64_t mip = static_cast<int64_t>(numMips) - 1; mip >= 0; mip--)
  {
    // First the mip padding if not supercompressed:
    if(writeSettings.supercompression == SupercompressionScheme::eNone)
    {
      const size_t posFromStart   = output.tellp() - startPos;
      const size_t mipPaddingSize = roundUp(posFromStart, lcm4(texelBlockSize)) - posFromStart;
      if(mipPaddingSize > 0)
      {
        // NOTE: Could be better
        std::vector<char> nulls(mipPaddingSize, 0);
        if(!output.write(nulls.data(), nulls.size()))
        {
          return "Writing mip padding failed.";
        }
      }
    }
    // We now know levels[mip].byteOffset, which comes after mip padding.
    levelIndex[mip].fileOffset = output.tellp() - startPos;

    const size_t mipWidth  = std::max(1u, mip0Width >> mip);
    const size_t mipHeight = std::max(1u, mip0Height >> mip);
    const size_t mipDepth  = std::max(1u, mip0Depth >> mip);

    // The size of each subresource of this mip in bytes.
    size_t subresourceSizeBytes = 0;
    UNWRAP_ERROR(exportSizeExtended(mipWidth, mipHeight, mipDepth, format, subresourceSizeBytes, writeSettings.customSizeCallback));

    // Compute the size of this mip in bytes.
    if(!checked_math::mul3(numLayersOr1, numFaces, subresourceSizeBytes, levelIndex[mip].uncompressedByteSize))
    {
      return "Computing the uncompressed byte size for mip " + std::to_string(mip) + " overflowed a size_t.";
    }

    // If not supercompressing, write each face to the file.
    if(writeSettings.supercompression == SupercompressionScheme::eNone)
    {
      for(uint32_t layer = 0; layer < numLayersOr1; layer++)
      {
        for(uint32_t face = 0; face < numFaces; face++)
        {
          const std::vector<char>& thisSubresource = subresource(static_cast<uint32_t>(mip), layer, face);
          assert(thisSubresource.size() == subresourceSizeBytes);
          if(!output.write(thisSubresource.data(), thisSubresource.size()))
          {
            return "Writing mip " + std::to_string(mip) + " layer " + std::to_string(layer) + " face "
                   + std::to_string(face) + " failed.";
          }
        }
      }
      levelIndex[mip].fileByteSize = levelIndex[mip].uncompressedByteSize;
    }
    else
    {
// We currently only support Zstd for writing.
#ifndef NVP_SUPPORTS_ZSTD
      return "Zstandard supercompression was selected, but nv_ktx was built without Zstd and execution reached the "
             "levelImages loop. This should never happen.";
#else
      if(writeSettings.supercompression != SupercompressionScheme::eZstd)
      {
        return "Only Zstandard supercompression is currently supported.";
      }
      // Concatenate all face data into a single buffer.
      // (Note: could potentially have lower peak memory usage but be more
      // complex using the Zstandard streaming API.)
      std::vector<char> rawData;
      {
        UNWRAP_ERROR(resizeVectorOrError(rawData, levelIndex[mip].uncompressedByteSize));
        size_t posInRawData = 0;
        for(uint32_t layer = 0; layer < numLayersOr1; layer++)
        {
          for(uint32_t face = 0; face < numFaces; face++)
          {
            const std::vector<char>& thisSubresource = subresource(uint32_t(mip), layer, face);
            assert(thisSubresource.size() == subresourceSizeBytes);
            memcpy(&rawData[posInRawData], thisSubresource.data(), thisSubresource.size());
            posInRawData += thisSubresource.size();
          }
        }
      }

      // Also allocate a buffer with the maximum possible compressed size needed.
      // (Note that this is always larger than the source!)
      // Also note that we'll always write the supercompressed data even when
      // it's larger, as the client controls whether supercompression is used.
      const size_t      fullMipUncompressedLength = levelIndex[mip].uncompressedByteSize;
      const size_t      supercompressedMaxSize    = ZSTD_COMPRESSBOUND(fullMipUncompressedLength);
      std::vector<char> supercompressedData;
      try
      {
        supercompressedData.resize(supercompressedMaxSize);
      }
      catch(...)
      {
        return "Allocating memory for Zstandard supercompressed output failed!";
      }

      // Compress!
      size_t errOrSize = ZSTD_compressCCtx(zstdContext.pCtx, supercompressedData.data(), supercompressedData.size(),
                                           rawData.data(), rawData.size(), zstdClampedSupercompressionLevel);
      if(ZSTD_isError(errOrSize))
      {
        return "Zstandard supercompression returned error " + std::to_string(errOrSize) + ".";
      }

      if(errOrSize > supercompressedData.size())
      {
        assert(false);  // This should never happen
        return "ZSTD_compressCCtx returned a number that was larger than the size of the supercompressed data "
               "buffer.";
      }

      // Write the supercompressed data to the file.
      if(!output.write(supercompressedData.data(), errOrSize))
      {
        return "Writing mip " + std::to_string(mip) + "'s supercompressed data to the file failed!";
      }
      levelIndex[mip].fileByteSize = errOrSize;
#endif
    }
  }

  // Now, fill in the header, then go back and write the identifier, header, and level index.
  header.vkFormat = format;
  switch(format)
  {
    case VK_FORMAT_R16_SFLOAT:
    case VK_FORMAT_R16G16_SFLOAT:
    case VK_FORMAT_R16G16B16A16_SFLOAT:
      header.typeSize = 2;
      break;
    case VK_FORMAT_R32_SFLOAT:
    case VK_FORMAT_R32G32_SFLOAT:
    case VK_FORMAT_R32G32B32A32_SFLOAT:
      header.typeSize = 4;
      break;
    default:
      header.typeSize = 1;
      break;
  }
  header.pixelWidth  = mip0Width;
  header.pixelHeight = mip0Height;
  header.pixelDepth  = mip0Depth;
  header.layerCount  = numLayersPossibly0;
  header.faceCount   = numFaces;
  header.levelCount  = appShouldGenerateMips ? 0 : numMips;
  switch(writeSettings.supercompression)
  {
    case SupercompressionScheme::eNone:
      header.supercompressionScheme = 0;
      break;
    case SupercompressionScheme::eZstd:
      header.supercompressionScheme = 2;
      break;
    default:
      return "Unsupported WriteSupercompression type while writing KTX2 header.";
  }

  if(!output.seekp(startPos, std::ios::beg))
  {
    return "Failed to seek back to the start to write the header.";
  }

  if(!output.write(reinterpret_cast<const char*>(ktx2Identifier), IDENTIFIER_LEN))
  {
    return "Failed to write the KTX2 identifier the second time.";
  }

  if(!output.write(reinterpret_cast<const char*>(&header), sizeof(header)))
  {
    return "Failed to write the KTX2 header.";
  }

  if(!output.write(reinterpret_cast<const char*>(levelIndex.data()), vectorByteSize(levelIndex)))
  {
    return "Failed to write the KTX2 level index.";
  }

  // And we're done!
  return {};
}

ErrorWithText Image::writeKTX2File(const char* filename, const WriteSettings& writeSettings)
{
  std::ofstream output(filename, std::ofstream::binary | std::ofstream::out | std::ofstream::trunc);
  return writeKTX2Stream(output, writeSettings);
}

//-----------------------------------------------------------------------------
// KTX1/KTX2 READING BRANCH
//-----------------------------------------------------------------------------

ErrorWithText Image::readHeaderFromStream(std::istream& input, const ReadSettings& readSettings)
{
  // Read the identifier.
  uint8_t identifier[IDENTIFIER_LEN]{};
  if(!input.read(reinterpret_cast<char*>(identifier), IDENTIFIER_LEN))
  {
    return "Reading the identifier failed!";
  }

  // Check if the identifier matches either the KTX 1 identifier or the KTX 2 identifier.
  if(memcmp(identifier, ktx1Identifier, IDENTIFIER_LEN) == 0)
  {
    m_fileInfo.readKtxVersion = 1;
    return readHeaderFromKTX1Stream(input, readSettings);
  }
  else if(memcmp(identifier, ktx2Identifier, IDENTIFIER_LEN) == 0)
  {
    m_fileInfo.readKtxVersion = 2;
    return readHeaderFromKTX2Stream(input, readSettings);
  }

  // Otherwise,
  return "Not a KTX1 or KTX2 file (first 12 bytes weren't a valid identifier).";
}

ErrorWithText Image::readSubresourcesFromStream(std::istream& input, const SubresourceRange& range, SubresourceTarget* outSubresources)
{
  // The app could have given us incorrect `range` and `outSubresources`,
  // so check them against what we know.
  if(range.numMips == 0 || range.numLayers == 0 || range.numFaces == 0)
  {
    return {};  // Nothing to do
  }

  if(!outSubresources)
  {
    return "`outSubresources` was null.";
  }

  if(range.firstMip > numMips || range.numMips > numMips - range.firstMip)
  {
    return "Requested range is out-of-bounds (requested " + std::to_string(range.numMips) + " starting at mip "
           + std::to_string(range.firstMip) + ", but the image only contains " + std::to_string(numMips) + " mips).";
  }

  const size_t numLayersClamped = std::max(1u, numLayersPossibly0);
  if(range.firstLayer > numLayersClamped || range.numLayers > numLayersClamped - range.firstLayer)
  {
    return "Requested range is out-of-bounds (requested " + std::to_string(range.numLayers) + " starting at layer "
           + std::to_string(range.firstLayer) + ", but the image only contains " + std::to_string(numLayersClamped) + " layers).";
  }

  if(range.firstFace > numFaces || range.numFaces > numFaces - range.firstFace)
  {
    return "Requested range is out-of-bounds (requested " + std::to_string(range.numFaces) + " starting at face "
           + std::to_string(range.firstFace) + ", but the image only contains " + std::to_string(numFaces) + " faces).";
  }

  {
    size_t outSubresourceIdx = 0;
    for(uint32_t mip = range.firstMip; mip < range.firstMip + range.numMips; mip++)
    {
      for(uint32_t layer = range.firstLayer; layer < range.firstLayer + range.numLayers; layer++)
      {
        for(uint32_t face = range.firstFace; face < range.firstFace + range.numFaces; face++)
        {
          const SubresourceLayout& source = getSubresourceLayout(mip, layer, face);
          const SubresourceTarget& target = outSubresources[outSubresourceIdx];

          if(!target.data)
          {
            return "outSubresources[" + std::to_string(outSubresourceIdx) + "].data was null.";
          }
          if(target.capacityInBytes < source.uncompressedByteSize)
          {
            return "outSubresources[" + std::to_string(outSubresourceIdx) + "] was too small to contain the "
                   + std::to_string(source.uncompressedByteSize) + " bytes of data for mip " + std::to_string(mip)
                   + ", layer " + std::to_string(layer) + ", face " + std::to_string(face) + ".";
          }

          outSubresourceIdx++;
        }
      }
    }
  }

  // OK, now we branch!
  if(m_fileInfo.readKtxVersion == 1)
  {
    return readSubresourcesFromKTX1Stream(input, range, outSubresources);
  }
  else if(m_fileInfo.readKtxVersion == 2)
  {
    return readSubresourcesFromKTX2Stream(input, range, outSubresources);
  }

  return "The read KTX version wasn't 1 or 2.";
}

ErrorWithText Image::readFromStream(std::istream& input, const ReadSettings& readSettings)
{
  const std::streampos startPos = input.tellg();
  UNWRAP_ERROR(readHeaderFromStream(input, readSettings));

  // In the high-level API, we write to our own subresource buffer. Set that up:
  if(!readSettings.mips)
  {
    numMips = 1;  // This is valid because of the mip-major layout of m_subresourceLayouts
  }
  UNWRAP_ERROR(allocate(numMips, numLayersPossibly0, numFaces));

  const SubresourceRange readRange{.numMips = numMips, .numLayers = std::max(1u, numLayersPossibly0), .numFaces = numFaces};

  std::vector<SubresourceTarget> targets;
  UNWRAP_ERROR(resizeVectorOrError(targets, m_data.size()));
  size_t subresourceIdx = 0;
  for(uint32_t mip = 0; mip < readRange.numMips; mip++)
  {
    for(uint32_t layer = 0; layer < readRange.numLayers; layer++)
    {
      for(uint32_t face = 0; face < readRange.numFaces; face++)
      {
        std::vector<char>& dst = subresource(mip, layer, face);
        UNWRAP_ERROR(resizeVectorOrError(dst, getSubresourceLayout(mip, layer, face).uncompressedByteSize));
        targets[subresourceIdx] = SubresourceTarget{.data = dst.data(), .capacityInBytes = dst.size()};
        subresourceIdx++;
      }
    }
  }

  // Rewind to where we started, then read the contents:
  input.seekg(startPos, std::ios::beg);
  return readSubresourcesFromStream(input, readRange, targets.data());
}

//-----------------------------------------------------------------------------
// Wrappers.

ErrorWithText Image::readHeaderFromFile(const char* filename, const ReadSettings& readSettings)
{
  std::ifstream inputStream(filename, std::ifstream::in | std::ifstream::binary);
  return readHeaderFromStream(inputStream, readSettings);
}

ErrorWithText Image::readSubresourcesFromFile(const char* filename, const SubresourceRange& range, SubresourceTarget* outSubresources)
{
  std::ifstream inputStream(filename, std::ifstream::in | std::ifstream::binary);
  return readSubresourcesFromStream(inputStream, range, outSubresources);
}

ErrorWithText Image::readFromFile(const char* filename, const ReadSettings& readSettings)
{
  std::ifstream inputStream(filename, std::ifstream::in | std::ifstream::binary);
  return readFromStream(inputStream, readSettings);
}

ErrorWithText Image::readHeaderFromMemory(const char* buffer, size_t bufferSize, const ReadSettings& readSettings)
{
  if(!buffer)
  {
    return "`buffer` was null.";
  }
  if(bufferSize > static_cast<size_t>(std::numeric_limits<std::streamsize>::max()))
  {
    return "The `bufferSize` parameter was too large to be stored in an std::streamsize.";
  }
  MemoryStream stream(buffer, static_cast<std::streamsize>(bufferSize));
  return readHeaderFromStream(stream, readSettings);
}

ErrorWithText Image::readSubresourcesFromMemory(const char* buffer, size_t bufferSize, const SubresourceRange& range, SubresourceTarget* outSubresources)
{
  if(!buffer)
  {
    return "`buffer` was null.";
  }
  if(bufferSize > static_cast<size_t>(std::numeric_limits<std::streamsize>::max()))
  {
    return "The `bufferSize` parameter was too large to be stored in an std::streamsize.";
  }
  MemoryStream stream(buffer, static_cast<std::streamsize>(bufferSize));
  return readSubresourcesFromStream(stream, range, outSubresources);
}

ErrorWithText Image::readFromMemory(const char* buffer, size_t bufferSize, const ReadSettings& readSettings)
{
  if(!buffer)
  {
    return "`buffer` was null.";
  }
  if(bufferSize > static_cast<size_t>(std::numeric_limits<std::streamsize>::max()))
  {
    return "The `bufferSize` parameter was too large to be stored in an std::streamsize.";
  }
  MemoryStream stream(buffer, static_cast<std::streamsize>(bufferSize));
  return readFromStream(stream, readSettings);
}
}  // namespace nv_ktx

//-----------------------------------------------------------------------------
// Sample code

#include <stdio.h>
[[maybe_unused]] static void usage_nv_ktx()
{
  // We have multiple examples here.

  //---------------------------------------------------------------------------
  // 1. How to load an image using the high-level API.
  nv_ktx::Image         image;
  nv_ktx::ErrorWithText maybeError = image.readFromFile("data/image.ktx2", {});
  // ErrorWithText is either empty (success), or has an error message.
  if(maybeError.has_value())
  {
    fprintf(stderr, "Could not read data/image.ktx2. Error information: %s\n", maybeError.value().c_str());
  }

  //---------------------------------------------------------------------------
  // 2. Inspecting the KTX image.
  // Typically, after reading a KTX file, you'd access subresources using
  // image.subresource(...) and upload them to the GPU using your graphics
  // API of choice.
  // Here, we'll print out some information about the image.
  printf("Image size: %u x %u x %u\n", image.mip0Width, image.mip0Height, image.mip0Depth);
  printf("Mips: %u\n", image.numMips);
  printf("Layers: %u\n", image.numLayersPossibly0);
  printf("Faces: %u\n", image.numFaces);
  printf("Image type: %u\n", static_cast<unsigned>(image.getImageType()));
  printf("Is premultiplied: %s\n", image.isPremultiplied ? "true" : "false");
  printf("Is sRGB: %s\n", image.isSrgb ? "true" : "false");
  printf("Swizzle:");
  for(size_t component = 0; component < image.swizzle.size(); component++)
  {
    switch(image.swizzle[component])
    {
      case nv_ktx::Swizzle::R:
        printf(" R");
        break;
      case nv_ktx::Swizzle::G:
        printf(" G");
        break;
      case nv_ktx::Swizzle::B:
        printf(" B");
        break;
      case nv_ktx::Swizzle::A:
        printf(" A");
        break;
      case nv_ktx::Swizzle::ZERO:
        printf(" 0");
        break;
      case nv_ktx::Swizzle::ONE:
        printf(" 1");
        break;
    }
  }
  printf("\n");
  printf("Key/value data:\n");
  for(const auto& keyValuePair : image.keyValueData)
  {
    printf("\t%s:\t", keyValuePair.first.c_str());  // Key
    // The value can be an arbitrary byte array.
    // Here, we check to see if it's a printable, null-terimated
    // ASCII string.
    // If not, we print the byte values.
    const std::vector<char>& value              = keyValuePair.second;
    bool                     isPrintableCString = true;
    for(char c : value)
    {
      if(c < 33 || c > 126)
      {
        isPrintableCString = false;
        break;
      }
    }
    isPrintableCString = isPrintableCString && (value.back() == '\0');

    if(isPrintableCString)
    {
      printf("%s", keyValuePair.second.data());
    }
    else
    {
      for(char c : value)
      {
        printf(" %u", static_cast<uint32_t>(c));
      }
    }
    printf("\n");
  }

  // Iterate over subresources and print a few bytes of the data of each one.
  for(uint32_t mip = 0; mip < image.numMips; mip++)
  {
    for(uint32_t layer = 0; layer < std::max(image.numLayersPossibly0, 1U); layer++)
    {
      for(uint32_t face = 0; face < std::max(image.numFaces, 1U); face++)
      {
        printf("mip %u, layer %u, face %u:\n", mip, layer, face);
        const std::vector<char>& subresource = image.subresource(mip, layer, face);
        printf("%u x %u x %u, %zu bytes\n", std::max(1u, image.mip0Width >> mip), std::max(1u, image.mip0Height >> mip),
               std::max(1u, image.mip0Depth >> mip), subresource.size());

        printf("data:");
        constexpr size_t kMaxBytesToPrint = 10;
        for(size_t i = 0; i < kMaxBytesToPrint && i < subresource.size(); i++)
        {
          printf(" %u", static_cast<uint32_t>(subresource[i]));
        }
        if(subresource.size() > kMaxBytesToPrint)
        {
          printf("...");
        }
        printf("\n");
      }
    }
  }

  //---------------------------------------------------------------------------
  // 3. How to use the low-level API.
  // readFromFile() copies into Image's internal subresources, which means
  // you then have to copy the data out of there. In some cases, you can avoid
  // a copy by using the lower-level readHeader + readSubresources API.
  {
    // First, we open the file -- let's use a stream this time to show one
    // thing that only applies if you're using streams here later on:
    std::ifstream file("data/image2.ktx2", std::ios::binary);
    // The readHeader functions read only the header, none of the image contents:
    nv_ktx::Image image2;
    if(nv_ktx::ErrorWithText maybeError = image2.readHeaderFromStream(file, {}))
    {
      fprintf(stderr, "Could not read data/image.ktx2. Error information: %s\n", maybeError.value().c_str());
    }

    // Now we can load individual subresources into buffers we allocate in
    // advance.
    // The API allows us to load a range of mips, layers, and faces.
    // For instance,
    const nv_ktx::SubresourceRange range{.firstMip   = 5,  // 5...6
                                         .numMips    = 2,
                                         .firstLayer = 0,  // 0...0
                                         .numLayers  = 1,
                                         .firstFace  = 0,  // 0...0
                                         .numFaces   = 1};
    // will load the subresources at
    // (mip 5 layer 0 face 0) and (mip 6 layer 0 face 0).
    // We then need to allocate data for the output, and tell the
    // readSubresourcesFromStream API where to put each subrsource. Like this:
    size_t            outputDataSize = image2.getSubresourceByteSizeSum(range);
    std::vector<char> outputData(outputDataSize);

    // Then tell the readSubresourcesFromStreamAPI where to put
    // each subresource, like this:
    std::vector<nv_ktx::SubresourceTarget> targets;
    size_t                                 nextPos = 0;
    for(uint32_t mip = range.firstMip; mip < range.firstMip + range.numMips; mip++)
    {
      // All subresources within a mip have the same decompressed size.
      const size_t subresourceSize = image2.getSubresourceLayout(mip, 0, 0).uncompressedByteSize;
      for(uint32_t layer = range.firstLayer; layer < range.firstLayer + range.numLayers; layer++)
      {
        for(uint32_t face = range.firstFace; face < range.firstFace + range.numFaces; face++)
        {
          targets.push_back(nv_ktx::SubresourceTarget{.data = &outputData[nextPos], .capacityInBytes = subresourceSize});
          nextPos += subresourceSize;
        }
      }
    }

    // Now, if we're using streams, we have to rewind back to the start of the
    // file:
    file.seekg(0, std::ios::beg);
    // And then read the subresources into `outputData`:
    if(nv_ktx::ErrorWithText maybeError = image2.readSubresourcesFromStream(file, range, targets.data()))
    {
      fprintf(stderr, "Could not read data/image.ktx2. Error information: %s\n", maybeError.value().c_str());
    }
  }

  //---------------------------------------------------------------------------
  // 4. How to create and write a simple image.
  // We'll make a 11x5 VK_FORMAT_R8G8B8A8_UNORM image with 2 mip levels.
  nv_ktx::Image outImage;
  maybeError = outImage.allocate(2 /* numMips */, 1 /* numLayers */, 1 /* numFaces */);
  if(maybeError.has_value())
  {
    printf("Error: %s", maybeError.value().c_str());
    return;
  }

  // Set its format and other information.
  outImage.format     = VK_FORMAT_R8G8B8A8_UNORM;
  outImage.mip0Width  = 11;
  outImage.mip0Height = 5;
  outImage.numFaces   = 1;

  // Fill it with some data. This can be arbitrary. Here we'll draw a pattern
  // using some bit operations.
  std::vector<char>& mip0 = outImage.subresource(0, 0, 0);
  for(uint32_t y = 0; y < outImage.mip0Height; y++)
  {
    for(uint32_t x = 0; x < outImage.mip0Width; x++)
    {
      const uint32_t row   = 0x11U | (0x1U << y) | (0x40U << (y / 2)) | (0x400U >> (y / 2));
      const uint32_t rgba  = ((row >> x) & 1U) * 0x00FF4689U + 0xFF00B976U;
      void*          pixel = reinterpret_cast<void*>(&mip0[(y * outImage.mip0Width + x) * 4]);
      memcpy(pixel, &rgba, sizeof(rgba));
    }
  }

  // Mip 1 will be a solid color of (0x76, 0xB9, 0x00, 0xFF).
  std::vector<char>& mip1       = outImage.subresource(1, 0, 0);
  const uint32_t     mip1Width  = outImage.mip0Width >> 1;
  const uint32_t     mip1Height = outImage.mip0Height >> 1;
  for(uint32_t y = 0; y < mip1Height; y++)
  {
    for(uint32_t x = 0; x < mip1Width; x++)
    {
      void*          pixel = reinterpret_cast<void*>(&mip0[(y * mip1Width + x) * 4]);
      const uint32_t rgba  = 0xFF00B976U;
      memcpy(pixel, &rgba, sizeof(rgba));
    }
  }

  // And write it out:
  nv_ktx::WriteSettings writeSettings = {};  // The defaults are fine
  maybeError                          = outImage.writeKTX2File("example.ktx2", writeSettings);
  if(maybeError.has_value())
  {
    printf("Write failed: %s\n", maybeError.value().c_str());
    return;
  }
}
