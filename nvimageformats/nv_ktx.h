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

/*-----------------------------------------------------------------------------
 
nv_ktx 2.0.0

This is a mostly self-contained reader and writer for KTX2 files and reader
for KTX1 files. It only relies on Vulkan (for KTX2), GL (for KTX1), and the
Khronos Data Format.

For example usage, please see usage_nv_ktx() at the end of nv_ktx.cpp.

Define `NVP_SUPPORTS_ZSTD`, `NVP_SUPPORTS_GZLIB`, and `NVP_SUPPORTS_BASISU` to
include the Zstd, Zlib, and Basis Universal headers respectively, and to
enable reading these formats. This will also enable writing Zstd and
Basis Universal-compressed formats.

Changelog for nv_ktx 2.0.0:
- Adds functions that let you read only the header, so you can copy data
directly to the GPU or even potentially undo supercompression there.
- [API break] WriteSupercompressionType and KTXImage::InputSupercompression
are now SupercompressionScheme. In particular, KTXImage::input_supercompression
is now in the read-only KTXImage::getFileInfo()::supercompression_scheme.
- [API break] getKTXVersion() is now in the read-only KTXImage::getFileInfo().
- [API break] ReadSettings::max_resource_size_in_bytes has been replaced by
ReadSettings::max_size_in_bytes, which limits the total uncompressed size of
all subresources together. This protects against resource exhaustion attacks
when there are many subresources.

-----------------------------------------------------------------------------*/

#ifndef __NV_KTX_H__
#define __NV_KTX_H__

#include <array>
#include <iostream>
#include <map>
#include <optional>
#include <string>
#include <vector>
#include <vulkan/vulkan_core.h>

namespace nv_ktx {
// These functions return an empty std::optional if they succeeded, and a
// value with text describing the error if they failed.
using ErrorWithText = std::optional<std::string>;

// KTX files can store key/value pairs, where the key is a UTF-8
// null-terminated string and the value is an arbitrary byte array
// (but often a null-terminated ASCII string).
using KeyValueData = std::map<std::string, std::vector<char>>;

// Apps can define custom functions that return the size in bytes of new
// VkFormats. Functions of this type should take in the width, height, and
// depth of a format in the first 3 parameters, the VkFormat in the 4th, and
// return the size in bytes of an image with those dimensions in the last
// parameter. Passing in an image size of (1, 1, 1) should give the size of
// the smallest possible nonzero image. If the format is unknown, it should
// return a string; if it succeeds, it should return {}.
using CustomExportSizeFuncPtr = ErrorWithText (*)(size_t, size_t, size_t, VkFormat, size_t&);

// Configurable settings for reading files. This is a struct so that it can
// be extended in the future.
struct ReadSettings
{
  // Whether to read all mips (true), or only the base mip (false).
  bool mips = true;
  // See docs for CustomExportSizeFuncPtr
  CustomExportSizeFuncPtr custom_size_callback = nullptr;
  // If true, the reader will validate that the KTX file contains at least 1
  // byte per subresource. This will involve seeking to the end of the stream
  // to determine the length of the stream or file.
  bool validate_input_size = true;
  // Limits the maximum total uncompressed image size and supercompression
  // global data size in bytes; produces errors for any files with a larger size.
  size_t max_size_in_bytes = size_t(1) << 30;
  // By default, UASTC is transcoded to BC7 instead of ASTC. Setting this to
  // true will transcode UASTC to ASTC.
  bool device_supports_astc = false;
};

// Names for the KTX2 supercompression schemes
// (so you don't need to use the raw values from
// https://registry.khronos.org/KTX/specs/2.0/ktxspec.v2.html#_supercompressionscheme )
enum class SupercompressionScheme : uint32_t
{
  eNone    = 0,  // Apply no supercompression.
  eBasisLZ = 1,  // UASTC or ETC1S.
  eZstd    = 2,  // Zstandard.
  eZlib    = 3,  // Zlib.
};

// Describes the four valid combinations of ETC1S slices in the KTX2
// specification. These are listed in the order they appear there.
enum class ETC1SCombination
{
  RGB,   // One slice, RGB
  RGBA,  // Two slices, RGB + AAA
  R,     // One slice, RRR
  RG     // Two slices, RRR + GGG
};

enum class EncodeRGBA8ToFormat
{
  NO,  // Don't encode the data to a Basis Universal format.
  // Note that an option other than NO overrides the SupercompressionType when writing.
  // For the following modes, the image format must be VK_FORMAT_B8G8R8A8_SRGB
  // or VK_FORMAT_B8G8R8A8_UNORM. Basis Universal will then be called to encode
  // the data and write the KTX2 file.
  UASTC,       // Highest-quality format; RGBA data, usually decodes to ASTC or BC7.
  ETC1S_RGBA,  // RGBA data; usually decodes to BC7 (8bpp).
  ETC1S_RGB    // RGB channels only; usually decodes to BC7 (8bpp).
};

enum class UASTCEncodingQuality
{
  FASTEST  = 0,
  FASTER   = 1,
  DEFAULT  = 2,
  SLOWER   = 3,
  VERYSLOW = 4
};

// Configurable settings for writing files. This is a struct so that it can
// be extended in the future.
struct WriteSettings
{
  // Type of supercompression to apply if any
  SupercompressionScheme supercompression = SupercompressionScheme::eNone;
  // Supercompression quality level for Zstandard, which is supported by all
  // formats other than ETC1s. This ranges from ZSTD_minCLevel() to
  // ZSTD_maxCLevel().
  // Higher levels are slower.
  int supercompression_level = 0;
  // See docs for CustomExportSizeFuncPtr
  CustomExportSizeFuncPtr custom_size_callback = nullptr;
  // Whether to encode the data to a Basis format. If not NO, the image format
  // must be VK_FORMAT_B8G8R8A8_SRGB or VK_FORMAT_B8G8R8A8_UNORM.
  EncodeRGBA8ToFormat encode_rgba8_to_format = EncodeRGBA8ToFormat::NO;
  // Applies when encoding RGBA8 to UASTC. Corresponds to cPackUASTCLevel in Basis.
  UASTCEncodingQuality uastc_encoding_quality = UASTCEncodingQuality::DEFAULT;
  // Applies when encoding RGBA8 to ETC1S. Ranges from 0 to BASISU_MAX_COMPRESSION_LEVEL.
  // Higher levels are slower.
  int etc1s_encoding_level = 3;
  // Lambda for UASTC Rate-Distortion Optimization, from 0 to 50. Higher numbers
  // compress more at lower quality.
  float rdo_lambda = 10.0f;
  // Enables Rate-Distortion Optimization for ETC1S.
  bool rdo_etc1s = true;
};

// An enum for each of the possible elements in a ktxSwizzle value.
enum class KTX_SWIZZLE
{
  ZERO = 0,
  ONE,
  R,
  G,
  B,
  A
};

// A range of subresources given by their mips, layers, and faces.
struct SubresourceRange
{
  uint32_t firstMip   = 0;
  uint32_t numMips    = 1;
  uint32_t firstLayer = 0;
  uint32_t numLayers  = 1;
  uint32_t firstFace  = 0;
  uint32_t numFaces   = 1;
};

// For each subresource, lists its position/layout info.
// Also matches the byte layout of a KTX2 level index.
struct SubresourceLayout
{
  // These two parameters specify the start and length of bytes, relative to
  // the start of the file, containing the (possibly supercompressed) data
  // for this resource.
  // This is mainly useful when there's no supercompression, because it tells
  // you where to memcpy each subresource from. In the special cases of ZSTD
  // and ZLIB supercompression, which apply compression to entire mip levels,
  // this spans the entire mip's data.
  uint64_t fileOffset{};
  uint64_t fileByteSize{};
  // The size in bytes of a single subresource, after undoing any
  // supercompression. All subresources within a mip have the same size.
  uint64_t uncompressedByteSize{};
};

// Where data for a subresource should be copied to, when using
// readSubresources*().
struct SubresourceTarget
{
  // The first byte of the target buffer.
  void* data = nullptr;
  // The size of the target buffer, so readSubresources*() can make sure it's
  // large enough.
  size_t capacityInBytes = 0;
};

// Represents a KTX or KTX2 file. This includes:
// - header information like the VkFormat of the image data,
// - the table of key/value pairs,
// - and optionally the formatted (i.e. encoded/compressed, but not
// supercompressed -- we supercompress data when reading from/writing to files)
// image data for each mip level, array element, and face.
struct KTXImage
{
public:
  // Clears, then sets up storage for an image with the given dimensions. These
  // can be set to 0 instead of 1 along each dimension to indicate different
  // texture types, such as 1D or 2D. See table 4.1 in the KTX 2.0
  // specification, or the comments on these variables below.
  //
  // Width, height, depth, and VkFormat should be set manually using the
  // member variables. This does not allocate the encoded subresources.
  // This can fail e.g. if the parameters are so large that the app runs out of
  // memory when allocating space.
  ErrorWithText allocate(
      // The number of mips (levels) in the image, including the base mip.
      uint32_t _num_mips = 1,
      // The number of array elements (layers) in the image. 0 for a non-array
      // texture (this has meaning in OpenGL, but not in Vulkan).
      // If representing an incomplete cube map (i.e. a cube map where not all
      // faces are stored), this is
      //   (faces per cube map) * (number of cube maps)
      // and _num_faces is 1.
      uint32_t _num_layers = 0,
      // The number of faces in the image (1 for a 2D texture, 6 for a cube map)
      uint32_t _num_faces = 1);

  // Clears all stored image data.
  void clear();

  // Determines the VkImageType corresponding to this KTXImage based on the
  // dimensions, according to Table 4.1 of the KTX 2.0 specification.
  // In the invalid case where mip_0_width == 0, returns VK_IMAGE_TYPE_1D.
  VkImageType getImageType() const;

  // Mutably accesses the subresource at the given mip, layer, and face. If the
  // given indices are out of range, throws an std::out_of_range exception.
  std::vector<char>& subresource(uint32_t mip = 0, uint32_t layer = 0, uint32_t face = 0);

  // Reads this structure from a KTX stream, advancing the stream as well.
  // Returns an optional error message if the read failed.
  ErrorWithText readFromStream(std::istream&       input,          // The input stream, at the start of the KTX data
                               const ReadSettings& readSettings);  // Settings for the reader.

  // Wrapper for readFromStream for a filename.
  ErrorWithText readFromFile(const char*         filename,       // The .ktx or .ktx2 file to read from.
                             const ReadSettings& readSettings);  // Settings for the reader.

  // Wrapper for readFromStream for a buffer in memory.
  ErrorWithText readFromMemory(const char*         buffer,         // The buffer in memory.
                               size_t              bufferSize,     // Its length in bytes.
                               const ReadSettings& readSettings);  // Settings for the reader.

  // Writes this structure in KTX2 format to a stream.
  ErrorWithText writeKTX2Stream(std::ostream&        output,  // The output stream, at the point to start writing
                                const WriteSettings& writeSettings);  // Settings for the writer.

  // Wrapper for writeKTX2Stream for a filename. Customarily, the filename ends
  // in .ktx2.
  ErrorWithText writeKTX2File(const char*          filename,        // The output stream, at the point to start writing
                              const WriteSettings& writeSettings);  // Settings for the writer.

  // Read-only info from the file header.
  struct FileInfo
  {
    // Whether the loaded file was a KTX1 (1) or KTX2 (2) file.
    uint32_t read_ktx_version = 1;
    // The KTX2 supercompression scheme and supercompression global data.
    // This should be one of the values from SupercompressionType,
    // unless the file used something new.
    uint32_t ktx2_supercompression_scheme = 0;
    uint64_t ktx2_global_data_offset      = 0;
    uint64_t ktx2_global_data_byte_size   = 0;
    // KTX2 Basis ETC1S textures can have 1 or 2 slices:
    size_t           ktx2_basis_etc1s_num_slices  = 1;
    ETC1SCombination ktx2_basis_etc1s_combination = {};
    // The KTX2 Khronos Data Format color model. This is mainly important for
    // the cases 163 (ETC1S) and 166 (UASTC).
    uint8_t ktx2_color_model = 0;
    // KTX1 data might be encoded in a way that requires you to swap element
    // endianness.
    // These two fields contain the info you need to do the swap yourself:
    // Whether the KTX1 data is endian swapped relative to this system:
    bool ktx1_needs_endian_swap = false;
    // KTX1 endian swap element size (glTypeSize from the header).
    uint32_t ktx1_gl_type_size = 0;
  };

  const FileInfo& getFileInfo() const { return m_file_info; }

  //---------------------------------------------------------------------------
  // We also provide a lower-level API where you can read the file header
  // and then copy/decompress subresources however you want (e.g. copying
  // data directly from a memory-mapped file to GPU memory if the file doesn't
  // use supercompression, or even inflate data on the GPU).

  // Reads only the header of a KTX stream (does not read textures), advancing
  // the stream as well. Returns an optional error message if the read failed.
  // This is useful if your code can copy non-supercompressed textures
  // directly to GPU memory.
  ErrorWithText readHeaderFromStream(std::istream&       input,  // The input stream, at the start of the KTX data
                                     const ReadSettings& readSettings);  // Settings for the reader.

  // Wrapper for readHeaderFromStream for a filename.
  ErrorWithText readHeaderFromFile(const char*         filename,       // The .ktx or .ktx2 file to read from.
                                   const ReadSettings& readSettings);  // Settings for the reader.

  // Wrapper for readHeaderFromStream for a buffer in memory.
  ErrorWithText readHeaderFromMemory(const char*         buffer,         // The buffer in memory.
                                     size_t              bufferSize,     // Its length in bytes.
                                     const ReadSettings& readSettings);  // Settings for the reader.

  // Returns whether subresources within the file require additional steps
  // (e.g. KTX2 supercompression inflation or KTX1 endian swapping) before they
  // can be used as subresources of the KTX file's `format` on a GPU.
  bool requiresComplexDecoding() const;

  // Returns where a subresource's (possibly supercompressed) data exists in the file.
  const SubresourceLayout& getSubresourceLayout(uint32_t mip, uint32_t layer, uint32_t face) const;

  // Returns the number of bytes for a single subresource within a single mip,
  // without supercompression.
  size_t getSubresourceByteSize(uint32_t mip) const { return getSubresourceLayout(mip, 0, 0).uncompressedByteSize; }

  // Returns the total number of bytes required to store all the subresources
  // within the given range, assuming there's no padding between them, without supercompression.
  // `range` must be in-bounds, and the calculation assumes size_t
  // does not overflow.
  size_t getSubresourceByteSizeSum(const SubresourceRange& range) const;

  // Returns the total number of bytes required to store all the subresources
  // within a given mip, assuming there's no padding between them, without supercompression.
  // `getSubresourceByteLengthSum` is more flexible, but this is the usual use case.
  size_t getMipByteSizeSum(uint32_t mip) const;

  // Given a stream set to the start of a KTX file (i.e. at the start of the
  // 12-byte magic number), extracts the subresources within the range of
  // mips, layers, and faces given by `range`, inflating any supercompression,
  // and writes each one to the corresponding SubresourceTarget.
  //
  // `outSubresources` must be in [mip, layer, face] order; i.e. it must be
  // an array of length `range.numMips * range.numLayers * range.numFaces`,
  // and the data for the subresource at mip `range.firstMip + i`,
  // layer `range.firstLayer + j`, face `range.firstFace + k` will be inflated
  // to outSubresources[(i * range.numLayers + j) * range.numFaces + k].
  ErrorWithText readSubresourcesFromStream(std::istream& input, const SubresourceRange& range, SubresourceTarget* outSubresources);

  // Wrapper for readSubresourcesFromStream for a file.
  ErrorWithText readSubresourcesFromFile(const char* filename, const SubresourceRange& range, SubresourceTarget* outSubresources);

  // Wrapper for readSubresourcesFromStream for a buffer in memory.
  ErrorWithText readSubresourcesFromMemory(const char* buffer, size_t bufferSize, const SubresourceRange& range, SubresourceTarget* outSubresources);

public:
  //---------------------------------------------------------------------------
  // These members can be freely modified.

  // The format of the data in this image. When reading a KTX1 file (which
  // specifies a GL format), we automatically convert to a VkFormat.
  VkFormat format = VK_FORMAT_UNDEFINED;
  // The width in pixels of the largest mip. Must be > 0.
  uint32_t mip_0_width = 1;
  // The height in pixels of the largest mip. 0 for a 1D texture.
  uint32_t mip_0_height = 0;
  // The depth in pixels of the largest mip. 0 for a 1D or 2D texture.
  uint32_t mip_0_depth = 0;
  // The number of mips (levels) in the image, including the base mip. Always
  // greater than or equal to 1.
  uint32_t num_mips = 1;
  // The number of array elements (layers) in the image. 0 for a non-array
  // texture (this has meaning in OpenGL, but not in Vulkan).
  // If representing an incomplete cube map (i.e. a cube map where not all
  // faces are stored), this is
  //   (faces per cube map) * (number of cube maps)
  // and _num_faces is 1.
  uint32_t num_layers_possibly_0 = 0;
  // The number of faces in the image (1 for a 2D texture, 6 for a cube map)
  uint32_t num_faces = 0;
  // This file's key/value table. Note that for the ktxSwizzle key, one should
  // use the swizzle element instead!
  KeyValueData key_value_data{};

  // KTX files can set the number of mips to 0 to indicate that
  // the application should generate a full mip chain.
  bool app_should_generate_mips = false;

  // Whether this data represents an image with premultiplied alpha
  // (generally, storing (r*a, g*a, b*a, a) instead of (r, g, b, a)).
  // This is used when writing the Data Format Descriptor in KTX2.
  bool is_premultiplied = false;

  // Whether the Data Format Descriptor transferFunction for this data is
  // KHR_DF_TRANSFER_SRGB. (Otherwise, it is KHR_DF_TRANSFER_LINEAR.)
  // More informally, says "when a GPU accesses this texture, should it perform
  // sRGB-to-linear conversion". For instance, this is usually true for color
  // textures, and false for normal maps and depth maps. Validation requires
  // this to match the VkFormat - except in special cases such as Basis UASTC
  // and Universal.
  bool is_srgb = true;

  // Specifies how the red, green, blue, and alpha channels should be sampled
  // from the source data. For instance, {R, G, ZERO, ONE} means the red and
  // green channels should be sampled from the red and green texture components
  // respectively, the blue channel is sampled as 0, and the alpha channel is
  // sampled as 1.
  // Note that values here should be read in lieu of the key_value_data's
  // ktxSwizzle key! This is to make Basis Universal usage easier in the future.
  std::array<KTX_SWIZZLE, 4> swizzle = {KTX_SWIZZLE::R, KTX_SWIZZLE::G, KTX_SWIZZLE::B, KTX_SWIZZLE::A};

private:
  // Internal functions.
  SubresourceLayout& subresourceLayout(uint32_t mip, uint32_t layer, uint32_t face);

  ErrorWithText readHeaderFromKTX1Stream(std::istream& input, const ReadSettings& readSettings);
  ErrorWithText readSubresourcesFromKTX1Stream(std::istream& input, const SubresourceRange& range, SubresourceTarget* outSubresources);

  ErrorWithText readHeaderFromKTX2Stream(std::istream& input, const ReadSettings& readSettings);
  ErrorWithText readSubresourcesFromKTX2Stream(std::istream& input, const SubresourceRange& range, SubresourceTarget* outSubresources);

private:
  // A structure containing all the image's encoded, non-supercompressed
  // image data. We store this in a buffer with an entry per subresource, and
  // provide accessors to it.
  std::vector<std::vector<char>> m_data;
  std::vector<SubresourceLayout> m_level_indices;
  std::vector<SubresourceLayout> m_subresource_layouts;
  FileInfo                       m_file_info{};
};

}  // namespace nv_ktx

#endif
