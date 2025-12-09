/*
 * Copyright (c) 2020-2025, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2020-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <assert.h>

#if defined(__linux__)
#include <errno.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

#include "file_mapping.hpp"

#if defined(_WIN32)
inline DWORD HIDWORD(size_t x)
{
  return (DWORD)(x >> 32);
}
inline DWORD LODWORD(size_t x)
{
  return (DWORD)x;
}
#endif

namespace nvutils {

nvutils::FileMapping& FileMapping::operator=(FileMapping&& other) noexcept
{
  if(this != &other)
  {
    // close our own handle before move assignment of other
    if(m_isValid)
    {
      close();
    }

    m_isValid     = other.m_isValid;
    m_fileSize    = other.m_fileSize;
    m_mappingType = other.m_mappingType;
    m_mappingPtr  = other.m_mappingPtr;
    m_mappingSize = other.m_mappingSize;
#ifdef _WIN32
    m_win32.file              = other.m_win32.file;
    m_win32.fileMapping       = other.m_win32.fileMapping;
    other.m_win32.file        = nullptr;
    other.m_win32.fileMapping = nullptr;
#else
    m_unix.file       = other.m_unix.file;
    other.m_unix.file = -1;
#endif
    other.m_isValid    = false;
    other.m_mappingPtr = nullptr;
  }

  return *this;
}

bool FileMapping::open(const std::filesystem::path& filePath, MappingType mappingType, size_t fileSize)
{
  // wchar_t* on Windows, char* on Linux
  const std::filesystem::path::value_type* nativePath = filePath.c_str();

  assert(!m_isValid && "must call close before open");

  if(!g_pageSize)
  {
#if defined(_WIN32)
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    g_pageSize = (size_t)si.dwAllocationGranularity;
#elif defined(__linux__)
    g_pageSize = (size_t)getpagesize();
#endif
  }

  m_mappingType = mappingType;

  if(mappingType == MAPPING_READOVERWRITE)
  {
    assert(fileSize);
    m_fileSize    = fileSize;
    m_mappingSize = ((fileSize + g_pageSize - 1) / g_pageSize) * g_pageSize;

    // check if the current process is allowed to save a file of that size
#if defined(_WIN32)
    WCHAR          dir[MAX_PATH + 1];
    BOOL           success = FALSE;
    ULARGE_INTEGER numFreeBytes;

    DWORD length = GetVolumePathNameW(nativePath, dir, MAX_PATH + 1);

    if(length > 0)
    {
      success = GetDiskFreeSpaceExW(dir, NULL, NULL, &numFreeBytes);
    }

    m_isValid = (!!success) && (m_mappingSize <= numFreeBytes.QuadPart);
#elif defined(__linux__)
    struct rlimit rlim;
    getrlimit(RLIMIT_FSIZE, &rlim);
    m_isValid = (m_mappingSize <= rlim.rlim_cur);
#endif
    if(!m_isValid)
    {
      return false;
    }
  }

#if defined(_WIN32)
  m_win32.file =
      mappingType == MAPPING_READONLY ?
          CreateFileW(nativePath, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_READONLY, NULL) :
          CreateFileW(nativePath, GENERIC_READ | GENERIC_WRITE, 0, NULL, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);

  m_isValid = (m_win32.file != INVALID_HANDLE_VALUE);
  if(m_isValid)
  {
    if(mappingType == MAPPING_READONLY)
    {
      DWORD sizeHi  = 0;
      DWORD sizeLo  = GetFileSize(m_win32.file, &sizeHi);
      m_mappingSize = (static_cast<size_t>(sizeHi) << 32) | sizeLo;
      m_fileSize    = m_mappingSize;
    }

    m_win32.fileMapping = CreateFileMapping(m_win32.file, NULL, mappingType == MAPPING_READONLY ? PAGE_READONLY : PAGE_READWRITE,
                                            HIDWORD(m_mappingSize), LODWORD(m_mappingSize), NULL);

    m_isValid = (m_win32.fileMapping != NULL);
    if(m_isValid)
    {
      m_mappingPtr = MapViewOfFile(m_win32.fileMapping, mappingType == MAPPING_READONLY ? FILE_MAP_READ : FILE_MAP_ALL_ACCESS,
                                   HIDWORD(0), LODWORD(0), (SIZE_T)0);
      if(!m_mappingPtr)
      {
#if 0
      DWORD err = GetLastError();
#endif
        CloseHandle(m_win32.file);
        m_isValid = false;
      }
    }
    else
    {
      CloseHandle(m_win32.file);
    }
  }
#elif defined(__linux__)
  m_unix.file = mappingType == MAPPING_READONLY ? ::open(nativePath, O_RDONLY) :
                                                  ::open(nativePath, O_RDWR | O_CREAT | O_TRUNC, 0666);

  m_isValid = (m_unix.file != -1);
  if(m_isValid)
  {
    if(mappingType == MAPPING_READONLY)
    {
      struct stat s;
      m_isValid &= (fstat(m_unix.file, &s) >= 0);
      m_mappingSize = s.st_size;
    }
    else
    {
      // make file large enough to hold the complete scene
      m_isValid &= (lseek(m_unix.file, m_mappingSize - 1, SEEK_SET) >= 0);
      m_isValid &= (write(m_unix.file, "", 1) >= 0);
      m_isValid &= (lseek(m_unix.file, 0, SEEK_SET) >= 0);
    }
    m_fileSize = m_mappingSize;
    if(m_isValid)
    {
      m_mappingPtr = mmap(0, m_mappingSize, mappingType == MAPPING_READONLY ? PROT_READ : (PROT_READ | PROT_WRITE),
                          MAP_SHARED, m_unix.file, 0);
      m_isValid    = (m_mappingPtr != MAP_FAILED);
    }
    if(!m_isValid)
    {
      ::close(m_unix.file);
      m_unix.file = -1;
    }
  }
#endif
  return m_isValid;
}

void FileMapping::close()
{
  if(m_isValid)
  {
#if defined(_WIN32)
    assert((m_win32.file != INVALID_HANDLE_VALUE) && (m_win32.fileMapping != NULL));

    UnmapViewOfFile(m_mappingPtr);
    CloseHandle(m_win32.fileMapping);

    if(m_mappingType == MAPPING_READOVERWRITE)
    {
      // truncate file to minimum size
      // To work with 64-bit file pointers, you can declare a LONG, treat it as the upper half
      // of the 64-bit file pointer, and pass its address in lpDistanceToMoveHigh. This means
      // you have to treat two different variables as a logical unit, which is error-prone.
      // The problems can be ameliorated by using the LARGE_INTEGER structure to create a 64-bit
      // value and passing the two 32-bit values by means of the appropriate elements of the union.
      // (see msdn documentation on SetFilePointer)
      LARGE_INTEGER li;
      li.QuadPart = (__int64)m_fileSize;
      SetFilePointer(m_win32.file, li.LowPart, &li.HighPart, FILE_BEGIN);

      SetEndOfFile(m_win32.file);
    }
    CloseHandle(m_win32.file);

    m_mappingPtr        = nullptr;
    m_win32.fileMapping = nullptr;
    m_win32.file        = nullptr;

#elif defined(__linux__)
    assert(m_unix.file != -1);

    munmap(m_mappingPtr, m_mappingSize);
    ::close(m_unix.file);

    m_mappingPtr = nullptr;
    m_unix.file  = -1;
#endif

    m_isValid = false;
  }
}

size_t FileMapping::g_pageSize = 0;

}  // namespace nvutils

//--------------------------------------------------------------------------------------------------
// Usage example
//--------------------------------------------------------------------------------------------------
[[maybe_unused]] static void usage_FileMapping()
{
  // use the class to memory map some input file
  nvutils::FileReadMapping readOnlyMapping;

  if(!readOnlyMapping.open("input.bin"))
    return;

  size_t elementCount = readOnlyMapping.size() / sizeof(float);

  // and then the appropriate output file
  nvutils::FileReadOverWriteMapping readOverWriteMapping;
  if(!readOverWriteMapping.open("output.bin", elementCount * sizeof(float)))
    return;

  // use the pointers directly to read or write to the files!
  const float* inputData  = static_cast<const float*>(readOnlyMapping.data());
  float*       outputData = static_cast<float*>(readOverWriteMapping.data());

  for(size_t i = 0; i < elementCount; i++)
  {
    outputData[i] = inputData[i] * 2.0f;
  }

  // The destructor will close the handle automatically.
  // One can use move assignment as well to close
  readOverWriteMapping = {};

  // or fully manually close
  readOnlyMapping.close();
  // and open something else
  readOnlyMapping.open("blubb.blah");
}
