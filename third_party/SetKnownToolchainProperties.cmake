# Various packages call try_compile() to see if compiler features are available.
# This can take a long time on Windows!
# Based on our knowledge of supported platforms and minimum compiler versions,
# we can set whether features are available in advance, and speed up
# CMake configuration.
# If we don't recognize the platform, we fall back to autodetection.
set(NVPRO2_SKIP_COMPILER_FEATURE_DETECTION_DEFAULT OFF)
if("${CMAKE_SYSTEM_PROCESSOR}" MATCHES "(x86_64|AMD64)") # Is x86_64?
  if(WIN32 OR UNIX) # One of the OSes we know?
    set(NVPRO2_SKIP_COMPILER_FEATURE_DETECTION_DEFAULT ON)
  endif()
endif()
option(NVPRO2_SKIP_COMPILER_FEATURE_DETECTION
       "Assume minimum compiler features exist instead of running test compiles"
	   ${NVPRO2_SKIP_COMPILER_FEATURE_DETECTION_DEFAULT}
)
if(NVPRO2_SKIP_COMPILER_FEATURE_DETECTION)
  message(STATUS "Skipping toolchain feature tests, since we recognize this toolchain. \
If you want to force feature autodetection, set NVPRO2_SKIP_COMPILER_FEATURE_DETECTION to OFF.")
  # GLFW queries FindThreads.
  # The quotes make it so that these variables are always 'defined'.
  # On Win32, we assume we're using Microsoft's STL, and on UNIX, we
  # assume we're using glibc or musl and libstdc++ or libc++.
  set(CMAKE_HAVE_LIBC_PTHREAD "${UNIX}")
  set(CMAKE_HAVE_PTHREADS_CREATE OFF)
  set(CMAKE_HAVE_PTHREAD_CREATE OFF)
  # zlib-ng queries architecture information.
  set(HAVE_ARM_ACLE_H OFF) # ARM
  set(HAVE_SYS_AUXV_H "${UNIX}")
  set(HAVE_SYS_SDT_H OFF) # System390
  set(HAVE_UNISTD_H "${UNIX}")
  set(HAVE_LINUX_AUXVEC_H "${UNIX}")
  set(HAVE_SYS_TYPES_H ON)
  set(HAVE_STDINT_H ON)
  set(HAVE_STDDEF_H ON)
  set(HAVE_OFF64_T "${UNIX}")
  set(OFF64_T 8)
  set(HAVE__OFF64_T OFF)
  set(HAVE___OFF64_T OFF)
  set(HAVE_FSEEKO "${UNIX}")
  set(HAVE_STRERROR ON)
  set(HAVE_POSIX_MEMALIGN "${UNIX}")
  set(HAVE_ALIGNED_ALLOC "${UNIX}")
  set(HAVE_NO_INTERPOSITION "${UNIX}")
  set(HAVE_ATTRIBUTE_VISIBILITY_HIDDEN "${UNIX}")
  set(HAVE_ATTRIBUTE_VISIBILITY_INTERNAL "${UNIX}")
  set(HAVE_ATTRIBUTE_ALIGNED "${UNIX}")
  set(HAVE_BUILTIN_ASSUME_ALIGNED ON)
  set(HAVE_PTRDIFF_T ON)
  set(HAVE_XSAVE_INTRIN ON)
  if ((CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang") AND (CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64"))
    set(HAVE_BUILTIN_CTZ ON)
    set(HAVE_BUILTIN_CTZLL ON)
  elseif (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    #zLib comes with a fallback for MSVC but that requires us to provide OFF here
    set(HAVE_BUILTIN_CTZ OFF)
    set(HAVE_BUILTIN_CTZLL OFF)
  else()
     # on other toolchains, just run the test
  endif()

  # Note that these are just whether the compiler has the header files for the
  # intrinsics -- most x86_64 machines don't support AVX512. This simply allows
  # zlib-ng to compile in code paths that are taken if the runtime system
  # supports these architectures.
  set(HAVE_SSE2_INTRIN ON)
  set(HAVE_SSSE3_INTRIN ON)
  set(HAVE_SSE42_INTRIN ON)
  set(HAVE_PCLMULQDQ_INTRIN ON)
  set(HAVE_AVX2_INTRIN ON)
  set(HAVE_AVX512_INTRIN ON)
  set(HAVE_AVX512VNNI_INTRIN ON)
  set(HAVE_VPCLMULQDQ_INTRIN ON)
  # Zstd checks for various compiler flags; it only checks for these if the
  # compiler is of a certain type.
  foreach(LANG C CXX)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang" OR MINGW)
	  foreach(FLAG WALL WEXTRA WUNDEF WSHADOW WCAST_ALIGN WCAST_QUAL WSTRICT_PROTOTYPES WA_NOEXECSTACK)
	    set(${LANG}_FLAG_${FLAG} ON)
	  endforeach()
	  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
		set(_IS_CLANG ON)
	  endif()
	  set(${LANG}_FLAG_QUNUSED_ARGUMENTS "${_IS_CLANG}")
	  set(LD_FLAG_WL_Z_NOEXECSTACK ON)
	elseif(MSVC)
	  foreach(FLAG MP D_UNICODE DUNICODE)
		set(${LANG}_FLAG_${FLAG} ON)
	  endforeach()
	endif()
  endforeach()
endif()