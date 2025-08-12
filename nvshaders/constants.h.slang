/*
 * Copyright (c) 2024-2025, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */


#ifndef CONSTANTS_SLANG
#define CONSTANTS_SLANG

#include "slang_types.h"

NAMESPACE_SHADERIO_BEGIN()

#ifndef M_PI
static const float M_PI = 3.1415926535897F;  // PI
#endif

#ifndef M_TWO_PI
static const float M_TWO_PI = 6.2831853071795F;  // 2*PI
#endif

#ifndef M_PI_2
static const float M_PI_2 = 1.5707963267948F;  // PI/2
#endif

#ifndef M_PI_4
static const float M_PI_4 = 0.7853981633974F;  // PI/4
#endif

#ifndef M_1_OVER_PI
static const float M_1_OVER_PI = 0.3183098861837F;  // 1/PI
#endif

#ifndef M_2_OVER_PI
static const float M_2_OVER_PI = 0.6366197723675F;  // 2/PI
#endif

#ifndef M_1_PI
static const float M_1_PI = 0.3183098861837F;  // 1/PI
#endif

#ifndef INFINITE
static const float INFINITE = 1e32F;
#endif

// Special PDF value returned by `bsdfSample()` to represent an infinite impulse or singularity.
#ifndef DIRAC
static const float DIRAC = -1.0F;
#endif

NAMESPACE_SHADERIO_END()

#endif  // CONSTANTS_GLSL