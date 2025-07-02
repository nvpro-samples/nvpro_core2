/*
 * Copyright (c) 2022-2025, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */


#ifndef TONEMAMP_SHADERIO_H
#define TONEMAMP_SHADERIO_H 1

enum ToneMapMethod
{
  eFilmic = 0,
  eUncharted2,
  eClip,
  eACES,
  eAgX,
  eKhronosPBR,
  eCount
};


// Bindings
enum TonemapBinding
{
  eInput = 0,
  eOutput
};


/*-------------------------------------------------------------------------------------------------
# Struct `Tonemapper`
> Tonemapper settings.
-------------------------------------------------------------------------------------------------*/
struct TonemapperData
{
  int   method     = ToneMapMethod::eFilmic;
  int   isActive   = 1;
  float exposure   = 1.0f;
  float brightness = 1.0f;
  float contrast   = 1.0f;
  float saturation = 1.0f;
  float vignette   = 0.0f;
};


#endif  // TONEMAMP_SHADERIO_H
