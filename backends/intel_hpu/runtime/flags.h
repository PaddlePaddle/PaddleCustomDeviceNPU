// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may
// not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Part of the following code in this file is from
//     https://github.com/google/glog/blob/master/src/base/commandlineflags.h
//     Git commit hash: 9f0b7d3bfe1542848f784e8d1c545b916cec6b3e
// Retain the following license from the original files:

// Copyright (c) 2008, Google Inc.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above
// copyright notice, this list of conditions and the following disclaimer
// in the documentation and/or other materials provided with the
// distribution.
//     * Neither the name of Google Inc. nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#ifndef BACKENDS_INTEL_HPU_RUNTIME_FLAGS_H_
#define BACKENDS_INTEL_HPU_RUNTIME_FLAGS_H_
#include "gflags/gflags.h"
#define FLAGS_DEFINE_bool(name, value, meaning) \
  DEFINE_bool(name, EnvToBool("FLAGS_" #name, value), meaning)

#define FLAGS_DEFINE_int32(name, value, meaning) \
  DEFINE_int32(name, EnvToInt("FLAGS_" #name, value), meaning)

#define FLAGS_DEFINE_uint32(name, value, meaning) \
  DEFINE_uint32(name, EnvToUInt("FLAGS_" #name, value), meaning)

#define FLAGS_DEFINE_uint64(name, value, meaning) \
  DEFINE_uint64(name, EnvToUInt("FLAGS_" #name, value), meaning)

#define FLAGS_DEFINE_string(name, value, meaning) \
  DEFINE_string(name, EnvToString("FLAGS_" #name, value), meaning)

#define EnvToString(envname, dflt) (!getenv(envname) ? (dflt) : getenv(envname))

#define EnvToBool(envname, dflt) \
  (!getenv(envname) ? (dflt) : memchr("tTyY1\0", getenv(envname)[0], 6) != NULL)

#define EnvToInt(envname, dflt) \
  (!getenv(envname) ? (dflt) : strtol(getenv(envname), NULL, 10))

#define EnvToUInt(envname, dflt) \
  (!getenv(envname) ? (dflt) : strtoul(getenv(envname), NULL, 10))

#endif  // BACKENDS_INTEL_HPU_RUNTIME_FLAGS_H_
