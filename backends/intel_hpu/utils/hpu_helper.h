// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once

#include <map>
#include <mutex>
#include <set>
#include <unordered_map>
#include <vector>

#include "habanalabs/synapse_api.h"
#include "habanalabs/synapse_common_types.h"

synStatus HostMap(const synDeviceId deviceId,
                  const uint64_t size,
                  const void* buffer);
synStatus HostUnmap(const synDeviceId deviceId, const void* buffer);

synStatus hbmAlloc(synDeviceId deviceId,
                   uint64_t size,
                   uint64_t* addr,
                   std::string name);
synStatus hbmFree(synDeviceId deviceId, uint64_t addr, const char* name);
synStatus hostAlloc(synDeviceId deviceId,
                    uint64_t size,
                    uint64_t* addr,
                    std::string name);
synStatus hostFree(synDeviceId deviceId, uint64_t addr, const char* name);

void resetTensorSections();
