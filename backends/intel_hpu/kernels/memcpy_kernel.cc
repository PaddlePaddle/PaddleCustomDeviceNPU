// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/phi/capi/all.h"

namespace custom_kernel {

template <typename T>
void MemcpyH2DKernel(const phi::Context& dev_ctx,
                     const phi::DenseTensor& x,
                     int dst_place_type,
                     phi::DenseTensor* out) {
}

template <typename T>
void MemcpyD2HKernel(const phi::Context& dev_ctx,
                     const phi::DenseTensor& x,
                     int dst_place_type,
                     phi::DenseTensor* out) {
}

}  // namespace custom_kernel

PD_BUILD_PHI_KERNEL(
    memcpy_h2d, intel_hpu, ALL_LAYOUT, custom_kernel::MemcpyH2DKernel, float) {}

PD_BUILD_PHI_KERNEL(
    memcpy_d2h, intel_hpu, ALL_LAYOUT, custom_kernel::MemcpyD2HKernel, float) {}
