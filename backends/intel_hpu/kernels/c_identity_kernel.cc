// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include <stdio.h>

#include "kernels/funcs.h"

namespace custom_kernel {

template <typename T, typename Context>
void CIdentityKernel(const Context& dev_ctx,
                     const phi::DenseTensor& x,
                     int ring_id,
                     bool use_calc_stream,
                     bool use_model_parallel,
                     phi::DenseTensor* out) {
  PADDLE_ENFORCE_GE(
      ring_id,
      0,
      phi::errors::InvalidArgument(
          "The ring_id (%d) for c_identity op must be non-negative.", ring_id));

  dev_ctx.template Alloc<T>(out);

  TensorCopy(dev_ctx, x, false, out);
  dev_ctx.Wait();
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(c_identity,
                          intel_hpu,
                          ALL_LAYOUT,
                          custom_kernel::CIdentityKernel,
                          float,
                          double,
                          int,
                          int64_t,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}
