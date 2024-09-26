// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include "common/gcu_op_runner.h"
#include "kernels/funcs/gcu_kernel_funcs.h"

namespace custom_kernel {

template <typename T, typename Context>
void FlipKernel(const Context& dev_ctx,
                const phi::DenseTensor& x,
                const std::vector<int>& axis,
                phi::DenseTensor* out) {
  PADDLE_GCU_KERNEL_TRACE("flip");
  dev_ctx.template Alloc<T>(out);

  if (LaunchAOTKernel()) {
    auto x_dims_size = x.dims().size();
    std::vector<int64_t> dimensions(axis.size(), 0);
    for (size_t i = 0; i < axis.size(); ++i) {
      int64_t dim = axis[i];
      dimensions[i] = dim < 0 ? (dim + x_dims_size) : dim;
    }
    LAUNCH_TOPSATENOP(topsatenFlip, dev_ctx, *out, x, dimensions);

  } else {  // kernel impl base on JIT
    THROW_JIT_UNIMPLEMENTED();
  }
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(flip,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::FlipKernel,
                          int,
                          float,
                          phi::dtype::float16) {}
