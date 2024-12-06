// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
void TakeAlongAxisKernel(const Context& dev_ctx,
                         const DenseTensor& x,
                         const DenseTensor& index,
                         int axis,
                         DenseTensor* out) {
  PADDLE_GCU_KERNEL_TRACE("take_along_axis");
  if (LaunchAOTKernel()) {
    dev_ctx.template Alloc<T>(out);

    const auto x_shape = x.dims();
    const auto x_rank = x_shape.size();
    axis = axis < 0 ? axis + x_rank : axis;

    // check valid
    // average divide if num_split has only one value
    PADDLE_ENFORCE_GT(axis,
                      0,
                      phi::errors::InvalidArgument(
                          "axis should be in [-%zu, %zu)!", x_rank, x_rank));

    int64_t axis_64 = axis;
    phi::DenseTensor out_tmp = custom_kernel::TensorEmpty(dev_ctx, x.meta());
    LAUNCH_TOPSATENOP(
        topsatenGather, dev_ctx, out_tmp, x, index, axis_64, false);

    LAUNCH_TOPSATENOP(topsatenCopy, dev_ctx, *out, x, false);
  } else {  // kernel impl base on JIT
    THROW_JIT_UNIMPLEMENTED();
  }
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(take_along_axis,
                          GPU,
                          ALL_LAYOUT,
                          custom_kernel::TakeAlongAxisKernel,
                          float,
                          double,
                          int64_t,
                          int,
                          phi::dtype::float16) {}
