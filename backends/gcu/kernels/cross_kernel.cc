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
void CrossKernel(const Context& dev_ctx,
                 const phi::DenseTensor& x,
                 const phi::DenseTensor& y,
                 int axis,
                 phi::DenseTensor* out) {
  PADDLE_GCU_KERNEL_TRACE("copysign");
  int64_t dim = axis;

  auto x_dims = x.dims();

  if (dim != common::DDim::kMaxRank) {
    PADDLE_ENFORCE_EQ(
        dim < x_dims.size() && dim >= (0 - x_dims.size()),
        true,
        phi::errors::OutOfRange(
            "Attr(dim) is out of range, It's expected "
            "to be in range of [-%d, %d]. But received Attr(dim) = %d.",
            x_dims.size(),
            x_dims.size() - 1,
            dim));
    if (dim < 0) {
      dim += x_dims.size();
    }

    PADDLE_ENFORCE_EQ(
        x_dims[dim] == 3,
        true,
        phi::errors::InvalidArgument(
            "Input(X/Y).dims[dim] must be equal to 3. But received: "
            "Input(X/Y).dims[dim] = [%d].",
            x_dims[dim]));
  } else {
    for (auto i = 0; i < x_dims.size(); i++) {
      if (x_dims[i] == 3) {
        dim = i;
        break;
      }
    }
    PADDLE_ENFORCE_EQ(dim == common::DDim::kMaxRank,
                      false,
                      phi::errors::InvalidArgument(
                          "There must be at least one dimension 'd' so that "
                          "Input(X/Y).dims()[d] is equal to 3. "
                          "But received: Input(X/Y).dims() == [%s].",
                          x_dims));
  }

  dev_ctx.template Alloc<T>(out);
  if (LaunchAOTKernel()) {
    phi::Scalar cross_dim(dim);
    LAUNCH_TOPSATENOP(topsatenCross, dev_ctx, *out, x, y, cross_dim);

  } else {  // kernel impl base on JIT
    THROW_JIT_UNIMPLEMENTED();
  }
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(cross,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::CrossKernel,
                          int,
                          int64_t,
                          float,
                          double,
                          phi::dtype::float16) {}
