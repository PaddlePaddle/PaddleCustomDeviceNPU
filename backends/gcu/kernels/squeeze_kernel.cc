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
void SqueezeKernel(const Context& dev_ctx,
                   const phi::DenseTensor& x,
                   const phi::IntArray& axes_int_array,
                   phi::DenseTensor* out) {
  PADDLE_GCU_KERNEL_TRACE("squeeze_infer");
  VLOG(6) << "[HOST_KERNEL] Impl on host for squeeze_infer";
  auto out_dims = out->dims();
  out->Resize(out_dims);
  dev_ctx.template Alloc<T>(out);

  TensorCopy(dev_ctx, x, false, out);
  out->Resize(out_dims);
}

template <typename T, typename Context>
void SqueezeWithXShapeKernel(const Context& dev_ctx,
                             const phi::DenseTensor& x,
                             const phi::IntArray& axes_int_array,
                             phi::DenseTensor* out,
                             phi::DenseTensor* xshape) {
  PADDLE_GCU_KERNEL_TRACE("squeeze");
  VLOG(6) << "[HOST_KERNEL] Impl on host for squeeze";
  custom_kernel::SqueezeKernel<T, Context>(dev_ctx, x, axes_int_array, out);
}

template <typename T, typename Context>
void SqueezeGradKernel(const Context& dev_ctx,
                       const phi::DenseTensor& x,
                       const phi::DenseTensor& dout,
                       const phi::IntArray& axes_int_array,
                       phi::DenseTensor* dx) {
  PADDLE_GCU_KERNEL_TRACE("squeeze_grad");
  VLOG(6) << "[HOST_KERNEL] Impl on host for squeeze_grad";
  auto x_dims = dx->dims();
  dev_ctx.template Alloc<T>(dx);

  TensorCopy(dev_ctx, dout, false, dx);
  dx->Resize(x_dims);
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(squeeze,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::SqueezeKernel,
                          bool,
                          int,
                          uint8_t,
                          int8_t,
                          int64_t,
                          float,
                          phi::dtype::float16,
                          double) {}

PD_REGISTER_PLUGIN_KERNEL(squeeze_with_xshape,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::SqueezeWithXShapeKernel,
                          bool,
                          int,
                          uint8_t,
                          int8_t,
                          int64_t,
                          float,
                          phi::dtype::float16,
                          double) {}

PD_REGISTER_PLUGIN_KERNEL(squeeze_grad,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::SqueezeGradKernel,
                          bool,
                          int,
                          uint8_t,
                          int8_t,
                          int64_t,
                          float,
                          phi::dtype::float16,
                          double) {}
