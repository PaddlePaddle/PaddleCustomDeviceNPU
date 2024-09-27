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
void IscloseKernel(const Context& dev_ctx,
                   const phi::DenseTensor& x,
                   const phi::DenseTensor& y,
                   const phi::Scalar& rtol,
                   const phi::Scalar& atol,
                   bool equal_nan,
                   phi::DenseTensor* out) {
  PADDLE_GCU_KERNEL_TRACE("isclose");
  PADDLE_ENFORCE_EQ(
      rtol.dtype(),
      phi::DataType::FLOAT64,
      phi::errors::InvalidArgument("Input(Rtol) type must be double"));

  PADDLE_ENFORCE_EQ(
      atol.dtype(),
      phi::DataType::FLOAT64,
      phi::errors::InvalidArgument("Input(Atol) type must be double"));

  dev_ctx.template Alloc<bool>(out);
  double cmp_rtol = rtol.to<double>();
  double cmp_atol = atol.to<double>();

  if (LaunchAOTKernel()) {
    LAUNCH_TOPSATENOP(
        topsatenIsclose, dev_ctx, *out, x, y, cmp_rtol, cmp_atol, equal_nan);

  } else {  // kernel impl base on JIT
    THROW_JIT_UNIMPLEMENTED();
  }
}
}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(isclose,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::IscloseKernel,
                          float,
                          double,
                          phi::dtype::float16) {}
