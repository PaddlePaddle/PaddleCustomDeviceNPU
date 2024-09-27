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
void Atan2Kernel(const Context& dev_ctx,
                 const phi::DenseTensor& x,
                 const phi::DenseTensor& y,
                 phi::DenseTensor* out) {
  PADDLE_GCU_KERNEL_TRACE("atan2");
  PADDLE_ENFORCE_LE(
      x.numel(),
      y.numel(),
      phi::errors::InvalidArgument("The count (%d) of elements of X shall not "
                                   "greater than count (%d) of elements of Y.",
                                   x.numel(),
                                   y.numel()));
  if (LaunchAOTKernel()) {
    dev_ctx.template Alloc<T>(out);
    LAUNCH_TOPSATENOP(topsatenAtan2, dev_ctx, *out, x, y);

  } else {  // kernel impl base on JIT
    THROW_JIT_UNIMPLEMENTED();
  }
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(atan2,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::Atan2Kernel,
                          int,
                          int64_t,
                          float,
                          double,
                          phi::dtype::float16) {
  kernel->OutputAt(0).SetDataType(phi::DataType::UNDEFINED);
}
