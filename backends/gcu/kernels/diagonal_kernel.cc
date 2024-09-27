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
void DiagonalKernel(const Context& dev_ctx,
                    const phi::DenseTensor& x,
                    int offset,
                    int axis1,
                    int axis2,
                    phi::DenseTensor* out) {
  PADDLE_GCU_KERNEL_TRACE("diagonal");
  dev_ctx.template Alloc<T>(out);
  LAUNCH_TOPSCLOP(diagonal,
                  dev_ctx,
                  *out,
                  x,
                  static_cast<int64_t>(offset),
                  static_cast<int64_t>(axis1),
                  static_cast<int64_t>(axis2));
}
}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(diagonal,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::DiagonalKernel,
                          float,
                          double,
                          int,
                          int64_t,
                          bool,
                          phi::dtype::float16) {}
