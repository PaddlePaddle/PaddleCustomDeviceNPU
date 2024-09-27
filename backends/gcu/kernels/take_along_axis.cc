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
    LAUNCH_TOPSCLOP(take_along_axis, dev_ctx, *out, x, index, axis)
  } else {  // kernel impl base on JIT
    THROW_JIT_UNIMPLEMENTED();
  }
}

template <typename T, typename Context>
void UnStackKernel(const Context& dev_ctx,
                   const phi::DenseTensor& x,
                   int axis,
                   int num,
                   std::vector<phi::DenseTensor*> outs) {
  PADDLE_GCU_KERNEL_TRACE("unstack");
  if (LaunchAOTKernel()) {
    for (auto y : outs) {
      dev_ctx.template Alloc<T>(y);
    }
    LAUNCH_TOPSCLOP(unstack, dev_ctx, &outs, x, axis)
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
