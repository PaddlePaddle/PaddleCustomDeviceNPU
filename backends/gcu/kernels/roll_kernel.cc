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
void RollKernel(const Context& dev_ctx,
                const phi::DenseTensor& x,
                const phi::IntArray& shifts,
                const std::vector<int64_t>& axis,
                phi::DenseTensor* out) {
  PADDLE_GCU_KERNEL_TRACE("roll");

  if (LaunchAOTKernel()) {
    dev_ctx.template Alloc<T>(out);
    auto roll_shifts = shifts.GetData();
    auto roll_axis = axis;
    if (!(roll_axis.empty())) {
      int64_t rank = x.dims().size();
      for (size_t i = 0; i < roll_axis.size(); ++i) {
        if (roll_axis[i] < 0) {
          roll_axis[i] += rank;
        }
      }
    }

    LAUNCH_TOPSATENOP(topsatenRoll, dev_ctx, *out, x, roll_shifts, roll_axis);

  } else {  // kernel impl base on JIT
    THROW_JIT_UNIMPLEMENTED();
  }
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(roll,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::RollKernel,
                          int,
                          int64_t,
                          float,
                          double,
                          phi::dtype::float16) {}
