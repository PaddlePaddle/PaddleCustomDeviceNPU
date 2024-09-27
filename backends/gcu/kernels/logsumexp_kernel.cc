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
void LogsumexpKernel(const Context& dev_ctx,
                     const phi::DenseTensor& x,
                     const std::vector<int64_t>& axis,
                     bool keepdim,
                     bool reduce_all,
                     phi::DenseTensor* out) {
  PADDLE_GCU_KERNEL_TRACE("logsumexp");
  if (LaunchAOTKernel()) {
    dev_ctx.template Alloc<T>(out);

    auto reduce_axis = axis;
    int64_t rank = x.dims().size();
    if (reduce_all || reduce_axis.empty()) {
      reduce_axis.assign(rank, 0);
      std::iota(reduce_axis.begin(), reduce_axis.end(), 0);
    } else {
      for (size_t i = 0; i < reduce_axis.size(); ++i) {
        if (reduce_axis[i] < 0) {
          reduce_axis[i] += rank;
        }
      }
    }
    LAUNCH_TOPSATENOP(
        topsatenLogsumexp, dev_ctx, *out, x, reduce_axis, keepdim);

  } else {  // kernel impl base on JIT
    THROW_JIT_UNIMPLEMENTED();
  }
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(logsumexp,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::LogsumexpKernel,
                          float,
                          double,
                          phi::dtype::float16) {}
