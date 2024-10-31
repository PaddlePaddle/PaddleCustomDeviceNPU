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

#include "kernels/funcs/npu_funcs.h"
#include "kernels/funcs/npu_op_runner.h"

namespace custom_kernel {

template <typename T, typename Context>
void AclopFlipKernel(const Context& dev_ctx,
                     const phi::DenseTensor& x,
                     const std::vector<int>& axis,
                     phi::DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);
  auto stream = dev_ctx.stream();

  const int total_dims = x.dims().size();
  std::vector<int> axis_trans = axis;
  for (size_t i = 0; i < axis_trans.size(); ++i) {
    axis_trans[i] =
        axis_trans[i] < 0 ? axis_trans[i] + total_dims : axis_trans[i];
  }

  NpuOpRunner runner;
  runner.SetType("ReverseV2")
      .AddInput(x)
      .AddInput(dev_ctx, std::move(axis_trans))
      .AddOutput(*out);
  runner.Run(stream);
}

template <typename T, typename Context>
void FlipKernel(const Context& dev_ctx,
                const phi::DenseTensor& x,
                const std::vector<int>& axis,
                phi::DenseTensor* out) {
  DO_COMPATIBILITY(
      aclnnFlip,
      (custom_kernel::AclopFlipKernel<T, Context>(dev_ctx, x, axis, out)));
  dev_ctx.template Alloc<T>(out);
  std::vector<int64_t> axis_trans(axis.begin(), axis.end());
  EXEC_NPU_CMD(aclnnFlip, dev_ctx, x, axis_trans, *out);
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(flip,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::FlipKernel,
                          phi::dtype::float16,
                          float,
                          double,
                          int32_t,
                          int64_t,
                          bool) {}
