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
void HistogramKernel(const Context& dev_ctx,
                     const phi::DenseTensor& input,
                     const paddle::optional<phi::DenseTensor>& weight,
                     int64_t bins,
                     float min,
                     float max,
                     bool density,
                     phi::DenseTensor* output) {
  PADDLE_ENFORCE_EQ(
      weight || density,
      false,
      phi::errors::InvalidArgument("PaddlePaddle does not support parameters "
                                   "weight and density on the NPU."));

  dev_ctx.template Alloc<T>(output);
  EXEC_NPU_CMD(aclnnInplaceZero, dev_ctx, *output);

  int bins_trans = bins;
  float min_trans = min;
  float max_trans = max;

  NpuOpRunner runner;
  runner.SetType("Histogram")
      .AddInput(input)
      .AddOutput(*output)
      .AddAttr("bins", bins_trans)
      .AddAttr("min", min_trans)
      .AddAttr("max", max_trans);
  runner.Run(dev_ctx.stream());
}
};  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(histogram,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::HistogramKernel,
                          float,
                          int,
                          int64_t) {}
