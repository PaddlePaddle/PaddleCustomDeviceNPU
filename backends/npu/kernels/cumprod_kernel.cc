/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "kernels/funcs/npu_funcs.h"
#include "kernels/funcs/npu_op_runner.h"

namespace custom_kernel {

template <typename T, typename Context>
void CumprodKernel(const Context& dev_ctx,
                   const phi::DenseTensor& input,
                   const int dim,
                   bool exclusive,
                   bool reverse,
                   phi::DenseTensor* out) {
  auto stream = dev_ctx.stream();
  dev_ctx.template Alloc<T>(out);

  NPUAttributeMap attr_input = {
      {"axis", dim}, {"exclusive", exclusive}, {"reverse", reverse}};

  const auto& runner = NpuOpRunner("Cumprod", {input}, {*out}, attr_input);
  runner.Run(stream);
}
}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(cumprod,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::CumprodKernel,
                          float,
                          double,
                          int,
                          int64_t,
                          phi::dtype::complex<float>,
                          phi::dtype::complex<double>) {}
