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
void NumelKernel(const Context& dev_ctx,
                 const phi::DenseTensor& input,
                 phi::DenseTensor* out) {
  PADDLE_GCU_KERNEL_TRACE("numel");
  dev_ctx.template Alloc<int64_t>(out);

  if (LaunchAOTKernel()) {
    int64_t size = input.numel();
    FillGcuTensorWithConstant<int64_t>(out, dev_ctx, size);

  } else {  // kernel impl base on JIT
    TensorNameMap input_names;
    input_names["Input"] = {"input"};

    TensorValueMap inputs;
    inputs["Input"] = {const_cast<DenseTensor*>(&input)};

    TensorNameMap output_names;
    output_names["Out"] = {"out"};

    TensorValueMap outputs;
    outputs["Out"] = {out};

    GcuAttributeMap attrs;

    GcuRunner(
        input_names, inputs, output_names, outputs, attrs, "size", dev_ctx);
  }
}
}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(numel,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::NumelKernel,
                          uint8_t,
                          int16_t,
                          int,
                          int64_t,
                          phi::dtype::float16,
                          float,
                          double,
                          bool) {
  kernel->OutputAt(0).SetDataType(phi::DataType::INT64);
}
