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
void TransposeKernel(const Context& dev_ctx,
                     const phi::DenseTensor& x,
                     const std::vector<int>& axis,
                     phi::DenseTensor* out) {
  PADDLE_GCU_KERNEL_TRACE("transpose");
  if (LaunchAOTKernel()) {
    std::vector<int64_t> in_axis(axis.begin(), axis.end());
    *out = custom_kernel::Transpose(dev_ctx, x, in_axis);

  } else {  // kernel impl base on JIT
    dev_ctx.template Alloc<T>(out);

    TensorNameMap input_names;
    input_names["X"] = {"x"};

    TensorValueMap inputs;
    inputs["X"] = {const_cast<DenseTensor*>(&x)};

    TensorNameMap output_names;
    output_names["Out"] = {"out"};

    TensorValueMap outputs;
    outputs["Out"] = {out};

    GcuAttributeMap attrs;
    attrs["axis"] = axis;

    GcuRunner(input_names,
              inputs,
              output_names,
              outputs,
              attrs,
              "transpose2",
              dev_ctx);
  }
}

template <typename T, typename Context>
void TransposeGradKernel(const Context& dev_ctx,
                         const phi::DenseTensor& dout,
                         const std::vector<int>& axis,
                         phi::DenseTensor* dx) {
  PADDLE_GCU_KERNEL_TRACE("transpose_grad");
  dev_ctx.template Alloc<T>(dx);
  if (LaunchAOTKernel()) {
    THROW_AOT_UNIMPLEMENTED();
  } else {  // kernel impl base on JIT
    TensorNameMap input_names;
    input_names[GradVarName("Out")] = {"dout"};

    TensorValueMap inputs;
    inputs[GradVarName("Out")] = {const_cast<DenseTensor*>(&dout)};

    TensorNameMap output_names;
    TensorValueMap outputs;

    output_names[GradVarName("X")] = {"dx"};
    outputs[GradVarName("X")] = {dx};

    GcuAttributeMap attrs;
    attrs["axis"] = axis;

    GcuRunner(input_names,
              inputs,
              output_names,
              outputs,
              attrs,
              "transpose2_grad",
              dev_ctx);
  }
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(transpose,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::TransposeKernel,
                          int,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(transpose_grad,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::TransposeGradKernel,
                          int,
                          int64_t,
                          uint8_t,
                          int8_t,
                          float,
                          double,
                          phi::dtype::float16) {}
