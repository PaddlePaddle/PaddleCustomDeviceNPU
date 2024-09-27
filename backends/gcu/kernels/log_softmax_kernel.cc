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
void LogSoftmaxKernel(const Context& dev_ctx,
                      const phi::DenseTensor& x,
                      int axis,
                      phi::DenseTensor* out) {
  PADDLE_GCU_KERNEL_TRACE("log_softmax");
  dev_ctx.template Alloc<T>(out);

  if (LaunchAOTKernel()) {
    LAUNCH_TOPSATENOP(topsatenLogSoftmaxForward, dev_ctx, *out, x, axis);

  } else {  // kernel impl base on JIT
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
              "log_softmax",
              dev_ctx);
  }
}

template <typename T, typename Context>
void LogSoftmaxGradKernel(const Context& dev_ctx,
                          const phi::DenseTensor& out,
                          const phi::DenseTensor& dout,
                          int axis,
                          phi::DenseTensor* dx) {
  PADDLE_GCU_KERNEL_TRACE("log_softmax_grad");
  dev_ctx.template Alloc<T>(dx);

  if (LaunchAOTKernel()) {
    THROW_AOT_UNIMPLEMENTED();
  } else {  // kernel impl base on JIT
    TensorNameMap input_names;
    input_names["Out"] = {"out"};
    input_names[GradVarName("Out")] = {"dout"};

    TensorValueMap inputs;
    inputs["Out"] = {const_cast<DenseTensor*>(&out)};
    inputs[GradVarName("Out")] = {const_cast<DenseTensor*>(&dout)};

    TensorNameMap output_names;
    output_names[GradVarName("X")] = {"dx"};

    TensorValueMap outputs;
    outputs[GradVarName("X")] = {dx};

    GcuAttributeMap attrs;
    attrs["axis"] = axis;

    GcuRunner(input_names,
              inputs,
              output_names,
              outputs,
              attrs,
              "log_softmax_grad",
              dev_ctx);
  }
}
}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(log_softmax,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::LogSoftmaxKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(log_softmax_grad,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::LogSoftmaxGradKernel,
                          float,
                          phi::dtype::float16,
                          double) {}
