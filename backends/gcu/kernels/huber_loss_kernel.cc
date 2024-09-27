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
void HuberLossKernel(const Context& dev_ctx,
                     const phi::DenseTensor& input,
                     const phi::DenseTensor& label,
                     float delta,
                     phi::DenseTensor* out,
                     phi::DenseTensor* residual) {
  PADDLE_GCU_KERNEL_TRACE("huber_loss");
  if (LaunchAOTKernel()) {
    dev_ctx.template Alloc<T>(out);
    int64_t reduction = 0;  // none, the unreduced loss is returned
    double d_delta = delta;
    LAUNCH_TOPSATENOP(
        topsatenHuberLoss, dev_ctx, *out, input, label, reduction, d_delta);
    *residual = custom_kernel::Subtract(dev_ctx, label, input);

  } else {  // kernel impl base on JIT
    out->Resize(input.dims());
    residual->Resize(label.dims());
    dev_ctx.template Alloc<T>(out);
    dev_ctx.template Alloc<T>(residual);

    TensorNameMap input_names;
    input_names["X"] = {"input"};
    input_names["Y"] = {"label"};

    TensorValueMap inputs;
    inputs["X"] = {const_cast<DenseTensor*>(&input)};
    inputs["Y"] = {const_cast<DenseTensor*>(&label)};

    TensorNameMap output_names;
    output_names["Out"] = {"out"};
    output_names["Residual"] = {"residual"};

    TensorValueMap outputs;
    outputs["Out"] = {out};
    outputs["Residual"] = {residual};

    GcuAttributeMap attrs;
    attrs["delta"] = delta;

    GcuRunner(input_names,
              inputs,
              output_names,
              outputs,
              attrs,
              "huber_loss",
              dev_ctx);
  }
}

template <typename T, typename Context>
void HuberLossGradKernel(const Context& dev_ctx,
                         const phi::DenseTensor& residual,
                         const phi::DenseTensor& dout,
                         float delta,
                         phi::DenseTensor* dx,
                         phi::DenseTensor* dy) {
  PADDLE_GCU_KERNEL_TRACE("huber_loss_grad");
  if (LaunchAOTKernel()) {
    THROW_AOT_UNIMPLEMENTED();
  } else {  // kernel impl base on JIT
    TensorNameMap input_names;
    input_names["Residual"] = {"residual"};
    input_names[GradVarName("Out")] = {"dout"};

    TensorValueMap inputs;
    inputs["Residual"] = {const_cast<DenseTensor*>(&residual)};
    inputs[GradVarName("Out")] = {const_cast<DenseTensor*>(&dout)};

    TensorNameMap output_names;
    TensorValueMap outputs;
    if (dx) {
      dev_ctx.template Alloc<T>(dx);
      output_names[GradVarName("X")] = {"dx"};
      outputs[GradVarName("X")] = {dx};
    }
    if (dy) {
      dev_ctx.template Alloc<T>(dy);
      output_names[GradVarName("Y")] = {"dy"};
      outputs[GradVarName("Y")] = {dy};
    }

    GcuAttributeMap attrs;
    attrs["delta"] = delta;

    GcuRunner(input_names,
              inputs,
              output_names,
              outputs,
              attrs,
              "huber_loss_grad",
              dev_ctx);
  }
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(huber_loss,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::HuberLossKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(huber_loss_grad,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::HuberLossGradKernel,
                          float,
                          double,
                          phi::dtype::float16) {}
