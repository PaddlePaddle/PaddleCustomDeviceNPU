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
void TrilTriuCommon(const std::string& op_type,
                    const Context& ctx,
                    const phi::DenseTensor& x,
                    int diagonal,
                    bool lower,
                    phi::DenseTensor* out) {
  ctx.template Alloc<T>(out);

  TensorNameMap input_names;
  input_names["X"] = {"x"};

  TensorValueMap inputs;
  inputs["X"] = {const_cast<DenseTensor*>(&x)};

  TensorNameMap output_names;
  output_names["Out"] = {"out"};

  TensorValueMap outputs;
  outputs["Out"] = {out};

  GcuAttributeMap attrs;
  attrs["diagonal"] = diagonal;
  attrs["lower"] = lower;

  GcuRunner(input_names, inputs, output_names, outputs, attrs, op_type, ctx);
}

template <typename T, typename Context>
void TrilTriuGradCommon(const std::string& op_type,
                        const Context& ctx,
                        const phi::DenseTensor& out_grad,
                        int diagonal,
                        bool lower,
                        phi::DenseTensor* x_grad) {
  ctx.template Alloc<T>(x_grad);

  TensorNameMap input_names;
  input_names[GradVarName("Out")] = {"out_grad"};

  TensorValueMap inputs;
  inputs[GradVarName("Out")] = {const_cast<DenseTensor*>(&out_grad)};

  TensorNameMap output_names;
  output_names[GradVarName("X")] = {"x_grad"};

  TensorValueMap outputs;
  outputs[GradVarName("X")] = {x_grad};

  GcuAttributeMap attrs;
  attrs["diagonal"] = diagonal;
  attrs["lower"] = lower;

  GcuRunner(input_names, inputs, output_names, outputs, attrs, op_type, ctx);
}

template <typename T, typename Context>
void TrilTriuKernel(const Context& ctx,
                    const phi::DenseTensor& x,
                    int diagonal,
                    bool lower,
                    phi::DenseTensor* out) {
  PADDLE_GCU_KERNEL_TRACE("tril_triu");
  if (LaunchAOTKernel()) {
    ctx.template Alloc<T>(out);
    if (lower) {
      LAUNCH_TOPSATENOP(topsatenTril, ctx, *out, x, diagonal);
    } else {
      LAUNCH_TOPSATENOP(topsatenTriu, ctx, *out, x, diagonal);
    }

  } else {  // kernel impl base on JIT
    custom_kernel::TrilTriuCommon<T, Context>(
        "tril_triu", ctx, x, diagonal, lower, out);
  }
}

template <typename T, typename Context>
void TrilTriuGradKernel(const Context& ctx,
                        const phi::DenseTensor& out_grad,
                        int diagonal,
                        bool lower,
                        phi::DenseTensor* x_grad) {
  PADDLE_GCU_KERNEL_TRACE("tril_triu_grad");
  if (LaunchAOTKernel()) {
    THROW_AOT_UNIMPLEMENTED();
  } else {  // kernel impl base on JIT
    custom_kernel::TrilTriuGradCommon<T, Context>(
        "tril_triu_grad", ctx, out_grad, diagonal, lower, x_grad);
  }
}

template <typename T, typename Context>
void TrilKernel(const Context& ctx,
                const phi::DenseTensor& x,
                int diagonal,
                phi::DenseTensor* out) {
  PADDLE_GCU_KERNEL_TRACE("tril");
  if (LaunchAOTKernel()) {
    ctx.template Alloc<T>(out);
    LAUNCH_TOPSATENOP(topsatenTril, ctx, *out, x, diagonal);

  } else {  // kernel impl base on JIT
    custom_kernel::TrilTriuCommon<T, Context>(
        "tril", ctx, x, diagonal, true, out);
  }
}

template <typename T, typename Context>
void TrilGradKernel(const Context& ctx,
                    const phi::DenseTensor& out_grad,
                    int diagonal,
                    phi::DenseTensor* x_grad) {
  PADDLE_GCU_KERNEL_TRACE("tril_grad");
  if (LaunchAOTKernel()) {
    THROW_AOT_UNIMPLEMENTED();
  } else {  // kernel impl base on JIT
    custom_kernel::TrilTriuGradCommon<T, Context>(
        "tril_grad", ctx, out_grad, diagonal, true, x_grad);
  }
}

template <typename T, typename Context>
void TriuKernel(const Context& ctx,
                const phi::DenseTensor& x,
                int diagonal,
                phi::DenseTensor* out) {
  PADDLE_GCU_KERNEL_TRACE("triu");
  if (LaunchAOTKernel()) {
    ctx.template Alloc<T>(out);
    LAUNCH_TOPSATENOP(topsatenTriu, ctx, *out, x, diagonal);

  } else {  // kernel impl base on JIT
    custom_kernel::TrilTriuCommon<T, Context>(
        "triu", ctx, x, diagonal, false, out);
  }
}

template <typename T, typename Context>
void TriuGradKernel(const Context& ctx,
                    const phi::DenseTensor& out_grad,
                    int diagonal,
                    phi::DenseTensor* x_grad) {
  PADDLE_GCU_KERNEL_TRACE("triu_grad");
  if (LaunchAOTKernel()) {
    THROW_AOT_UNIMPLEMENTED();
  } else {  // kernel impl base on JIT
    custom_kernel::TrilTriuGradCommon<T, Context>(
        "triu_grad", ctx, out_grad, diagonal, false, x_grad);
  }
}
}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(tril_triu,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::TrilTriuKernel,
                          bool,
                          float,
                          int,
                          int64_t,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(tril_triu_grad,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::TrilTriuGradKernel,
                          bool,
                          float,
                          int,
                          int64_t,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(tril,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::TrilKernel,
                          bool,
                          float,
                          int,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(tril_grad,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::TrilGradKernel,
                          bool,
                          float,
                          int,
                          int64_t,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(triu,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::TriuKernel,
                          bool,
                          float,
                          int,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(triu_grad,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::TriuGradKernel,
                          bool,
                          float,
                          int,
                          int64_t,
                          phi::dtype::float16) {}
