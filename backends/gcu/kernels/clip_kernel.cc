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
void ClipKernel(const Context& dev_ctx,
                const phi::DenseTensor& x,
                const phi::Scalar& min,
                const phi::Scalar& max,
                phi::DenseTensor* out) {
  PADDLE_GCU_KERNEL_TRACE("clip");
  dev_ctx.template Alloc<T>(out);

  if (LaunchAOTKernel()) {
    LAUNCH_TOPSATENOP(topsatenClip, dev_ctx, *out, x, min, max);

  } else {  // kernel impl base on JIT
    auto max_ = max.to<T>();
    auto min_ = min.to<T>();

    PADDLE_ENFORCE_LE(min_,
                      max_,
                      phi::errors::InvalidArgument(
                          "max should be greater than or equal to min. "
                          "But received min = %f, max = %f",
                          static_cast<float>(min_),
                          static_cast<float>(max_)));

    TensorNameMap input_names;
    input_names["X"] = {"x"};

    TensorValueMap inputs;
    inputs["X"] = {const_cast<DenseTensor*>(&x)};

    TensorNameMap output_names;
    output_names["Out"] = {"out"};

    TensorValueMap outputs;
    outputs["Out"] = {out};

    GcuAttributeMap attrs;
    attrs["min"] = static_cast<float>(min_);
    attrs["max"] = static_cast<float>(max_);

    GcuRunner(
        input_names, inputs, output_names, outputs, attrs, "clip", dev_ctx);
  }
}

template <typename T, typename Context>
void ClipGradKernel(const Context& dev_ctx,
                    const phi::DenseTensor& x,
                    const phi::DenseTensor& dout,
                    const phi::Scalar& min,
                    const phi::Scalar& max,
                    phi::DenseTensor* dx) {
  PADDLE_GCU_KERNEL_TRACE("clip_grad");
  dev_ctx.template Alloc<T>(dx);
  if (LaunchAOTKernel()) {
    THROW_AOT_UNIMPLEMENTED();
  } else {  // kernel impl base on JIT
    auto max_ = max.to<T>();
    auto min_ = min.to<T>();

    TensorNameMap input_names;
    input_names["X"] = {"x"};
    input_names[GradVarName("Out")] = {"dout"};

    TensorValueMap inputs;
    inputs["X"] = {const_cast<DenseTensor*>(&x)};
    inputs[GradVarName("Out")] = {const_cast<DenseTensor*>(&dout)};

    TensorNameMap output_names;
    output_names[GradVarName("X")] = {"dx"};

    TensorValueMap outputs;
    outputs[GradVarName("X")] = {dx};

    GcuAttributeMap attrs;
    attrs["min"] = static_cast<float>(min_);
    attrs["max"] = static_cast<float>(max_);

    GcuRunner(input_names,
              inputs,
              output_names,
              outputs,
              attrs,
              "clip_grad",
              dev_ctx);
  }
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(clip,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::ClipKernel,
                          int,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(clip_grad,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::ClipGradKernel,
                          int,
                          int64_t,
                          float,
                          phi::dtype::float16,
                          double) {}
