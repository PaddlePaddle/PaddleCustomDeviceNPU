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
void StackKernel(const Context& dev_ctx,
                 const std::vector<const phi::DenseTensor*>& x,
                 int axis,
                 phi::DenseTensor* y) {
  PADDLE_GCU_KERNEL_TRACE("stack");
  dev_ctx.template Alloc<T>(y);
  if (LaunchAOTKernel()) {
    phi::DenseTensor output = MaybeCreateOrTrans64To32bits(dev_ctx, *y, false);
    auto out_tensor = CreateTopsatenTensor(output);

    std::vector<phi::DenseTensor> input_tensors;
    for (const auto& in : x) {
      input_tensors.emplace_back(MaybeCreateOrTrans64To32bits(dev_ctx, *in));
    }
    std::vector<topsatenTensor> in_tensors;
    for (const auto& in : input_tensors) {
      in_tensors.emplace_back(CreateTopsatenTensor(in));
    }
    int64_t dim = static_cast<int64_t>(axis);
    if (dim < 0 && !x.empty()) {
      dim += x[0]->dims().size() + 1;
    }
    std::string abstract_info = custom_kernel::GetAbstractInfo(
        "topsatenStack", output, input_tensors, dim);
    LAUNCH_TOPSATENOP_WITH_RAW_ATEN_DEF(
        topsatenStack, dev_ctx, abstract_info, out_tensor, in_tensors, dim);
    MaybeTransResult(dev_ctx, output, y);

  } else {  // kernel impl base on JIT
    TensorNameMap input_names;
    TensorValueMap inputs;
    std::vector<std::string> names;
    names.reserve(x.size());
    std::vector<phi::DenseTensor*> values;
    values.reserve(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
      names.emplace_back(std::string("x_") + std::to_string(i));
      values.emplace_back(const_cast<DenseTensor*>(x[i]));
    }
    input_names["X"] = names;
    inputs["X"] = values;

    TensorNameMap output_names;
    output_names["Y"] = {"y"};

    TensorValueMap outputs;
    outputs["Y"] = {y};

    GcuAttributeMap attrs;
    attrs["axis"] = axis;

    GcuRunner(
        input_names, inputs, output_names, outputs, attrs, "stack", dev_ctx);
  }
}

template <typename T, typename Context>
void StackGradKernel(const Context& dev_ctx,
                     const phi::DenseTensor& dy,
                     int axis,
                     std::vector<phi::DenseTensor*> dx) {
  PADDLE_GCU_KERNEL_TRACE("stack_grad");
  if (LaunchAOTKernel()) {
    THROW_AOT_UNIMPLEMENTED();
  } else {  // kernel impl base on JIT
    TensorNameMap input_names;
    TensorValueMap inputs;

    input_names[GradVarName("Y")] = {"dy"};
    inputs[GradVarName("Y")] = {const_cast<DenseTensor*>(&dy)};

    TensorNameMap output_names;
    TensorValueMap outputs;

    std::vector<std::string> names;
    names.reserve(dx.size());
    std::vector<phi::DenseTensor*> values;
    values.reserve(dx.size());
    for (size_t i = 0; i < dx.size(); ++i) {
      dev_ctx.template Alloc<T>(dx[i]);
      names.emplace_back(std::string("x_grad_") + std::to_string(i));
      values.emplace_back(dx[i]);
    }
    output_names[GradVarName("X")] = names;
    outputs[GradVarName("X")] = values;

    GcuAttributeMap attrs;
    attrs["axis"] = axis;

    GcuRunner(input_names,
              inputs,
              output_names,
              outputs,
              attrs,
              "stack_grad",
              dev_ctx);
  }
}

template <typename T, typename Context>
void UnStackKernel(const Context& dev_ctx,
                   const phi::DenseTensor& x,
                   int axis,
                   int num,
                   std::vector<phi::DenseTensor*> outs) {
  PADDLE_GCU_KERNEL_TRACE("unstack");
  if (LaunchAOTKernel()) {
    for (auto y : outs) {
      dev_ctx.template Alloc<T>(y);
    }
    LAUNCH_TOPSCLOP(unstack, dev_ctx, &outs, x, axis)
  } else {  // kernel impl base on JIT
    THROW_JIT_UNIMPLEMENTED();
  }
}

template <typename T, typename Context>
void UnbindKernel(const Context& dev_ctx,
                  const phi::DenseTensor& x,
                  int axis,
                  std::vector<phi::DenseTensor*> outs) {
  PADDLE_GCU_KERNEL_TRACE("unbind");
  if (LaunchAOTKernel()) {
    for (auto y : outs) {
      dev_ctx.template Alloc<T>(y);
    }
    LAUNCH_TOPSCLOP(unstack, dev_ctx, &outs, x, axis)
  } else {  // kernel impl base on JIT
    THROW_JIT_UNIMPLEMENTED();
  }
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(stack,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::StackKernel,
                          int,
                          int64_t,
                          double,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(stack_grad,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::StackGradKernel,
                          int,
                          int64_t,
                          float,
                          double) {}

PD_REGISTER_PLUGIN_KERNEL(unstack,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::UnStackKernel,
                          int,
                          int64_t,
                          double,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(unbind,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::UnbindKernel,
                          int,
                          int64_t,
                          double,
                          float,
                          phi::dtype::float16) {}
