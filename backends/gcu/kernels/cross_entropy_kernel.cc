/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "common/gcu_op_runner.h"
#include "kernels/funcs/gcu_kernel_funcs.h"

namespace custom_kernel {

template <typename T, typename Context>
void CrossEntropyWithSoftmaxKernel(const Context& dev_ctx,
                                   const phi::DenseTensor& logits,
                                   const phi::DenseTensor& labels,
                                   bool soft_label,
                                   bool use_softmax,
                                   bool numeric_stable_mode,
                                   int ignore_index,
                                   int axis,
                                   phi::DenseTensor* softmax,
                                   phi::DenseTensor* loss) {
  PADDLE_GCU_KERNEL_TRACE("cross_entropy_with_softmax");
  dev_ctx.template Alloc<T>(loss);
  dev_ctx.template Alloc<T>(softmax);

  if (LaunchAOTKernel()) {
    THROW_AOT_UNIMPLEMENTED();
  } else {  // kernel impl base on JIT
    TensorNameMap input_names;
    input_names["Logits"] = {"logits"};
    input_names["Label"] = {"labels"};

    TensorValueMap inputs;
    inputs["Logits"] = {const_cast<DenseTensor*>(&logits)};
    inputs["Label"] = {const_cast<DenseTensor*>(&labels)};

    TensorNameMap output_names;
    output_names["Softmax"] = {"softmax"};
    output_names["Loss"] = {"loss"};

    TensorValueMap outputs;
    outputs["Softmax"] = {softmax};
    outputs["Loss"] = {loss};

    GcuAttributeMap attrs;
    attrs["soft_label"] = soft_label;
    attrs["use_softmax"] = use_softmax;
    attrs["numeric_stable_mode"] = numeric_stable_mode;
    attrs["ignore_index"] = ignore_index;
    attrs["axis"] = axis;

    GcuRunner(input_names,
              inputs,
              output_names,
              outputs,
              attrs,
              "softmax_with_cross_entropy",
              dev_ctx);
  }
}

template <typename T, typename Context>
void CrossEntropyWithSoftmaxGradKernel(const Context& dev_ctx,
                                       const phi::DenseTensor& labels,
                                       const phi::DenseTensor& softmax,
                                       const phi::DenseTensor& loss_grad,
                                       bool soft_label,
                                       bool use_softmax,
                                       bool numeric_stable_mode,
                                       int ignore_index,
                                       int axis,
                                       phi::DenseTensor* logits_grad) {
  PADDLE_GCU_KERNEL_TRACE("cross_entropy_with_softmax_grad");
  dev_ctx.template Alloc<T>(logits_grad);

  if (LaunchAOTKernel()) {
    THROW_AOT_UNIMPLEMENTED();
  } else {  // kernel impl base on JIT
    TensorNameMap input_names;
    input_names["Label"] = {"labels"};
    input_names["Softmax"] = {"softmax"};
    input_names[GradVarName("Loss")] = {"loss_grad"};

    TensorValueMap inputs;
    inputs["Label"] = {const_cast<DenseTensor*>(&labels)};
    inputs["Softmax"] = {const_cast<DenseTensor*>(&softmax)};
    inputs[GradVarName("Loss")] = {const_cast<DenseTensor*>(&loss_grad)};

    TensorNameMap output_names;
    output_names[GradVarName("Logits")] = {"logits_grad"};

    TensorValueMap outputs;
    outputs[GradVarName("Logits")] = {logits_grad};

    GcuAttributeMap attrs;
    attrs["soft_label"] = soft_label;
    attrs["use_softmax"] = use_softmax;
    attrs["numeric_stable_mode"] = numeric_stable_mode;
    attrs["ignore_index"] = ignore_index;
    attrs["axis"] = axis;

    GcuRunner(input_names,
              inputs,
              output_names,
              outputs,
              attrs,
              "softmax_with_cross_entropy_grad",
              dev_ctx);
  }
}
}  // namespace custom_kernel

// PD_REGISTER_PLUGIN_KERNEL(cross_entropy_with_softmax,
//                           gcu,
//                           ALL_LAYOUT,
//                           custom_kernel::CrossEntropyWithSoftmaxKernel,
//                           phi::dtype::float16,
//                           float) {}

// PD_REGISTER_PLUGIN_KERNEL(cross_entropy_with_softmax_grad,
//                           gcu,
//                           ALL_LAYOUT,
//                           custom_kernel::CrossEntropyWithSoftmaxGradKernel,
//                           phi::dtype::float16,
//                           float) {}
