// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

const int kIgnoreIndex = -100;

void CheckAttrs(bool normalize, int ignore_index) {
  // Add this check is is due to npu SigmoidCrossEntropyWithLogits
  // and SigmoidCrossEntropyWithLogitsGrad does't supoort
  // attr normalize and ignore_index
  PADDLE_ENFORCE_EQ(normalize,
                    false,
                    phi::errors::InvalidArgument(
                        "attr normalize must be false, but got true"));
  PADDLE_ENFORCE_EQ(ignore_index,
                    kIgnoreIndex,
                    phi::errors::InvalidArgument(
                        "attr ignore_index must be default %d, but got %d",
                        kIgnoreIndex,
                        ignore_index));
}

template <typename T, typename Context>
void SigmoidCrossEntropyWithLogitsKernel(
    const Context& dev_ctx,
    const phi::DenseTensor& x,
    const phi::DenseTensor& label,
    const paddle::optional<phi::DenseTensor>& pos_weight,
    bool normalize,
    int ignore_index,
    phi::DenseTensor* out) {
  CheckAttrs(normalize, ignore_index);

  std::string reduction = "none";
  phi::DenseTensor weight_tensor;
  phi::DenseTensor pos_weight_tensor;
  phi::DenseTensorMeta weight_tensor_meta = {phi::DataType::FLOAT32, x.dims()};
  weight_tensor.set_meta(weight_tensor_meta);
  FillNpuTensorWithConstant<float>(&weight_tensor, dev_ctx, 1.0);
  weight_tensor.Resize(x.dims());

  if (pos_weight.get_ptr() == nullptr) {
    pos_weight_tensor.set_meta(weight_tensor_meta);
    FillNpuTensorWithConstant<float>(&pos_weight_tensor, dev_ctx, 1.0);
    pos_weight_tensor.Resize(x.dims());
  } else {
    pos_weight_tensor = *pos_weight.get_ptr();
  }

  dev_ctx.template Alloc<T>(out);
  auto stream = dev_ctx.stream();

  const auto& runner = NpuOpRunner("SigmoidCrossEntropyWithLogitsV2",
                                   {x, label, weight_tensor, pos_weight_tensor},
                                   {*out},
                                   {{"reduction", reduction}});
  runner.Run(stream);
}

template <typename T, typename Context>
void SigmoidCrossEntropyWithLogitsGradKernel(
    const Context& dev_ctx,
    const phi::DenseTensor& x,
    const phi::DenseTensor& label,
    const paddle::optional<phi::DenseTensor>& pos_weight,
    const phi::DenseTensor& dout,
    bool normalize,
    int ignore_index,
    phi::DenseTensor* dx) {
  CheckAttrs(normalize, ignore_index);

  std::string reduction = "none";
  phi::DenseTensor weight_tensor;
  phi::DenseTensor pos_weight_tensor;
  phi::DenseTensorMeta weight_tensor_meta = {phi::DataType::FLOAT32, x.dims()};
  weight_tensor.set_meta(weight_tensor_meta);
  FillNpuTensorWithConstant<float>(&weight_tensor, dev_ctx, 1.0);
  weight_tensor.Resize(x.dims());

  if (pos_weight.get_ptr() == nullptr) {
    pos_weight_tensor.set_meta(weight_tensor_meta);
    FillNpuTensorWithConstant<float>(&pos_weight_tensor, dev_ctx, 1.0);
    pos_weight_tensor.Resize(x.dims());
  } else {
    pos_weight_tensor = *pos_weight.get_ptr();
  }

  dev_ctx.template Alloc<T>(dx);
  auto stream = dev_ctx.stream();

  const auto& runner =
      NpuOpRunner("SigmoidCrossEntropyWithLogitsGradV2",
                  {x, label, dout, weight_tensor, pos_weight_tensor},
                  {*dx},
                  {{"reduction", reduction}});
  runner.Run(stream);
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(sigmoid_cross_entropy_with_logits,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::SigmoidCrossEntropyWithLogitsKernel,
                          float,
                          double) {}

PD_REGISTER_PLUGIN_KERNEL(
    sigmoid_cross_entropy_with_logits_grad,
    npu,
    ALL_LAYOUT,
    custom_kernel::SigmoidCrossEntropyWithLogitsGradKernel,
    float,
    double) {}
