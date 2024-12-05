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

template <typename T, typename Context>
void ScaleKernel(const Context& dev_ctx,
                 const phi::DenseTensor& x,
                 const phi::Scalar& in_scale,
                 float bias,
                 bool bias_after_scale,
                 phi::DenseTensor* out);

template <typename T, typename Context>
void AddKernel(const Context& dev_ctx,
               const phi::DenseTensor& x,
               const phi::DenseTensor& y,
               phi::DenseTensor* out);

template <typename T, typename Context>
void MomentumKernel(const Context& dev_ctx,
                    const phi::DenseTensor& param,
                    const phi::DenseTensor& grad,
                    const phi::DenseTensor& velocity,
                    const phi::DenseTensor& learning_rate,
                    const paddle::optional<phi::DenseTensor>& master_param,
                    float mu_f,
                    bool use_nesterov,
                    const std::string& regularization_method,
                    float regularization_coeff,
                    bool multi_precision,
                    float rescale_grad,
                    phi::DenseTensor* param_out,
                    phi::DenseTensor* velocity_out,
                    phi::DenseTensor* master_param_out) {
  auto mu = static_cast<T>(mu_f);

  dev_ctx.template Alloc<T>(param_out);
  dev_ctx.template Alloc<T>(velocity_out);

  phi::DenseTensor mu_tensor;
  mu_tensor.Resize({1});
  dev_ctx.template Alloc<T>(&mu_tensor);
  FillNpuTensorWithConstant<T>(&mu_tensor, dev_ctx, mu);

  phi::DenseTensor regularized_grad;
  if (regularization_method == "l2_decay") {
    regularized_grad.Resize(grad.dims());
    dev_ctx.template Alloc<T>(&regularized_grad);

    phi::Scalar in_scale = regularization_coeff;
    custom_kernel::ScaleKernel<T, Context>(
        dev_ctx, param, in_scale, 0.0f, false, &regularized_grad);

    custom_kernel::AddKernel<T, Context>(
        dev_ctx, grad, regularized_grad, &regularized_grad);
  } else {
    regularized_grad = grad;
  }
  TensorCopy(dev_ctx, param, false, param_out);
  TensorCopy(dev_ctx, velocity, false, velocity_out);
  // NOTE: ApplyMomentum will change the input
  const auto& runner = NpuOpRunner(
      "ApplyMomentum",
      {*param_out, *velocity_out, learning_rate, regularized_grad, mu_tensor},
      {*param_out},
      {{"use_nesterov", use_nesterov}});
  runner.Run(dev_ctx.stream());
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(momentum,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::MomentumKernel,
                          phi::dtype::float16,
                          float,
                          double) {}
