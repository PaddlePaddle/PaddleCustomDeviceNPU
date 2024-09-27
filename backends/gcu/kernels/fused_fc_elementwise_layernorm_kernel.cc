// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
#include "custom_op/custom_op_common.h"
#include "kernels/funcs/gcu_kernel_funcs.h"

namespace custom_kernel {
template <typename T, typename Context>
extern void FCKernel(const Context& dev_ctx,
                     const phi::DenseTensor& input,
                     const phi::DenseTensor& w,
                     const paddle::optional<phi::DenseTensor>& bias,
                     const int in_num_col_dims,
                     const std::string& activation_type,
                     const bool padding_weights,
                     phi::DenseTensor* out);

template <typename T, typename Context>
extern void AddKernel(const Context& dev_ctx,
                      const phi::DenseTensor& x,
                      const phi::DenseTensor& y,
                      phi::DenseTensor* out);

template <typename T, typename Context>
void LayerNormKernel(const Context& dev_ctx,
                     const phi::DenseTensor& x,
                     const paddle::optional<phi::DenseTensor>& scale_opt,
                     const paddle::optional<phi::DenseTensor>& bias_opt,
                     float epsilon,
                     int begin_norm_axis,
                     phi::DenseTensor* out,
                     phi::DenseTensor* mean,
                     phi::DenseTensor* variance);

template <typename T, typename Context>
void FusedFCElementwiseLayerNormKernel(
    const Context& dev_ctx,
    const phi::DenseTensor& x,
    const phi::DenseTensor& w,
    const phi::DenseTensor& y,
    const paddle::optional<phi::DenseTensor>& bias0,
    const paddle::optional<phi::DenseTensor>& scale,
    const paddle::optional<phi::DenseTensor>& bias1,
    const int x_num_col_dims,
    const std::string& activation_type,
    const float epsilon,
    const int begin_norm_axis,
    phi::DenseTensor* out,
    phi::DenseTensor* mean,
    phi::DenseTensor* variance) {
  PADDLE_GCU_KERNEL_TRACE("fused_fc_elementwise_layernorm");

  if (LaunchAOTKernel()) {
    phi::DenseTensor fc_out = TensorEmpty(dev_ctx, out->meta());
    custom_kernel::FCKernel<T, Context>(
        dev_ctx, x, w, bias0, x_num_col_dims, activation_type, false, &fc_out);
    if (mean != nullptr && variance != nullptr) {
      phi::DenseTensor add_out = TensorEmpty(dev_ctx, out->meta());
      custom_kernel::AddKernel<T, Context>(dev_ctx, y, fc_out, &add_out);
      custom_kernel::LayerNormKernel<T, Context>(dev_ctx,
                                                 add_out,
                                                 scale,
                                                 bias1,
                                                 epsilon,
                                                 begin_norm_axis,
                                                 out,
                                                 mean,
                                                 variance);
    } else {
      auto fc = custom_op_common::CreateTensorFromDenseTensor(fc_out);
      auto residual = custom_op_common::CreateTensorFromDenseTensor(y);
      auto res_add = paddle::experimental::add(residual, fc);
      auto norm_scale =
          custom_op_common::CreateOptionalTensorFromOptionalDense(scale);
      auto norm_bias1 =
          custom_op_common::CreateOptionalTensorFromOptionalDense(bias1);
      auto norm = paddle::experimental::layer_norm(
          res_add, norm_scale, norm_bias1, epsilon, begin_norm_axis);
      *out = custom_op_common::CreateDenseTensorFromTernsor(norm);
    }

  } else {  // kernel impl base on JIT
    THROW_JIT_UNIMPLEMENTED();
  }
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(fused_fc_elementwise_layernorm,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::FusedFCElementwiseLayerNormKernel,
                          float,
                          phi::dtype::float16) {}
