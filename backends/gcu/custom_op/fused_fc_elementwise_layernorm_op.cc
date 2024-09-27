/* Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "custom_op/custom_op_common.h"

extern std::vector<std::vector<int64_t>> CustomLinearInferShape(
    const std::vector<int64_t>& x_shape,
    const std::vector<int64_t>& weight_shape,
    const std::vector<int64_t>& bias_shape);

extern std::vector<paddle::Tensor> CustomLinear(const paddle::Tensor& x,
                                                const paddle::Tensor& weight,
                                                const paddle::Tensor& bias);

std::vector<std::vector<int64_t>> FusedFcElementwiseLayerNormInferShape(
    const std::vector<int64_t>& x_shape,
    const std::vector<int64_t>& y_shape,
    const std::vector<int64_t>& weight_shape,
    const std::vector<int64_t>& scale_shape,
    const std::vector<int64_t>& bias0_shape,
    const std::vector<int64_t>& bias1_shape,
    float epsilon,
    int begin_norm_axis) {
  auto fc_shape = CustomLinearInferShape(x_shape, weight_shape, bias0_shape);
  return fc_shape;
}

std::vector<paddle::DataType> FusedFcElementwiseLayerNormInferDtype(
    const paddle::DataType& x_dtype,
    const paddle::DataType& y_dtype,
    const paddle::DataType& weight_dtype,
    const paddle::DataType& scale_dtype,
    const paddle::DataType& bias0_dtype,
    const paddle::DataType& bias1_dtype,
    float epsilon,
    int begin_norm_axis) {
  return {x_dtype};
}

// The linear implemented here must be passed in bias
std::vector<paddle::Tensor> FusedFcElementwiseLayerNorm(
    const paddle::Tensor& x,
    const paddle::Tensor& y,
    const paddle::Tensor& weight,
    const paddle::Tensor& scale,
    const paddle::Tensor& bias0,
    const paddle::Tensor& bias1,
    float epsilon,
    int begin_norm_axis) {
  PADDLE_GCU_KERNEL_TRACE("fused_fc_elementwise_layernorm");
  VLOG(6) << "[CUSTOM_KERNEL] Custom Operator: fused_fc_elementwise_layernorm";
  auto fc_out = CustomLinear(x, weight, bias0);
  auto res_add = paddle::experimental::add(y, fc_out[0]);
  auto norm = paddle::experimental::layer_norm(
      res_add, scale, bias1, epsilon, begin_norm_axis);

  return {norm};
}

// PD_BUILD_OP(fused_fc_elementwise_layernorm)
//     .Inputs({"X", "Y", "W", "Scale", "Bias0", "Bias1"})
//     .Outputs({"Out"})
//     .Attrs({"epsilon: float", "begin_norm_axis: int"})
//     .SetKernelFn(PD_KERNEL(FusedFcElementwiseLayerNorm))
//     .SetInferShapeFn(PD_INFER_SHAPE(FusedFcElementwiseLayerNormInferShape))
//     .SetInferDtypeFn(PD_INFER_DTYPE(FusedFcElementwiseLayerNormInferDtype));
