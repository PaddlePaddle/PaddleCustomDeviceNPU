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

namespace custom_kernel {
template <typename T, typename Context>
extern void Conv2dBiasKernel(const Context& dev_ctx,
                             const DenseTensor& input,
                             const DenseTensor& filter,
                             const paddle::optional<phi::DenseTensor>& bias,
                             const std::vector<int>& strides,
                             const std::vector<int>& paddings,
                             const std::string& padding_algorithm,
                             const std::vector<int>& dilations,
                             int groups,
                             const std::string& data_format,
                             DenseTensor* out);

template <typename T, typename Context>
extern void FusedConv2dAddActKernel(
    const Context& dev_ctx,
    const phi::DenseTensor& input,
    const phi::DenseTensor& filter,
    const phi::DenseTensor& bias,
    const paddle::optional<phi::DenseTensor>& residual,
    const std::vector<int>& strides,
    const std::vector<int>& paddings,
    const std::string& padding_algorithm,
    const std::vector<int>& dilations,
    int groups,
    const std::string& data_format,
    const std::string& activation,
    const std::vector<int>& split_channels,
    bool exhaustive_search,
    int workspace_size_MB,
    float fuse_alpha,
    phi::DenseTensor* output,
    std::vector<phi::DenseTensor*> outputs);
}  // namespace custom_kernel

std::vector<std::vector<int64_t>> FusedConv2dAddInferShape(
    const std::vector<int64_t>& x_shape,
    const std::vector<int64_t>& filter_shape,
    const std::vector<int64_t>& bias_shape,
    const std::vector<int64_t>& residual_shape,
    const std::vector<int>& strides,
    const std::vector<int>& paddings,
    const std::string& padding_algorithm,
    const std::vector<int>& dilations,
    int groups,
    const std::string& data_format,
    const std::string& activation,
    const std::vector<int>& split_channels,
    bool exhaustive_search,
    int workspace_size_MB,
    float fuse_alpha) {
  return {{-1}};
}

std::vector<paddle::DataType> FusedConv2dAddInferDtype(
    const paddle::DataType& x_dtype,
    const paddle::DataType& filter_dtype,
    const paddle::DataType& bias_dtype,
    const paddle::DataType& residual_dtype,
    const std::vector<int>& strides,
    const std::vector<int>& paddings,
    const std::string& padding_algorithm,
    const std::vector<int>& dilations,
    int groups,
    const std::string& data_format,
    const std::string& activation,
    const std::vector<int>& split_channels,
    bool exhaustive_search,
    int workspace_size_MB,
    float fuse_alpha) {
  return {x_dtype};
}

template <typename T>
std::vector<paddle::Tensor> GCUFusedConv2dAddTemplate(
    const paddle::Tensor& x,
    const paddle::Tensor& filter,
    const paddle::Tensor& bias,
    const paddle::Tensor& residual,
    const std::vector<int>& strides,
    const std::vector<int>& paddings,
    const std::string& padding_algorithm,
    const std::vector<int>& dilations,
    int groups,
    const std::string& data_format,
    const std::string& activation,
    const std::vector<int>& split_channels,
    bool exhaustive_search,
    int workspace_size_MB,
    float fuse_alpha) {
  PADDLE_GCU_KERNEL_TRACE("fused_conv2d_add");
  VLOG(6) << "[CUSTOM_KERNEL] Custom Operator: fused_conv2d_add";
  auto dev_ctx = static_cast<const phi::CustomContext*>(
      paddle::experimental::DeviceContextPool::Instance().Get(x.place()));

  auto x_tensor = static_cast<const phi::DenseTensor*>(x.impl().get());
  auto filter_tensor =
      static_cast<const phi::DenseTensor*>(filter.impl().get());
  auto bias_tensor = static_cast<const phi::DenseTensor*>(bias.impl().get());
  auto residual_tensor =
      static_cast<const phi::DenseTensor*>(residual.impl().get());

  std::shared_ptr<phi::DenseTensor> output =
      std::make_shared<phi::DenseTensor>();
  phi::MetaTensor meta_out(*output);
  phi::ConvInferMeta(*x_tensor,
                     *filter_tensor,
                     strides,
                     paddings,
                     padding_algorithm,
                     dilations,
                     groups,
                     data_format,
                     &meta_out);
  output->Resize(meta_out.dims());
  dev_ctx->Alloc(output.get(), x.dtype());

  // conv2d + residual
  if (bias.dims().size() == x.dims().size()) {
    auto meta =
        phi::DenseTensorMeta(x.dtype(), phi::make_ddim({filter.dims().at(0)}));
    auto bias_zero = custom_kernel::TensorZeros(*dev_ctx, meta);
    auto residual_bcast = *residual_tensor;
    if (residual_bcast.dims() != output->dims()) {
      residual_bcast = custom_kernel::Broadcast(
          *dev_ctx, residual_bcast, phi::vectorize(output->dims()));
    }
    custom_kernel::FusedConv2dAddActKernel<T, phi::CustomContext>(
        *dev_ctx,
        *x_tensor,
        *filter_tensor,
        bias_zero,
        paddle::make_optional<phi::DenseTensor>(residual_bcast),
        strides,
        paddings,
        padding_algorithm,
        dilations,
        groups,
        data_format,
        activation,
        split_channels,
        exhaustive_search,
        workspace_size_MB,
        fuse_alpha,
        output.get(),
        {});
  } else if (bias.dims().size() == 1) {  // conv2d + bias
    custom_kernel::Conv2dBiasKernel<T, phi::CustomContext>(*dev_ctx,
                                                           *x_tensor,
                                                           *filter_tensor,
                                                           *bias_tensor,
                                                           strides,
                                                           paddings,
                                                           padding_algorithm,
                                                           dilations,
                                                           groups,
                                                           data_format,
                                                           output.get());
  } else {
    PADDLE_THROW(phi::errors::Unimplemented(
        "Unsupported fused type, x dims: %s, filter dims: %s, bias dims: %s, "
        "residual dims: %s.",
        x.dims().to_str().c_str(),
        filter.dims().to_str().c_str(),
        bias.dims().to_str().c_str(),
        residual.dims().to_str().c_str()));
  }

  return {paddle::Tensor(output)};
}

std::vector<paddle::Tensor> FusedConv2dAdd(
    const paddle::Tensor& x,
    const paddle::Tensor& filter,
    const paddle::Tensor& bias,
    const paddle::Tensor& residual,
    const std::vector<int>& strides,
    const std::vector<int>& paddings,
    const std::string& padding_algorithm,
    const std::vector<int>& dilations,
    int groups,
    const std::string& data_format,
    const std::string& activation,
    const std::vector<int>& split_channels,
    bool exhaustive_search,
    int workspace_size_MB,
    float fuse_alpha) {
  if (x.dtype() == phi::DataType::FLOAT16) {
    return GCUFusedConv2dAddTemplate<phi::dtype::float16>(x,
                                                          filter,
                                                          bias,
                                                          residual,
                                                          strides,
                                                          paddings,
                                                          padding_algorithm,
                                                          dilations,
                                                          groups,
                                                          data_format,
                                                          activation,
                                                          split_channels,
                                                          exhaustive_search,
                                                          workspace_size_MB,
                                                          fuse_alpha);
  } else if (x.dtype() == phi::DataType::FLOAT32) {
    return GCUFusedConv2dAddTemplate<float>(x,
                                            filter,
                                            bias,
                                            residual,
                                            strides,
                                            paddings,
                                            padding_algorithm,
                                            dilations,
                                            groups,
                                            data_format,
                                            activation,
                                            split_channels,
                                            exhaustive_search,
                                            workspace_size_MB,
                                            fuse_alpha);
  } else {
    PADDLE_THROW(phi::errors::Unimplemented(
        "Unsupported data type: %s", phi::DataTypeToString(x.dtype()).c_str()));
  }
}

PD_BUILD_OP(fused_conv2d_add)
    .Inputs({"Input", "Filter", "Bias", "ResidualData"})
    .Outputs({"Output"})
    .Attrs({"strides: std::vector<int>",
            "paddings: std::vector<int>",
            "padding_algorithm: std::string",
            "dilations: std::vector<int>",
            "groups: int",
            "data_format: std::string",
            "activation: std::string",
            "split_channels: std::vector<int>",
            "exhaustive_search: bool",
            "workspace_size_MB: int",
            "fuse_alpha: float"})
    .SetKernelFn(PD_KERNEL(FusedConv2dAdd))
    .SetInferShapeFn(PD_INFER_SHAPE(FusedConv2dAddInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(FusedConv2dAddInferDtype));
