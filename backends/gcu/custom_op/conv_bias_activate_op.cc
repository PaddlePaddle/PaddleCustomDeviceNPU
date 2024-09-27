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

#include <iostream>
#include <vector>

#include "common/gcu_op_runner.h"
#include "kernels/funcs/gcu_kernel_funcs.h"
// #include "common/phi_funcs.h"
#include "paddle/extension.h"
#include "paddle/phi/backends/custom/custom_context.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/kernels/cpu/conv_util.h"

template <typename T = int32_t>
static void UpdatePaddingAndDilation(std::vector<T>& paddings,  // NOLINT
                                     std::vector<T>& dilation,  // NOLINT
                                     const std::string& padding_algorithm,
                                     const phi::DDim& data_dims,
                                     const std::vector<T>& strides,
                                     const std::vector<T>& ksize) {
  // set padding size == data_dims.size() * 2
  auto data_shape = phi::vectorize<T>(data_dims);
  if (static_cast<int>(paddings.size()) == data_dims.size()) {
    for (int i = 0; i < data_dims.size(); ++i) {
      T copy_pad = *(paddings.begin() + 2 * i);
      paddings.insert(paddings.begin() + 2 * i + 1, copy_pad);
    }
  } else {
    PADDLE_ENFORCE_EQ(
        data_dims.size() * 2,
        paddings.size(),
        phi::errors::InvalidArgument(
            "Attribute padding's size should be the same or twice as the "
            "input's dimension. "
            "But received: padding's size is %d, padding is [%s]; input's "
            "dimension is %d, input's shape is [%s].",
            paddings.size(),
            phi::make_ddim(paddings),
            data_dims.size(),
            data_dims));
  }

  // when padding_algorithm is "VALID" or "SAME"
  if (padding_algorithm == "SAME") {
    for (int i = 0; i < data_dims.size(); ++i) {
      T out_size = (data_dims[i] + strides[i] - 1) / strides[i];
      T pad_sum =
          std::max((out_size - 1) * strides[i] + ksize[i] - data_shape[i],
                   static_cast<T>(0));
      T pad_0 = pad_sum / 2;
      T pad_1 = pad_sum - pad_0;
      *(paddings.begin() + i * 2) = pad_0;
      *(paddings.begin() + i * 2 + 1) = pad_1;

      // dilation
      *(dilation.begin() + i) = 1;
    }

  } else if (padding_algorithm == "VALID") {
    for (auto it = paddings.begin(); it != paddings.end(); it++) {
      *it = 0;
    }
  }
}

std::vector<std::vector<int64_t>> ConvBiasOpInferShape(
    const std::vector<int64_t>& x_shape,
    const std::vector<int64_t>& filter_shape,
    const std::vector<int64_t>& Y,
    std::string data_format,
    int64_t groups,
    std::string padding_algorithm,
    std::vector<int> strides,
    std::vector<int> paddings,
    std::vector<int> dilations) {
  VLOG(6) << "---------- here is ConvBiasOpInferShape ------";
  auto in_dims = phi::make_ddim(x_shape);
  auto filter_dims = phi::make_ddim(filter_shape);
  int dilation_size = dilations.size();
  for (int i = 0; i < dilation_size; ++i) {
    PADDLE_ENFORCE_GT(
        dilations[i],
        0,
        phi::errors::InvalidArgument(
            "The dilation of Op(Conv) should be larget than 0, but received "
            "dilation is %d.",
            dilations[i]));
  }
  const bool channel_last = (data_format == "NHWC" || data_format == "NDHWC");

  for (int i = 0; i < 2; ++i) {
    PADDLE_ENFORCE_NE(in_dims[i],
                      0,
                      phi::errors::InvalidArgument(
                          "The size of Op(Conv) inputs should not be 0."));
  }

  PADDLE_ENFORCE_EQ(
      in_dims.size() == 4 || in_dims.size() == 5,
      true,
      phi::errors::InvalidArgument(
          "The input of Op(Conv) should be a 4-D or 5-D Tensor. But "
          "received: input's dimension is %u, input's shape is [%s].",
          in_dims.size(),
          in_dims));

  PADDLE_ENFORCE_EQ(
      in_dims.size(),
      filter_dims.size(),
      phi::errors::InvalidArgument(
          "The input's dimension and filter's dimension of "
          "Op(Conv) should be equal. But received: the input's shape is [%s], "
          "the input's dimension is %d; the filter's shape is [%s],  "
          "the filter's dimension is %d.",
          in_dims,
          in_dims.size(),
          filter_dims,
          filter_dims.size()));

  int stride_size = strides.size();
  for (int i = 0; i < stride_size; ++i) {
    PADDLE_ENFORCE_GT(
        strides[i],
        0,
        phi::errors::InvalidArgument(
            "The stride of Op(Conv) should be larget than 0, but received "
            "stride is %d.",
            strides[i]));
  }

  int in_sub_stride_size = in_dims.size() - stride_size;
  PADDLE_ENFORCE_EQ(
      in_dims.size(),
      strides.size() + 2U,
      phi::errors::InvalidArgument(
          "The difference of input's dimension and Attr(strides)'s "
          "length must be euqal to 2 for Op(Conv). "
          "But received: input's dimension is %d, input's shape is [%s]; "
          "Attr(stride)'s length is %d, Attr(stride) is [%s]; "
          "difference of input's dimention and Attr(strides)'s length = %u.",
          in_dims.size(),
          in_dims,
          strides.size(),
          phi::make_ddim(strides),
          in_sub_stride_size));

  const auto input_channels =
      channel_last ? in_dims[in_dims.size() - 1] : in_dims[1];

  PADDLE_ENFORCE_EQ(
      input_channels,
      filter_dims[1] * groups,
      phi::errors::InvalidArgument(
          "The number of input's channels should be equal to filter's channels "
          "* groups for Op(Conv). But received: the input's channels is %d, "
          "the input's shape is [%s]; the filter's channels is %d, the "
          "filter's shape is [%s]; the groups is %d, the data_format is %s. "
          "The error may come from wrong data_format setting.",
          input_channels,
          in_dims,
          filter_dims[1],
          filter_dims,
          groups,
          data_format));
  PADDLE_ENFORCE_EQ(
      filter_dims[0] % groups,
      0,
      phi::errors::InvalidArgument(
          "The number of output's channels (filter's first dimension) of "
          "Op(Conv) should be divided by groups. But received: "
          "the output channels is %d, the filter's shape is [%s], "
          "the groups is %d.",
          filter_dims[0],
          filter_dims,
          groups));

  phi::DDim in_data_dims;
  if (channel_last) {
    in_data_dims = phi::slice_ddim(in_dims, 1, in_dims.size() - 1);
  } else {
    in_data_dims = phi::slice_ddim(in_dims, 2, in_dims.size());
  }

  phi::DDim filter_data_dims =
      phi::slice_ddim(filter_dims, 2, filter_dims.size());

  std::vector<int> ksize = phi::vectorize<int>(filter_data_dims);
  UpdatePaddingAndDilation<int>(
      paddings, dilations, padding_algorithm, in_data_dims, strides, ksize);

  std::vector<int64_t> output_shape({in_dims[0]});
  if (!channel_last) {
    output_shape.push_back(filter_dims[0]);
  }
  for (int i = 0; i < in_data_dims.size(); ++i) {
    if (in_data_dims[i] <= 0 || filter_dims[i + 2] <= 0) {
      output_shape.push_back(-1);
    } else {
      const int dkernel = dilations[i] * (filter_data_dims[i] - 1) + 1;
      int output_size =
          (in_data_dims[i] + paddings[2 * i] + paddings[2 * i + 1] - dkernel) /
              strides[i] +
          1;
      output_shape.push_back(output_size);
    }
  }
  if (channel_last) {
    output_shape.push_back(filter_dims[0]);
  }
  return {output_shape};
}

std::vector<paddle::Tensor> ConvBiasOp(const paddle::Tensor& x,
                                       const paddle::Tensor& filter,
                                       const paddle::Tensor& y,
                                       std::string data_format,
                                       int64_t groups,
                                       std::string padding_algorithm,
                                       std::vector<int> strides,
                                       std::vector<int> paddings,
                                       std::vector<int> dilations,
                                       int axis) {
  auto dev_ctx = static_cast<const phi::CustomContext*>(
      paddle::experimental::DeviceContextPool::Instance().Get(x.place()));

  auto x_tensor = static_cast<const phi::DenseTensor*>(x.impl().get());
  auto filter_tensor =
      static_cast<const phi::DenseTensor*>(filter.impl().get());
  auto y_tensor = static_cast<const phi::DenseTensor*>(y.impl().get());
  // Infershape
  auto out_shapes = ConvBiasOpInferShape(x.shape(),
                                         filter.shape(),
                                         y.shape(),
                                         data_format,
                                         groups,
                                         padding_algorithm,
                                         strides,
                                         paddings,
                                         dilations);
  paddle::Tensor out = paddle::empty(out_shapes[0], x.dtype(), x.place());
  if (custom_kernel::LaunchAOTKernel()) {
    THROW_AOT_UNIMPLEMENTED();
  } else {
    custom_kernel::TensorNameMap input_names;
    custom_kernel::TensorValueMap inputs;
    input_names["X"] = {"x"};
    input_names["Filter"] = {"filter"};
    input_names["Y"] = {"y"};
    inputs["X"] = {const_cast<phi::DenseTensor*>(x_tensor)};
    inputs["Filter"] = {const_cast<phi::DenseTensor*>(filter_tensor)};
    inputs["Y"] = {const_cast<phi::DenseTensor*>(y_tensor)};

    custom_kernel::GcuAttributeMap attrs;
    attrs["groups"] = groups;
    attrs["data_format"] = data_format;
    attrs["padding_algorithm"] = padding_algorithm;
    attrs["strides"] = strides;
    attrs["paddings"] = paddings;
    attrs["dilations"] = dilations;
    attrs["axis"] = axis;

    custom_kernel::TensorNameMap output_names;
    custom_kernel::TensorValueMap outputs;
    output_names["Out"] = {"Out"};
    outputs["Out"] = {static_cast<phi::DenseTensor*>(out.impl().get())};

    custom_kernel::GcuRunner(input_names,
                             inputs,
                             output_names,
                             outputs,
                             attrs,
                             "conv_bias",
                             *dev_ctx);
  }
  return {out};
}

std::vector<paddle::Tensor> ConvBiasReluOp(const paddle::Tensor& x,
                                           const paddle::Tensor& filter,
                                           const paddle::Tensor& y,
                                           std::string data_format,
                                           int64_t groups,
                                           std::string padding_algorithm,
                                           std::vector<int> strides,
                                           std::vector<int> paddings,
                                           std::vector<int> dilations,
                                           int axis) {
  auto dev_ctx = static_cast<const phi::CustomContext*>(
      paddle::experimental::DeviceContextPool::Instance().Get(x.place()));

  auto x_tensor = static_cast<const phi::DenseTensor*>(x.impl().get());
  auto filter_tensor =
      static_cast<const phi::DenseTensor*>(filter.impl().get());
  auto y_tensor = static_cast<const phi::DenseTensor*>(y.impl().get());
  // Infershape
  auto out_shapes = ConvBiasOpInferShape(x.shape(),
                                         filter.shape(),
                                         y.shape(),
                                         data_format,
                                         groups,
                                         padding_algorithm,
                                         strides,
                                         paddings,
                                         dilations);
  paddle::Tensor out = paddle::empty(out_shapes[0], x.dtype(), x.place());
  if (custom_kernel::LaunchAOTKernel()) {
    THROW_AOT_UNIMPLEMENTED();
  } else {
    custom_kernel::TensorNameMap input_names;
    custom_kernel::TensorValueMap inputs;
    input_names["X"] = {"x"};
    input_names["Filter"] = {"filter"};
    input_names["Y"] = {"y"};
    inputs["X"] = {const_cast<phi::DenseTensor*>(x_tensor)};
    inputs["Filter"] = {const_cast<phi::DenseTensor*>(filter_tensor)};
    inputs["Y"] = {const_cast<phi::DenseTensor*>(y_tensor)};

    custom_kernel::GcuAttributeMap attrs;
    attrs["groups"] = groups;
    attrs["data_format"] = data_format;
    attrs["padding_algorithm"] = padding_algorithm;
    attrs["strides"] = strides;
    attrs["paddings"] = paddings;
    attrs["dilations"] = dilations;
    attrs["axis"] = axis;

    custom_kernel::TensorNameMap output_names;
    custom_kernel::TensorValueMap outputs;
    output_names["Out"] = {"Out"};
    outputs["Out"] = {static_cast<phi::DenseTensor*>(out.impl().get())};

    custom_kernel::GcuRunner(input_names,
                             inputs,
                             output_names,
                             outputs,
                             attrs,
                             "conv_bias_relu",
                             *dev_ctx);
  }
  return {out};
}

std::vector<paddle::Tensor> ConvBiasHardSigmoidOp(const paddle::Tensor& x,
                                                  const paddle::Tensor& filter,
                                                  const paddle::Tensor& y,
                                                  std::string data_format,
                                                  int64_t groups,
                                                  std::string padding_algorithm,
                                                  std::vector<int> strides,
                                                  std::vector<int> paddings,
                                                  std::vector<int> dilations,
                                                  int axis,
                                                  float slope,
                                                  float offset) {
  auto dev_ctx = static_cast<const phi::CustomContext*>(
      paddle::experimental::DeviceContextPool::Instance().Get(x.place()));

  auto x_tensor = static_cast<const phi::DenseTensor*>(x.impl().get());
  auto filter_tensor =
      static_cast<const phi::DenseTensor*>(filter.impl().get());
  auto y_tensor = static_cast<const phi::DenseTensor*>(y.impl().get());
  // Infershape
  auto out_shapes = ConvBiasOpInferShape(x.shape(),
                                         filter.shape(),
                                         y.shape(),
                                         data_format,
                                         groups,
                                         padding_algorithm,
                                         strides,
                                         paddings,
                                         dilations);
  paddle::Tensor out = paddle::empty(out_shapes[0], x.dtype(), x.place());

  custom_kernel::TensorNameMap input_names;
  custom_kernel::TensorValueMap inputs;
  input_names["X"] = {"x"};
  input_names["Filter"] = {"filter"};
  input_names["Y"] = {"y"};
  inputs["X"] = {const_cast<phi::DenseTensor*>(x_tensor)};
  inputs["Filter"] = {const_cast<phi::DenseTensor*>(filter_tensor)};
  inputs["Y"] = {const_cast<phi::DenseTensor*>(y_tensor)};

  custom_kernel::GcuAttributeMap attrs;
  attrs["groups"] = groups;
  attrs["data_format"] = data_format;
  attrs["padding_algorithm"] = padding_algorithm;
  attrs["strides"] = strides;
  attrs["paddings"] = paddings;
  attrs["dilations"] = dilations;
  attrs["axis"] = axis;
  attrs["slope"] = slope;
  attrs["offset"] = offset;

  custom_kernel::TensorNameMap output_names;
  custom_kernel::TensorValueMap outputs;
  output_names["Out"] = {"Out"};
  outputs["Out"] = {static_cast<phi::DenseTensor*>(out.impl().get())};

  custom_kernel::GcuRunner(input_names,
                           inputs,
                           output_names,
                           outputs,
                           attrs,
                           "conv_bias_hard_sigmoid",
                           *dev_ctx);

  return {out};
}

PD_BUILD_OP(GCUConvBias)
    .Inputs({"X", "Filter", "Y"})
    .Outputs({"Out"})
    .Attrs({"data_format: std::string",
            "groups: int64_t",
            "padding_algorithm: std::string",
            "strides: std::vector<int>",
            "paddings: std::vector<int>",
            "dilations: std::vector<int>",
            "axis: int"})
    .SetKernelFn(PD_KERNEL(ConvBiasOp))
    .SetInferShapeFn(PD_INFER_SHAPE(ConvBiasOpInferShape));

PD_BUILD_OP(GCUConvBiasRelu)
    .Inputs({"X", "Filter", "Y"})
    .Outputs({"Out"})
    .Attrs({"data_format: std::string",
            "groups: int64_t",
            "padding_algorithm: std::string",
            "strides: std::vector<int>",
            "paddings: std::vector<int>",
            "dilations: std::vector<int>",
            "axis: int"})
    .SetKernelFn(PD_KERNEL(ConvBiasReluOp))
    .SetInferShapeFn(PD_INFER_SHAPE(ConvBiasOpInferShape));

PD_BUILD_OP(GCUConvBiasHardSigmoid)
    .Inputs({"X", "Filter", "Y"})
    .Outputs({"Out"})
    .Attrs({"data_format: std::string",
            "groups: int64_t",
            "padding_algorithm: std::string",
            "strides: std::vector<int>",
            "paddings: std::vector<int>",
            "dilations: std::vector<int>",
            "axis: int",
            "slope: float",
            "offset: float"})
    .SetKernelFn(PD_KERNEL(ConvBiasHardSigmoidOp))
    .SetInferShapeFn(PD_INFER_SHAPE(ConvBiasOpInferShape));
