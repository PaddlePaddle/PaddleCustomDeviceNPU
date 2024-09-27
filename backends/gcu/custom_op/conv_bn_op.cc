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

std::vector<std::vector<int64_t>> ConvBnOpInferShape(
    const std::vector<int64_t>& x_shape,
    const std::vector<int64_t>& filter_shape,
    const std::vector<int64_t>& mean_shape,
    const std::vector<int64_t>& bias_shape,
    const std::vector<int64_t>& scale_shape,
    const std::vector<int64_t>& var_shape,
    std::string data_format,
    int64_t groups,
    std::string padding_algorithm,
    std::vector<int> strides,
    std::vector<int> paddings,
    std::vector<int> dilations,
    bool is_test,
    std::string data_layout,
    float momentum,
    float epsilon,
    bool trainable_statistics,
    bool use_global_stats) {
  VLOG(3) << "---------- here is conv_bn op ------";
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
  return {output_shape, mean_shape, mean_shape, var_shape, var_shape};
}

void FuseConvBnWeights(const phi::CustomContext* dev_ctx,
                       const phi::DenseTensor& filter,
                       const phi::DenseTensor& conv_bias,
                       const phi::DenseTensor& scale,
                       const phi::DenseTensor& bn_bias,
                       const phi::DenseTensor& mean,
                       const phi::DenseTensor& var,
                       const std::string data_format,
                       const float epsilon,
                       phi::DenseTensor*& fuse_filter_out,  // NOLINT
                       phi::DenseTensor*& fuse_bias_out) {  // NOLINT
  VLOG(6) << "start fuse conv bn weigths process.  " << filter.data() << " "
          << conv_bias.data() << " " << scale.data() << " " << bn_bias.data()
          << " " << mean.data() << " " << var.data() << " " << epsilon;
  static std::map<const void*, std::vector<std::shared_ptr<phi::DenseTensor>>>
      tensors2fuse_tensors;
  if (tensors2fuse_tensors.count(filter.data()) != 0) {
    fuse_filter_out = tensors2fuse_tensors.at(filter.data()).at(0).get();
    fuse_bias_out = tensors2fuse_tensors.at(filter.data()).at(1).get();
    VLOG(3) << "Conv Weight [" << filter.name() << ", " << conv_bias.name()
            << "] have already done fuse batchnorm weghts.";
    return;
  }
  // new_filter = filter * scale / sqrt(var + epsilon)
  // new_bias = scale * (conv_bias - mean) / sqrt(var + epsilon) + bn_bias
  // reshape
  auto rank = filter.dims().size();
  auto dims = phi::vectorize(scale.dims());
  PADDLE_ENFORCE_EQ(
      dims.size(),
      1,
      phi::errors::InvalidArgument("Scale should be 1-D tensor."));
  if (data_format == "NCHW" || data_format == "NCDHW") {
    for (int i = 1; i < rank; ++i) {
      dims.push_back(1);
    }
  }
  PADDLE_ENFORCE_EQ(dims.size(),
                    rank,
                    phi::errors::PreconditionNotMet(
                        "reshaped dims rank should be same with input rank."));
  auto new_filter = std::make_shared<phi::DenseTensor>();
  auto new_conv_bias = std::make_shared<phi::DenseTensor>();
  phi::DenseTensor new_scale;
  phi::DenseTensor new_bn_bias;
  phi::DenseTensor new_mean;
  phi::DenseTensor new_var;
  new_filter->set_meta(filter.meta());
  new_conv_bias->set_meta(conv_bias.meta());
  new_scale.set_meta(scale.meta());
  new_bn_bias.set_meta(bn_bias.meta());
  new_mean.set_meta(mean.meta());
  new_var.set_meta(var.meta());
  if (filter.dtype() == phi::DataType::FLOAT16) {
    auto meta = new_scale.meta();
    meta.dtype = phi::DataType::FLOAT16;
    new_scale.set_meta(meta);
    new_bn_bias.set_meta(meta);
    new_mean.set_meta(meta);
    new_var.set_meta(meta);
  }
  if (new_filter->dtype() == phi::DataType::FLOAT32) {
    dev_ctx->Alloc<float>(new_filter.get());
    dev_ctx->Alloc<float>(new_conv_bias.get());
    dev_ctx->Alloc<float>(&new_scale);
    dev_ctx->Alloc<float>(&new_bn_bias);
    dev_ctx->Alloc<float>(&new_mean);
    dev_ctx->Alloc<float>(&new_var);
  } else {  // fp16
    dev_ctx->Alloc<phi::float16>(new_filter.get());
    dev_ctx->Alloc<phi::float16>(new_conv_bias.get());
    dev_ctx->Alloc<phi::float16>(&new_scale);
    dev_ctx->Alloc<phi::float16>(&new_bn_bias);
    dev_ctx->Alloc<phi::float16>(&new_mean);
    dev_ctx->Alloc<phi::float16>(&new_var);
  }
  phi::Copy(*dev_ctx, filter, dev_ctx->GetPlace(), false, new_filter.get());
  // Do Cast
  if (filter.dtype() == phi::DataType::FLOAT16) {
    if (new_scale.dtype() == phi::DataType::FLOAT32) {
      custom_kernel::Cast(*dev_ctx, scale, phi::DataType::FLOAT16, &new_scale);
    }
    if (new_bn_bias.dtype() == phi::DataType::FLOAT32) {
      custom_kernel::Cast(
          *dev_ctx, bn_bias, phi::DataType::FLOAT16, &new_bn_bias);
    }
    if (new_mean.dtype() == phi::DataType::FLOAT32) {
      custom_kernel::Cast(*dev_ctx, mean, phi::DataType::FLOAT16, &new_mean);
    }
    if (new_var.dtype() == phi::DataType::FLOAT32) {
      custom_kernel::Cast(*dev_ctx, var, phi::DataType::FLOAT16, &new_var);
    }
  } else {
    if (new_scale.dtype() == phi::DataType::FLOAT16) {
      phi::Copy(*dev_ctx, scale, dev_ctx->GetPlace(), false, &new_scale);
    }
    if (new_bn_bias.dtype() == phi::DataType::FLOAT16) {
      phi::Copy(*dev_ctx, bn_bias, dev_ctx->GetPlace(), false, &new_bn_bias);
    }
    if (new_mean.dtype() == phi::DataType::FLOAT16) {
      phi::Copy(*dev_ctx, mean, dev_ctx->GetPlace(), false, &new_mean);
    }
    if (new_var.dtype() == phi::DataType::FLOAT16) {
      phi::Copy(*dev_ctx, var, dev_ctx->GetPlace(), false, &new_var);
    }
  }
  // new_filter = filter * scale / sqrt(var + epsilon)
  // new_bias = scale * (conv_bias - mean) / sqrt(var + epsilon) + bn_bias
  phi::Scalar factor(1.0f);
  phi::Scalar eps(epsilon);
  LAUNCH_TOPSATENOP(topsatenAdd, (*dev_ctx), new_var, new_var, eps, factor);
  LAUNCH_TOPSATENOP(topsatenRsqrt, (*dev_ctx), new_var, new_var);
  LAUNCH_TOPSATENOP(topsatenMul, (*dev_ctx), new_scale, new_scale, new_var);
  // new_filter
  auto tmp_scale = new_scale;
  if (data_format == "NCHW" || data_format == "NCDHW") {
    tmp_scale.Resize(phi::make_ddim(dims));
  }
  LAUNCH_TOPSATENOP(topsatenMul, (*dev_ctx), *new_filter, tmp_scale, filter);
  // new_bias
  LAUNCH_TOPSATENOP(topsatenSub,
                    (*dev_ctx),
                    *new_conv_bias,
                    *new_conv_bias,
                    new_mean,
                    factor);
  LAUNCH_TOPSATENOP(
      topsatenMul, (*dev_ctx), *new_conv_bias, *new_conv_bias, new_scale);
  LAUNCH_TOPSATENOP(topsatenAdd,
                    (*dev_ctx),
                    *new_conv_bias,
                    *new_conv_bias,
                    new_bn_bias,
                    factor);
  tensors2fuse_tensors[filter.data()] = {new_filter, new_conv_bias};
  fuse_filter_out = new_filter.get();
  fuse_bias_out = new_conv_bias.get();
  VLOG(6) << "end fuse conv bn weigths process.";
}

std::vector<paddle::Tensor> ConvBnOp(const paddle::Tensor& x,
                                     const paddle::Tensor& filter,
                                     const paddle::Tensor& mean,
                                     const paddle::Tensor& bias,
                                     const paddle::Tensor& scale,
                                     const paddle::Tensor& var,
                                     std::string data_format,
                                     int64_t groups,
                                     std::string padding_algorithm,
                                     std::vector<int> strides,
                                     std::vector<int> paddings,
                                     std::vector<int> dilations,
                                     bool is_test,
                                     std::string data_layout,
                                     float momentum,
                                     float epsilon,
                                     bool trainable_statistics,
                                     bool use_global_stats) {
  PADDLE_GCU_KERNEL_TRACE("conv_bn");
  auto dev_ctx = reinterpret_cast<const phi::CustomContext*>(
      paddle::experimental::DeviceContextPool::Instance().Get(x.place()));

  auto x_tensor = reinterpret_cast<const phi::DenseTensor*>(x.impl().get());
  auto filter_tensor =
      reinterpret_cast<const phi::DenseTensor*>(filter.impl().get());
  auto mean_tensor =
      reinterpret_cast<const phi::DenseTensor*>(mean.impl().get());
  auto bias_tensor =
      reinterpret_cast<const phi::DenseTensor*>(bias.impl().get());
  auto scale_tensor =
      reinterpret_cast<const phi::DenseTensor*>(scale.impl().get());
  auto var_tensor = reinterpret_cast<const phi::DenseTensor*>(var.impl().get());
  PADDLE_ENFORCE_EQ(
      x_tensor->dtype() == phi::DataType::FLOAT32 ||
          x_tensor->dtype() == phi::DataType::FLOAT16,
      true,
      phi::errors::InvalidArgument(
          "Only support float32/float16 data type in conv_bn op, but "
          "received: %s.",
          x_tensor->dtype()));
  // Infershape
  auto out_shapes = ConvBnOpInferShape(x.shape(),
                                       filter.shape(),
                                       mean.shape(),
                                       bias.shape(),
                                       scale.shape(),
                                       var.shape(),
                                       data_format,
                                       groups,
                                       padding_algorithm,
                                       strides,
                                       paddings,
                                       dilations,
                                       is_test,
                                       data_layout,
                                       momentum,
                                       epsilon,
                                       trainable_statistics,
                                       use_global_stats);
  paddle::Tensor out = paddle::empty(out_shapes[0], x.dtype(), x.place());
  auto out_tensor = reinterpret_cast<phi::DenseTensor*>(out.impl().get());
  if (custom_kernel::LaunchAOTKernel()) {
    PADDLE_ENFORCE_EQ(
        data_layout == "NCHW" || data_layout == "NCDHW",
        true,
        phi::errors::InvalidArgument(
            "Only support NCHW/NCDHW layout in AOT mode, but received: %s.",
            data_layout.c_str()));
    paddle::Tensor conv_out =
        paddle::empty(out_shapes[0], x.dtype(), x.place());
    auto conv_out_tensor =
        reinterpret_cast<phi::DenseTensor*>(conv_out.impl().get());

    auto meta = phi::DenseTensorMeta(filter.dtype(),
                                     phi::make_ddim({filter.dims().at(0)}));
    phi::DenseTensor conv_bias = custom_kernel::TensorZeros(*dev_ctx, meta);

    std::vector<int64_t> strides_v;
    strides_v.insert(strides_v.begin(), strides.begin(), strides.end());
    std::vector<int64_t> paddings_v;
    paddings_v.insert(paddings_v.begin(), paddings.begin(), paddings.end());
    std::vector<int64_t> dilations_v;
    dilations_v.insert(dilations_v.begin(), dilations.begin(), dilations.end());
    int64_t groups_64 = groups;

    phi::DenseTensor* fused_filter;
    phi::DenseTensor* fused_bias;
    FuseConvBnWeights(dev_ctx,
                      *filter_tensor,
                      conv_bias,
                      *scale_tensor,
                      *bias_tensor,
                      *mean_tensor,
                      *var_tensor,
                      data_format,
                      epsilon,
                      fused_filter,
                      fused_bias);

    LAUNCH_TOPSATENOP(topsatenConv2d,
                      (*dev_ctx),
                      (*conv_out_tensor),
                      (*x_tensor),
                      (*fused_filter),
                      (*fused_bias),
                      strides_v,
                      paddings_v,
                      dilations_v,
                      groups_64);
    return {out, mean, mean, var, var};
  } else {
    custom_kernel::TensorNameMap input_names;
    custom_kernel::TensorValueMap inputs;
    input_names["X"] = {"x"};
    input_names["Filter"] = {"filter"};
    input_names["Mean"] = {"mean"};
    input_names["Bias"] = {"bias"};
    input_names["Scale"] = {"scale"};
    input_names["Variance"] = {"variance"};
    inputs["X"] = {const_cast<phi::DenseTensor*>(x_tensor)};
    inputs["Filter"] = {const_cast<phi::DenseTensor*>(filter_tensor)};
    inputs["Mean"] = {const_cast<phi::DenseTensor*>(mean_tensor)};
    inputs["Bias"] = {const_cast<phi::DenseTensor*>(bias_tensor)};
    inputs["Scale"] = {const_cast<phi::DenseTensor*>(scale_tensor)};
    inputs["Variance"] = {const_cast<phi::DenseTensor*>(var_tensor)};

    custom_kernel::GcuAttributeMap attrs;
    attrs["groups"] = groups;
    attrs["data_format"] = data_format;
    attrs["padding_algorithm"] = padding_algorithm;
    attrs["strides"] = strides;
    attrs["paddings"] = paddings;
    attrs["dilations"] = dilations;
    attrs["is_test"] = is_test;
    attrs["data_layout"] = data_layout;
    attrs["momentum"] = momentum;
    attrs["epsilon"] = epsilon;
    attrs["trainable_statistics"] = trainable_statistics;
    attrs["use_global_stats"] = use_global_stats;

    custom_kernel::TensorNameMap output_names;
    custom_kernel::TensorValueMap outputs;
    output_names["Out"] = {"Out"};
    outputs["Out"] = {static_cast<phi::DenseTensor*>(out.impl().get())};

    custom_kernel::GcuRunner(
        input_names, inputs, output_names, outputs, attrs, "conv_bn", *dev_ctx);

    return {out, mean, mean, var, var};
  }
}

std::vector<paddle::Tensor> ConvBnSwishOp(const paddle::Tensor& x,
                                          const paddle::Tensor& filter,
                                          const paddle::Tensor& mean,
                                          const paddle::Tensor& bias,
                                          const paddle::Tensor& scale,
                                          const paddle::Tensor& var,
                                          std::string data_format,
                                          int64_t groups,
                                          std::string padding_algorithm,
                                          std::vector<int> strides,
                                          std::vector<int> paddings,
                                          std::vector<int> dilations,
                                          bool is_test,
                                          std::string data_layout,
                                          float momentum,
                                          float epsilon,
                                          bool trainable_statistics,
                                          bool use_global_stats,
                                          int beta) {
  auto dev_ctx = static_cast<const phi::CustomContext*>(
      paddle::experimental::DeviceContextPool::Instance().Get(x.place()));

  auto x_tensor = static_cast<const phi::DenseTensor*>(x.impl().get());
  auto filter_tensor =
      static_cast<const phi::DenseTensor*>(filter.impl().get());
  auto mean_tensor = static_cast<const phi::DenseTensor*>(mean.impl().get());
  auto bias_tensor = static_cast<const phi::DenseTensor*>(bias.impl().get());
  auto scale_tensor = static_cast<const phi::DenseTensor*>(scale.impl().get());
  auto var_tensor = static_cast<const phi::DenseTensor*>(var.impl().get());
  // Infershape
  auto out_shapes = ConvBnOpInferShape(x.shape(),
                                       filter.shape(),
                                       mean.shape(),
                                       bias.shape(),
                                       scale.shape(),
                                       var.shape(),
                                       data_format,
                                       groups,
                                       padding_algorithm,
                                       strides,
                                       paddings,
                                       dilations,
                                       is_test,
                                       data_layout,
                                       momentum,
                                       epsilon,
                                       trainable_statistics,
                                       use_global_stats);
  paddle::Tensor out = paddle::empty(out_shapes[0], x.dtype(), x.place());

  custom_kernel::TensorNameMap input_names;
  custom_kernel::TensorValueMap inputs;
  input_names["X"] = {"x"};
  input_names["Filter"] = {"filter"};
  input_names["Mean"] = {"mean"};
  input_names["Bias"] = {"bias"};
  input_names["Scale"] = {"scale"};
  input_names["Variance"] = {"variance"};
  inputs["X"] = {const_cast<phi::DenseTensor*>(x_tensor)};
  inputs["Filter"] = {const_cast<phi::DenseTensor*>(filter_tensor)};
  inputs["Mean"] = {const_cast<phi::DenseTensor*>(mean_tensor)};
  inputs["Bias"] = {const_cast<phi::DenseTensor*>(bias_tensor)};
  inputs["Scale"] = {const_cast<phi::DenseTensor*>(scale_tensor)};
  inputs["Variance"] = {const_cast<phi::DenseTensor*>(var_tensor)};

  custom_kernel::GcuAttributeMap attrs;
  attrs["groups"] = groups;
  attrs["data_format"] = data_format;
  attrs["padding_algorithm"] = padding_algorithm;
  attrs["strides"] = strides;
  attrs["paddings"] = paddings;
  attrs["dilations"] = dilations;
  attrs["is_test"] = is_test;
  attrs["data_layout"] = data_layout;
  attrs["momentum"] = momentum;
  attrs["epsilon"] = epsilon;
  attrs["trainable_statistics"] = trainable_statistics;
  attrs["use_global_stats"] = use_global_stats;
  attrs["beta"] = beta;

  custom_kernel::TensorNameMap output_names;
  custom_kernel::TensorValueMap outputs;
  output_names["Out"] = {"Out"};
  outputs["Out"] = {static_cast<phi::DenseTensor*>(out.impl().get())};

  custom_kernel::GcuRunner(input_names,
                           inputs,
                           output_names,
                           outputs,
                           attrs,
                           "conv_bn_swish",
                           *dev_ctx);

  return {out, mean, mean, var, var};
}

PD_BUILD_OP(GCUConvBn)
    .Inputs({"X", "Filter", "Mean", "Bias", "Scale", "Var"})
    .Outputs({"Out", "MeanOut", "SavedMean", "SavedVariance", "VarianceOut"})
    .Attrs({"data_format: std::string",
            "groups: int64_t",
            "padding_algorithm: std::string",
            "strides: std::vector<int>",
            "paddings: std::vector<int>",
            "dilations: std::vector<int>",
            "is_test: bool",
            "data_layout: std::string",
            "momentum: float",
            "epsilon: float",
            "trainable_statistics: bool",
            "use_global_stats: bool"})
    .SetKernelFn(PD_KERNEL(ConvBnOp))
    .SetInferShapeFn(PD_INFER_SHAPE(ConvBnOpInferShape));

PD_BUILD_OP(GCUConvBnSwish)
    .Inputs({"X", "Filter", "Mean", "Bias", "Scale", "Var"})
    .Outputs({"Out", "MeanOut", "SavedMean", "SavedVariance", "VarianceOut"})
    .Attrs({"data_format: std::string",
            "groups: int64_t",
            "padding_algorithm: std::string",
            "strides: std::vector<int>",
            "paddings: std::vector<int>",
            "dilations: std::vector<int>",
            "is_test: bool",
            "data_layout: std::string",
            "momentum: float",
            "epsilon: float",
            "trainable_statistics: bool",
            "use_global_stats: bool",
            "beta: int"})
    .SetKernelFn(PD_KERNEL(ConvBnSwishOp))
    .SetInferShapeFn(PD_INFER_SHAPE(ConvBnOpInferShape));
