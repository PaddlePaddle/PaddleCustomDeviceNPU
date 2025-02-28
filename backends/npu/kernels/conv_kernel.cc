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

#include "kernels/funcs/conv_util.h"
#include "kernels/funcs/npu_funcs.h"
#include "kernels/funcs/npu_op_runner.h"

namespace custom_kernel {

template <typename T, typename Context>
void AclopDepthwiseConv2dKernel(const Context& dev_ctx,
                                const phi::DenseTensor& input,
                                const phi::DenseTensor& filter,
                                const std::vector<int>& stride,
                                const std::vector<int>& padding,
                                int groups,
                                const std::vector<int>& dilation,
                                const std::string& data_format,
                                const bool channel_last,
                                phi::DenseTensor* out) {
  if (FLAGS_npu_storage_format) {
    AllocNPUTensor<T>(dev_ctx, ACL_FORMAT_NC1HWC0, out);
  } else {
    dev_ctx.template Alloc<T>(out);
  }

  PADDLE_ENFORCE_EQ(channel_last && FLAGS_npu_storage_format,
                    false,
                    phi::errors::InvalidArgument(
                        "PaddlePaddle do not support NPU storage format when "
                        "Conv2D in NHWC format, but got data_format [%s] and "
                        "FLAGS_npu_storage_format [%d]. Please execute 'export "
                        "FLAGS_npu_storage_format=0' in your environment.",
                        data_format,
                        FLAGS_npu_storage_format));

  std::vector<int> strides(4, 1);
  std::vector<int> dilations(4, 1);

  phi::DenseTensor input_tensor(input), output_tensor(*out);

  if (channel_last) {
    phi::DenseTensorMeta input_meta = {
        input.dtype(), input.dims(), phi::DataLayout::kNHWC};
    phi::DenseTensorMeta output_meta = {
        out->dtype(), out->dims(), phi::DataLayout::kNHWC};
    input_tensor.set_meta(input_meta);
    output_tensor.set_meta(output_meta);
    dev_ctx.template Alloc<T>(&input_tensor);
    dev_ctx.template Alloc<T>(&output_tensor);
    strides[1] = stride[0];
    strides[2] = stride[1];
    dilations[1] = dilation[0];
    dilations[2] = dilation[1];
  } else {
    strides[2] = stride[0];
    strides[3] = stride[1];
    dilations[2] = dilation[0];
    dilations[3] = dilation[1];
  }

  auto stream = dev_ctx.stream();

  NpuOpRunner runner_conv2d;
  runner_conv2d.SetType("Conv2D")
      .AddInput(input_tensor)
      .AddInput(filter)
      .AddOutput(output_tensor)
      .AddAttrs({{"strides", strides}})
      .AddAttrs({{"pads", padding}})
      .AddAttrs({{"dilations", dilations}})
      .AddAttrs({{"groups", groups}})
      .AddAttrs({{"data_format", data_format}})
      .Run(stream);
}

template <typename T, typename Context>
void DepthwiseConv2dKernel(const Context& dev_ctx,
                           const phi::DenseTensor& input,
                           const phi::DenseTensor& filter,
                           const std::vector<int>& stride,
                           const std::vector<int>& paddings_in,
                           const std::string& padding_algorithm,
                           int groups,
                           const std::vector<int>& dilations_in,
                           const std::string& data_format,
                           phi::DenseTensor* out) {
  std::vector<int> padding = paddings_in;
  std::vector<int> dilation = dilations_in;

  const bool channel_last = data_format == "NHWC";
  auto in_dims = input.dims();
  auto filter_dims = filter.dims();
  phi::DDim in_data_dims;
  phi::DDim filter_data_dims;

  if (channel_last) {
    in_data_dims = phi::slice_ddim(in_dims, 1, in_dims.size() - 1);
  } else {
    in_data_dims = phi::slice_ddim(in_dims, 2, in_dims.size());
  }
  filter_data_dims = phi::slice_ddim(filter_dims, 2, in_dims.size());

  std::vector<int> ksize = phi::vectorize<int>(filter_data_dims);
  custom_kernel::UpdatePaddingAndDilation(
      &padding, &dilation, padding_algorithm, in_data_dims, stride, ksize);

  if (padding[0] != padding[1] || padding[2] != padding[3]) {
    VLOG(2) << "Fallback to AclopDepthwiseConv2dKernel due to asymmetric "
               "padding: {"
            << padding[0] << ", " << padding[1] << ", " << padding[2] << ", "
            << padding[3] << "}";
    return custom_kernel::AclopDepthwiseConv2dKernel<T, Context>(dev_ctx,
                                                                 input,
                                                                 filter,
                                                                 stride,
                                                                 padding,
                                                                 groups,
                                                                 dilation,
                                                                 data_format,
                                                                 channel_last,
                                                                 out);
  }

  if (FLAGS_npu_storage_format) {
    VLOG(2) << "Fallback to AclopDepthwiseConv2dKernel since "
               "`FLAGS_npu_storage_format` is ON";
    return custom_kernel::AclopDepthwiseConv2dKernel<T, Context>(dev_ctx,
                                                                 input,
                                                                 filter,
                                                                 stride,
                                                                 padding,
                                                                 groups,
                                                                 dilation,
                                                                 data_format,
                                                                 channel_last,
                                                                 out);
  }

  if (channel_last) {
    VLOG(2) << "Fallback to AclopDepthwiseConv2dKernel since "
               "`dataformat` is `NHWC`";
    return custom_kernel::AclopDepthwiseConv2dKernel<T, Context>(dev_ctx,
                                                                 input,
                                                                 filter,
                                                                 stride,
                                                                 padding,
                                                                 groups,
                                                                 dilation,
                                                                 data_format,
                                                                 channel_last,
                                                                 out);
  }

  DO_COMPATIBILITY(
      aclnnConvolution,
      (custom_kernel::AclopDepthwiseConv2dKernel<T, Context>(dev_ctx,
                                                             input,
                                                             filter,
                                                             stride,
                                                             padding,
                                                             groups,
                                                             dilation,
                                                             data_format,
                                                             channel_last,
                                                             out)));

  dev_ctx.template Alloc<T>(out);

  phi::DenseTensor input_tensor(input), output_tensor(*out);

  // prepare an zeros-filled bias tensor
  phi::DenseTensor bias_tensor;
  phi::DenseTensorMeta bias_meta = {input.dtype(),
                                    phi::slice_ddim(filter_dims, 0, 1)};
  bias_tensor.set_meta(bias_meta);
  FillNpuTensorWithConstant<T>(&bias_tensor, dev_ctx, static_cast<T>(0));

  std::vector<int64_t> ksize_(ksize.begin(), ksize.end());
  std::vector<int64_t> stride_(stride.begin(), stride.end());
  std::vector<int64_t> padding_ = {padding[0], padding[2]};
  std::vector<int64_t> dilation_(dilation.begin(), dilation.end());
  bool transposed = false;
  std::vector<int64_t> output_padding = {0, 0};
  int64_t groups_ = groups;
  int8_t cubeMathType = 0;

  EXEC_NPU_CMD(aclnnConvolution,
               dev_ctx,
               input_tensor,
               filter,
               bias_tensor,
               stride_,
               padding_,
               dilation_,
               transposed,
               output_padding,
               groups_,
               output_tensor,
               cubeMathType);
}

template <typename T, typename Context>
void AclopDepthwiseConv2dGradKernel(const Context& dev_ctx,
                                    const phi::DenseTensor& input,
                                    const phi::DenseTensor& filter,
                                    const phi::DenseTensor& out_grad,
                                    const std::vector<int>& stride,
                                    const std::vector<int>& padding,
                                    int groups,
                                    const std::vector<int>& dilation,
                                    const std::string& data_format,
                                    const bool channel_last,
                                    phi::DenseTensor* input_grad,
                                    phi::DenseTensor* filter_grad) {
  auto stream = dev_ctx.stream();

  // Transform filter (n, 1, h, w) --> (1, n, h, w)
  phi::DenseTensor transformed_filter;
  phi::DenseTensorMeta meta = {
      filter.dtype(),
      {filter.dims()[1], filter.dims()[0], filter.dims()[2], filter.dims()[3]}};

  // construct NPU attr
  std::vector<int> strides(4, 1);
  std::vector<int> dilations(4, 1);

  phi::DenseTensor input_tensor(input), output_grad_tensor(out_grad);
  if (channel_last) {
    phi::DenseTensorMeta input_meta = {
        input.dtype(), input.dims(), phi::DataLayout::kNHWC};
    phi::DenseTensorMeta output_grad_meta = {
        out_grad.dtype(), out_grad.dims(), phi::DataLayout::kNHWC};
    input_tensor.set_meta(input_meta);
    output_grad_tensor.set_meta(output_grad_meta);
    strides[1] = stride[0];
    strides[2] = stride[1];
    dilations[1] = dilation[0];
    dilations[2] = dilation[1];
  } else {
    strides[2] = stride[0];
    strides[3] = stride[1];
    dilations[2] = dilation[0];
    dilations[3] = dilation[1];
  }
  if (filter_grad) {
    if (groups == 1 && FLAGS_npu_storage_format) {
      AllocNPUTensor<T>(dev_ctx, ACL_FORMAT_FRACTAL_Z, filter_grad);
    } else {
      dev_ctx.template Alloc<T>(filter_grad);
    }

    NpuOpRunner runner_filter;
    runner_filter.SetType("Conv2DBackpropFilter")
        .AddInput(input_tensor)
        .AddInput(dev_ctx, phi::vectorize<int>(filter.dims()))
        .AddInput(output_grad_tensor)
        .AddOutput(*filter_grad)
        .AddAttrs({{"strides", strides}})
        .AddAttrs({{"pads", padding}})
        .AddAttrs({{"dilations", dilations}})
        .AddAttrs({{"groups", groups}})
        .AddAttrs({{"data_format", data_format}})
        .Run(stream);
  }
  if (input_grad) {
    if (FLAGS_npu_storage_format) {
      AllocNPUTensor<T>(dev_ctx, ACL_FORMAT_NC1HWC0, input_grad);
    } else {
      dev_ctx.template Alloc<T>(input_grad);
    }

    phi::DenseTensor input_grad_tensor(*input_grad);
    if (channel_last) {
      phi::DenseTensorMeta input_grad_meta = {
          input_grad->dtype(), input_grad->dims(), phi::DataLayout::kNHWC};
      input_grad_tensor.set_meta(input_grad_meta);
    }

    NpuOpRunner runner_filter;
    runner_filter.SetType("Conv2DBackpropInput")
        .AddInput(dev_ctx, phi::vectorize<int>(input_tensor.dims()))
        .AddInput(filter)
        .AddInput(output_grad_tensor)
        .AddOutput(input_grad_tensor)
        .AddAttrs({{"strides", strides}})
        .AddAttrs({{"pads", padding}})
        .AddAttrs({{"dilations", dilations}})
        .AddAttrs({{"groups", groups}})
        .AddAttrs({{"data_format", data_format}})
        .Run(stream);
  }
}

template <typename T, typename Context>
void DepthwiseConv2dGradKernel(const Context& dev_ctx,
                               const phi::DenseTensor& input,
                               const phi::DenseTensor& filter,
                               const phi::DenseTensor& out_grad,
                               const std::vector<int>& stride,
                               const std::vector<int>& paddings_in,
                               const std::string& padding_algorithm,
                               int groups,
                               const std::vector<int>& dilations_in,
                               const std::string& data_format,
                               phi::DenseTensor* input_grad,
                               phi::DenseTensor* filter_grad) {
  std::vector<int> padding = paddings_in;
  std::vector<int> dilation = dilations_in;
  const bool channel_last = data_format == "NHWC";
  // update padding and dilation
  auto in_dims = input.dims();
  auto filter_dims = filter.dims();
  phi::DDim in_data_dims;
  phi::DDim filter_data_dims;

  if (channel_last) {
    in_data_dims = phi::slice_ddim(in_dims, 1, in_dims.size() - 1);
  } else {
    in_data_dims = phi::slice_ddim(in_dims, 2, in_dims.size());
  }
  filter_data_dims = phi::slice_ddim(filter_dims, 2, in_dims.size());

  std::vector<int> ksize = phi::vectorize<int>(filter_data_dims);
  custom_kernel::UpdatePaddingAndDilation(
      &padding, &dilation, padding_algorithm, in_data_dims, stride, ksize);

  DO_COMPATIBILITY(
      aclnnConvolutionBackward,
      (custom_kernel::AclopDepthwiseConv2dGradKernel<T, Context>(dev_ctx,
                                                                 input,
                                                                 filter,
                                                                 out_grad,
                                                                 stride,
                                                                 padding,
                                                                 groups,
                                                                 dilation,
                                                                 data_format,
                                                                 channel_last,
                                                                 input_grad,
                                                                 filter_grad)));

  if (padding[0] != padding[1] || padding[2] != padding[3]) {
    VLOG(2) << "Fallback to AclopDepthwiseConv2dGradKernel due to asymmetric "
               "padding: {"
            << padding[0] << ", " << padding[1] << ", " << padding[2] << ", "
            << padding[3] << "}";
    return custom_kernel::AclopDepthwiseConv2dGradKernel<T, Context>(
        dev_ctx,
        input,
        filter,
        out_grad,
        stride,
        padding,
        groups,
        dilation,
        data_format,
        channel_last,
        input_grad,
        filter_grad);
  }
  if (FLAGS_npu_storage_format) {
    VLOG(2) << "Fallback to AclopDepthwiseConv2dGradKernel since "
               "`FLAGS_npu_storage_format` is ON";
    return custom_kernel::AclopDepthwiseConv2dGradKernel<T, Context>(
        dev_ctx,
        input,
        filter,
        out_grad,
        stride,
        padding,
        groups,
        dilation,
        data_format,
        channel_last,
        input_grad,
        filter_grad);
  }

  if (channel_last) {
    VLOG(2) << "Fallback to AclopDepthwiseConv2dGradKernel since "
               "`dataformat` is `NHWC`";
    return custom_kernel::AclopDepthwiseConv2dGradKernel<T, Context>(
        dev_ctx,
        input,
        filter,
        out_grad,
        stride,
        padding,
        groups,
        dilation,
        data_format,
        channel_last,
        input_grad,
        filter_grad);
  }

  phi::DenseTensor input_tensor(input), output_grad_tensor(out_grad);

  phi::DenseTensor filter_grad_tensor;
  phi::DenseTensor input_grad_tensor;
  phi::DenseTensor bias_grad_tensor;

  if (filter_grad) {
    dev_ctx.template Alloc<T>(filter_grad);
    filter_grad_tensor = phi::DenseTensor(*filter_grad);
  } else {
    phi::DenseTensorMeta filter_grad_meta = {filter.dtype(), filter.dims()};
    filter_grad_tensor.set_meta(filter_grad_meta);
    dev_ctx.template Alloc<T>(&filter_grad_tensor);
  }

  if (input_grad) {
    dev_ctx.template Alloc<T>(input_grad);
    input_grad_tensor = phi::DenseTensor(*input_grad);
  } else {
    phi::DenseTensorMeta input_grad_meta = {input.dtype(), input.dims()};
    input_grad_tensor.set_meta(input_grad_meta);
    dev_ctx.template Alloc<T>(&input_grad_tensor);
  }

  phi::DenseTensorMeta bias_grad_meta = {
      input.dtype(), phi::make_ddim({filter_grad_tensor.dims()[0]})};
  bias_grad_tensor.set_meta(bias_grad_meta);
  dev_ctx.template Alloc<T>(&bias_grad_tensor);

  std::vector<int64_t> bias_sizes =
      phi::vectorize(phi::slice_ddim(filter_dims, 0, 1));
  std::vector<int64_t> stride_(stride.begin(), stride.end());
  std::vector<int64_t> padding_ = {padding[0], padding[2]};
  std::vector<int64_t> dilation_(dilation.begin(), dilation.end());
  bool transposed = false;
  std::vector<int64_t> output_padding = {0, 0};
  std::array<bool, 3> output_mask = {
      input_grad != nullptr, filter_grad != nullptr, false};
  int8_t cubeMathType = 0;

  EXEC_NPU_CMD(aclnnConvolutionBackward,
               dev_ctx,
               output_grad_tensor,
               input_tensor,
               filter,
               bias_sizes,
               stride_,
               padding_,
               dilation_,
               transposed,
               output_padding,
               groups,
               output_mask,
               cubeMathType,
               input_grad_tensor,
               filter_grad_tensor,
               bias_grad_tensor);
}

template <typename T, typename Context>
void Conv3dKernel(const Context& dev_ctx,
                  const phi::DenseTensor& input,
                  const phi::DenseTensor& filter,
                  const std::vector<int>& strides,
                  const std::vector<int>& padding,
                  const std::string& padding_algorithm,
                  int groups,
                  const std::vector<int>& dilation,
                  const std::string& data_format,
                  phi::DenseTensor* out) {
  auto paddings = padding;
  auto dilations = dilation;

  PADDLE_ENFORCE_EQ(data_format,
                    "NCDHW",
                    phi::errors::Unimplemented(
                        "the data_format must be NCDHW in "
                        "the npu kernel of conv3d, but got data_format "
                        "= [%s]",
                        data_format));

  PADDLE_ENFORCE_EQ(
      groups,
      1,
      phi::errors::Unimplemented("the groups must be 1 in "
                                 "the npu kernel of conv3d, but got groups "
                                 "= [%d]",
                                 groups));

  dev_ctx.template Alloc<T>(out);

  phi::DenseTensor input_tensor(input);
  phi::DenseTensor filter_tensor(filter);
  phi::DenseTensor output_tensor(*out);

  phi::DenseTensorMeta input_meta = {
      input_tensor.dtype(), input_tensor.dims(), phi::DataLayout::kNCDHW};
  input_tensor.set_meta(input_meta);

  phi::DenseTensorMeta filter_meta = {
      filter_tensor.dtype(), filter_tensor.dims(), phi::DataLayout::kNCDHW};
  filter_tensor.set_meta(filter_meta);

  phi::DenseTensorMeta output_meta = {
      output_tensor.dtype(), output_tensor.dims(), phi::DataLayout::kNCDHW};
  output_tensor.set_meta(output_meta);

  // update padding and dilation
  auto in_dims = input.dims();
  auto filter_dims = filter.dims();
  phi::DDim in_data_dims;
  phi::DDim filter_data_dims;

  in_data_dims = phi::slice_ddim(in_dims, 2, in_dims.size());
  filter_data_dims = phi::slice_ddim(filter_dims, 2, in_dims.size());

  std::vector<int> ksize = phi::vectorize<int>(filter_data_dims);
  custom_kernel::UpdatePaddingAndDilation(
      &paddings, &dilations, padding_algorithm, in_data_dims, strides, ksize);

  std::vector<int> strides_vec(5, 1);
  std::vector<int> dilations_vec(5, 1);

  strides_vec[2] = strides[0];
  strides_vec[3] = strides[1];
  strides_vec[4] = strides[2];
  dilations_vec[2] = dilations[0];
  dilations_vec[3] = dilations[1];
  dilations_vec[4] = dilations[2];

  auto stream = dev_ctx.stream();
  const auto& runner = NpuOpRunner("Conv3D",
                                   {input_tensor, filter_tensor},
                                   {output_tensor},
                                   {{"strides", strides_vec},
                                    {"pads", paddings},
                                    {"dilations", dilations_vec},
                                    {"groups", groups},
                                    {"data_format", data_format}});
  runner.Run(stream);
}

template <typename T, typename Context>
void Conv3dGradKernel(const Context& dev_ctx,
                      const phi::DenseTensor& input,
                      const phi::DenseTensor& filter,
                      const phi::DenseTensor& out_grad,
                      const std::vector<int>& strides,
                      const std::vector<int>& padding,
                      const std::string& padding_algorithm,
                      int groups,
                      const std::vector<int>& dilation,
                      const std::string& data_format,
                      phi::DenseTensor* input_grad,
                      phi::DenseTensor* filter_grad) {
  auto paddings = padding;
  auto dilations = dilation;

  PADDLE_ENFORCE_EQ(data_format,
                    "NCDHW",
                    phi::errors::Unimplemented(
                        "the data_format must be NCDHW in "
                        "the npu kernel of conv3d, but got data_format "
                        "= [%s]",
                        data_format));

  PADDLE_ENFORCE_EQ(
      groups,
      1,
      phi::errors::Unimplemented("the groups must be 1 in "
                                 "the npu kernel of conv3d, but got groups "
                                 "= [%d]",
                                 groups));

  phi::DenseTensor input_tensor(input);
  phi::DenseTensor filter_tensor(filter);
  phi::DenseTensor output_grad_tensor(out_grad);

  phi::DenseTensorMeta input_meta = {
      input_tensor.dtype(), input_tensor.dims(), phi::DataLayout::kNCDHW};
  input_tensor.set_meta(input_meta);

  phi::DenseTensorMeta filter_meta = {
      filter_tensor.dtype(), filter_tensor.dims(), phi::DataLayout::kNCDHW};
  filter_tensor.set_meta(filter_meta);

  phi::DenseTensorMeta output_meta = {output_grad_tensor.dtype(),
                                      output_grad_tensor.dims(),
                                      phi::DataLayout::kNCDHW};
  output_grad_tensor.set_meta(output_meta);

  // update padding and dilation
  auto in_dims = input.dims();
  auto filter_dims = filter.dims();
  phi::DDim in_data_dims;
  phi::DDim filter_data_dims;

  in_data_dims = phi::slice_ddim(in_dims, 1, in_dims.size() - 1);
  filter_data_dims = phi::slice_ddim(filter_dims, 2, in_dims.size());

  std::vector<int> ksize = phi::vectorize<int>(filter_data_dims);
  custom_kernel::UpdatePaddingAndDilation(
      &paddings, &dilations, padding_algorithm, in_data_dims, strides, ksize);

  std::vector<int> strides_vec(5, 1);
  std::vector<int> dilations_vec(5, 1);

  strides_vec[2] = strides[0];
  strides_vec[3] = strides[1];
  strides_vec[4] = strides[2];
  dilations_vec[2] = dilations[0];
  dilations_vec[3] = dilations[1];
  dilations_vec[4] = dilations[2];

  auto stream = dev_ctx.stream();

  if (filter_grad) {
    dev_ctx.template Alloc<T>(filter_grad);
    std::vector<int> filter_shape_vec = phi::vectorize<int>(filter.dims());

    phi::DenseTensor filter_grad_tensor(*filter_grad);
    phi::DenseTensorMeta filter_grad_meta = {filter_grad_tensor.dtype(),
                                             filter_grad_tensor.dims(),
                                             phi::DataLayout::kNCDHW};
    filter_grad_tensor.set_meta(filter_grad_meta);

    // Conv3DBackpropFilterD only support fp32 output, so we need cast the
    // output when the out dtype is fp16.
    phi::DenseTensor filter_grad_tmp;
    if (filter_grad->dtype() == phi::DataType::FLOAT16) {
      phi::DenseTensorMeta filter_grad_tmp_meta = {phi::DataType::FLOAT32,
                                                   filter_grad_tensor.dims(),
                                                   phi::DataLayout::kNCDHW};
      filter_grad_tmp.set_meta(filter_grad_tmp_meta);
      dev_ctx.template Alloc<float>(&filter_grad_tmp);
    } else {
      filter_grad_tmp = filter_grad_tensor;
    }

    const auto& runner = NpuOpRunner("Conv3DBackpropFilterD",
                                     {input_tensor, output_grad_tensor},
                                     {filter_grad_tmp},
                                     {{"filter_size", filter_shape_vec},
                                      {"strides", strides_vec},
                                      {"pads", paddings},
                                      {"dilations", dilations_vec},
                                      {"groups", groups},
                                      {"data_format", data_format}});
    runner.Run(stream);
    dev_ctx.Wait();
    if (filter_grad->dtype() == phi::DataType::FLOAT16) {
      const auto& cast_runner = NpuOpRunner("Cast",
                                            {filter_grad_tmp},
                                            {*filter_grad},
                                            {{"dst_type", ACL_FLOAT16}});
      cast_runner.Run(stream);
    }
  }

  if (input_grad) {
    dev_ctx.template Alloc<T>(input_grad);
    std::vector<int> input_shape_vec = phi::vectorize<int>(input.dims());

    phi::DenseTensor input_grad_tensor(*input_grad);
    phi::DenseTensorMeta input_grad_meta = {input_grad_tensor.dtype(),
                                            input_grad_tensor.dims(),
                                            phi::DataLayout::kNCDHW};
    input_grad_tensor.set_meta(input_grad_meta);

    const auto& runner = NpuOpRunner("Conv3DBackpropInputD",
                                     {filter_tensor, output_grad_tensor},
                                     {input_grad_tensor},
                                     {{"input_size", input_shape_vec},
                                      {"strides", strides_vec},
                                      {"pads", paddings},
                                      {"dilations", dilations_vec},
                                      {"groups", groups},
                                      {"data_format", data_format}});
    runner.Run(stream);
  }
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(depthwise_conv2d,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::DepthwiseConv2dKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(depthwise_conv2d_grad,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::DepthwiseConv2dGradKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(conv3d,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::Conv3dKernel,
                          float,
                          phi::dtype::bfloat16,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(conv3d_grad,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::Conv3dGradKernel,
                          float,
                          phi::dtype::bfloat16,
                          phi::dtype::float16) {}
