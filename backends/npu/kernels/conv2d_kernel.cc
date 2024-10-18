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

#include "acl/acl_op_compiler.h"
#include "kernels/funcs/conv_util.h"
#include "kernels/funcs/npu_funcs.h"
#include "kernels/funcs/npu_op_runner.h"

namespace custom_kernel {

template <typename T, typename Context>
void AclopConv2dKernel(const Context& dev_ctx,
                       const phi::DenseTensor& input,
                       const phi::DenseTensor& filter,
                       const std::vector<int>& strides,
                       const std::vector<int>& paddings,
                       const std::vector<int>& dilations,
                       int groups,
                       const std::string& data_format,
                       const bool channel_last,
                       phi::DenseTensor* output) {
  PADDLE_ENFORCE_EQ(channel_last && FLAGS_npu_storage_format,
                    false,
                    phi::errors::InvalidArgument(
                        "PaddlePaddle do not support NPU storage format when "
                        "Conv2D in NHWC format, but got data_format [%s] and "
                        "FLAGS_npu_storage_format [%d]. Please execute 'export "
                        "FLAGS_npu_storage_format=0' in your environment.",
                        data_format,
                        FLAGS_npu_storage_format));

  if (FLAGS_npu_storage_format) {
    AllocNPUTensor<T>(dev_ctx, ACL_FORMAT_NC1HWC0, output);
  } else {
    dev_ctx.template Alloc<T>(output);
  }

  std::vector<int> strides_vec(4, 1);
  std::vector<int> dilations_vec(4, 1);

  phi::DenseTensor input_tensor(input), output_tensor(*output);
  if (channel_last) {
    phi::DenseTensorMeta input_meta = {
        input.dtype(), input.dims(), phi::DataLayout::kNHWC};
    phi::DenseTensorMeta output_meta = {
        output->dtype(), output->dims(), phi::DataLayout::kNHWC};
    input_tensor.set_meta(input_meta);
    output_tensor.set_meta(output_meta);
    dev_ctx.template Alloc<T>(&input_tensor);
    dev_ctx.template Alloc<T>(&output_tensor);
    strides_vec[1] = strides[0];
    strides_vec[2] = strides[1];
    dilations_vec[1] = dilations[0];
    dilations_vec[2] = dilations[1];
  } else {
    strides_vec[2] = strides[0];
    strides_vec[3] = strides[1];
    dilations_vec[2] = dilations[0];
    dilations_vec[3] = dilations[1];
  }

  auto stream = dev_ctx.stream();

  NpuOpRunner runner_conv2d;
  runner_conv2d.SetType("Conv2D")
      .AddInput(input_tensor)
      .AddInput(filter)
      .AddOutput(output_tensor)
      .AddAttrs({{"strides", strides_vec}})
      .AddAttrs({{"pads", paddings}})
      .AddAttrs({{"dilations", dilations_vec}})
      .AddAttrs({{"groups", groups}})
      .AddAttrs({{"data_format", data_format}})
      .Run(stream);
}

template <typename T, typename Context>
void Conv2dKernel(const Context& dev_ctx,
                  const phi::DenseTensor& input,
                  const phi::DenseTensor& filter,
                  const std::vector<int>& strides_t,
                  const std::vector<int>& paddings_t,
                  const std::string& padding_algorithm,
                  const std::vector<int>& dilations_t,
                  int groups,
                  const std::string& data_format,
                  phi::DenseTensor* output) {
  if (FLAGS_npu_jit_compile) {
    aclSetCompileopt(ACL_OP_JIT_COMPILE, "disable");
  }

  auto strides = strides_t;
  auto paddings = paddings_t;
  auto dilations = dilations_t;

  const bool channel_last = data_format == "NHWC";

  PADDLE_ENFORCE_EQ(channel_last && FLAGS_npu_storage_format,
                    false,
                    phi::errors::InvalidArgument(
                        "PaddlePaddle do not support NPU storage format when "
                        "Conv2D in NHWC format, but got data_format [%s] and "
                        "FLAGS_npu_storage_format [%d]. Please execute 'export "
                        "FLAGS_npu_storage_format=0' in your environment.",
                        data_format,
                        FLAGS_npu_storage_format));

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
      &paddings, &dilations, padding_algorithm, in_data_dims, strides, ksize);

  if (paddings[0] != paddings[1] || paddings[2] != paddings[3]) {
    VLOG(2) << "Fallback to AclopConv2dKernel due to asymmetric padding : {"
            << paddings[0] << ", " << paddings[1] << ", " << paddings[2] << ", "
            << paddings[3] << "}";
    return custom_kernel::AclopConv2dKernel<T, Context>(dev_ctx,
                                                        input,
                                                        filter,
                                                        strides,
                                                        paddings,
                                                        dilations,
                                                        groups,
                                                        data_format,
                                                        channel_last,
                                                        output);
  }

  if (FLAGS_npu_storage_format) {
    VLOG(2) << "Fallback to AclopConv2dKernel since `FLAGS_npu_storage_format` "
            << "is ON.";
    return custom_kernel::AclopConv2dKernel<T, Context>(dev_ctx,
                                                        input,
                                                        filter,
                                                        strides,
                                                        paddings,
                                                        dilations,
                                                        groups,
                                                        data_format,
                                                        channel_last,
                                                        output);
  }

  if (channel_last) {
    VLOG(2) << "Fallback to AclopConv2dKernel since `data_format` is NHWC.";
    return custom_kernel::AclopConv2dKernel<T, Context>(dev_ctx,
                                                        input,
                                                        filter,
                                                        strides,
                                                        paddings,
                                                        dilations,
                                                        groups,
                                                        data_format,
                                                        channel_last,
                                                        output);
  }

  DO_COMPATIBILITY(aclnnConvolution,
                   (custom_kernel::AclopConv2dKernel<T, Context>(dev_ctx,
                                                                 input,
                                                                 filter,
                                                                 strides,
                                                                 paddings,
                                                                 dilations,
                                                                 groups,
                                                                 data_format,
                                                                 channel_last,
                                                                 output)));

  dev_ctx.template Alloc<T>(output);

  phi::DenseTensor input_tensor(input), output_tensor(*output);

  // prepare an zeros-filled bias tensor
  phi::DenseTensor bias_tensor;
  phi::DenseTensorMeta bias_meta = {input.dtype(),
                                    phi::slice_ddim(filter_dims, 0, 1)};
  bias_tensor.set_meta(bias_meta);
  FillNpuTensorWithConstant<T>(&bias_tensor, dev_ctx, static_cast<T>(0));

  std::vector<int64_t> stride_(strides.begin(), strides.end());
  std::vector<int64_t> padding_ = {paddings[0], paddings[2]};
  std::vector<int64_t> dilation_(dilations.begin(), dilations.end());
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

  if (FLAGS_npu_jit_compile) {
    aclSetCompileopt(ACL_OP_JIT_COMPILE, "enable");
  }
}

template <typename T, typename Context>
void AclopConv2DGradKernel(const Context& dev_ctx,
                           const phi::DenseTensor& input,
                           const phi::DenseTensor& filter,
                           const phi::DenseTensor& output_grad,
                           const std::vector<int>& strides,
                           const std::vector<int>& paddings,
                           const std::vector<int>& dilations,
                           int groups,
                           const std::string& data_format,
                           const bool channel_last,
                           phi::DenseTensor* input_grad,
                           phi::DenseTensor* filter_grad) {
  std::vector<int> strides_vec(4, 1);
  std::vector<int> dilations_vec(4, 1);

  phi::DenseTensor input_tensor(input), output_grad_tensor(output_grad);
  if (channel_last) {
    phi::DenseTensorMeta input_meta = {
        input.dtype(), input.dims(), phi::DataLayout::kNHWC};
    phi::DenseTensorMeta output_grad_meta = {
        output_grad.dtype(), output_grad.dims(), phi::DataLayout::kNHWC};
    input_tensor.set_meta(input_meta);
    output_grad_tensor.set_meta(output_grad_meta);
    strides_vec[1] = strides[0];
    strides_vec[2] = strides[1];
    dilations_vec[1] = dilations[0];
    dilations_vec[2] = dilations[1];
  } else {
    strides_vec[2] = strides[0];
    strides_vec[3] = strides[1];
    dilations_vec[2] = dilations[0];
    dilations_vec[3] = dilations[1];
  }

  auto stream = dev_ctx.stream();

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
        .AddAttrs({{"strides", strides_vec}})
        .AddAttrs({{"pads", paddings}})
        .AddAttrs({{"dilations", dilations_vec}})
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
        .AddAttrs({{"strides", strides_vec}})
        .AddAttrs({{"pads", paddings}})
        .AddAttrs({{"dilations", dilations_vec}})
        .AddAttrs({{"groups", groups}})
        .AddAttrs({{"data_format", data_format}})
        .Run(stream);
  }
}

template <typename T, typename Context>
void Conv2DGradKernel(const Context& dev_ctx,
                      const phi::DenseTensor& input,
                      const phi::DenseTensor& filter,
                      const phi::DenseTensor& output_grad,
                      const std::vector<int>& strides_t,
                      const std::vector<int>& paddings_t,
                      const std::string& padding_algorithm,
                      const std::vector<int>& dilations_t,
                      int groups,
                      const std::string& data_format,
                      phi::DenseTensor* input_grad,
                      phi::DenseTensor* filter_grad) {
  if (FLAGS_npu_jit_compile) {
    aclSetCompileopt(ACL_OP_JIT_COMPILE, "disable");
  }

  auto strides = strides_t;
  auto paddings = paddings_t;
  auto dilations = dilations_t;
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
      &paddings, &dilations, padding_algorithm, in_data_dims, strides, ksize);

  DO_COMPATIBILITY(
      aclnnConvolutionBackward,
      (custom_kernel::AclopConv2DGradKernel<T, Context>(dev_ctx,
                                                        input,
                                                        filter,
                                                        output_grad,
                                                        strides,
                                                        paddings,
                                                        dilations,
                                                        groups,
                                                        data_format,
                                                        channel_last,
                                                        input_grad,
                                                        filter_grad)));

  if (paddings[0] != paddings[1] || paddings[2] != paddings[3]) {
    VLOG(2) << "Fallback to AclopConv2DGradKernel due to asymmetric padding : {"
            << paddings[0] << ", " << paddings[1] << ", " << paddings[2] << ", "
            << paddings[3] << "}";
    return custom_kernel::AclopConv2DGradKernel<T, Context>(dev_ctx,
                                                            input,
                                                            filter,
                                                            output_grad,
                                                            strides,
                                                            paddings,
                                                            dilations,
                                                            groups,
                                                            data_format,
                                                            channel_last,
                                                            input_grad,
                                                            filter_grad);
  }

  if (FLAGS_npu_storage_format) {
    VLOG(2)
        << "Fallback to AclopConv2DGradKernel since `FLAGS_npu_storage_format` "
        << "is ON.";
    return custom_kernel::AclopConv2DGradKernel<T, Context>(dev_ctx,
                                                            input,
                                                            filter,
                                                            output_grad,
                                                            strides,
                                                            paddings,
                                                            dilations,
                                                            groups,
                                                            data_format,
                                                            channel_last,
                                                            input_grad,
                                                            filter_grad);
  }

  if (channel_last) {
    VLOG(2) << "Fallback to AclopConv2DGradKernel since `data_format` is NHWC.";
    return custom_kernel::AclopConv2DGradKernel<T, Context>(dev_ctx,
                                                            input,
                                                            filter,
                                                            output_grad,
                                                            strides,
                                                            paddings,
                                                            dilations,
                                                            groups,
                                                            data_format,
                                                            channel_last,
                                                            input_grad,
                                                            filter_grad);
  }

  phi::DenseTensor input_tensor(input), output_grad_tensor(output_grad);

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
  std::vector<int64_t> stride_(strides.begin(), strides.end());
  std::vector<int64_t> padding_ = {paddings[0], paddings[2]};
  std::vector<int64_t> dilation_(dilations.begin(), dilations.end());
  bool transposed = false;
  std::vector<int64_t> output_padding = {0, 0};
  int64_t groups_ = groups;
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
               groups_,
               output_mask,
               cubeMathType,
               input_grad_tensor,
               filter_grad_tensor,
               bias_grad_tensor);

  if (FLAGS_npu_jit_compile) {
    aclSetCompileopt(ACL_OP_JIT_COMPILE, "enable");
  }
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(conv2d,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::Conv2dKernel,
                          float,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}

PD_REGISTER_PLUGIN_KERNEL(conv2d_grad,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::Conv2DGradKernel,
                          float,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}
