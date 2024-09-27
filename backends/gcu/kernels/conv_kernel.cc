// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
#include "kernels/funcs/gcu_kernel_funcs.h"
#include "paddle/phi/kernels/cpu/conv_util.h"

namespace custom_kernel {
static std::unordered_set<const void*> g_conv2d_weights_nhwc;
static std::unordered_set<const void*> g_conv3d_weights_nhwc;
static std::unordered_set<const void*> g_depthwise_conv2d_weights_nhwc;

template <typename T, typename Context>
void GcuConvKernel(const Context& dev_ctx,
                   const phi::DenseTensor& input,
                   const phi::DenseTensor& filter,
                   const std::vector<int>& strides,
                   const std::vector<int>& paddings,
                   const std::string& padding_algorithm,
                   int groups,
                   const std::vector<int>& dilations,
                   const std::string& data_format,
                   phi::DenseTensor* out,
                   const std::string& op_type) {
  dev_ctx.template Alloc<T>(out);

  TensorNameMap input_names;
  input_names["Input"] = {"input"};
  input_names["Filter"] = {"filter"};

  TensorValueMap inputs;
  inputs["Input"] = {const_cast<DenseTensor*>(&input)};
  inputs["Filter"] = {const_cast<DenseTensor*>(&filter)};

  TensorNameMap output_names;
  output_names["Output"] = {"out"};

  TensorValueMap outputs;
  outputs["Output"] = {out};

  GcuAttributeMap attrs;
  attrs["strides"] = strides;
  attrs["paddings"] = paddings;
  attrs["padding_algorithm"] = padding_algorithm;
  attrs["dilations"] = dilations;
  attrs["groups"] = groups;
  attrs["data_format"] = data_format;

  GcuRunner(
      input_names, inputs, output_names, outputs, attrs, op_type, dev_ctx);
}

template <typename T, typename Context>
void GcuConvGradKernel(const Context& dev_ctx,
                       const phi::DenseTensor& input,
                       const phi::DenseTensor& filter,
                       const phi::DenseTensor& out_grad,
                       const std::vector<int>& strides,
                       const std::vector<int>& paddings,
                       const std::string& padding_algorithm,
                       int groups,
                       const std::vector<int>& dilations,
                       const std::string& data_format,
                       phi::DenseTensor* input_grad,
                       phi::DenseTensor* filter_grad,
                       const std::string& op_type) {
  TensorNameMap input_names;
  input_names["Input"] = {"input"};
  input_names["Filter"] = {"filter"};
  input_names[GradVarName("Output")] = {"out_grad"};

  TensorValueMap inputs;
  inputs["Input"] = {const_cast<DenseTensor*>(&input)};
  inputs["Filter"] = {const_cast<DenseTensor*>(&filter)};
  inputs[GradVarName("Output")] = {const_cast<DenseTensor*>(&out_grad)};

  TensorNameMap output_names;
  TensorValueMap outputs;
  if (input_grad) {
    dev_ctx.template Alloc<T>(input_grad);
    output_names[GradVarName("Input")] = {"input_grad"};
    outputs[GradVarName("Input")] = {input_grad};
  }
  if (filter_grad) {
    dev_ctx.template Alloc<T>(filter_grad);
    output_names[GradVarName("Filter")] = {"filter_grad"};
    outputs[GradVarName("Filter")] = {filter_grad};
  }

  GcuAttributeMap attrs;
  attrs["strides"] = strides;
  attrs["paddings"] = paddings;
  attrs["padding_algorithm"] = padding_algorithm;
  attrs["dilations"] = dilations;
  attrs["groups"] = groups;
  attrs["data_format"] = data_format;

  GcuRunner(
      input_names, inputs, output_names, outputs, attrs, op_type, dev_ctx);
}

template <typename T, typename Context>
void Conv2dBiasKernel(const Context& dev_ctx,
                      const DenseTensor& input,
                      const DenseTensor& filter,
                      const paddle::optional<phi::DenseTensor>& bias,
                      const std::vector<int>& strides,
                      const std::vector<int>& paddings,
                      const std::string& padding_algorithm,
                      const std::vector<int>& dilations,
                      int groups,
                      const std::string& data_format,
                      DenseTensor* out) {
  PADDLE_GCU_KERNEL_TRACE("conv2d_bias");
  if (LaunchAOTKernel()) {
    phi::DenseTensor input_x = MaybeCreateOrTrans64To32bits(dev_ctx, input);
    // if (data_format == "NHWC") {
    //   input_x = custom_kernel::Transpose(dev_ctx, input, {0, 3, 1, 2});
    // }
    dev_ctx.template Alloc<T>(out);
    phi::DenseTensor filter_x = MaybeCreateOrTrans64To32bits(dev_ctx, filter);
    phi::DenseTensor output =
        MaybeCreateOrTrans64To32bits(dev_ctx, *out, false);

    // phi::DenseTensor input_x = input;
    // phi::DenseTensor filter_x = filter;
    // phi::DenseTensor output = *out;
    // update paddings and dilations according to padding_algorithm
    auto input_dims = input.dims();
    auto filter_dims = filter.dims();
    std::vector<int> paddings_vec = paddings;
    std::vector<int> dilations_vec = dilations;
    phi::DDim in_data_dims =
        common::slice_ddim(input_dims, 2, input_dims.size());
    phi::DDim filter_data_dims =
        common::slice_ddim(filter_dims, 2, filter_dims.size());
    std::vector<int> ksize = common::vectorize<int>(filter_data_dims);
    phi::UpdatePaddingAndDilation(&paddings_vec,
                                  &dilations_vec,
                                  padding_algorithm,
                                  in_data_dims,
                                  strides,
                                  ksize);
    PADDLE_ENFORCE_EQ(
        paddings_vec.size(),
        4,
        phi::errors::Fatal("Paddings size should be the same as 4 after update "
                           "padding and dilation process."));
    if (EnableTransposeOptimize()) {
      PADDLE_ENFORCE_EQ(data_format,
                        "NCHW",
                        phi::errors::InvalidArgument(
                            "Layout of kernel attr should be NCHW."));

      VLOG(6) << "Transpose debug, conv2d input:"
              << custom_kernel::TensorDetailsToString(input);
      VLOG(6) << "Transpose debug, conv2d filter:"
              << custom_kernel::TensorDetailsToString(filter);

      if (input.layout() == common::DataLayout::kNCHW) {
        input_x = custom_kernel::NCHWTransToPdCustomNHWC(dev_ctx, input_x);
      }

      PdCustomNHWCRepresentAsAtenNHWC(input_x);
      PdCustomNHWCRepresentAsAtenNHWC(filter_x, true);
      PdCustomNHWCRepresentAsAtenNHWC(output, true);
      if (g_conv2d_weights_nhwc.count(filter.data()) == 0) {
        auto filter_trans = NCHWTransToPdCustomNHWC(dev_ctx, filter);
        phi::DenseTensor* filter_ptr = const_cast<phi::DenseTensor*>(&filter);
        TensorCopy(dev_ctx, filter_trans, false, filter_ptr);
        g_conv2d_weights_nhwc.emplace(filter.data());
        VLOG(6) << "Transpose debug, trans filter for conv2d.";
      }
    } else {
      if (data_format == "NHWC") {
        OriginNHWCRepresentAsAtenNHWC(input_x);
        // OriginNHWCRepresentAsAtenNHWC(filter_x);
        OriginNHWCRepresentAsAtenNHWC(output);
      }
    }

    phi::DenseTensor input_bias;
    if (bias) {
      input_bias = bias.get();
    } else {
      auto meta = phi::DenseTensorMeta(input.dtype(),
                                       phi::make_ddim({filter_x.dims().at(0)}));
      input_bias = TensorZeros(dev_ctx, meta);
    }

    std::vector<int64_t> strides_v = {strides.begin(), strides.end()};
    std::vector<int64_t> paddings_v = {paddings_vec.begin(),
                                       paddings_vec.end()};
    std::vector<int64_t> dilations_v = {dilations_vec.begin(),
                                        dilations_vec.end()};

    int64_t groups_64 = groups;
    LAUNCH_TOPSATENOP(topsatenConv2d,
                      dev_ctx,
                      output,
                      input_x,
                      filter_x,
                      input_bias,
                      strides_v,
                      paddings_v,
                      dilations_v,
                      groups_64);

    if (EnableTransposeOptimize()) {
      AtenNHWCRepresentAsPdCustomNHWC(output);
      AtenNHWCRepresentAsPdCustomNHWC(*out, true);
      VLOG(6) << "Transpose debug, conv2d output:"
              << custom_kernel::TensorDetailsToString(*out);
    } else {
      if (data_format == "NHWC") {
        AtenNHWCRepresentAsOriginNHWC(output);
      }
    }

    MaybeTransResult(dev_ctx, output, out);

  } else {  // kernel impl base on JIT
    if (bias) {
      THROW_JIT_UNIMPLEMENTED();
    }
    GcuConvKernel<T, Context>(dev_ctx,
                              input,
                              filter,
                              strides,
                              paddings,
                              padding_algorithm,
                              groups,
                              dilations,
                              data_format,
                              out,
                              "conv2d");
  }
}

template <typename T, typename Context>
void Conv2dKernel(const Context& dev_ctx,
                  const DenseTensor& input,
                  const DenseTensor& filter,
                  const std::vector<int>& strides,
                  const std::vector<int>& paddings,
                  const std::string& padding_algorithm,
                  const std::vector<int>& dilations,
                  int groups,
                  const std::string& data_format,
                  DenseTensor* out) {
  PADDLE_GCU_KERNEL_TRACE("conv2d");
  custom_kernel::Conv2dBiasKernel<T, Context>(
      dev_ctx,
      input,
      filter,
      paddle::optional<phi::DenseTensor>(),
      strides,
      paddings,
      padding_algorithm,
      dilations,
      groups,
      data_format,
      out);
}

template <typename T, typename Context>
void Conv2DGradKernel(const Context& dev_ctx,
                      const phi::DenseTensor& input,
                      const phi::DenseTensor& filter,
                      const phi::DenseTensor& output_grad,
                      const std::vector<int>& strides,
                      const std::vector<int>& paddings,
                      const std::string& padding_algorithm,
                      const std::vector<int>& dilations,
                      int groups,
                      const std::string& data_format,
                      phi::DenseTensor* input_grad,
                      phi::DenseTensor* filter_grad) {
  PADDLE_GCU_KERNEL_TRACE("conv2d_grad");
  if (LaunchAOTKernel()) {
    THROW_AOT_UNIMPLEMENTED();
  } else {  // kernel impl base on JIT
    GcuConvGradKernel<T, Context>(dev_ctx,
                                  input,
                                  filter,
                                  output_grad,
                                  strides,
                                  paddings,
                                  padding_algorithm,
                                  groups,
                                  dilations,
                                  data_format,
                                  input_grad,
                                  filter_grad,
                                  "conv2d_grad");
  }
}

template <typename T, typename Context>
void DepthwiseConv2dKernel(const Context& dev_ctx,
                           const phi::DenseTensor& input,
                           const phi::DenseTensor& filter,
                           const std::vector<int>& strides,
                           const std::vector<int>& paddings,
                           const std::string& padding_algorithm,
                           int groups,
                           const std::vector<int>& dilations,
                           const std::string& data_format,
                           phi::DenseTensor* out) {
  PADDLE_GCU_KERNEL_TRACE("depthwise_conv2d");
  if (LaunchAOTKernel()) {
    dev_ctx.template Alloc<T>(out);
    // phi::DenseTensor input_x = MaybeCreateOrTrans64To32bits(dev_ctx, input);
    // if (data_format == "NHWC") {
    //   input_x = custom_kernel::Transpose(dev_ctx, input, {0, 3, 1, 2});
    // }
    // phi::DenseTensor filter_x = MaybeCreateOrTrans64To32bits(dev_ctx,
    // filter); phi::DenseTensor output =
    //     MaybeCreateOrTrans64To32bits(dev_ctx, *out, false);

    phi::DenseTensor input_x = input;
    phi::DenseTensor filter_x = filter;
    phi::DenseTensor output = *out;

    if (EnableTransposeOptimize()) {
      PADDLE_ENFORCE_EQ(data_format,
                        "NCHW",
                        phi::errors::InvalidArgument(
                            "Layout of kernel attr should be NCHW."));

      VLOG(6) << "Transpose debug, depthwise_conv2d input:"
              << custom_kernel::TensorDetailsToString(input);
      VLOG(6) << "Transpose debug, depthwise_conv2d filter:"
              << custom_kernel::TensorDetailsToString(filter);

      if (input.layout() == common::DataLayout::kNCHW) {
        input_x = custom_kernel::NCHWTransToPdCustomNHWC(dev_ctx, input);
      }

      PdCustomNHWCRepresentAsAtenNHWC(input_x);
      PdCustomNHWCRepresentAsAtenNHWC(filter_x, true);
      PdCustomNHWCRepresentAsAtenNHWC(output, true);
      if (g_depthwise_conv2d_weights_nhwc.count(filter.data()) == 0) {
        auto filter_trans = NCHWTransToPdCustomNHWC(dev_ctx, filter);
        phi::DenseTensor* filter_ptr = const_cast<phi::DenseTensor*>(&filter);
        TensorCopy(dev_ctx, filter_trans, false, filter_ptr);
        g_depthwise_conv2d_weights_nhwc.emplace(filter.data());
        VLOG(6) << "Transpose debug, trans filter for depthwise_conv2d.";
      }
    } else {
      if (data_format == "NHWC") {
        OriginNHWCRepresentAsAtenNHWC(input_x);
        // OriginNHWCRepresentAsAtenNHWC(filter_x);
        OriginNHWCRepresentAsAtenNHWC(output);
      }
    }

    auto meta = phi::DenseTensorMeta(input.dtype(),
                                     phi::make_ddim({filter.dims().at(0)}));
    auto bias = TensorZeros(dev_ctx, meta);

    std::vector<int64_t> strides_v = {strides.begin(), strides.end()};
    std::vector<int64_t> paddings_v = {paddings.begin(), paddings.end()};
    std::vector<int64_t> dilations_v = {dilations.begin(), dilations.end()};

    LAUNCH_TOPSATENOP(topsatenConvDepthwise2d,
                      dev_ctx,
                      output,
                      input_x,
                      filter_x,
                      bias,
                      strides_v,
                      paddings_v,
                      dilations_v);
    // MaybeTransResult(dev_ctx, output, out);
    if (EnableTransposeOptimize()) {
      AtenNHWCRepresentAsPdCustomNHWC(*out, true);
      VLOG(6) << "Transpose debug, depthwise_conv2d output:"
              << custom_kernel::TensorDetailsToString(*out);
    } else {
      if (data_format == "NHWC") {
        AtenNHWCRepresentAsOriginNHWC(output);
      }
    }
  } else {  // kernel impl base on JIT
    GcuConvKernel<T, Context>(dev_ctx,
                              input,
                              filter,
                              strides,
                              paddings,
                              padding_algorithm,
                              groups,
                              dilations,
                              data_format,
                              out,
                              "depthwise_conv2d");
  }
}

template <typename T, typename Context>
void DepthwiseConv2dGradKernel(const Context& dev_ctx,
                               const phi::DenseTensor& input,
                               const phi::DenseTensor& filter,
                               const phi::DenseTensor& out_grad,
                               const std::vector<int>& strides,
                               const std::vector<int>& paddings,
                               const std::string& padding_algorithm,
                               int groups,
                               const std::vector<int>& dilations,
                               const std::string& data_format,
                               phi::DenseTensor* input_grad,
                               phi::DenseTensor* filter_grad) {
  PADDLE_GCU_KERNEL_TRACE("depthwise_conv2d_grad");
  if (LaunchAOTKernel()) {
    THROW_AOT_UNIMPLEMENTED();
  } else {  // kernel impl base on JIT
    GcuConvGradKernel<T, Context>(dev_ctx,
                                  input,
                                  filter,
                                  out_grad,
                                  strides,
                                  paddings,
                                  padding_algorithm,
                                  groups,
                                  dilations,
                                  data_format,
                                  input_grad,
                                  filter_grad,
                                  "depthwise_conv2d_grad");
  }
}

template <typename T, typename Context>
void Conv3dKernel(const Context& dev_ctx,
                  const phi::DenseTensor& input,
                  const phi::DenseTensor& filter,
                  const std::vector<int>& strides,
                  const std::vector<int>& paddings,
                  const std::string& padding_algorithm,
                  int groups,
                  const std::vector<int>& dilations,
                  const std::string& data_format,
                  phi::DenseTensor* out) {
  PADDLE_GCU_KERNEL_TRACE("conv3d");
  if (LaunchAOTKernel()) {
    dev_ctx.template Alloc<T>(out);
    phi::DenseTensor input_x = MaybeCreateOrTrans64To32bits(dev_ctx, input);
    phi::DenseTensor filter_x = MaybeCreateOrTrans64To32bits(dev_ctx, filter);
    phi::DenseTensor output =
        MaybeCreateOrTrans64To32bits(dev_ctx, *out, false);

    if (EnableTransposeOptimize()) {
      PADDLE_ENFORCE_EQ(data_format,
                        "NCHW",
                        phi::errors::InvalidArgument(
                            "Layout of kernel attr should be NCHW."));

      VLOG(6) << "Transpose debug, conv3d input:"
              << custom_kernel::TensorDetailsToString(input);
      VLOG(6) << "Transpose debug, conv3d filter:"
              << custom_kernel::TensorDetailsToString(filter);

      if (input.layout() == common::DataLayout::kNCHW) {
        input_x = custom_kernel::NCHWTransToPdCustomNHWC(dev_ctx, input_x);
      }

      PdCustomNHWCRepresentAsAtenNHWC(input_x);
      PdCustomNHWCRepresentAsAtenNHWC(filter_x, true);
      PdCustomNHWCRepresentAsAtenNHWC(output, true);
      if (g_conv3d_weights_nhwc.count(filter.data()) == 0) {
        auto filter_trans = NCHWTransToPdCustomNHWC(dev_ctx, filter);
        phi::DenseTensor* filter_ptr = const_cast<phi::DenseTensor*>(&filter);
        TensorCopy(dev_ctx, filter_trans, false, filter_ptr);
        g_conv3d_weights_nhwc.emplace(filter.data());
        VLOG(6) << "Transpose debug, trans filter for conv3d.";
      }
    } else {
      if (data_format == "NHWC") {
        OriginNHWCRepresentAsAtenNHWC(input_x);
        // OriginNHWCRepresentAsAtenNHWC(filter_x);
        OriginNHWCRepresentAsAtenNHWC(output);
      }
    }

    auto meta = phi::DenseTensorMeta(input.dtype(),
                                     phi::make_ddim({filter_x.dims().at(0)}));
    auto bias = TensorZeros(dev_ctx, meta);

    std::vector<int64_t> strides_v = {strides.begin(), strides.end()};
    std::vector<int64_t> paddings_v = {paddings.begin(), paddings.end()};
    std::vector<int64_t> dilations_v = {dilations.begin(), dilations.end()};
    std::vector<int64_t> output_padding_v = {0, 0, 0};

    int64_t groups_64 = groups;
    bool transposed = false;

    LAUNCH_TOPSATENOP(topsatenConvolution,
                      dev_ctx,
                      output,
                      input_x,
                      filter_x,
                      bias,
                      strides_v,
                      paddings_v,
                      dilations_v,
                      transposed,
                      output_padding_v,
                      groups_64);

    if (EnableTransposeOptimize()) {
      AtenNHWCRepresentAsPdCustomNHWC(output);
      AtenNHWCRepresentAsPdCustomNHWC(*out, true);
      VLOG(6) << "Transpose debug, conv3d output:"
              << custom_kernel::TensorDetailsToString(*out);
    } else {
      if (data_format == "NHWC") {
        AtenNHWCRepresentAsOriginNHWC(output);
      }
    }

    MaybeTransResult(dev_ctx, output, out);

  } else {  // kernel impl base on JIT
    GcuConvKernel<T, Context>(dev_ctx,
                              input,
                              filter,
                              strides,
                              paddings,
                              padding_algorithm,
                              groups,
                              dilations,
                              data_format,
                              out,
                              "conv3d");
  }
}

template <typename T, typename Context>
void Conv3dGradKernel(const Context& dev_ctx,
                      const phi::DenseTensor& input,
                      const phi::DenseTensor& filter,
                      const phi::DenseTensor& out_grad,
                      const std::vector<int>& strides,
                      const std::vector<int>& paddings,
                      const std::string& padding_algorithm,
                      int groups,
                      const std::vector<int>& dilations,
                      const std::string& data_format,
                      phi::DenseTensor* input_grad,
                      phi::DenseTensor* filter_grad) {
  PADDLE_GCU_KERNEL_TRACE("conv3d_grad");
  if (LaunchAOTKernel()) {
    THROW_AOT_UNIMPLEMENTED();
  } else {  // kernel impl base on JIT
    GcuConvGradKernel<T, Context>(dev_ctx,
                                  input,
                                  filter,
                                  out_grad,
                                  strides,
                                  paddings,
                                  padding_algorithm,
                                  groups,
                                  dilations,
                                  data_format,
                                  input_grad,
                                  filter_grad,
                                  "conv3d_grad");
  }
}
}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(depthwise_conv2d,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::DepthwiseConv2dKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(depthwise_conv2d_grad,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::DepthwiseConv2dGradKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(conv3d,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::Conv3dKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(conv3d_grad,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::Conv3dGradKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(conv2d,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::Conv2dKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(conv2d_grad,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::Conv2DGradKernel,
                          float,
                          phi::dtype::float16) {}
