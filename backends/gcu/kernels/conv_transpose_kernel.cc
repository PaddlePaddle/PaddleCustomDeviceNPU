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

namespace custom_kernel {
static std::unordered_set<const void*> g_conv2d_transpose_weights_nhwc;
static std::unordered_set<const void*> g_conv3d_transpose_weights_nhwc;

template <typename T, typename Context>
void ConvTransposeRawKernel(const std::string& conv_type,
                            const Context& dev_ctx,
                            const phi::DenseTensor& x,
                            const phi::DenseTensor& filter,
                            const std::vector<int>& strides,
                            const std::vector<int>& paddings,
                            const std::vector<int>& output_padding,
                            const phi::IntArray& output_size,
                            const std::string& padding_algorithm,
                            int groups,
                            const std::vector<int>& dilations,
                            const std::string& data_format,
                            phi::DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);

  TensorNameMap input_names;
  input_names["Input"] = {"input"};
  input_names["Filter"] = {"filter"};

  TensorValueMap inputs;
  inputs["Input"] = {const_cast<phi::DenseTensor*>(&x)};
  inputs["Filter"] = {const_cast<phi::DenseTensor*>(&filter)};

  TensorNameMap output_names;
  output_names["Out"] = {"out"};

  TensorValueMap outputs;
  outputs["Out"] = {out};

  std::vector<int> output_size_list = GetIntList(output_size.GetData());

  GcuAttributeMap attrs;
  attrs["strides"] = strides;
  attrs["paddings"] = paddings;
  attrs["output_padding"] = output_padding;
  attrs["output_size"] = output_size_list;
  attrs["padding_algorithm"] = padding_algorithm;
  attrs["groups"] = groups;
  attrs["dilations"] = dilations;
  attrs["data_format"] = data_format;

  GcuRunner(
      input_names, inputs, output_names, outputs, attrs, conv_type, dev_ctx);
}

template <typename T, typename Context>
void ConvTransposeGradRawKernel(const std::string& conv_grad_type,
                                const Context& dev_ctx,
                                const phi::DenseTensor& x,
                                const phi::DenseTensor& filter,
                                const phi::DenseTensor& dout,
                                const std::vector<int>& strides,
                                const std::vector<int>& paddings,
                                const std::vector<int>& output_padding,
                                const phi::IntArray& output_size,
                                const std::string& padding_algorithm,
                                int groups,
                                const std::vector<int>& dilations,
                                const std::string& data_format,
                                phi::DenseTensor* dx,
                                phi::DenseTensor* dfilter) {
  dev_ctx.template Alloc<T>(dx);
  dev_ctx.template Alloc<T>(dfilter);

  TensorNameMap input_names;
  input_names["Input"] = {"input"};
  input_names["Filter"] = {"filter"};
  input_names[GradVarName("Output")] = {"dout"};

  TensorValueMap inputs;
  inputs["Input"] = {const_cast<phi::DenseTensor*>(&x)};
  inputs["Filter"] = {const_cast<phi::DenseTensor*>(&filter)};
  inputs[GradVarName("Output")] = {const_cast<phi::DenseTensor*>(&dout)};

  TensorNameMap output_names;
  output_names[GradVarName("Input")] = {"dx"};
  output_names[GradVarName("Filter")] = {"dfilter"};

  TensorValueMap outputs;
  outputs[GradVarName("Input")] = {dx};
  outputs[GradVarName("Filter")] = {dfilter};

  std::vector<int> output_size_list = GetIntList(output_size.GetData());

  GcuAttributeMap attrs;
  attrs["strides"] = strides;
  attrs["paddings"] = paddings;
  attrs["output_padding"] = output_padding;
  attrs["output_size"] = output_size_list;
  attrs["padding_algorithm"] = padding_algorithm;
  attrs["groups"] = groups;
  attrs["dilations"] = dilations;
  attrs["data_format"] = data_format;

  GcuRunner(input_names,
            inputs,
            output_names,
            outputs,
            attrs,
            conv_grad_type,
            dev_ctx);
}

template <typename T, typename Context>
void Conv2dTransposeBiasKernel(const Context& dev_ctx,
                               const phi::DenseTensor& x,
                               const phi::DenseTensor& filter,
                               const paddle::optional<phi::DenseTensor>& bias,
                               const std::vector<int>& strides,
                               const std::vector<int>& paddings,
                               const std::vector<int>& output_padding UNUSED,
                               const phi::IntArray& output_size UNUSED,
                               const std::string& padding_algorithm,
                               int groups,
                               const std::vector<int>& dilations,
                               const std::string& data_format UNUSED,
                               phi::DenseTensor* out) {
  PADDLE_GCU_KERNEL_TRACE("conv2d_transpose_bias");
  if (LaunchAOTKernel()) {
    dev_ctx.template Alloc<T>(out);
    phi::DenseTensor input_x = MaybeCreateOrTrans64To32bits(dev_ctx, x);
    phi::DenseTensor filter_x = MaybeCreateOrTrans64To32bits(dev_ctx, filter);
    phi::DenseTensor output =
        MaybeCreateOrTrans64To32bits(dev_ctx, *out, false);

    if (EnableTransposeOptimize()) {
      PADDLE_ENFORCE_EQ(data_format,
                        "NCHW",
                        phi::errors::InvalidArgument(
                            "Layout of kernel attr should be NCHW."));

      VLOG(6) << "Transpose debug, conv2d_transpose_bias input:"
              << custom_kernel::TensorDetailsToString(x);
      VLOG(6) << "Transpose debug, conv2d_transpose_bias filter:"
              << custom_kernel::TensorDetailsToString(filter);

      if (x.layout() == common::DataLayout::kNCHW) {
        input_x = custom_kernel::NCHWTransToPdCustomNHWC(dev_ctx, input_x);
      }

      PdCustomNHWCRepresentAsAtenNHWC(input_x);
      PdCustomNHWCRepresentAsAtenNHWC(filter_x, true);
      PdCustomNHWCRepresentAsAtenNHWC(output, true);
      if (g_conv2d_transpose_weights_nhwc.count(filter.data()) == 0) {
        auto filter_trans = NCHWTransToPdCustomNHWC(dev_ctx, filter);
        phi::DenseTensor* filter_ptr = const_cast<phi::DenseTensor*>(&filter);
        TensorCopy(dev_ctx, filter_trans, false, filter_ptr);
        g_conv2d_transpose_weights_nhwc.emplace(filter.data());
        VLOG(6) << "Transpose debug, trans filter for conv2d_transpose_bias.";
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
      auto meta = phi::DenseTensorMeta(x.dtype(),
                                       phi::make_ddim({filter_x.dims().at(1)}));
      input_bias = TensorZeros(dev_ctx, meta);
    }

    std::vector<int64_t> strides_v = {strides.begin(), strides.end()};
    std::vector<int64_t> paddings_v = {paddings.begin(), paddings.end()};
    // std::vector<int64_t> output_padding_v = {output_padding.begin(),
    //                                          output_padding.end()};
    std::vector<int64_t> dilations_v = {dilations.begin(), dilations.end()};

    int64_t groups_64 = groups;
    LAUNCH_TOPSATENOP(topsatenConvTranspose2d,
                      dev_ctx,
                      output,
                      input_x,
                      filter_x,
                      input_bias,
                      strides_v,
                      paddings_v,
                      paddings_v,
                      groups_64,
                      dilations_v);

    if (EnableTransposeOptimize()) {
      AtenNHWCRepresentAsPdCustomNHWC(output);
      AtenNHWCRepresentAsPdCustomNHWC(*out, true);
      VLOG(6) << "Transpose debug, conv2d_transpose_bias output:"
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
    ConvTransposeRawKernel<T, Context>("conv2d_transpose",
                                       dev_ctx,
                                       x,
                                       filter,
                                       strides,
                                       paddings,
                                       output_padding,
                                       output_size,
                                       padding_algorithm,
                                       groups,
                                       dilations,
                                       data_format,
                                       out);
  }
}

template <typename T, typename Context>
void Conv2dTransposeKernel(const Context& dev_ctx,
                           const phi::DenseTensor& x,
                           const phi::DenseTensor& filter,
                           const std::vector<int>& strides,
                           const std::vector<int>& paddings,
                           const std::vector<int>& output_padding UNUSED,
                           const phi::IntArray& output_size UNUSED,
                           const std::string& padding_algorithm,
                           int groups,
                           const std::vector<int>& dilations,
                           const std::string& data_format UNUSED,
                           phi::DenseTensor* out) {
  PADDLE_GCU_KERNEL_TRACE("conv2d_transpose");
  custom_kernel::Conv2dTransposeBiasKernel<T, Context>(
      dev_ctx,
      x,
      filter,
      paddle::optional<phi::DenseTensor>(),
      strides,
      paddings,
      output_padding,
      output_size,
      padding_algorithm,
      groups,
      dilations,
      data_format,
      out);
}

template <typename T, typename Context>
void Conv2dTransposeGradKernel(const Context& dev_ctx,
                               const phi::DenseTensor& x,
                               const phi::DenseTensor& filter,
                               const phi::DenseTensor& dout,
                               const std::vector<int>& strides,
                               const std::vector<int>& paddings,
                               const std::vector<int>& output_padding,
                               const phi::IntArray& output_size,
                               const std::string& padding_algorithm,
                               int groups,
                               const std::vector<int>& dilations,
                               const std::string& data_format,
                               phi::DenseTensor* dx,
                               phi::DenseTensor* dfilter) {
  PADDLE_GCU_KERNEL_TRACE("conv2d_transpose_grad");
  if (LaunchAOTKernel()) {
    THROW_AOT_UNIMPLEMENTED();
  } else {  // kernel impl base on JIT
    ConvTransposeGradRawKernel<T, Context>("conv2d_transpose_grad",
                                           dev_ctx,
                                           x,
                                           filter,
                                           dout,
                                           strides,
                                           paddings,
                                           output_padding,
                                           output_size,
                                           padding_algorithm,
                                           groups,
                                           dilations,
                                           data_format,
                                           dx,
                                           dfilter);
  }
}

template <typename T, typename Context>
void Conv3dTransposeKernel(const Context& dev_ctx,
                           const phi::DenseTensor& x,
                           const phi::DenseTensor& filter,
                           const std::vector<int>& strides,
                           const std::vector<int>& paddings,
                           const std::vector<int>& output_padding,
                           const std::vector<int>& output_size,
                           const std::string& padding_algorithm,
                           int groups,
                           const std::vector<int>& dilations,
                           const std::string& data_format,
                           phi::DenseTensor* out) {
  PADDLE_GCU_KERNEL_TRACE("conv3d_transpose");
  if (LaunchAOTKernel()) {
    // The aten operator library does not support conv3d_transpose
    THROW_AOT_UNIMPLEMENTED();
    // dev_ctx.template Alloc<T>(out);
    // phi::DenseTensor input_x = MaybeCreateOrTrans64To32bits(dev_ctx, x);
    // phi::DenseTensor filter_x = MaybeCreateOrTrans64To32bits(dev_ctx,
    // filter); phi::DenseTensor output =
    //     MaybeCreateOrTrans64To32bits(dev_ctx, *out, false);

    // if (EnableTransposeOptimize()) {
    //   PADDLE_ENFORCE_EQ(data_format, "NCHW",
    //                     phi::errors::InvalidArgument(
    //                         "Layout of kernel attr should be NCHW."));

    //   VLOG(6) << "Transpose debug, conv3d_transpose input:"
    //           << custom_kernel::TensorDetailsToString(x);
    //   VLOG(6) << "Transpose debug, conv3d_transpose filter:"
    //           << custom_kernel::TensorDetailsToString(filter);

    //   if (x.layout() == common::DataLayout::kNCHW) {
    //     input_x = custom_kernel::NCHWTransToPdCustomNHWC(dev_ctx, input_x);
    //   }

    //   PdCustomNHWCRepresentAsAtenNHWC(input_x);
    //   PdCustomNHWCRepresentAsAtenNHWC(filter_x, true);
    //   PdCustomNHWCRepresentAsAtenNHWC(output, true);
    //   if (g_conv3d_transpose_weights_nhwc.count(filter.data()) == 0) {
    //     auto filter_trans = NCHWTransToPdCustomNHWC(dev_ctx, filter);
    //     phi::DenseTensor* filter_ptr =
    //     const_cast<phi::DenseTensor*>(&filter); TensorCopy(dev_ctx,
    //     filter_trans, false, filter_ptr);
    //     g_conv3d_transpose_weights_nhwc.emplace(filter.data());
    //     VLOG(6) << "Transpose debug, trans filter for conv3d_transpose.";
    //   }
    // } else {
    //   if (data_format == "NHWC") {
    //     OriginNHWCRepresentAsAtenNHWC(input_x);
    //     // OriginNHWCRepresentAsAtenNHWC(filter_x);
    //     OriginNHWCRepresentAsAtenNHWC(output);
    //   }
    // }

    // auto meta = phi::DenseTensorMeta(x.dtype(),
    //                                  phi::make_ddim({filter_x.dims().at(1)}));
    // auto bias = TensorZeros(dev_ctx, meta);

    // std::vector<int64_t> strides_v = {strides.begin(), strides.end()};
    // std::vector<int64_t> paddings_v = {paddings.begin(), paddings.end()};
    // std::vector<int64_t> dilations_v = {dilations.begin(), dilations.end()};
    // std::vector<int64_t> output_padding_v = {output_padding.begin(),
    //                                          output_padding.end()};

    // int64_t groups_64 = groups;
    // bool transposed = true;

    // LAUNCH_TOPSATENOP(topsatenConvolution, dev_ctx, output, input_x,
    // filter_x,
    //                   bias, strides_v, paddings_v, dilations_v, transposed,
    //                   output_padding_v, groups_64);

    // if (EnableTransposeOptimize()) {
    //   AtenNHWCRepresentAsPdCustomNHWC(output);
    //   AtenNHWCRepresentAsPdCustomNHWC(*out, true);
    //   VLOG(6) << "Transpose debug, conv3d_transpose output:"
    //           << custom_kernel::TensorDetailsToString(*out);
    // } else {
    //   if (data_format == "NHWC") {
    //     AtenNHWCRepresentAsOriginNHWC(output);
    //   }
    // }

    // MaybeTransResult(dev_ctx, output, out);

  } else {  // kernel impl base on JIT
    ConvTransposeRawKernel<T, Context>("conv3d_transpose",
                                       dev_ctx,
                                       x,
                                       filter,
                                       strides,
                                       paddings,
                                       output_padding,
                                       output_size,
                                       padding_algorithm,
                                       groups,
                                       dilations,
                                       data_format,
                                       out);
  }
}

template <typename T, typename Context>
void Conv3dTransposeGradKernel(const Context& dev_ctx,
                               const phi::DenseTensor& x,
                               const phi::DenseTensor& filter,
                               const phi::DenseTensor& dout,
                               const std::vector<int>& strides,
                               const std::vector<int>& paddings,
                               const std::vector<int>& output_padding,
                               const std::vector<int>& output_size,
                               const std::string& padding_algorithm,
                               int groups,
                               const std::vector<int>& dilations,
                               const std::string& data_format,
                               phi::DenseTensor* dx,
                               phi::DenseTensor* dfilter) {
  PADDLE_GCU_KERNEL_TRACE("conv3d_transpose_grad");
  if (LaunchAOTKernel()) {
    THROW_AOT_UNIMPLEMENTED();
  } else {  // kernel impl base on JIT
    ConvTransposeGradRawKernel<T, Context>("conv3d_transpose_grad",
                                           dev_ctx,
                                           x,
                                           filter,
                                           dout,
                                           strides,
                                           paddings,
                                           output_padding,
                                           output_size,
                                           padding_algorithm,
                                           groups,
                                           dilations,
                                           data_format,
                                           dx,
                                           dfilter);
  }
}
}  // namespace custom_kernel
PD_REGISTER_PLUGIN_KERNEL(conv2d_transpose_bias,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::Conv2dTransposeBiasKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(conv2d_transpose,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::Conv2dTransposeKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(conv2d_transpose_grad,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::Conv2dTransposeGradKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(conv3d_transpose,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::Conv3dTransposeKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(conv3d_transpose_grad,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::Conv3dTransposeGradKernel,
                          float,
                          phi::dtype::float16) {}
