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
#include "kernels/funcs/gcu_kernel_funcs.h"

namespace custom_kernel {
static std::unordered_set<const void*> g_weights_nhwc;
template <typename T, typename Context>
extern void AddKernel(const Context& dev_ctx,
                      const phi::DenseTensor& x,
                      const phi::DenseTensor& y,
                      phi::DenseTensor* out);

template <typename T, typename Context>
void FusedConv2dAddActKernel(const Context& dev_ctx,
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
                             std::vector<phi::DenseTensor*> outputs) {
  PADDLE_GCU_KERNEL_TRACE("fused_conv2d_add_act");

  if (LaunchAOTKernel()) {
    dev_ctx.template Alloc<T>(output);

    phi::DenseTensor input_perm = input;
    phi::DenseTensor filter_perm = filter;
    phi::DenseTensor conv_out_perm = *output;
    phi::DenseTensor residual_perm;
    if (residual) {
      residual_perm = residual.get();
    }

    if (EnableTransposeOptimize()) {
      PADDLE_ENFORCE_EQ(data_format,
                        "NCHW",
                        phi::errors::InvalidArgument(
                            "Layout of kernel attr should be NCHW."));

      VLOG(6) << "Transpose debug, input:"
              << custom_kernel::TensorDetailsToString(input);
      VLOG(6) << "Transpose debug, filter:"
              << custom_kernel::TensorDetailsToString(filter);

      if (input.layout() == common::DataLayout::kNCHW) {
        input_perm = custom_kernel::NCHWTransToPdCustomNHWC(dev_ctx, input);
      }
      if (residual && residual_perm.layout() == common::DataLayout::kNCHW) {
        residual_perm =
            custom_kernel::NCHWTransToPdCustomNHWC(dev_ctx, residual_perm);
      }

      PdCustomNHWCRepresentAsAtenNHWC(input_perm);
      PdCustomNHWCRepresentAsAtenNHWC(filter_perm, true);
      PdCustomNHWCRepresentAsAtenNHWC(conv_out_perm, true);
      if (g_weights_nhwc.count(filter.data()) == 0) {
        auto filter_trans = NCHWTransToPdCustomNHWC(dev_ctx, filter);
        phi::DenseTensor* filter_ptr = const_cast<phi::DenseTensor*>(&filter);
        TensorCopy(dev_ctx, filter_trans, false, filter_ptr);
        g_weights_nhwc.emplace(filter.data());
        VLOG(6) << "Transpose debug, trans filter for fused_conv2d_add_act.";
      }
    } else {
      if (data_format == "NHWC") {
        OriginNHWCRepresentAsAtenNHWC(input_perm);
        OriginNHWCRepresentAsAtenNHWC(filter_perm);
        OriginNHWCRepresentAsAtenNHWC(conv_out_perm);
      }
    }

    std::vector<int64_t> strides_v = {strides.begin(), strides.end()};
    std::vector<int64_t> paddings_v = {paddings.begin(), paddings.end()};
    std::vector<int64_t> dilations_v = {dilations.begin(), dilations.end()};

    int ic = input_perm.dims().at(1);
    int oc = filter_perm.dims().at(0);
    bool depthwise_conv = (groups == ic) && (ic == oc);
    if (!depthwise_conv) {
      PADDLE_ENFORCE_EQ(groups,
                        1,
                        phi::errors::InvalidArgument(
                            "The groups must be 1, but got %d.", groups));

      int64_t groups_64 = groups;
      topsatenActivationMode_t act_mode = TOPSATEN_ACTIVATION_RELU;
      phi::Scalar coef(1.0f);
      if (activation == "identity") {
        act_mode = TOPSATEN_ACTIVATION_IDENTITY;
      } else if (activation == "relu") {
        act_mode = TOPSATEN_ACTIVATION_RELU;
      } else if (activation == "sigmoid") {
        act_mode = TOPSATEN_ACTIVATION_SIGMOID;
      } else if (activation == "swish") {
        act_mode = TOPSATEN_ACTIVATION_SWISH;
      } else {
        PADDLE_THROW(phi::errors::Unimplemented(
            "Unsupport activation string: %s.", activation));
      }

      if (!residual) {
        LAUNCH_TOPSATENOP(topsatenConvBiasActivation,
                          dev_ctx,
                          conv_out_perm,
                          input_perm,
                          filter_perm,
                          bias,
                          strides_v,
                          paddings_v,
                          dilations_v,
                          groups_64,
                          act_mode,
                          coef);
      } else {
        phi::Scalar alpha(1.0f);
        phi::Scalar beta(1.0f);
        if (EnableTransposeOptimize()) {
          PdCustomNHWCRepresentAsAtenNHWC(residual_perm);
        }
        LAUNCH_TOPSATENOP(topsatenConvScaledBiasActivation,
                          dev_ctx,
                          conv_out_perm,
                          input_perm,
                          filter_perm,
                          bias,
                          residual_perm,
                          strides_v,
                          paddings_v,
                          dilations_v,
                          groups_64,
                          act_mode,
                          coef,
                          alpha,
                          beta);
      }
    } else {
      VLOG(6) << "Conv2dDepthwiseBias, groups:" << groups;
      if (residual) {
        PADDLE_ENFORCE_EQ(residual->data<T>(),
                          nullptr,
                          phi::errors::InvalidArgument(
                              "The pointer of residual's data must be null."));
      }
      PADDLE_ENFORCE_EQ(activation,
                        "identity",
                        phi::errors::InvalidArgument(
                            "Conv2dDepthwiseBias not support activation now."));
      LAUNCH_TOPSATENOP(topsatenConvDepthwise2d,
                        dev_ctx,
                        conv_out_perm,
                        input_perm,
                        filter_perm,
                        bias,
                        strides_v,
                        paddings_v,
                        dilations_v);
    }

    if (EnableTransposeOptimize()) {
      AtenNHWCRepresentAsPdCustomNHWC(*output, true);
      VLOG(6) << "Transpose debug, conv_out_perm:"
              << custom_kernel::TensorDetailsToString(*output);
    } else {
      if (data_format == "NHWC") {
        AtenNHWCRepresentAsOriginNHWC(*output);
      }
    }

  } else {  // kernel impl base on JIT
    THROW_JIT_UNIMPLEMENTED();
  }
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(fused_conv2d_add_act,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::FusedConv2dAddActKernel,
                          float,
                          phi::dtype::float16) {}
