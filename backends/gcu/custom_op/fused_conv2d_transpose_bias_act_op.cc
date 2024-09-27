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
static std::unordered_set<const void*> g_conv2d_transpose_bias_act_weights_nhwc;

void Conv2dTransposeBiasActKernel(const phi::CustomContext& dev_ctx,
                                  const phi::DenseTensor& x,
                                  const phi::DenseTensor& filter,
                                  const phi::DenseTensor& bias,
                                  const std::vector<int>& strides,
                                  const std::vector<int>& paddings,
                                  const std::vector<int>& output_padding,
                                  const phi::IntArray& output_size,
                                  const std::string& padding_algorithm,
                                  int groups,
                                  const std::vector<int>& dilations,
                                  const std::string& data_format,
                                  const std::string& activation,
                                  phi::DenseTensor* out) {
  // dev_ctx->Alloc(out, out->dtype());
  phi::DenseTensor input_x = MaybeCreateOrTrans64To32bits(dev_ctx, x);
  phi::DenseTensor filter_x = MaybeCreateOrTrans64To32bits(dev_ctx, filter);
  phi::DenseTensor output = MaybeCreateOrTrans64To32bits(dev_ctx, *out, false);

  if (EnableTransposeOptimize()) {
    PADDLE_ENFORCE_EQ(
        data_format,
        "NCHW",
        phi::errors::InvalidArgument("Layout of kernel attr should be NCHW."));

    VLOG(6) << "Transpose debug, conv2d_transpose_bias_act input:"
            << custom_kernel::TensorDetailsToString(x);
    VLOG(6) << "Transpose debug, conv2d_transpose_bias_act filter:"
            << custom_kernel::TensorDetailsToString(filter);

    if (x.layout() == common::DataLayout::kNCHW) {
      input_x = custom_kernel::NCHWTransToPdCustomNHWC(dev_ctx, input_x);
    }

    PdCustomNHWCRepresentAsAtenNHWC(input_x);
    PdCustomNHWCRepresentAsAtenNHWC(filter_x, true);
    PdCustomNHWCRepresentAsAtenNHWC(output, true);
    if (g_conv2d_transpose_bias_act_weights_nhwc.count(filter.data()) == 0) {
      auto filter_trans = NCHWTransToPdCustomNHWC(dev_ctx, filter);
      phi::DenseTensor* filter_ptr = const_cast<phi::DenseTensor*>(&filter);
      TensorCopy(dev_ctx, filter_trans, false, filter_ptr);
      g_conv2d_transpose_bias_act_weights_nhwc.emplace(filter.data());
      VLOG(6) << "Transpose debug, trans filter for conv2d_transpose_bias_act.";
    }
  } else {
    if (data_format == "NHWC") {
      OriginNHWCRepresentAsAtenNHWC(input_x);
      // OriginNHWCRepresentAsAtenNHWC(filter_x);
      OriginNHWCRepresentAsAtenNHWC(output);
    }
  }

  std::vector<int64_t> strides_v = {strides.begin(), strides.end()};
  std::vector<int64_t> paddings_v = {paddings.begin(), paddings.end()};
  std::vector<int64_t> output_padding_v = {output_padding.begin(),
                                           output_padding.end()};
  if (output_padding_v.empty()) {
    output_padding_v = paddings_v;
  }
  std::vector<int64_t> dilations_v = {dilations.begin(), dilations.end()};

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
        "Unsupported activation string: %s.", activation));
  }

  LAUNCH_TOPSATENOP(topsatenConvTransposeActivation,
                    dev_ctx,
                    output,
                    input_x,
                    filter_x,
                    bias,
                    strides_v,
                    paddings_v,
                    output_padding_v,
                    groups_64,
                    dilations_v,
                    act_mode,
                    coef);

  if (EnableTransposeOptimize()) {
    AtenNHWCRepresentAsPdCustomNHWC(output);
    AtenNHWCRepresentAsPdCustomNHWC(*out, true);
    VLOG(6) << "Transpose debug, conv2d_transpose_bias_act output:"
            << custom_kernel::TensorDetailsToString(*out);
  } else {
    if (data_format == "NHWC") {
      AtenNHWCRepresentAsOriginNHWC(output);
    }
  }

  MaybeTransResult(dev_ctx, output, out);
}
}  // namespace custom_kernel

std::vector<std::vector<int64_t>> FusedConv2dTransposeBiasActInferShape(
    const std::vector<int64_t>& input_shape,
    const std::vector<int64_t>& filter_shape,
    const std::vector<int64_t>& bias_shape,
    const std::vector<int>& strides,
    const std::vector<int>& paddings,
    const std::vector<int>& output_padding,
    const std::vector<int>& output_size,
    const std::string& padding_algorithm,
    int groups,
    const std::vector<int>& dilations,
    const std::string& data_format,
    const std::string& activation) {
  return {{-1}};
}

std::vector<paddle::DataType> FusedConv2dTransposeBiasActInferDtype(
    const paddle::DataType& input_dtype,
    const paddle::DataType& filter_dtype,
    const paddle::DataType& bias_dtype,
    const std::vector<int>& strides,
    const std::vector<int>& paddings,
    const std::vector<int>& output_padding,
    const std::vector<int>& output_size,
    const std::string& padding_algorithm,
    int groups,
    const std::vector<int>& dilations,
    const std::string& data_format,
    const std::string& activation) {
  return {input_dtype};
}

std::vector<paddle::Tensor> FusedConv2dTransposeBiasAct(
    const paddle::Tensor& input,
    const paddle::Tensor& filter,
    const paddle::Tensor& bias,
    const std::vector<int>& strides,
    const std::vector<int>& paddings,
    const std::vector<int>& output_padding,
    const std::vector<int>& output_size,
    const std::string& padding_algorithm,
    int groups,
    const std::vector<int>& dilations,
    const std::string& data_format,
    const std::string& activation) {
  PADDLE_GCU_KERNEL_TRACE("fused_conv2d_transpose_bias_act");
  VLOG(6) << "[CUSTOM_KERNEL] Custom Operator: fused_conv2d_transpose_bias_act";

  auto dev_ctx = static_cast<const phi::CustomContext*>(
      paddle::experimental::DeviceContextPool::Instance().Get(input.place()));

  auto input_tensor = static_cast<const phi::DenseTensor*>(input.impl().get());
  auto filter_tensor =
      static_cast<const phi::DenseTensor*>(filter.impl().get());
  auto bias_tensor = static_cast<const phi::DenseTensor*>(bias.impl().get());

  std::shared_ptr<phi::DenseTensor> output =
      std::make_shared<phi::DenseTensor>();
  phi::MetaTensor meta_out(*output);
  phi::Conv2dTransposeInferMeta(*input_tensor,
                                *filter_tensor,
                                strides,
                                paddings,
                                output_padding,
                                output_size,
                                padding_algorithm,
                                groups,
                                dilations,
                                data_format,
                                &meta_out);
  output->Resize(meta_out.dims());
  dev_ctx->Alloc(output.get(), input.dtype());

  if (input.dtype() == phi::DataType::FLOAT16 ||
      input.dtype() == phi::DataType::FLOAT32) {
    custom_kernel::Conv2dTransposeBiasActKernel(*dev_ctx,
                                                *input_tensor,
                                                *filter_tensor,
                                                *bias_tensor,
                                                strides,
                                                paddings,
                                                output_padding,
                                                output_size,
                                                padding_algorithm,
                                                groups,
                                                dilations,
                                                data_format,
                                                activation,
                                                output.get());
  } else {
    PADDLE_THROW(phi::errors::Unimplemented(
        "Unsupported data type: %s",
        phi::DataTypeToString(input.dtype()).c_str()));
  }

  return {paddle::Tensor(output)};
}

PD_BUILD_OP(fused_conv2d_transpose_bias_act)
    .Inputs({"Input", "Filter", "Bias"})
    .Outputs({"Output"})
    .Attrs({"strides: std::vector<int>",
            "paddings: std::vector<int>",
            "output_padding: std::vector<int>",
            "output_size: std::vector<int>",
            "padding_algorithm: std::string",
            "groups: int",
            "dilations: std::vector<int>",
            "data_format: std::string",
            "activation: std::string"})
    .SetKernelFn(PD_KERNEL(FusedConv2dTransposeBiasAct))
    .SetInferShapeFn(PD_INFER_SHAPE(FusedConv2dTransposeBiasActInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(FusedConv2dTransposeBiasActInferDtype));
