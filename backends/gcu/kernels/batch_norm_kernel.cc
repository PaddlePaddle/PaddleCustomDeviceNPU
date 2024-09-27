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

template <typename T, typename Context>
void BatchNormKernel(const Context& dev_ctx,
                     const phi::DenseTensor& x,
                     const phi::DenseTensor& running_mean,
                     const phi::DenseTensor& running_var,
                     const paddle::optional<phi::DenseTensor>& scale,
                     const paddle::optional<phi::DenseTensor>& bias,
                     bool is_test,
                     float momentum,
                     float epsilon,
                     const std::string& data_layout_str,
                     bool use_global_stats,
                     bool trainable_stats,
                     phi::DenseTensor* y,
                     phi::DenseTensor* mean_out,
                     phi::DenseTensor* variance_out,
                     phi::DenseTensor* saved_mean,
                     phi::DenseTensor* saved_variance,
                     phi::DenseTensor* reserve_space) {
  PADDLE_GCU_KERNEL_TRACE("batch_norm");
  PADDLE_ENFORCE_EQ(data_layout_str == "NCHW" || data_layout_str == "NHWC",
                    true,
                    phi::errors::InvalidArgument(
                        "The 'data_layout' attribute must be NCHW or NHWC. "
                        "But recevived 'data_layout' is [%s].",
                        data_layout_str));
  const auto& x_dims = x.dims();
  PADDLE_ENFORCE_EQ(
      x_dims.size() >= 2 && x_dims.size() <= 5,
      true,
      phi::errors::InvalidArgument(
          "The size of input's dimensions should be between 2 and 5"
          "But received: the size of input's dimensions is [%d]",
          x_dims.size()));

  int C = 1;
  if (x_dims.size() == 2) {
    C = x_dims[1];
  } else {
    C = (data_layout_str == "NCHW") ? x_dims[1] : x_dims[x_dims.size() - 1];
  }

  auto* scale_ptr = scale.get_ptr();
  auto* bias_ptr = bias.get_ptr();

  phi::DenseTensor new_scale;
  phi::DenseTensor new_bias;
  if (scale_ptr) {
    new_scale = scale.get();
  } else {
    new_scale.Resize({C});
    FillGcuTensorWithConstant<T>(&new_scale, dev_ctx, static_cast<T>(1));
  }

  if (bias_ptr) {
    new_bias = bias.get();
  } else {
    new_bias.Resize({C});
    FillGcuTensorWithConstant<T>(&new_bias, dev_ctx, static_cast<T>(0));
  }

  // The data type is not set in BatchNormOp::InferShape.
  // This may be a bug in the execution of NaiveExecutor.
  auto meta = y->meta();
  meta.dtype = x.dtype();
  y->set_meta(meta);

  dev_ctx.Alloc(y, y->dtype());
  dev_ctx.Alloc(mean_out, mean_out->dtype());
  dev_ctx.Alloc(variance_out, variance_out->dtype());
  dev_ctx.Alloc(saved_mean, saved_mean->dtype());
  dev_ctx.Alloc(saved_variance, saved_variance->dtype());

  if (LaunchAOTKernel()) {
    phi::DenseTensor input_x = MaybeCreateOrTrans64To32bits(dev_ctx, x);
    phi::DenseTensor scale_x = MaybeCreateOrTrans64To32bits(dev_ctx, new_scale);
    phi::DenseTensor bias_x = MaybeCreateOrTrans64To32bits(dev_ctx, new_bias);
    phi::DenseTensor mean_x =
        MaybeCreateOrTrans64To32bits(dev_ctx, running_mean);
    phi::DenseTensor variance_x =
        MaybeCreateOrTrans64To32bits(dev_ctx, running_var);

    phi::DenseTensor output = MaybeCreateOrTrans64To32bits(dev_ctx, *y, false);
    phi::DenseTensor saved_mean_output =
        MaybeCreateOrTrans64To32bits(dev_ctx, *saved_mean, false);
    phi::DenseTensor saved_variance_output =
        MaybeCreateOrTrans64To32bits(dev_ctx, *saved_variance, false);

    double momentum_d = 1.0 - momentum;
    double epsilon_d = epsilon;

    LAUNCH_TOPSATENOP(topsatenNativeBatchNorm,
                      dev_ctx,
                      output,
                      saved_mean_output,
                      saved_variance_output,
                      input_x,
                      scale_x,
                      bias_x,
                      mean_x,
                      variance_x,
                      !is_test,
                      momentum_d,
                      epsilon_d);
    MaybeTransResult(dev_ctx, output, y);
    MaybeTransResult(dev_ctx, saved_mean_output, saved_mean);
    MaybeTransResult(dev_ctx, saved_variance_output, saved_variance);
    // TensorCopy(dev_ctx, running_mean, false, mean_out);
    // TensorCopy(dev_ctx, running_var, false, variance_out);
    *mean_out = running_mean;
    *variance_out = running_var;

  } else {  // kernel impl base on JIT
    phi::DenseTensor mean_out_tmp;
    mean_out_tmp.set_meta(mean_out->meta());
    dev_ctx.template Alloc<T>(&mean_out_tmp);

    phi::DenseTensor variance_out_tmp;
    variance_out_tmp.set_meta(variance_out->meta());
    dev_ctx.template Alloc<T>(&variance_out_tmp);

    TensorNameMap input_names;
    input_names["X"] = {"x"};
    input_names["Mean"] = {"running_mean"};
    input_names["Variance"] = {"running_var"};
    input_names["Bias"] = {"bias"};
    input_names["Scale"] = {"scale"};

    TensorValueMap inputs;
    inputs["X"] = {const_cast<DenseTensor*>(&x)};
    inputs["Mean"] = {const_cast<DenseTensor*>(&running_mean)};
    inputs["Variance"] = {const_cast<DenseTensor*>(&running_var)};
    inputs["Bias"] = {&new_bias};
    inputs["Scale"] = {&new_scale};

    TensorNameMap output_names;
    output_names["Y"] = {"y"};
    output_names["MeanOut"] = {"mean_out"};
    output_names["VarianceOut"] = {"variance_out"};
    output_names["SavedMean"] = {"saved_mean"};
    output_names["SavedVariance"] = {"saved_variance"};

    TensorValueMap outputs;
    outputs["Y"] = {y};
    outputs["MeanOut"] = {&mean_out_tmp};
    outputs["VarianceOut"] = {&variance_out_tmp};
    outputs["SavedMean"] = {saved_mean};
    outputs["SavedVariance"] = {saved_variance};

    GcuAttributeMap attrs;
    attrs["is_test"] = is_test;
    attrs["momentum"] = momentum;
    attrs["epsilon"] = epsilon;
    attrs["data_layout"] = data_layout_str;
    attrs["use_global_stats"] = use_global_stats;
    attrs["trainable_statistics"] = trainable_stats;

    if (saved_mean && saved_mean->numel() > 0) {
      attrs["init_saved_mean"] = true;
    } else {
      attrs["init_saved_mean"] = false;
    }
    if (saved_variance && saved_variance->numel() > 0) {
      attrs["init_saved_variance"] = true;
    } else {
      attrs["init_saved_variance"] = false;
    }

    GcuRunner(input_names,
              inputs,
              output_names,
              outputs,
              attrs,
              "batch_norm",
              dev_ctx);
    TensorCopy(dev_ctx, mean_out_tmp, false, mean_out);
    TensorCopy(dev_ctx, variance_out_tmp, false, variance_out);
  }
}

template <typename T, typename Context>
void BatchNormGradKernel(
    const Context& dev_ctx,
    const phi::DenseTensor& x,
    const paddle::optional<phi::DenseTensor>& scale,
    const paddle::optional<phi::DenseTensor>& bias,
    const paddle::optional<phi::DenseTensor>& mean,
    const paddle::optional<phi::DenseTensor>& variance,
    const phi::DenseTensor& saved_mean,
    const phi::DenseTensor& saved_variance,
    const paddle::optional<phi::DenseTensor>& reserve_space,
    const phi::DenseTensor& y_grad,
    float momentum,
    float epsilon,
    const std::string& data_layout_str,
    bool is_test,
    bool use_global_stats,
    bool trainable_statistics,
    phi::DenseTensor* x_grad,
    phi::DenseTensor* scale_grad,
    phi::DenseTensor* bias_grad) {
  PADDLE_GCU_KERNEL_TRACE("batch_norm_grad");
  const auto& x_dims = x.dims();
  int C = 1;
  if (x_dims.size() == 2) {
    C = x_dims[1];
  } else {
    C = (data_layout_str == "NCHW") ? x_dims[1] : x_dims[x_dims.size() - 1];
  }

  auto* scale_ptr = scale.get_ptr();
  auto* bias_ptr = bias.get_ptr();

  phi::DenseTensor new_scale;
  phi::DenseTensor new_bias;
  if (scale_ptr) {
    new_scale = scale.get();
  } else {
    new_scale.Resize({C});
    FillGcuTensorWithConstant<T>(&new_scale, dev_ctx, static_cast<T>(1));
  }

  if (bias_ptr) {
    new_bias = bias.get();
  } else {
    new_bias.Resize({C});
    FillGcuTensorWithConstant<T>(&new_bias, dev_ctx, static_cast<T>(0));
  }

  if (LaunchAOTKernel()) {
    THROW_AOT_UNIMPLEMENTED();
  } else {  // kernel impl base on JIT
    TensorNameMap input_names;
    input_names["X"] = {"x"};
    input_names["SavedMean"] = {"saved_mean"};
    input_names["SavedVariance"] = {"saved_variance"};
    input_names["Bias"] = {"bias"};
    input_names["Scale"] = {"scale"};
    input_names[GradVarName("Y")] = {"y_grad"};

    TensorValueMap inputs;
    inputs["X"] = {const_cast<DenseTensor*>(&x)};
    inputs["SavedMean"] = {const_cast<DenseTensor*>(&saved_mean)};
    inputs["SavedVariance"] = {const_cast<DenseTensor*>(&saved_variance)};
    inputs["Bias"] = {&new_bias};
    inputs["Scale"] = {&new_scale};
    inputs[GradVarName("Y")] = {const_cast<DenseTensor*>(&y_grad)};

    TensorNameMap output_names;
    TensorValueMap outputs;
    if (x_grad) {
      dev_ctx.template Alloc<T>(x_grad);
      output_names[GradVarName("X")] = {"x_grad"};
      outputs[GradVarName("X")] = {x_grad};
    }
    if (scale_grad) {
      dev_ctx.template Alloc<T>(scale_grad);
      output_names[GradVarName("Scale")] = {"scale_grad"};
      outputs[GradVarName("Scale")] = {scale_grad};
    }
    if (bias_grad) {
      dev_ctx.template Alloc<T>(bias_grad);
      output_names[GradVarName("Bias")] = {"bias_grad"};
      outputs[GradVarName("Bias")] = {bias_grad};
    }

    GcuAttributeMap attrs;
    attrs["is_test"] = is_test;
    attrs["momentum"] = momentum;
    attrs["epsilon"] = epsilon;
    attrs["data_layout"] = data_layout_str;
    attrs["use_global_stats"] = use_global_stats;
    attrs["trainable_statistics"] = trainable_statistics;

    GcuRunner(input_names,
              inputs,
              output_names,
              outputs,
              attrs,
              "batch_norm_grad",
              dev_ctx);
  }
}

template <typename T, typename Context>
void BatchNormInferKernel(const Context& dev_ctx,
                          const phi::DenseTensor& x,
                          const phi::DenseTensor& mean,
                          const phi::DenseTensor& variance,
                          const phi::DenseTensor& scale,
                          const phi::DenseTensor& bias,
                          float momentum,
                          float epsilon,
                          const std::string& data_layout_str,
                          phi::DenseTensor* y,
                          phi::DenseTensor* mean_out,
                          phi::DenseTensor* variance_out) {
  PADDLE_GCU_KERNEL_TRACE("batch_norm_infer");
  const auto& x_dims = x.dims();
  PADDLE_ENFORCE_EQ(
      x_dims.size() >= 2 && x_dims.size() <= 5,
      true,
      phi::errors::InvalidArgument(
          "The size of input's dimensions should be between 2 and 5"
          "But received: the size of input's dimensions is [%d]",
          x_dims.size()));

  // The data type is not set in BatchNormOp::InferShape.
  // This may be a bug in the execution of NaiveExecutor.
  auto meta = y->meta();
  meta.dtype = x.dtype();
  y->set_meta(meta);

  dev_ctx.Alloc(y, y->dtype());
  dev_ctx.Alloc(mean_out, mean_out->dtype());
  dev_ctx.Alloc(variance_out, variance_out->dtype());

  if (LaunchAOTKernel()) {
    phi::DenseTensor input_x = MaybeCreateOrTrans64To32bits(dev_ctx, x);
    phi::DenseTensor scale_x = MaybeCreateOrTrans64To32bits(dev_ctx, scale);
    phi::DenseTensor bias_x = MaybeCreateOrTrans64To32bits(dev_ctx, bias);
    phi::DenseTensor mean_x = MaybeCreateOrTrans64To32bits(dev_ctx, mean);
    phi::DenseTensor variance_x =
        MaybeCreateOrTrans64To32bits(dev_ctx, variance);

    phi::DenseTensor output = MaybeCreateOrTrans64To32bits(dev_ctx, *y, false);
    phi::DenseTensor mean_output =
        MaybeCreateOrTrans64To32bits(dev_ctx, *mean_out, false);
    phi::DenseTensor variance_output =
        MaybeCreateOrTrans64To32bits(dev_ctx, *variance_out, false);

    double momentum_d = 1.0 - momentum;
    double epsilon_d = epsilon;

    LAUNCH_TOPSATENOP(topsatenNativeBatchNorm,
                      dev_ctx,
                      output,
                      mean_output,
                      variance_output,
                      input_x,
                      scale_x,
                      bias_x,
                      mean_x,
                      variance_x,
                      false,
                      momentum_d,
                      epsilon_d);
    MaybeTransResult(dev_ctx, output, y);
    MaybeTransResult(dev_ctx, mean_output, mean_out);
    MaybeTransResult(dev_ctx, variance_output, variance_out);

  } else {  // kernel impl base on JIT
    TensorNameMap input_names;
    input_names["X"] = {"x"};
    input_names["Mean"] = {"mean"};
    input_names["Variance"] = {"var"};
    input_names["Bias"] = {"bias"};
    input_names["Scale"] = {"scale"};

    TensorValueMap inputs;
    inputs["X"] = {const_cast<DenseTensor*>(&x)};
    inputs["Mean"] = {const_cast<DenseTensor*>(&mean)};
    inputs["Variance"] = {const_cast<DenseTensor*>(&variance)};
    inputs["Bias"] = {const_cast<DenseTensor*>(&bias)};
    inputs["Scale"] = {const_cast<DenseTensor*>(&scale)};

    TensorNameMap output_names;
    output_names["Y"] = {"y"};

    TensorValueMap outputs;
    outputs["Y"] = {y};

    GcuAttributeMap attrs;
    attrs["epsilon"] = epsilon;
    attrs["data_layout"] = data_layout_str;

    GcuRunner(input_names,
              inputs,
              output_names,
              outputs,
              attrs,
              "batch_norm_infer",
              dev_ctx);
  }
}
}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(batch_norm,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::BatchNormKernel,
                          float,
                          double,
                          phi::dtype::float16) {
  if (kernel_key.dtype() == phi::DataType::FLOAT16) {
    kernel->InputAt(1).SetDataType(phi::DataType::FLOAT32);   // mean
    kernel->InputAt(2).SetDataType(phi::DataType::FLOAT32);   // variance
    kernel->InputAt(3).SetDataType(phi::DataType::FLOAT32);   // scale
    kernel->InputAt(4).SetDataType(phi::DataType::FLOAT32);   // bias
    kernel->OutputAt(1).SetDataType(phi::DataType::FLOAT32);  // mean_out
    kernel->OutputAt(2).SetDataType(phi::DataType::FLOAT32);  // variance_out
    kernel->OutputAt(3).SetDataType(phi::DataType::FLOAT32);  // saved_mean
    kernel->OutputAt(4).SetDataType(phi::DataType::FLOAT32);  // saved_variance
  }
}

PD_REGISTER_PLUGIN_KERNEL(batch_norm_grad,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::BatchNormGradKernel,
                          float,
                          double,
                          phi::dtype::float16) {
  if (kernel_key.dtype() == phi::DataType::FLOAT16) {
    kernel->OutputAt(0).SetDataType(phi::DataType::FLOAT32);  // x_grad
    kernel->OutputAt(1).SetDataType(phi::DataType::FLOAT32);  // scale_grad
    kernel->OutputAt(2).SetDataType(phi::DataType::FLOAT32);  // bias_grad
  }
}

PD_REGISTER_PLUGIN_KERNEL(batch_norm_infer,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::BatchNormInferKernel,
                          float,
                          double,
                          phi::dtype::float16) {
  if (kernel_key.dtype() == phi::DataType::FLOAT16) {
    kernel->OutputAt(1).SetDataType(phi::DataType::FLOAT32);  // mean_out
    kernel->OutputAt(2).SetDataType(phi::DataType::FLOAT32);  // variance_out
  }
}
