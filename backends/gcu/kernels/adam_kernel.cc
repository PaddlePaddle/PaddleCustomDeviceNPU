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
void AdamBaseKernel(const Context& dev_ctx,
                    const phi::DenseTensor& param,
                    const phi::DenseTensor& grad,
                    const phi::DenseTensor& learning_rate,
                    const phi::DenseTensor& moment1,
                    const phi::DenseTensor& moment2,
                    const phi::DenseTensor& beta1_pow_in,
                    const phi::DenseTensor& beta2_pow_in,
                    const paddle::optional<phi::DenseTensor>& master_param,
                    const paddle::optional<phi::DenseTensor>& skip_update,
                    const GcuAttributeMap& attrs,
                    bool use_global_beta_pow,
                    phi::DenseTensor* param_out,
                    phi::DenseTensor* moment1_out,
                    phi::DenseTensor* moment2_out,
                    phi::DenseTensor* beta1_pow_out,
                    phi::DenseTensor* beta2_pow_out,
                    phi::DenseTensor* master_param_out,
                    const std::string& op_type) {
  bool skip_update_ = false;
  if (skip_update.is_initialized()) {
    PADDLE_ENFORCE_EQ(skip_update->numel(),
                      1,
                      phi::errors::InvalidArgument(
                          "Input(SkipUpdate) size must be 1, but get %d",
                          skip_update->numel()));
    std::vector<bool> skip_update_vec;
    custom_kernel::TensorToVector(
        dev_ctx, *skip_update, dev_ctx, &skip_update_vec);
    skip_update_ = skip_update_vec[0];
  }

  // skip_update=true, just copy input to output, and TensorCopy will call
  // mutable_data
  if (skip_update_) {
    VLOG(4) << "Adam skip update";
    TensorCopy(dev_ctx, param, false, param_out);
    TensorCopy(dev_ctx, moment1, false, moment1_out);
    TensorCopy(dev_ctx, moment2, false, moment2_out);
    TensorCopy(dev_ctx, beta1_pow_in, false, beta1_pow_out);
    TensorCopy(dev_ctx, beta2_pow_in, false, beta2_pow_out);
    return;
  }

  phi::DenseTensor* beta1_pow = const_cast<phi::DenseTensor*>(&beta1_pow_in);
  phi::DenseTensor* beta2_pow = const_cast<phi::DenseTensor*>(&beta2_pow_in);

  VLOG(4) << "use_global_beta_pow:" << use_global_beta_pow;

  dev_ctx.template Alloc<T>(param_out);
  dev_ctx.template Alloc<T>(moment1_out);
  dev_ctx.template Alloc<T>(moment2_out);

  // beta1_pow and beta2_pow may on CPU and not transform
  // place.
  phi::DenseTensor beta1_pow_tmp;
  phi::DenseTensor beta2_pow_tmp;
  if (beta1_pow->place().GetType() == phi::AllocationType::CPU) {
    T beta1 = *beta1_pow->data<T>();
    beta1_pow_tmp.Resize({1});
    dev_ctx.template Alloc<T>(&beta1_pow_tmp);
    FillGcuTensorWithConstant<T>(&beta1_pow_tmp, dev_ctx, beta1);
    beta1_pow = &beta1_pow_tmp;
  }
  if (beta2_pow->place().GetType() == phi::AllocationType::CPU) {
    T beta2 = *beta2_pow->data<T>();
    beta2_pow_tmp.Resize({1});
    dev_ctx.template Alloc<T>(&beta2_pow_tmp);
    FillGcuTensorWithConstant<T>(&beta2_pow_tmp, dev_ctx, beta2);
    beta2_pow = &beta2_pow_tmp;
  }

  TensorNameMap input_names;
  input_names["Param"] = {"param"};
  input_names["Grad"] = {"grad"};
  input_names["LearningRate"] = {"learning_rate"};
  input_names["Moment1"] = {"moment1"};
  input_names["Moment2"] = {"moment2"};
  input_names["Beta1Pow"] = {"beta1_pow_in"};
  input_names["Beta2Pow"] = {"beta2_pow_in"};

  TensorValueMap inputs;
  inputs["Param"] = {const_cast<DenseTensor*>(&param)};
  inputs["Grad"] = {const_cast<DenseTensor*>(&grad)};
  inputs["LearningRate"] = {const_cast<DenseTensor*>(&learning_rate)};
  inputs["Moment1"] = {const_cast<DenseTensor*>(&moment1)};
  inputs["Moment2"] = {const_cast<DenseTensor*>(&moment2)};
  inputs["Beta1Pow"] = {beta1_pow};
  inputs["Beta2Pow"] = {beta2_pow};

  phi::DenseTensor param_out_tmp;
  param_out_tmp.set_meta(param_out->meta());
  dev_ctx.template Alloc<T>(&param_out_tmp);

  phi::DenseTensor moment1_out_tmp;
  moment1_out_tmp.set_meta(moment1_out->meta());
  dev_ctx.template Alloc<T>(&moment1_out_tmp);

  phi::DenseTensor moment2_out_tmp;
  moment2_out_tmp.set_meta(moment2_out->meta());
  dev_ctx.template Alloc<T>(&moment2_out_tmp);

  phi::DenseTensor beta1_pow_out_tmp;
  beta1_pow_out_tmp.set_meta(beta1_pow_out->meta());
  dev_ctx.template Alloc<T>(&beta1_pow_out_tmp);

  phi::DenseTensor beta2_pow_out_tmp;
  beta2_pow_out_tmp.set_meta(beta2_pow_out->meta());
  dev_ctx.template Alloc<T>(&beta2_pow_out_tmp);

  TensorNameMap output_names;
  output_names["ParamOut"] = {"param_out"};
  output_names["Moment1Out"] = {"moment1_out"};
  output_names["Moment2Out"] = {"moment2_out"};
  output_names["Beta1PowOut"] = {"beta1_pow_out"};
  output_names["Beta2PowOut"] = {"beta2_pow_out"};

  TensorValueMap outputs;
  outputs["ParamOut"] = {&param_out_tmp};
  outputs["Moment1Out"] = {&moment1_out_tmp};
  outputs["Moment2Out"] = {&moment2_out_tmp};
  outputs["Beta1PowOut"] = {&beta1_pow_out_tmp};
  outputs["Beta2PowOut"] = {&beta2_pow_out_tmp};

  GcuRunner(
      input_names, inputs, output_names, outputs, attrs, op_type, dev_ctx);

  TensorCopy(dev_ctx, param_out_tmp, false, param_out);
  TensorCopy(dev_ctx, moment1_out_tmp, false, moment1_out);
  TensorCopy(dev_ctx, moment2_out_tmp, false, moment2_out);
  if (!use_global_beta_pow) {
    dev_ctx.template Alloc<T>(beta1_pow_out);
    dev_ctx.template Alloc<T>(beta2_pow_out);
    TensorCopy(dev_ctx, beta1_pow_out_tmp, false, beta1_pow_out);
    TensorCopy(dev_ctx, beta2_pow_out_tmp, false, beta2_pow_out);
  }

  // updates params inplace, so
  // if param and param_out is not same, we need to do copy.
  if (param_out->data<T>() != param.data<T>()) {
    TensorCopy(dev_ctx, *param_out, false, const_cast<DenseTensor*>(&param));
  }
  if (moment1_out->data<T>() != moment1.data<T>()) {
    TensorCopy(
        dev_ctx, *moment1_out, false, const_cast<DenseTensor*>(&moment1));
  }
  if (moment2_out->data<T>() != moment2.data<T>()) {
    TensorCopy(
        dev_ctx, *moment2_out, false, const_cast<DenseTensor*>(&moment2));
  }
}

template <typename T, typename Context>
void AdamKernel(const Context& dev_ctx,
                const phi::DenseTensor& param,
                const phi::DenseTensor& grad,
                const phi::DenseTensor& learning_rate,
                const phi::DenseTensor& moment1,
                const phi::DenseTensor& moment2,
                const phi::DenseTensor& beta1_pow_in,
                const phi::DenseTensor& beta2_pow_in,
                const paddle::optional<phi::DenseTensor>& master_param,
                const paddle::optional<phi::DenseTensor>& skip_update,
                const phi::Scalar& beta1_in,
                const phi::Scalar& beta2_in,
                const phi::Scalar& epsilon_in,
                bool lazy_mode,
                int64_t min_row_size_to_use_multithread,
                bool multi_precision,
                bool use_global_beta_pow,
                phi::DenseTensor* param_out,
                phi::DenseTensor* moment1_out,
                phi::DenseTensor* moment2_out,
                phi::DenseTensor* beta1_pow_out,
                phi::DenseTensor* beta2_pow_out,
                phi::DenseTensor* master_param_out) {
  PADDLE_GCU_KERNEL_TRACE("adam");
  if (LaunchAOTKernel()) {
    THROW_AOT_UNIMPLEMENTED();
  } else {  // kernel impl base on JIT
    GcuAttributeMap attrs;
    attrs["beta1"] = beta1_in.to<float>();
    attrs["beta2"] = beta2_in.to<float>();
    attrs["epsilon"] = epsilon_in.to<float>();
    attrs["use_global_beta_pow"] = use_global_beta_pow;

    AdamBaseKernel<T, Context>(dev_ctx,
                               param,
                               grad,
                               learning_rate,
                               moment1,
                               moment2,
                               beta1_pow_in,
                               beta2_pow_in,
                               master_param,
                               skip_update,
                               attrs,
                               use_global_beta_pow,
                               param_out,
                               moment1_out,
                               moment2_out,
                               beta1_pow_out,
                               beta2_pow_out,
                               master_param_out,
                               "adam");
  }
}

template <typename T, typename Context>
void AdamwKernel(const Context& dev_ctx,
                 const phi::DenseTensor& param,
                 const phi::DenseTensor& grad,
                 const phi::DenseTensor& learning_rate,
                 const phi::DenseTensor& moment1,
                 const phi::DenseTensor& moment2,
                 const phi::DenseTensor& beta1_pow_in,
                 const phi::DenseTensor& beta2_pow_in,
                 const paddle::optional<phi::DenseTensor>& master_param,
                 const paddle::optional<phi::DenseTensor>& skip_update,
                 const phi::Scalar& beta1_in,
                 const phi::Scalar& beta2_in,
                 const phi::Scalar& epsilon_in,
                 float lr_ratio,
                 float coeff,
                 bool with_decay,
                 bool lazy_mode,
                 int64_t min_row_size_to_use_multithread,
                 bool multi_precision,
                 bool use_global_beta_pow,
                 phi::DenseTensor* param_out,
                 phi::DenseTensor* moment1_out,
                 phi::DenseTensor* moment2_out,
                 phi::DenseTensor* beta1_pow_out,
                 phi::DenseTensor* beta2_pow_out,
                 phi::DenseTensor* master_param_out) {
  PADDLE_GCU_KERNEL_TRACE("adamw");
  if (LaunchAOTKernel()) {
    THROW_AOT_UNIMPLEMENTED();
  } else {  // kernel impl base on JIT
    GcuAttributeMap attrs;
    attrs["beta1"] = beta1_in.to<float>();
    attrs["beta2"] = beta2_in.to<float>();
    attrs["epsilon"] = epsilon_in.to<float>();
    attrs["use_global_beta_pow"] = use_global_beta_pow;
    attrs["lr_ratio"] = lr_ratio;
    attrs["coeff"] = coeff;
    attrs["with_decay"] = with_decay;

    AdamBaseKernel<T, Context>(dev_ctx,
                               param,
                               grad,
                               learning_rate,
                               moment1,
                               moment2,
                               beta1_pow_in,
                               beta2_pow_in,
                               master_param,
                               skip_update,
                               attrs,
                               use_global_beta_pow,
                               param_out,
                               moment1_out,
                               moment2_out,
                               beta1_pow_out,
                               beta2_pow_out,
                               master_param_out,
                               "adamw");
  }
}

}  // namespace custom_kernel

// PD_REGISTER_PLUGIN_KERNEL(adam,
//                           gcu,
//                           ALL_LAYOUT,
//                           custom_kernel::AdamKernel,
//                           phi::dtype::float16,
//                           float,
//                           double) {
//   // Skip beta1_pow, beta2_pow, skip_update data transform
//   kernel->InputAt(5).SetBackend(phi::Backend::ALL_BACKEND);
//   kernel->InputAt(6).SetBackend(phi::Backend::ALL_BACKEND);
//   kernel->InputAt(8).SetBackend(phi::Backend::ALL_BACKEND);
//   if (kernel_key.dtype() == phi::DataType::FLOAT16) {
//     kernel->OutputAt(1).SetDataType(phi::DataType::FLOAT32);
//     kernel->OutputAt(2).SetDataType(phi::DataType::FLOAT32);
//     kernel->OutputAt(3).SetDataType(phi::DataType::FLOAT32);
//     kernel->OutputAt(4).SetDataType(phi::DataType::FLOAT32);
//     kernel->OutputAt(5).SetDataType(phi::DataType::FLOAT32);
//   }
//   kernel->OutputAt(3).SetBackend(phi::Backend::UNDEFINED);
//   kernel->OutputAt(4).SetBackend(phi::Backend::UNDEFINED);
// }

// PD_REGISTER_PLUGIN_KERNEL(adamw,
//                           gcu,
//                           ALL_LAYOUT,
//                           custom_kernel::AdamwKernel,
//                           phi::dtype::float16,
//                           float,
//                           double) {
//   // Skip beta1_pow, beta2_pow, skip_update data transform
//   kernel->InputAt(5).SetBackend(phi::Backend::ALL_BACKEND);
//   kernel->InputAt(6).SetBackend(phi::Backend::ALL_BACKEND);
//   kernel->InputAt(8).SetBackend(phi::Backend::ALL_BACKEND);
//   if (kernel_key.dtype() == phi::DataType::FLOAT16) {
//     kernel->OutputAt(1).SetDataType(phi::DataType::FLOAT32);
//     kernel->OutputAt(2).SetDataType(phi::DataType::FLOAT32);
//     kernel->OutputAt(3).SetDataType(phi::DataType::FLOAT32);
//     kernel->OutputAt(4).SetDataType(phi::DataType::FLOAT32);
//     kernel->OutputAt(5).SetDataType(phi::DataType::FLOAT32);
//   }
//   kernel->OutputAt(3).SetBackend(phi::Backend::UNDEFINED);
//   kernel->OutputAt(4).SetBackend(phi::Backend::UNDEFINED);
// }
