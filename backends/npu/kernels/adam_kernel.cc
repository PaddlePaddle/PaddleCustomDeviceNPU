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

#include "kernels/funcs/npu_funcs.h"
#include "kernels/funcs/npu_op_runner.h"

namespace custom_kernel {

template <typename T, typename Context>
void CastKernel(const Context& dev_ctx,
                const phi::DenseTensor& x,
                phi::DataType dtype,
                phi::DenseTensor* out);

template <typename T, typename Context>
void AdamImplKernel(const Context& dev_ctx,
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
  using MPDType = typename phi::dtype::MPTypeTrait<T>::Type;

  phi::DenseTensor* beta1_pow = const_cast<phi::DenseTensor*>(&beta1_pow_in);
  phi::DenseTensor* beta2_pow = const_cast<phi::DenseTensor*>(&beta2_pow_in);

  VLOG(4) << "use_global_beta_pow:" << use_global_beta_pow;

  dev_ctx.template Alloc<T>(param_out);
  dev_ctx.template Alloc<MPDType>(moment1_out);
  dev_ctx.template Alloc<MPDType>(moment2_out);

  // NOTE(zhiqiu): beta1_pow and beta2_pow may on CPU and not transform
  // place.
  phi::DenseTensor beta1_pow_tmp;
  phi::DenseTensor beta2_pow_tmp;
  if (beta1_pow->place().GetType() == phi::AllocationType::CPU) {
    MPDType beta1 = *beta1_pow->data<MPDType>();
    beta1_pow_tmp.Resize({1});
    dev_ctx.template Alloc<MPDType>(&beta1_pow_tmp);
    FillNpuTensorWithConstant<MPDType>(&beta1_pow_tmp, dev_ctx, beta1);
    beta1_pow = &beta1_pow_tmp;
  }
  if (beta2_pow->place().GetType() == phi::AllocationType::CPU) {
    MPDType beta2 = *beta2_pow->data<MPDType>();
    beta2_pow_tmp.Resize({1});
    dev_ctx.template Alloc<MPDType>(&beta2_pow_tmp);
    FillNpuTensorWithConstant<MPDType>(&beta2_pow_tmp, dev_ctx, beta2);
    beta2_pow = &beta2_pow_tmp;
  }

  const phi::DenseTensor* beta1_tensor = nullptr;
  const phi::DenseTensor* beta2_tensor = nullptr;
  const phi::DenseTensor* epsilon_tensor = nullptr;

  phi::DenseTensor beta1_tmp;
  phi::DenseTensor beta2_tmp;
  phi::DenseTensor epsilon_tmp;
  beta1_tmp.Resize({1});
  beta2_tmp.Resize({1});
  epsilon_tmp.Resize({1});

  MPDType beta1 = beta1_in.to<MPDType>();
  dev_ctx.template Alloc<MPDType>(&beta1_tmp);
  FillNpuTensorWithConstant<MPDType>(&beta1_tmp, dev_ctx, beta1);
  beta1_tensor = &beta1_tmp;

  MPDType beta2 = beta2_in.to<MPDType>();
  dev_ctx.template Alloc<MPDType>(&beta2_tmp);
  FillNpuTensorWithConstant<MPDType>(&beta2_tmp, dev_ctx, beta2);
  beta2_tensor = &beta2_tmp;

  MPDType epsilon = epsilon_in.to<MPDType>();
  dev_ctx.template Alloc<MPDType>(&epsilon_tmp);
  FillNpuTensorWithConstant<MPDType>(&epsilon_tmp, dev_ctx, epsilon);
  epsilon_tensor = &epsilon_tmp;

  VLOG(3) << "beta1_pow.numel() : " << beta1_pow->numel()
          << "beta2_pow.numel() : " << beta2_pow->numel();
  VLOG(3) << "param.numel(): " << param.numel();

  auto stream = dev_ctx.stream();

  const phi::DenseTensor* calc_param =
      multi_precision ? &master_param.get() : &param;

  // NOTE(zhiqiu): ApplyAdamD updates params inplace
  TensorCopy(dev_ctx, moment1, false, moment1_out);
  TensorCopy(dev_ctx, moment2, false, moment2_out);

  if (multi_precision) {
    dev_ctx.template Alloc<MPDType>(master_param_out);
    phi::DenseTensor master_param_t;
    auto tmp_master_param = master_param.get();
    if (tmp_master_param.dtype() == phi::DataType::FLOAT64 ||
        tmp_master_param.dtype() == phi::DataType::FLOAT32) {
      phi::DenseTensorMeta master_param_meta = {phi::DataType::FLOAT32,
                                                tmp_master_param.dims(),
                                                tmp_master_param.layout()};
      master_param_t.set_meta(master_param_meta);
      custom_kernel::CastKernel<T, Context>(
          dev_ctx, tmp_master_param, phi::DataType::FLOAT32, &master_param_t);
    }
    TensorCopy(dev_ctx, master_param_t, false, master_param_out);
    const auto& runner = NpuOpRunner("ApplyAdamD",
                                     {
                                         *master_param_out,
                                         *moment1_out,
                                         *moment2_out,
                                         *beta1_pow,
                                         *beta2_pow,
                                         learning_rate,
                                         *beta1_tensor,
                                         *beta2_tensor,
                                         *epsilon_tensor,
                                         grad,
                                     },
                                     {
                                         *master_param_out,
                                         *moment1_out,
                                         *moment2_out,
                                     },
                                     {});
    runner.Run(stream);

    custom_kernel::CastKernel<T, Context>(
        dev_ctx, *master_param_out, param_out->dtype(), param_out);
  } else if (param.dtype() == phi::DataType::FLOAT16) {
    phi::DenseTensor param_fp32;
    param_fp32.Resize(calc_param->dims());
    dev_ctx.template Alloc<MPDType>(&param_fp32);

    custom_kernel::CastKernel<T, Context>(
        dev_ctx, param, param_fp32.dtype(), &param_fp32);

    const auto& runner = NpuOpRunner("ApplyAdamD",
                                     {
                                         param_fp32,
                                         *moment1_out,
                                         *moment2_out,
                                         *beta1_pow,
                                         *beta2_pow,
                                         learning_rate,
                                         *beta1_tensor,
                                         *beta2_tensor,
                                         *epsilon_tensor,
                                         grad,
                                     },
                                     {
                                         param_fp32,
                                         *moment1_out,
                                         *moment2_out,
                                     },
                                     {});
    runner.Run(stream);

    custom_kernel::CastKernel<T, Context>(
        dev_ctx, param_fp32, param_out->dtype(), param_out);
  } else {
    TensorCopy(dev_ctx, param, false, param_out);
    const auto& runner = NpuOpRunner("ApplyAdamD",
                                     {
                                         *param_out,
                                         *moment1_out,
                                         *moment2_out,
                                         *beta1_pow,
                                         *beta2_pow,
                                         learning_rate,
                                         *beta1_tensor,
                                         *beta2_tensor,
                                         *epsilon_tensor,
                                         grad,
                                     },
                                     {
                                         *param_out,
                                         *moment1_out,
                                         *moment2_out,
                                     },
                                     {});
    runner.Run(stream);
  }

  if (!use_global_beta_pow) {
    dev_ctx.template Alloc<MPDType>(beta1_pow_out);
    dev_ctx.template Alloc<MPDType>(beta2_pow_out);
    const auto& runner_m1 =
        NpuOpRunner("Mul", {*beta1_pow, *beta1_tensor}, {*beta1_pow_out}, {});
    runner_m1.Run(stream);
    const auto& runner_m2 =
        NpuOpRunner("Mul", {*beta2_pow, *beta2_tensor}, {*beta2_pow_out}, {});
    runner_m2.Run(stream);
  }
}

template <typename Context>
void CastFP64toFP32Kernel(const Context& dev_ctx,
                          const phi::DenseTensor& in,
                          phi::DenseTensor* out) {
  auto stream = dev_ctx.stream();
  out->Resize(in.dims());
  dev_ctx.template Alloc<float>(out);
  const auto& cast_runner =
      NpuOpRunner("Cast", {in}, {*out}, {{"dst_type", ACL_FLOAT}});
  cast_runner.Run(stream);
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
    if (!use_global_beta_pow) {
      TensorCopy(dev_ctx, beta1_pow_in, false, beta1_pow_out);
      TensorCopy(dev_ctx, beta2_pow_in, false, beta2_pow_out);
    }
    return;
  }

  if (param.dtype() == phi::DataType::FLOAT64) {
    // trans input
    auto stream = dev_ctx.stream();
    phi::DenseTensor trans_param, trans_grad, trans_learning_rate,
        trans_moment1, trans_moment2, trans_beta1_pow_in, trans_beta2_pow_in;
    CastFP64toFP32Kernel<Context>(dev_ctx, param, &trans_param);
    CastFP64toFP32Kernel<Context>(dev_ctx, grad, &trans_grad);
    CastFP64toFP32Kernel<Context>(dev_ctx, learning_rate, &trans_learning_rate);
    CastFP64toFP32Kernel<Context>(dev_ctx, moment1, &trans_moment1);
    CastFP64toFP32Kernel<Context>(dev_ctx, moment2, &trans_moment2);
    CastFP64toFP32Kernel<Context>(dev_ctx, beta1_pow_in, &trans_beta1_pow_in);
    CastFP64toFP32Kernel<Context>(dev_ctx, beta2_pow_in, &trans_beta2_pow_in);
    phi::DenseTensor trans_param_out;
    phi::DenseTensorMeta meta = {
        phi::DataType::FLOAT32, param_out->dims(), param_out->layout()};
    trans_param_out.set_meta(meta);

    // impl kernel
    custom_kernel::AdamImplKernel<float, Context>(
        dev_ctx,
        trans_param,
        trans_grad,
        trans_learning_rate,
        trans_moment1,
        trans_moment2,
        trans_beta1_pow_in,
        trans_beta2_pow_in,
        master_param,
        skip_update,
        beta1_in,
        beta2_in,
        epsilon_in,
        lazy_mode,
        min_row_size_to_use_multithread,
        multi_precision,
        use_global_beta_pow,
        &trans_param_out,
        moment1_out,
        moment2_out,
        beta1_pow_out,
        beta2_pow_out,
        master_param_out);
    // trans output
    dev_ctx.template Alloc<T>(param_out);
    const auto& cast_runner = NpuOpRunner(
        "Cast", {trans_param_out}, {*param_out}, {{"dst_type", ACL_DOUBLE}});
    cast_runner.Run(stream);
  } else {
    custom_kernel::AdamImplKernel<T, Context>(dev_ctx,
                                              param,
                                              grad,
                                              learning_rate,
                                              moment1,
                                              moment2,
                                              beta1_pow_in,
                                              beta2_pow_in,
                                              master_param,
                                              skip_update,
                                              beta1_in,
                                              beta2_in,
                                              epsilon_in,
                                              lazy_mode,
                                              min_row_size_to_use_multithread,
                                              multi_precision,
                                              use_global_beta_pow,
                                              param_out,
                                              moment1_out,
                                              moment2_out,
                                              beta1_pow_out,
                                              beta2_pow_out,
                                              master_param_out);
  }
}

template <typename T, typename Context>
void AdamwKernel(const Context& dev_ctx,
                 const phi::DenseTensor& param,
                 const phi::DenseTensor& grad,
                 const phi::DenseTensor& learning_rate,
                 const phi::DenseTensor& moment1,
                 const phi::DenseTensor& moment2,
                 const phi::DenseTensor& beta1_pow,
                 const phi::DenseTensor& beta2_pow,
                 const paddle::optional<phi::DenseTensor>& master_param,
                 const paddle::optional<phi::DenseTensor>& skip_update,
                 const phi::Scalar& beta1,
                 const phi::Scalar& beta2,
                 const phi::Scalar& epsilon,
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
                 phi::DenseTensor* master_param_outs) {
  using MPDType = typename phi::dtype::MPTypeTrait<T>::Type;

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

  VLOG(3) << "Skip update" << skip_update_;
  if (skip_update_) {
    TensorCopy(dev_ctx, param, false, param_out);
    TensorCopy(dev_ctx, moment1, false, moment1_out);
    TensorCopy(dev_ctx, moment2, false, moment2_out);
    if (!use_global_beta_pow) {
      TensorCopy(dev_ctx, beta1_pow, false, beta1_pow_out);
      TensorCopy(dev_ctx, beta2_pow, false, beta2_pow_out);
    }
    return;
  }

  if (!skip_update_) {
    phi::DenseTensor one;
    phi::DenseTensor decay;
    phi::DenseTensor tmp;
    phi::DenseTensorMeta meta = {phi::DataType::FLOAT32, {1}};
    one.set_meta(meta);
    decay.set_meta(meta);
    tmp.set_meta(meta);

    dev_ctx.template Alloc<MPDType>(&tmp);
    dev_ctx.template Alloc<MPDType>(&one);
    dev_ctx.template Alloc<MPDType>(&decay);

    FillNpuTensorWithConstant<MPDType>(
        &one, dev_ctx, static_cast<MPDType>(1.0f));
    if (!with_decay) {
      coeff = 0.0f;
    }
    NPUAttributeMap attr_input = {{"value", coeff}};

    auto stream = dev_ctx.stream();

    const auto& runner1 =
        NpuOpRunner("Muls", {learning_rate}, {tmp}, attr_input);
    runner1.Run(stream);

    const auto& runner2 = NpuOpRunner("Sub", {one, tmp}, {decay}, {});
    runner2.Run(stream);

    if (multi_precision) {
      phi::DenseTensor new_master_param;
      new_master_param.Resize(master_param->dims());
      dev_ctx.template Alloc<MPDType>(&new_master_param);
      const auto& runner = NpuOpRunner(
          "Mul", {master_param.get(), decay}, {new_master_param}, {});
      runner.Run(stream);
      custom_kernel::AdamImplKernel<T, Context>(dev_ctx,
                                                param,
                                                grad,
                                                learning_rate,
                                                moment1,
                                                moment2,
                                                beta1_pow,
                                                beta2_pow,
                                                new_master_param,
                                                skip_update,
                                                beta1,
                                                beta2,
                                                epsilon,
                                                lazy_mode,
                                                min_row_size_to_use_multithread,
                                                multi_precision,
                                                use_global_beta_pow,
                                                param_out,
                                                moment1_out,
                                                moment2_out,
                                                beta1_pow_out,
                                                beta2_pow_out,
                                                master_param_outs);
    } else {
      phi::DenseTensor new_param;
      new_param.Resize(param.dims());
      dev_ctx.template Alloc<T>(&new_param);
      const auto& runner = NpuOpRunner("Mul", {param, decay}, {new_param}, {});
      runner.Run(stream);
      custom_kernel::AdamImplKernel<T, Context>(dev_ctx,
                                                new_param,
                                                grad,
                                                learning_rate,
                                                moment1,
                                                moment2,
                                                beta1_pow,
                                                beta2_pow,
                                                master_param,
                                                skip_update,
                                                beta1,
                                                beta2,
                                                epsilon,
                                                lazy_mode,
                                                min_row_size_to_use_multithread,
                                                multi_precision,
                                                use_global_beta_pow,
                                                param_out,
                                                moment1_out,
                                                moment2_out,
                                                beta1_pow_out,
                                                beta2_pow_out,
                                                master_param_outs);
    }
  }
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(adam,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::AdamKernel,
                          phi::dtype::float16,
                          phi::dtype::bfloat16,
                          float,
                          double) {
  // Skip beta1_pow, beta2_pow, skip_update data transform
  kernel->InputAt(5).SetBackend(phi::Backend::ALL_BACKEND);
  kernel->InputAt(6).SetBackend(phi::Backend::ALL_BACKEND);
  kernel->InputAt(8).SetBackend(phi::Backend::ALL_BACKEND);
  if (kernel_key.dtype() == phi::DataType::FLOAT16) {
    kernel->OutputAt(1).SetDataType(phi::DataType::FLOAT32);
    kernel->OutputAt(2).SetDataType(phi::DataType::FLOAT32);
    kernel->OutputAt(3).SetDataType(phi::DataType::FLOAT32);
    kernel->OutputAt(4).SetDataType(phi::DataType::FLOAT32);
    kernel->OutputAt(5).SetDataType(phi::DataType::FLOAT32);
  }
  kernel->OutputAt(3).SetBackend(phi::Backend::UNDEFINED);
  kernel->OutputAt(4).SetBackend(phi::Backend::UNDEFINED);
}

PD_REGISTER_PLUGIN_KERNEL(adamw,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::AdamwKernel,
                          phi::dtype::float16,
                          phi::dtype::bfloat16,
                          float,
                          double) {
  // Skip beta1_pow, beta2_pow, skip_update data transform
  kernel->InputAt(5).SetBackend(phi::Backend::ALL_BACKEND);
  kernel->InputAt(6).SetBackend(phi::Backend::ALL_BACKEND);
  kernel->InputAt(8).SetBackend(phi::Backend::ALL_BACKEND);
  if (kernel_key.dtype() == phi::DataType::FLOAT16) {
    kernel->OutputAt(1).SetDataType(phi::DataType::FLOAT32);
    kernel->OutputAt(2).SetDataType(phi::DataType::FLOAT32);
    kernel->OutputAt(3).SetDataType(phi::DataType::FLOAT32);
    kernel->OutputAt(4).SetDataType(phi::DataType::FLOAT32);
    kernel->OutputAt(5).SetDataType(phi::DataType::FLOAT32);
  }
  kernel->OutputAt(3).SetBackend(phi::Backend::UNDEFINED);
  kernel->OutputAt(4).SetBackend(phi::Backend::UNDEFINED);
}
