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

#include "kernels/funcs/mlu_baseop.h"
#include "kernels/funcs/mlu_funcs.h"

namespace custom_kernel {

template <typename T, typename Context>
void AdamKernel(
    const Context& dev_ctx,
    const phi::DenseTensor& param,
    const phi::DenseTensor& grad,
    const phi::DenseTensor& learning_rate,
    const phi::DenseTensor& moment1,
    const phi::DenseTensor& moment2,
    const paddle::optional<phi::DenseTensor>& moment2_max,  // UNUSED
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
    bool amsgrad,  // UNUSED
    phi::DenseTensor* param_out,
    phi::DenseTensor* moment1_out,
    phi::DenseTensor* moment2_out,
    phi::DenseTensor* moment2_max_out,  // UNUSED
    phi::DenseTensor* beta1_pow_out,
    phi::DenseTensor* beta2_pow_out,
    phi::DenseTensor* master_param_out) {
  PADDLE_ENFORCE_NE(
      amsgrad,
      true,
      phi::errors::Unimplemented("Operation amsgrad is not supported yet."));
  using MPDType = typename MPTypeTrait<T>::Type;

  bool skip_update_ = false;
  if (skip_update.is_initialized()) {
    PADDLE_ENFORCE_EQ(skip_update->numel(),
                      1,
                      phi::errors::InvalidArgument(
                          "Input(SkipUpdate) size must be 1, but get %d",
                          skip_update->numel()));
    std::vector<bool> skip_update_vec;
    TensorToVector(dev_ctx, *skip_update, dev_ctx, &skip_update_vec);
    dev_ctx.Wait();
    skip_update_ = skip_update_vec[0];
  }

  // skip_update_=true, just copy input to output asynchronously on the device,
  // and TensorCopy will call mutable_data
  if (skip_update_) {
    VLOG(4) << "Adam skip update";
    *param_out = param;
    *moment1_out = moment1;
    *moment2_out = moment2;
    if (!use_global_beta_pow) {
      *beta1_pow_out = beta1_pow_in;
      *beta2_pow_out = beta2_pow_in;
    }
    return;
  }

  phi::DenseTensor* beta1_pow = const_cast<phi::DenseTensor*>(&beta1_pow_in);
  phi::DenseTensor* beta2_pow = const_cast<phi::DenseTensor*>(&beta2_pow_in);

  VLOG(4) << "use_global_beta_pow:" << use_global_beta_pow;

  // param, moment1(momentum), moment2(velocity) in cnnl are in&out tensors.
  // *param_out = param; // alloc later
  *moment1_out = moment1;
  *moment2_out = moment2;

  phi::DenseTensor beta1_pow_tmp;
  phi::DenseTensor beta2_pow_tmp;
  if (beta1_pow->place().GetType() == phi::AllocationType::CPU) {
    MPDType beta1 = *beta1_pow->data<MPDType>();
    beta1_pow_tmp.Resize({1});
    dev_ctx.template Alloc<MPDType>(&beta1_pow_tmp);
    FillMLUTensorWithHostValue<MPDType>(dev_ctx, beta1, &beta1_pow_tmp);
    beta1_pow = &beta1_pow_tmp;
  }
  if (beta2_pow->place().GetType() == phi::AllocationType::CPU) {
    MPDType beta2 = *beta2_pow->data<MPDType>();
    beta2_pow_tmp.Resize({1});
    dev_ctx.template Alloc<MPDType>(&beta2_pow_tmp);
    FillMLUTensorWithHostValue<MPDType>(dev_ctx, beta2, &beta2_pow_tmp);
    beta2_pow = &beta2_pow_tmp;
  }

  VLOG(3) << "beta1_pow->numel() : " << beta1_pow->numel()
          << " beta2_pow->numel() : " << beta2_pow->numel();
  VLOG(3) << "param.numel(): " << param.numel();

  PADDLE_ENFORCE_EQ(beta1_pow_out->numel(),
                    1,
                    phi::errors::InvalidArgument(
                        "beta1 pow output size should be 1, but received "
                        "value is:%d.",
                        beta1_pow_out->numel()));

  PADDLE_ENFORCE_EQ(beta2_pow_out->numel(),
                    1,
                    phi::errors::InvalidArgument(
                        "beta2 pow output size should be 1, but received "
                        "value is:%d.",
                        beta2_pow_out->numel()));

  const phi::DenseTensor* beta1_tensor = nullptr;
  const phi::DenseTensor* beta2_tensor = nullptr;
  const phi::DenseTensor* epsilon_tensor = nullptr;

  phi::DenseTensor beta1_tmp;
  phi::DenseTensor beta2_tmp;
  phi::DenseTensor epsilon_tmp;
  phi::DenseTensorMeta meta = {phi::DataType::FLOAT32, {1}};
  beta1_tmp.Resize({1});
  beta2_tmp.Resize({1});
  epsilon_tmp.Resize({1});

  MPDType beta1 = beta1_in.to<MPDType>();
  dev_ctx.template Alloc<MPDType>(&beta1_tmp);
  FillMLUTensorWithHostValue<MPDType>(dev_ctx, beta1, &beta1_tmp);
  beta1_tensor = &beta1_tmp;

  MPDType beta2 = beta2_in.to<MPDType>();
  dev_ctx.template Alloc<MPDType>(&beta2_tmp);
  FillMLUTensorWithHostValue<MPDType>(dev_ctx, beta2, &beta2_tmp);
  beta2_tensor = &beta2_tmp;

  MPDType epsilon = epsilon_in.to<MPDType>();
  dev_ctx.template Alloc<MPDType>(&epsilon_tmp);
  FillMLUTensorWithHostValue<MPDType>(dev_ctx, epsilon, &epsilon_tmp);
  epsilon_tensor = &epsilon_tmp;

  Tensor t_param_in_out, t_grad;
  t_param_in_out.Resize(param.dims());
  t_grad.Resize(grad.dims());
  if (multi_precision) {
    // for multi_precision attribute, master_param_out should be float32.
    dev_ctx.template Alloc<MPDType>(master_param_out);
    TensorCopy(dev_ctx, master_param.get(), false, master_param_out);
    t_param_in_out = *master_param_out;
  } else if (param.dtype() != phi::DataType::FLOAT32) {
    // cast param(T) to t_param_in_out(MPDType)
    dev_ctx.template Alloc<MPDType>(&t_param_in_out);
    MLUCnnlTensorDesc param_out_desc(param);
    MLUCnnlTensorDesc param_in_out_desc(t_param_in_out);
    cnnlCastDataType_t cast_type =
        GetCastDataType(param.dtype(), DataType::FLOAT32);
    MLUCnnl::Cast(dev_ctx,
                  cast_type,
                  param_out_desc.get(),
                  GetBasePtr(&param),
                  param_in_out_desc.get(),
                  GetBasePtr(&t_param_in_out));
  } else {
    *param_out = param;
    t_param_in_out = *param_out;
  }

  if (grad.dtype() == phi::DataType::FLOAT16) {  // cast half grad to float
    dev_ctx.template Alloc<MPDType>(&t_grad);
    MLUCnnlTensorDesc grad_desc(grad);
    MLUCnnlTensorDesc cast_grad_desc(t_grad);
    cnnlCastDataType_t cast_type =
        GetCastDataType(grad.dtype(), DataType::FLOAT32);
    MLUCnnl::Cast(dev_ctx,
                  cast_type,
                  grad_desc.get(),
                  GetBasePtr(&grad),
                  cast_grad_desc.get(),
                  GetBasePtr(&t_grad));
  } else {
    t_grad = grad;
  }

  MLUCnnlTensorDesc param_desc(t_param_in_out);
  MLUCnnlTensorDesc mom1_desc(moment1);
  MLUCnnlTensorDesc mom2_desc(moment2);
  MLUCnnlTensorDesc grad_desc(t_grad);
  MLUCnnl::ApplyAdam(dev_ctx,
                     param_desc.get(),
                     GetBasePtr(&t_param_in_out),
                     mom1_desc.get(),
                     GetBasePtr(moment1_out),
                     mom2_desc.get(),
                     GetBasePtr(moment2_out),
                     grad_desc.get(),
                     GetBasePtr(&t_grad),
                     GetBasePtr(&learning_rate),
                     GetBasePtr(beta1_tensor),
                     GetBasePtr(beta2_tensor),
                     GetBasePtr(beta1_pow),
                     GetBasePtr(beta2_pow),
                     GetBasePtr(epsilon_tensor),
                     /*use_nesterov*/ false);

  if (param.dtype() != phi::DataType::FLOAT32) {
    // 1. cast param_in_out(MPDType) to param_out(T) anyway.
    phi::DenseTensorMeta meta = {param.dtype(), param.dims()};
    param_out->set_meta(meta);
    dev_ctx.template Alloc<T>(param_out);
    MLUCnnlTensorDesc param_out_desc(*param_out);
    MLUCnnlTensorDesc param_in_out_desc(t_param_in_out);
    cnnlCastDataType_t cast_type =
        GetCastDataType(DataType::FLOAT32, param_out->dtype());
    MLUCnnl::Cast(dev_ctx,
                  cast_type,
                  param_in_out_desc.get(),
                  GetBasePtr(&t_param_in_out),
                  param_out_desc.get(),
                  GetBasePtr(param_out));
  }

  if (!use_global_beta_pow) {
    if (beta1_pow->place().GetType() == phi::AllocationType::CPU &&
        beta2_pow->place().GetType() == phi::AllocationType::CPU) {
      // cpu update
      dev_ctx.template HostAlloc<MPDType>(beta1_pow_out)[0] =
          beta1 * beta1_pow->data<MPDType>()[0];
      dev_ctx.template HostAlloc<MPDType>(beta2_pow_out)[0] =
          beta2 * beta2_pow->data<MPDType>()[0];
    } else {
      // mlu update
      dev_ctx.template Alloc<MPDType>(beta1_pow_out);
      dev_ctx.template Alloc<MPDType>(beta2_pow_out);

      MLUCnnlTensorDesc beta1_desc(*beta1_tensor);
      MLUCnnlOpTensorDesc mul_op_desc(CNNL_OP_TENSOR_MUL,
                                      ToCnnlDataType<MPDType>(),
                                      CNNL_NOT_PROPAGATE_NAN);

      MLUCnnl::OpTensor(dev_ctx,
                        mul_op_desc.get(),
                        beta1_desc.get(),
                        GetBasePtr(beta1_pow),
                        beta1_desc.get(),
                        GetBasePtr(beta1_tensor),
                        beta1_desc.get(),
                        GetBasePtr(beta1_pow_out),
                        ToCnnlDataType<MPDType>());

      MLUCnnl::OpTensor(dev_ctx,
                        mul_op_desc.get(),
                        beta1_desc.get(),
                        GetBasePtr(beta2_pow),
                        beta1_desc.get(),
                        GetBasePtr(beta2_tensor),
                        beta1_desc.get(),
                        GetBasePtr(beta2_pow_out),
                        ToCnnlDataType<MPDType>());
    }
  }
}

template <typename T, typename Context>
void AdamWKernel(
    const Context& dev_ctx,
    const phi::DenseTensor& param,
    const phi::DenseTensor& grad,
    const phi::DenseTensor& learning_rate,
    const phi::DenseTensor& moment1,
    const phi::DenseTensor& moment2,
    const paddle::optional<phi::DenseTensor>& moment2_max,  // UNUSED
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
    bool amsgrad,  // UNUSED
    phi::DenseTensor* param_out,
    phi::DenseTensor* moment1_out,
    phi::DenseTensor* moment2_out,
    phi::DenseTensor* moment2_max_out,  // UNUSED
    phi::DenseTensor* beta1_pow_out,
    phi::DenseTensor* beta2_pow_out,
    phi::DenseTensor* master_param_outs) {
  PADDLE_ENFORCE_NE(
      amsgrad,
      true,
      phi::errors::Unimplemented("Operation amsgrad is not supported yet."));
  using MPDType = typename MPTypeTrait<T>::Type;
  bool skip_update_ = false;
  if (skip_update.is_initialized()) {
    PADDLE_ENFORCE_EQ(skip_update->numel(),
                      1,
                      phi::errors::InvalidArgument(
                          "Input(SkipUpdate) size must be 1, but get %d",
                          skip_update->numel()));
    std::vector<bool> skip_update_vec;
    TensorToVector(dev_ctx, *skip_update, dev_ctx, &skip_update_vec);
    dev_ctx.Wait();
    skip_update_ = skip_update_vec[0];
  }

  VLOG(3) << "Skip update" << skip_update_ << ", With decay: " << with_decay;
  if (skip_update_) {
    *param_out = param;
    *moment1_out = moment1;
    *moment2_out = moment2;
    if (!use_global_beta_pow) {
      *beta1_pow_out = beta1_pow;
      *beta2_pow_out = beta2_pow;
    }
    return;
  }

  Tensor t_param_bak, t_lr;
  t_param_bak.Resize(param.dims());
  t_lr.Resize(param.dims());
  dev_ctx.template Alloc<MPDType>(&t_lr);
  FillMLUTensorWithDeviceValue<MPDType>(
      dev_ctx,
      reinterpret_cast<MPDType*>(const_cast<void*>(GetBasePtr(&learning_rate))),
      &t_lr);
  if (multi_precision) {
    dev_ctx.template Alloc<MPDType>(&t_param_bak);
    TensorCopy(dev_ctx, master_param.get(), false, &t_param_bak);
  } else {
    dev_ctx.template Alloc<T>(&t_param_bak);
    TensorCopy(dev_ctx, param, false, &t_param_bak);
  }

  // do adam, then decay
  custom_kernel::AdamKernel<T, Context>(dev_ctx,
                                        param,
                                        grad,
                                        learning_rate,
                                        moment1,
                                        moment2,
                                        moment2_max,
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
                                        amsgrad,
                                        param_out,
                                        moment1_out,
                                        moment2_out,
                                        moment2_max_out,
                                        beta1_pow_out,
                                        beta2_pow_out,
                                        master_param_outs);
  if (!skip_update_ && with_decay) {
    MLUCnnlTensorDesc lr_desc(t_lr);
    MLUCnnlTensorDesc param_desc(t_param_bak);
    MLUCnnlOpTensorDesc mul_op_desc(
        CNNL_OP_TENSOR_MUL, ToCnnlDataType<T>(), CNNL_NOT_PROPAGATE_NAN);

    if (multi_precision) {
      MLUCnnlTensorDesc out_desc(*master_param_outs);
      MLUCnnl::OpTensor(dev_ctx,
                        mul_op_desc.get(),
                        param_desc.get(),
                        GetBasePtr(&t_param_bak),
                        lr_desc.get(),
                        GetBasePtr(&t_lr),
                        out_desc.get(),
                        const_cast<void*>(GetBasePtr(master_param_outs)),
                        ToCnnlDataType<T>(),
                        /*alpha1*/ -1.f,
                        /*alpha2*/ coeff,
                        /*beta*/ 1.f);
    } else {
      MLUCnnlTensorDesc out_desc(*param_out);
      MLUCnnl::OpTensor(dev_ctx,
                        mul_op_desc.get(),
                        param_desc.get(),
                        GetBasePtr(&t_param_bak),
                        lr_desc.get(),
                        GetBasePtr(&t_lr),
                        out_desc.get(),
                        const_cast<void*>(GetBasePtr(param_out)),
                        ToCnnlDataType<T>(),
                        /*alpha1*/ -1.f,
                        /*alpha2*/ coeff,
                        /*beta*/ 1.f);
    }
  }
}

template <typename T, typename Context>
void MergedAdamKernel(
    const Context& dev_ctx,
    const std::vector<const phi::DenseTensor*>& param,
    const std::vector<const phi::DenseTensor*>& grad,
    const std::vector<const phi::DenseTensor*>& learning_rate,
    const std::vector<const phi::DenseTensor*>& moment1,
    const std::vector<const phi::DenseTensor*>& moment2,
    const paddle::optional<std::vector<const phi::DenseTensor*>>& moment2_max,
    const std::vector<const phi::DenseTensor*>& beta1_pow,
    const std::vector<const phi::DenseTensor*>& beta2_pow,
    const paddle::optional<std::vector<const phi::DenseTensor*>>& master_param,
    const phi::Scalar& beta1,
    const phi::Scalar& beta2,
    const phi::Scalar& epsilon,
    bool multi_precision,
    bool use_global_beta_pow,
    bool amsgrad,
    std::vector<phi::DenseTensor*> param_out,
    std::vector<phi::DenseTensor*> moment1_out,
    std::vector<phi::DenseTensor*> moment2_out,
    std::vector<phi::DenseTensor*> moment2_max_out,
    std::vector<phi::DenseTensor*> beta1_pow_out,
    std::vector<phi::DenseTensor*> beta2_pow_out,
    std::vector<phi::DenseTensor*> master_param_out) {
  size_t param_num = param.size();
  PADDLE_ENFORCE_EQ(param_num,
                    grad.size(),
                    phi::errors::InvalidArgument(
                        "The size of Input(grad) must be equal to "
                        "Input(param), but got the size of Input(grad) "
                        "is %d, the size of Input(param) is %d.",
                        grad.size(),
                        param_num));
  PADDLE_ENFORCE_EQ(
      param_num,
      learning_rate.size(),
      phi::errors::InvalidArgument(
          "The size of Input(learning_rate) must be equal to "
          "Input(param), but got the size of Input(learning_rate) "
          "is %d, the size of Input(param) is %d.",
          learning_rate.size(),
          param_num));
  PADDLE_ENFORCE_EQ(param_num,
                    moment1.size(),
                    phi::errors::InvalidArgument(
                        "The size of Input(moment1) must be equal to "
                        "Input(param), but got the size of Input(moment1) "
                        "is %d, the size of Input(param) is %d.",
                        moment1.size(),
                        param_num));
  PADDLE_ENFORCE_EQ(param_num,
                    moment2.size(),
                    phi::errors::InvalidArgument(
                        "The size of Input(moment2) must be equal to "
                        "Input(param), but got the size of Input(moment2) "
                        "is %d, the size of Input(param) is %d.",
                        moment2.size(),
                        param_num));
  PADDLE_ENFORCE_EQ(param_num,
                    beta1_pow.size(),
                    phi::errors::InvalidArgument(
                        "The size of Input(beta1_pow) must be equal to "
                        "Input(param), but got the size of Input(beta1_pow) "
                        "is %d, the size of Input(param) is %d.",
                        beta1_pow.size(),
                        param_num));
  PADDLE_ENFORCE_EQ(param_num,
                    beta2_pow.size(),
                    phi::errors::InvalidArgument(
                        "The size of Input(beta2_pow) must be equal to "
                        "Input(param), but got the size of Input(beta2_pow) "
                        "is %d, the size of Input(param) is %d.",
                        beta2_pow.size(),
                        param_num));

  VLOG(4) << "use_global_beta_pow:" << use_global_beta_pow;
  PADDLE_ENFORCE_NE(
      amsgrad,
      true,
      phi::errors::Unimplemented("Operation amsgrad is not supported yet."));

  const phi::DenseTensor* beta1_tensor = nullptr;
  const phi::DenseTensor* beta2_tensor = nullptr;
  const phi::DenseTensor* epsilon_tensor = nullptr;

  phi::DenseTensor beta1_tmp;
  phi::DenseTensor beta2_tmp;
  phi::DenseTensor epsilon_tmp;
  phi::DenseTensorMeta meta = {phi::DataType::FLOAT32, {1}};
  beta1_tmp.set_meta(meta);
  beta2_tmp.set_meta(meta);
  epsilon_tmp.set_meta(meta);

  T beta1_ = beta1.to<T>();
  dev_ctx.template Alloc<T>(&beta1_tmp);
  FillMLUTensorWithHostValue<T>(dev_ctx, beta1_, &beta1_tmp);
  beta1_tensor = &beta1_tmp;

  T beta2_ = beta2.to<T>();
  dev_ctx.template Alloc<T>(&beta2_tmp);
  FillMLUTensorWithHostValue<T>(dev_ctx, beta2_, &beta2_tmp);
  beta2_tensor = &beta2_tmp;

  T epsilon_ = epsilon.to<T>();
  dev_ctx.template Alloc<T>(&epsilon_tmp);
  FillMLUTensorWithHostValue<T>(dev_ctx, epsilon_, &epsilon_tmp);
  epsilon_tensor = &epsilon_tmp;

  for (size_t idx = 0; idx < param_num; idx++) {
    VLOG(4) << "[MergedAdam] loop: " << idx;
    *param_out[idx] = *param[idx];
    *moment1_out[idx] = *moment1[idx];
    *moment2_out[idx] = *moment2[idx];

    phi::DenseTensor* beta1_pow_tensor =
        const_cast<phi::DenseTensor*>(beta1_pow[idx]);
    phi::DenseTensor* beta2_pow_tensor =
        const_cast<phi::DenseTensor*>(beta2_pow[idx]);
    phi::DenseTensor beta1_pow_tmp;
    phi::DenseTensor beta2_pow_tmp;
    if (beta1_pow_tensor->place().GetType() == phi::AllocationType::CPU) {
      T beta1_pow_ = *beta1_pow_tensor->data<T>();
      beta1_pow_tmp.Resize({1});
      dev_ctx.template Alloc<T>(&beta1_pow_tmp);
      FillMLUTensorWithHostValue(dev_ctx, beta1_pow_, &beta1_pow_tmp);
      beta1_pow_tensor = &beta1_pow_tmp;
    }
    if (beta2_pow_tensor->place().GetType() == phi::AllocationType::CPU) {
      T beta2_pow_ = *beta2_pow_tensor->data<T>();
      beta2_pow_tmp.Resize({1});
      dev_ctx.template Alloc<T>(&beta2_pow_tmp);
      FillMLUTensorWithHostValue(dev_ctx, beta2_pow_, &beta2_pow_tmp);
      beta2_pow_tensor = &beta2_pow_tmp;
    }

    VLOG(3) << "beta1_pow_tensor.numel() : " << beta1_pow_tensor->numel()
            << "beta2_pow_tensor.numel() : " << beta2_pow_tensor->numel();
    VLOG(3) << "param.numel(): " << param[idx]->numel();
    PADDLE_ENFORCE_EQ(beta1_pow_out[idx]->numel(),
                      1,
                      phi::errors::InvalidArgument(
                          "beta1 pow output size should be 1, but received "
                          "value is:%d.",
                          beta1_pow_out[idx]->numel()));

    PADDLE_ENFORCE_EQ(beta2_pow_out[idx]->numel(),
                      1,
                      phi::errors::InvalidArgument(
                          "beta2 pow output size should be 1, but received "
                          "value is:%d.",
                          beta2_pow_out[idx]->numel()));
    MLUCnnlTensorDesc param_desc(*param[idx]);
    MLUCnnlTensorDesc mom1_desc(*moment1[idx]);
    MLUCnnlTensorDesc mom2_desc(*moment2[idx]);
    MLUCnnlTensorDesc grad_desc(*grad[idx]);
    MLUCnnl::ApplyAdam(dev_ctx,
                       param_desc.get(),
                       GetBasePtr(param_out[idx]),
                       mom1_desc.get(),
                       GetBasePtr(moment1_out[idx]),
                       mom2_desc.get(),
                       GetBasePtr(moment2_out[idx]),
                       grad_desc.get(),
                       GetBasePtr(grad[idx]),
                       GetBasePtr(learning_rate[idx]),
                       GetBasePtr(beta1_tensor),
                       GetBasePtr(beta2_tensor),
                       GetBasePtr(beta1_pow_tensor),
                       GetBasePtr(beta2_pow_tensor),
                       GetBasePtr(epsilon_tensor),
                       /*use_nesterov*/ false);
    if (!use_global_beta_pow) {
      dev_ctx.template Alloc<T>(beta1_pow_out[idx]);
      dev_ctx.template Alloc<T>(beta2_pow_out[idx]);

      MLUCnnlTensorDesc beta1_desc(*beta1_tensor);
      MLUCnnlOpTensorDesc mul_op_desc(
          CNNL_OP_TENSOR_MUL, ToCnnlDataType<T>(), CNNL_NOT_PROPAGATE_NAN);

      MLUCnnl::OpTensor(dev_ctx,
                        mul_op_desc.get(),
                        beta1_desc.get(),
                        GetBasePtr(beta1_pow_tensor),
                        beta1_desc.get(),
                        GetBasePtr(beta1_tensor),
                        beta1_desc.get(),
                        GetBasePtr(beta1_pow_out[idx]),
                        ToCnnlDataType<T>());

      MLUCnnl::OpTensor(dev_ctx,
                        mul_op_desc.get(),
                        beta1_desc.get(),
                        GetBasePtr(beta2_pow_tensor),
                        beta1_desc.get(),
                        GetBasePtr(beta2_tensor),
                        beta1_desc.get(),
                        GetBasePtr(beta2_pow_out[idx]),
                        ToCnnlDataType<T>());
    }
  }
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(adam,
                          mlu,
                          ALL_LAYOUT,
                          custom_kernel::AdamKernel,
                          phi::dtype::float16,
                          float) {
  // Skip beta1_pow, beta2_pow, skip_update data transform
  kernel->InputAt(6).SetBackend(phi::Backend::ALL_BACKEND);
  kernel->InputAt(7).SetBackend(phi::Backend::ALL_BACKEND);
  kernel->InputAt(9).SetBackend(phi::Backend::ALL_BACKEND);
  if (kernel_key.dtype() == phi::DataType::FLOAT16) {
    kernel->OutputAt(1).SetDataType(phi::DataType::FLOAT32);
    kernel->OutputAt(2).SetDataType(phi::DataType::FLOAT32);
    kernel->OutputAt(3).SetDataType(phi::DataType::FLOAT32);
    kernel->OutputAt(4).SetDataType(phi::DataType::FLOAT32);
    kernel->OutputAt(5).SetDataType(phi::DataType::FLOAT32);
    kernel->OutputAt(6).SetDataType(phi::DataType::FLOAT32);
  }
  kernel->OutputAt(4).SetBackend(phi::Backend::UNDEFINED);
  kernel->OutputAt(5).SetBackend(phi::Backend::UNDEFINED);
}

PD_REGISTER_PLUGIN_KERNEL(adamw,
                          mlu,
                          ALL_LAYOUT,
                          custom_kernel::AdamWKernel,
                          phi::dtype::float16,
                          float) {
  // Skip beta1_pow, beta2_pow, skip_update data transform
  kernel->InputAt(6).SetBackend(phi::Backend::ALL_BACKEND);
  kernel->InputAt(7).SetBackend(phi::Backend::ALL_BACKEND);
  kernel->InputAt(9).SetBackend(phi::Backend::ALL_BACKEND);
  if (kernel_key.dtype() == phi::DataType::FLOAT16) {
    kernel->OutputAt(1).SetDataType(phi::DataType::FLOAT32);
    kernel->OutputAt(2).SetDataType(phi::DataType::FLOAT32);
    kernel->OutputAt(3).SetDataType(phi::DataType::FLOAT32);
    kernel->OutputAt(4).SetDataType(phi::DataType::FLOAT32);
    kernel->OutputAt(5).SetDataType(phi::DataType::FLOAT32);
    kernel->OutputAt(6).SetDataType(phi::DataType::FLOAT32);
  }
  kernel->OutputAt(4).SetBackend(phi::Backend::UNDEFINED);
  kernel->OutputAt(5).SetBackend(phi::Backend::UNDEFINED);
}

PD_REGISTER_PLUGIN_KERNEL(merged_adam,
                          mlu,
                          ALL_LAYOUT,
                          custom_kernel::MergedAdamKernel,
                          phi::dtype::float16,
                          float) {}
