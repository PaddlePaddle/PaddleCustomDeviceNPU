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
#include "paddle/phi/kernels/funcs/tensor_formatter.h"

namespace custom_kernel {

using Tensor = phi::DenseTensor;
using DDim = phi::DDim;

using DataLayout = phi::DataLayout;

template <typename T>
class NormDataType;

template <>
class NormDataType<phi::dtype::float16> {
 public:
  // The scaling param type is float for HALF and FLOAT tensors
  using ScalingParamType = const float;
  using BatchNormParamType = float;
};

template <>
class NormDataType<phi::dtype::bfloat16> {
 public:
  // The scaling param type is float for HALF and FLOAT tensors
  using ScalingParamType = const float;
  using BatchNormParamType = float;
};

template <>
class NormDataType<float> {
 public:
  using ScalingParamType = const float;
  using BatchNormParamType = float;
};

template <typename T>
using NormDataType = NormDataType<T>;
template <typename T>
using LayerNormParamType = typename NormDataType<T>::BatchNormParamType;

template <typename T, typename Context>
void CastKernel(const Context& dev_ctx,
                const phi::DenseTensor& x,
                phi::DataType dtype,
                phi::DenseTensor* out);

template <typename T, typename Context>
void FillKernel(const Context& dev_ctx,
                const phi::DenseTensor& x,
                const phi::Scalar& val);

template <typename T, typename Context>
void AclopLayerNormNPUKernel(
    const Context& dev_ctx,
    const phi::DenseTensor& x,
    const paddle::optional<phi::DenseTensor>& scale_opt,
    const paddle::optional<phi::DenseTensor>& bias_opt,
    float epsilon,
    int begin_norm_axis,
    phi::DenseTensor* out,
    phi::DenseTensor* mean,
    phi::DenseTensor* variance) {
  using U = LayerNormParamType<T>;
  auto* scale = scale_opt.get_ptr();
  auto* bias = bias_opt.get_ptr();
  const auto& x_dims = x.dims();
  std::vector<int> axes;
  auto matrix_dim = phi::flatten_to_2d(x_dims, begin_norm_axis);
  int right = static_cast<int>(matrix_dim[1]);

  // The shape of scale and bias should be equal to x.shape[begin_norm_axis:],
  // required by npu.
  for (auto i = begin_norm_axis; i < x_dims.size(); ++i) {
    axes.push_back(x_dims[i]);
  }

  auto stream = dev_ctx.stream();

  Tensor default_scale;
  if (!scale) {
    phi::DenseTensorMeta default_scale_meta = {x.dtype(), phi::make_ddim(axes)};
    default_scale.set_meta(default_scale_meta);
    dev_ctx.template Alloc<T>(&default_scale);

    if (default_scale.numel() > 1) {
      Tensor value;
      phi::DenseTensorMeta value_meta = {x.dtype(), phi::make_ddim({1})};
      value.set_meta(value_meta);
      dev_ctx.template Alloc<T>(&value);

      FillNpuTensorWithConstant<T>(&value, dev_ctx, static_cast<T>(1.0));

      NpuOpRunner runner;
      runner.SetType("Fill")
          .AddInput(dev_ctx, std::move(axes))
          .AddInput(value)
          .AddOutput(default_scale);
      runner.Run(stream);
    } else {
      // CANN op Fill/FillD would raise error when output's numel is 1.
      FillNpuTensorWithConstant<T>(
          &default_scale, dev_ctx, static_cast<T>(1.0));
    }
    scale = &default_scale;
  } else {
    const_cast<Tensor*>(scale)->Resize(phi::make_ddim(axes));
  }

  Tensor default_bias;
  if (!bias) {
    phi::DenseTensorMeta default_bias_meta = {x.dtype(), phi::make_ddim(axes)};
    default_bias.set_meta(default_bias_meta);
    dev_ctx.template Alloc<T>(&default_bias);

    if (default_bias.numel() > 1) {
      Tensor value;
      phi::DenseTensorMeta value_meta = {x.dtype(), phi::make_ddim({1})};
      value.set_meta(value_meta);
      dev_ctx.template Alloc<T>(&value);

      FillNpuTensorWithConstant<T>(&value, dev_ctx, static_cast<T>(0));

      NpuOpRunner runner;
      runner.SetType("Fill")
          .AddInput(dev_ctx, std::move(axes))
          .AddInput(value)
          .AddOutput(default_bias);
      runner.Run(stream);
    } else {
      // CANN op Fill/FillD would raise error when output's numel is 1.
      FillNpuTensorWithConstant<T>(&default_bias, dev_ctx, static_cast<T>(0));
    }
    bias = &default_bias;
  } else {
    const_cast<Tensor*>(bias)->Resize(phi::make_ddim(axes));
  }

  // cast scale from LayerNormParamType to T if needed
  Tensor cast_scale;
  if (x.dtype() == phi::DataType::FLOAT16 &&
      scale->dtype() == phi::DataType::FLOAT32) {
    cast_scale.Resize(scale->dims());
    dev_ctx.template Alloc<T>(&cast_scale);

    auto dst_dtype = ConvertToNpuDtype(x.dtype());
    const auto& runner_cast_scale =
        NpuOpRunner("Cast",
                    {*scale},
                    {cast_scale},
                    {{"dst_type", static_cast<int>(dst_dtype)}});
    runner_cast_scale.Run(stream);
  } else {
    cast_scale = *scale;
  }

  // cast bias from LayerNormParamType to T if needed
  Tensor cast_bias;
  if (x.dtype() == phi::DataType::FLOAT16 &&
      bias->dtype() == phi::DataType::FLOAT32) {
    cast_bias.Resize(bias->dims());
    dev_ctx.template Alloc<T>(&cast_bias);

    auto dst_dtype = ConvertToNpuDtype(x.dtype());
    const auto& runner_cast_bias =
        NpuOpRunner("Cast",
                    {*bias},
                    {cast_bias},
                    {{"dst_type", static_cast<int>(dst_dtype)}});
    runner_cast_bias.Run(stream);
  } else {
    cast_bias = *bias;
  }

  dev_ctx.template Alloc<T>(out);

  // mean should be of  U type
  Tensor* tmp_mean = mean;
  Tensor cast_mean;
  if (x.dtype() == phi::DataType::FLOAT16 &&
      (scale->dtype() == phi::DataType::FLOAT32 ||
       bias->dtype() == phi::DataType::FLOAT32)) {
    cast_mean.Resize(mean->dims());
    dev_ctx.template Alloc<T>(&cast_mean);
    tmp_mean = &cast_mean;
    dev_ctx.template Alloc<U>(mean);
  } else {
    dev_ctx.template Alloc<T>(mean);
  }

  // same for variance
  Tensor* tmp_variance = variance;
  Tensor cast_variance;
  if (x.dtype() == phi::DataType::FLOAT16 &&
      (scale->dtype() == phi::DataType::FLOAT32 ||
       bias->dtype() == phi::DataType::FLOAT32)) {
    cast_variance.Resize(variance->dims());
    dev_ctx.template Alloc<T>(&cast_variance);
    tmp_variance = &cast_variance;
    dev_ctx.template Alloc<U>(variance);
  } else {
    dev_ctx.template Alloc<T>(variance);
  }

  const auto& runner = NpuOpRunner("LayerNorm",
                                   {x, cast_scale, cast_bias},
                                   {*out, *tmp_mean, *tmp_variance},
                                   {{"begin_norm_axis", begin_norm_axis},
                                    {"begin_params_axis", begin_norm_axis},
                                    {"epsilon", epsilon}});
  runner.Run(stream);

  // cast back from FLOAT16 to FLOAT32
  if (x.dtype() == phi::DataType::FLOAT16 &&
      mean->dtype() == phi::DataType::FLOAT32) {
    auto dst_dtype = ConvertToNpuDtype(mean->dtype());
    const auto& runner_cast_mean =
        NpuOpRunner("Cast",
                    {*tmp_mean},
                    {*mean},
                    {{"dst_type", static_cast<int>(dst_dtype)}});
    runner_cast_mean.Run(stream);
  }
  // same for variance
  if (x.dtype() == phi::DataType::FLOAT16 &&
      variance->dtype() == phi::DataType::FLOAT32) {
    auto dst_dtype = ConvertToNpuDtype(variance->dtype());
    const auto& runner_cast_variance =
        NpuOpRunner("Cast",
                    {*tmp_variance},
                    {*variance},
                    {{"dst_type", static_cast<int>(dst_dtype)}});
    runner_cast_variance.Run(stream);
  }

  // revert shape of scale and bias
  // TODO(zhiqiu): better implementation, use tmp tensor to avoid write input
  // tensor.
  const_cast<Tensor*>(scale)->Resize(phi::make_ddim({right}));
  const_cast<Tensor*>(bias)->Resize(phi::make_ddim({right}));
}

template <typename T, typename Context>
void AclnnLayerNormNPUKernel(
    const Context& dev_ctx,
    const phi::DenseTensor& x,
    const paddle::optional<phi::DenseTensor>& scale_opt,
    const paddle::optional<phi::DenseTensor>& bias_opt,
    float epsilon,
    int begin_norm_axis,
    phi::DenseTensor* out,
    phi::DenseTensor* mean,
    phi::DenseTensor* variance) {
  using U = LayerNormParamType<T>;
  auto* scale = scale_opt.get_ptr();
  auto* bias = bias_opt.get_ptr();
  const auto& x_dims = x.dims();
  std::vector<int> axes;
  auto matrix_dim = phi::flatten_to_2d(x_dims, begin_norm_axis);
  int right = static_cast<int>(matrix_dim[1]);

  // The shape of scale and bias should be equal to x.shape[begin_norm_axis:],
  // required by npu.
  for (auto i = begin_norm_axis; i < x_dims.size(); ++i) {
    axes.push_back(x_dims[i]);
  }
  auto normalizedShape = phi::vectorize<int64_t>(phi::make_ddim(axes));
  auto stream = dev_ctx.stream();

  Tensor default_scale;
  if (!scale) {
    phi::DenseTensorMeta default_scale_meta = {x.dtype(), phi::make_ddim(axes)};
    default_scale.set_meta(default_scale_meta);
    dev_ctx.template Alloc<T>(&default_scale);

    if (default_scale.numel() > 1) {
      Tensor value;
      phi::DenseTensorMeta value_meta = {x.dtype(), phi::make_ddim({1})};
      value.set_meta(value_meta);
      dev_ctx.template Alloc<T>(&value);

      FillNpuTensorWithConstant<T>(&value, dev_ctx, static_cast<T>(1.0));

      NpuOpRunner runner;
      runner.SetType("Fill")
          .AddInput(dev_ctx, std::move(axes))
          .AddInput(value)
          .AddOutput(default_scale);
      runner.Run(stream);
    } else {
      // CANN op Fill/FillD would raise error when output's numel is 1.
      FillNpuTensorWithConstant<T>(
          &default_scale, dev_ctx, static_cast<T>(1.0));
    }
    scale = &default_scale;
  } else {
    const_cast<Tensor*>(scale)->Resize(phi::make_ddim(axes));
  }

  Tensor default_bias;
  if (!bias) {
    phi::DenseTensorMeta default_bias_meta = {x.dtype(), phi::make_ddim(axes)};
    default_bias.set_meta(default_bias_meta);
    dev_ctx.template Alloc<T>(&default_bias);

    if (default_bias.numel() > 1) {
      Tensor value;
      phi::DenseTensorMeta value_meta = {x.dtype(), phi::make_ddim({1})};
      value.set_meta(value_meta);
      dev_ctx.template Alloc<T>(&value);

      FillNpuTensorWithConstant<T>(&value, dev_ctx, static_cast<T>(0));

      NpuOpRunner runner;
      runner.SetType("Fill")
          .AddInput(dev_ctx, std::move(axes))
          .AddInput(value)
          .AddOutput(default_bias);
      runner.Run(stream);
      // custom_kernel::FillKernel<T, Context>(dev_ctx, default_bias, value);

    } else {
      // CANN op Fill/FillD would raise error when output's numel is 1.
      FillNpuTensorWithConstant<T>(&default_bias, dev_ctx, static_cast<T>(0));
    }
    bias = &default_bias;
  } else {
    const_cast<Tensor*>(bias)->Resize(phi::make_ddim(axes));
  }

  // cast scale from LayerNormParamType to T if needed
  Tensor cast_scale;
  if (x.dtype() == phi::DataType::FLOAT16 &&
      scale->dtype() == phi::DataType::FLOAT32) {
    cast_scale.Resize(scale->dims());
    dev_ctx.template Alloc<T>(&cast_scale);

    custom_kernel::CastKernel<T, Context>(
        dev_ctx, *scale, x.dtype(), &cast_scale);
  } else {
    cast_scale = *scale;
  }

  // cast bias from LayerNormParamType to T if needed
  Tensor cast_bias;
  if (x.dtype() == phi::DataType::FLOAT16 &&
      bias->dtype() == phi::DataType::FLOAT32) {
    cast_bias.Resize(bias->dims());
    dev_ctx.template Alloc<T>(&cast_bias);

    custom_kernel::CastKernel<T, Context>(
        dev_ctx, *bias, x.dtype(), &cast_bias);
  } else {
    cast_bias = *bias;
  }
  out->Resize(x.dims());
  dev_ctx.template Alloc<T>(out);

  // mean should be of  U type
  Tensor* tmp_mean = mean;
  Tensor cast_mean;
  if (x.dtype() == phi::DataType::FLOAT16 &&
      (scale->dtype() == phi::DataType::FLOAT32 ||
       bias->dtype() == phi::DataType::FLOAT32)) {
    cast_mean.Resize(mean->dims());
    dev_ctx.template Alloc<T>(&cast_mean);
    tmp_mean = &cast_mean;
    dev_ctx.template Alloc<U>(mean);
  } else {
    dev_ctx.template Alloc<T>(mean);
  }

  // same for variance
  Tensor* tmp_variance = variance;
  Tensor cast_variance;
  if (x.dtype() == phi::DataType::FLOAT16 &&
      (scale->dtype() == phi::DataType::FLOAT32 ||
       bias->dtype() == phi::DataType::FLOAT32)) {
    cast_variance.Resize(variance->dims());
    dev_ctx.template Alloc<T>(&cast_variance);
    tmp_variance = &cast_variance;
    dev_ctx.template Alloc<U>(variance);
  } else {
    dev_ctx.template Alloc<T>(variance);
  }

  // resize mean and variance to append 1s
  // add 1s after begin norm axis
  std::vector<int64_t> tmp_mean_dim;
  for (auto i = 0; i < x.dims().size(); i++) {
    if (i < begin_norm_axis)
      tmp_mean_dim.push_back(x.dims()[i]);
    else
      tmp_mean_dim.push_back(1);
  }
  auto mean_dims = mean->dims();
  tmp_mean->Resize(phi::make_ddim(tmp_mean_dim));

  // variance dims

  // add 1s after begin norm axis
  std::vector<int64_t> tmp_variance_dim;
  for (auto i = 0; i < x.dims().size(); i++) {
    if (i < begin_norm_axis)
      tmp_variance_dim.push_back(x.dims()[i]);
    else
      tmp_variance_dim.push_back(1);
  }
  auto variance_dims = variance->dims();
  tmp_variance->Resize(phi::make_ddim(tmp_variance_dim));

  double eps = static_cast<double>(epsilon);

  EXEC_NPU_CMD(aclnnLayerNorm,
               dev_ctx,
               x,
               normalizedShape,
               cast_scale,
               cast_bias,
               eps,
               *out,
               *tmp_mean,
               *tmp_variance);  // output rstd

  // mean shape change back
  tmp_mean->Resize(mean_dims);
  tmp_variance->Resize(variance_dims);

  // cast back from FLOAT16 to FLOAT32
  if (x.dtype() == phi::DataType::FLOAT16 &&
      mean->dtype() == phi::DataType::FLOAT32) {
    custom_kernel::CastKernel<T, Context>(
        dev_ctx, *tmp_mean, mean->dtype(), mean);
  }
  // same for variance
  if (x.dtype() == phi::DataType::FLOAT16 &&
      variance->dtype() == phi::DataType::FLOAT32) {
    custom_kernel::CastKernel<T, Context>(
        dev_ctx, *tmp_variance, variance->dtype(), variance);
  }

  // revert shape of scale and bias
  // TODO(zhiqiu): better implementation, use tmp tensor to avoid write input
  // tensor.
  const_cast<Tensor*>(scale)->Resize(phi::make_ddim({right}));
  const_cast<Tensor*>(bias)->Resize(phi::make_ddim({right}));
}

template <typename T, typename Context>
void AclopLayerNormGradNPUKernel(
    const Context& dev_ctx,
    const phi::DenseTensor& x,
    const paddle::optional<phi::DenseTensor>& scale_opt,
    const paddle::optional<phi::DenseTensor>& bias,
    const phi::DenseTensor& mean,
    const phi::DenseTensor& variance,
    const phi::DenseTensor& out_grad,
    float epsilon,
    int begin_norm_axis,
    phi::DenseTensor* x_grad,
    phi::DenseTensor* scale_grad,
    phi::DenseTensor* bias_grad) {
  using U = LayerNormParamType<T>;
  const auto& x_dims = x.dims();
  auto* scale = scale_opt.get_ptr();

  auto matrix_dim = phi::flatten_to_2d(x_dims, begin_norm_axis);
  int right = static_cast<int>(matrix_dim[1]);

  std::vector<int> axes;
  for (auto i = begin_norm_axis; i < x_dims.size(); ++i) {
    axes.push_back(x_dims[i]);
  }

  auto stream = dev_ctx.stream();

  // No need to compute any gradient, jusr return
  if (!x_grad && !scale_grad && !bias_grad) {
    return;
  }

  // The rank of mean should be equal to x, required by npu.
  std::vector<int> new_shape;
  for (auto i = 0; i < begin_norm_axis; ++i) {
    new_shape.push_back(x_dims[i]);
  }
  for (auto i = begin_norm_axis; i < x_dims.size(); ++i) {
    new_shape.push_back(1);
  }

  auto mean_dims = mean.dims();
  const_cast<Tensor*>(&mean)->Resize(phi::make_ddim({new_shape}));
  const_cast<Tensor*>(&variance)->Resize(phi::make_ddim({new_shape}));

  Tensor default_scale;
  if (!scale) {
    phi::DenseTensorMeta default_scale_meta = {x.dtype(), phi::make_ddim(axes)};
    default_scale.set_meta(default_scale_meta);
    dev_ctx.template Alloc<T>(&default_scale);

    if (default_scale.numel() > 0) {
      Tensor value;
      phi::DenseTensorMeta value_meta = {x.dtype(), phi::make_ddim({1})};
      value.set_meta(value_meta);
      dev_ctx.template Alloc<T>(&value);

      FillNpuTensorWithConstant<T>(&value, dev_ctx, static_cast<T>(1.0));

      NpuOpRunner runner;
      runner.SetType("Fill")
          .AddInput(dev_ctx, std::move(axes))
          .AddInput(value)
          .AddOutput(default_scale);
      runner.Run(stream);
    } else {
      // CANN op Fill/FillD would raise error when output's numel is 1.
      FillNpuTensorWithConstant<T>(
          &default_scale, dev_ctx, static_cast<T>(1.0));
    }
    scale = &default_scale;
  } else {
    const_cast<Tensor*>(scale)->Resize(phi::make_ddim(axes));
  }

  // cast scale from LayerNormParamType to T if needed
  Tensor cast_scale;
  if (x.dtype() == phi::DataType::FLOAT16 &&
      scale->dtype() == phi::DataType::FLOAT32) {
    cast_scale.Resize(scale->dims());
    dev_ctx.template Alloc<T>(&cast_scale);

    auto dst_dtype = ConvertToNpuDtype(x.dtype());
    const auto& runner_cast_scale =
        NpuOpRunner("Cast",
                    {*scale},
                    {cast_scale},
                    {{"dst_type", static_cast<int>(dst_dtype)}});
    runner_cast_scale.Run(stream);
  } else {
    cast_scale = *scale;
  }

  // cast mean from LayerNormParamType to T if needed
  Tensor cast_mean;
  if (x.dtype() == phi::DataType::FLOAT16 &&
      mean.dtype() == phi::DataType::FLOAT32) {
    cast_mean.Resize(mean.dims());
    dev_ctx.template Alloc<T>(&cast_mean);
    auto dst_dtype = ConvertToNpuDtype(x.dtype());
    const auto& runner_cast_mean =
        NpuOpRunner("Cast",
                    {mean},
                    {cast_mean},
                    {{"dst_type", static_cast<int>(dst_dtype)}});
    runner_cast_mean.Run(stream);
  } else {
    cast_mean = mean;
  }

  // cast variance from LayerNormParamType to T if needed
  Tensor cast_variance;
  if (x.dtype() == phi::DataType::FLOAT16 &&
      variance.dtype() == phi::DataType::FLOAT32) {
    cast_variance.Resize(variance.dims());
    dev_ctx.template Alloc<T>(&cast_variance);
    auto dst_dtype = ConvertToNpuDtype(x.dtype());
    const auto& runner_cast_variance =
        NpuOpRunner("Cast",
                    {variance},
                    {cast_variance},
                    {{"dst_type", static_cast<int>(dst_dtype)}});
    runner_cast_variance.Run(stream);
  } else {
    cast_variance = variance;
  }

  Tensor x_grad_, scale_grad_, bias_grad_;
  x_grad = (x_grad == nullptr) ? &x_grad_ : x_grad;
  scale_grad = (scale_grad == nullptr) ? &scale_grad_ : scale_grad;
  bias_grad = (bias_grad == nullptr) ? &bias_grad_ : bias_grad;

  x_grad->Resize(x.dims());
  scale_grad->Resize(phi::make_ddim(axes));
  bias_grad->Resize(phi::make_ddim(axes));

  dev_ctx.template Alloc<T>(x_grad);
  // scale_grad should be of  U type
  Tensor* tmp_scale_grad = scale_grad;
  Tensor cast_scale_grad;
  if (x.dtype() == phi::DataType::FLOAT16 &&
      (mean.dtype() == phi::DataType::FLOAT32 ||
       variance.dtype() == phi::DataType::FLOAT32)) {
    cast_scale_grad.Resize(scale_grad->dims());
    dev_ctx.template Alloc<T>(&cast_scale_grad);
    tmp_scale_grad = &cast_scale_grad;
    dev_ctx.template Alloc<U>(scale_grad);
  } else {
    dev_ctx.template Alloc<T>(scale_grad);
  }

  // same for bias_grad
  Tensor* tmp_bias_grad = bias_grad;
  Tensor cast_bias_grad;
  if (x.dtype() == phi::DataType::FLOAT16 &&
      (mean.dtype() == phi::DataType::FLOAT32 ||
       variance.dtype() == phi::DataType::FLOAT32)) {
    cast_bias_grad.Resize(bias_grad->dims());
    dev_ctx.template Alloc<T>(&cast_bias_grad);
    tmp_bias_grad = &cast_bias_grad;
    dev_ctx.template Alloc<U>(bias_grad);
  } else {
    dev_ctx.template Alloc<T>(bias_grad);
  }

  const auto& runner =
      NpuOpRunner("LayerNormGrad",
                  {out_grad, x, cast_variance, cast_mean, cast_scale},
                  {*x_grad, *tmp_scale_grad, *tmp_bias_grad},
                  {});
  runner.Run(stream);

  // cast back from FLOAT16 to FLOAT32
  if (x.dtype() == phi::DataType::FLOAT16 &&
      scale_grad->dtype() == phi::DataType::FLOAT32) {
    auto dst_dtype = ConvertToNpuDtype(scale_grad->dtype());
    const auto& runner_cast_scale_grad =
        NpuOpRunner("Cast",
                    {*tmp_scale_grad},
                    {*scale_grad},
                    {{"dst_type", static_cast<int>(dst_dtype)}});
    runner_cast_scale_grad.Run(stream);
  }
  // same for bias_grad
  if (x.dtype() == phi::DataType::FLOAT16 &&
      bias_grad->dtype() == phi::DataType::FLOAT32) {
    auto dst_dtype = ConvertToNpuDtype(bias_grad->dtype());
    const auto& runner_cast_bias_grad =
        NpuOpRunner("Cast",
                    {*tmp_bias_grad},
                    {*bias_grad},
                    {{"dst_type", static_cast<int>(dst_dtype)}});
    runner_cast_bias_grad.Run(stream);
  }

  const_cast<Tensor*>(&mean)->Resize(mean_dims);
  const_cast<Tensor*>(&variance)->Resize(mean_dims);
  const_cast<Tensor*>(scale)->Resize(phi::make_ddim({right}));
  scale_grad->Resize(phi::make_ddim({right}));
  bias_grad->Resize(phi::make_ddim({right}));
}

template <typename T, typename Context>
void AclnnLayerNormGradNPUKernel(
    const Context& dev_ctx,
    const phi::DenseTensor& x,
    const paddle::optional<phi::DenseTensor>& scale_opt,
    const paddle::optional<phi::DenseTensor>& bias,
    const phi::DenseTensor& mean,
    const phi::DenseTensor& variance,
    const phi::DenseTensor& out_grad,
    float epsilon,
    int begin_norm_axis,
    phi::DenseTensor* x_grad,
    phi::DenseTensor* scale_grad,
    phi::DenseTensor* bias_grad) {
  using U = LayerNormParamType<T>;
  const auto& x_dims = x.dims();
  auto* scale = scale_opt.get_ptr();
  auto* bias_ptr = bias.get_ptr();

  auto matrix_dim = phi::flatten_to_2d(x_dims, begin_norm_axis);
  int right = static_cast<int>(matrix_dim[1]);

  std::vector<int> axes;
  for (auto i = begin_norm_axis; i < x_dims.size(); ++i) {
    axes.push_back(x_dims[i]);
  }

  auto stream = dev_ctx.stream();

  // No need to compute any gradient, jusr return
  if (!x_grad && !scale_grad && !bias_grad) {
    return;
  }

  // The rank of mean should be equal to x, required by npu.
  std::vector<int> new_shape;
  for (auto i = 0; i < begin_norm_axis; ++i) {
    new_shape.push_back(x_dims[i]);
  }
  for (auto i = begin_norm_axis; i < x_dims.size(); ++i) {
    new_shape.push_back(1);
  }

  auto mean_dims = mean.dims();
  const_cast<Tensor*>(&mean)->Resize(phi::make_ddim({new_shape}));
  const_cast<Tensor*>(&variance)->Resize(phi::make_ddim({new_shape}));

  Tensor default_scale;
  if (!scale) {
    phi::DenseTensorMeta default_scale_meta = {x.dtype(), phi::make_ddim(axes)};
    default_scale.set_meta(default_scale_meta);
    dev_ctx.template Alloc<T>(&default_scale);

    if (default_scale.numel() > 0) {
      Tensor value;
      phi::DenseTensorMeta value_meta = {x.dtype(), phi::make_ddim({1})};
      value.set_meta(value_meta);
      dev_ctx.template Alloc<T>(&value);

      FillNpuTensorWithConstant<T>(&value, dev_ctx, static_cast<T>(1.0));

      NpuOpRunner runner;
      runner.SetType("Fill")
          .AddInput(dev_ctx, std::move(axes))
          .AddInput(value)
          .AddOutput(default_scale);
      runner.Run(stream);
    } else {
      // CANN op Fill/FillD would raise error when output's numel is 1.
      FillNpuTensorWithConstant<T>(
          &default_scale, dev_ctx, static_cast<T>(1.0));
    }
    scale = &default_scale;
  } else {
    const_cast<Tensor*>(scale)->Resize(phi::make_ddim(axes));
  }

  // cast scale from LayerNormParamType to T if needed
  Tensor cast_scale;
  if (x.dtype() == phi::DataType::FLOAT16 &&
      scale->dtype() == phi::DataType::FLOAT32) {
    cast_scale.Resize(scale->dims());
    dev_ctx.template Alloc<T>(&cast_scale);
    custom_kernel::CastKernel<T, Context>(
        dev_ctx, *scale, x.dtype(), &cast_scale);
  } else {
    cast_scale = *scale;
  }

  Tensor default_bias;
  phi::DenseTensorMeta default_bias_meta = {x.dtype(), phi::make_ddim(axes)};
  default_bias.set_meta(default_bias_meta);
  dev_ctx.template Alloc<T>(&default_bias);

  if (default_bias.numel() > 0) {
    Tensor bias_value;
    phi::DenseTensorMeta bias_value_meta = {x.dtype(), phi::make_ddim({1})};
    bias_value.set_meta(bias_value_meta);
    dev_ctx.template Alloc<T>(&bias_value);

    FillNpuTensorWithConstant<T>(&bias_value, dev_ctx, static_cast<T>(0));

    NpuOpRunner runner;
    runner.SetType("Fill")
        .AddInput(dev_ctx, std::move(axes))
        .AddInput(bias_value)
        .AddOutput(default_bias);
    runner.Run(stream);
  } else {
    // CANN op Fill/FillD would raise error when output's numel is 1.
    FillNpuTensorWithConstant<T>(&default_bias, dev_ctx, static_cast<T>(0));
  }
  bias_ptr = &default_bias;

  // cast bias from LayerNormParamType to T if needed
  Tensor cast_bias;
  if (x.dtype() == phi::DataType::FLOAT16 &&
      bias_ptr->dtype() == phi::DataType::FLOAT32) {
    cast_bias.Resize(bias_ptr->dims());
    dev_ctx.template Alloc<T>(&cast_bias);
    custom_kernel::CastKernel<T, Context>(
        dev_ctx, *bias_ptr, x.dtype(), &cast_bias);
  } else {
    cast_bias = *bias_ptr;
  }

  // cast mean from LayerNormParamType to T if needed
  Tensor cast_mean;
  if (x.dtype() == phi::DataType::FLOAT16 &&
      mean.dtype() == phi::DataType::FLOAT32) {
    cast_mean.Resize(mean.dims());
    dev_ctx.template Alloc<T>(&cast_mean);
    custom_kernel::CastKernel<T, Context>(dev_ctx, mean, x.dtype(), &cast_mean);
  } else {
    cast_mean = mean;
  }
  // cast variance from LayerNormParamType to T if needed
  Tensor cast_variance;
  if (x.dtype() == phi::DataType::FLOAT16 &&
      variance.dtype() == phi::DataType::FLOAT32) {
    cast_variance.Resize(variance.dims());
    dev_ctx.template Alloc<T>(&cast_variance);
    custom_kernel::CastKernel<T, Context>(
        dev_ctx, variance, x.dtype(), &cast_variance);
  } else {
    cast_variance = variance;
  }
  Tensor x_grad_, scale_grad_, bias_grad_;
  x_grad = (x_grad == nullptr) ? &x_grad_ : x_grad;
  scale_grad = (scale_grad == nullptr) ? &scale_grad_ : scale_grad;
  bias_grad = (bias_grad == nullptr) ? &bias_grad_ : bias_grad;

  x_grad->Resize(x.dims());
  scale_grad->Resize(phi::make_ddim(axes));
  bias_grad->Resize(phi::make_ddim(axes));

  dev_ctx.template Alloc<T>(x_grad);
  // scale_grad should be of  U type
  Tensor* tmp_scale_grad = scale_grad;
  Tensor cast_scale_grad;
  if (x.dtype() == phi::DataType::FLOAT16 &&
      (mean.dtype() == phi::DataType::FLOAT32 ||
       variance.dtype() == phi::DataType::FLOAT32)) {
    cast_scale_grad.Resize(scale_grad->dims());
    dev_ctx.template Alloc<T>(&cast_scale_grad);
    tmp_scale_grad = &cast_scale_grad;
    dev_ctx.template Alloc<U>(scale_grad);
  } else {
    dev_ctx.template Alloc<T>(scale_grad);
  }

  // same for bias_grad
  Tensor* tmp_bias_grad = bias_grad;
  Tensor cast_bias_grad;
  if (x.dtype() == phi::DataType::FLOAT16 &&
      (mean.dtype() == phi::DataType::FLOAT32 ||
       variance.dtype() == phi::DataType::FLOAT32)) {
    cast_bias_grad.Resize(bias_grad->dims());
    dev_ctx.template Alloc<T>(&cast_bias_grad);
    tmp_bias_grad = &cast_bias_grad;
    dev_ctx.template Alloc<U>(bias_grad);
  } else {
    dev_ctx.template Alloc<T>(bias_grad);
  }

  auto normalizedShape = phi::vectorize<int64_t>(phi::make_ddim(axes));
  std::array<bool, 3> output_mask{false};
  if (x_grad) {
    output_mask[0] = true;
  }
  if (tmp_scale_grad) {
    output_mask[1] = true;
  }
  if (tmp_bias_grad) {
    output_mask[2] = true;
  }
  EXEC_NPU_CMD(aclnnLayerNormBackward,
               dev_ctx,
               out_grad,
               x,
               normalizedShape,
               cast_mean,
               cast_variance,
               cast_scale,
               cast_bias,
               output_mask,
               *x_grad,
               *tmp_scale_grad,
               *tmp_bias_grad);  // output rstd

  // cast back from FLOAT16 to FLOAT32
  if (x.dtype() == phi::DataType::FLOAT16 &&
      scale_grad->dtype() == phi::DataType::FLOAT32) {
    custom_kernel::CastKernel<T, Context>(
        dev_ctx, *tmp_scale_grad, scale_grad->dtype(), scale_grad);
  }
  // same for bias_grad
  if (x.dtype() == phi::DataType::FLOAT16 &&
      bias_grad->dtype() == phi::DataType::FLOAT32) {
    custom_kernel::CastKernel<T, Context>(
        dev_ctx, *tmp_bias_grad, bias_grad->dtype(), bias_grad);
  }

  const_cast<Tensor*>(&mean)->Resize(mean_dims);
  const_cast<Tensor*>(&variance)->Resize(mean_dims);
  const_cast<Tensor*>(scale)->Resize(phi::make_ddim({right}));
  scale_grad->Resize(phi::make_ddim({right}));
  bias_grad->Resize(phi::make_ddim({right}));
}

template <typename T, typename Context>
void LayerNormNPUKernel(const Context& dev_ctx,
                        const phi::DenseTensor& x,
                        const paddle::optional<phi::DenseTensor>& scale_opt,
                        const paddle::optional<phi::DenseTensor>& bias_opt,
                        float epsilon,
                        int begin_norm_axis,
                        phi::DenseTensor* out,
                        phi::DenseTensor* mean,
                        phi::DenseTensor* variance) {
  DO_COMPATIBILITY(
      aclnnLayerNorm,
      (custom_kernel::AclopLayerNormNPUKernel<T, Context>(dev_ctx,
                                                          x,
                                                          scale_opt,
                                                          bias_opt,
                                                          epsilon,
                                                          begin_norm_axis,
                                                          out,
                                                          mean,
                                                          variance)));
  custom_kernel::AclnnLayerNormNPUKernel<T, Context>(dev_ctx,
                                                     x,
                                                     scale_opt,
                                                     bias_opt,
                                                     epsilon,
                                                     begin_norm_axis,
                                                     out,
                                                     mean,
                                                     variance);
}

template <typename T, typename Context>
void LayerNormGradNPUKernel(const Context& dev_ctx,
                            const phi::DenseTensor& x,
                            const paddle::optional<phi::DenseTensor>& scale_opt,
                            const paddle::optional<phi::DenseTensor>& bias,
                            const phi::DenseTensor& mean,
                            const phi::DenseTensor& variance,
                            const phi::DenseTensor& out_grad,
                            float epsilon,
                            int begin_norm_axis,
                            phi::DenseTensor* x_grad,
                            phi::DenseTensor* scale_grad,
                            phi::DenseTensor* bias_grad) {
  DO_COMPATIBILITY(
      aclnnLayerNormBackward,
      (custom_kernel::AclopLayerNormGradNPUKernel<T, Context>(dev_ctx,
                                                              x,
                                                              scale_opt,
                                                              bias,
                                                              mean,
                                                              variance,
                                                              out_grad,
                                                              epsilon,
                                                              begin_norm_axis,
                                                              x_grad,
                                                              scale_grad,
                                                              bias_grad)));
  custom_kernel::AclnnLayerNormGradNPUKernel<T, Context>(dev_ctx,
                                                         x,
                                                         scale_opt,
                                                         bias,
                                                         mean,
                                                         variance,
                                                         out_grad,
                                                         epsilon,
                                                         begin_norm_axis,
                                                         x_grad,
                                                         scale_grad,
                                                         bias_grad);
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(layer_norm,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::LayerNormNPUKernel,
                          float,
                          phi::dtype::bfloat16,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(layer_norm_grad,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::LayerNormGradNPUKernel,
                          float,
                          phi::dtype::bfloat16,
                          phi::dtype::float16) {}
