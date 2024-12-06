/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the Licnse. */

#include "kernels/funcs/elementwise_utils.h"
#include "kernels/funcs/mlu_baseop.h"
#include "kernels/funcs/mlu_funcs.h"

namespace custom_kernel {

template <typename T, typename Context>
void ActivationKernel(const Context& dev_ctx,
                      const phi::DenseTensor& x,
                      float alpha,
                      cnnlActivationMode_t act_mode,
                      phi::DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);

  MLUCnnlActivationDesc act_desc(act_mode, alpha);
  MLUCnnlTensorDesc input_desc(x);
  MLUCnnlTensorDesc output_desc(*out);

  MLUCnnl::Active(dev_ctx,
                  act_desc.get(),
                  input_desc.get(),
                  GetBasePtr(&x),
                  output_desc.get(),
                  GetBasePtr(out));
}

template <typename T, typename Context>
void ActivationGradKernelV1(const Context& dev_ctx,
                            const phi::DenseTensor& x,
                            const phi::DenseTensor& dout,
                            float alpha,
                            cnnlActivationMode_t act_mode,
                            phi::DenseTensor* dx) {
  dev_ctx.template Alloc<T>(dx);

  MLUCnnlTensorDesc x_desc(x);
  MLUCnnlTensorDesc dout_desc(dout);
  MLUCnnlTensorDesc dx_desc(*dx);
  MLUCnnlActivationDesc act_desc(act_mode, alpha);
  MLUCnnl::ActiveGrad(dev_ctx,
                      act_desc.get(),
                      nullptr,
                      nullptr,
                      nullptr,
                      nullptr,
                      dout_desc.get(),
                      GetBasePtr(&dout),
                      x_desc.get(),
                      GetBasePtr(&x),
                      dx_desc.get(),
                      GetBasePtr(dx));
}

template <typename T, typename Context>
void ActivationGradKernelV2(const Context& dev_ctx,
                            const phi::DenseTensor& out,
                            const phi::DenseTensor& dout,
                            cnnlActivationMode_t act_mode,
                            phi::DenseTensor* dx) {
  dev_ctx.template Alloc<T>(dx);

  MLUCnnlTensorDesc out_desc(out);
  MLUCnnlTensorDesc dout_desc(dout);
  MLUCnnlTensorDesc dx_desc(*dx);
  MLUCnnlActivationDesc act_desc(act_mode, 1.0);
  MLUCnnl::ActiveGrad(dev_ctx,
                      act_desc.get(),
                      nullptr,
                      nullptr,
                      out_desc.get(),
                      GetBasePtr(&out),
                      dout_desc.get(),
                      GetBasePtr(&dout),
                      nullptr,
                      nullptr,
                      dx_desc.get(),
                      GetBasePtr(dx));
}

template <typename T, typename Context>
void ActivationGradKernelV3(const Context& dev_ctx,
                            const phi::DenseTensor& out,
                            const phi::DenseTensor& dout,
                            cnnlActivationMode_t act_mode,
                            phi::DenseTensor* dx) {
  dev_ctx.template Alloc<T>(dx);

  MLUCnnlTensorDesc out_desc(out);
  MLUCnnlTensorDesc dout_desc(dout);
  MLUCnnlTensorDesc dx_desc(*dx);
  MLUCnnlActivationDesc act_desc(act_mode, 1.0);
  MLUCnnl::ActiveGrad(dev_ctx,
                      act_desc.get(),
                      nullptr,
                      nullptr,
                      nullptr,
                      nullptr,
                      dout_desc.get(),
                      GetBasePtr(&dout),
                      out_desc.get(),
                      GetBasePtr(&out),
                      dx_desc.get(),
                      GetBasePtr(dx));
}

template <typename T, typename Context>
void ReluKernel(const Context& dev_ctx,
                const phi::DenseTensor& x,
                phi::DenseTensor* out) {
  ActivationKernel<T, Context>(dev_ctx, x, 1.0, CNNL_ACTIVATION_RELU, out);
}

template <typename T, typename Context>
void ReluGradKernel(const Context& dev_ctx,
                    const phi::DenseTensor& out,
                    const phi::DenseTensor& dout,
                    phi::DenseTensor* dx) {
  ActivationGradKernelV3<T, Context>(
      dev_ctx, out, dout, CNNL_ACTIVATION_RELU, dx);
}

template <typename T, typename Context>
void Relu6RawKernel(const Context& dev_ctx,
                    const phi::DenseTensor& x,
                    float threshold,
                    phi::DenseTensor* out) {
  ActivationKernel<T, Context>(dev_ctx, x, 1.0, CNNL_ACTIVATION_RELU6, out);
}

template <typename T, typename Context>
void Relu6Kernel(const Context& dev_ctx,
                 const phi::DenseTensor& x,
                 phi::DenseTensor* out) {
  custom_kernel::Relu6RawKernel<T, Context>(dev_ctx, x, 6.0, out);
}

template <typename T, typename Context>
void Relu6GradKernel(const Context& dev_ctx,
                     const phi::DenseTensor& out,
                     const phi::DenseTensor& dout,
                     phi::DenseTensor* dx) {
  ActivationGradKernelV3<T, Context>(
      dev_ctx, out, dout, CNNL_ACTIVATION_RELU6, dx);
}

template <typename T, typename Context>
void SigmoidKernel(const Context& dev_ctx,
                   const phi::DenseTensor& x,
                   phi::DenseTensor* out) {
  ActivationKernel<T, Context>(dev_ctx, x, 1.0, CNNL_ACTIVATION_SIGMOID, out);
}

template <typename T, typename Context>
void SigmoidGradKernel(const Context& dev_ctx,
                       const phi::DenseTensor& out,
                       const phi::DenseTensor& dout,
                       phi::DenseTensor* dx) {
  ActivationGradKernelV2<T, Context>(
      dev_ctx, out, dout, CNNL_ACTIVATION_SIGMOID, dx);
}

template <typename T, typename Context>
void TanhKernel(const Context& dev_ctx,
                const phi::DenseTensor& x,
                phi::DenseTensor* out) {
  ActivationKernel<T, Context>(dev_ctx, x, 1.0, CNNL_ACTIVATION_TANH, out);
}

template <typename T, typename Context>
void TanhGradKernel(const Context& dev_ctx,
                    const phi::DenseTensor& out,
                    const phi::DenseTensor& out_grad,
                    phi::DenseTensor* x_grad) {
  ActivationGradKernelV2<T, Context>(
      dev_ctx, out, out_grad, CNNL_ACTIVATION_TANH, x_grad);
}

template <typename T, typename Context>
void LeakyReluKernel(const Context& dev_ctx,
                     const phi::DenseTensor& x,
                     float alpha,
                     phi::DenseTensor* out) {
  ActivationKernel<T, Context>(
      dev_ctx, x, alpha, CNNL_ACTIVATION_LEAKYRELU, out);
}

template <typename T, typename Context>
void LeakyReluGradKernel(const Context& dev_ctx,
                         const phi::DenseTensor& x,
                         const phi::DenseTensor& dout,
                         float alpha,
                         phi::DenseTensor* dx) {
  ActivationGradKernelV1<T, Context>(
      dev_ctx, x, dout, alpha, CNNL_ACTIVATION_LEAKYRELU, dx);
}

template <typename T, typename Context>
void GeluKernel(const Context& dev_ctx,
                const phi::DenseTensor& x,
                bool approximate,
                phi::DenseTensor* out) {
  ActivationKernel<T, Context>(dev_ctx, x, 1.0, CNNL_ACTIVATION_GELU, out);
}

template <typename T, typename Context>
void GeluGradKernel(const Context& dev_ctx,
                    const phi::DenseTensor& x,
                    const phi::DenseTensor& out_grad,
                    bool approximate,
                    phi::DenseTensor* x_grad) {
  ActivationGradKernelV1<T, Context>(
      dev_ctx, x, out_grad, 1.0, CNNL_ACTIVATION_GELU, x_grad);
}

template <typename T, typename Context>
void SiluKernel(const Context& dev_ctx,
                const phi::DenseTensor& x,
                phi::DenseTensor* out) {
  ActivationKernel<T, Context>(dev_ctx, x, 1.0, CNNL_ACTIVATION_SILU, out);
}

template <typename T, typename Context>
void SiluGradKernel(const Context& dev_ctx,
                    const phi::DenseTensor& x,
                    const phi::DenseTensor& out,
                    const phi::DenseTensor& dout,
                    phi::DenseTensor* dx) {
  ActivationGradKernelV3<T, Context>(
      dev_ctx, x, dout, CNNL_ACTIVATION_SILU, dx);
}

template <typename T, typename Context>
void SquareKernel(const Context& dev_ctx,
                  const phi::DenseTensor& x,
                  phi::DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);

  MLUCnnlTensorDesc input_desc(x);
  MLUCnnlTensorDesc output_desc(*out);

  MLUCnnl::Square(dev_ctx,
                  input_desc.get(),
                  GetBasePtr(&x),
                  output_desc.get(),
                  GetBasePtr(out));
}

template <typename T, typename Context>
void SquareGradKernel(const Context& dev_ctx,
                      const phi::DenseTensor& x,
                      const phi::DenseTensor& dout,
                      phi::DenseTensor* dx) {
  dev_ctx.template Alloc<T>(dx);
  auto factor_1 = static_cast<float>(1.0);
  auto factor_2 = static_cast<float>(2.0);

  // compute dx = dout * factor_2 * x
  MLUCnnlTensorDesc dout_desc(dout);
  MLUCnnlTensorDesc x_desc(x);
  MLUCnnlTensorDesc dx_desc(*dx);
  MLUCnnlOpTensorDesc mul_op_desc(
      CNNL_OP_TENSOR_MUL, ToCnnlDataType<T>(), CNNL_NOT_PROPAGATE_NAN);
  MLUCnnl::OpTensor(dev_ctx,
                    mul_op_desc.get(),
                    dout_desc.get(),
                    GetBasePtr(&dout),
                    x_desc.get(),
                    GetBasePtr(&x),
                    dx_desc.get(),
                    GetBasePtr(dx),
                    ToCnnlDataType<T>(),
                    /*alpha1*/ factor_1,
                    /*alpha2*/ factor_2,
                    /*beta*/ 0.f);
}

template <typename T, typename Context>
void PowKernel(const Context& dev_ctx,
               const phi::DenseTensor& x,
               const phi::Scalar& factor_scalar,
               phi::DenseTensor* out) {
  auto factor = factor_scalar.to<T>();
  dev_ctx.template Alloc<T>(out);
  phi::DenseTensor factor_tensor;
  factor_tensor.Resize(x.dims());
  dev_ctx.template Alloc<T>(&factor_tensor);
  MLUCnnlTensorDesc factor_desc(factor_tensor);
  MLUCnnlTensorDesc input_desc(x);
  MLUCnnlTensorDesc output_desc(*out);
  cnnlComputationPreference_t prefer = CNNL_COMPUTATION_FAST;
  MLUCnnl::Fill(dev_ctx,
                CNNL_POINTER_MODE_HOST,
                &factor,
                factor_desc.get(),
                GetBasePtr(&factor_tensor));

  MLUCnnl::Pow(dev_ctx,
               prefer,
               input_desc.get(),
               GetBasePtr(&x),
               factor_desc.get(),
               GetBasePtr(&factor_tensor),
               output_desc.get(),
               GetBasePtr(out));
}

// dx = dout * factor * x.pow(factor-1)
template <typename T, typename Context>
void PowGradKernel(const Context& dev_ctx,
                   const phi::DenseTensor& x,
                   const phi::DenseTensor& dout,
                   const phi::Scalar& factor_scalar,
                   phi::DenseTensor* dx) {
  auto factor = factor_scalar.to<T>();

  // Step1: Compute x_pow = x.pow(factor-1)
  phi::DenseTensor x_pow;
  x_pow.Resize(x.dims());
  auto factor_1 = phi::Scalar(factor - static_cast<T>(1.0));
  custom_kernel::PowKernel<T>(dev_ctx, x, factor_1, &x_pow);

  // Step 2: Construct a broadcast factor, which has the same shape with x.
  phi::DenseTensor factor_tensor;
  factor_tensor.Resize(x.dims());
  dev_ctx.template Alloc<T>(&factor_tensor);
  MLUCnnlTensorDesc factor_desc(factor_tensor);
  MLUCnnl::Fill(dev_ctx,
                CNNL_POINTER_MODE_HOST,
                &factor,
                factor_desc.get(),
                GetBasePtr(&factor_tensor));

  // Step 3: Compute x_power_mul_factor = factor * x_pow
  phi::DenseTensor x_power_mul_factor;
  x_power_mul_factor.Resize(x.dims());
  dev_ctx.template Alloc<T>(&x_power_mul_factor);
  MLUOpTensorKernel<T>(dev_ctx,
                       factor_tensor,
                       x_pow,
                       -1,
                       CNNL_OP_TENSOR_MUL,
                       &x_power_mul_factor);

  // Step 4: Compute dx = dout * x_power_mul_factor
  dev_ctx.template Alloc<T>(dx);
  MLUOpTensorKernel<T>(
      dev_ctx, dout, x_power_mul_factor, -1, CNNL_OP_TENSOR_MUL, dx);
}

template <typename T, typename Context>
void AtanKernel(const Context& dev_ctx,
                const phi::DenseTensor& x,
                phi::DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);
  cnnlComputationPreference_t prefer = CNNL_COMPUTATION_FAST;
  MLUCnnlTrigonDesc trigon_desc(CNNL_TRIGON_ATAN, prefer);
  MLUCnnlTensorDesc x_desc(x);
  MLUCnnlTensorDesc out_desc(*out);
  MLUCnnl::TrigonForward(dev_ctx,
                         trigon_desc.get(),
                         x_desc.get(),
                         GetBasePtr(&x),
                         out_desc.get(),
                         GetBasePtr(out));
}

// dx = dout * 1 / (1 + x.pow(2))
template <typename T, typename Context>
void AtanGradKernel(const Context& dev_ctx,
                    const phi::DenseTensor& x,
                    const phi::DenseTensor& dout,
                    phi::DenseTensor* dx) {
  // Step1: Compute x_pow = x.pow(2)
  phi::DenseTensor x_pow;
  x_pow.Resize(x.dims());
  auto factor = phi::Scalar(static_cast<T>(2.0));
  custom_kernel::PowKernel<T>(dev_ctx, x, factor, &x_pow);

  // Step2: x_pow_1 = x_pow + 1
  phi::DenseTensor factor_tensor, x_pow_1;
  factor_tensor.Resize(x.dims());
  x_pow_1.Resize(x.dims());
  dev_ctx.template Alloc<T>(&x_pow_1);
  dev_ctx.template Alloc<T>(&factor_tensor);
  MLUCnnlTensorDesc factor_desc(factor_tensor);
  MLUCnnlTensorDesc x_pow_1_desc(x_pow_1);
  auto one = static_cast<T>(1.0);
  MLUCnnl::Fill(dev_ctx,
                CNNL_POINTER_MODE_HOST,
                &one,
                factor_desc.get(),
                GetBasePtr(&factor_tensor));
  MLUOpTensorKernel<T>(
      dev_ctx, factor_tensor, x_pow, -1, CNNL_OP_TENSOR_ADD, &x_pow_1);

  // Step3: dx = dout / x_pow_1
  dev_ctx.template Alloc<T>(dx);
  MLUBinaryOp<DIV, T>(dev_ctx, dout, x_pow_1, -1, dx);
}

template <typename T, typename Context>
void ReciprocalKernel(const Context& dev_ctx,
                      const phi::DenseTensor& x,
                      phi::DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);
  MLUCnnlTensorDesc x_desc(x);
  MLUCnnlTensorDesc out_desc(*out);
  MLUCnnl::Reciprocal(
      dev_ctx, x_desc.get(), GetBasePtr(&x), out_desc.get(), GetBasePtr(out));
}

template <typename T, typename Context>
void ReciprocalGradKernel(const Context& dev_ctx,
                          const phi::DenseTensor& out,
                          const phi::DenseTensor& dout,
                          phi::DenseTensor* dx) {
  dev_ctx.template Alloc<T>(dx);
  Tensor square_out;
  square_out.Resize(out.dims());
  dev_ctx.template Alloc<T>(&square_out);
  MLUCnnlTensorDesc out_desc(out);
  MLUCnnlTensorDesc dout_desc(dout);
  MLUCnnlTensorDesc dx_desc(*dx);
  MLUCnnlTensorDesc square_out_desc(square_out);
  MLUCnnl::Square(dev_ctx,
                  out_desc.get(),
                  GetBasePtr(&out),
                  square_out_desc.get(),
                  GetBasePtr(&square_out));
  cnnlOpTensorDesc_t op_tensor_op = CNNL_OP_TENSOR_MUL;
  cnnlDataType_t op_tensor_comp_type = CNNL_DTYPE_FLOAT;
  cnnlNanPropagation_t op_tensor_nan_opt = CNNL_NOT_PROPAGATE_NAN;
  MLUCnnlOpTensorDesc op_tensor_desc(
      op_tensor_op, op_tensor_comp_type, op_tensor_nan_opt);
  float alpha1_float = -1;
  float alpha2_float = 1;
  float beta_float = 0;
  MLUCnnl::OpTensor(dev_ctx,
                    op_tensor_desc.get(),
                    dout_desc.get(),
                    GetBasePtr(&dout),
                    square_out_desc.get(),
                    GetBasePtr(&square_out),
                    dx_desc.get(),
                    GetBasePtr(dx),
                    op_tensor_comp_type,
                    alpha1_float,
                    alpha2_float,
                    beta_float);
}

template <typename T, typename Context>
void SqrtKernel(const Context& dev_ctx,
                const phi::DenseTensor& x,
                phi::DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);

  MLUCnnlTensorDesc input_desc(x);
  MLUCnnlTensorDesc output_desc(*out);

  cnnlComputationPreference_t prefer = CNNL_COMPUTATION_HIGH_PRECISION;
  MLUCnnl::Sqrt(dev_ctx,
                prefer,
                input_desc.get(),
                GetBasePtr(&x),
                output_desc.get(),
                GetBasePtr(out));
}

template <typename T, typename Context>
void SqrtGradKernel(const Context& dev_ctx,
                    const phi::DenseTensor& out,
                    const phi::DenseTensor& dout,
                    phi::DenseTensor* dx) {
  dev_ctx.template Alloc<T>(dx);

  MLUCnnlTensorDesc data_desc(out);
  MLUCnnl::SqrtGrad(dev_ctx,
                    data_desc.get(),
                    GetBasePtr(&out),
                    GetBasePtr(&dout),
                    GetBasePtr(dx));
}

template <typename T, typename Context>
void RsqrtKernel(const Context& dev_ctx,
                 const phi::DenseTensor& x,
                 phi::DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);

  MLUCnnlTensorDesc input_desc(x);
  MLUCnnlTensorDesc output_desc(*out);

  cnnlComputationPreference_t prefer = CNNL_COMPUTATION_HIGH_PRECISION;
  MLUCnnl::Rsqrt(dev_ctx,
                 prefer,
                 input_desc.get(),
                 GetBasePtr(&x),
                 output_desc.get(),
                 GetBasePtr(out));
}

template <typename T, typename Context>
void RsqrtGradKernel(const Context& dev_ctx,
                     const phi::DenseTensor& out,
                     const phi::DenseTensor& dout,
                     phi::DenseTensor* dx) {
  dev_ctx.template Alloc<T>(dx);

  MLUCnnlTensorDesc data_desc(out);
  MLUCnnl::RsqrtGrad(dev_ctx,
                     data_desc.get(),
                     GetBasePtr(&out),
                     GetBasePtr(&dout),
                     GetBasePtr(dx));
}

template <typename T, typename Context>
void CosKernel(const Context& dev_ctx,
               const phi::DenseTensor& x,
               phi::DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);

  MLUCnnlTensorDesc input_desc(x);
  MLUCnnlTensorDesc output_desc(*out);

  cnnlComputationPreference_t prefer = CNNL_COMPUTATION_HIGH_PRECISION;
  MLUCnnl::Cos(dev_ctx,
               prefer,
               input_desc.get(),
               GetBasePtr(&x),
               output_desc.get(),
               GetBasePtr(out));
}

template <typename T, typename Context>
void CosGradKernel(const Context& dev_ctx,
                   const phi::DenseTensor& x,
                   const phi::DenseTensor& dout,
                   phi::DenseTensor* dx) {
  dev_ctx.template Alloc<T>(dx);

  phi::DenseTensor sin_out;
  sin_out.Resize(x.dims());
  dev_ctx.template Alloc<T>(&sin_out);

  MLUCnnlTensorDesc input_desc(x);
  MLUCnnlTensorDesc sin_out_desc(sin_out);
  MLUCnnlTensorDesc dout_desc(dout);
  MLUCnnlTensorDesc dx_desc(*dx);

  cnnlComputationPreference_t prefer = CNNL_COMPUTATION_HIGH_PRECISION;
  MLUCnnl::Sin(dev_ctx,
               prefer,
               input_desc.get(),
               GetBasePtr(&x),
               sin_out_desc.get(),
               GetBasePtr(&sin_out));

  MLUCnnlOpTensorDesc mul_op_desc(
      CNNL_OP_TENSOR_MUL, ToCnnlDataType<T>(), CNNL_NOT_PROPAGATE_NAN);

  MLUCnnl::OpTensor(dev_ctx,
                    mul_op_desc.get(),
                    sin_out_desc.get(),
                    GetBasePtr(&sin_out),
                    dout_desc.get(),
                    GetBasePtr(&dout),
                    dx_desc.get(),
                    GetBasePtr(dx),
                    ToCnnlDataType<T>(),
                    -1 /*alpha -1,for -1*sin(x) */);
}

// CNNL_LOG_E = 0,
// CNNL_LOG_2 = 1,
// CNNL_LOG_10 = 2,
template <typename T, typename Context>
void LogMLUKernel(const Context& dev_ctx,
                  const phi::DenseTensor& x,
                  cnnlLogBase_t log_base,
                  phi::DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);

  MLUCnnlTensorDesc input_desc(x);
  MLUCnnlTensorDesc output_desc(*out);
  cnnlComputationPreference_t prefer = CNNL_COMPUTATION_HIGH_PRECISION;

  MLUCnnl::Log(dev_ctx,
               prefer,
               log_base,
               input_desc.get(),
               GetBasePtr(&x),
               output_desc.get(),
               GetBasePtr(out));
}

template <typename T, typename Context>
void LogKernel(const Context& dev_ctx,
               const phi::DenseTensor& x,
               phi::DenseTensor* out) {
  LogMLUKernel<T, Context>(dev_ctx, x, CNNL_LOG_E, out);
}

template <typename T, typename Context>
void Log2Kernel(const Context& dev_ctx,
                const phi::DenseTensor& x,
                phi::DenseTensor* out) {
  LogMLUKernel<T, Context>(dev_ctx, x, CNNL_LOG_2, out);
}

template <typename T, typename Context>
void Log10Kernel(const Context& dev_ctx,
                 const phi::DenseTensor& x,
                 phi::DenseTensor* out) {
  LogMLUKernel<T, Context>(dev_ctx, x, CNNL_LOG_10, out);
}

template <typename T, typename Context>
void ExpKernel(const Context& dev_ctx,
               const phi::DenseTensor& x,
               phi::DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);

  MLUCnnlTensorDesc input_desc(x);
  MLUCnnlTensorDesc output_desc(*out);
  cnnlComputationPreference_t prefer = CNNL_COMPUTATION_HIGH_PRECISION;

  MLUCnnl::Exp(dev_ctx,
               prefer,
               input_desc.get(),
               GetBasePtr(&x),
               output_desc.get(),
               GetBasePtr(out));
}

template <typename T, typename Context>
void ExpGradKernel(const Context& dev_ctx,
                   const phi::DenseTensor& out,
                   const phi::DenseTensor& dout,
                   phi::DenseTensor* dx) {
  dev_ctx.template Alloc<T>(dx);

  MLUCnnlTensorDesc dout_desc(dout);
  MLUCnnlTensorDesc out_desc(out);
  MLUCnnlTensorDesc dx_desc(*dx);

  MLUCnnlOpTensorDesc op_tensor_desc(
      CNNL_OP_TENSOR_MUL, ToCnnlDataType<T>(), CNNL_NOT_PROPAGATE_NAN);

  MLUCnnl::OpTensor(dev_ctx,
                    op_tensor_desc.get(),
                    dout_desc.get(),
                    GetBasePtr(&dout),
                    out_desc.get(),
                    GetBasePtr(&out),
                    dx_desc.get(),
                    GetBasePtr(dx),
                    ToCnnlDataType<T>());
}

template <typename T, typename Context>
void SinKernel(const Context& dev_ctx,
               const phi::DenseTensor& x,
               phi::DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);
  MLUCnnlTensorDesc input_desc(x);
  MLUCnnlTensorDesc output_desc(*out);
  cnnlComputationPreference_t prefer = CNNL_COMPUTATION_HIGH_PRECISION;
  MLUCnnl::Sin(dev_ctx,
               prefer,
               input_desc.get(),
               GetBasePtr(&x),
               output_desc.get(),
               GetBasePtr(out));
}

template <typename T, typename Context>
void SinGradKernel(const Context& dev_ctx,
                   const phi::DenseTensor& x,
                   const phi::DenseTensor& dout,
                   phi::DenseTensor* dx) {
  dev_ctx.template Alloc<T>(dx);
  phi::DenseTensor cos_out;
  phi::DenseTensorMeta meta = {x.dtype(), x.dims()};
  cos_out.set_meta(meta);
  dev_ctx.template Alloc<T>(&cos_out);

  MLUCnnlTensorDesc x_desc(x);
  MLUCnnlTensorDesc cos_out_desc(cos_out);
  MLUCnnlTensorDesc dout_desc(dout);
  MLUCnnlTensorDesc dx_desc(*dx);

  cnnlComputationPreference_t prefer = CNNL_COMPUTATION_HIGH_PRECISION;
  MLUCnnl::Cos(dev_ctx,
               prefer,
               x_desc.get(),
               GetBasePtr(&x),
               cos_out_desc.get(),
               GetBasePtr(&cos_out));

  MLUCnnlOpTensorDesc mul_op_desc(
      CNNL_OP_TENSOR_MUL, ToCnnlDataType<T>(), CNNL_NOT_PROPAGATE_NAN);
  MLUCnnl::OpTensor(dev_ctx,
                    mul_op_desc.get(),
                    cos_out_desc.get(),
                    GetBasePtr(&cos_out),
                    dout_desc.get(),
                    GetBasePtr(&dout),
                    dx_desc.get(),
                    GetBasePtr(dx),
                    ToCnnlDataType<T>());
}

template <typename T, typename Context>
void HardSwishKernel(const Context& dev_ctx,
                     const phi::DenseTensor& x,
                     phi::DenseTensor* out) {
  float threshold = 6;
  float scale = 6;
  float offset = 3;
  PADDLE_ENFORCE_EQ(
      threshold,
      6.0f,
      phi::errors::External("Not support threshold [%f] in MLU", threshold));
  PADDLE_ENFORCE_EQ(
      scale,
      6.0f,
      phi::errors::External("Not support scale [%f] in MLU", scale));
  PADDLE_ENFORCE_EQ(
      offset,
      3.0f,
      phi::errors::External("Not support offset [%f] in MLU", offset));

  dev_ctx.template Alloc<T>(out);
  MLUCnnlActivationDesc act_desc(CNNL_ACTIVATION_HARDSWISH,
                                 1.0f /*ceof useless*/);
  MLUCnnlTensorDesc input_desc(x);
  MLUCnnlTensorDesc output_desc(*out);

  MLUCnnl::Active(dev_ctx,
                  act_desc.get(),
                  input_desc.get(),
                  GetBasePtr(&x),
                  output_desc.get(),
                  GetBasePtr(out));
}

template <typename T, typename Context>
void SwishKernel(const Context& dev_ctx,
                 const phi::DenseTensor& x,
                 phi::DenseTensor* out) {
  ActivationKernel<T, Context>(
      dev_ctx, x, 1.0 /* ceof */, CNNL_ACTIVATION_SILU, out);
}

template <typename T, typename Context>
void SwishGradKernel(const Context& dev_ctx,
                     const phi::DenseTensor& x,
                     const phi::DenseTensor& dout,
                     phi::DenseTensor* dx) {
  ActivationGradKernelV3<T, Context>(
      dev_ctx, x, dout, CNNL_ACTIVATION_SILU, dx);
}

template <typename T, typename Context>
void HardSwishGradKernel(const Context& dev_ctx,
                         const phi::DenseTensor& x,
                         const phi::DenseTensor& dout,
                         phi::DenseTensor* dx) {
  float threshold = 6;
  float scale = 6;
  float offset = 3;
  PADDLE_ENFORCE_EQ(
      threshold,
      6.0f,
      phi::errors::External("Not support threshold [%f] in MLU", threshold));
  PADDLE_ENFORCE_EQ(
      scale,
      6.0f,
      phi::errors::External("Not support scale [%f] in MLU", scale));
  PADDLE_ENFORCE_EQ(
      offset,
      3.0f,
      phi::errors::External("Not support offset [%f] in MLU", offset));

  dev_ctx.template Alloc<T>(dx);

  MLUCnnlTensorDesc x_desc(x);
  MLUCnnlTensorDesc dout_desc(dout);
  MLUCnnlTensorDesc dx_desc(*dx);
  MLUCnnlActivationDesc act_desc(CNNL_ACTIVATION_HARDSWISH,
                                 1.0f /*ceof useless*/);
  MLUCnnl::ActiveGrad(dev_ctx,
                      act_desc.get(),
                      nullptr,
                      nullptr,
                      nullptr,
                      nullptr,
                      dout_desc.get(),
                      GetBasePtr(&dout),
                      x_desc.get(),
                      GetBasePtr(&x),
                      dx_desc.get(),
                      GetBasePtr(dx));
}

template <typename T, typename Context>
void HardSigmoidKernel(const Context& dev_ctx,
                       const phi::DenseTensor& x,
                       float slope,
                       float offset,
                       phi::DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);

  MLUCnnlActivationDesc act_desc(CNNL_ACTIVATION_HARDSIGMOID,
                                 1.0f /*ceof useless*/,
                                 1.0f /*sliced_dim useless*/,
                                 slope,
                                 offset);
  MLUCnnlTensorDesc input_desc(x);
  MLUCnnlTensorDesc output_desc(*out);

  MLUCnnl::Active(dev_ctx,
                  act_desc.get(),
                  input_desc.get(),
                  GetBasePtr(&x),
                  output_desc.get(),
                  GetBasePtr(out));
}

template <typename T, typename Context>
void HardSigmoidGradKernel(const Context& dev_ctx,
                           const phi::DenseTensor& out,
                           const phi::DenseTensor& dout,
                           float slope,
                           float offset,
                           phi::DenseTensor* dx) {
  dev_ctx.template Alloc<T>(dx);
  MLUCnnlActivationDesc act_desc(CNNL_ACTIVATION_HARDSIGMOID,
                                 1.0f /*ceof useless*/,
                                 1.0f /*sliced_dim useless*/,
                                 slope,
                                 offset);
  MLUCnnlTensorDesc out_desc(out);
  MLUCnnlTensorDesc dout_desc(dout);
  MLUCnnlTensorDesc dx_desc(*dx);
  MLUCnnl::ActiveGrad(dev_ctx,
                      act_desc.get(),
                      nullptr,
                      nullptr,
                      out_desc.get(),
                      GetBasePtr(&out),
                      dout_desc.get(),
                      GetBasePtr(&dout),
                      nullptr,
                      nullptr,
                      dx_desc.get(),
                      GetBasePtr(dx));
}

template <typename T, typename Context>
void FloorKernel(const Context& dev_ctx,
                 const phi::DenseTensor& x,
                 phi::DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);

  MLUCnnlTensorDesc input_desc(x);
  MLUCnnlTensorDesc output_desc(*out);
  MLUCnnl::Floor(dev_ctx,
                 input_desc.get(),
                 GetBasePtr(&x),
                 output_desc.get(),
                 GetBasePtr(out));
}

template <typename T, typename Context>
void RoundKernel(const Context& dev_ctx,
                 const phi::DenseTensor& x,
                 const int decimals
                     UNUSED,  // add decimals, Now it's just for CI
                 phi::DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);

  MLUCnnlTensorDesc input_desc(x);
  MLUCnnlTensorDesc output_desc(*out);
  MLUCnnl::Round(dev_ctx,
                 input_desc.get(),
                 GetBasePtr(&x),
                 output_desc.get(),
                 GetBasePtr(out));
}

template <typename T, typename Context>
void RoundGradKernel(const Context& dev_ctx,
                     const phi::DenseTensor& dout,
                     phi::DenseTensor* dx) {
  dev_ctx.template Alloc<T>(dx);
  FillMLUTensorWithHostValue<T>(dev_ctx, static_cast<T>(0), dx);
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(relu,
                          mlu,
                          ALL_LAYOUT,
                          custom_kernel::ReluKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(relu_grad,
                          mlu,
                          ALL_LAYOUT,
                          custom_kernel::ReluGradKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(relu6,
                          mlu,
                          ALL_LAYOUT,
                          custom_kernel::Relu6Kernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(relu6_grad,
                          mlu,
                          ALL_LAYOUT,
                          custom_kernel::Relu6GradKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(leaky_relu,
                          mlu,
                          ALL_LAYOUT,
                          custom_kernel::LeakyReluKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(leaky_relu_grad,
                          mlu,
                          ALL_LAYOUT,
                          custom_kernel::LeakyReluGradKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(reciprocal,
                          mlu,
                          ALL_LAYOUT,
                          custom_kernel::ReciprocalKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(reciprocal_grad,
                          mlu,
                          ALL_LAYOUT,
                          custom_kernel::ReciprocalGradKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(sqrt,
                          mlu,
                          ALL_LAYOUT,
                          custom_kernel::SqrtKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(sqrt_grad,
                          mlu,
                          ALL_LAYOUT,
                          custom_kernel::SqrtGradKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(rsqrt,
                          mlu,
                          ALL_LAYOUT,
                          custom_kernel::RsqrtKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(rsqrt_grad,
                          mlu,
                          ALL_LAYOUT,
                          custom_kernel::RsqrtGradKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(cos,
                          mlu,
                          ALL_LAYOUT,
                          custom_kernel::CosKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(cos_grad,
                          mlu,
                          ALL_LAYOUT,
                          custom_kernel::CosGradKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(log,
                          mlu,
                          ALL_LAYOUT,
                          custom_kernel::LogKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(log2,
                          mlu,
                          ALL_LAYOUT,
                          custom_kernel::Log2Kernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(log10,
                          mlu,
                          ALL_LAYOUT,
                          custom_kernel::Log10Kernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(gelu,
                          mlu,
                          ALL_LAYOUT,
                          custom_kernel::GeluKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(gelu_grad,
                          mlu,
                          ALL_LAYOUT,
                          custom_kernel::GeluGradKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(tanh,
                          mlu,
                          ALL_LAYOUT,
                          custom_kernel::TanhKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(tanh_grad,
                          mlu,
                          ALL_LAYOUT,
                          custom_kernel::TanhGradKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(exp,
                          mlu,
                          ALL_LAYOUT,
                          custom_kernel::ExpKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(exp_grad,
                          mlu,
                          ALL_LAYOUT,
                          custom_kernel::ExpGradKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(sin,
                          mlu,
                          ALL_LAYOUT,
                          custom_kernel::SinKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(sin_grad,
                          mlu,
                          ALL_LAYOUT,
                          custom_kernel::SinGradKernel,
                          float,
                          phi::dtype::float16) {}
PD_REGISTER_PLUGIN_KERNEL(swish,
                          mlu,
                          ALL_LAYOUT,
                          custom_kernel::SwishKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(swish_grad,
                          mlu,
                          ALL_LAYOUT,
                          custom_kernel::SwishGradKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(hardswish,
                          mlu,
                          ALL_LAYOUT,
                          custom_kernel::HardSwishKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(hardswish_grad,
                          mlu,
                          ALL_LAYOUT,
                          custom_kernel::HardSwishGradKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(hardsigmoid,
                          mlu,
                          ALL_LAYOUT,
                          custom_kernel::HardSigmoidKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(hardsigmoid_grad,
                          mlu,
                          ALL_LAYOUT,
                          custom_kernel::HardSigmoidGradKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(floor,
                          mlu,
                          ALL_LAYOUT,
                          custom_kernel::FloorKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(sigmoid,
                          mlu,
                          ALL_LAYOUT,
                          custom_kernel::SigmoidKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(sigmoid_grad,
                          mlu,
                          ALL_LAYOUT,
                          custom_kernel::SigmoidGradKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(silu,
                          mlu,
                          ALL_LAYOUT,
                          custom_kernel::SiluKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(silu_grad,
                          mlu,
                          ALL_LAYOUT,
                          custom_kernel::SiluGradKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(square,
                          mlu,
                          ALL_LAYOUT,
                          custom_kernel::SquareKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(square_grad,
                          mlu,
                          ALL_LAYOUT,
                          custom_kernel::SquareGradKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(pow,
                          mlu,
                          ALL_LAYOUT,
                          custom_kernel::PowKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(pow_grad,
                          mlu,
                          ALL_LAYOUT,
                          custom_kernel::PowGradKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(atan,
                          mlu,
                          ALL_LAYOUT,
                          custom_kernel::AtanKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(atan_grad,
                          mlu,
                          ALL_LAYOUT,
                          custom_kernel::AtanGradKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(round,
                          mlu,
                          ALL_LAYOUT,
                          custom_kernel::RoundKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(round_grad,
                          mlu,
                          ALL_LAYOUT,
                          custom_kernel::RoundGradKernel,
                          float,
                          phi::dtype::float16) {}
