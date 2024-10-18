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
void ScaleKernel(const Context& dev_ctx,
                 const phi::DenseTensor& x,
                 const phi::Scalar& in_scale,
                 const phi::Scalar& in_bias,
                 bool bias_after_scale,
                 phi::DenseTensor* out);

template <typename T, typename Context>
void ClipKernel(const Context& dev_ctx,
                const phi::DenseTensor& x,
                const phi::Scalar& min,
                const phi::Scalar& max,
                phi::DenseTensor* out);

template <typename T, typename Context>
void GreaterThanKernel(const Context& dev_ctx,
                       const phi::DenseTensor& x,
                       const phi::DenseTensor& y,
                       phi::DenseTensor* out);

template <typename T, typename Context>
void LessThanKernel(const Context& dev_ctx,
                    const phi::DenseTensor& x,
                    const phi::DenseTensor& y,
                    phi::DenseTensor* out);

template <typename T, typename Context>
void LogicalAndNPUKernel(const Context& dev_ctx,
                         const phi::DenseTensor& x,
                         const phi::DenseTensor& y,
                         phi::DenseTensor* out);

template <typename T, typename Context>
void WhereKernel(const Context& dev_ctx,
                 const phi::DenseTensor& condition,
                 const phi::DenseTensor& x,
                 const phi::DenseTensor& y,
                 phi::DenseTensor* out);

template <typename T, typename Context>
void AclopCosKernel(const Context& dev_ctx,
                    const phi::DenseTensor& x,
                    phi::DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);
  auto stream = dev_ctx.stream();
  const auto& runner = NpuOpRunner("Cos", {x}, {*out}, {});
  runner.Run(stream);
}

template <typename T, typename Context>
void CosKernel(const Context& dev_ctx,
               const phi::DenseTensor& x,
               phi::DenseTensor* out) {
  DO_COMPATIBILITY(
      aclnnCos, (custom_kernel::AclopCosKernel<T, Context>(dev_ctx, x, out)));
  dev_ctx.template Alloc<T>(out);
  EXEC_NPU_CMD(aclnnCos, dev_ctx, x, *out);
}

template <typename T, typename Context>
void AclopAtanKernel(const Context& dev_ctx,
                     const phi::DenseTensor& x,
                     phi::DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);
  auto stream = dev_ctx.stream();
  const auto& runner = NpuOpRunner("Atan", {x}, {*out}, {});
  runner.Run(stream);
}

template <typename T, typename Context>
void AtanKernel(const Context& dev_ctx,
                const phi::DenseTensor& x,
                phi::DenseTensor* out) {
  DO_COMPATIBILITY(
      aclnnAtan, (custom_kernel::AclopAtanKernel<T, Context>(dev_ctx, x, out)));
  dev_ctx.template Alloc<T>(out);
  EXEC_NPU_CMD(aclnnAtan, dev_ctx, x, *out);
}

template <typename T, typename Context>
void AtanGradKernel(const Context& dev_ctx,
                    const phi::DenseTensor& x,
                    const phi::DenseTensor& dout,
                    phi::DenseTensor* dx) {
  dev_ctx.template Alloc<T>(dx);
  auto stream = dev_ctx.stream();

  const auto& runner = NpuOpRunner("AtanGrad", {x, dout}, {*dx}, {});
  runner.Run(stream);
}

template <typename T, typename Context>
void AclopExpKernel(const Context& dev_ctx,
                    const phi::DenseTensor& x,
                    phi::DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);
  auto stream = dev_ctx.stream();
  const auto& runner = NpuOpRunner("Exp", {x}, {*out}, {});
  runner.Run(stream);
}

template <typename T, typename Context>
void ExpKernel(const Context& dev_ctx,
               const phi::DenseTensor& x,
               phi::DenseTensor* out) {
  DO_COMPATIBILITY(
      aclnnExp, (custom_kernel::AclopExpKernel<T, Context>(dev_ctx, x, out)));
  dev_ctx.template Alloc<T>(out);
  EXEC_NPU_CMD(aclnnExp, dev_ctx, x, *out);
}

template <typename T, typename Context>
void AclopExpGradKernel(const Context& dev_ctx,
                        const phi::DenseTensor& out,
                        const phi::DenseTensor& dout,
                        phi::DenseTensor* dx) {
  dev_ctx.template Alloc<T>(dx);
  auto stream = dev_ctx.stream();
  const auto& runner = NpuOpRunner("Mul", {dout, out}, {*dx}, {});
  runner.Run(stream);
}

template <typename T, typename Context>
void ExpGradKernel(const Context& dev_ctx,
                   const phi::DenseTensor& out,
                   const phi::DenseTensor& dout,
                   phi::DenseTensor* dx) {
  DO_COMPATIBILITY(
      aclnnMul,
      (custom_kernel::AclopExpGradKernel<T, Context>(dev_ctx, dout, out, dx)));
  dev_ctx.template Alloc<T>(dx);
  EXEC_NPU_CMD(aclnnMul, dev_ctx, dout, out, *dx);
}

template <typename T, typename Context>
void AclopFloorKernel(const Context& dev_ctx,
                      const phi::DenseTensor& x,
                      phi::DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);
  auto stream = dev_ctx.stream();
  const auto& runner = NpuOpRunner("Floor", {x}, {*out}, {});
  runner.Run(stream);
}

template <typename T, typename Context>
void FloorKernel(const Context& dev_ctx,
                 const phi::DenseTensor& x,
                 phi::DenseTensor* out) {
  DO_COMPATIBILITY(
      aclnnFloor,
      (custom_kernel::AclopFloorKernel<T, Context>(dev_ctx, x, out)));
  dev_ctx.template Alloc<T>(out);
  EXEC_NPU_CMD(aclnnFloor, dev_ctx, x, *out);
}

template <typename T, typename Context>
void AclopFloorGradKernel(const Context& dev_ctx,
                          const phi::DenseTensor& dout,
                          phi::DenseTensor* dx) {
  dev_ctx.template Alloc<T>(dx);
  auto stream = dev_ctx.stream();
  const auto& runner =
      NpuOpRunner("Fills", {*dx}, {*dx}, {{"value", static_cast<float>(0)}});
  runner.Run(stream);
}

template <typename T, typename Context>
void FloorGradKernel(const Context& dev_ctx,
                     const phi::DenseTensor& dout,
                     phi::DenseTensor* dx) {
  DO_COMPATIBILITY(
      aclnnInplaceZero,
      (custom_kernel::AclopFloorGradKernel<T, Context>(dev_ctx, dout, dx)));
  dev_ctx.template Alloc<T>(dx);
  EXEC_NPU_CMD(aclnnInplaceZero, dev_ctx, *dx);
}

template <typename T, typename Context>
void AclopRsqrtKernel(const Context& dev_ctx,
                      const phi::DenseTensor& x,
                      phi::DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);
  auto stream = dev_ctx.stream();

  const auto& runner = NpuOpRunner("Rsqrt", {x}, {*out}, {});
  runner.Run(stream);
}

template <typename T, typename Context>
void RsqrtKernel(const Context& dev_ctx,
                 const phi::DenseTensor& x,
                 phi::DenseTensor* out) {
  DO_COMPATIBILITY(
      aclnnRsqrt,
      (custom_kernel::AclopRsqrtKernel<T, Context>(dev_ctx, x, out)));
  dev_ctx.template Alloc<T>(out);
  EXEC_NPU_CMD(aclnnRsqrt, dev_ctx, x, *out);
}

template <typename T, typename Context>
void RsqrtGradAclop(const Context& dev_ctx,
                    const phi::DenseTensor& out,
                    const phi::DenseTensor& dout,
                    phi::DenseTensor* dx) {
  dev_ctx.template Alloc<T>(dx);
  auto stream = dev_ctx.stream();

  const auto& runner = NpuOpRunner("RsqrtGrad", {out, dout}, {*dx}, {});
  runner.Run(stream);
}

template <typename T, typename Context>
void RsqrtGradKernel(const Context& dev_ctx,
                     const phi::DenseTensor& out,
                     const phi::DenseTensor& dout,
                     phi::DenseTensor* dx) {
  DO_COMPATIBILITY(
      aclnnInplaceZero,
      (custom_kernel::RsqrtGradAclop<T, Context>(dev_ctx, out, dout, dx)));
  dev_ctx.template Alloc<T>(dx);
  auto stream = dev_ctx.stream();
  phi::DenseTensor x_tmp;
  x_tmp.Resize(out.dims());
  dev_ctx.template Alloc<T>(&x_tmp);
  auto neg_two_value = static_cast<T>(-0.5);
  auto three_value = static_cast<T>(3.0);
  aclDataType acl_data_type = ConvertToNpuDtype(out.dtype());
  static const auto aclCreateScalar = GET_OP_API_FUNC(aclCreateScalar);
  aclScalar* acl_scalar_two = aclCreateScalar(&neg_two_value, acl_data_type);
  aclScalar* acl_scalar_three = aclCreateScalar(&three_value, acl_data_type);
  EXEC_NPU_CMD(aclnnPowTensorScalar, dev_ctx, out, acl_scalar_three, x_tmp);
  EXEC_NPU_CMD(aclnnMuls, dev_ctx, x_tmp, acl_scalar_two, x_tmp);
  EXEC_NPU_CMD(aclnnMul, dev_ctx, dout, x_tmp, *dx);
}

template <typename T, typename Context>
void AclopSinKernel(const Context& dev_ctx,
                    const phi::DenseTensor& x,
                    phi::DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);
  auto stream = dev_ctx.stream();
  const auto& runner = NpuOpRunner("Sin", {x}, {*out}, {});
  runner.Run(stream);
}

template <typename T, typename Context>
void SinKernel(const Context& dev_ctx,
               const phi::DenseTensor& x,
               phi::DenseTensor* out) {
  DO_COMPATIBILITY(
      aclnnSin, (custom_kernel::AclopSinKernel<T, Context>(dev_ctx, x, out)));
  dev_ctx.template Alloc<T>(out);
  EXEC_NPU_CMD(aclnnSin, dev_ctx, x, *out);
}

// Swish = x * sigmoid(beta * x)
template <typename T, typename Context>
void AclopSwishKernel(const Context& dev_ctx,
                      const phi::DenseTensor& x,
                      phi::DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);
  auto stream = dev_ctx.stream();
  const auto& runner =
      NpuOpRunner("Swish", {x}, {*out}, {{"scale", static_cast<float>(1.0)}});
  runner.Run(stream);
}

template <typename T, typename Context>
void SwishKernel(const Context& dev_ctx,
                 const phi::DenseTensor& x,
                 phi::DenseTensor* out) {
  DO_COMPATIBILITY(
      aclnnSilu,
      (custom_kernel::AclopSwishKernel<T, Context>(dev_ctx, x, out)));
  dev_ctx.template Alloc<T>(out);
  EXEC_NPU_CMD(aclnnSilu, dev_ctx, x, *out);
}

template <typename T, typename Context>
void AclopSwishGradKernel(const Context& dev_ctx,
                          const phi::DenseTensor& x,
                          const phi::DenseTensor& dout,
                          phi::DenseTensor* dx) {
  dev_ctx.template Alloc<T>(dx);
  auto stream = dev_ctx.stream();

  phi::DenseTensor beta_x, sigmoid_out, swish_out;
  beta_x.Resize(x.dims());
  sigmoid_out.Resize(x.dims());
  swish_out.Resize(x.dims());
  dev_ctx.template Alloc<T>(&beta_x);
  dev_ctx.template Alloc<T>(&sigmoid_out);
  dev_ctx.template Alloc<T>(&swish_out);

  const auto& muls_runner =
      NpuOpRunner("Muls", {x}, {beta_x}, {{"value", static_cast<float>(1.0)}});
  muls_runner.Run(stream);

  const auto& sigmoid_runner =
      NpuOpRunner("Sigmoid", {beta_x}, {sigmoid_out}, {});
  sigmoid_runner.Run(stream);

  const auto& mul_runner =
      NpuOpRunner("Mul", {sigmoid_out, x}, {swish_out}, {});
  mul_runner.Run(stream);

  const auto& muls_runner2 = NpuOpRunner(
      "Muls", {swish_out}, {swish_out}, {{"value", static_cast<float>(1.0)}});
  muls_runner2.Run(stream);

  const auto& mul_runner1 =
      NpuOpRunner("Mul", {sigmoid_out, swish_out}, {*dx}, {});
  mul_runner1.Run(stream);

  const auto& sub_runner = NpuOpRunner("Sub", {swish_out, *dx}, {*dx}, {});
  sub_runner.Run(stream);

  const auto& add_runner = NpuOpRunner("Add", {sigmoid_out, *dx}, {*dx}, {});
  add_runner.Run(stream);

  const auto& mul_runner2 = NpuOpRunner("Mul", {dout, *dx}, {*dx}, {});
  mul_runner2.Run(stream);
}

template <typename T, typename Context>
void SwishGradKernel(const Context& dev_ctx,
                     const phi::DenseTensor& x,
                     const phi::DenseTensor& dout,
                     phi::DenseTensor* dx) {
  DO_COMPATIBILITY(
      aclnnSiluBackward,
      (custom_kernel::AclopSwishGradKernel<T, Context>(dev_ctx, x, dout, dx)));

  dev_ctx.template Alloc<T>(dx);
  EXEC_NPU_CMD(aclnnSiluBackward, dev_ctx, dout, x, *dx);
}

// Silu = x * sigmoid(x)
template <typename T, typename Context>
void AclopSiluKernel(const Context& dev_ctx,
                     const phi::DenseTensor& x,
                     phi::DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);
  auto stream = dev_ctx.stream();
  const auto& runner =
      NpuOpRunner("Swish", {x}, {*out}, {{"scale", static_cast<float>(1.0)}});
  runner.Run(stream);
}

template <typename T, typename Context>
void SiluKernel(const Context& dev_ctx,
                const phi::DenseTensor& x,
                phi::DenseTensor* out) {
  DO_COMPATIBILITY(
      aclnnSilu, (custom_kernel::AclopSiluKernel<T, Context>(dev_ctx, x, out)));
  dev_ctx.template Alloc<T>(out);
  EXEC_NPU_CMD(aclnnSilu, dev_ctx, x, *out);
}

template <typename T, typename Context>
void AclopSiluGradKernel(const Context& dev_ctx,
                         const phi::DenseTensor& x,
                         const phi::DenseTensor& out,
                         const phi::DenseTensor& dout,
                         phi::DenseTensor* dx) {
  AclopSwishGradKernel<T, Context>(dev_ctx, x, dout, dx);
}

template <typename T, typename Context>
void SiluGradKernel(const Context& dev_ctx,
                    const phi::DenseTensor& x,
                    const phi::DenseTensor& out,
                    const phi::DenseTensor& dout,
                    phi::DenseTensor* dx) {
  DO_COMPATIBILITY(aclnnSiluBackward,
                   (custom_kernel::AclopSiluGradKernel<T, Context>(
                       dev_ctx, x, out, dout, dx)));

  dev_ctx.template Alloc<T>(dx);
  EXEC_NPU_CMD(aclnnSiluBackward, dev_ctx, dout, x, *dx);
}

template <typename T, typename Context>
void AclopReluKernel(const Context& dev_ctx,
                     const phi::DenseTensor& x,
                     phi::DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);
  const auto& runner = NpuOpRunner("Relu", {x}, {*out}, {});

  auto stream = dev_ctx.stream();
  runner.Run(stream);
}

template <typename T, typename Context>
void ReluKernel(const Context& dev_ctx,
                const phi::DenseTensor& x,
                phi::DenseTensor* out) {
  DO_COMPATIBILITY(
      aclnnRelu, (custom_kernel::AclopReluKernel<T, Context>(dev_ctx, x, out)));
  dev_ctx.template Alloc<T>(out);
  EXEC_NPU_CMD(aclnnRelu, dev_ctx, x, *out);
}

template <typename T, typename Context>
void AclopReluGradKernel(const Context& dev_ctx,
                         const phi::DenseTensor& out,
                         const phi::DenseTensor& dout,
                         phi::DenseTensor* dx) {
  auto stream = dev_ctx.stream();
  dev_ctx.template Alloc<T>(dx);
  const auto& runner = NpuOpRunner("ReluGrad", {dout, out}, {*dx}, {});
  runner.Run(stream);
}

template <typename T, typename Context>
void ReluGradKernel(const Context& dev_ctx,
                    const phi::DenseTensor& out,
                    const phi::DenseTensor& dout,
                    phi::DenseTensor* dx) {
  DO_COMPATIBILITY(
      aclnnThresholdBackward,
      (custom_kernel::AclopReluGradKernel<T, Context>(dev_ctx, out, dout, dx)));
  dev_ctx.template Alloc<T>(dx);
  phi::Scalar threshold = 0.0;
  EXEC_NPU_CMD(aclnnThresholdBackward, dev_ctx, dout, out, threshold, *dx);
}

template <typename T, typename Context>
void Relu6RawKernel(const Context& dev_ctx,
                    const phi::DenseTensor& x,
                    float threshold,
                    phi::DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);
  const auto& runner = NpuOpRunner("Relu6", {x}, {*out}, {});
  auto stream = dev_ctx.stream();
  runner.Run(stream);
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
  auto stream = dev_ctx.stream();
  dev_ctx.template Alloc<T>(dx);
  const auto& runner = NpuOpRunner("Relu6Grad", {dout, out}, {*dx}, {});
  runner.Run(stream);
}

template <typename T, typename Context>
void LeakyReluKernel(const Context& dev_ctx,
                     const phi::DenseTensor& x,
                     float alpha,
                     phi::DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);

  auto stream = dev_ctx.stream();
  const auto& runner =
      NpuOpRunner("LeakyRelu", {x}, {*out}, {{"negative_slope", alpha}});
  runner.Run(stream);
}

template <typename T, typename Context>
void LeakyReluGradKernel(const Context& dev_ctx,
                         const phi::DenseTensor& x,
                         const phi::DenseTensor& dout,
                         float alpha,
                         phi::DenseTensor* dx) {
  dev_ctx.template Alloc<T>(dx);

  auto stream = dev_ctx.stream();
  const auto& runner = NpuOpRunner(
      "LeakyReluGrad", {dout, x}, {*dx}, {{"negative_slope", alpha}});
  runner.Run(stream);
}

template <typename T, typename Context>
void AclopGeluKernel(const Context& dev_ctx,
                     const phi::DenseTensor& x,
                     bool approximate,
                     phi::DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);

  auto stream = dev_ctx.stream();
  const auto& runner = NpuOpRunner("Gelu", {x}, {*out}, {});
  runner.Run(stream);
}

template <typename T, typename Context>
void GeluKernel(const Context& dev_ctx,
                const phi::DenseTensor& x,
                bool approximate,
                phi::DenseTensor* out) {
  DO_COMPATIBILITY(aclnnGelu,
                   (custom_kernel::AclopGeluKernel<T, Context>(
                       dev_ctx, x, approximate, out)));
  dev_ctx.template Alloc<T>(out);
  EXEC_NPU_CMD(aclnnGelu, dev_ctx, x, *out);
}

template <typename T, typename Context>
void AclopGeluGradKernel(const Context& dev_ctx,
                         const phi::DenseTensor& x,
                         const phi::DenseTensor& out_grad,
                         bool approximate,
                         phi::DenseTensor* x_grad) {
  dev_ctx.template Alloc<T>(x_grad);
  auto stream = dev_ctx.stream();

  // NOTE(pangyoki): In the original implementation of GeluGrad op, the input
  // is {*dout, *x, out}, where out = Gelu(x). However, we find that variable
  // `out` was not actually used. In order to improve performance, the
  // useless GELU operation was deleted.
  // We directly use `*dout` as a placeholder to replace `out`, it will not
  // be used in calculations.
  const auto& runner_dx =
      NpuOpRunner("GeluGrad", {out_grad, x, out_grad}, {*x_grad}, {});
  runner_dx.Run(stream);
}

template <typename T, typename Context>
void GeluGradKernel(const Context& dev_ctx,
                    const phi::DenseTensor& x,
                    const phi::DenseTensor& out_grad,
                    bool approximate,
                    phi::DenseTensor* x_grad) {
  DO_COMPATIBILITY(aclnnGeluBackward,
                   (custom_kernel::AclopGeluGradKernel<T, Context>(
                       dev_ctx, x, out_grad, approximate, x_grad)));
  dev_ctx.template Alloc<T>(x_grad);
  EXEC_NPU_CMD(aclnnGeluBackward, dev_ctx, out_grad, x, *x_grad);
}

template <typename T, typename Context>
void TanhKernel(const Context& dev_ctx,
                const phi::DenseTensor& x,
                phi::DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);

  auto stream = dev_ctx.stream();
  const auto& runner = NpuOpRunner("Tanh", {x}, {*out}, {});
  runner.Run(stream);
}

template <typename T, typename Context>
void TanhGradKernel(const Context& dev_ctx,
                    const phi::DenseTensor& out,
                    const phi::DenseTensor& out_grad,
                    phi::DenseTensor* x_grad) {
  dev_ctx.template Alloc<T>(x_grad);
  auto stream = dev_ctx.stream();

  const auto& runner_dx =
      NpuOpRunner("TanhGrad", {out, out_grad}, {*x_grad}, {});
  runner_dx.Run(stream);
}

template <typename T, typename Context>
void AclopSigmoidKernel(const Context& dev_ctx,
                        const phi::DenseTensor& x,
                        phi::DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);
  auto stream = dev_ctx.stream();

  const auto& runner = NpuOpRunner("Sigmoid", {x}, {*out}, {});
  runner.Run(stream);
}

template <typename T, typename Context>
void SigmoidKernel(const Context& dev_ctx,
                   const phi::DenseTensor& x,
                   phi::DenseTensor* out) {
  DO_COMPATIBILITY(
      aclnnSigmoid,
      (custom_kernel::AclopSigmoidKernel<T, Context>(dev_ctx, x, out)));
  dev_ctx.template Alloc<T>(out);
  EXEC_NPU_CMD(aclnnSigmoid, dev_ctx, x, *out);
}

template <typename T, typename Context>
void AclopSigmoidGradKernel(const Context& dev_ctx,
                            const phi::DenseTensor& out,
                            const phi::DenseTensor& dout,
                            phi::DenseTensor* dx) {
  dev_ctx.template Alloc<T>(dx);
  auto stream = dev_ctx.stream();

  const auto& runner = NpuOpRunner("SigmoidGrad", {out, dout}, {*dx}, {});
  runner.Run(stream);
}

template <typename T, typename Context>
void SigmoidGradKernel(const Context& dev_ctx,
                       const phi::DenseTensor& out,
                       const phi::DenseTensor& dout,
                       phi::DenseTensor* dx) {
  DO_COMPATIBILITY(aclnnSigmoidBackward,
                   (custom_kernel::AclopSigmoidGradKernel<T, Context>(
                       dev_ctx, out, dout, dx)));
  dev_ctx.template Alloc<T>(dx);
  EXEC_NPU_CMD(aclnnSigmoidBackward, dev_ctx, dout, out, *dx);
}

template <typename T, typename Context>
void EluKernel(const Context& dev_ctx,
               const phi::DenseTensor& x,
               float alpha,
               phi::DenseTensor* out) {
  float scaleValue = 1.0f;
  float inputScaleValue = 1.0f;
  NPUAttributeMap attr_input = {{"alpha", alpha},
                                {"scale", scaleValue},
                                {"input_scale", inputScaleValue}};
  dev_ctx.template Alloc<T>(out);
  const auto& runner = NpuOpRunner("Elu", {x}, {*out}, attr_input);
  auto stream = dev_ctx.stream();
  runner.Run(stream);
}

template <typename T, typename Context>
void AclopEluGradKernel(const Context& dev_ctx,
                        const phi::DenseTensor& x,  // output
                        const phi::DenseTensor& out,
                        const phi::DenseTensor& dout,
                        float alpha,
                        phi::DenseTensor* dx) {
  dev_ctx.template Alloc<T>(dx);
  auto stream = dev_ctx.stream();
  bool isResult = true;
  const auto& runner = NpuOpRunner("EluGradV2",
                                   {dout, out},
                                   {*dx},
                                   {{"alpha", alpha}, {"is_result", isResult}});
  runner.Run(stream);
}

template <typename T, typename Context>
void EluGradKernel(const Context& dev_ctx,
                   const phi::DenseTensor& x,  // output
                   const phi::DenseTensor& out,
                   const phi::DenseTensor& dout,
                   float alpha,
                   phi::DenseTensor* dx) {
  DO_COMPATIBILITY(aclnnEluBackward,
                   (custom_kernel::AclopEluGradKernel<T, Context>(
                       dev_ctx, x, out, dout, alpha, dx)));
  dev_ctx.template Alloc<T>(dx);
  phi::Scalar alpha_ = alpha;
  phi::Scalar input_scale = 1.0f;
  phi::Scalar scale = 1.0f;
  bool isResult = false;
  EXEC_NPU_CMD(aclnnEluBackward,
               dev_ctx,
               dout,
               alpha_,
               scale,
               input_scale,
               isResult,
               x,
               *dx);
}

template <typename T, typename Context>
void CeluKernel(const Context& dev_ctx,
                const phi::DenseTensor& x,
                float alpha,
                phi::DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);
  const auto& runner = NpuOpRunner("CeluV2", {x}, {*out}, {{"alpha", alpha}});
  auto stream = dev_ctx.stream();
  runner.Run(stream);
}

template <typename T, typename Context>
void AclopCeluGradKernel(const Context& dev_ctx,
                         const phi::DenseTensor& x,
                         const phi::DenseTensor& dout,
                         float alpha,
                         phi::DenseTensor* dx) {
  auto stream = dev_ctx.stream();
  phi::DenseTensor tmp_out;
  phi::DenseTensorMeta meta = {x.dtype(), x.dims()};
  tmp_out.set_meta(meta);

  float times = alpha != 0.0 ? (1 / alpha) : 0.0;
  dev_ctx.template Alloc<T>(&tmp_out);

  const auto& runner =
      NpuOpRunner("CeluV2", {x}, {tmp_out}, {{"alpha", alpha}});
  runner.Run(stream);

  const auto& runner_mul =
      NpuOpRunner("Muls", {tmp_out}, {tmp_out}, {{"value", times}});
  runner_mul.Run(stream);

  bool isResult = true;
  dev_ctx.template Alloc<T>(dx);
  const auto& runner_1 =
      NpuOpRunner("EluGradV2",
                  {dout, tmp_out},
                  {*dx},
                  {{"alpha", times * alpha}, {"is_result", isResult}});
  runner_1.Run(stream);
}

template <typename T, typename Context>
void CeluGradKernel(const Context& dev_ctx,
                    const phi::DenseTensor& x,
                    const phi::DenseTensor& dout,
                    float alpha,
                    phi::DenseTensor* dx) {
  DO_COMPATIBILITY(aclnnEluBackward,
                   (custom_kernel::AclopCeluGradKernel<T, Context>(
                       dev_ctx, x, dout, alpha, dx)));

  float times = alpha != 0.0 ? (1 / alpha) : 0.0;
  dev_ctx.template Alloc<T>(dx);
  phi::Scalar alpha_ = alpha;
  phi::Scalar input_scale = times;
  phi::Scalar scale = 1.0f;
  bool isResult = false;
  EXEC_NPU_CMD(aclnnEluBackward,
               dev_ctx,
               dout,
               alpha_,
               scale,
               input_scale,
               isResult,
               x,
               *dx);
}

template <typename T, typename Context>
void SeluKernel(const Context& dev_ctx,
                const phi::DenseTensor& x,
                float scale,
                float alpha,
                phi::DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);
  auto stream = dev_ctx.stream();

  NpuOpRunner runner;
  runner.SetType("Selu")
      .AddInput(x)
      .AddOutput(*out)
      .AddAttr("scale", scale)
      .AddAttr("alpha", alpha);
  runner.Run(stream);
}

template <typename T, typename Context>
void SeluGradKernel(const Context& dev_ctx,
                    const phi::DenseTensor& out,
                    const phi::DenseTensor& dout,
                    float scale,
                    float alpha,
                    phi::DenseTensor* dx) {
  dev_ctx.template Alloc<T>(dx);
  auto stream = dev_ctx.stream();

  NpuOpRunner runner;
  // NOTE(songkai05): SeluGrad do not support double dtype
  runner.SetType("SeluGrad")
      .AddInput(dout)
      .AddInput(out)
      .AddOutput(*dx)
      .AddAttr("scale", scale)
      .AddAttr("alpha", alpha);
  runner.Run(stream);
}

template <typename T, typename Context>
void AclopSqrtKernel(const Context& dev_ctx,
                     const phi::DenseTensor& x,
                     phi::DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);
  auto stream = dev_ctx.stream();

  const auto& runner = NpuOpRunner("Sqrt", {x}, {*out}, {});
  runner.Run(stream);
}

template <typename T, typename Context>
void SqrtKernel(const Context& dev_ctx,
                const phi::DenseTensor& x,
                phi::DenseTensor* out) {
  DO_COMPATIBILITY(
      aclnnSqrt, (custom_kernel::AclopSqrtKernel<T, Context>(dev_ctx, x, out)));
  dev_ctx.template Alloc<T>(out);
  EXEC_NPU_CMD(aclnnSqrt, dev_ctx, x, *out);
}

template <typename T, typename Context>
void SqrtGradAClop(const Context& dev_ctx,
                   const phi::DenseTensor& out,
                   const phi::DenseTensor& dout,
                   phi::DenseTensor* dx) {
  dev_ctx.template Alloc<T>(dx);
  auto stream = dev_ctx.stream();

  const auto& runner = NpuOpRunner("SqrtGrad", {out, dout}, {*dx}, {});
  runner.Run(stream);
}

template <typename T, typename Context>
void SqrtGradKernel(const Context& dev_ctx,
                    const phi::DenseTensor& out,
                    const phi::DenseTensor& dout,
                    phi::DenseTensor* dx) {
  DO_COMPATIBILITY(
      aclnnMuls,
      (custom_kernel::SqrtGradAClop<T, Context>(dev_ctx, out, dout, dx)));
  dev_ctx.template Alloc<T>(dx);
  auto stream = dev_ctx.stream();
  phi::DenseTensor x_tmp;
  x_tmp.Resize(out.dims());
  dev_ctx.template Alloc<T>(&x_tmp);
  auto two_value = static_cast<T>(2.0);
  aclDataType acl_data_type = ConvertToNpuDtype(out.dtype());
  static const auto aclCreateScalar = GET_OP_API_FUNC(aclCreateScalar);
  aclScalar* acl_scalar = aclCreateScalar(&two_value, acl_data_type);
  EXEC_NPU_CMD(aclnnMuls, dev_ctx, out, acl_scalar, x_tmp);
  EXEC_NPU_CMD(aclnnDiv, dev_ctx, dout, x_tmp, *dx);
}

template <typename T, typename Context>
void AclopLogKernel(const Context& dev_ctx,
                    const phi::DenseTensor& x,
                    phi::DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);
  auto stream = dev_ctx.stream();

  phi::DenseTensor one;
  phi::DenseTensorMeta one_meta = {x.dtype(), x.dims()};
  one.set_meta(one_meta);
  dev_ctx.template Alloc<T>(&one);
  const auto& runner_one = NpuOpRunner("OnesLike", {x}, {one}, {});
  runner_one.Run(stream);

  phi::DenseTensor sub;
  phi::DenseTensorMeta sub_meta = {x.dtype(), x.dims()};
  sub.set_meta(sub_meta);
  dev_ctx.template Alloc<T>(&sub);
  const auto& runner_sub = NpuOpRunner("Sub", {x, one}, {sub}, {});
  runner_sub.Run(stream);

  const auto& runner_out = NpuOpRunner("Log1p", {sub}, {*out}, {});
  runner_out.Run(stream);
}

template <typename T, typename Context>
void LogKernel(const Context& dev_ctx,
               const phi::DenseTensor& x,
               phi::DenseTensor* out) {
  DO_COMPATIBILITY(
      aclnnLog1p, (custom_kernel::AclopLogKernel<T, Context>(dev_ctx, x, out)));
  dev_ctx.template Alloc<T>(out);
  phi::DenseTensor one;
  phi::DenseTensorMeta one_meta = {x.dtype(), x.dims()};
  one.set_meta(one_meta);
  dev_ctx.template Alloc<T>(&one);
  EXEC_NPU_CMD(aclnnInplaceOne, dev_ctx, one);

  phi::DenseTensor sub;
  phi::DenseTensorMeta sub_meta = {x.dtype(), x.dims()};
  sub.set_meta(sub_meta);
  dev_ctx.template Alloc<T>(&sub);

  aclDataType acl_data_type = ConvertToNpuDtype(x.dtype());
  static const auto aclCreateScalar = GET_OP_API_FUNC(aclCreateScalar);
  auto alpha_value = static_cast<T>(1.0);
  aclScalar* alpha = aclCreateScalar(&alpha_value, acl_data_type);

  EXEC_NPU_CMD(aclnnSub, dev_ctx, x, one, alpha, sub);
  EXEC_NPU_CMD(aclnnLog1p, dev_ctx, sub, *out);
}

template <typename T, typename Context>
void AclopLogGradKernel(const Context& dev_ctx,
                        const phi::DenseTensor& x,
                        const phi::DenseTensor& dout,
                        phi::DenseTensor* dx) {
  dev_ctx.template Alloc<T>(dx);
  auto stream = dev_ctx.stream();

  const auto& runner = NpuOpRunner("DivNoNan", {dout, x}, {*dx}, {});
  runner.Run(stream);
}

template <typename T, typename Context>
void LogGradKernel(const Context& dev_ctx,
                   const phi::DenseTensor& x,
                   const phi::DenseTensor& dout,
                   phi::DenseTensor* dx) {
  DO_COMPATIBILITY(
      aclnnDiv,
      (custom_kernel::AclopLogGradKernel<T, Context>(dev_ctx, x, dout, dx)));
  dev_ctx.template Alloc<T>(dx);
  EXEC_NPU_CMD(aclnnDiv, dev_ctx, dout, x, *dx);
}

template <typename T, typename Context>
void Log2Kernel(const Context& dev_ctx,
                const phi::DenseTensor& x,
                phi::DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);
  auto stream = dev_ctx.stream();

  const auto& runner = NpuOpRunner("Log",
                                   {x},
                                   {*out},
                                   {{"base", static_cast<float>(2.0)},
                                    {"scale", static_cast<float>(1.0)},
                                    {"shift", static_cast<float>(0.0)}});
  runner.Run(stream);
}

template <typename T, typename Context>
void Log2GradKernel(const Context& dev_ctx,
                    const phi::DenseTensor& x,
                    const phi::DenseTensor& dout,
                    phi::DenseTensor* dx) {
  dev_ctx.template Alloc<T>(dx);
  auto stream = dev_ctx.stream();

  phi::DenseTensor x_log2;
  x_log2.Resize(x.dims());
  dev_ctx.template Alloc<T>(&x_log2);

  const auto& runner_mul = NpuOpRunner(
      "Muls", {x}, {x_log2}, {{"value", static_cast<float>(log(2))}});
  runner_mul.Run(stream);

  const auto& runner_div = NpuOpRunner("DivNoNan", {dout, x_log2}, {*dx}, {});
  runner_div.Run(stream);
}

template <typename T, typename Context>
void AclopPowKernel(const Context& dev_ctx,
                    const phi::DenseTensor& x,
                    const phi::Scalar& factor_scalar,
                    phi::DenseTensor* out) {
  auto factor = factor_scalar.to<float>();
  dev_ctx.template Alloc<T>(out);

  const auto& runner = NpuOpRunner("Power",
                                   {x},
                                   {*out},
                                   {{"power", factor},
                                    {"scale", static_cast<float>(1.0)},
                                    {"shift", static_cast<float>(0.0)}});
  auto stream = dev_ctx.stream();

  runner.Run(stream);
}

template <typename T, typename Context>
void PowKernel(const Context& dev_ctx,
               const phi::DenseTensor& x,
               const phi::Scalar& factor_scalar,
               phi::DenseTensor* out) {
  DO_COMPATIBILITY(aclnnPowTensorScalar,
                   (custom_kernel::AclopPowKernel<T, Context>(
                       dev_ctx, x, factor_scalar, out)));
  dev_ctx.template Alloc<T>(out);
  EXEC_NPU_CMD(aclnnPowTensorScalar, dev_ctx, x, factor_scalar, *out);
}

template <typename T, typename Context>
void PowGradAclop(const Context& dev_ctx,
                  const phi::DenseTensor& x,
                  const phi::DenseTensor& dout,
                  const phi::Scalar& factor_scalar,
                  phi::DenseTensor* dx) {
  auto factor = factor_scalar.to<float>();

  auto x_dims = x.dims();

  auto stream = dev_ctx.stream();

  // dx = dout * factor * x.pow(factor-1)

  // Step1: Compute x_pow = x.pow(factor-1)
  phi::DenseTensor x_pow;
  phi::DenseTensorMeta x_pow_meta = {x.dtype(), x_dims};
  x_pow.set_meta(x_pow_meta);
  dev_ctx.template Alloc<T>(&x_pow);

  const auto& runner_pow = NpuOpRunner(
      "Power", {x}, {x_pow}, {{"power", factor - static_cast<float>(1)}});
  runner_pow.Run(stream);

  // Step 2: Construct a broadcast factor, which has the same shape with x.

  // 2.1 Get a factor tensor with shape [1].
  phi::DenseTensor factor_tensor;
  phi::DenseTensorMeta factor_tensor_meta = {x.dtype(), {1}};
  factor_tensor.set_meta(factor_tensor_meta);
  dev_ctx.template Alloc<T>(&factor_tensor);

  FillNpuTensorWithConstant<T>(&factor_tensor, dev_ctx, static_cast<T>(factor));

  // 2.2 Get the factor which has the shape with x and the same value with
  // factor.
  phi::DenseTensor factor_bc_tensor;
  if (x_dims.size() > 0) {
    phi::DenseTensorMeta factor_bc_tensor_meta = {x.dtype(), x_dims};
    factor_bc_tensor.set_meta(factor_bc_tensor_meta);
    dev_ctx.template Alloc<T>(&factor_bc_tensor);
    if (factor_bc_tensor.numel() > 1) {
#if (CANN_VERSION_CODE >= 700000)
      phi::DenseTensor factor_cast_tensor;
      factor_cast_tensor.set_meta(factor_tensor_meta);
      dev_ctx.template Alloc<T>(&factor_cast_tensor);
      custom_kernel::CastKernel<T, Context>(
          dev_ctx, factor_tensor, phi::DataType::INT32, &factor_cast_tensor);
      NpuOpRunner runner;
      runner.SetType("Fill")
          .AddInput(dev_ctx, phi::vectorize(x_dims))
          .AddInput(factor_cast_tensor)
          .AddOutput(factor_bc_tensor)
          .Run(stream);
#else
      const auto& runner_bc = NpuOpRunner("FillD",
                                          {factor_tensor},
                                          {factor_bc_tensor},
                                          {{"dims", phi::vectorize(x_dims)}});
      runner_bc.Run(stream);
#endif
    } else {
      // CANN op Fill/FillD would raise error when output's numel is 1.
      FillNpuTensorWithConstant<T>(
          &factor_bc_tensor, dev_ctx, static_cast<T>(factor));
    }
  } else {
    factor_bc_tensor = factor_tensor;
    factor_bc_tensor.Resize(x_dims);
  }

  // Step 3: Compute x_power_mul_factor = factor * x.pow(factor-1)
  phi::DenseTensor x_power_mul_factor;
  phi::DenseTensorMeta x_power_mul_factor_meta = {x.dtype(), x_dims};
  x_power_mul_factor.set_meta(x_power_mul_factor_meta);
  dev_ctx.template Alloc<T>(&x_power_mul_factor);
  const auto& runner_mul_1 =
      NpuOpRunner("Mul", {factor_bc_tensor, x_pow}, {x_power_mul_factor}, {});
  runner_mul_1.Run(stream);

  // Step 4: Compute dx = dout * factor * x.pow(factor-1)
  dev_ctx.template Alloc<T>(dx);
  const auto& runner_mul_2 =
      NpuOpRunner("Mul", {dout, x_power_mul_factor}, {*dx}, {});
  runner_mul_2.Run(stream);
}

template <typename T, typename Context>
void PowGradKernel(const Context& dev_ctx,
                   const phi::DenseTensor& x,
                   const phi::DenseTensor& dout,
                   const phi::Scalar& factor_scalar,
                   phi::DenseTensor* dx) {
  DO_COMPATIBILITY(aclnnPowTensorScalar,
                   (custom_kernel::PowGradAclop<T, Context>(
                       dev_ctx, x, dout, factor_scalar, dx)));
  auto factor = factor_scalar.to<float>();
  auto x_dims = x.dims();
  phi::DenseTensor x_pow;
  phi::DenseTensorMeta x_pow_meta = {x.dtype(), x_dims};
  x_pow.set_meta(x_pow_meta);
  dev_ctx.template Alloc<T>(&x_pow);
  auto factor_scalar1 = static_cast<T>(factor - 1.0);
  aclDataType acl_data_type = ConvertToNpuDtype(x.dtype());
  static const auto aclCreateScalar = GET_OP_API_FUNC(aclCreateScalar);
  aclScalar* acl_scalar = aclCreateScalar(&factor_scalar1, acl_data_type);
  EXEC_NPU_CMD(aclnnPowTensorScalar, dev_ctx, x, acl_scalar, x_pow);
  auto factor_scalar2 = static_cast<T>(factor);
  aclScalar* acl_scalar2 = aclCreateScalar(&factor_scalar2, acl_data_type);
  dev_ctx.template Alloc<T>(dx);
  EXEC_NPU_CMD(aclnnMuls, dev_ctx, x_pow, acl_scalar2, x_pow);
  EXEC_NPU_CMD(aclnnMul, dev_ctx, dout, x_pow, *dx);
}

template <typename T, typename Context>
void AclopSquareKernel(const Context& dev_ctx,
                       const phi::DenseTensor& x,
                       phi::DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);
  auto stream = dev_ctx.stream();

  const auto& runner = NpuOpRunner("Square", {x}, {*out}, {});
  runner.Run(stream);
}

template <typename T, typename Context>
void SquareKernel(const Context& dev_ctx,
                  const phi::DenseTensor& x,
                  phi::DenseTensor* out) {
  DO_COMPATIBILITY(
      aclnnMul,
      (custom_kernel::AclopSquareKernel<T, Context>(dev_ctx, x, out)));
  dev_ctx.template Alloc<T>(out);
  EXEC_NPU_CMD(aclnnMul, dev_ctx, x, x, *out);
}

template <typename T, typename Context>
void SquareGradKernel(const Context& dev_ctx,
                      const phi::DenseTensor& x,
                      const phi::DenseTensor& dout,
                      phi::DenseTensor* dx) {
  auto stream = dev_ctx.stream();

  auto factor = static_cast<float>(2.0);

  // Step 1: Compute x_muls_factor = factor * x
  phi::DenseTensor x_muls_factor;
  phi::DenseTensorMeta x_muls_factor_meta = {x.dtype(), x.dims()};
  x_muls_factor.set_meta(x_muls_factor_meta);
  dev_ctx.template Alloc<T>(&x_muls_factor);

  const auto& runner_muls_1 =
      NpuOpRunner("Muls", {x}, {x_muls_factor}, {{"value", factor}});
  runner_muls_1.Run(stream);

  // Step 2: Compute dx = dout * factor * x
  dev_ctx.template Alloc<T>(dx);
  const auto& runner_mul_2 =
      NpuOpRunner("Mul", {dout, x_muls_factor}, {*dx}, {});
  runner_mul_2.Run(stream);
}

template <typename T, typename Context>
void HardTanhKernel(const Context& dev_ctx,
                    const phi::DenseTensor& x,
                    float t_min,
                    float t_max,
                    phi::DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);
  // t_min and t_max
  phi::DenseTensor min_tensor, max_tensor;
  phi::DenseTensorMeta meta1 = {x.dtype(), {1}};
  min_tensor.set_meta(meta1);
  max_tensor.set_meta(meta1);
  dev_ctx.template Alloc<T>(&min_tensor);
  dev_ctx.template Alloc<T>(&max_tensor);

  FillNpuTensorWithConstant<T>(&min_tensor, dev_ctx, static_cast<T>(t_min));
  FillNpuTensorWithConstant<T>(&max_tensor, dev_ctx, static_cast<T>(t_max));

  auto stream = dev_ctx.stream();
  const auto& runner =
      NpuOpRunner("ClipByValue", {x, min_tensor, max_tensor}, {*out}, {});
  runner.Run(stream);
}

template <typename T, typename Context>
void HardTanhGradKernel(const Context& dev_ctx,
                        const phi::DenseTensor& out,
                        const phi::DenseTensor& dout,
                        float t_min,
                        float t_max,
                        phi::DenseTensor* dx) {
  dev_ctx.template Alloc<T>(dx);

  NPUAttributeMap attr_input = {{"min_val", t_min}, {"max_val", t_max}};
  auto stream = dev_ctx.stream();
  const auto& runner_dx =
      NpuOpRunner("HardtanhGrad", {out, dout}, {*dx}, attr_input);
  runner_dx.Run(stream);
}

template <typename T, typename Context>
void HardSigmoidKernel(const Context& dev_ctx,
                       const phi::DenseTensor& x,
                       float slope,
                       float offset,
                       phi::DenseTensor* out) {
  phi::DenseTensor scaled_x;
  scaled_x.set_meta(x.meta());
  custom_kernel::ScaleKernel<T, Context>(
      dev_ctx, x, phi::Scalar(slope), phi::Scalar(offset), true, &scaled_x);

  custom_kernel::ClipKernel<T, Context>(
      dev_ctx, scaled_x, phi::Scalar(0.0f), phi::Scalar(1.0f), out);
}

template <typename T, typename Context>
void AclopHardSigmoidGradKernel(const Context& dev_ctx,
                                const phi::DenseTensor& out,
                                const phi::DenseTensor& dout,
                                float slope,
                                float offset,
                                phi::DenseTensor* dx) {
  dev_ctx.template Alloc<T>(dx);

  NPUAttributeMap attr_input = {{"alpha", slope}, {"beta", offset}};

  auto stream = dev_ctx.stream();
  const auto& runner_dx =
      NpuOpRunner("HardSigmoidGrad", {dout, out}, {*dx}, attr_input);
  runner_dx.Run(stream);
}

template <typename T, typename Context>
void HardSigmoidGradKernel(const Context& dev_ctx,
                           const phi::DenseTensor& out,
                           const phi::DenseTensor& dout,
                           float slope,
                           float offset,
                           phi::DenseTensor* dx) {
  dev_ctx.template Alloc<T>(dx);
  int max_ = 1;
  int min_ = 0;
  phi::DenseTensorMeta number_meta = {phi::DataType::INT32, out.dims()};

  phi::DenseTensor max_tensor;
  max_tensor.set_meta(number_meta);
  FillNpuTensorWithConstant<int>(&max_tensor, dev_ctx, max_);
  max_tensor.Resize(out.dims());

  phi::DenseTensor min_tensor;
  min_tensor.set_meta(number_meta);
  FillNpuTensorWithConstant<int>(&min_tensor, dev_ctx, min_);
  min_tensor.Resize(out.dims());

  phi::DenseTensorMeta out_meta = {phi::DataType::BOOL, out.dims()};
  phi::DenseTensor min_out;
  min_out.set_meta(out_meta);
  phi::DenseTensor max_out;
  max_out.set_meta(out_meta);
  phi::DenseTensor logical_out;
  logical_out.set_meta(out_meta);

  phi::DenseTensor dout_scale;
  phi::DenseTensorMeta dout_scale_meta = {dout.dtype(), dout.dims()};
  dout_scale.set_meta(dout_scale_meta);
  dev_ctx.template Alloc<T>(&dout_scale);
  phi::Scalar scale = slope;
  phi::Scalar bias = 0;
  custom_kernel::ScaleKernel<T, Context>(
      dev_ctx, dout, scale, bias, false, &dout_scale);

  custom_kernel::GreaterThanKernel<T, Context>(
      dev_ctx, out, min_tensor, &min_out);
  custom_kernel::LessThanKernel<T, Context>(dev_ctx, out, max_tensor, &max_out);
  custom_kernel::LogicalAndNPUKernel<T, Context>(
      dev_ctx, min_out, max_out, &logical_out);
  custom_kernel::WhereKernel<T, Context>(
      dev_ctx, logical_out, dout_scale, min_tensor, dx);
}

template <typename T, typename Context>
void AclopHardSwishKernel(const Context& dev_ctx,
                          const phi::DenseTensor& x,
                          phi::DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);
  auto stream = dev_ctx.stream();
  float threshold = 6;
  float scale = 6;
  float offset = 3;
  phi::DenseTensorMeta meta_1 = {x.dtype(), {1}};
  phi::DenseTensorMeta meta_x = {x.dtype(), x.dims()};
  phi::DenseTensor tensor_offset;
  tensor_offset.set_meta(meta_1);
  dev_ctx.template Alloc<T>(&tensor_offset);
  FillNpuTensorWithConstant<T>(&tensor_offset, dev_ctx, static_cast<T>(offset));

  phi::DenseTensor add_offset_val;
  add_offset_val.set_meta(meta_x);
  dev_ctx.template Alloc<T>(&add_offset_val);
  const auto& runner_add =
      NpuOpRunner("AddV2", {x, tensor_offset}, {add_offset_val});
  runner_add.Run(stream);

  phi::DenseTensor tensor_threshold;
  tensor_threshold.set_meta(meta_1);
  dev_ctx.template Alloc<T>(&tensor_threshold);
  FillNpuTensorWithConstant<T>(
      &tensor_threshold, dev_ctx, static_cast<T>(threshold));

  phi::DenseTensor tensor_zero;
  tensor_zero.set_meta(meta_1);
  dev_ctx.template Alloc<T>(&tensor_zero);
  FillNpuTensorWithConstant<T>(&tensor_zero, dev_ctx, static_cast<T>(0.0));

  phi::DenseTensor clip_val;
  clip_val.set_meta(meta_x);
  dev_ctx.template Alloc<T>(&clip_val);
  const auto& runner_clip =
      NpuOpRunner("ClipByValue",
                  {add_offset_val, tensor_zero, tensor_threshold},
                  {clip_val});
  runner_clip.Run(stream);

  phi::DenseTensor tensor_scale_tmp;
  tensor_scale_tmp.set_meta(meta_1);
  dev_ctx.template Alloc<T>(&tensor_scale_tmp);
  FillNpuTensorWithConstant<T>(
      &tensor_scale_tmp, dev_ctx, static_cast<T>(scale));
  phi::DenseTensor tensor_scale;
  if (x.dims().size() > 0) {
    tensor_scale.set_meta(meta_x);
    dev_ctx.template Alloc<T>(&tensor_scale);
    if (tensor_scale.numel() > 1) {
      const auto& runner_fill =
          NpuOpRunner("FillD",
                      {tensor_scale_tmp},
                      {tensor_scale},
                      {{"dims", phi::vectorize(x.dims())}});
      runner_fill.Run(stream);
    } else {
      // CANN op Fill/FillD would raise error when output's numel is 1.
      FillNpuTensorWithConstant<T>(
          &tensor_scale, dev_ctx, static_cast<T>(scale));
    }
  } else {
    tensor_scale = tensor_scale_tmp;
    tensor_scale.Resize(x.dims());
  }

  phi::DenseTensor div_val;
  div_val.set_meta(meta_x);
  dev_ctx.template Alloc<T>(&div_val);
  const auto& runner_div =
      NpuOpRunner("Div", {clip_val, tensor_scale}, {div_val});
  runner_div.Run(stream);

  const auto& runner_mul = NpuOpRunner("Mul", {x, div_val}, {*out});
  runner_mul.Run(stream);
}

template <typename T, typename Context>
void HardSwishKernel(const Context& dev_ctx,
                     const phi::DenseTensor& x,
                     phi::DenseTensor* out) {
  DO_COMPATIBILITY(
      aclnnHardswish,
      (custom_kernel::AclopHardSwishKernel<T, Context>(dev_ctx, x, out)));
  dev_ctx.template Alloc<T>(out);
  EXEC_NPU_CMD(aclnnHardswish, dev_ctx, x, *out);
}

template <typename T, typename Context>
void AclopHardSwishGradKernel(const Context& dev_ctx,
                              const phi::DenseTensor& x,
                              const phi::DenseTensor& dout,
                              phi::DenseTensor* dx) {
  dev_ctx.template Alloc<T>(dx);
  auto stream = dev_ctx.stream();
  float threshold = 6;
  float scale = 6;
  float offset = 3;
  phi::DenseTensorMeta meta_1 = {x.dtype(), {1}};
  phi::DenseTensorMeta meta_x = {x.dtype(), x.dims()};
  phi::DenseTensor tensor_offset;
  tensor_offset.set_meta(meta_1);
  dev_ctx.template Alloc<T>(&tensor_offset);
  FillNpuTensorWithConstant<T>(&tensor_offset, dev_ctx, static_cast<T>(offset));

  phi::DenseTensor add_offset_val;
  add_offset_val.set_meta(meta_x);
  dev_ctx.template Alloc<T>(&add_offset_val);
  const auto& runner_add =
      NpuOpRunner("AddV2", {x, tensor_offset}, {add_offset_val});
  runner_add.Run(stream);

  phi::DenseTensor tmp1;
  tmp1.set_meta(meta_x);
  dev_ctx.template Alloc<T>(&tmp1);
  const auto& runner_pow1 =
      NpuOpRunner("Power", {x}, {tmp1}, {{"scale", 2.0f}, {"shift", offset}});
  runner_pow1.Run(stream);

  phi::DenseTensor tmp2;
  tmp2.set_meta(meta_x);
  dev_ctx.template Alloc<T>(&tmp2);
  const auto& runner_ht_grad =
      NpuOpRunner("HardtanhGrad",
                  {add_offset_val, tmp1},
                  {tmp2},
                  {{"min_val", 0.0f}, {"max_val", threshold}});
  runner_ht_grad.Run(stream);

  phi::DenseTensor tmp3;
  tmp3.set_meta(meta_x);
  dev_ctx.template Alloc<T>(&tmp3);
  const auto& runner_pow2 = NpuOpRunner(
      "Power", {tmp2}, {tmp3}, {{"scale", 1.0f / scale}, {"shift", 1.0f}});
  runner_pow2.Run(stream);

  phi::DenseTensor tensor_threshold_tmp;
  tensor_threshold_tmp.set_meta(meta_1);
  dev_ctx.template Alloc<T>(&tensor_threshold_tmp);
  FillNpuTensorWithConstant<T>(
      &tensor_threshold_tmp, dev_ctx, static_cast<T>(threshold));
  phi::DenseTensor tensor_threshold;
  if (x.dims().size() > 0) {
    tensor_threshold.set_meta(meta_x);
    dev_ctx.template Alloc<T>(&tensor_threshold);
    if (tensor_threshold.numel() > 1) {
      const auto& runner_fill =
          NpuOpRunner("FillD",
                      {tensor_threshold_tmp},
                      {tensor_threshold},
                      {{"dims", phi::vectorize(x.dims())}});
      runner_fill.Run(stream);
    } else {
      // CANN op Fill/FillD would raise error when output's numel is 1.
      FillNpuTensorWithConstant<T>(
          &tensor_threshold, dev_ctx, static_cast<T>(threshold));
    }
  } else {
    tensor_threshold = tensor_threshold_tmp;
    tensor_threshold.Resize(x.dims());
  }

  phi::DenseTensor tmp_bool;
  phi::DenseTensorMeta meta_tmp = {phi::DataType::BOOL, x.dims()};
  tmp_bool.set_meta(meta_tmp);
  dev_ctx.template Alloc<bool>(&tmp_bool);
  const auto& runner_less =
      NpuOpRunner("Less", {add_offset_val, tensor_threshold}, {tmp_bool});
  runner_less.Run(stream);
  phi::DenseTensor tmp4;
  tmp4.set_meta(meta_x);
  dev_ctx.template Alloc<T>(&tmp4);
  auto dst_dtype = ConvertToNpuDtype(x.dtype());
  const auto& runner_cast = NpuOpRunner(
      "Cast", {tmp_bool}, {tmp4}, {{"dst_type", static_cast<int>(dst_dtype)}});
  runner_cast.Run(stream);

  phi::DenseTensor tmp5;
  tmp5.set_meta(meta_x);
  dev_ctx.template Alloc<T>(&tmp5);
  const auto& runner_sub = NpuOpRunner("Sub", {tmp3, tmp4}, {tmp5});
  runner_sub.Run(stream);

  const auto& runner_final = NpuOpRunner("Mul", {tmp5, dout}, {*dx});
  runner_final.Run(stream);
}

template <typename T, typename Context>
void HardSwishGradKernel(const Context& dev_ctx,
                         const phi::DenseTensor& x,
                         const phi::DenseTensor& dout,
                         phi::DenseTensor* dx) {
  DO_COMPATIBILITY(aclnnHardswishBackward,
                   (custom_kernel::AclopHardSwishGradKernel<T, Context>(
                       dev_ctx, x, dout, dx)));
  dev_ctx.template Alloc<T>(dx);
  EXEC_NPU_CMD(aclnnHardswishBackward, dev_ctx, dout, x, *dx);
}

template <typename T, typename Context>
void SoftplusKernel(const Context& dev_ctx,
                    const phi::DenseTensor& x,
                    const float beta,
                    const float threshold,
                    phi::DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);
  auto stream = dev_ctx.stream();
  const auto& runner = NpuOpRunner(
      "SoftplusV2", {x}, {*out}, {{"beta", beta}, {"threshold", threshold}});
  runner.Run(stream);
}

template <typename T, typename Context>
void SoftplusGradKernel(const Context& dev_ctx,
                        const phi::DenseTensor& a,
                        const phi::DenseTensor& dout,
                        const float beta,
                        const float threshold,
                        phi::DenseTensor* dx) {
  dev_ctx.template Alloc<T>(dx);
  auto stream = dev_ctx.stream();
  const auto& runner = NpuOpRunner("SoftplusV2Grad",
                                   {dout, a},
                                   {*dx},
                                   {{"beta", beta}, {"threshold", threshold}});
  runner.Run(stream);
}

template <typename T, typename Context>
void SoftshrinkKernel(const Context& dev_ctx,
                      const phi::DenseTensor& x,
                      const float lambd,
                      phi::DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);
  PD_CHECK(lambd > 0, "lambd should be greater than 0");
  auto stream = dev_ctx.stream();
  const auto& runner =
      NpuOpRunner("SoftShrink", {x}, {*out}, {{"lambd", lambd}});
  runner.Run(stream);
}

template <typename T, typename Context>
void SoftshrinkGradKernel(const Context& dev_ctx,
                          const phi::DenseTensor& a,
                          const phi::DenseTensor& dout,
                          const float lambd,
                          phi::DenseTensor* dx) {
  dev_ctx.template Alloc<T>(dx);
  auto stream = dev_ctx.stream();
  const auto& runner =
      NpuOpRunner("SoftShrinkGrad", {dout, a}, {*dx}, {{"lambd", lambd}});
  runner.Run(stream);
}

template <typename T, typename Context>
void CosGradKernel(const Context& dev_ctx,
                   const phi::DenseTensor& x,
                   const phi::DenseTensor& dout,
                   phi::DenseTensor* dx) {
  dev_ctx.template Alloc<T>(dx);
  auto stream = dev_ctx.stream();

  phi::DenseTensor sin_out;
  phi::DenseTensorMeta meta = {x.dtype(), x.dims()};
  sin_out.set_meta(meta);
  custom_kernel::SinKernel<T, Context>(dev_ctx, x, &sin_out);
  custom_kernel::ExpGradKernel<T, Context>(dev_ctx, dout, sin_out, dx);

  phi::DenseTensor tmp;
  phi::DenseTensorMeta tmp_meta = {x.dtype(), {1, 1}};
  tmp.set_meta(tmp_meta);
  dev_ctx.template Alloc<T>(&tmp);
  float factor = -1.;
  FillNpuTensorWithConstant<T>(&tmp, dev_ctx, static_cast<T>(factor));

  const auto& runner_dx_ = NpuOpRunner("Xdivy", {*dx, tmp}, {*dx}, {});
  runner_dx_.Run(stream);
}

template <typename T, typename Context>
void HardshrinkKernel(const Context& dev_ctx,
                      const phi::DenseTensor& x,
                      const float lambd,
                      phi::DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);
  auto stream = dev_ctx.stream();
  const auto& runner =
      NpuOpRunner("HardShrink", {x}, {*out}, {{"lambd", lambd}});
  runner.Run(stream);
}

template <typename T, typename Context>
void HardshrinkGradKernel(const Context& dev_ctx,
                          const phi::DenseTensor& a,
                          const phi::DenseTensor& dout,
                          const float lambd,
                          phi::DenseTensor* dx) {
  dev_ctx.template Alloc<T>(dx);
  auto stream = dev_ctx.stream();
  const auto& runner =
      NpuOpRunner("HardShrinkGrad", {dout, a}, {*dx}, {{"lambd", lambd}});
  runner.Run(stream);
}

template <typename T, typename Context>
void AclopReciprocalKernel(const Context& dev_ctx,
                           const phi::DenseTensor& x,
                           phi::DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);
  auto stream = dev_ctx.stream();
  const auto& runner = NpuOpRunner("Reciprocal", {x}, {*out}, {});
  runner.Run(stream);
}

template <typename T, typename Context>
void ReciprocalKernel(const Context& dev_ctx,
                      const phi::DenseTensor& x,
                      phi::DenseTensor* out) {
  DO_COMPATIBILITY(
      aclnnReciprocal,
      (custom_kernel::AclopReciprocalKernel<T, Context>(dev_ctx, x, out)));
  dev_ctx.template Alloc<T>(out);
  EXEC_NPU_CMD(aclnnReciprocal, dev_ctx, x, *out);
}

template <typename T, typename Context>
void AclopReciprocalGradKernel(const Context& dev_ctx,
                               const phi::DenseTensor& out,
                               const phi::DenseTensor& dout,
                               phi::DenseTensor* dx) {
  dev_ctx.template Alloc<T>(dx);
  auto stream = dev_ctx.stream();
  const auto& runner = NpuOpRunner("ReciprocalGrad", {out, dout}, {*dx}, {});
  runner.Run(stream);
}

template <typename T, typename Context>
void ReciprocalGradKernel(const Context& dev_ctx,
                          const phi::DenseTensor& out,
                          const phi::DenseTensor& dout,
                          phi::DenseTensor* dx) {
  DO_COMPATIBILITY(aclnnMul,
                   (custom_kernel::AclopReciprocalGradKernel<T, Context>(
                       dev_ctx, out, dout, dx)));
  dev_ctx.template Alloc<T>(dx);
  phi::DenseTensor out_tmp;
  out_tmp.Resize(out.dims());
  dev_ctx.template Alloc<T>(&out_tmp);
  auto neg_one_value = static_cast<T>(-1.0);
  aclDataType acl_data_type = ConvertToNpuDtype(out.dtype());
  static const auto aclCreateScalar = GET_OP_API_FUNC(aclCreateScalar);
  aclScalar* acl_scalar_one = aclCreateScalar(&neg_one_value, acl_data_type);
  EXEC_NPU_CMD(aclnnMul, dev_ctx, out, out, out_tmp);
  EXEC_NPU_CMD(aclnnMuls, dev_ctx, out_tmp, acl_scalar_one, out_tmp);
  EXEC_NPU_CMD(aclnnMul, dev_ctx, dout, out_tmp, *dx);
}

template <typename T, typename Context>
void MishKernel(const Context& dev_ctx,
                const phi::DenseTensor& x,
                float threshold,
                phi::DenseTensor* out) {
  auto stream = dev_ctx.stream();
  dev_ctx.template Alloc<T>(out);

  const auto& runner = NpuOpRunner("Mish", {x}, {*out}, {});
  runner.Run(stream);
}

template <typename T, typename Context>
void MishGradKernel(const Context& dev_ctx,
                    const phi::DenseTensor& x,
                    const phi::DenseTensor& dout,
                    float threshold,
                    phi::DenseTensor* dx) {
  auto stream = dev_ctx.stream();
  dev_ctx.template Alloc<T>(dx);

  const auto& runner = NpuOpRunner("MishGrad", {dout, x}, {*dx}, {});
  runner.Run(stream);
}

template <typename T, typename Context>
void RoundKernel(const Context& dev_ctx,
                 const phi::DenseTensor& x,
                 const int decimals,
                 phi::DenseTensor* out) {
  int64_t decimals_trans = decimals;
  dev_ctx.template Alloc<T>(out);
  EXEC_NPU_CMD(aclnnRoundDecimals, dev_ctx, x, decimals_trans, *out);
}

template <typename T, typename Context>
void RoundGradKernel(const Context& dev_ctx,
                     const phi::DenseTensor& dout,
                     phi::DenseTensor* dx) {
  auto dx_dims = dx->dims();
  dev_ctx.template Alloc<T>(dx);
  FillNpuTensorWithConstant<T>(dx, dev_ctx, static_cast<T>(0));
  dx->Resize(dx_dims);
}

template <typename T, typename Context>
void LogSigmoidKernel(const Context& dev_ctx,
                      const phi::DenseTensor& x,
                      phi::DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);
  EXEC_NPU_CMD(aclnnLogSigmoid, dev_ctx, x, *out);
}

template <typename T, typename Context>
void LogSigmoidGradKernel(const Context& dev_ctx,
                          const phi::DenseTensor& out,
                          const phi::DenseTensor& dout,
                          phi::DenseTensor* dx) {
  phi::DenseTensor buffer;
  phi::DenseTensorMeta buffer_meta = {out.dtype(), out.dims()};
  buffer.set_meta(buffer_meta);
  dev_ctx.template Alloc<T>(&buffer);
  EXEC_NPU_CMD(aclnnInplaceZero, dev_ctx, buffer);
  dev_ctx.template Alloc<T>(dx);
  EXEC_NPU_CMD(aclnnLogSigmoidBackward, dev_ctx, dout, out, buffer, *dx);
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(cos,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::CosKernel,
                          float,
                          double,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}

PD_REGISTER_PLUGIN_KERNEL(cos_grad,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::CosGradKernel,
                          float,
                          double,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}

PD_REGISTER_PLUGIN_KERNEL(atan,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::AtanKernel,
                          float,
                          double,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(atan_grad,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::AtanGradKernel,
                          float,
                          double,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(exp,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::ExpKernel,
                          float,
                          double,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}

PD_REGISTER_PLUGIN_KERNEL(exp_grad,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::ExpGradKernel,
                          float,
                          double,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}

PD_REGISTER_PLUGIN_KERNEL(sin,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::SinKernel,
                          float,
                          double,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}

PD_REGISTER_PLUGIN_KERNEL(swish,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::SwishKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(swish_grad,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::SwishGradKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(silu,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::SiluKernel,
                          float,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}

PD_REGISTER_PLUGIN_KERNEL(silu_grad,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::SiluGradKernel,
                          float,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}

PD_REGISTER_PLUGIN_KERNEL(relu,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::ReluKernel,
                          float,
                          double,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(relu_grad,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::ReluGradKernel,
                          float,
                          double,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(relu6,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::Relu6Kernel,
                          float,
                          double,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(relu6_grad,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::Relu6GradKernel,
                          float,
                          double,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(leaky_relu,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::LeakyReluKernel,
                          float,
                          phi::dtype::float16,
                          double) {}

PD_REGISTER_PLUGIN_KERNEL(leaky_relu_grad,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::LeakyReluGradKernel,
                          float,
                          phi::dtype::float16,
                          double) {}

PD_REGISTER_PLUGIN_KERNEL(pow,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::PowKernel,
                          float,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}

PD_REGISTER_PLUGIN_KERNEL(pow_grad,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::PowGradKernel,
                          float,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}

PD_REGISTER_PLUGIN_KERNEL(log,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::LogKernel,
                          double,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(log_grad,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::LogGradKernel,
                          double,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(log2,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::Log2Kernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(log2_grad,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::Log2GradKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(floor,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::FloorKernel,
                          double,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(floor_grad,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::FloorGradKernel,
                          double,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(gelu,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::GeluKernel,
                          float,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}

PD_REGISTER_PLUGIN_KERNEL(gelu_grad,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::GeluGradKernel,
                          float,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}

PD_REGISTER_PLUGIN_KERNEL(tanh,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::TanhKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(tanh_grad,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::TanhGradKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(sigmoid,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::SigmoidKernel,
                          float,
                          phi::dtype::float16,
                          phi::dtype::bfloat16,
                          double) {}

PD_REGISTER_PLUGIN_KERNEL(sigmoid_grad,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::SigmoidGradKernel,
                          float,
                          phi::dtype::float16,
                          phi::dtype::bfloat16,
                          double) {}

PD_REGISTER_PLUGIN_KERNEL(sqrt,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::SqrtKernel,
                          float,
                          phi::dtype::float16,
                          phi::dtype::bfloat16,
                          double) {}

PD_REGISTER_PLUGIN_KERNEL(sqrt_grad,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::SqrtGradKernel,
                          float,
                          phi::dtype::float16,
                          phi::dtype::bfloat16,
                          double) {}

PD_REGISTER_PLUGIN_KERNEL(square,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::SquareKernel,
                          float,
                          phi::dtype::float16,
                          phi::dtype::bfloat16,
                          double) {}

PD_REGISTER_PLUGIN_KERNEL(square_grad,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::SquareGradKernel,
                          float,
                          phi::dtype::float16,
                          phi::dtype::bfloat16,
                          double) {}

PD_REGISTER_PLUGIN_KERNEL(hardtanh,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::HardTanhKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(hardtanh_grad,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::HardTanhGradKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(hardsigmoid,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::HardSigmoidKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(hardsigmoid_grad,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::HardSigmoidGradKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(hardswish,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::HardSwishKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(hardswish_grad,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::HardSwishGradKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(softplus,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::SoftplusKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(softplus_grad,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::SoftplusGradKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(softshrink,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::SoftshrinkKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(softshrink_grad,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::SoftshrinkGradKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(hard_shrink,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::HardshrinkKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(hard_shrink_grad,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::HardshrinkGradKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(reciprocal,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::ReciprocalKernel,
                          float,
                          double,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(reciprocal_grad,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::ReciprocalGradKernel,
                          float,
                          double,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(selu,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::SeluKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(selu_grad,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::SeluGradKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(rsqrt,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::RsqrtKernel,
                          float,
                          double,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}

PD_REGISTER_PLUGIN_KERNEL(rsqrt_grad,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::RsqrtGradKernel,
                          float,
                          double,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}

PD_REGISTER_PLUGIN_KERNEL(elu,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::EluKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(elu_grad,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::EluGradKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(celu,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::CeluKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(celu_grad,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::CeluGradKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(mish,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::MishKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(mish_grad,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::MishGradKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(round,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::RoundKernel,
                          float,
                          phi::dtype::float16,
                          double) {}

PD_REGISTER_PLUGIN_KERNEL(round_grad,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::RoundGradKernel,
                          float,
                          phi::dtype::float16,
                          double) {}

PD_REGISTER_PLUGIN_KERNEL(logsigmoid,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::LogSigmoidKernel,
                          float,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}

PD_REGISTER_PLUGIN_KERNEL(logsigmoid_grad,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::LogSigmoidGradKernel,
                          float,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}
