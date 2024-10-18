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
#include "paddle/phi/common/type_traits.h"

namespace custom_kernel {

#define DEFINE_UNARY_AOT_ACTIVATION_KERNEL(name, functor_prefix)     \
  template <typename T, typename Context>                            \
  void functor_prefix##Kernel(const Context& dev_ctx,                \
                              const phi::DenseTensor& x,             \
                              phi::DenseTensor* out) {               \
    PADDLE_GCU_KERNEL_TRACE(#name);                                  \
    if (LaunchAOTKernel()) {                                         \
      dev_ctx.template Alloc<T>(out);                                \
      LAUNCH_TOPSATENOP(topsaten##functor_prefix, dev_ctx, *out, x); \
    } else { /* kernel impl base on JIT */                           \
      THROW_JIT_UNIMPLEMENTED();                                     \
    }                                                                \
  }

template <typename T, typename Context>
void ActivationBaseKernel(const Context& dev_ctx,
                          const phi::DenseTensor& x,
                          const GcuAttributeMap& attrs,
                          phi::DenseTensor* out,
                          const std::string& op_type) {
  dev_ctx.template Alloc<T>(out);

  TensorNameMap input_names;
  input_names["X"] = {"x"};

  TensorValueMap inputs;
  inputs["X"] = {const_cast<DenseTensor*>(&x)};

  TensorNameMap output_names;
  output_names["Out"] = {"out"};

  TensorValueMap outputs;
  outputs["Out"] = {out};

  GcuRunner(
      input_names, inputs, output_names, outputs, attrs, op_type, dev_ctx);
}

template <typename T, typename Context>
void ActivationGradBaseKernel(const Context& dev_ctx,
                              const std::string& x_name,
                              const phi::DenseTensor& x,
                              const phi::DenseTensor& dout,
                              const GcuAttributeMap& attrs,
                              phi::DenseTensor* dx,
                              const std::string& op_type) {
  dev_ctx.template Alloc<T>(dx);

  TensorNameMap input_names;
  input_names[x_name] = {"x"};
  input_names[GradVarName("Out")] = {"dout"};

  TensorValueMap inputs;
  inputs[x_name] = {const_cast<DenseTensor*>(&x)};
  inputs[GradVarName("Out")] = {const_cast<DenseTensor*>(&dout)};

  TensorNameMap output_names;
  output_names[GradVarName("X")] = {"dx"};

  TensorValueMap outputs;
  outputs[GradVarName("X")] = {dx};

  GcuRunner(
      input_names, inputs, output_names, outputs, attrs, op_type, dev_ctx);
}

template <typename T, typename Context>
void AbsKernel(const Context& dev_ctx,
               const phi::DenseTensor& x,
               phi::DenseTensor* out) {
  PADDLE_GCU_KERNEL_TRACE("abs");
  if (LaunchAOTKernel()) {
    dev_ctx.template Alloc<T>(out);
    LAUNCH_TOPSATENOP(topsatenAbs, dev_ctx, *out, x);

  } else {  // kernel impl base on JIT
    ActivationBaseKernel<T, Context>(dev_ctx, x, {}, out, "abs");
  }
}

template <typename T, typename Context>
void AbsGradKernel(const Context& dev_ctx,
                   const phi::DenseTensor& x,
                   const phi::DenseTensor& dout,
                   phi::DenseTensor* dx) {
  PADDLE_GCU_KERNEL_TRACE("abs_grad");
  if (LaunchAOTKernel()) {
    THROW_AOT_UNIMPLEMENTED();
  } else {  // kernel impl base on JIT
    ActivationGradBaseKernel<T, Context>(
        dev_ctx, "X", x, dout, {}, dx, "abs_grad");
  }
}

template <typename T, typename Context>
void CosKernel(const Context& dev_ctx,
               const phi::DenseTensor& x,
               phi::DenseTensor* out) {
  PADDLE_GCU_KERNEL_TRACE("cos");
  if (LaunchAOTKernel()) {
    dev_ctx.template Alloc<T>(out);
    LAUNCH_TOPSATENOP(topsatenCos, dev_ctx, *out, x);

  } else {  // kernel impl base on JIT
    ActivationBaseKernel<T, Context>(dev_ctx, x, {}, out, "cos");
  }
}

template <typename T, typename Context>
void CosGradKernel(const Context& dev_ctx,
                   const phi::DenseTensor& x,
                   const phi::DenseTensor& dout,
                   phi::DenseTensor* dx) {
  PADDLE_GCU_KERNEL_TRACE("cos_grad");
  if (LaunchAOTKernel()) {
    THROW_AOT_UNIMPLEMENTED();
  } else {  // kernel impl base on JIT
    ActivationGradBaseKernel<T, Context>(
        dev_ctx, "X", x, dout, {}, dx, "cos_grad");
  }
}

template <typename T, typename Context>
void SinKernel(const Context& dev_ctx,
               const phi::DenseTensor& x,
               phi::DenseTensor* out) {
  PADDLE_GCU_KERNEL_TRACE("sin");
  if (LaunchAOTKernel()) {
    dev_ctx.template Alloc<T>(out);
    LAUNCH_TOPSATENOP(topsatenSin, dev_ctx, *out, x);

  } else {  // kernel impl base on JIT
    THROW_JIT_UNIMPLEMENTED();
  }
}

template <typename T, typename Context>
void AtanKernel(const Context& dev_ctx,
                const phi::DenseTensor& x,
                phi::DenseTensor* out) {
  PADDLE_GCU_KERNEL_TRACE("atan");
  if (LaunchAOTKernel()) {
    dev_ctx.template Alloc<T>(out);
    LAUNCH_TOPSATENOP(topsatenAtan, dev_ctx, *out, x);

  } else {  // kernel impl base on JIT
    ActivationBaseKernel<T, Context>(dev_ctx, x, {}, out, "atan");
  }
}

template <typename T, typename Context>
void AtanGradKernel(const Context& dev_ctx,
                    const phi::DenseTensor& x,
                    const phi::DenseTensor& dout,
                    phi::DenseTensor* dx) {
  PADDLE_GCU_KERNEL_TRACE("atan_grad");
  if (LaunchAOTKernel()) {
    THROW_AOT_UNIMPLEMENTED();
  } else {  // kernel impl base on JIT
    ActivationGradBaseKernel<T, Context>(
        dev_ctx, "X", x, dout, {}, dx, "atan_grad");
  }
}

template <typename T, typename Context>
void ExpKernel(const Context& dev_ctx,
               const phi::DenseTensor& x,
               phi::DenseTensor* out) {
  PADDLE_GCU_KERNEL_TRACE("exp");
  if (LaunchAOTKernel()) {
    dev_ctx.template Alloc<T>(out);
    LAUNCH_TOPSATENOP(topsatenExp, dev_ctx, *out, x);

  } else {  // kernel impl base on JIT
    ActivationBaseKernel<T, Context>(dev_ctx, x, {}, out, "exp");
  }
}

template <typename T, typename Context>
void ExpGradKernel(const Context& dev_ctx,
                   const phi::DenseTensor& out,
                   const phi::DenseTensor& dout,
                   phi::DenseTensor* dx) {
  PADDLE_GCU_KERNEL_TRACE("exp_grad");
  if (LaunchAOTKernel()) {
    THROW_AOT_UNIMPLEMENTED();
  } else {  // kernel impl base on JIT
    ActivationGradBaseKernel<T, Context>(
        dev_ctx, "Out", out, dout, {}, dx, "exp_grad");
  }
}

template <typename T, typename Context>
void FloorKernel(const Context& dev_ctx,
                 const phi::DenseTensor& x,
                 phi::DenseTensor* out) {
  PADDLE_GCU_KERNEL_TRACE("floor");
  if (LaunchAOTKernel()) {
    dev_ctx.template Alloc<T>(out);
    LAUNCH_TOPSATENOP(topsatenFloor, dev_ctx, *out, x);

  } else {  // kernel impl base on JIT
    ActivationBaseKernel<T, Context>(dev_ctx, x, {}, out, "floor");
  }
}

template <typename T, typename Context>
void FloorGradKernel(const Context& dev_ctx,
                     const phi::DenseTensor& dout,
                     phi::DenseTensor* dx) {
  PADDLE_GCU_KERNEL_TRACE("floor_grad");
  if (LaunchAOTKernel()) {
    THROW_AOT_UNIMPLEMENTED();
  } else {  // kernel impl base on JIT
    dev_ctx.template Alloc<T>(dx);

    TensorNameMap input_names;
    input_names[GradVarName("Out")] = {"dout"};

    TensorValueMap inputs;
    inputs[GradVarName("Out")] = {const_cast<DenseTensor*>(&dout)};

    TensorNameMap output_names;
    output_names[GradVarName("X")] = {"dx"};

    TensorValueMap outputs;
    outputs[GradVarName("X")] = {dx};

    GcuRunner(
        input_names, inputs, output_names, outputs, {}, "floor_grad", dev_ctx);
  }
}

template <typename T, typename Context>
void CeilKernel(const Context& dev_ctx,
                const phi::DenseTensor& x,
                phi::DenseTensor* out) {
  PADDLE_GCU_KERNEL_TRACE("ceil");
  if (LaunchAOTKernel()) {
    dev_ctx.template Alloc<T>(out);
    LAUNCH_TOPSATENOP(topsatenCeil, dev_ctx, *out, x);

  } else {  // kernel impl base on JIT
    THROW_JIT_UNIMPLEMENTED();
  }
}

template <typename T, typename Context>
void SwishKernel(const Context& dev_ctx,
                 const phi::DenseTensor& x,
                 phi::DenseTensor* out) {
  PADDLE_GCU_KERNEL_TRACE("swish");
  if (LaunchAOTKernel()) {
    dev_ctx.template Alloc<T>(out);
    LAUNCH_TOPSATENOP(topsatenSilu, dev_ctx, *out, x);
  } else {  // kernel impl base on JIT
    ActivationBaseKernel<T, Context>(dev_ctx, x, {}, out, "swish");
  }
}

template <typename T, typename Context>
void SwishGradKernel(const Context& dev_ctx,
                     const phi::DenseTensor& x,
                     const phi::DenseTensor& dout,
                     phi::DenseTensor* dx) {
  PADDLE_GCU_KERNEL_TRACE("swish_grad");
  if (LaunchAOTKernel()) {
    THROW_AOT_UNIMPLEMENTED();
  } else {  // kernel impl base on JIT
    float beta = 1.0;
    GcuAttributeMap attrs;
    attrs["beta"] = beta;
    ActivationGradBaseKernel<T, Context>(
        dev_ctx, "X", x, dout, attrs, dx, "swish_grad");
  }
}

template <typename T, typename Context>
void ReluKernel(const Context& dev_ctx,
                const phi::DenseTensor& x,
                phi::DenseTensor* out) {
  PADDLE_GCU_KERNEL_TRACE("relu");
  if (LaunchAOTKernel()) {
    dev_ctx.template Alloc<T>(out);
    LAUNCH_TOPSATENOP(topsatenRelu, dev_ctx, *out, x);
  } else {  // kernel impl base on JIT
    ActivationBaseKernel<T, Context>(dev_ctx, x, {}, out, "relu");
  }
}

template <typename T, typename Context>
void ReluGradKernel(const Context& dev_ctx,
                    const phi::DenseTensor& out,
                    const phi::DenseTensor& dout,
                    phi::DenseTensor* dx) {
  PADDLE_GCU_KERNEL_TRACE("relu_grad");
  if (LaunchAOTKernel()) {
    THROW_AOT_UNIMPLEMENTED();
  } else {  // kernel impl base on JIT
    ActivationGradBaseKernel<T, Context>(
        dev_ctx, "Out", out, dout, {}, dx, "relu_grad");
  }
}

template <typename T, typename Context>
void Relu6Kernel(const Context& dev_ctx,
                 const phi::DenseTensor& x,
                 phi::DenseTensor* out) {
  PADDLE_GCU_KERNEL_TRACE("relu6");
  if (LaunchAOTKernel()) {
    dev_ctx.template Alloc<T>(out);
    LAUNCH_TOPSATENOP(topsatenRelu6, dev_ctx, *out, x);
  } else {  // kernel impl base on JIT
    ActivationBaseKernel<T, Context>(dev_ctx, x, {}, out, "relu6");
  }
}

template <typename T, typename Context>
void Relu6GradKernel(const Context& dev_ctx,
                     const phi::DenseTensor& out,
                     const phi::DenseTensor& dout,
                     phi::DenseTensor* dx) {
  PADDLE_GCU_KERNEL_TRACE("relu6_grad");
  if (LaunchAOTKernel()) {
    THROW_AOT_UNIMPLEMENTED();
  } else {  // kernel impl base on JIT
    ActivationGradBaseKernel<T, Context>(
        dev_ctx, "Out", out, dout, {}, dx, "relu6_grad");
  }
}

template <typename T, typename Context>
void LeakyReluKernel(const Context& dev_ctx,
                     const phi::DenseTensor& x,
                     float alpha,
                     phi::DenseTensor* out) {
  PADDLE_GCU_KERNEL_TRACE("leaky_relu");
  if (LaunchAOTKernel()) {
    dev_ctx.template Alloc<T>(out);
    phi::Scalar negative_slope(alpha);
    LAUNCH_TOPSATENOP(topsatenLeakyRelu, dev_ctx, *out, x, negative_slope);

  } else {  // kernel impl base on JIT
    GcuAttributeMap attrs;
    attrs["alpha"] = alpha;

    ActivationBaseKernel<T, Context>(dev_ctx, x, attrs, out, "leaky_relu");
  }
}

template <typename T, typename Context>
void LeakyReluGradKernel(const Context& dev_ctx,
                         const phi::DenseTensor& x,
                         const phi::DenseTensor& dout,
                         float alpha,
                         phi::DenseTensor* dx) {
  PADDLE_GCU_KERNEL_TRACE("leaky_relu_grad");
  if (LaunchAOTKernel()) {
    THROW_AOT_UNIMPLEMENTED();
  } else {  // kernel impl base on JIT
    GcuAttributeMap attrs;
    attrs["alpha"] = alpha;

    ActivationGradBaseKernel<T, Context>(
        dev_ctx, "X", x, dout, attrs, dx, "leaky_relu_grad");
  }
}

template <typename T, typename Context>
void GeluKernel(const Context& dev_ctx,
                const phi::DenseTensor& x,
                bool approximate,
                phi::DenseTensor* out) {
  PADDLE_GCU_KERNEL_TRACE("gelu");
  if (LaunchAOTKernel()) {
    const char* gelu_approximate = "none";
    if (approximate) {
      gelu_approximate = "tanh";
    }
    dev_ctx.template Alloc<T>(out);
    LAUNCH_TOPSATENOP(topsatenGelu, dev_ctx, *out, x, gelu_approximate);

  } else {  // kernel impl base on JIT
    GcuAttributeMap attrs;
    attrs["approximate"] = approximate;
    attrs["use_mkldnn"] = false;
    attrs["use_cudnn"] = false;

    ActivationBaseKernel<T, Context>(dev_ctx, x, attrs, out, "gelu");
  }
}

template <typename T, typename Context>
void GeluGradKernel(const Context& dev_ctx,
                    const phi::DenseTensor& x,
                    const phi::DenseTensor& dout,
                    bool approximate,
                    phi::DenseTensor* dx) {
  PADDLE_GCU_KERNEL_TRACE("gelu_grad");
  if (LaunchAOTKernel()) {
    THROW_AOT_UNIMPLEMENTED();
  } else {  // kernel impl base on JIT
    GcuAttributeMap attrs;
    attrs["approximate"] = approximate;
    attrs["use_mkldnn"] = false;
    attrs["use_cudnn"] = false;

    ActivationGradBaseKernel<T, Context>(
        dev_ctx, "X", x, dout, attrs, dx, "gelu_grad");
  }
}

template <typename T, typename Context>
void TanhKernel(const Context& dev_ctx,
                const phi::DenseTensor& x,
                phi::DenseTensor* out) {
  PADDLE_GCU_KERNEL_TRACE("tanh");
  if (LaunchAOTKernel()) {
    dev_ctx.template Alloc<T>(out);
    LAUNCH_TOPSATENOP(topsatenTanh, dev_ctx, *out, x);

  } else {  // kernel impl base on JIT
    ActivationBaseKernel<T, Context>(dev_ctx, x, {}, out, "tanh");
  }
}

template <typename T, typename Context>
void TanhGradKernel(const Context& dev_ctx,
                    const phi::DenseTensor& out,
                    const phi::DenseTensor& dout,
                    phi::DenseTensor* dx) {
  PADDLE_GCU_KERNEL_TRACE("tanh_grad");
  if (LaunchAOTKernel()) {
    THROW_AOT_UNIMPLEMENTED();
  } else {  // kernel impl base on JIT
    ActivationGradBaseKernel<T, Context>(
        dev_ctx, "Out", out, dout, {}, dx, "tanh_grad");
  }
}

template <typename T, typename Context>
void SigmoidKernel(const Context& dev_ctx,
                   const phi::DenseTensor& x,
                   phi::DenseTensor* out) {
  PADDLE_GCU_KERNEL_TRACE("sigmoid");
  if (LaunchAOTKernel()) {
    dev_ctx.template Alloc<T>(out);
    LAUNCH_TOPSATENOP(topsatenSigmoid, dev_ctx, *out, x);
  } else {  // kernel impl base on JIT
    ActivationBaseKernel<T, Context>(dev_ctx, x, {}, out, "sigmoid");
  }
}

template <typename T, typename Context>
void SigmoidGradKernel(const Context& dev_ctx,
                       const phi::DenseTensor& out,
                       const phi::DenseTensor& dout,
                       phi::DenseTensor* dx) {
  PADDLE_GCU_KERNEL_TRACE("sigmoid_grad");
  if (LaunchAOTKernel()) {
    THROW_AOT_UNIMPLEMENTED();
  } else {  // kernel impl base on JIT
    ActivationGradBaseKernel<T, Context>(
        dev_ctx, "Out", out, dout, {}, dx, "sigmoid_grad");
  }
}

template <typename T, typename Context>
void SqrtKernel(const Context& dev_ctx,
                const phi::DenseTensor& x,
                phi::DenseTensor* out) {
  PADDLE_GCU_KERNEL_TRACE("sqrt");
  if (LaunchAOTKernel()) {
    dev_ctx.template Alloc<T>(out);
    LAUNCH_TOPSATENOP(topsatenSqrt, dev_ctx, *out, x);

  } else {  // kernel impl base on JIT
    ActivationBaseKernel<T, Context>(dev_ctx, x, {}, out, "sqrt");
  }
}

template <typename T, typename Context>
void LogKernel(const Context& dev_ctx,
               const phi::DenseTensor& x,
               phi::DenseTensor* out) {
  PADDLE_GCU_KERNEL_TRACE("log");
  if (LaunchAOTKernel()) {
    dev_ctx.template Alloc<T>(out);
    LAUNCH_TOPSATENOP(topsatenLog, dev_ctx, *out, x);

  } else {  // kernel impl base on JIT
    ActivationBaseKernel<T, Context>(dev_ctx, x, {}, out, "log");
  }
}

template <typename T, typename Context>
void LogGradKernel(const Context& dev_ctx,
                   const phi::DenseTensor& x,
                   const phi::DenseTensor& dout,
                   phi::DenseTensor* dx) {
  PADDLE_GCU_KERNEL_TRACE("log_grad");
  if (LaunchAOTKernel()) {
    THROW_AOT_UNIMPLEMENTED();
  } else {  // kernel impl base on JIT
    ActivationGradBaseKernel<T, Context>(
        dev_ctx, "X", x, dout, {}, dx, "log_grad");
  }
}

template <typename T, typename Context>
void PowKernel(const Context& dev_ctx,
               const phi::DenseTensor& x,
               const phi::Scalar& factor_scalar,
               phi::DenseTensor* out) {
  PADDLE_GCU_KERNEL_TRACE("pow");
  if (LaunchAOTKernel()) {
    dev_ctx.template Alloc<T>(out);
    LAUNCH_TOPSATENOP(topsatenPow, dev_ctx, *out, x, factor_scalar);

  } else {  // kernel impl base on JIT
    GcuAttributeMap attrs;
    attrs["factor"] = factor_scalar.to<float>();
    ActivationBaseKernel<T, Context>(dev_ctx, x, attrs, out, "pow");
  }
}

template <typename T, typename Context>
void PowGradKernel(const Context& dev_ctx,
                   const phi::DenseTensor& x,
                   const phi::DenseTensor& dout,
                   const phi::Scalar& factor_scalar,
                   phi::DenseTensor* dx) {
  PADDLE_GCU_KERNEL_TRACE("pow_grad");
  if (LaunchAOTKernel()) {
    THROW_AOT_UNIMPLEMENTED();
  } else {  // kernel impl base on JIT
    GcuAttributeMap attrs;
    attrs["factor"] = factor_scalar.to<float>();
    ActivationGradBaseKernel<T, Context>(
        dev_ctx, "X", x, dout, attrs, dx, "pow_grad");
  }
}

template <typename T, typename Context>
void SquareKernel(const Context& dev_ctx,
                  const phi::DenseTensor& x,
                  phi::DenseTensor* out) {
  PADDLE_GCU_KERNEL_TRACE("square");
  if (LaunchAOTKernel()) {
    dev_ctx.template Alloc<T>(out);
    LAUNCH_TOPSATENOP(topsatenSquare, dev_ctx, *out, x);

  } else {  // kernel impl base on JIT
    ActivationBaseKernel<T, Context>(dev_ctx, x, {}, out, "square");
  }
}

template <typename T, typename Context>
void SquareGradKernel(const Context& dev_ctx,
                      const phi::DenseTensor& x,
                      const phi::DenseTensor& dout,
                      phi::DenseTensor* dx) {
  PADDLE_GCU_KERNEL_TRACE("square_grad");
  if (LaunchAOTKernel()) {
    THROW_AOT_UNIMPLEMENTED();
  } else {  // kernel impl base on JIT
    ActivationGradBaseKernel<T, Context>(
        dev_ctx, "X", x, dout, {}, dx, "square_grad");
  }
}

template <typename T, typename Context>
void Hard_SigmoidKernel(const Context& dev_ctx,
                        const phi::DenseTensor& x,
                        float slope,
                        float offset,
                        phi::DenseTensor* out) {
  PADDLE_GCU_KERNEL_TRACE("hard_sigmoid");
  if (LaunchAOTKernel()) {
    dev_ctx.template Alloc<T>(out);
    LAUNCH_TOPSATENOP(topsatenHardsigmoid, dev_ctx, *out, x);
  } else {  // kernel impl base on JIT
    GcuAttributeMap attrs;
    attrs["slope"] = slope;
    attrs["offset"] = offset;

    ActivationBaseKernel<T, Context>(dev_ctx, x, attrs, out, "hard_sigmoid");
  }
}

template <typename T, typename Context>
void HardSigmoidKernel(const Context& dev_ctx,
                       const phi::DenseTensor& x,
                       float slope,
                       float offset,
                       phi::DenseTensor* out) {
  PADDLE_GCU_KERNEL_TRACE("hardsigmoid");
  Hard_SigmoidKernel<T, Context>(dev_ctx, x, slope, offset, out);
}

template <typename T, typename Context>
void HardSigmoidGradKernel(const Context& dev_ctx,
                           const phi::DenseTensor& out,
                           const phi::DenseTensor& dout,
                           float slope,
                           float offset,
                           phi::DenseTensor* dx) {
  PADDLE_GCU_KERNEL_TRACE("hard_sigmoid_grad");
  if (LaunchAOTKernel()) {
    THROW_AOT_UNIMPLEMENTED();
  } else {  // kernel impl base on JIT
    GcuAttributeMap attrs;
    attrs["slope"] = slope;
    attrs["offset"] = offset;

    ActivationGradBaseKernel<T, Context>(
        dev_ctx, "Out", out, dout, attrs, dx, "hard_sigmoid_grad");
  }
}

template <typename T, typename Context>
void HardSwishKernel(const Context& dev_ctx,
                     const phi::DenseTensor& x,
                     phi::DenseTensor* out) {
  PADDLE_GCU_KERNEL_TRACE("hard_swish");
  if (LaunchAOTKernel()) {
    dev_ctx.template Alloc<T>(out);
    LAUNCH_TOPSATENOP(topsatenHardswish, dev_ctx, *out, x);
  } else {  // kernel impl base on JIT
    ActivationBaseKernel<T, Context>(dev_ctx, x, {}, out, "hard_swish");
  }
}

template <typename T, typename Context>
void HardSwishGradKernel(const Context& dev_ctx,
                         const phi::DenseTensor& x,
                         const phi::DenseTensor& dout,
                         phi::DenseTensor* dx) {
  PADDLE_GCU_KERNEL_TRACE("hard_swish_grad");
  if (LaunchAOTKernel()) {
    THROW_AOT_UNIMPLEMENTED();
  } else {  // kernel impl base on JIT
    float offset = 3;
    float threshold = 6;
    float scale = 6;
    GcuAttributeMap attrs;
    attrs["offset"] = offset;
    attrs["threshold"] = threshold;
    attrs["scale"] = scale;
    ActivationGradBaseKernel<T, Context>(
        dev_ctx, "X", x, dout, attrs, dx, "hard_swish_grad");
  }
}

template <typename T, typename Context>
void LogitKernel(const Context& dev_ctx,
                 const phi::DenseTensor& x,
                 float eps,
                 phi::DenseTensor* out) {
  PADDLE_GCU_KERNEL_TRACE("logit");
  if (LaunchAOTKernel()) {
    dev_ctx.template Alloc<T>(out);
    phi::Scalar epsilon(eps);
    LAUNCH_TOPSATENOP(topsatenLogit, dev_ctx, *out, x, epsilon);

  } else {  // kernel impl base on JIT
    THROW_JIT_UNIMPLEMENTED();
  }
}

template <typename T, typename Context>
void CeluKernel(const Context& dev_ctx,
                const phi::DenseTensor& x,
                float alpha,
                phi::DenseTensor* out) {
  PADDLE_GCU_KERNEL_TRACE("celu");
  if (LaunchAOTKernel()) {
    dev_ctx.template Alloc<T>(out);
    phi::Scalar a(alpha);
    LAUNCH_TOPSATENOP(topsatenCelu, dev_ctx, *out, x, a);

  } else {  // kernel impl base on JIT
    THROW_JIT_UNIMPLEMENTED();
  }
}

template <typename T, typename Context>
void HardShrinkKernel(const Context& dev_ctx,
                      const phi::DenseTensor& x,
                      float threshold,
                      phi::DenseTensor* out) {
  PADDLE_GCU_KERNEL_TRACE("hard_shrink");
  if (LaunchAOTKernel()) {
    dev_ctx.template Alloc<T>(out);
    phi::Scalar lambd(threshold);
    LAUNCH_TOPSATENOP(topsatenHardshrink, dev_ctx, *out, x, lambd);

  } else {  // kernel impl base on JIT
    THROW_JIT_UNIMPLEMENTED();
  }
}

template <typename T, typename Context>
void SoftShrinkKernel(const Context& dev_ctx,
                      const phi::DenseTensor& x,
                      float lambda,
                      phi::DenseTensor* out) {
  PADDLE_GCU_KERNEL_TRACE("softshrink");
  if (LaunchAOTKernel()) {
    dev_ctx.template Alloc<T>(out);
    phi::Scalar lambd(lambda);
    LAUNCH_TOPSATENOP(topsatenSoftshrink, dev_ctx, *out, x, lambd);

  } else {  // kernel impl base on JIT
    THROW_JIT_UNIMPLEMENTED();
  }
}

template <typename T, typename Context>
void SoftplusKernel(const Context& dev_ctx,
                    const phi::DenseTensor& x,
                    float beta,
                    float threshold,
                    phi::DenseTensor* out) {
  PADDLE_GCU_KERNEL_TRACE("softplus");
  if (LaunchAOTKernel()) {
    dev_ctx.template Alloc<T>(out);
    phi::Scalar beta_s(beta);
    phi::Scalar threshold_s(threshold);
    LAUNCH_TOPSATENOP(topsatenSoftplus, dev_ctx, *out, x, beta_s, threshold_s);

  } else {  // kernel impl base on JIT
    THROW_JIT_UNIMPLEMENTED();
  }
}

template <typename T, typename Context>
void HardtanhKernel(const Context& dev_ctx,
                    const phi::DenseTensor& x,
                    float t_min,
                    float t_max,
                    phi::DenseTensor* out) {
  PADDLE_GCU_KERNEL_TRACE("hardtanh");
  if (LaunchAOTKernel()) {
    dev_ctx.template Alloc<T>(out);
    phi::Scalar min_val(t_min);
    phi::Scalar max_val(t_max);
    LAUNCH_TOPSATENOP(topsatenHardtanh, dev_ctx, *out, x, min_val, max_val);

  } else {  // kernel impl base on JIT
    THROW_JIT_UNIMPLEMENTED();
  }
}

template <typename T, typename Context>
void EluKernel(const Context& dev_ctx,
               const phi::DenseTensor& x,
               float alpha,
               phi::DenseTensor* out) {
  PADDLE_GCU_KERNEL_TRACE("elu");
  if (LaunchAOTKernel()) {
    dev_ctx.template Alloc<T>(out);
    phi::Scalar alpha_s(alpha);
    phi::Scalar scale_s(1.0f);
    phi::Scalar in_scale_s(1.0f);
    LAUNCH_TOPSATENOP(
        topsatenElu, dev_ctx, *out, x, alpha_s, scale_s, in_scale_s);

  } else {  // kernel impl base on JIT
    THROW_JIT_UNIMPLEMENTED();
  }
}

template <typename T, typename Context>
void RoundKernel(const Context& dev_ctx,
                 const phi::DenseTensor& x,
                 const int decimals,
                 phi::DenseTensor* out) {
  PADDLE_GCU_KERNEL_TRACE("round");
  if (LaunchAOTKernel()) {
    dev_ctx.template Alloc<T>(out);
    LAUNCH_TOPSATENOP(topsatenRound, dev_ctx, *out, x, decimals);

  } else {  // kernel impl base on JIT
    THROW_JIT_UNIMPLEMENTED();
  }
}

DEFINE_UNARY_AOT_ACTIVATION_KERNEL(logsigmoid, LogSigmoid)
DEFINE_UNARY_AOT_ACTIVATION_KERNEL(rsqrt, Rsqrt)
DEFINE_UNARY_AOT_ACTIVATION_KERNEL(log2, Log2)
DEFINE_UNARY_AOT_ACTIVATION_KERNEL(log10, Log10)
DEFINE_UNARY_AOT_ACTIVATION_KERNEL(log1p, Log1p)
DEFINE_UNARY_AOT_ACTIVATION_KERNEL(silu, Silu)
DEFINE_UNARY_AOT_ACTIVATION_KERNEL(reciprocal, Reciprocal)
DEFINE_UNARY_AOT_ACTIVATION_KERNEL(acos, Acos)
DEFINE_UNARY_AOT_ACTIVATION_KERNEL(acosh, Acosh)
DEFINE_UNARY_AOT_ACTIVATION_KERNEL(asin, Asin)
DEFINE_UNARY_AOT_ACTIVATION_KERNEL(asinh, Asinh)
DEFINE_UNARY_AOT_ACTIVATION_KERNEL(atanh, Atanh)
DEFINE_UNARY_AOT_ACTIVATION_KERNEL(cosh, Cosh)
DEFINE_UNARY_AOT_ACTIVATION_KERNEL(sinh, Sinh)
DEFINE_UNARY_AOT_ACTIVATION_KERNEL(tan, Tan)
DEFINE_UNARY_AOT_ACTIVATION_KERNEL(erf, Erf)
DEFINE_UNARY_AOT_ACTIVATION_KERNEL(expm1, Expm1)

#undef DEFINE_UNARY_AOT_ACTIVATION_KERNEL

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(abs,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::AbsKernel,
                          float,
                          double,
                          int64_t,
                          int32_t,
                          phi::dtype::float16) {
  kernel->InputAt(0).SetDataType(phi::dtype::ToReal(kernel_key.dtype()));
}

PD_REGISTER_PLUGIN_KERNEL(abs_grad,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::AbsGradKernel,
                          float,
                          double,
                          int64_t,
                          int32_t) {
  kernel->InputAt(1).SetDataType(phi::dtype::ToReal(kernel_key.dtype()));
}

#define PD_REGISTER_PLUGIN_GCU_ACTIVATION_KERNEL(name, func) \
  PD_REGISTER_PLUGIN_KERNEL(name,                            \
                            gcu,                             \
                            ALL_LAYOUT,                      \
                            custom_kernel::func,             \
                            float,                           \
                            phi::dtype::float16,             \
                            double) {}

PD_REGISTER_PLUGIN_GCU_ACTIVATION_KERNEL(cos, CosKernel)
PD_REGISTER_PLUGIN_GCU_ACTIVATION_KERNEL(cos_grad, CosGradKernel)
PD_REGISTER_PLUGIN_GCU_ACTIVATION_KERNEL(sin, SinKernel)
PD_REGISTER_PLUGIN_GCU_ACTIVATION_KERNEL(atan, AtanKernel)
PD_REGISTER_PLUGIN_GCU_ACTIVATION_KERNEL(atan_grad, AtanGradKernel)
PD_REGISTER_PLUGIN_GCU_ACTIVATION_KERNEL(exp, ExpKernel)
PD_REGISTER_PLUGIN_GCU_ACTIVATION_KERNEL(exp_grad, ExpGradKernel)
PD_REGISTER_PLUGIN_GCU_ACTIVATION_KERNEL(swish, SwishKernel)
PD_REGISTER_PLUGIN_GCU_ACTIVATION_KERNEL(swish_grad, SwishGradKernel)
PD_REGISTER_PLUGIN_GCU_ACTIVATION_KERNEL(relu, ReluKernel)
PD_REGISTER_PLUGIN_GCU_ACTIVATION_KERNEL(relu_grad, ReluGradKernel)
PD_REGISTER_PLUGIN_GCU_ACTIVATION_KERNEL(relu6, Relu6Kernel)
PD_REGISTER_PLUGIN_GCU_ACTIVATION_KERNEL(relu6_grad, Relu6GradKernel)
PD_REGISTER_PLUGIN_GCU_ACTIVATION_KERNEL(leaky_relu, LeakyReluKernel)
PD_REGISTER_PLUGIN_GCU_ACTIVATION_KERNEL(leaky_relu_grad, LeakyReluGradKernel)
PD_REGISTER_PLUGIN_GCU_ACTIVATION_KERNEL(pow, PowKernel)
PD_REGISTER_PLUGIN_GCU_ACTIVATION_KERNEL(pow_grad, PowGradKernel)
PD_REGISTER_PLUGIN_GCU_ACTIVATION_KERNEL(log, LogKernel)
PD_REGISTER_PLUGIN_GCU_ACTIVATION_KERNEL(log2, Log2Kernel)
PD_REGISTER_PLUGIN_GCU_ACTIVATION_KERNEL(log10, Log10Kernel)
PD_REGISTER_PLUGIN_GCU_ACTIVATION_KERNEL(log1p, Log1pKernel)
PD_REGISTER_PLUGIN_GCU_ACTIVATION_KERNEL(floor, FloorKernel)
PD_REGISTER_PLUGIN_GCU_ACTIVATION_KERNEL(floor_grad, FloorGradKernel)
PD_REGISTER_PLUGIN_GCU_ACTIVATION_KERNEL(ceil, CeilKernel)
PD_REGISTER_PLUGIN_GCU_ACTIVATION_KERNEL(gelu, GeluKernel)
PD_REGISTER_PLUGIN_GCU_ACTIVATION_KERNEL(gelu_grad, GeluGradKernel)
PD_REGISTER_PLUGIN_GCU_ACTIVATION_KERNEL(tanh, TanhKernel)
PD_REGISTER_PLUGIN_GCU_ACTIVATION_KERNEL(tanh_grad, TanhGradKernel)
PD_REGISTER_PLUGIN_GCU_ACTIVATION_KERNEL(sigmoid, SigmoidKernel)
PD_REGISTER_PLUGIN_GCU_ACTIVATION_KERNEL(sigmoid_grad, SigmoidGradKernel)
PD_REGISTER_PLUGIN_GCU_ACTIVATION_KERNEL(logsigmoid, LogSigmoidKernel)
PD_REGISTER_PLUGIN_GCU_ACTIVATION_KERNEL(sqrt, SqrtKernel)
PD_REGISTER_PLUGIN_GCU_ACTIVATION_KERNEL(rsqrt, RsqrtKernel)
PD_REGISTER_PLUGIN_GCU_ACTIVATION_KERNEL(square, SquareKernel)
PD_REGISTER_PLUGIN_GCU_ACTIVATION_KERNEL(square_grad, SquareGradKernel)
PD_REGISTER_PLUGIN_GCU_ACTIVATION_KERNEL(hard_sigmoid, Hard_SigmoidKernel)
PD_REGISTER_PLUGIN_GCU_ACTIVATION_KERNEL(hard_sigmoid_grad,
                                         HardSigmoidGradKernel)
PD_REGISTER_PLUGIN_GCU_ACTIVATION_KERNEL(hardsigmoid, HardSigmoidKernel)
PD_REGISTER_PLUGIN_GCU_ACTIVATION_KERNEL(hardswish, HardSwishKernel)
PD_REGISTER_PLUGIN_GCU_ACTIVATION_KERNEL(hardswish_grad, HardSwishGradKernel)
PD_REGISTER_PLUGIN_GCU_ACTIVATION_KERNEL(silu, SiluKernel)
PD_REGISTER_PLUGIN_GCU_ACTIVATION_KERNEL(reciprocal, ReciprocalKernel)
PD_REGISTER_PLUGIN_GCU_ACTIVATION_KERNEL(logit, LogitKernel)
PD_REGISTER_PLUGIN_GCU_ACTIVATION_KERNEL(celu, CeluKernel)
PD_REGISTER_PLUGIN_GCU_ACTIVATION_KERNEL(hard_shrink, HardShrinkKernel)
PD_REGISTER_PLUGIN_GCU_ACTIVATION_KERNEL(softshrink, SoftShrinkKernel)
PD_REGISTER_PLUGIN_GCU_ACTIVATION_KERNEL(softplus, SoftplusKernel)
PD_REGISTER_PLUGIN_GCU_ACTIVATION_KERNEL(acos, AcosKernel)
PD_REGISTER_PLUGIN_GCU_ACTIVATION_KERNEL(acosh, AcoshKernel)
PD_REGISTER_PLUGIN_GCU_ACTIVATION_KERNEL(asin, AsinKernel)
PD_REGISTER_PLUGIN_GCU_ACTIVATION_KERNEL(asinh, AsinhKernel)
PD_REGISTER_PLUGIN_GCU_ACTIVATION_KERNEL(atanh, AtanhKernel)
PD_REGISTER_PLUGIN_GCU_ACTIVATION_KERNEL(cosh, CoshKernel)
PD_REGISTER_PLUGIN_GCU_ACTIVATION_KERNEL(sinh, SinhKernel)
PD_REGISTER_PLUGIN_GCU_ACTIVATION_KERNEL(tan, TanKernel)
PD_REGISTER_PLUGIN_GCU_ACTIVATION_KERNEL(round, RoundKernel)
PD_REGISTER_PLUGIN_GCU_ACTIVATION_KERNEL(elu, EluKernel)
PD_REGISTER_PLUGIN_GCU_ACTIVATION_KERNEL(erf, ErfKernel)
PD_REGISTER_PLUGIN_GCU_ACTIVATION_KERNEL(expm1, Expm1Kernel)
PD_REGISTER_PLUGIN_GCU_ACTIVATION_KERNEL(hardtanh, HardtanhKernel)

#undef PD_REGISTER_PLUGIN_GCU_ACTIVATION_KERNEL
