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
void AclopNormKernel(const Context& dev_ctx,
                     const phi::DenseTensor& x,
                     phi::DenseTensor* x_norm) {
  auto stream = dev_ctx.stream();

  PADDLE_ENFORCE_NOT_NULL(&x,
                          phi::errors::InvalidArgument(
                              "Input(X) of ClipByNormOp should not be null. "
                              "Please check if it is created correctly."));
  phi::DenseTensor square_sum;
  phi::DenseTensorMeta square_sum_meta = {x.dtype(), phi::DDim({1})};
  square_sum.set_meta(square_sum_meta);
  dev_ctx.template Alloc<T>(&square_sum);
  const auto& x_dims = x.dims();
  std::vector<int> axis;
  for (int i = 0; i < x_dims.size(); ++i) {
    axis.push_back(i);
  }
  const auto& square_sum_runner = NpuOpRunner(
      "SquareSumV1", {x}, {square_sum}, {{"axis", axis}, {"keep_dims", false}});
  square_sum_runner.Run(stream);

  // sqrt
  dev_ctx.template Alloc<T>(x_norm);
  const auto& x_norm_runner = NpuOpRunner("Sqrt", {square_sum}, {*x_norm}, {});
  x_norm_runner.Run(stream);
}

template <typename T, typename Context>
void NormKernel(const Context& dev_ctx,
                const phi::DenseTensor& x,
                phi::DenseTensor* x_norm) {
  DO_COMPATIBILITY(
      aclnnNorm,
      (custom_kernel::AclopNormKernel<T, Context>(dev_ctx, x, x_norm)));

  dev_ctx.template Alloc<T>(x_norm);

  phi::Scalar p = 2.0f;
  const auto& x_dims = x.dims();
  std::vector<int64_t> axis, resize_list;
  for (int64_t i = 0; i < x_dims.size(); ++i) {
    axis.push_back(i);
    resize_list.push_back(1);
  }
  bool keepdim = true;
  x_norm->Resize(phi::make_ddim(resize_list));
  EXEC_NPU_CMD(aclnnNorm, dev_ctx, x, p, axis, keepdim, *x_norm);
  x_norm->Resize({1});
}

template <typename T, typename Context>
void AclopMulsKernel(const Context& dev_ctx,
                     const phi::DenseTensor& x,
                     const float scaling,
                     phi::DenseTensor* out) {
  auto stream = dev_ctx.stream();
  dev_ctx.template Alloc<T>(out);

  const auto& muls_runner =
      NpuOpRunner("Muls", {x}, {*out}, {{"value", scaling}});
  muls_runner.Run(stream);
}

template <typename T, typename Context>
void MulsKernel(const Context& dev_ctx,
                const phi::DenseTensor& x,
                const float scaling,
                phi::DenseTensor* out) {
  DO_COMPATIBILITY(
      aclnnMuls,
      (custom_kernel::AclopMulsKernel<T, Context>(dev_ctx, x, scaling, out)));
  dev_ctx.template Alloc<T>(out);
  phi::Scalar scaling_ = scaling;
  EXEC_NPU_CMD(aclnnMuls, dev_ctx, x, scaling_, *out);
}

template <typename T, typename Context>
void ClipByNormKernel(const Context& dev_ctx,
                      const phi::DenseTensor& x,
                      float max_norm,
                      phi::DenseTensor* out) {
  PADDLE_ENFORCE_NOT_NULL(&x,
                          phi::errors::InvalidArgument(
                              "Input(X) of ClipByNormOp should not be null. "
                              "Please check if it is created correctly."));

  phi::DenseTensor x_norm;
  phi::DenseTensorMeta x_norm_meta = {x.dtype(), phi::DDim({1})};
  x_norm.set_meta(x_norm_meta);
  NormKernel<T, Context>(dev_ctx, x, &x_norm);
  dev_ctx.Wait();

  phi::DenseTensor x_norm_t;
  phi::DenseTensorMeta x_norm_t_meta = {
      x_norm.dtype(), x_norm.dims(), x_norm.layout()};
  x_norm_t.set_meta(x_norm_t_meta);
  // sync copy
  TensorCopy(dev_ctx, x_norm, true, &x_norm_t, phi::CPUPlace());
  auto x_norm_v = static_cast<float>(*(x_norm_t.data<T>()));

  if (x_norm_v <= max_norm) {
    TensorCopy(dev_ctx, x, false, out);
  } else {
    auto epsilon = x_norm_v <= static_cast<float>(1e-30)
                       ? static_cast<float>(1e-6)
                       : static_cast<float>(0);
    float scaling = max_norm / (x_norm_v + epsilon);
    MulsKernel<T, Context>(dev_ctx, x, scaling, out);
  }
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(clip_by_norm,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::ClipByNormKernel,
                          float,
                          phi::dtype::float16) {}
