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

#include "kernels/funcs.h"
#include "kernels/hpu_operator.h"

namespace custom_kernel {

template <typename T, typename Context>
void AddKernel(const Context& dev_ctx,
               const phi::DenseTensor& x,
               const phi::DenseTensor& y,
               phi::DenseTensor* out);

template <typename T, typename Context>
void MultKernel(const Context& dev_ctx,
                const phi::DenseTensor& x,
                const phi::DenseTensor& y,
                phi::DenseTensor* out);

template <typename T, typename Context>
void FullKernel(const Context& dev_ctx,
                const phi::IntArray& shape,
                const phi::Scalar& val,
                phi::DataType dtype,
                phi::DenseTensor* out);

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
                 phi::DenseTensor* out) {
  phi::DenseTensor scale_tensor;
  phi::DenseTensorMeta tensor_meta({phi::DataType::FLOAT32, {1}});
  scale_tensor.set_meta(tensor_meta);
  std::vector<int64_t> shape_vec = {1};
  phi::IntArray scalar_shape(shape_vec);
  custom_kernel::FullKernel<float, Context>(
      dev_ctx, scalar_shape, in_scale, phi::DataType::FLOAT32, &scale_tensor);

  phi::DenseTensor x_f32;
  phi::DenseTensorMeta x_f32_meta({phi::DataType::FLOAT32, x.dims()});
  x_f32.set_meta(x_f32_meta);
  custom_kernel::CastKernel<T, Context>(
      dev_ctx, x, phi::DataType::FLOAT32, &x_f32);

  phi::DenseTensor mul_out;
  phi::DenseTensorMeta out_meta({x_f32.dtype(), x_f32.dims()});
  mul_out.set_meta(out_meta);
  custom_kernel::MultKernel<float, Context>(
      dev_ctx, x_f32, scale_tensor, &mul_out);

  phi::DenseTensor bias_tensor;
  bias_tensor.set_meta(tensor_meta);
  auto scale = in_scale.to<float>();
  auto bias = in_bias.to<float>();
  if (!bias_after_scale) {
    bias = bias * scale;
  }
  auto bias_scalar = phi::Scalar(bias);
  custom_kernel::FullKernel<float, Context>(
      dev_ctx, scalar_shape, bias_scalar, phi::DataType::FLOAT32, &bias_tensor);

  phi::DenseTensor add_out;
  add_out.set_meta(out_meta);
  custom_kernel::AddKernel<float, Context>(
      dev_ctx, mul_out, bias_tensor, &add_out);

  custom_kernel::CastKernel<float, Context>(dev_ctx, add_out, x.dtype(), out);
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(scale,
                          intel_hpu,
                          ALL_LAYOUT,
                          custom_kernel::ScaleKernel,
                          float,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}
