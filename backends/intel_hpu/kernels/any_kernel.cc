// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not
// use this file except in compliance with the License.
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
void AbsKernel(const Context& dev_ctx,
               const phi::DenseTensor& x,
               phi::DenseTensor* out);

template <typename T, typename Context>
void SumKernel(const Context& dev_ctx,
               const phi::DenseTensor& x,
               const phi::IntArray& dims,
               phi::DataType out_dtype,
               bool keep_dim,
               phi::DenseTensor* out);

template <typename T, typename Context>
void NotEqualKernel(const Context& dev_ctx,
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
void AnyKernel(const Context& dev_ctx,
               const phi::DenseTensor& x,
               const std::vector<int64_t>& dims,
               bool keep_dim,
               phi::DenseTensor* out) {
  phi::DenseTensor x_f32;
  phi::DenseTensorMeta x_f32_meta({phi::DataType::FLOAT32, x.dims()});
  x_f32.set_meta(x_f32_meta);
  custom_kernel::CastKernel<T, Context>(
      dev_ctx, x, phi::DataType::FLOAT32, &x_f32);

  phi::DenseTensor abs_out;
  phi::DenseTensorMeta abs_meta({x_f32.dtype(), x_f32.dims()});
  abs_out.set_meta(abs_meta);
  custom_kernel::AbsKernel<float, Context>(dev_ctx, x_f32, &abs_out);

  phi::DenseTensor sum_out;
  auto new_shape = x_f32.dims();
  for (size_t i = 0; i < dims.size(); i++) {
    auto d = CanonicalAxis(dims[i], x_f32.dims().size());
    new_shape[d] = 1;
  }
  phi::DenseTensorMeta sum_meta({x_f32.dtype(), new_shape});
  sum_out.set_meta(sum_meta);
  custom_kernel::SumKernel<float, Context>(
      dev_ctx, abs_out, dims, x_f32.dtype(), true, &sum_out);

  phi::DenseTensor zero;
  phi::DenseTensorMeta zero_meta({x_f32.dtype(), {1}});
  zero.set_meta(zero_meta);
  std::vector<int64_t> shape_vec = {1};
  phi::IntArray scalar_shape(shape_vec);
  custom_kernel::FullKernel<float, Context>(
      dev_ctx, scalar_shape, phi::Scalar(0.), x_f32.dtype(), &zero);

  custom_kernel::NotEqualKernel<float, Context>(dev_ctx, sum_out, zero, out);
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(
    any, intel_hpu, ALL_LAYOUT, custom_kernel::AnyKernel, bool) {}
