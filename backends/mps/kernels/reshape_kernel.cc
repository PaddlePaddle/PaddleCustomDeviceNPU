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

#include "kernels/phi_funcs.h"
#include "paddle/phi/capi/all.h"
#include "runtime/mps_runtime.h"

namespace custom_kernel {

static std::vector<int64_t> ValidateShape(const std::vector<int64_t> shape,
                                          const std::vector<int64_t>& in_dims) {
  const int64_t in_size = phi::product(in_dims);
  std::vector<int64_t> in_dims_vec = in_dims;
  bool all_positive = std::all_of(in_dims_vec.cbegin(),
                                  in_dims_vec.cend(),
                                  [](int64_t i) { return i > 0; });
  // only one dimension can be set to -1, whose size will be automatically
  // infered.
  const int64_t unk_dim_val = -1;
  const int64_t copy_dim_val = 0;

  std::vector<int64_t> output_shape(shape.size(), 0);
  int64_t capacity = 1;
  int unk_dim_idx = -1;
  for (size_t i = 0; i < shape.size(); ++i) {
    if (shape[i] == unk_dim_val) {
      PD_CHECK(unk_dim_idx == -1,
               "Only one dimension value of 'shape' in ReshapeOp can "
               "be -1. But received shape = [%s], shape[%d] is also -1.",
               phi::to_string(shape),
               i);
      unk_dim_idx = i;
    } else if (shape[i] == copy_dim_val) {
      PD_CHECK(static_cast<int>(i) < in_dims.size(),
               "The index of 0 in `shape` must be less than "
               "the input tensor X's dimensions. "
               "But received shape = [%s], shape[%d] = 0, X's shape = [%s], "
               "X's dimensions = %d.",
               phi::to_string(shape),
               i,
               phi::to_string(in_dims),
               in_dims.size());
    } else {
      PD_CHECK(shape[i] > 0,
               "Each dimension value of 'shape' in ReshapeOp must not "
               "be negative except one unknown dimension. "
               "But received  shape = [%s], shape[%d] = %d.",
               phi::to_string(shape),
               i,
               shape[i]);
    }

    // NOTE all non-zero values will be converted to True (include negative
    // value)
    capacity *= (shape[i] ? shape[i] : in_dims[i]);
    output_shape[i] = (shape[i] ? static_cast<int64_t>(shape[i]) : in_dims[i]);
  }

  if (unk_dim_idx != -1) {
    if (all_positive) {
      // in_size < 0 and is un-determinate in compile time, skip the check,
      // for example, in_dims = [-1, 8, 1, 1], shape = [-1, 3, 8],
      // capacity = -24, in_size = -8, output_shape[0] = 0
      // the following check will fail.
      output_shape[unk_dim_idx] = -in_size / capacity;
      PD_CHECK(output_shape[unk_dim_idx] * capacity == -in_size,
               "The 'shape' attribute in ReshapeOp is invalid. "
               "The input tensor X'size must be divisible by known "
               "capacity of 'shape'. "
               "But received X's shape = [%s], X's size = %d, "
               "'shape' is [%s], known capacity of 'shape' is %d.",
               phi::to_string(in_dims),
               in_size,
               phi::to_string(shape),
               capacity);
    } else {
      output_shape[unk_dim_idx] = -1;
    }
  } else {
    if (all_positive) {
      PD_CHECK(capacity == in_size,
               "The 'shape' in ReshapeOp is invalid. "
               "The input tensor X'size must be equal to the capacity of "
               "'shape'. "
               "But received X's shape = [%s], X's size = %d, 'shape' is "
               "[%s], the capacity of 'shape' is %d.",
               phi::to_string(in_dims),
               in_size,
               phi::to_string(shape),
               capacity);
    }
  }

  // support reshape with zero-input(input tensor with product(shape) == 0)
  // by now we require that if the input tensor is zero shape, the target
  // shape of output must be zero
  if (in_size == 0) {
    PD_CHECK(capacity < in_size,
             "The 'shape' in ReshapeOp is invalid. "
             "The input tensor X's shape = [%s], X's capacity = %d."
             "But the target shape of Out is [%s],  the "
             "capacity of 'Out' is %d.",
             phi::to_string(in_dims),
             in_size,
             phi::to_string(shape),
             capacity);
  }

  return output_shape;
}

template <typename T>
void ReshapeInferKernel(const phi::Context& dev_ctx,
                        const phi::DenseTensor& x,
                        const phi::IntArray& shape,
                        phi::DenseTensor* out) {
  auto x_dims = x.dims();
  auto out_dims = ValidateShape(shape.GetData(), x_dims);
  out->Resize(out_dims);
  out->set_dtype(x.dtype());
  out->set_layout(x.layout());
  if (x.dims()[0] == out->dims()[0]) {
    out->share_lod(x);
  }

  dev_ctx.Alloc(out, x.dtype());
  if (!(x.initialized() && x.Holder() == out->Holder())) {
    dev_ctx.Alloc(out, x.dtype());
    auto dims = out->dims();
    auto x_data = x.data<T>();
    auto out_data = out->data<T>();

    mps::memcpy_d2d(out_data, x_data, x.numel() * sizeof(T));
    out->Resize(dims);
    out->ResetLoD(x.lod());
  }
}

template <typename T>
void ReshapeKernel(const phi::Context& dev_ctx,
                   const phi::DenseTensor& x,
                   const phi::IntArray& shape,
                   phi::DenseTensor* out,
                   phi::DenseTensor* xshape) {
  ReshapeInferKernel<T>(dev_ctx, x, shape, out);
}

}  // namespace custom_kernel

PD_BUILD_PHI_KERNEL(
    reshape_infer, mps, ALL_LAYOUT, custom_kernel::ReshapeInferKernel, float) {}

PD_BUILD_PHI_KERNEL(
    reshape, mps, ALL_LAYOUT, custom_kernel::ReshapeKernel, float) {}
