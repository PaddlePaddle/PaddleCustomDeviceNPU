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

#include <vector>

#include "kernels/phi_funcs.h"
#include "paddle/phi/capi/all.h"
#include "runtime/mps_runtime.h"

namespace custom_kernel {

std::vector<int64_t> GetOutputShape(const std::vector<int> squeeze_dims,
                                    const std::vector<int64_t>& in_dims,
                                    bool is_runtime) {
  size_t num_squeeze_dims = squeeze_dims.size();
  std::vector<bool> should_squeeze(in_dims.size(), false);

  // Mark dimensions need to be squeezed.
  if (num_squeeze_dims == 0) {
    for (int i = 0; i < in_dims.size(); ++i) {
      if (in_dims[i] == 1) {
        should_squeeze[i] = true;
      }
    }
  } else {
    for (size_t i = 0; i < num_squeeze_dims; ++i) {
      int current = squeeze_dims[i] < 0 ? squeeze_dims[i] + in_dims.size()
                                        : squeeze_dims[i];

      if (!should_squeeze[current]) {
        if (is_runtime) {
          // At run time, dim of 1 is allowed to squeeze
          if (in_dims[current] == 1) {
            should_squeeze[current] = true;
          }
        } else {
          // At compile time, dim of -1 or 1 is allowed to squeeze
          if (in_dims[current] == 1 || in_dims[current] == -1) {
            should_squeeze[current] = true;
          }
        }
      }
    }
  }
  // Make output dimensions
  std::vector<int64_t> output_shape;
  for (int i = 0; i < in_dims.size(); ++i) {
    if (!should_squeeze[i]) {
      output_shape.push_back(in_dims[i]);
    }
  }
  return output_shape;
}

template <typename T>
void SqueezeKernel(const phi::Context& dev_ctx,
                   const phi::DenseTensor& x,
                   const phi::IntArray& axes_int_array,
                   phi::DenseTensor* out) {
  auto stream = dev_ctx.stream();
  std::vector<int32_t> axes(axes_int_array.GetData().begin(),
                            axes_int_array.GetData().end());

  auto x_dims = x.dims();
  auto out_dims = custom_kernel::GetOutputShape(axes, x_dims, true);

  auto x_data = x.data<T>();
  auto out_data = out->data<T>();
  mps::memcpy_d2d(out_data, x_data, x.numel() * sizeof(T));

  out->Resize(out_dims);
}

template <typename T>
void SqueezeWithXShapeKernel(const phi::Context& dev_ctx,
                             const phi::DenseTensor& x,
                             const phi::IntArray& axes_int_array,
                             phi::DenseTensor* out,
                             phi::DenseTensor* xshape) {
  custom_kernel::SqueezeKernel<T>(dev_ctx, x, axes_int_array, out);
}

}  // namespace custom_kernel

PD_BUILD_PHI_KERNEL(
    squeeze, mps, ALL_LAYOUT, custom_kernel::SqueezeKernel, float) {}

PD_BUILD_PHI_KERNEL(squeeze_with_xshape,
                    mps,
                    ALL_LAYOUT,
                    custom_kernel::SqueezeWithXShapeKernel,
                    float) {}
