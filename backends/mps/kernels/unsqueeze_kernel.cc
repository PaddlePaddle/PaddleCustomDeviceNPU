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

inline std::vector<int64_t> GetUnsqueezeShape(
    const std::vector<int> unsqz_dims, const std::vector<int64_t>& in_dims) {
  int output_size = in_dims.size() + static_cast<int>(unsqz_dims.size());
  int cur_output_size = in_dims.size();
  std::vector<int64_t> output_shape(output_size, 0);

  for (int axis : unsqz_dims) {
    int cur = axis < 0 ? axis + cur_output_size + 1 : axis;
    // Move old axis, and insert new axis
    for (int i = cur_output_size; i >= cur; --i) {
      if (output_shape[i] == 1) {
        // Move axis
        output_shape[i + 1] = 1;
        output_shape[i] = 0;
      }
    }
    output_shape[cur] = 1;
    // Add the output size.
    cur_output_size++;
  }

  // Make output shape
  for (int in_idx = 0, out_idx = 0; out_idx < output_size; ++out_idx) {
    if (output_shape[out_idx] == 0) {
      output_shape[out_idx] = in_dims[in_idx++];
    }
  }

  return output_shape;
}

template <typename T>
void UnsqueezeKernel(const phi::Context& dev_ctx,
                     const phi::DenseTensor& x,
                     const phi::IntArray& axes,
                     phi::DenseTensor* out) {
  auto x_dims = x.dims();
  auto out_dims = out->dims();

  std::vector<int32_t> tmp;
  tmp.reserve(axes.GetData().size());
  std::for_each(axes.GetData().begin(),
                axes.GetData().end(),
                [&tmp](const int64_t& t) { tmp.push_back(t); });
  out_dims = GetUnsqueezeShape(tmp, x_dims);

  out->Resize(out_dims);
  dev_ctx.template Alloc<T>(out);

  auto x_data = x.data<T>();
  auto out_data = out->data<T>();
  mps::memcpy_d2d(out_data, x_data, x.numel() * sizeof(T));
  out->Resize(out_dims);  // copy will reset the dims.
}

template <typename T>
void UnsqueezeWithXShapeKernel(const phi::Context& dev_ctx,
                               const phi::DenseTensor& x,
                               const phi::IntArray& axes,
                               phi::DenseTensor* out,
                               phi::DenseTensor* xshape) {
  custom_kernel::UnsqueezeKernel<T>(dev_ctx, x, axes, out);
}

template <typename T>
void UnsqueezeGradKernel(const phi::Context& dev_ctx,
                         const phi::DenseTensor& x,
                         const phi::DenseTensor& dout,
                         phi::DenseTensor* dx) {
  auto x_dims = dx->dims();

  dev_ctx.template Alloc<T>(dx);
  auto dout_data = dout.data<T>();
  auto dx_data = dx->data<T>();
  mps::memcpy_d2d(dx_data, dout_data, dout.numel() * sizeof(T));
  dx->Resize(x_dims);
}

}  // namespace custom_kernel

PD_BUILD_PHI_KERNEL(
    unsqueeze, mps, ALL_LAYOUT, custom_kernel::UnsqueezeKernel, float) {}

PD_BUILD_PHI_KERNEL(unsqueeze_with_xshape,
                    mps,
                    ALL_LAYOUT,
                    custom_kernel::UnsqueezeWithXShapeKernel,
                    float) {}

PD_BUILD_PHI_KERNEL(unsqueeze_grad,
                    mps,
                    ALL_LAYOUT,
                    custom_kernel::UnsqueezeGradKernel,
                    float) {}
