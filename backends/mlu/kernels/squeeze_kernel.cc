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

#include "kernels/funcs/mlu_baseop.h"
#include "kernels/funcs/mlu_funcs.h"

namespace custom_kernel {

phi::DDim GetOutputShape(const std::vector<int> squeeze_dims,
                         const phi::DDim& in_dims,
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

      PADDLE_ENFORCE_GE(
          current,
          0,
          phi::errors::InvalidArgument(
              "Each axis in Attr(axes) should be in the range of [%d, %d]"
              "But current axis is:%d, input tensor's shape = [%s].",
              -in_dims.size(),
              in_dims.size() - 1,
              current,
              in_dims));
      PADDLE_ENFORCE_LT(
          current,
          in_dims.size(),
          phi::errors::InvalidArgument(
              "Each axis in Attr(axes) should be in the range of [%d, %d]"
              "But current axis is:%d, input tensor's shape = [%s].",
              -in_dims.size(),
              in_dims.size() - 1,
              current,
              in_dims));

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
  return phi::make_ddim(output_shape);
}

template <typename T, typename Context>
void SqueezeKernel(const Context& dev_ctx,
                   const phi::DenseTensor& x,
                   const phi::IntArray& axes_int_array,
                   phi::DenseTensor* out) {
  auto stream = dev_ctx.stream();
  std::vector<int32_t> axes(axes_int_array.GetData().begin(),
                            axes_int_array.GetData().end());

  auto x_dims = x.dims();
  auto out_dims = custom_kernel::GetOutputShape(axes, x_dims, true);

  TensorCopy(dev_ctx, x, false, out);

  out->Resize(out_dims);
}

template <typename T, typename Context>
void SqueezeWithXShapeKernel(const Context& dev_ctx,
                             const phi::DenseTensor& x,
                             const phi::IntArray& axes_int_array,
                             phi::DenseTensor* out,
                             phi::DenseTensor* xshape) {
  custom_kernel::SqueezeKernel<T, Context>(dev_ctx, x, axes_int_array, out);
}

template <typename T, typename Context>
void SqueezeGradKernel(const Context& dev_ctx,
                       const phi::DenseTensor& x,
                       const phi::DenseTensor& dout,
                       const phi::IntArray& axes_int_array,
                       phi::DenseTensor* dx) {
  auto stream = dev_ctx.stream();

  auto x_dims = dx->dims();

  TensorCopy(dev_ctx, dout, false, dx);
  dx->Resize(x_dims);
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(squeeze,
                          mlu,
                          ALL_LAYOUT,
                          custom_kernel::SqueezeKernel,
                          bool,
                          int,
                          uint8_t,
                          int8_t,
                          int64_t,
                          float,
                          phi::dtype::float16,
                          double) {}

PD_REGISTER_PLUGIN_KERNEL(squeeze_with_xshape,
                          mlu,
                          ALL_LAYOUT,
                          custom_kernel::SqueezeWithXShapeKernel,
                          bool,
                          int,
                          uint8_t,
                          int8_t,
                          int64_t,
                          float,
                          phi::dtype::float16,
                          double) {}

PD_REGISTER_PLUGIN_KERNEL(squeeze_grad,
                          mlu,
                          ALL_LAYOUT,
                          custom_kernel::SqueezeGradKernel,
                          bool,
                          int,
                          uint8_t,
                          int8_t,
                          int64_t,
                          float,
                          phi::dtype::float16,
                          double) {}
