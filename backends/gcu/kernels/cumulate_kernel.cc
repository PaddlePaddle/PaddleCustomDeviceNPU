// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include "kernels/funcs/gcu_kernel_funcs.h"

namespace custom_kernel {
template <typename T, typename Context>
void CumsumKernel(const Context& dev_ctx,
                  const phi::DenseTensor& x,
                  const phi::Scalar& axis_scalar,
                  bool flatten,
                  bool exclusive,
                  bool reverse,
                  phi::DenseTensor* out) {
  PADDLE_GCU_KERNEL_TRACE("cumsum");
  if (LaunchAOTKernel()) {
    auto axis = axis_scalar.to<int>();
    phi::DenseTensor input_tensor(x);
    if (flatten) {
      PADDLE_ENFORCE_EQ(
          axis,
          -1,
          phi::errors::InvalidArgument(
              "when flatten is true, attr axis must be default %d, but got %d",
              -1,
              axis));
      input_tensor.Resize(phi::make_ddim({x.numel()}));
    }
    if (axis < 0) {
      axis += input_tensor.dims().size();
    }
    dev_ctx.template Alloc<T>(out);
    LAUNCH_TOPSATENOP(topsatenCumsum,
                      dev_ctx,
                      *out,
                      input_tensor,
                      axis,
                      input_tensor.dtype());

  } else {  // kernel impl base on JIT
    THROW_JIT_UNIMPLEMENTED();
  }
}

template <typename T, typename Context>
void CummaxKernel(const Context& dev_ctx,
                  const phi::DenseTensor& x,
                  int axis,
                  phi::DataType dtype,
                  phi::DenseTensor* out,
                  phi::DenseTensor* indices) {
  PADDLE_GCU_KERNEL_TRACE("cummax");
  if (LaunchAOTKernel()) {
    if (axis < 0) {
      axis += x.dims().size();
    }

    dev_ctx.template Alloc<T>(out);

    phi::DenseTensor indices_out;
    if (dtype == phi::DataType::INT64) {
      dev_ctx.template Alloc<int64_t>(indices);
      indices_out = MaybeCreateOrTrans64To32bits(dev_ctx, *indices, false);
    } else if (dtype == phi::DataType::INT32) {
      dev_ctx.template Alloc<int32_t>(indices);
      indices_out = *indices;
    } else {
      PADDLE_THROW(
          phi::errors::InvalidArgument("Unsupported indices dtype: %s.",
                                       phi::DataTypeToString(dtype).c_str()));
    }
    LAUNCH_TOPSATENOP(topsatenCummax, dev_ctx, *out, indices_out, x, axis);
    MaybeTransResult(dev_ctx, indices_out, indices);

  } else {  // kernel impl base on JIT
    THROW_JIT_UNIMPLEMENTED();
  }
}

template <typename T, typename Context>
void CumminKernel(const Context& dev_ctx,
                  const phi::DenseTensor& x,
                  int axis,
                  phi::DataType dtype,
                  phi::DenseTensor* out,
                  phi::DenseTensor* indices) {
  PADDLE_GCU_KERNEL_TRACE("cummin");
  if (LaunchAOTKernel()) {
    if (axis < 0) {
      axis += x.dims().size();
    }

    dev_ctx.template Alloc<T>(out);

    phi::DenseTensor indices_out;
    if (dtype == phi::DataType::INT64) {
      dev_ctx.template Alloc<int64_t>(indices);
      indices_out = MaybeCreateOrTrans64To32bits(dev_ctx, *indices, false);
    } else if (dtype == phi::DataType::INT32) {
      dev_ctx.template Alloc<int32_t>(indices);
      indices_out = *indices;
    } else {
      PADDLE_THROW(
          phi::errors::InvalidArgument("Unsupported indices dtype: %s.",
                                       phi::DataTypeToString(dtype).c_str()));
    }
    LAUNCH_TOPSATENOP(topsatenCummin, dev_ctx, *out, indices_out, x, axis);
    MaybeTransResult(dev_ctx, indices_out, indices);

  } else {  // kernel impl base on JIT
    THROW_JIT_UNIMPLEMENTED();
  }
}

template <typename T, typename Context>
void CumprodKernel(const Context& dev_ctx,
                   const phi::DenseTensor& input,
                   int dim,
                   bool exclusive,
                   bool reverse,
                   phi::DenseTensor* out) {
  PADDLE_GCU_KERNEL_TRACE("cumprod");
  if (LaunchAOTKernel()) {
    if (dim < 0) {
      dim += input.dims().size();
    }

    dev_ctx.template Alloc<T>(out);
    LAUNCH_TOPSATENOP(
        topsatenCumprod, dev_ctx, *out, input, dim, input.dtype());

  } else {  // kernel impl base on JIT
    THROW_JIT_UNIMPLEMENTED();
  }
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(cumsum,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::CumsumKernel,
                          int,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(cummax,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::CummaxKernel,
                          int,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(cummin,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::CumminKernel,
                          int,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(cumprod,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::CumprodKernel,
                          int,
                          float,
                          phi::dtype::float16) {}
