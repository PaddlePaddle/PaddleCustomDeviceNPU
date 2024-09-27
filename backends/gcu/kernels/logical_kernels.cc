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

namespace custom_kernel {
template <typename T, typename Context>
void LogicalAndKernel(const Context& dev_ctx,
                      const phi::DenseTensor& x,
                      const phi::DenseTensor& y,
                      phi::DenseTensor* out) {
  PADDLE_GCU_KERNEL_TRACE("logical_and");
  dev_ctx.template Alloc<bool>(out);

  if (LaunchAOTKernel()) {
    // phi::DenseTensor real_x = x;
    // phi::DenseTensor real_y = y;
    // if (x.dtype() != phi::DataType::BOOL) {
    //   real_x = custom_kernel::Cast(dev_ctx, x, phi::DataType::BOOL);
    // }
    // if (y.dtype() != phi::DataType::BOOL) {
    //   real_y = custom_kernel::Cast(dev_ctx, y, phi::DataType::BOOL);
    // }
    // LAUNCH_TOPSATENOP(topsatenBitwiseAnd, dev_ctx, *out, real_x, real_y);
    LAUNCH_TOPSATENOP(topsatenLogicalAnd, dev_ctx, *out, x, y);

  } else {  // kernel impl base on JIT
    TensorNameMap input_names;
    input_names["X"] = {"x"};
    input_names["Y"] = {"y"};

    TensorValueMap inputs;
    inputs["X"] = {const_cast<DenseTensor*>(&x)};
    inputs["Y"] = {const_cast<DenseTensor*>(&y)};

    TensorNameMap output_names;
    output_names["Out"] = {"out"};

    TensorValueMap outputs;
    outputs["Out"] = {out};

    GcuAttributeMap attrs;

    GcuRunner(input_names,
              inputs,
              output_names,
              outputs,
              attrs,
              "logical_and",
              dev_ctx);
  }
}

template <typename T, typename Context>
void LogicalNotKernel(const Context& dev_ctx,
                      const phi::DenseTensor& x,
                      phi::DenseTensor* out) {
  PADDLE_GCU_KERNEL_TRACE("logical_not");
  dev_ctx.template Alloc<bool>(out);

  if (LaunchAOTKernel()) {
    // phi::DenseTensor real_x = x;
    // if (x.dtype() != phi::DataType::BOOL) {
    //   real_x = custom_kernel::Cast(dev_ctx, x, phi::DataType::BOOL);
    // }
    // LAUNCH_TOPSATENOP(topsatenBitwiseNot, dev_ctx, *out, real_x);
    LAUNCH_TOPSATENOP(topsatenLogicalNot, dev_ctx, *out, x);

  } else {  // kernel impl base on JIT
    TensorNameMap input_names;
    input_names["X"] = {"x"};

    TensorValueMap inputs;
    inputs["X"] = {const_cast<DenseTensor*>(&x)};

    TensorNameMap output_names;
    output_names["Out"] = {"out"};

    TensorValueMap outputs;
    outputs["Out"] = {out};

    GcuAttributeMap attrs;

    GcuRunner(input_names,
              inputs,
              output_names,
              outputs,
              attrs,
              "logical_not",
              dev_ctx);
  }
}

template <typename T, typename Context>
void LogicalOrKernel(const Context& dev_ctx,
                     const phi::DenseTensor& x,
                     const phi::DenseTensor& y,
                     phi::DenseTensor* out) {
  PADDLE_GCU_KERNEL_TRACE("logical_or");
  dev_ctx.template Alloc<bool>(out);

  if (LaunchAOTKernel()) {
    // phi::DenseTensor real_x = x;
    // phi::DenseTensor real_y = y;
    // if (x.dtype() != phi::DataType::BOOL) {
    //   real_x = custom_kernel::Cast(dev_ctx, x, phi::DataType::BOOL);
    // }
    // if (y.dtype() != phi::DataType::BOOL) {
    //   real_y = custom_kernel::Cast(dev_ctx, y, phi::DataType::BOOL);
    // }
    // LAUNCH_TOPSATENOP(topsatenBitwiseOr, dev_ctx, *out, real_x, real_y);
    LAUNCH_TOPSATENOP(topsatenLogicalOr, dev_ctx, *out, x, y);

  } else {  // kernel impl base on JIT
    THROW_JIT_UNIMPLEMENTED();
  }
}

template <typename T, typename Context>
void LogicalXorKernel(const Context& dev_ctx,
                      const phi::DenseTensor& x,
                      const phi::DenseTensor& y,
                      phi::DenseTensor* out) {
  PADDLE_GCU_KERNEL_TRACE("logical_xor");
  dev_ctx.template Alloc<bool>(out);

  if (LaunchAOTKernel()) {
    // phi::DenseTensor real_x = x;
    // phi::DenseTensor real_y = y;
    // if (x.dtype() != phi::DataType::BOOL) {
    //   real_x = custom_kernel::Cast(dev_ctx, x, phi::DataType::BOOL);
    // }
    // if (y.dtype() != phi::DataType::BOOL) {
    //   real_y = custom_kernel::Cast(dev_ctx, y, phi::DataType::BOOL);
    // }
    // LAUNCH_TOPSATENOP(topsatenBitwiseOr, dev_ctx, *out, real_x, real_y);
    LAUNCH_TOPSATENOP(topsatenLogicalXor, dev_ctx, *out, x, y);

  } else {  // kernel impl base on JIT
    THROW_JIT_UNIMPLEMENTED();
  }
}
}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(logical_and,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::LogicalAndKernel,
                          bool,
                          float,
                          phi::dtype::float16) {
  kernel->OutputAt(0).SetDataType(phi::DataType::BOOL);
}

PD_REGISTER_PLUGIN_KERNEL(logical_not,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::LogicalNotKernel,
                          bool,
                          float,
                          phi::dtype::float16) {
  kernel->OutputAt(0).SetDataType(phi::DataType::BOOL);
}

PD_REGISTER_PLUGIN_KERNEL(logical_or,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::LogicalOrKernel,
                          bool,
                          float,
                          phi::dtype::float16) {
  kernel->OutputAt(0).SetDataType(phi::DataType::BOOL);
}

PD_REGISTER_PLUGIN_KERNEL(logical_xor,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::LogicalXorKernel,
                          bool,
                          float,
                          phi::dtype::float16) {
  kernel->OutputAt(0).SetDataType(phi::DataType::BOOL);
}
