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
namespace {
void CheckParam(const std::string& name, int axis, size_t rank) {
  int axis_c = axis < 0 ? (axis + rank) : axis;
  PADDLE_ENFORCE_EQ(axis_c,
                    (rank - 1),
                    phi::errors::InvalidArgument(
                        "Not support axis %d for %s", axis, name.c_str()));
}
}  // namespace

template <typename T, typename Context>
void CompareBaseKernel(const Context& dev_ctx,
                       const phi::DenseTensor& x,
                       const phi::DenseTensor& y,
                       int axis,
                       phi::DenseTensor* out,
                       const std::string& op_type) {
  dev_ctx.template Alloc<bool>(out);

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
  attrs["axis"] = axis;

  GcuRunner(
      input_names, inputs, output_names, outputs, attrs, op_type, dev_ctx);
}

template <typename T, typename Context>
void EqualKernelRaw(const Context& dev_ctx,
                    const phi::DenseTensor& x,
                    const phi::DenseTensor& y,
                    int axis,
                    phi::DenseTensor* out) {
  PADDLE_GCU_KERNEL_TRACE("equal_raw");
  if (LaunchAOTKernel()) {
    CheckParam("equal_raw", axis, x.dims().size());
    dev_ctx.template Alloc<bool>(out);
    phi::DenseTensor input_x = MaybeCreateOrTrans64To32bits(dev_ctx, x);
    phi::DenseTensor input_y = MaybeCreateOrTrans64To32bits(dev_ctx, y);
    LAUNCH_TOPSATENOP(topsatenEq, dev_ctx, *out, input_x, input_y);

  } else {  // kernel impl base on JIT
    CompareBaseKernel<T, Context>(dev_ctx, x, y, -1, out, "equal");
  }
}

template <typename T, typename Context>
void EqualKernel(const Context& dev_ctx,
                 const phi::DenseTensor& x,
                 const phi::DenseTensor& y,
                 phi::DenseTensor* out) {
  PADDLE_GCU_KERNEL_TRACE("equal");
  if (LaunchAOTKernel()) {
    dev_ctx.template Alloc<bool>(out);
    phi::DenseTensor input_x = MaybeCreateOrTrans64To32bits(dev_ctx, x);
    phi::DenseTensor input_y = MaybeCreateOrTrans64To32bits(dev_ctx, y);
    LAUNCH_TOPSATENOP(topsatenEq, dev_ctx, *out, input_x, input_y);

  } else {  // kernel impl base on JIT
    CompareBaseKernel<T, Context>(dev_ctx, x, y, -1, out, "equal");
  }
}

template <typename T, typename Context>
void NotEqualKernel(const Context& dev_ctx,
                    const phi::DenseTensor& x,
                    const phi::DenseTensor& y,
                    phi::DenseTensor* out) {
  PADDLE_GCU_KERNEL_TRACE("not_equal");
  if (LaunchAOTKernel()) {
    dev_ctx.template Alloc<bool>(out);
    phi::DenseTensor input_x = MaybeCreateOrTrans64To32bits(dev_ctx, x);
    phi::DenseTensor input_y = MaybeCreateOrTrans64To32bits(dev_ctx, y);
    LAUNCH_TOPSATENOP(topsatenNe, dev_ctx, *out, input_x, input_y);

  } else {  // kernel impl base on JIT
    CompareBaseKernel<T, Context>(dev_ctx, x, y, -1, out, "not_equal");
  }
}

template <typename T, typename Context>
void LessEqualKernel(const Context& dev_ctx,
                     const phi::DenseTensor& x,
                     const phi::DenseTensor& y,
                     phi::DenseTensor* out) {
  PADDLE_GCU_KERNEL_TRACE("less_equal");
  if (LaunchAOTKernel()) {
    dev_ctx.template Alloc<bool>(out);
    phi::DenseTensor input_x = MaybeCreateOrTrans64To32bits(dev_ctx, x);
    phi::DenseTensor input_y = MaybeCreateOrTrans64To32bits(dev_ctx, y);
    LAUNCH_TOPSATENOP(topsatenLe, dev_ctx, *out, input_x, input_y);

  } else {  // kernel impl base on JIT
    CompareBaseKernel<T, Context>(dev_ctx, x, y, -1, out, "less_equal");
  }
}

template <typename T, typename Context>
void LessThanKernelRaw(const Context& dev_ctx,
                       const phi::DenseTensor& x,
                       const phi::DenseTensor& y,
                       int axis,
                       phi::DenseTensor* out) {
  PADDLE_GCU_KERNEL_TRACE("less_than_raw");
  if (LaunchAOTKernel()) {
    CheckParam("less_than_raw", axis, x.dims().size());
    dev_ctx.template Alloc<bool>(out);
    phi::DenseTensor input_x = MaybeCreateOrTrans64To32bits(dev_ctx, x);
    phi::DenseTensor input_y = MaybeCreateOrTrans64To32bits(dev_ctx, y);
    LAUNCH_TOPSATENOP(topsatenLt, dev_ctx, *out, input_x, input_y);
  } else {  // kernel impl base on JIT
    CompareBaseKernel<T, Context>(dev_ctx, x, y, -1, out, "less_than");
  }
}

template <typename T, typename Context>
void LessThanKernel(const Context& dev_ctx,
                    const phi::DenseTensor& x,
                    const phi::DenseTensor& y,
                    phi::DenseTensor* out) {
  PADDLE_GCU_KERNEL_TRACE("less_than");
  if (LaunchAOTKernel()) {
    dev_ctx.template Alloc<bool>(out);
    phi::DenseTensor input_x = MaybeCreateOrTrans64To32bits(dev_ctx, x);
    phi::DenseTensor input_y = MaybeCreateOrTrans64To32bits(dev_ctx, y);
    LAUNCH_TOPSATENOP(topsatenLt, dev_ctx, *out, input_x, input_y);

  } else {  // kernel impl base on JIT
    CompareBaseKernel<T, Context>(dev_ctx, x, y, -1, out, "less_than");
  }
}

template <typename T, typename Context>
void GreaterEqualKernelRaw(const Context& dev_ctx,
                           const phi::DenseTensor& x,
                           const phi::DenseTensor& y,
                           int axis,
                           phi::DenseTensor* out) {
  PADDLE_GCU_KERNEL_TRACE("greater_equal_raw");
  if (LaunchAOTKernel()) {
    CheckParam("greater_equal_raw", axis, x.dims().size());
    dev_ctx.template Alloc<bool>(out);
    phi::DenseTensor input_x = MaybeCreateOrTrans64To32bits(dev_ctx, x);
    phi::DenseTensor input_y = MaybeCreateOrTrans64To32bits(dev_ctx, y);
    LAUNCH_TOPSATENOP(topsatenGe, dev_ctx, *out, input_x, input_y);
  } else {  // kernel impl base on JIT
    CompareBaseKernel<T, Context>(dev_ctx, x, y, -1, out, "greater_equal");
  }
}

template <typename T, typename Context>
void GreaterEqualKernel(const Context& dev_ctx,
                        const phi::DenseTensor& x,
                        const phi::DenseTensor& y,
                        phi::DenseTensor* out) {
  PADDLE_GCU_KERNEL_TRACE("greater_equal");
  if (LaunchAOTKernel()) {
    dev_ctx.template Alloc<bool>(out);
    phi::DenseTensor input_x = MaybeCreateOrTrans64To32bits(dev_ctx, x);
    phi::DenseTensor input_y = MaybeCreateOrTrans64To32bits(dev_ctx, y);
    LAUNCH_TOPSATENOP(topsatenGe, dev_ctx, *out, input_x, input_y);
  } else {  // kernel impl base on JIT
    CompareBaseKernel<T, Context>(dev_ctx, x, y, -1, out, "greater_equal");
  }
}

template <typename T, typename Context>
void GreaterThanKernel(const Context& dev_ctx,
                       const phi::DenseTensor& x,
                       const phi::DenseTensor& y,
                       phi::DenseTensor* out) {
  PADDLE_GCU_KERNEL_TRACE("greater_than");
  if (LaunchAOTKernel()) {
    dev_ctx.template Alloc<bool>(out);
    phi::DenseTensor input_x = MaybeCreateOrTrans64To32bits(dev_ctx, x);
    phi::DenseTensor input_y = MaybeCreateOrTrans64To32bits(dev_ctx, y);
    LAUNCH_TOPSATENOP(topsatenGt, dev_ctx, *out, input_x, input_y);

  } else {  // kernel impl base on JIT
    CompareBaseKernel<T, Context>(dev_ctx, x, y, -1, out, "greater_than");
  }
}

}  // namespace custom_kernel

#define PD_REGISTER_COMPARE_KERNEL(name, func)            \
  PD_REGISTER_PLUGIN_KERNEL(name,                         \
                            gcu,                          \
                            ALL_LAYOUT,                   \
                            custom_kernel::func##Kernel,  \
                            bool,                         \
                            int,                          \
                            int64_t,                      \
                            float,                        \
                            phi::dtype::float16) {        \
    kernel->OutputAt(0).SetDataType(phi::DataType::BOOL); \
  }

#define PD_REGISTER_COMPARE_RAW_KERNEL(name, func)          \
  PD_REGISTER_PLUGIN_KERNEL(name##_raw,                     \
                            gcu,                            \
                            ALL_LAYOUT,                     \
                            custom_kernel::func##KernelRaw, \
                            bool,                           \
                            int,                            \
                            int64_t,                        \
                            float,                          \
                            phi::dtype::float16) {          \
    kernel->OutputAt(0).SetDataType(phi::DataType::BOOL);   \
  }

PD_REGISTER_COMPARE_KERNEL(less_than, LessThan)
PD_REGISTER_COMPARE_KERNEL(less_equal, LessEqual)
PD_REGISTER_COMPARE_KERNEL(greater_than, GreaterThan)
PD_REGISTER_COMPARE_KERNEL(greater_equal, GreaterEqual)
PD_REGISTER_COMPARE_KERNEL(equal, Equal)
PD_REGISTER_COMPARE_KERNEL(not_equal, NotEqual)

PD_REGISTER_COMPARE_RAW_KERNEL(less_than, LessThan)
PD_REGISTER_COMPARE_RAW_KERNEL(greater_equal, GreaterEqual)
PD_REGISTER_COMPARE_RAW_KERNEL(equal, Equal)
