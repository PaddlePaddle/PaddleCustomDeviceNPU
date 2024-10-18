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
void FlattenKernel(const Context& dev_ctx,
                   const phi::DenseTensor& x,
                   int start_axis UNUSED,
                   int stop_axis UNUSED,
                   phi::DenseTensor* out) {
  PADDLE_GCU_KERNEL_TRACE("flatten");
  if (LaunchAOTKernel()) {
    VLOG(6) << "[HOST_KERNEL] Impl on host for flatten";
    if (x.numel() == 0) {
      return;
    }
    auto recovery = x;
    if (DataPdCustomNHWC(x)) {
      recovery = PdCustomNHWCTransToNCHW(dev_ctx, recovery);
    }
    dev_ctx.template Alloc<T>(out);
    if (recovery.initialized() && recovery.data() == out->data()) {
      return;
    }

    PADDLE_ENFORCE_EQ(x.numel(),
                      out->numel(),
                      phi::errors::InvalidArgument(
                          "src tensor shape is %s, dst tensor shape is %s",
                          x.dims().to_str().c_str(),
                          out->dims().to_str().c_str()));

    // the output dims are overwrite after copying,
    // here we need to use copy method that only copy data
    auto dims = out->dims();
    auto lod = out->lod();
    TensorCopy(dev_ctx, recovery, false, out);
    out->Resize(dims);
    out->ResetLoD(lod);
    SetLayout(*out, common::DataLayout::kNCHW);

  } else {  // kernel impl base on JIT
    THROW_JIT_UNIMPLEMENTED();
  }
}

template <typename T, typename Context>
void FlattenWithXShapeKernel(const Context& dev_ctx,
                             const phi::DenseTensor& x,
                             int start_axis,
                             int stop_axis,
                             phi::DenseTensor* out,
                             phi::DenseTensor* xshape) {
  PADDLE_GCU_KERNEL_TRACE("flatten_with_xshape");
  if (LaunchAOTKernel()) {
    custom_kernel::FlattenKernel<T, Context>(
        dev_ctx, x, start_axis, stop_axis, out);

  } else {  // kernel impl base on JIT
    THROW_JIT_UNIMPLEMENTED();
    dev_ctx.template Alloc<T>(out);
    dev_ctx.template Alloc<T>(xshape);

    TensorNameMap input_names;
    input_names["X"] = {"x"};

    TensorValueMap inputs;
    inputs["X"] = {const_cast<DenseTensor*>(&x)};

    TensorNameMap output_names;
    output_names["Out"] = {"out"};
    output_names["XShape"] = {"xshape"};

    TensorValueMap outputs;
    outputs["Out"] = {out};
    outputs["XShape"] = {xshape};

    GcuAttributeMap attrs;
    attrs["start_axis"] = start_axis;
    attrs["stop_axis"] = stop_axis;

    GcuRunner(input_names,
              inputs,
              output_names,
              outputs,
              attrs,
              "flatten_contiguous_range",
              dev_ctx);
  }
}

template <typename T, typename Context>
void FlattenGradKernel(const Context& dev_ctx,
                       const phi::DenseTensor& xshape,
                       const phi::DenseTensor& out_grad,
                       DenseTensor* x_grad) {
  PADDLE_GCU_KERNEL_TRACE("flatten_grad");
  dev_ctx.template Alloc<T>(x_grad);

  if (LaunchAOTKernel()) {
    THROW_AOT_UNIMPLEMENTED();
  } else {  // kernel impl base on JIT
    phi::DenseTensor* tmp_tensor = nullptr;

    TensorNameMap input_names;
    input_names["XShape"] = {"xshape"};
    input_names[GradVarName("Out")] = {"out_grad"};

    TensorValueMap inputs;
    inputs["XShape"] = {const_cast<DenseTensor*>(&xshape)};
    inputs[GradVarName("Out")] = {const_cast<DenseTensor*>(&out_grad)};

    TensorNameMap output_names;
    output_names[GradVarName("X")] = {"x_grad"};

    TensorValueMap outputs;
    outputs[GradVarName("X")] = {
        ((tmp_tensor == nullptr ? x_grad : tmp_tensor))};

    GcuAttributeMap attrs;

    GcuRunner(input_names,
              inputs,
              output_names,
              outputs,
              attrs,
              "flatten_contiguous_range_grad",
              dev_ctx);

    if (tmp_tensor != nullptr) {
      *x_grad = *tmp_tensor;
      delete tmp_tensor;
    }
  }
}
}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(flatten,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::FlattenKernel,
                          phi::dtype::bfloat16,
                          phi::dtype::float16,
                          float,
                          double,
                          uint8_t,
                          int8_t,
                          int,
                          int64_t) {}

PD_REGISTER_PLUGIN_KERNEL(flatten_with_xshape,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::FlattenWithXShapeKernel,
                          phi::dtype::bfloat16,
                          phi::dtype::float16,
                          float,
                          double,
                          uint8_t,
                          int8_t,
                          int,
                          int64_t) {}

PD_REGISTER_PLUGIN_KERNEL(flatten_grad,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::FlattenGradKernel,
                          phi::dtype::bfloat16,
                          float,
                          double,
                          uint8_t,
                          int8_t,
                          int,
                          int64_t) {}
