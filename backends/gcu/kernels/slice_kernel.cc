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
#include "paddle/phi/kernels/funcs/slice_utils.h"

namespace custom_kernel {

template <typename T, typename Context>
void SliceKernel(const Context& dev_ctx,
                 const phi::DenseTensor& x,
                 const std::vector<int64_t>& axes_t,
                 const phi::IntArray& starts_array,
                 const phi::IntArray& ends_array,
                 const std::vector<int64_t>& infer_flags,
                 const std::vector<int64_t>& decrease_axis,
                 phi::DenseTensor* out) {
  PADDLE_GCU_KERNEL_TRACE("slice");
  if (LaunchAOTKernel()) {
    auto axes = axes_t;
    auto starts = starts_array.GetData();
    auto ends = ends_array.GetData();

    PADDLE_ENFORCE_EQ(
        starts.size(),
        axes.size(),
        phi::errors::InvalidArgument(
            "The size of starts must be equal to the size of axes."));
    PADDLE_ENFORCE_EQ(
        ends.size(),
        axes.size(),
        phi::errors::InvalidArgument(
            "The size of ends must be equal to the size of axes."));

    auto in_dims = x.dims();
    auto out_dims = out->dims();
    auto slice_dims = out_dims;

    // Infer output dims
    for (size_t i = 0; i < axes.size(); ++i) {
      // when start == -1 && end == start+1
      if (starts[i] == -1 && ends[i] == 0 && infer_flags[i] == -1) {
        auto ret =
            std::find(decrease_axis.begin(), decrease_axis.end(), axes[i]);
        if (ret != decrease_axis.end()) {
          ends[i] = in_dims[axes[i]];
        }
      }
    }

    phi::funcs::UpdateSliceAttrs<int64_t>(in_dims, axes, &starts, &ends);
    slice_dims = phi::funcs::GetSliceDims<int64_t>(
        in_dims, axes, starts, ends, nullptr, nullptr);
    out_dims = phi::funcs::GetDecreasedDims<int64_t>(slice_dims, decrease_axis);

    out->Resize(slice_dims);
    dev_ctx.template Alloc<T>(out);
    if (out->data() == x.data()) {
      *out = TensorEmpty(dev_ctx, out->meta());
    }

    custom_kernel::SliceBase(dev_ctx, x, axes, starts, out);
    out->Resize(out_dims);

  } else {  // kernel impl base on JIT
    std::vector<int> axes(axes_t.begin(), axes_t.end());
    auto starts_int = starts_array.GetData();
    auto ends_int = ends_array.GetData();
    std::vector<int> starts(starts_int.begin(), starts_int.end());
    std::vector<int> ends(ends_int.begin(), ends_int.end());

    PADDLE_ENFORCE_EQ(
        starts.size(),
        axes.size(),
        phi::errors::InvalidArgument(
            "The size of starts must be equal to the size of axes."));
    PADDLE_ENFORCE_EQ(
        ends.size(),
        axes.size(),
        phi::errors::InvalidArgument(
            "The size of ends must be equal to the size of axes."));

    dev_ctx.template Alloc<T>(out);

    TensorNameMap input_names;
    input_names["Input"] = {"x"};

    TensorValueMap inputs;
    inputs["Input"] = {const_cast<DenseTensor*>(&x)};

    TensorNameMap output_names;
    output_names["Out"] = {"out"};

    TensorValueMap outputs;
    outputs["Out"] = {out};

    GcuAttributeMap attrs;
    attrs["axes"] = axes;
    attrs["starts"] = starts;
    attrs["ends"] = ends;
    attrs["infer_flags"] =
        std::vector<int>(infer_flags.begin(), infer_flags.end());
    attrs["decrease_axis"] =
        std::vector<int>(decrease_axis.begin(), decrease_axis.end());

    GcuRunner(
        input_names, inputs, output_names, outputs, attrs, "slice", dev_ctx);
  }
}

template <typename T, typename Context>
void SliceGradKernel(const Context& dev_ctx,
                     const phi::DenseTensor& x,
                     const phi::DenseTensor& out_grad,
                     const std::vector<int64_t>& axes_t,
                     const phi::IntArray& starts_array,
                     const phi::IntArray& ends_array,
                     const std::vector<int64_t>& infer_flags,
                     const std::vector<int64_t>& decrease_axis,
                     phi::DenseTensor* x_grad) {
  PADDLE_GCU_KERNEL_TRACE("slice_grad");
  std::vector<int> axes(axes_t.begin(), axes_t.end());
  auto starts_int = starts_array.GetData();
  auto ends_int = ends_array.GetData();

  std::vector<int> starts(starts_int.begin(), starts_int.end());
  std::vector<int> ends(ends_int.begin(), ends_int.end());

  dev_ctx.template Alloc<T>(x_grad);

  if (LaunchAOTKernel()) {
    THROW_AOT_UNIMPLEMENTED();
  } else {  // kernel impl base on JIT
    TensorNameMap input_names;
    input_names["Input"] = {"x"};
    input_names[GradVarName("Out")] = {"out_grad"};

    TensorValueMap inputs;
    inputs["Input"] = {const_cast<DenseTensor*>(&x)};
    inputs[GradVarName("Out")] = {const_cast<DenseTensor*>(&out_grad)};

    TensorNameMap output_names;
    TensorValueMap outputs;

    output_names[GradVarName("Input")] = {"x_grad"};
    outputs[GradVarName("Input")] = {x_grad};

    GcuAttributeMap attrs;
    attrs["axes"] = axes;
    attrs["starts"] = starts;
    attrs["ends"] = ends;
    attrs["infer_flags"] =
        std::vector<int>(infer_flags.begin(), infer_flags.end());
    attrs["decrease_axis"] =
        std::vector<int>(decrease_axis.begin(), decrease_axis.end());

    GcuRunner(input_names,
              inputs,
              output_names,
              outputs,
              attrs,
              "slice_grad",
              dev_ctx);
  }
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(slice,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::SliceKernel,
                          phi::dtype::float16,
                          float,
                          double,
                          int32_t,
                          int64_t) {}

PD_REGISTER_PLUGIN_KERNEL(slice_grad,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::SliceGradKernel,
                          phi::dtype::float16,
                          float,
                          double,
                          int16_t,
                          int32_t,
                          int64_t,
                          bool) {}
