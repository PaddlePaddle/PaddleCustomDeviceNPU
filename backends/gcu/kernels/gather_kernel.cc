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
void GatherKernel(const Context& dev_ctx,
                  const phi::DenseTensor& x,
                  const phi::DenseTensor& index,
                  const phi::Scalar& axis,
                  phi::DenseTensor* out) {
  PADDLE_GCU_KERNEL_TRACE("gather");
  dev_ctx.template Alloc<T>(out);

  if (LaunchAOTKernel()) {
    VLOG(6) << "GatherKernel, x dims:" << x.dims()
            << ", out dims:" << out->dims();
    phi::DenseTensor input_x = MaybeCreateOrTrans64To32bits(dev_ctx, x);
    phi::DenseTensor output =
        MaybeCreateOrTrans64To32bits(dev_ctx, *out, false);
    auto gather_axis = axis.to<int64_t>();
    if (gather_axis < 0) {
      gather_axis += x.dims().size();
    }
    LAUNCH_TOPSATENOP(
        topsatenIndexSelect, dev_ctx, output, input_x, gather_axis, index);
    MaybeTransResult(dev_ctx, output, out);

    // phi::DenseTensor input_x(x);
    // phi::DenseTensor input_index(index);
    // phi::DenseTensor output(*out);

    // if (x.dtype() == phi::DataType::INT64) {
    //   input_x = custom_kernel::Cast(dev_ctx, x, phi::DataType::INT32);
    // }

    // if (index.dtype() == phi::DataType::INT64) {
    //   input_index = custom_kernel::Cast(dev_ctx, index,
    //   phi::DataType::INT32);
    // }

    // if (out->dtype() == phi::DataType::INT64) {
    //   auto meta = out->meta();
    //   meta.dtype = phi::DataType::INT32;
    //   output.set_meta(meta);
    //   dev_ctx.template Alloc(&output, output.dtype());
    // }

    // auto rank = x.dims().size();
    // auto gather_axis = axis.to<int64_t>();
    // if (gather_axis < 0) {
    //   gather_axis += rank;
    // }

    // // * @param offset_dims: The set of dimensions in the output shape that
    // // offset
    // //                       into an array sliced from x.
    // std::vector<int64_t> offset_dims;
    // for (auto i = 0; i < gather_axis; ++i) {
    //   offset_dims.push_back(i);
    // }
    // for (auto i = gather_axis + 1; i < rank; ++i) {
    //   offset_dims.push_back(i);
    // }

    // // * @param slice_sizes: slice_sizes[i] is the bounds for the slice on
    // //                       dimension i.
    // std::vector<int64_t> slice_sizes = phi::vectorize(x.dims());
    // slice_sizes.at(gather_axis) = 1;

    // // * @param collapsed_slice_dims: The set of dimensions in each slice
    // that
    // // are
    // //                                collapsed away.
    // //                                These dimensions must have size 1.
    // std::vector<int64_t> collapsed_slice_dims = {gather_axis};

    // // * @param start_index_map: A map that describes how to map indices in
    // //                           start_indices to legal indices into x.
    // std::vector<int64_t> start_index_map = {gather_axis};

    // // * @param index_vector_dim: The dimension in start_indices that
    // "contains"
    // //                            the starting indices.
    // int64_t index_vector_dim = 1;

    // // * @param indices_are_sorted: Whether the indices are guaranteed to be
    // //                              sorted by the caller.
    // bool indices_are_sorted = false;

    // // * @param unique_indices: Whether the indices are guaranteed to be
    // unique
    // //                          by the caller.
    // bool unique_indices = false;

    // LAUNCH_TOPSATENOP(topsxlaGather,
    //                   dev_ctx,
    //                   output,
    //                   input_x,
    //                   input_index,
    //                   offset_dims,
    //                   slice_sizes,
    //                   collapsed_slice_dims,
    //                   start_index_map,
    //                   index_vector_dim,
    //                   indices_are_sorted,
    //                   unique_indices);

    // if (out->dtype() == phi::DataType::INT64) {
    //   custom_kernel::Cast(dev_ctx, output, phi::DataType::INT64, out);
    // }

  } else {  // kernel impl base on JIT
    TensorNameMap input_names;
    input_names["X"] = {"x"};
    input_names["Index"] = {"index"};

    TensorValueMap inputs;
    inputs["X"] = {const_cast<DenseTensor*>(&x)};
    inputs["Index"] = {const_cast<DenseTensor*>(&index)};

    TensorNameMap output_names;
    output_names["Out"] = {"out"};

    TensorValueMap outputs;
    outputs["Out"] = {out};

    int gather_axis = axis.to<int>();

    GcuAttributeMap attrs;
    attrs["axis"] = gather_axis;

    GcuRunner(
        input_names, inputs, output_names, outputs, attrs, "gather", dev_ctx);
  }
}

template <typename T, typename Context>
void GatherGradKernel(const Context& dev_ctx,
                      const phi::DenseTensor& x,
                      const phi::DenseTensor& index,
                      const phi::DenseTensor& out_grad,
                      const phi::Scalar& axis,
                      phi::DenseTensor* x_grad) {
  PADDLE_GCU_KERNEL_TRACE("gather_grad");
  dev_ctx.template Alloc<T>(x_grad);

  if (LaunchAOTKernel()) {
    THROW_AOT_UNIMPLEMENTED();
  } else {  // kernel impl base on JIT
    TensorNameMap input_names;
    input_names["X"] = {"x"};
    input_names["Index"] = {"index"};
    input_names[GradVarName("Out")] = {"out_grad"};

    TensorValueMap inputs;
    inputs["X"] = {const_cast<DenseTensor*>(&x)};
    inputs["Index"] = {const_cast<DenseTensor*>(&index)};
    inputs[GradVarName("Out")] = {const_cast<DenseTensor*>(&out_grad)};

    TensorNameMap output_names;
    output_names[GradVarName("X")] = {"x_grad"};

    TensorValueMap outputs;
    outputs[GradVarName("X")] = {x_grad};

    int gather_axis = axis.to<int>();

    GcuAttributeMap attrs;
    attrs["axis"] = gather_axis;

    GcuRunner(input_names,
              inputs,
              output_names,
              outputs,
              attrs,
              "gather_grad",
              dev_ctx);
  }
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(gather,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::GatherKernel,
                          int,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(gather_grad,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::GatherGradKernel,
                          int,
                          int64_t,
                          float,
                          phi::dtype::float16) {}
