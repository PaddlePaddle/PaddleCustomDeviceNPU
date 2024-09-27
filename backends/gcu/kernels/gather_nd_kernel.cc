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
void GatherNdKernel(const Context &dev_ctx,
                    const phi::DenseTensor &x,
                    const phi::DenseTensor &index,
                    phi::DenseTensor *out) {
  PADDLE_GCU_KERNEL_TRACE("gather_nd");
  dev_ctx.template Alloc<T>(out);

  if (x.numel() == 0) {
    return;
  }

  bool empty_index = false;
  std::vector<int> result_dims;
  if (index.numel() == 0) {
    int diff = out->dims().size() - x.dims().size();
    if (diff == 0) {
      TensorCopy(dev_ctx, x, false, out);
      return;
    } else {
      empty_index = true;
      result_dims = phi::vectorize<int>(out->dims());
    }
  }

  if (LaunchAOTKernel()) {
    phi::DenseTensor input_x(x);
    phi::DenseTensor input_index(index);
    phi::DenseTensor output(*out);

    if (x.dtype() == phi::DataType::INT64) {
      input_x = custom_kernel::Cast(dev_ctx, x, phi::DataType::INT32);
    }

    if (index.dtype() == phi::DataType::INT64) {
      input_index = custom_kernel::Cast(dev_ctx, index, phi::DataType::INT32);
    }

    if (out->dtype() == phi::DataType::INT64) {
      auto meta = out->meta();
      meta.dtype = phi::DataType::INT32;
      output.set_meta(meta);
      dev_ctx.template Alloc(&output, output.dtype());
    }

    std::vector<int64_t> src_shape = phi::vectorize(x.dims());
    std::vector<int64_t> index_shape = phi::vectorize(index.dims());
    int64_t src_rank = src_shape.size();
    int64_t index_rank = index_shape.size();

    int64_t indices_last_dim_size = index_shape.back();
    int64_t slices_rank = src_rank - indices_last_dim_size;
    int64_t batch_dims_size = index_rank - 1;

    // * @param offset_dims: The set of dimensions in the output shape that
    // offset
    //                       into an array sliced from x.
    std::vector<int64_t> offset_dims;
    for (auto i = 0; i < slices_rank; ++i) {
      offset_dims.push_back(batch_dims_size + i);
    }

    // * @param slice_sizes: slice_sizes[i] is the bounds for the slice on
    //                       dimension i.
    std::vector<int64_t> slice_sizes(indices_last_dim_size, 1);
    for (int64_t i = indices_last_dim_size; i < src_rank; ++i) {
      slice_sizes.push_back(src_shape.at(i));
    }

    // * @param collapsed_slice_dims: The set of dimensions in each slice that
    // are
    //                                collapsed away.
    //                                These dimensions must have size 1.
    std::vector<int64_t> collapsed_slice_dims;
    for (auto i = 0; i < indices_last_dim_size; ++i) {
      collapsed_slice_dims.push_back(i);
    }

    // * @param start_index_map: A map that describes how to map indices in
    //                           start_indices to legal indices into x.
    std::vector<int64_t> start_index_map = collapsed_slice_dims;

    // * @param index_vector_dim: The dimension in start_indices that "contains"
    //                            the starting indices.
    int64_t index_vector_dim = index_rank - 1;

    // * @param indices_are_sorted: Whether the indices are guaranteed to be
    //                              sorted by the caller.
    bool indices_are_sorted = false;

    // * @param unique_indices: Whether the indices are guaranteed to be unique
    //                          by the caller.
    bool unique_indices = false;

    LAUNCH_TOPSATENOP(topsxlaGather,
                      dev_ctx,
                      output,
                      input_x,
                      input_index,
                      offset_dims,
                      slice_sizes,
                      collapsed_slice_dims,
                      start_index_map,
                      index_vector_dim,
                      indices_are_sorted,
                      unique_indices);

    if (out->dtype() == phi::DataType::INT64) {
      custom_kernel::Cast(dev_ctx, output, phi::DataType::INT64, out);
    }

  } else {  // kernel impl base on JIT
    TensorNameMap input_names;
    input_names["X"] = {"x"};
    input_names["Index"] = {"index"};

    TensorValueMap inputs;
    inputs["X"] = {const_cast<DenseTensor *>(&x)};
    inputs["Index"] = {const_cast<DenseTensor *>(&index)};

    TensorNameMap output_names;
    output_names["Out"] = {"out"};

    TensorValueMap outputs;
    outputs["Out"] = {out};

    GcuAttributeMap attrs;
    attrs["gather_nd_with_empty_index"] = empty_index;
    attrs["result_dims"] = result_dims;

    GcuRunner(input_names,
              inputs,
              output_names,
              outputs,
              attrs,
              "gather_nd",
              dev_ctx);
  }
}

template <typename T, typename Context>
void GatherNdGradKernel(const Context &dev_ctx,
                        const phi::DenseTensor &x,
                        const phi::DenseTensor &index,
                        const phi::DenseTensor &dout,
                        phi::DenseTensor *dx) {
  PADDLE_GCU_KERNEL_TRACE("gather_nd_grad");
  auto x_dims = dx->dims();
  dev_ctx.template Alloc<T>(dx);

  if (dx->numel() == 0) return;

  bool empty_index = false;
  std::vector<int> reduce_axes;
  if (index.numel() == 0) {
    int diff = dout.dims().size() - x.dims().size();
    if (diff == 0) {
      TensorCopy(dev_ctx, dout, false, dx);
      return;
    } else {
      for (size_t i = 0; i < diff; ++i) {
        reduce_axes.emplace_back(i);
      }
      empty_index = true;
    }
  }

  if (LaunchAOTKernel()) {
    THROW_AOT_UNIMPLEMENTED();
  } else {  // kernel impl base on JIT
    TensorNameMap input_names;
    input_names["X"] = {"x"};
    input_names["Index"] = {"index"};
    input_names[GradVarName("Out")] = {"dout"};

    TensorValueMap inputs;
    inputs["X"] = {const_cast<DenseTensor *>(&x)};
    inputs["Index"] = {const_cast<DenseTensor *>(&index)};
    inputs[GradVarName("Out")] = {const_cast<DenseTensor *>(&dout)};

    TensorNameMap output_names;
    output_names[GradVarName("X")] = {"dx"};

    TensorValueMap outputs;
    outputs[GradVarName("X")] = {dx};

    GcuAttributeMap attrs;
    attrs["gather_nd_grad_with_empty_index"] = empty_index;
    attrs["reduce_axes"] = reduce_axes;

    GcuRunner(input_names,
              inputs,
              output_names,
              outputs,
              attrs,
              "gather_nd_grad",
              dev_ctx);
  }
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(gather_nd,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::GatherNdKernel,
                          int,
                          phi::dtype::float16,
                          float,
                          bool) {}

PD_REGISTER_PLUGIN_KERNEL(gather_nd_grad,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::GatherNdGradKernel,
                          int,
                          int64_t,
                          phi::dtype::float16,
                          float) {}
