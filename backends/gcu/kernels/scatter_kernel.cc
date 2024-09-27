
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
void ScatterKernel(const Context& dev_ctx,
                   const phi::DenseTensor& x,
                   const phi::DenseTensor& index,
                   const phi::DenseTensor& updates,
                   bool overwrite,
                   phi::DenseTensor* out) {
  PADDLE_GCU_KERNEL_TRACE("scatter");
  dev_ctx.template Alloc<T>(out);

  if (LaunchAOTKernel()) {
    std::vector<int64_t> input_shape = phi::vectorize(x.dims());
    std::vector<int64_t> updates_shape = phi::vectorize(updates.dims());
    std::vector<int64_t> index_shape = phi::vectorize(index.dims());

    int64_t num_index_dims = 1;
    int64_t num_indices = index_shape.size();

    if (num_indices == 0) {
      PADDLE_THROW(
          phi::errors::InvalidArgument("num_indices should greater than 0."));
    }

    // If any of the indexed dimensions are zero in the input shape,
    // the update cannotsucceed since it updates a slice of size 1.
    for (int64_t i = 0; i < num_index_dims; ++i) {
      PADDLE_ENFORCE(
          input_shape.at(i) != 0,
          phi::errors::InvalidArgument("Scatter dimension ", i, " is zero."));
    }

    int64_t updates_rank = updates_shape.size();
    int64_t input_rank = input_shape.size();
    int64_t num_window_dims_in_updates = input_rank - num_index_dims;

    // * @param computation: Computation to be used for combining the existing
    //                       values in the input array and the updates
    //                       during scatter.
    topsatenScatterComputationType_t computation_type =
        TOPSATEN_SCATTER_COMP_UPDATE;

    // * @param index_vector_dim: The dimension in indices that contains the
    //                            starting indices.
    int64_t index_vector_dim = index_shape.size();

    // * @param update_window_dims: The set of dimensions in updates shape that
    //                              are window dimensions.
    std::vector<int64_t> update_window_dims;
    if (updates_rank > 0) {
      for (int64_t i = (updates_rank - num_window_dims_in_updates);
           i < updates_rank;
           ++i) {
        update_window_dims.push_back(i);
      }
    }

    // * @param inserted_window_dims: The set of window dimensions that must be
    //                                inserted into updates shape.
    std::vector<int64_t> inserted_window_dims;
    for (int64_t i = 0; i < num_index_dims; ++i) {
      inserted_window_dims.push_back(i);
    }

    // * @param scatter_dims_to_operand_dims: A dimensions map from the scatter
    //                                        indices to the operand index
    //                                        space.
    std::vector<int64_t> scatter_dims_to_operand_dims = inserted_window_dims;

    // * @param indices_are_sorted: Whether the indices are guaranteed to be
    //                              sorted by the caller.
    bool indices_are_sorted = false;

    // * @param unique_indices: Whether the indices are guaranteed to be unique
    //                          by the caller.
    bool unique_indices = true;

    phi::DenseTensor input_index = MaybeCreateOrTrans64To32bits(dev_ctx, index);
    phi::DenseTensor input_x = MaybeCreateOrTrans64To32bits(dev_ctx, x);
    phi::DenseTensor input_updates =
        MaybeCreateOrTrans64To32bits(dev_ctx, updates);
    phi::DenseTensor output =
        MaybeCreateOrTrans64To32bits(dev_ctx, *out, false);

    phi::DenseTensor tmp_copy_x = input_x;
    if (!overwrite) {
      auto zero_updates = TensorZeros(dev_ctx, input_updates.meta());
      phi::DenseTensor tmp_out = TensorEmpty(dev_ctx, output.meta());

      LAUNCH_TOPSATENOP(topsxlaScatter,
                        dev_ctx,
                        tmp_out,
                        input_x,
                        input_index,
                        zero_updates,
                        computation_type,
                        index_vector_dim,
                        update_window_dims,
                        inserted_window_dims,
                        scatter_dims_to_operand_dims,
                        indices_are_sorted,
                        unique_indices);

      computation_type = TOPSATEN_SCATTER_COMP_ADD;
      tmp_copy_x = tmp_out;
    }

    LAUNCH_TOPSATENOP(topsxlaScatter,
                      dev_ctx,
                      output,
                      tmp_copy_x,
                      input_index,
                      input_updates,
                      computation_type,
                      index_vector_dim,
                      update_window_dims,
                      inserted_window_dims,
                      scatter_dims_to_operand_dims,
                      indices_are_sorted,
                      unique_indices);

    MaybeTransResult(dev_ctx, output, out);

  } else {  // kernel impl base on JIT
    TensorNameMap input_names;
    input_names["X"] = {"x"};
    input_names["Ids"] = {"index"};
    input_names["Updates"] = {"updates"};

    TensorValueMap inputs;
    inputs["X"] = {const_cast<DenseTensor*>(&x)};
    inputs["Ids"] = {const_cast<DenseTensor*>(&index)};
    inputs["Updates"] = {const_cast<DenseTensor*>(&updates)};

    TensorNameMap output_names;
    output_names["Out"] = {"out"};

    TensorValueMap outputs;
    outputs["Out"] = {out};

    GcuAttributeMap attrs;
    attrs["overwrite"] = overwrite;

    GcuRunner(
        input_names, inputs, output_names, outputs, attrs, "scatter", dev_ctx);
  }
}
}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(scatter,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::ScatterKernel,
                          float,
                          int64_t,
                          int,
                          phi::dtype::float16) {}
