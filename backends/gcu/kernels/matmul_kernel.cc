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
void AdjustStrides(phi::DenseTensor& tensor) {  // NOLINT
  size_t rank = tensor.dims().size();
  if (rank <= 1) {
    return;
  }
  auto meta = tensor.meta();
  meta.strides = meta.calc_strides(meta.dims);
  std::swap(meta.dims[rank - 1], meta.dims[rank - 2]);
  std::swap(meta.strides[rank - 1], meta.strides[rank - 2]);
  tensor.set_meta(meta);
}

template <typename T, typename Context>
void MatmulKernel(const Context& dev_ctx,
                  const phi::DenseTensor& x,
                  const phi::DenseTensor& y,
                  bool trans_x,
                  bool trans_y,
                  phi::DenseTensor* out) {
  PADDLE_GCU_KERNEL_TRACE("matmul");
  dev_ctx.template Alloc<T>(out);

  if (LaunchAOTKernel()) {
    phi::DenseTensor input_x = x;
    phi::DenseTensor input_y = y;
    if (trans_x) {
      AdjustStrides(input_x);
    }
    if (trans_y) {
      AdjustStrides(input_y);
    }
    LAUNCH_TOPSATENOP(topsatenMatmul, dev_ctx, *out, input_x, input_y);

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
    attrs["trans_x"] = trans_x;
    attrs["trans_y"] = trans_y;

    GcuRunner(input_names,
              inputs,
              output_names,
              outputs,
              attrs,
              "matmul_v2",
              dev_ctx);
  }
}

template <typename T, typename Context>
void MatmulGradKernel(const Context& dev_ctx,
                      const phi::DenseTensor& x,
                      const phi::DenseTensor& y,
                      const phi::DenseTensor& dout,
                      bool trans_x,
                      bool trans_y,
                      phi::DenseTensor* dx,
                      phi::DenseTensor* dy) {
  PADDLE_GCU_KERNEL_TRACE("matmul_grad");
  if (LaunchAOTKernel()) {
    THROW_AOT_UNIMPLEMENTED();
  } else {  // kernel impl base on JIT
    TensorNameMap input_names;
    input_names["X"] = {"x"};
    input_names["Y"] = {"y"};
    input_names[GradVarName("Out")] = {"dout"};

    TensorValueMap inputs;
    inputs["X"] = {const_cast<DenseTensor*>(&x)};
    inputs["Y"] = {const_cast<DenseTensor*>(&y)};
    inputs[GradVarName("Out")] = {const_cast<DenseTensor*>(&dout)};

    TensorNameMap output_names;
    TensorValueMap outputs;
    if (dx) {
      dev_ctx.template Alloc<T>(dx);
      output_names[GradVarName("X")] = {"dx"};
      outputs[GradVarName("X")] = {dx};
    }
    if (dy) {
      dev_ctx.template Alloc<T>(dy);
      output_names[GradVarName("Y")] = {"dy"};
      outputs[GradVarName("Y")] = {dy};
    }

    GcuAttributeMap attrs;
    attrs["trans_x"] = trans_x;
    attrs["trans_y"] = trans_y;

    GcuRunner(input_names,
              inputs,
              output_names,
              outputs,
              attrs,
              "matmul_v2_grad",
              dev_ctx);
  }
}

template <typename T, typename Context>
void MatmulWithFlattenKernel(const Context& dev_ctx,
                             const phi::DenseTensor& x,
                             const phi::DenseTensor& y,
                             int x_num_col_dims,
                             int y_num_col_dims,
                             phi::DenseTensor* out) {
  const phi::DenseTensor x_matrix =
      x.dims().size() > 2 ? phi::ReshapeToMatrix(x, x_num_col_dims) : x;
  const phi::DenseTensor y_matrix =
      y.dims().size() > 2 ? phi::ReshapeToMatrix(y, y_num_col_dims) : y;

  dev_ctx.template Alloc<T>(out);
  auto z_dim = out->dims();
  if (z_dim.size() != 2) {
    out->Resize({x_matrix.dims()[0], y_matrix.dims()[1]});
  }

  LAUNCH_TOPSATENOP(topsatenMatmul, dev_ctx, *out, x_matrix, y_matrix);
  if (z_dim.size() != 2) {
    out->Resize(z_dim);
  }
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(matmul,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::MatmulKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(matmul_grad,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::MatmulGradKernel,
                          float,
                          phi::dtype::float16,
                          double) {}

PD_REGISTER_PLUGIN_KERNEL(matmul_with_flatten,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::MatmulWithFlattenKernel,
                          float,
                          phi::dtype::float16) {}
