
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

#include "kernels/funcs/npu_funcs.h"
#include "kernels/funcs/npu_op_runner.h"

namespace custom_kernel {

template <typename T, typename Context>
void ScatterKernel(const Context& dev_ctx,
                   const phi::DenseTensor& x,
                   const phi::DenseTensor& index,
                   const phi::DenseTensor& updates,
                   bool overwrite,
                   phi::DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);

  phi::DenseTensor tmp_tensor(index);
  const auto index_dims = index.dims();
  if (index_dims.size() == 1 || index_dims.size() == 0) {
    std::vector<int64_t> new_dim = {index_dims.size() == 0 ? 1 : index_dims[0],
                                    1};
    tmp_tensor.Resize(phi::make_ddim(new_dim));
  }

  auto op_func_update = [](const std::vector<phi::DenseTensor>& inputs,
                           const std::vector<phi::DenseTensor>& outputs,
                           const NPUAttributeMap& attrs,
                           const Context& dev_ctx) {
    const auto& runner =
        NpuOpRunner("TensorScatterUpdate", inputs, outputs, attrs);
    runner.Run(dev_ctx.stream());
  };
  auto op_func_add = [](const std::vector<phi::DenseTensor>& inputs,
                        const std::vector<phi::DenseTensor>& outputs,
                        const NPUAttributeMap& attrs,
                        const Context& dev_ctx) {
    const auto& runner =
        NpuOpRunner("TensorScatterAdd", inputs, outputs, attrs);
    runner.Run(dev_ctx.stream());
  };

  if (overwrite) {
    if (x.dtype() == phi::DataType::INT64) {
      NpuOpRunner::TypeAdapter(
          {x, tmp_tensor, updates},
          {*out},
          {},
          dev_ctx,
          op_func_update,
          {phi::DataType::INT32, phi::DataType::INT32, phi::DataType::INT32},
          {phi::DataType::INT32});
    } else {
      const auto& runner_update = NpuOpRunner(
          "TensorScatterUpdate", {x, tmp_tensor, updates}, {*out}, {});
      runner_update.Run(dev_ctx.stream());
    }
  } else {
    if (x.dtype() == phi::DataType::INT64) {
      NpuOpRunner::TypeAdapter(
          {x, tmp_tensor, updates},
          {*out},
          {},
          dev_ctx,
          op_func_add,
          {phi::DataType::INT32, phi::DataType::INT32, phi::DataType::INT32},
          {phi::DataType::INT32});
    } else {
      const auto& runner_add =
          NpuOpRunner("TensorScatterAdd", {x, tmp_tensor, updates}, {*out}, {});
      runner_add.Run(dev_ctx.stream());
    }
  }
}

template <typename T, typename Context>
void ScatterGradKernel(const Context& dev_ctx,
                       const phi::DenseTensor& index,
                       const phi::DenseTensor& updates,
                       const phi::DenseTensor& out_grad,
                       bool overwrite,
                       phi::DenseTensor* x_grad,
                       phi::DenseTensor* updates_grad) {
  phi::DenseTensor tmp_tensor(index);
  const auto index_dims = index.dims();
  if (index_dims.size() == 1 || index_dims.size() == 0) {
    std::vector<int64_t> new_dim = {index_dims.size() == 0 ? 1 : index_dims[0],
                                    1};
    tmp_tensor.Resize(phi::make_ddim(new_dim));
  }
  const auto& index_type = index.dtype();
  bool index_type_match =
      index_type == phi::DataType::INT32 || index_type == phi::DataType::INT64;
  PADDLE_ENFORCE_EQ(index_type_match,
                    true,
                    phi::errors::InvalidArgument(
                        "scatter_op index holds the wrong type, it holds [%s],"
                        "but desires to be [%s] or [%s]",
                        index_type,
                        phi::DataType::INT32,
                        phi::DataType::INT64));
  if (x_grad) {
    dev_ctx.template Alloc<T>(x_grad);
    TensorCopy(dev_ctx, out_grad, true, x_grad);
    // zeros like
    phi::DenseTensor zeroslike_xout;
    phi::DenseTensorMeta meta = {updates.dtype(), updates.dims()};
    zeroslike_xout.set_meta(meta);
    dev_ctx.template Alloc<T>(&zeroslike_xout);

    const auto& runner_tensor_zeros =
        NpuOpRunner("ZerosLike", {zeroslike_xout}, {zeroslike_xout}, {});
    runner_tensor_zeros.Run(dev_ctx.stream());
    const auto& runner_add = NpuOpRunner("TensorScatterUpdate",
                                         {out_grad, tmp_tensor, zeroslike_xout},
                                         {*x_grad},
                                         {});
    runner_add.Run(dev_ctx.stream());
  }
  if (updates_grad) {
    dev_ctx.template Alloc<T>(updates_grad);
    NpuOpRunner runner;
    runner.SetType("GatherV2")
        .AddInput(out_grad)
        .AddInput(index)
        .AddInput(dev_ctx, std::vector<int32_t>({0}))
        .AddOutput(*updates_grad);
    runner.Run(dev_ctx.stream());
  }
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(scatter,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::ScatterKernel,
                          float,
                          int64_t,
                          int,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}

PD_REGISTER_PLUGIN_KERNEL(scatter_grad,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::ScatterGradKernel,
                          float,
                          int64_t,
                          int,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}
