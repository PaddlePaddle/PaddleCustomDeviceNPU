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
void AclopIndexSelectNPUKernel(const Context& dev_ctx,
                               const phi::DenseTensor& x,
                               const phi::DenseTensor& index,
                               int dim,
                               phi::DenseTensor* output) {
  dev_ctx.template Alloc<T>(output);

  auto stream = dev_ctx.stream();

  if (x.dtype() == phi::DataType::FLOAT64) {
    auto op_func = [](const std::vector<phi::DenseTensor>& inputs,
                      const std::vector<phi::DenseTensor>& outputs,
                      const NPUAttributeMap& attrs,
                      const phi::CustomContext& dev_ctx,
                      const auto& host_vecs) {
      NpuOpRunner runner;
      runner.SetType("GatherV2")
          .AddInput(inputs[0])
          .AddInput(inputs[1])
          .AddInput(dev_ctx, std::move(host_vecs[0]))
          .AddOutput(outputs[0])
          .AddAttrs(attrs);
      runner.Run(dev_ctx.stream());
    };

    NpuOpRunner::TypeAdapter<int32_t>({x, index},
                                      {*output},
                                      {},
                                      dev_ctx,
                                      op_func,
                                      {phi::DataType::FLOAT32, index.dtype()},
                                      {phi::DataType::FLOAT32},
                                      {std::vector<int32_t>({dim})});
  } else {
    NpuOpRunner runner;
    runner.SetType("GatherV2")
        .AddInput(x)
        .AddInput(index)
        .AddInput(dev_ctx, std::vector<int32_t>{dim})
        .AddOutput(*output);
    runner.Run(stream);
  }
}

template <typename T, typename Context>
void IndexSelectNPUKernel(const Context& dev_ctx,
                          const phi::DenseTensor& x,
                          const phi::DenseTensor& index,
                          int dim,
                          phi::DenseTensor* output) {
  DO_COMPATIBILITY(aclnnIndexSelect,
                   (custom_kernel::AclopIndexSelectNPUKernel<T, Context>(
                       dev_ctx, x, index, dim, output)));

  dev_ctx.template Alloc<T>(output);
  int64_t dim_ = dim;

  if (x.place() == index.place()) {
    EXEC_NPU_CMD(aclnnIndexSelect, dev_ctx, x, dim_, index, *output);
  } else {
    phi::DenseTensor index_copy;
    phi::DenseTensorMeta index_copy_meta = {index.dtype(), index.dims()};
    index_copy.set_meta(index_copy_meta);
    TensorCopy(dev_ctx, index, false, &index_copy, x.place());
    EXEC_NPU_CMD(aclnnIndexSelect, dev_ctx, x, dim_, index_copy, *output);
  }
}

template <typename T, typename Context>
void IndexSelectGradNPUKernel(const Context& dev_ctx,
                              const phi::DenseTensor& x,
                              const phi::DenseTensor& index,
                              const phi::DenseTensor& out_grad,
                              int dim,
                              phi::DenseTensor* x_grad) {
  auto stream = dev_ctx.stream();

  auto x_dims = x_grad->dims();
  auto out_dims = out_grad.dims();

  if (dim < 0) {
    dim += out_dims.size();
  }

  phi::DenseTensor casted_index;
  if (index.dtype() != phi::DataType::INT32) {
    casted_index.Resize(index.dims());
    dev_ctx.template Alloc<int32_t>(&casted_index);

    const auto& cast_runner =
        NpuOpRunner("Cast", {index}, {casted_index}, {{"dst_type", ACL_INT32}});
    cast_runner.Run(stream);
  } else {
    casted_index = index;
  }

  auto op_func = [](const std::vector<phi::DenseTensor>& inputs,
                    const std::vector<phi::DenseTensor>& outputs,
                    const NPUAttributeMap& attrs,
                    const phi::CustomContext& dev_ctx,
                    const auto& host_vecs) {
    NpuOpRunner runner;
    runner.SetType("UnsortedSegmentSum")
        .AddInput(inputs[0])
        .AddInput(inputs[1])
        .AddInput(dev_ctx, std::move(host_vecs[0]))
        .AddOutput(outputs[0])
        .AddAttrs(attrs);
    runner.Run(dev_ctx.stream());
  };

  if (dim == 0) {
    dev_ctx.template Alloc<T>(x_grad);
    const auto& zeros_runner = NpuOpRunner("ZerosLike", {*x_grad}, {*x_grad});
    zeros_runner.Run(stream);

    if (x.dtype() == phi::DataType::FLOAT64) {
      NpuOpRunner::TypeAdapter<int64_t>(
          {out_grad, casted_index},
          {*x_grad},
          {},
          dev_ctx,
          op_func,
          {phi::DataType::FLOAT32, casted_index.dtype()},
          {phi::DataType::FLOAT32},
          {std::vector<int64_t>({x_dims[dim]})});
    } else {
      NpuOpRunner runner;
      runner.SetType("UnsortedSegmentSum")
          .AddInput(out_grad)
          .AddInput(casted_index)
          .AddInput(dev_ctx, std::vector<int64_t>{x_dims[dim]})
          .AddOutput(*x_grad);
      runner.Run(stream);
    }
  } else {
    phi::DenseTensor transed_out_grad;
    std::vector<int> in_trans_perm;
    in_trans_perm.push_back(dim);
    for (int i = 0; i < out_dims.size(); ++i) {
      if (i == dim) continue;
      in_trans_perm.push_back(i);
    }
    phi::DDim transed_out_dims(out_dims);
    for (size_t i = 0; i < in_trans_perm.size(); ++i) {
      transed_out_dims[i] = out_dims[in_trans_perm[i]];
    }
    transed_out_grad.Resize(transed_out_dims);
    dev_ctx.template Alloc<T>(&transed_out_grad);
    NpuOpRunner in_trans_runner;
    in_trans_runner.SetType("Transpose")
        .AddInput(out_grad)
        .AddInput(dev_ctx, std::move(in_trans_perm))
        .AddOutput(transed_out_grad);
    in_trans_runner.Run(stream);

    phi::DenseTensor sum_out;
    phi::DDim sum_dims(x_dims);
    sum_dims[0] = x_dims[dim];
    auto idx = 1;
    for (int i = 0; i < x_dims.size(); ++i) {
      if (i == dim) continue;
      sum_dims[idx++] = x_dims[i];
    }
    sum_out.Resize(sum_dims);
    dev_ctx.template Alloc<T>(&sum_out);
    const auto& zeros_runner = NpuOpRunner("ZerosLike", {sum_out}, {sum_out});
    zeros_runner.Run(stream);

    if (x.dtype() == phi::DataType::FLOAT64) {
      NpuOpRunner::TypeAdapter<int64_t>(
          {transed_out_grad, casted_index},
          {sum_out},
          {},
          dev_ctx,
          op_func,
          {phi::DataType::FLOAT32, casted_index.dtype()},
          {phi::DataType::FLOAT32},
          {std::vector<int64_t>({x_dims[dim]})});
    } else {
      NpuOpRunner runner;
      runner.SetType("UnsortedSegmentSum")
          .AddInput(transed_out_grad)
          .AddInput(casted_index)
          .AddInput(dev_ctx, std::vector<int64_t>{x_dims[dim]})
          .AddOutput(sum_out);
      runner.Run(stream);
    }

    std::vector<int> out_trans_perm;
    for (int i = 1; i < 1 + dim; ++i) {
      out_trans_perm.push_back(i);
    }
    out_trans_perm.push_back(0);
    for (int i = 1 + dim; i < x_dims.size(); ++i) {
      out_trans_perm.push_back(i);
    }
    dev_ctx.template Alloc<T>(x_grad);
    NpuOpRunner out_trans_runner;
    out_trans_runner.SetType("Transpose")
        .AddInput(sum_out)
        .AddInput(dev_ctx, std::move(out_trans_perm))
        .AddOutput(*x_grad);
    out_trans_runner.Run(stream);
  }
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(index_select,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::IndexSelectNPUKernel,
                          float,
                          double,
                          int,
                          int64_t,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(index_select_grad,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::IndexSelectGradNPUKernel,
                          float,
                          double,
                          int,
                          int64_t,
                          phi::dtype::float16) {}
