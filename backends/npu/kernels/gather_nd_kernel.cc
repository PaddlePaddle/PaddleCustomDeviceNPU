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
void AclopGatherNdKernel(const Context &dev_ctx,
                         const phi::DenseTensor &x,
                         const phi::DenseTensor &index,
                         phi::DenseTensor *out) {
  dev_ctx.template Alloc<T>(out);
  auto stream = dev_ctx.stream();

  if (x.numel() == 0) return;

  if (index.numel() == 0) {
    int diff = out->dims().size() - x.dims().size();
    if (diff == 0) {
      TensorCopy(dev_ctx, x, false, out);
    } else {
      std::vector<int64_t> new_dims(diff, 1);
      for (size_t i = 0; i < x.dims().size(); ++i) {
        new_dims.emplace_back(x.dims()[i]);
      }

      phi::DenseTensor x_tmp(x);
      x_tmp.Resize(phi::make_ddim(new_dims));

      NpuOpRunner runner;
      runner.SetType("BroadcastTo")
          .AddInput(x_tmp)
          .AddInput(dev_ctx, phi::vectorize(out->dims()))
          .AddOutput(*out);
      runner.Run(stream);
    }
    return;
  }

  const auto &index_type = index.dtype();
  bool index_type_match =
      index_type == phi::DataType::INT32 || index_type == phi::DataType::INT64;
  PADDLE_ENFORCE_EQ(
      index_type_match,
      true,
      phi::errors::InvalidArgument("Index holds the wrong type, it holds [%s],"
                                   "but desires to be [%s] or [%s]",
                                   index_type,
                                   phi::DataType::INT32,
                                   phi::DataType::INT64));

  const auto &runner = NpuOpRunner("GatherNd", {x, index}, {*out}, {});
  runner.Run(stream);
}

template <typename T, typename Context>
void GatherNdKernel(const Context &dev_ctx,
                    const phi::DenseTensor &x,
                    const phi::DenseTensor &index,
                    phi::DenseTensor *out) {
  DO_COMPATIBILITY(
      aclnnGatherNd,
      (custom_kernel::AclopGatherNdKernel<T, Context>(dev_ctx, x, index, out)));

  if (x.numel() == 0) return;
  dev_ctx.template Alloc<T>(out);

  if (index.numel() == 0) {
    int diff = out->dims().size() - x.dims().size();
    if (diff == 0) {
      TensorCopy(dev_ctx, x, false, out);
    } else {
      std::vector<int64_t> new_dims(diff, 1);
      for (size_t i = 0; i < x.dims().size(); ++i) {
        new_dims.emplace_back(x.dims()[i]);
      }

      phi::DenseTensor x_tmp(x);
      x_tmp.Resize(phi::make_ddim(new_dims));

      auto out_dims = phi::vectorize(out->dims());
      EXEC_NPU_CMD(aclnnExpand, dev_ctx, x_tmp, out_dims, *out);
    }
    return;
  }

  const auto &index_type = index.dtype();
  bool index_type_match =
      index_type == phi::DataType::INT32 || index_type == phi::DataType::INT64;
  PADDLE_ENFORCE_EQ(
      index_type_match,
      true,
      phi::errors::InvalidArgument("Index holds the wrong type, it holds [%s],"
                                   "but desires to be [%s] or [%s]",
                                   index_type,
                                   phi::DataType::INT32,
                                   phi::DataType::INT64));

  bool negativeIndexSupport = false;
  EXEC_NPU_CMD(aclnnGatherNd, dev_ctx, x, index, negativeIndexSupport, *out);
}

template <typename T, typename Context>
void GatherNdGradKernel(const Context &dev_ctx,
                        const phi::DenseTensor &x,
                        const phi::DenseTensor &index,
                        const phi::DenseTensor &dout,
                        phi::DenseTensor *dx) {
  auto x_dims = dx->dims();
  dev_ctx.template Alloc<T>(dx);
  auto stream = dev_ctx.stream();

  if (dx->numel() == 0) return;

  if (index.numel() == 0) {
    int diff = dout.dims().size() - x_dims.size();
    if (diff == 0) {
      TensorCopy(dev_ctx, dout, false, dx);
    } else {
      std::vector<int> axes;
      for (size_t i = 0; i < diff; ++i) {
        axes.push_back(i);
      }

      NpuOpRunner runner;
      runner.SetType("ReduceSum")
          .AddInput(dout)
          .AddInput(dev_ctx, std::move(axes))
          .AddOutput(*dx)
          .AddAttr("keep_dims", false);
      runner.Run(stream);
    }
    return;
  }

  const phi::DenseTensor *p_index = &index;
  const phi::DenseTensor *p_dout = &dout;
  phi::DenseTensor tmp_tensor(index);
  phi::DenseTensor tmp_tensor2(dout);
  const auto index_dims = index.dims();
  if (index_dims.size() == 1) {
    std::vector<int64_t> new_dim = {1, index_dims[0]};
    tmp_tensor.Resize(phi::make_ddim(new_dim));
    p_index = &tmp_tensor;

    std::vector<int64_t> new_dim2{1};
    for (int i = p_index->numel(); i < x.dims().size(); i++) {
      new_dim2.push_back(x.dims()[i]);
    }
    tmp_tensor2.Resize(phi::make_ddim(new_dim2));
    p_dout = &tmp_tensor2;
  }

  FillNpuTensorWithConstant<T>(dx, dev_ctx, static_cast<T>(0));
  dx->Resize(x_dims);

  const auto &runner_scatter = NpuOpRunner("ScatterNdAdd",
                                           {*dx, *p_index, *p_dout},
                                           {*dx},
                                           {{"use_locking", false}});
  runner_scatter.Run(stream);
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(gather_nd,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::GatherNdKernel,
                          int64_t,
                          int32_t,
                          float,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}

PD_REGISTER_PLUGIN_KERNEL(gather_nd_grad,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::GatherNdGradKernel,
                          int64_t,
                          int32_t,
                          float,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}
