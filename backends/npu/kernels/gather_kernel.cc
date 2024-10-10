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
void AclopGatherKernel(const Context& dev_ctx,
                       const phi::DenseTensor& x,
                       const phi::DenseTensor& index,
                       const phi::Scalar& axis,
                       phi::DenseTensor* out) {
  auto stream = dev_ctx.stream();
  dev_ctx.template Alloc<T>(out);
  if (x.dtype() == phi::DataType::BOOL) {
    phi::DenseTensor trans_x, trans_y;
    phi::DenseTensorMeta trans_x_meta = {phi::DataType::UINT8, x.dims()};
    trans_x.set_meta(trans_x_meta);
    dev_ctx.template Alloc<uint8_t>(&trans_x);
    trans_y.Resize(out->dims());
    dev_ctx.template Alloc<uint8_t>(&trans_y);
    const auto& cast_runner1 =
        NpuOpRunner("Cast", {x}, {trans_x}, {{"dst_type", ACL_UINT8}});
    cast_runner1.Run(stream);
    NpuOpRunner gather_runner;
    gather_runner.SetType("GatherV2")
        .AddInput(trans_x)
        .AddInput(index)
        .AddInput(dev_ctx, std::vector<int32_t>({axis.to<int32_t>()}))
        .AddOutput(trans_y);
    gather_runner.Run(stream);
    const auto& cast_runner2 =
        NpuOpRunner("Cast", {trans_y}, {*out}, {{"dst_type", ACL_BOOL}});
    cast_runner2.Run(stream);
  } else {
    NpuOpRunner runner;
    runner.SetType("GatherV2")
        .AddInput(x)
        .AddInput(index)
        .AddInput(dev_ctx, std::vector<int32_t>({axis.to<int32_t>()}))
        .AddOutput(*out);
    runner.Run(stream);
  }
}

template <typename T, typename Context>
void GatherKernel(const Context& dev_ctx,
                  const phi::DenseTensor& x,
                  const phi::DenseTensor& index,
                  const phi::Scalar& axis,
                  phi::DenseTensor* out) {
  DO_COMPATIBILITY(aclnnGatherV2,
                   (custom_kernel::AclopGatherKernel<T, Context>(
                       dev_ctx, x, index, axis, out)));

  int64_t dim = axis.to<int64_t>();

  auto out_dims = out->dims();
  bool zero_index = false;
  int count = 0;
  if (index.dims().size() == 0) {
    std::vector<int64_t> out_dims_new(out_dims.size() + 1);
    for (int64_t i = 0; i <= out_dims_new.size() - 1; i++) {
      if (i == dim) {
        out_dims_new[i] = 1;
      } else {
        out_dims_new[i] = out_dims[count];
        count++;
      }
    }

    phi::DenseTensorMeta meta_1 = {x.dtype(), phi::make_ddim(out_dims_new)};
    out->set_meta(meta_1);
    zero_index = true;
  }

  dev_ctx.template Alloc<T>(out);

  auto index_shape_vec = phi::vectorize(index.dims());
  if (index_shape_vec.size() == 2 && index_shape_vec[1] == 1) {
    const phi::DenseTensor* p_index = &index;
    phi::DenseTensor tmp_tensor(index);
    const auto index_dims = index.dims();
    std::vector<int64_t> new_dim = {index_dims[0]};
    tmp_tensor.Resize(phi::make_ddim(new_dim));
    p_index = &tmp_tensor;
    EXEC_NPU_CMD(aclnnGatherV2, dev_ctx, x, dim, *p_index, *out);
  } else {
    EXEC_NPU_CMD(aclnnGatherV2, dev_ctx, x, dim, index, *out);
  }

  if (zero_index) {
    phi::DenseTensorMeta meta_0 = {x.dtype(), out_dims};
    out->set_meta(meta_0);
  }
}

template <typename T, typename Context>
void GatherGradKernel(const Context& dev_ctx,
                      const phi::DenseTensor& x,
                      const phi::DenseTensor& index,
                      const phi::DenseTensor& out_grad,
                      const phi::Scalar& axis,
                      phi::DenseTensor* x_grad) {
  dev_ctx.template Alloc<T>(x_grad);

  const phi::DenseTensor* p_index = &index;
  // step1: Unsqueeze index
  phi::DenseTensor tmp_tensor(index);
  const auto index_dims = index.dims();
  if (index_dims.size() == 1 || index_dims.size() == 0) {
    std::vector<int64_t> new_dim = {index_dims.size() == 0 ? 1 : index_dims[0],
                                    1};
    tmp_tensor.Resize(phi::make_ddim(new_dim));
    p_index = &tmp_tensor;
  }

  auto stream = dev_ctx.stream();

  // step2: ZerosLike x in device
  phi::DenseTensor zeroslike_xout;
  phi::DenseTensorMeta meta = {x_grad->dtype(), x.dims()};
  zeroslike_xout.set_meta(meta);
  dev_ctx.template Alloc<T>(&zeroslike_xout);

  const auto& runner_tensor_zeros =
      NpuOpRunner("ZerosLike", {*x_grad}, {zeroslike_xout}, {});
  runner_tensor_zeros.Run(stream);
  zeroslike_xout.Resize(x.dims());

  // step3: scatter(x_grad)
  phi::DenseTensor out_grad_(out_grad);

  int64_t dim = axis.to<int64_t>();
  int count = 0;
  if (index.dims().size() == 0) {
    std::vector<int64_t> out_dims_new(out_grad.dims().size() + 1);
    for (int64_t i = 0; i <= out_dims_new.size() - 1; i++) {
      if (i == dim) {
        out_dims_new[i] = 1;
      } else {
        out_dims_new[i] = out_grad.dims()[count];
        count++;
      }
    }

    phi::DenseTensorMeta meta_1 = {x.dtype(), phi::make_ddim(out_dims_new)};
    out_grad_.set_meta(meta_1);
  }

  EXEC_NPU_CMD(
      aclnnScatterNd, dev_ctx, zeroslike_xout, *p_index, out_grad_, *x_grad);
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(gather,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::GatherKernel,
                          bool,
                          float,
                          double,
                          int32_t,
                          int64_t,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(gather_grad,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::GatherGradKernel,
                          bool,
                          float,
                          double,
                          int32_t,
                          int64_t,
                          phi::dtype::float16) {}
