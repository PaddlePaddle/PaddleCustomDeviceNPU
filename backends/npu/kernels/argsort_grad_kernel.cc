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
#include "paddle/phi/backends/custom/custom_context.h"
#include "paddle/phi/kernels/funcs/tensor_formatter.h"

namespace custom_kernel {

template <typename T, typename Context>
void CastKernel(const Context& dev_ctx,
                const phi::DenseTensor& x,
                phi::DataType dtype,
                phi::DenseTensor* out);

template <typename T, typename Context>
void TransposeKernel(const Context& dev_ctx,
                     const phi::DenseTensor& x,
                     const std::vector<int>& axis,
                     phi::DenseTensor* out);

template <typename T, typename Context>
void AddKernel(const Context& dev_ctx,
               const phi::DenseTensor& x,
               const phi::DenseTensor& y,
               phi::DenseTensor* out);

template <typename Context, typename T>
static void TranposeNPU(const Context& dev_ctx,
                        const aclrtStream& stream,
                        std::vector<int64_t>* perm,
                        const phi::DenseTensor& in,
                        phi::DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);
  NpuOpRunner runner;
  runner.SetType("Transpose")
      .AddInput(in)
      .AddInput(dev_ctx, std::move(*perm))
      .AddOutput(*out)
      .Run(stream);
}

template <typename Context, typename T, typename Type>
static void AclopFullAssignNPU(const Context& dev_ctx,
                               const aclrtStream& stream,
                               const phi::DDim in_dims,
                               const phi::DenseTensor& input,
                               const phi::DenseTensor& indices,
                               phi::DenseTensor* t_out) {
  const int64_t input_height =
      phi::product(phi::slice_ddim(in_dims, 0, in_dims.size() - 1));
  const int64_t input_width = in_dims[in_dims.size() - 1];

  phi::DenseTensor input_tmp(input);
  input_tmp.Resize(
      phi::make_ddim(std::vector<int64_t>{input_height * input_width, 1}));

  phi::DenseTensor indices_tmp(indices);
  indices_tmp.Resize(
      phi::make_ddim(std::vector<int64_t>{input_height, input_width}));

  std::vector<int64_t> indexs_value;
  for (Type i = 0; i < input_height; i++) {
    indexs_value.push_back(i * input_width);
  }
  phi::DenseTensor indexs_tmp;
  phi::DenseTensorMeta indexs_tmp_meta = {
      indices.dtype(), phi::make_ddim(std::vector<int64_t>{input_height, 1})};
  indexs_tmp.set_meta(indexs_tmp_meta);
  dev_ctx.template Alloc<int64_t>(&indexs_tmp);
  TensorFromVector<int64_t>(dev_ctx, indexs_value, dev_ctx, &indexs_tmp);
  indexs_tmp.Resize(phi::make_ddim(std::vector<int64_t>{input_height, 1}));

  phi::DenseTensor indices_index;
  phi::DenseTensorMeta indices_index_meta = {indices.dtype(),
                                             indices_tmp.dims()};
  indices_index.set_meta(indices_index_meta);
  dev_ctx.template Alloc<int64_t>(&indices_index);

  const auto& runner_add =
      NpuOpRunner("AddV2", {indices_tmp, indexs_tmp}, {indices_index}, {});
  runner_add.Run(stream);

  indices_index.Resize(
      phi::make_ddim(std::vector<int64_t>{input_height * input_width, 1}));

  dev_ctx.template Alloc<T>(t_out);
  phi::DenseTensor out_tmp(*t_out);
  const auto& runner = NpuOpRunner("TensorScatterUpdate",
                                   {input_tmp, indices_index, input_tmp},
                                   {out_tmp},
                                   {});
  runner.Run(stream);
}

template <typename Context, typename T, typename Type>
static void FullAssignNPU(const Context& dev_ctx,
                          const phi::DDim in_dims,
                          const phi::DenseTensor& input,
                          const phi::DenseTensor& indices,
                          phi::DenseTensor* t_out) {
  DO_COMPATIBILITY(
      aclnnScatterNd,
      (custom_kernel::AclopFullAssignNPU<Context, T, Type>(
          dev_ctx, dev_ctx.stream(), in_dims, input, indices, t_out)));
  const int64_t input_height =
      phi::product(phi::slice_ddim(in_dims, 0, in_dims.size() - 1));
  const int64_t input_width = in_dims[in_dims.size() - 1];

  phi::DenseTensor input_tmp(input);
  input_tmp.Resize(
      phi::make_ddim(std::vector<int64_t>{input_height * input_width, 1}));

  phi::DenseTensor indices_tmp(indices);
  indices_tmp.Resize(
      phi::make_ddim(std::vector<int64_t>{input_height, input_width}));

  std::vector<int64_t> indexs_value;
  for (Type i = 0; i < input_height; i++) {
    indexs_value.push_back(i * input_width);
  }
  phi::DenseTensor indexs_tmp;
  phi::DenseTensorMeta indexs_tmp_meta = {
      indices.dtype(), phi::make_ddim(std::vector<int64_t>{input_height, 1})};
  indexs_tmp.set_meta(indexs_tmp_meta);
  dev_ctx.template Alloc<int64_t>(&indexs_tmp);
  TensorFromVector<int64_t>(dev_ctx, indexs_value, dev_ctx, &indexs_tmp);
  indexs_tmp.Resize(phi::make_ddim(std::vector<int64_t>{input_height, 1}));

  phi::DenseTensor indices_index;
  phi::DenseTensorMeta indices_index_meta = {indices.dtype(),
                                             indices_tmp.dims()};
  indices_index.set_meta(indices_index_meta);
  dev_ctx.template Alloc<int64_t>(&indices_index);
  custom_kernel::AddKernel<T, Context>(
      dev_ctx, indices_tmp, indexs_tmp, &indices_index);

  indices_index.Resize(
      phi::make_ddim(std::vector<int64_t>{input_height * input_width, 1}));

  phi::DenseTensor indices_index_int;
  phi::DenseTensorMeta meta = {phi::DataType::INT64, indices_index.dims()};
  indices_index_int.set_meta(meta);
  custom_kernel::CastKernel<T, Context>(
      dev_ctx, indices_index, phi::DataType::INT64, &indices_index_int);

  dev_ctx.template Alloc<T>(t_out);
  phi::DenseTensor out_tmp(*t_out);
  out_tmp.Resize(input_tmp.dims());
  EXEC_NPU_CMD(aclnnScatterNd,
               dev_ctx,
               input_tmp,
               indices_index_int,
               input_tmp,
               out_tmp);
  out_tmp.Resize(t_out->dims());
}

template <typename T, typename Context>
void AclopArgsortGradKernel(const Context& dev_ctx,
                            const phi::DenseTensor& indices,
                            const phi::DenseTensor& input,
                            const phi::DenseTensor& out_grad,
                            int axis,
                            bool descending,
                            bool stable,
                            phi::DenseTensor* in_grad) {
  auto stream = dev_ctx.stream();
  auto in_dims = indices.dims();
  auto rank = input.dims().size();
  axis = (axis < 0) ? (in_dims.size() + axis) : axis;
  dev_ctx.template Alloc<T>(in_grad);
  if (out_grad.numel() == 0) return;

  if (rank == 0) {
    phi::Copy<Context>(dev_ctx, out_grad, dev_ctx.GetPlace(), false, in_grad);
    return;
  }

  // Do full assign
  if (axis == -1 || axis + 1 == in_dims.size()) {
    AclopFullAssignNPU<Context, T, int64_t>(
        dev_ctx, stream, in_dims, out_grad, indices, in_grad);
  } else {
    std::vector<int64_t> perm;
    for (int64_t i = 0; i < in_dims.size(); i++) {
      perm.emplace_back(i);
    }
    std::swap(perm[axis], perm[in_dims.size() - 1]);

    std::vector<int64_t> shape;
    for (size_t i = 0; i < perm.size(); i++) {
      shape.emplace_back(in_dims[perm[i]]);
    }
    auto trans_dims = phi::make_ddim(shape);
    phi::DenseTensor trans_dout;
    phi::DenseTensor trans_ids;
    phi::DenseTensorMeta trans_dout_meta = {out_grad.dtype(), trans_dims};
    phi::DenseTensorMeta trans_ids_meta = {indices.dtype(), trans_dims};
    trans_dout.set_meta(trans_dout_meta);
    trans_ids.set_meta(trans_ids_meta);
    dev_ctx.template Alloc<T>(&trans_dout);
    dev_ctx.template Alloc<T>(&trans_ids);

    TranposeNPU<Context, T>(dev_ctx, stream, &perm, out_grad, &trans_dout);
    TranposeNPU<Context, int64_t>(dev_ctx, stream, &perm, indices, &trans_ids);

    phi::DenseTensor trans_dx;
    phi::DenseTensorMeta trans_dx_meta = {out_grad.dtype(), trans_dims};
    trans_dx.set_meta(trans_dx_meta);
    dev_ctx.template Alloc<T>(&trans_dx);

    AclopFullAssignNPU<Context, T, int64_t>(
        dev_ctx, stream, trans_dims, trans_dout, trans_ids, &trans_dx);

    TranposeNPU<Context, T>(dev_ctx, stream, &perm, trans_dx, in_grad);
  }
}

template <typename T, typename Context>
void ArgsortGradKernel(const Context& dev_ctx,
                       const phi::DenseTensor& indices,
                       const phi::DenseTensor& input,
                       const phi::DenseTensor& out_grad,
                       int axis,
                       bool descending,
                       bool stable,
                       phi::DenseTensor* in_grad) {
  auto stream = dev_ctx.stream();
  auto in_dims = indices.dims();
  auto rank = input.dims().size();
  axis = (axis < 0) ? (in_dims.size() + axis) : axis;
  dev_ctx.template Alloc<T>(in_grad);
  if (out_grad.numel() == 0) return;

  if (rank == 0) {
    phi::Copy<Context>(dev_ctx, out_grad, dev_ctx.GetPlace(), false, in_grad);
    return;
  }

  // Do full assign
  if (axis == -1 || axis + 1 == in_dims.size()) {
    // FullAssignNPU<Context, T, int64_t>(
    //     dev_ctx, in_dims, out_grad, indices, in_grad);
    AclopFullAssignNPU<Context, T, int64_t>(
        dev_ctx, stream, in_dims, out_grad, indices, in_grad);
  } else {
    std::vector<int> perm;
    for (int i = 0; i < in_dims.size(); i++) {
      perm.emplace_back(i);
    }
    std::swap(perm[axis], perm[in_dims.size() - 1]);

    std::vector<int64_t> shape;
    for (size_t i = 0; i < perm.size(); i++) {
      shape.emplace_back(in_dims[perm[i]]);
    }
    auto trans_dims = phi::make_ddim(shape);
    phi::DenseTensor trans_dout;
    phi::DenseTensor trans_ids;
    phi::DenseTensorMeta trans_dout_meta = {out_grad.dtype(), trans_dims};
    phi::DenseTensorMeta trans_ids_meta = {indices.dtype(), trans_dims};
    trans_dout.set_meta(trans_dout_meta);
    trans_ids.set_meta(trans_ids_meta);
    dev_ctx.template Alloc<T>(&trans_dout);
    dev_ctx.template Alloc<T>(&trans_ids);

    custom_kernel::TransposeKernel<T, Context>(
        dev_ctx, out_grad, perm, &trans_dout);
    custom_kernel::TransposeKernel<int64_t, Context>(
        dev_ctx, indices, perm, &trans_ids);

    phi::DenseTensor trans_dx;
    phi::DenseTensorMeta trans_dx_meta = {out_grad.dtype(), trans_dims};
    trans_dx.set_meta(trans_dx_meta);
    dev_ctx.template Alloc<T>(&trans_dx);

    // FullAssignNPU<Context, T, int64_t>(
    //     dev_ctx, trans_dims, trans_dout, trans_ids, &trans_dx);

    AclopFullAssignNPU<Context, T, int64_t>(
        dev_ctx, stream, trans_dims, trans_dout, trans_ids, &trans_dx);

    custom_kernel::TransposeKernel<T, Context>(
        dev_ctx, trans_dx, perm, in_grad);
  }
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(argsort_grad,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::ArgsortGradKernel,
                          float,
                          int64_t,
                          phi::dtype::float16) {}
