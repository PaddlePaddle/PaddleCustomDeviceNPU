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
void CastKernel(const Context& dev_ctx,
                const phi::DenseTensor& x,
                phi::DataType dtype,
                phi::DenseTensor* out);

template <typename T, typename Context>
void AclopSoftmaxKernel(const Context& dev_ctx,
                        const phi::DenseTensor& x,
                        int axis,
                        phi::DenseTensor* out) {
  const int rank = x.dims().size();
  if (rank == 0) {
    dev_ctx.template Alloc<T>(out);
    auto out_dim = out->dims();
    FillNpuTensorWithConstant<T>(out, dev_ctx, static_cast<T>(1));
    out->Resize(out_dim);
    return;
  }

  std::vector<int> axes;
  axes.push_back(axis);
  NPUAttributeMap attr_input = {{"axes", axes}};
  dev_ctx.template Alloc<T>(out);
  const auto& runner = NpuOpRunner("SoftmaxV2", {x}, {*out}, attr_input);
  auto stream = dev_ctx.stream();
  runner.Run(stream);
}

template <typename T, typename Context>
void SoftmaxKernel(const Context& dev_ctx,
                   const phi::DenseTensor& x,
                   int axis,
                   phi::DenseTensor* out) {
  DO_COMPATIBILITY(
      aclnnSoftmax,
      (custom_kernel::AclopSoftmaxKernel<T, Context>(dev_ctx, x, axis, out)));
  dev_ctx.template Alloc<T>(out);
  int64_t dim = static_cast<int64_t>(axis);

  phi::DenseTensor x_trans(x);
  if (x.dims().size() == 0) {
    phi::DenseTensorMeta meta_1 = {x.dtype(), phi::make_ddim({1})};
    x_trans.set_meta(meta_1);
  }

  auto out_dims = out->dims();
  if (out_dims.size() == 0) {
    out->Resize(phi::make_ddim({1}));
  }

  EXEC_NPU_CMD(aclnnSoftmax, dev_ctx, x_trans, dim, *out);

  if (out_dims.size() == 0) {
    out->Resize(out_dims);
  }
}

template <typename T, typename Context>
void AclopSoftmaxGradKernel(const Context& dev_ctx,
                            const phi::DenseTensor& out,
                            const phi::DenseTensor& out_grad,
                            int axis,
                            phi::DenseTensor* x_grad) {
  auto dims = x_grad->dims();
  const int rank = dims.size();
  if (out.dims().size() == 0) {
    dev_ctx.template Alloc<T>(x_grad);
    auto x_grad_dim = x_grad->dims();
    FillNpuTensorWithConstant<T>(x_grad, dev_ctx, static_cast<T>(0));
    x_grad->Resize(x_grad_dim);
    return;
  }

  axis = custom_kernel::CanonicalAxis(axis, rank);
  int64_t first_dim = 1;
  int64_t sec_dim = 1;
  for (int i = 0; i < axis; i++) {
    first_dim *= dims[i];
  }
  for (int i = axis; i < rank; i++) {
    sec_dim *= dims[i];
  }
  auto stream = dev_ctx.stream();
  NPUAttributeMap attr_input = {};
  phi::DenseTensor tmp_out, tmp_out_grad;
  if (out.dtype() == phi::DataType::FLOAT64) {
    phi::DenseTensorMeta tmp_out_meta = {phi::DataType::FLOAT32, out.dims()};
    phi::DenseTensorMeta tmp_out_grad_meta = {phi::DataType::FLOAT32,
                                              out_grad.dims()};
    tmp_out.set_meta(tmp_out_meta);
    tmp_out_grad.set_meta(tmp_out_grad_meta);
    dev_ctx.template Alloc<float>(&tmp_out);
    dev_ctx.template Alloc<float>(&tmp_out_grad);
    const auto& cast_runner1 =
        NpuOpRunner("Cast", {out}, {tmp_out}, {{"dst_type", ACL_FLOAT}});
    cast_runner1.Run(stream);
    const auto& cast_runner2 = NpuOpRunner(
        "Cast", {out_grad}, {tmp_out_grad}, {{"dst_type", ACL_FLOAT}});
    cast_runner2.Run(stream);

    phi::DenseTensor tmp_x_grad;
    tmp_x_grad.Resize(phi::make_ddim({first_dim, sec_dim}));
    dev_ctx.template Alloc<float>(&tmp_x_grad);

    const auto& runner = NpuOpRunner(std::string("SoftmaxGrad"),
                                     {tmp_out, tmp_out_grad},
                                     {tmp_x_grad},
                                     attr_input);
    runner.Run(stream);

    x_grad->Resize(phi::make_ddim({first_dim, sec_dim}));
    dev_ctx.template Alloc<T>(x_grad);
    const auto& cast_runner3 = NpuOpRunner(
        "Cast", {tmp_x_grad}, {*x_grad}, {{"dst_type", ACL_DOUBLE}});
    cast_runner3.Run(stream);
  } else {
    tmp_out = out;
    tmp_out.Resize({first_dim, sec_dim});
    tmp_out_grad = out_grad;
    tmp_out_grad.Resize({first_dim, sec_dim});

    x_grad->Resize(phi::make_ddim({first_dim, sec_dim}));
    dev_ctx.template Alloc<T>(x_grad);

    const auto& runner = NpuOpRunner(std::string("SoftmaxGrad"),
                                     {tmp_out, tmp_out_grad},
                                     {*x_grad},
                                     attr_input);
    runner.Run(stream);
  }

  x_grad->Resize(dims);
}

template <typename T, typename Context>
void SoftmaxGradKernel(const Context& dev_ctx,
                       const phi::DenseTensor& out,
                       const phi::DenseTensor& out_grad,
                       int axis,
                       phi::DenseTensor* x_grad) {
  DO_COMPATIBILITY(aclnnSoftmax,
                   (custom_kernel::AclopSoftmaxGradKernel<T, Context>(
                       dev_ctx, out, out_grad, axis, x_grad)));
  dev_ctx.template Alloc<T>(x_grad);
  int64_t dim = static_cast<int64_t>(axis);

  phi::DenseTensor x_trans(out_grad);
  if (out_grad.dims().size() == 0) {
    phi::DenseTensorMeta meta_1 = {out_grad.dtype(), phi::make_ddim({1})};
    x_trans.set_meta(meta_1);
  }

  phi::DenseTensor cast_x;
  if (x_trans.dtype() == phi::DataType::FLOAT64) {
    phi::DenseTensorMeta meta(x_trans.meta());
    meta.dtype = phi::DataType::FLOAT32;
    cast_x.set_meta(meta);

    custom_kernel::CastKernel<T, Context>(
        dev_ctx, x_trans, phi::DataType::FLOAT32, &cast_x);
  } else {
    cast_x = x_trans;
  }

  phi::DenseTensor y_trans(out);
  if (out.dims().size() == 0) {
    phi::DenseTensorMeta meta_1 = {out.dtype(), phi::make_ddim({1})};
    y_trans.set_meta(meta_1);
  }

  phi::DenseTensor cast_y;
  if (y_trans.dtype() == phi::DataType::FLOAT64) {
    phi::DenseTensorMeta meta(y_trans.meta());
    meta.dtype = phi::DataType::FLOAT32;
    cast_y.set_meta(meta);

    custom_kernel::CastKernel<T, Context>(
        dev_ctx, y_trans, phi::DataType::FLOAT32, &cast_y);
  } else {
    cast_y = y_trans;
  }

  auto x_grad_dims = x_grad->dims();
  if (x_grad_dims.size() == 0) {
    x_grad->Resize(phi::make_ddim({1}));
  }

  EXEC_NPU_CMD(aclnnSoftmaxBackward, dev_ctx, cast_x, cast_y, dim, *x_grad);

  if (x_grad_dims.size() == 0) {
    x_grad->Resize(x_grad_dims);
  }
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(softmax,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::SoftmaxKernel,
                          float,
                          double,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}

PD_REGISTER_PLUGIN_KERNEL(softmax_grad,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::SoftmaxGradKernel,
                          float,
                          double,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}
