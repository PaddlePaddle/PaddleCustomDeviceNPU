// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
#include "kernels/funcs/slice_utils.h"

namespace custom_kernel {
template <typename T, typename Context>
void CastKernel(const Context& dev_ctx,
                const phi::DenseTensor& x,
                phi::DataType dtype,
                phi::DenseTensor* out);

template <typename T, typename Context>
void NllLossRawKernel(const Context& dev_ctx,
                      const phi::DenseTensor& x,
                      const phi::DenseTensor& labels,
                      const paddle::optional<phi::DenseTensor>& weight,
                      int64_t ignore_index,
                      const std::string& reduction,
                      phi::DenseTensor* out,
                      phi::DenseTensor* total_weight) {
  auto x_dims = x.dims();
  phi::Scalar weight_default = 1.0;
  int64_t reduction_int = 1;
  if (reduction == "none") {
    reduction_int = 0;
  } else if (reduction == "sum") {
    reduction_int = 2;
  }

  phi::DenseTensor weight_tensor;
  auto weight_size = phi::make_ddim({x.dims()[1]});
  if (weight.get_ptr() == nullptr) {
    weight_tensor.ResizeAndAllocate(weight_size);
    dev_ctx.template Alloc<float>(&weight_tensor);
    EXEC_NPU_CMD(
        aclnnInplaceFillScalar, dev_ctx, weight_tensor, weight_default);
  } else {
    weight_tensor = *weight.get_ptr();
  }

  bool need_resize = false;
  if (x_dims.size() == 4 && total_weight->dims().size() == 0) {
    total_weight->Resize(phi::make_ddim({1}));
    need_resize = true;
  }
  dev_ctx.template Alloc<T>(out);
  dev_ctx.template Alloc<T>(total_weight);

  if (x.dtype() == phi::DataType::FLOAT32) {
    if (x_dims.size() == 2) {
      EXEC_NPU_CMD(aclnnNLLLoss,
                   dev_ctx,
                   x,
                   labels,
                   weight_tensor,
                   reduction_int,
                   ignore_index,
                   *out,
                   *total_weight);
    } else if (x_dims.size() == 4) {
      EXEC_NPU_CMD(aclnnNLLLoss2d,
                   dev_ctx,
                   x,
                   labels,
                   weight_tensor,
                   reduction_int,
                   ignore_index,
                   *out,
                   *total_weight);
    }

    if (need_resize) {
      total_weight->Resize(phi::make_ddim({}));
    }
  } else {
    // data trans: double to float32
    phi::DenseTensor x_cast, weight_tensor_cast, out_cast, total_weight_cast;
    phi::DenseTensorMeta x_cast_meta;
    phi::DenseTensorMeta weight_tensor_cast_meta;
    phi::DenseTensorMeta out_cast_meta;
    phi::DenseTensorMeta total_weight_cast_meta;

    x_cast_meta = {phi::DataType::FLOAT32, x.dims()};
    weight_tensor_cast_meta = {phi::DataType::FLOAT32, weight_tensor.dims()};
    out_cast_meta = {phi::DataType::FLOAT32, out->dims()};
    total_weight_cast_meta = {phi::DataType::FLOAT32, total_weight->dims()};

    x_cast.set_meta(x_cast_meta);
    weight_tensor_cast.set_meta(weight_tensor_cast_meta);
    out_cast.set_meta(out_cast_meta);
    total_weight_cast.set_meta(total_weight_cast_meta);

    dev_ctx.template Alloc<float>(&out_cast);
    dev_ctx.template Alloc<float>(&total_weight_cast);
    custom_kernel::CastKernel<T, Context>(
        dev_ctx, x, phi::DataType::FLOAT32, &x_cast);
    custom_kernel::CastKernel<T, Context>(
        dev_ctx, weight_tensor, phi::DataType::FLOAT32, &weight_tensor_cast);

    if (x_dims.size() == 2) {
      EXEC_NPU_CMD(aclnnNLLLoss,
                   dev_ctx,
                   x_cast,
                   labels,
                   weight_tensor_cast,
                   reduction_int,
                   ignore_index,
                   out_cast,
                   total_weight_cast);
    } else if (x_dims.size() == 4) {
      EXEC_NPU_CMD(aclnnNLLLoss2d,
                   dev_ctx,
                   x_cast,
                   labels,
                   weight_tensor_cast,
                   reduction_int,
                   ignore_index,
                   out_cast,
                   total_weight_cast);
    }

    custom_kernel::CastKernel<T, Context>(dev_ctx, out_cast, out->dtype(), out);
    custom_kernel::CastKernel<T, Context>(
        dev_ctx, total_weight_cast, total_weight->dtype(), total_weight);

    if (need_resize) {
      total_weight->Resize(phi::make_ddim({}));
    }
  }
}

template <typename T, typename Context>
void NllLossGradKernel(const Context& dev_ctx,
                       const phi::DenseTensor& x,
                       const phi::DenseTensor& labels,
                       const paddle::optional<phi::DenseTensor>& weight,
                       const phi::DenseTensor& total_weight,
                       const phi::DenseTensor& d_out,
                       int64_t ignore_index,
                       const std::string& reduction,
                       phi::DenseTensor* dx) {
  auto x_dims = x.dims();
  phi::Scalar weight_default = 1.0;
  int64_t reduction_int = 1;
  if (reduction == "none") {
    reduction_int = 0;
  } else if (reduction == "sum") {
    reduction_int = 2;
  }

  phi::DenseTensor weight_tensor;
  auto weight_size = phi::make_ddim({x.dims()[1]});
  if (weight.get_ptr() == nullptr) {
    weight_tensor.ResizeAndAllocate(weight_size);
    dev_ctx.template Alloc<float>(&weight_tensor);
    EXEC_NPU_CMD(
        aclnnInplaceFillScalar, dev_ctx, weight_tensor, weight_default);
  } else {
    weight_tensor = *weight.get_ptr();
  }
  dev_ctx.template Alloc<T>(dx);

  phi::DenseTensor total_weight_new;
  if (x_dims.size() == 4) {
    phi::DenseTensorMeta total_weight_new_meta = {phi::DataType::FLOAT32,
                                                  phi::make_ddim({1})};
    total_weight_new.set_meta(total_weight_new_meta);
    TensorCopy(dev_ctx, total_weight, true, &total_weight_new);
    total_weight_new.Resize(phi::make_ddim({1}));
  }

  if (x.dtype() == phi::DataType::FLOAT32) {
    if (x_dims.size() == 2) {
      EXEC_NPU_CMD(aclnnNLLLossBackward,
                   dev_ctx,
                   d_out,
                   x,
                   labels,
                   weight_tensor,
                   reduction_int,
                   ignore_index,
                   total_weight,
                   *dx);
    } else if (x_dims.size() == 4) {
      if (d_out.dims().size() == 0) {
        phi::DenseTensor d_out_new;
        phi::DenseTensorMeta d_out_new_meta = {phi::DataType::FLOAT32,
                                               phi::make_ddim({1})};
        d_out_new.set_meta(d_out_new_meta);
        TensorCopy(dev_ctx, d_out, true, &d_out_new);
        d_out_new.Resize(phi::make_ddim({1}));

        EXEC_NPU_CMD(aclnnNLLLoss2dBackward,
                     dev_ctx,
                     d_out_new,
                     x,
                     labels,
                     weight_tensor,
                     reduction_int,
                     ignore_index,
                     total_weight_new,
                     *dx);
      } else {
        EXEC_NPU_CMD(aclnnNLLLoss2dBackward,
                     dev_ctx,
                     d_out,
                     x,
                     labels,
                     weight_tensor,
                     reduction_int,
                     ignore_index,
                     total_weight_new,
                     *dx);
      }
    }
  } else {
    // data trans: double to float32
    phi::DenseTensor d_out_cast, x_cast, weight_tensor_cast, total_weight_cast,
        dx_cast;
    phi::DenseTensorMeta d_out_cast_meta;
    phi::DenseTensorMeta x_cast_meta;
    phi::DenseTensorMeta weight_tensor_cast_meta;
    phi::DenseTensorMeta total_weight_cast_meta;
    phi::DenseTensorMeta dx_cast_meta;

    d_out_cast_meta = {phi::DataType::FLOAT32, d_out.dims()};
    x_cast_meta = {phi::DataType::FLOAT32, x.dims()};
    weight_tensor_cast_meta = {phi::DataType::FLOAT32, weight_tensor.dims()};
    total_weight_cast_meta = {phi::DataType::FLOAT32, total_weight.dims()};
    dx_cast_meta = {phi::DataType::FLOAT32, dx->dims()};

    d_out_cast.set_meta(d_out_cast_meta);
    x_cast.set_meta(x_cast_meta);
    weight_tensor_cast.set_meta(weight_tensor_cast_meta);
    total_weight_cast.set_meta(total_weight_cast_meta);
    dx_cast.set_meta(dx_cast_meta);

    dev_ctx.template Alloc<float>(&dx_cast);
    custom_kernel::CastKernel<T, Context>(
        dev_ctx, d_out, phi::DataType::FLOAT32, &d_out_cast);
    custom_kernel::CastKernel<T, Context>(
        dev_ctx, x, phi::DataType::FLOAT32, &x_cast);
    custom_kernel::CastKernel<T, Context>(
        dev_ctx, weight_tensor, phi::DataType::FLOAT32, &weight_tensor_cast);
    custom_kernel::CastKernel<T, Context>(
        dev_ctx, total_weight, phi::DataType::FLOAT32, &total_weight_cast);

    if (x_dims.size() == 4 && total_weight_cast.dims().size() == 0) {
      total_weight_cast.Resize(phi::make_ddim({1}));
    }

    if (x_dims.size() == 4 && d_out_cast.dims().size() == 0) {
      d_out_cast.Resize(phi::make_ddim({1}));
    }

    if (x_dims.size() == 2) {
      EXEC_NPU_CMD(aclnnNLLLossBackward,
                   dev_ctx,
                   d_out_cast,
                   x_cast,
                   labels,
                   weight_tensor_cast,
                   reduction_int,
                   ignore_index,
                   total_weight_cast,
                   dx_cast);
    } else if (x_dims.size() == 4) {
      EXEC_NPU_CMD(aclnnNLLLoss2dBackward,
                   dev_ctx,
                   d_out_cast,
                   x_cast,
                   labels,
                   weight_tensor_cast,
                   reduction_int,
                   ignore_index,
                   total_weight_cast,
                   dx_cast);
    }

    custom_kernel::CastKernel<T, Context>(dev_ctx, dx_cast, dx->dtype(), dx);
  }
}
}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(nll_loss,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::NllLossRawKernel,
                          float,
                          phi::dtype::float16,
                          double) {}

PD_REGISTER_PLUGIN_KERNEL(nll_loss_grad,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::NllLossGradKernel,
                          float,
                          phi::dtype::float16,
                          double) {}
