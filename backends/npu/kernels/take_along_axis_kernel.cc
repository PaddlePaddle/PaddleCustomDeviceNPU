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
void AclopTakeAlongAxisKernel(const Context& dev_ctx,
                              const phi::DenseTensor& x,
                              const phi::DenseTensor& index,
                              int axis,
                              phi::DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);
  auto stream = dev_ctx.stream();
  NPUAttributeMap attr_input = {{"dim", axis}};
  auto op_func = [](const std::vector<phi::DenseTensor>& inputs,
                    const std::vector<phi::DenseTensor>& outputs,
                    const NPUAttributeMap& attrs,
                    const Context& dev_ctx) {
    const auto& runner = NpuOpRunner("GatherElements", inputs, outputs, attrs);
    runner.Run(dev_ctx.stream());
  };
  if (x.dtype() == phi::DataType::FLOAT64) {
    NpuOpRunner::TypeAdapter({x, index},
                             {*out},
                             attr_input,
                             dev_ctx,
                             op_func,
                             {phi::DataType::FLOAT32, phi::DataType::INT64},
                             {phi::DataType::FLOAT32});
  } else {
    const auto& runner =
        NpuOpRunner("GatherElements", {x, index}, {*out}, attr_input);
    runner.Run(stream);
  }
}

template <typename T, typename Context>
void TakeAlongAxisKernel(const Context& dev_ctx,
                         const phi::DenseTensor& x,
                         const phi::DenseTensor& index,
                         int axis,
                         phi::DenseTensor* out) {
  DO_COMPATIBILITY(aclnnGather,
                   (custom_kernel::AclopTakeAlongAxisKernel<T, Context>(
                       dev_ctx, x, index, axis, out)));
  dev_ctx.template Alloc<T>(out);

  int64_t axis_ = static_cast<int64_t>(axis);
  EXEC_NPU_CMD(aclnnGather, dev_ctx, x, axis_, index, *out);
}

template <typename T, typename Context>
void AclopTakeAlongAxisGradKernel(const Context& dev_ctx,
                                  const phi::DenseTensor& x,
                                  const phi::DenseTensor& index,
                                  const phi::DenseTensor& out_grad,
                                  int axis,
                                  phi::DenseTensor* x_grad) {
  dev_ctx.template Alloc<T>(x_grad);
  FillNpuTensorWithConstant<T>(x_grad, dev_ctx, static_cast<T>(0));
  x_grad->Resize(x.dims());
  auto stream = dev_ctx.stream();
  NPUAttributeMap attr_input = {{"axis", axis}};
  if (x_grad->dtype() == phi::DataType::FLOAT64) {
    phi::DenseTensor tmp_x_grad;
    tmp_x_grad.Resize(x_grad->dims());
    dev_ctx.template Alloc<float>(&tmp_x_grad);
    const auto& cast_runner1 =
        NpuOpRunner("Cast", {*x_grad}, {tmp_x_grad}, {{"dst_type", ACL_FLOAT}});
    cast_runner1.Run(dev_ctx.stream());
    phi::DenseTensor tmp_out_grad;
    tmp_out_grad.Resize(out_grad.dims());
    dev_ctx.template Alloc<float>(&tmp_out_grad);
    const auto& cast_runner2 = NpuOpRunner(
        "Cast", {out_grad}, {tmp_out_grad}, {{"dst_type", ACL_FLOAT}});
    cast_runner2.Run(dev_ctx.stream());
    const auto& runner = NpuOpRunner("ScatterAddWithAxis",
                                     {tmp_x_grad, index, tmp_out_grad},
                                     {tmp_x_grad},
                                     attr_input);
    runner.Run(stream);
    const auto& cast_runner3 = NpuOpRunner(
        "Cast", {tmp_x_grad}, {*x_grad}, {{"dst_type", ACL_DOUBLE}});
    cast_runner3.Run(dev_ctx.stream());
    const auto& cast_runner4 = NpuOpRunner(
        "Cast", {tmp_out_grad}, {out_grad}, {{"dst_type", ACL_DOUBLE}});
    cast_runner4.Run(dev_ctx.stream());
  } else {
    const auto& runner = NpuOpRunner("ScatterAddWithAxis",
                                     {*x_grad, index, out_grad},
                                     {*x_grad},
                                     {{"axis", axis}});
    runner.Run(stream);
  }
}

template <typename T, typename Context>
void TakeAlongAxisGradKernel(const Context& dev_ctx,
                             const phi::DenseTensor& x,
                             const phi::DenseTensor& index,
                             const phi::DenseTensor& out_grad,
                             int axis,
                             phi::DenseTensor* x_grad) {
  DO_COMPATIBILITY(aclnnInplaceScatter,
                   (custom_kernel::AclopTakeAlongAxisGradKernel<T, Context>(
                       dev_ctx, x, index, out_grad, axis, x_grad)));

  dev_ctx.template Alloc<T>(x_grad);
  FillNpuTensorWithConstant<T>(x_grad, dev_ctx, static_cast<T>(0));
  x_grad->Resize(x.dims());

  int64_t reduce = 1;  // add

  EXEC_NPU_CMD(
      aclnnInplaceScatter, dev_ctx, *x_grad, axis, index, out_grad, reduce);
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(take_along_axis,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::TakeAlongAxisKernel,
                          int,
                          int64_t,
                          phi::dtype::float16,
                          phi::dtype::bfloat16,
                          float,
                          double) {}

PD_REGISTER_PLUGIN_KERNEL(take_along_axis_grad,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::TakeAlongAxisGradKernel,
                          int,
                          int64_t,
                          phi::dtype::float16,
                          phi::dtype::bfloat16,
                          float,
                          double) {}
