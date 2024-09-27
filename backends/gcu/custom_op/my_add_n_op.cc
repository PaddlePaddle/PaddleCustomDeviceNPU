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

#include <iostream>
#include <vector>

#include "common/gcu_op_runner.h"
#include "kernels/funcs/gcu_kernel_funcs.h"
#include "paddle/extension.h"

std::vector<paddle::Tensor> MyAddNOp(const paddle::Tensor& x,
                                     const paddle::Tensor& y,
                                     const paddle::Tensor& z,
                                     int64_t axis) {
  auto dev_ctx = static_cast<const phi::CustomContext*>(
      paddle::experimental::DeviceContextPool::Instance().Get(x.place()));

  auto x_tensor = static_cast<const phi::DenseTensor*>(x.impl().get());
  auto y_tensor = static_cast<const phi::DenseTensor*>(y.impl().get());
  auto z_tensor = static_cast<const phi::DenseTensor*>(z.impl().get());

  std::shared_ptr<phi::DenseTensor> out_tensor =
      std::make_shared<phi::DenseTensor>();
  out_tensor->Resize(x_tensor->dims());
  dev_ctx->Alloc(out_tensor.get(), x_tensor->dtype());
  // for demo to show how to call custom topscc kernel
  //   extern void vec_add_cpp(int *from, int *to, size_t N);
  //   vec_add_cpp(static_cast<int*>(const_cast<void*>(x_tensor->data())),
  //               static_cast<int*>(out_tensor->data()), 1);
  custom_kernel::TensorNameMap input_names;
  custom_kernel::TensorValueMap inputs;
  input_names["X"] = {"x", "y", "z"};
  inputs["X"] = {const_cast<phi::DenseTensor*>(x_tensor),
                 const_cast<phi::DenseTensor*>(y_tensor),
                 const_cast<phi::DenseTensor*>(z_tensor)};

  custom_kernel::TensorNameMap output_names;
  custom_kernel::TensorValueMap outputs;
  output_names["Out"] = {"out"};
  outputs["Out"] = {out_tensor.get()};

  custom_kernel::GcuRunner(
      input_names, inputs, output_names, outputs, {}, "sum", *dev_ctx);

  return {paddle::Tensor(out_tensor)};
}

std::vector<std::vector<int64_t>> MyAddNOpInferShape(
    const std::vector<int64_t>& x_shape,
    const std::vector<int64_t>& y_shape,
    const std::vector<int64_t>& z_shape,
    int64_t axis) {
  VLOG(0) << "MyAddNOpInferShape : " << axis << "\n";
  return {x_shape};
}

PD_BUILD_OP(my_add_n)
    .Inputs({"X", "Y", "Z"})
    .Outputs({"Out"})
    .Attrs({"axis: int64_t"})
    .SetKernelFn(PD_KERNEL(MyAddNOp))
    .SetInferShapeFn(PD_INFER_SHAPE(
        MyAddNOpInferShape));  // neccessary if the op has muti_inputs
