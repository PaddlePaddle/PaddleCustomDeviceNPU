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

template <typename T, typename Context>
void AssignKernel(const Context& dev_ctx,
                  const phi::DenseTensor& x,
                  phi::DenseTensor* out) {
  PADDLE_GCU_KERNEL_TRACE("assign");
  VLOG(6) << "[HOST_KERNEL] Impl on host for assign";
  dev_ctx.template Alloc<T>(out);
  TensorCopy(dev_ctx, x, false, out);
}

template <typename T, typename Context>
void AssignRawKernel(const Context& dev_ctx,
                     const paddle::optional<phi::DenseTensor>& x,
                     phi::DenseTensor* out) {
  PADDLE_GCU_KERNEL_TRACE("assign_raw");
  VLOG(6) << "[HOST_KERNEL] Impl on host for assign_raw";
  if (x) {
    if (!x->initialized()) {
      return;
    }
    auto& x_tensor = *x.get_ptr();
    custom_kernel::AssignKernel<T, Context>(dev_ctx, x_tensor, out);
  }
}

template <typename T, typename Context>
void AssignArrayKernel(const Context& dev_ctx,
                       const std::vector<const phi::DenseTensor*>& x,
                       std::vector<phi::DenseTensor*> out) {
  PADDLE_GCU_KERNEL_TRACE("assign_array");
  VLOG(6) << "[HOST_KERNEL] Impl on host for assign_array";
  for (size_t i = 0; i < x.size(); ++i) {
    custom_kernel::AssignKernel<T, Context>(dev_ctx, *x[i], out.at(i));
  }
}

template <typename T, typename Context>
typename std::enable_if<std::is_same<T, bool>::value>::type CopyVectorToTensor(
    const Context& dev_ctx,
    const std::vector<phi::Scalar>& values,
    phi::DenseTensor* out) {
  // If attribute value dtype is vector<bool>, it will be converted to
  // vector<int>. at the same time, we can not use vector<bool> to hold
  // the value, because the c++ use bit value to replace byte value.
  std::vector<int> assign_values;
  assign_values.reserve(values.size());
  for (const auto& val : values) {
    assign_values.emplace_back(val.to<int>());
  }

  // use the array to replace to vector
  bool* array_ptr = new T[assign_values.size()];
  for (unsigned int i = 0; i < assign_values.size(); i++) {
    array_ptr[i] = static_cast<T>(assign_values[i]);
  }
  custom_kernel::TensorFromArray(
      dev_ctx, array_ptr, assign_values.size(), dev_ctx, out);
  dev_ctx.Wait();
  delete[] array_ptr;
}

template <typename T, typename Context>
typename std::enable_if<!std::is_same<T, bool>::value>::type CopyVectorToTensor(
    const Context& dev_ctx,
    const std::vector<phi::Scalar>& values,
    phi::DenseTensor* out) {
  std::vector<T, PinnedAllocatorForSTL<T>> assign_values;
  assign_values.reserve(values.size());
  for (const auto& val : values) {
    assign_values.emplace_back(val.to<T>());
  }
  custom_kernel::TensorFromVector(dev_ctx, assign_values, dev_ctx, out);
  //   dev_ctx.Wait();
}

template <typename T, typename Context>
void AssignValueKernel(const Context& dev_ctx,
                       const std::vector<int>& shape,
                       phi::DataType dtype,
                       const std::vector<phi::Scalar>& values,
                       phi::DenseTensor* out) {
  PADDLE_GCU_KERNEL_TRACE("assign_value");
  VLOG(6) << "[HOST_KERNEL] Impl on host for assign_value";
  auto template_dtype = phi::CppTypeToDataType<T>::Type();
  PADDLE_ENFORCE_EQ(
      dtype,
      template_dtype,
      phi::errors::InvalidArgument("Argument dtype mismatch for kernel dtype, "
                                   "argument dtype is %s, kernel dtype is %s.",
                                   dtype,
                                   template_dtype));
  CopyVectorToTensor<T>(dev_ctx, values, out);
  out->Resize(phi::make_ddim(shape));
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(assign,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::AssignKernel,
                          int,
                          float,
                          double,
                          int64_t,
                          bool,
                          phi::dtype::float16) {
  kernel->InputAt(0).SetBackend(phi::Backend::ALL_BACKEND);
}

PD_REGISTER_PLUGIN_KERNEL(assign_raw,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::AssignRawKernel,
                          int,
                          float,
                          double,
                          int64_t,
                          bool,
                          phi::dtype::float16) {
  kernel->InputAt(0).SetBackend(phi::Backend::ALL_BACKEND);
}

PD_REGISTER_PLUGIN_KERNEL(assign_array,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::AssignArrayKernel,
                          int,
                          float,
                          double,
                          int64_t,
                          bool,
                          phi::dtype::float16) {
  kernel->InputAt(0).SetBackend(phi::Backend::ALL_BACKEND);
}

PD_REGISTER_PLUGIN_KERNEL(assign_value,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::AssignValueKernel,
                          bool,
                          int,
                          int64_t,
                          double,
                          float,
                          phi::dtype::float16) {}
