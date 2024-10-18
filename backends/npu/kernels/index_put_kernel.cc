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
#include "paddle/phi/common/type_traits.h"
#include "paddle/phi/kernels/funcs/tensor_formatter.h"

namespace custom_kernel {

template <typename T, typename Context>
void StackKernel(const Context& dev_ctx,
                 const std::vector<const phi::DenseTensor*>& x,
                 int axis,
                 phi::DenseTensor* y);

template <typename T, typename Context>
void GatherNdKernel(const Context& dev_ctx,
                    const phi::DenseTensor& x,
                    const phi::DenseTensor& index,
                    phi::DenseTensor* out);

template <typename T, typename Context>
void NonZeroKernel(const Context& dev_ctx,
                   const phi::DenseTensor& condition,
                   phi::DenseTensor* out);

template <typename T, typename Context>
void CastKernel(const Context& dev_ctx,
                const phi::DenseTensor& x,
                phi::DataType dtype,
                phi::DenseTensor* out);

template <typename T, typename Context>
void IndexPutKernel(const Context& dev_ctx,
                    const phi::DenseTensor& x,
                    const std::vector<const phi::DenseTensor*>& indices,
                    const phi::DenseTensor& value,
                    bool accumulate,
                    phi::DenseTensor* out) {
  bool unsafe = true;

  std::vector<phi::DenseTensor*> tensor_list(indices.size());
  for (size_t i = 0; i < indices.size(); i++) {
    if (indices[i]->dtype() == phi::DataType::INT32) {
      tensor_list[i] = new phi::DenseTensor;
      tensor_list[i]->Resize(indices[i]->dims());
      dev_ctx.Alloc(tensor_list[i], phi::DataType::INT64);
      custom_kernel::CastKernel<T, Context>(
          dev_ctx, *(indices[i]), phi::DataType::INT64, tensor_list[i]);
    } else {
      tensor_list[i] = new phi::DenseTensor(*(indices[i]));
    }
  }

  EXEC_NPU_CMD(
      aclnnIndexPutImpl, dev_ctx, x, tensor_list, value, accumulate, unsafe);

  dev_ctx.template Alloc<T>(out);
  TensorCopy(dev_ctx, x, true, out);
}

template <typename T, typename Context>
void IndexPutGradKernel(const Context& dev_ctx,
                        const phi::DenseTensor& x,
                        const std::vector<const phi::DenseTensor*>& indices,
                        const phi::DenseTensor& value,
                        const phi::DenseTensor& out_grad,
                        bool accumulate,
                        phi::DenseTensor* x_grad,
                        phi::DenseTensor* value_grad) {
  bool unsafe = true;

  std::vector<phi::DenseTensor*> tensor_list(indices.size());
  for (size_t i = 0; i < indices.size(); i++) {
    if (indices[i]->dtype() == phi::DataType::INT32) {
      tensor_list[i] = new phi::DenseTensor;
      tensor_list[i]->Resize(indices[i]->dims());
      dev_ctx.Alloc(tensor_list[i], phi::DataType::INT64);
      custom_kernel::CastKernel<T, Context>(
          dev_ctx, *(indices[i]), phi::DataType::INT64, tensor_list[i]);
    } else {
      tensor_list[i] = new phi::DenseTensor(*(indices[i]));
    }
  }

  if (x_grad) {
    dev_ctx.template Alloc<T>(x_grad);
    TensorCopy(dev_ctx, out_grad, true, x_grad);
    phi::DenseTensorMeta value_zero_meta = {value.dtype(), value.dims()};
    phi::DenseTensor value_zero;
    value_zero.set_meta(value_zero_meta);
    dev_ctx.template Alloc<T>(&value_zero);
    EXEC_NPU_CMD(aclnnInplaceZero, dev_ctx, value_zero);
    EXEC_NPU_CMD(aclnnIndexPutImpl,
                 dev_ctx,
                 *x_grad,
                 tensor_list,
                 value_zero,
                 accumulate,
                 unsafe);
  }

  if (value_grad) {
    dev_ctx.template Alloc<T>(value_grad);
    if (tensor_list[0]->dtype() == phi::DataType::BOOL) {
      // deal with bool indices
      PADDLE_ENFORCE_EQ(
          tensor_list.size(),
          1,
          phi::errors::InvalidArgument("bool indices should be 1d"));

      phi::DenseTensor non_zero_index;
      custom_kernel::NonZeroKernel<int64_t, Context>(
          dev_ctx, *tensor_list[0], &non_zero_index);
      custom_kernel::GatherNdKernel<T, Context>(
          dev_ctx, out_grad, non_zero_index, value_grad);
    } else {
      phi::DenseTensorMeta index_tensor_meta = {
          tensor_list[0]->dtype(),
          phi::make_ddim({tensor_list[0]->dims()[0], tensor_list.size()})};
      phi::DenseTensor index_tensor;
      index_tensor.set_meta(index_tensor_meta);
      custom_kernel::StackKernel<int64_t, Context>(
          dev_ctx, indices, -1, &index_tensor);
      custom_kernel::GatherNdKernel<T, Context>(
          dev_ctx, out_grad, index_tensor, value_grad);
    }
  }
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(index_put,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::IndexPutKernel,
                          float,
                          double,
                          phi::dtype::bfloat16,
                          phi::dtype::float16,
                          bool,
                          int,
                          int64_t) {}

PD_REGISTER_PLUGIN_KERNEL(index_put_grad,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::IndexPutGradKernel,
                          float,
                          double,
                          phi::dtype::bfloat16,
                          phi::dtype::float16,
                          bool,
                          int,
                          int64_t) {}
