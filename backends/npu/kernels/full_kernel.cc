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
#include "paddle/phi/core/tensor_meta.h"

namespace custom_kernel {

template <typename T, typename Context>
void AclopFullKernel(const Context& dev_ctx,
                     const phi::IntArray& shape,
                     const phi::Scalar& val,
                     phi::DataType dtype,
                     phi::DenseTensor* out) {
  auto shape_vec = shape.GetData();
  auto out_dim = phi::make_ddim(shape_vec);
  out->ResizeAndAllocate(out_dim);
  dev_ctx.template Alloc<T>(out);
  aclrtStream stream = static_cast<aclrtStream>(dev_ctx.stream());

  FillNpuTensorWithConstant<T>(out, dev_ctx, static_cast<T>(val.to<T>()));

  out->Resize(out_dim);
}

template <typename T, typename Context>
void FullKernel(const Context& dev_ctx,
                const phi::IntArray& shape,
                const phi::Scalar& val,
                phi::DataType dtype,
                phi::DenseTensor* out) {
  DO_COMPATIBILITY(aclnnInplaceFillScalar,
                   (custom_kernel::AclopFullKernel<T, Context>(
                       dev_ctx, shape, val, dtype, out)));

  auto shape_vec = shape.GetData();
  auto out_dim = phi::make_ddim(shape_vec);
  out->ResizeAndAllocate(out_dim);
  dev_ctx.template Alloc<T>(out);
  aclrtStream stream = static_cast<aclrtStream>(dev_ctx.stream());

  EXEC_NPU_CMD(aclnnInplaceFillScalar, dev_ctx, *out, val);
}

template <typename T, typename Context>
void AclopFullLikeKernel(const Context& dev_ctx,
                         const phi::DenseTensor& x,
                         const phi::Scalar& val,
                         phi::DataType dtype,
                         phi::DenseTensor* out) {
  auto value = val.to<double>();
  using CommonType = typename std::common_type<
      float,
      typename std::conditional<
          std::is_same<T, phi::dtype::float16>::value ||
              std::is_same<T, phi::dtype::bfloat16>::value,
          float,
          T>::type>::type;

  auto common_type_value = static_cast<CommonType>(value);

  // Check whether the filled value is valid
  bool is_out_range = true;
  if (std::isinf(value) || std::isnan(value)) {
    is_out_range = false;
  }

  if ((common_type_value >=
       static_cast<CommonType>(std::numeric_limits<T>::lowest())) &&
      (common_type_value <=
       static_cast<CommonType>(std::numeric_limits<T>::max()))) {
    is_out_range = false;
  }

  PADDLE_ENFORCE_EQ(
      is_out_range,
      false,
      phi::errors::InvalidArgument(
          "The filled value is out of range for target type, "
          "current kernel type is %s, the range should between %f "
          "and %f, but now value is %f.",
          typeid(T).name(),
          static_cast<CommonType>(std::numeric_limits<T>::lowest()),
          static_cast<CommonType>(std::numeric_limits<T>::max()),
          static_cast<float>(value)));

  dev_ctx.template Alloc<T>(out);
  aclrtStream stream = static_cast<aclrtStream>(dev_ctx.stream());

  NpuOpRunner runner;
  runner.SetType("Fills")
      .AddInput(*out)
      .AddOutput(*out)
      .AddAttrs({{"value", val.to<float>()}})
      .Run(stream);
}

template <typename T, typename Context>
void FullLikeKernel(const Context& dev_ctx,
                    const phi::DenseTensor& x,
                    const phi::Scalar& val,
                    phi::DataType dtype,
                    phi::DenseTensor* out) {
  DO_COMPATIBILITY(aclnnInplaceFillScalar,
                   (custom_kernel::AclopFullLikeKernel<T, Context>(
                       dev_ctx, x, val, dtype, out)));

  auto value = val.to<double>();
  using CommonType = typename std::common_type<
      float,
      typename std::conditional<
          std::is_same<T, phi::dtype::float16>::value ||
              std::is_same<T, phi::dtype::bfloat16>::value,
          float,
          T>::type>::type;

  auto common_type_value = static_cast<CommonType>(value);

  // Check whether the filled value is valid
  bool is_out_range = true;
  if (std::isinf(value) || std::isnan(value)) {
    is_out_range = false;
  }

  if ((common_type_value >=
       static_cast<CommonType>(std::numeric_limits<T>::lowest())) &&
      (common_type_value <=
       static_cast<CommonType>(std::numeric_limits<T>::max()))) {
    is_out_range = false;
  }

  PADDLE_ENFORCE_EQ(
      is_out_range,
      false,
      phi::errors::InvalidArgument(
          "The filled value is out of range for target type, "
          "current kernel type is %s, the range should between %f "
          "and %f, but now value is %f.",
          typeid(T).name(),
          static_cast<CommonType>(std::numeric_limits<T>::lowest()),
          static_cast<CommonType>(std::numeric_limits<T>::max()),
          static_cast<float>(value)));

  dev_ctx.template Alloc<T>(out);
  EXEC_NPU_CMD(aclnnInplaceFillScalar, dev_ctx, *out, val);
}

template <typename T, typename Context>
void FullBatchSizeLikeKernel(const Context& dev_ctx,
                             const phi::DenseTensor& x,
                             const std::vector<int>& shape,
                             const phi::Scalar& val,
                             phi::DataType dtype,
                             int x_batch_size_dim,
                             int out_batch_size_dim,
                             phi::DenseTensor* out) {
  if (x.lod().size() && x_batch_size_dim == 0) {
    // set the correct batch size for the DenseTensor.
    auto odims = out->dims();
    odims[out_batch_size_dim] = static_cast<int>(x.lod().back().size()) - 1;
    custom_kernel::FullKernel<T, Context>(
        dev_ctx, phi::vectorize(odims), val, dtype, out);
  }
  custom_kernel::FullLikeKernel<T, Context>(dev_ctx, x, val, dtype, out);
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(full,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::FullKernel,
                          bool,
                          int,
                          int64_t,
                          float,
                          double,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}

PD_REGISTER_PLUGIN_KERNEL(full_like,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::FullLikeKernel,
                          bool,
                          int,
                          int64_t,
                          float,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {
  kernel->InputAt(0).SetBackend(phi::Backend::ALL_BACKEND);
}

PD_REGISTER_PLUGIN_KERNEL(full_batch_size_like,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::FullBatchSizeLikeKernel,
                          int,
                          float,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {
  kernel->InputAt(0).SetBackend(phi::Backend::ALL_BACKEND);
}
