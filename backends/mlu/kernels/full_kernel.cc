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

#include "kernels/funcs/mlu_baseop.h"

namespace custom_kernel {

template <typename T, typename Context>
void FullKernel(const Context& dev_ctx,
                const phi::IntArray& shape,
                const phi::Scalar& val,
                phi::DataType dtype,
                phi::DenseTensor* out) {
  auto shape_vec = shape.GetData();
  out->ResizeAndAllocate(phi::make_ddim(shape_vec));
  dev_ctx.template Alloc<T>(out);

  T value = val.to<T>();
  const T* value_data = &value;
  cnnlPointerMode_t pointer_mode = CNNL_POINTER_MODE_HOST;
  MLUCnnlTensorDesc output_desc(*out);
  MLUCnnl::Fill(
      dev_ctx, pointer_mode, value_data, output_desc.get(), GetBasePtr(out));
}

template <typename T, typename Context>
void FullLikeKernel(const Context& dev_ctx,
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
  auto value_t = static_cast<T>(value);
  MLUCnnlTensorDesc out_desc(*out, CNNL_LAYOUT_ARRAY, ToCnnlDataType<T>());

  MLUCnnl::Fill(dev_ctx,
                CNNL_POINTER_MODE_HOST,
                &value_t,
                out_desc.get(),
                GetBasePtr(out));
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
                          mlu,
                          ALL_LAYOUT,
                          custom_kernel::FullKernel,
                          bool,
                          uint8_t,
                          int16_t,
                          int,
                          int64_t,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(full_like,
                          mlu,
                          ALL_LAYOUT,
                          custom_kernel::FullLikeKernel,
                          int,
                          int64_t,
                          float,
                          phi::dtype::float16) {
  kernel->InputAt(0).SetBackend(phi::Backend::ALL_BACKEND);
}

PD_REGISTER_PLUGIN_KERNEL(full_batch_size_like,
                          mlu,
                          ALL_LAYOUT,
                          custom_kernel::FullBatchSizeLikeKernel,
                          int,
                          float,
                          phi::dtype::float16) {
  kernel->InputAt(0).SetBackend(phi::Backend::ALL_BACKEND);
}
