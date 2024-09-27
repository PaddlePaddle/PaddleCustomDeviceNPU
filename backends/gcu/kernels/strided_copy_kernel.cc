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

#include "common/gcu_op_runner.h"
#include "kernels/funcs/gcu_kernel_funcs.h"

namespace custom_kernel {

template <typename T, typename Context>
void StridedCopyKernel(const Context& dev_ctx,
                       const phi::DenseTensor& input,
                       const std::vector<int64_t>& dims,
                       const std::vector<int64_t>& out_stride,
                       int64_t offset,
                       phi::DenseTensor* out) {
  PADDLE_GCU_KERNEL_TRACE("contiguous");
  phi::DenseTensorMeta meta = input.meta();
  meta.strides = common::make_ddim(out_stride);
  meta.dims = common::make_ddim(dims);
  meta.offset = offset;
  out->set_meta(meta);

  PADDLE_ENFORCE_EQ(input.dims(),
                    out->dims(),
                    phi::errors::InvalidArgument(
                        "Input shape(%s) must be equal with out shape(%s).",
                        input.dims(),
                        out->dims()));

  PADDLE_ENFORCE_EQ(input.numel(),
                    out->numel(),
                    phi::errors::InvalidArgument(
                        "Input numel(%d) must be equal with out numel(%d).",
                        input.numel(),
                        out->numel()));

  if (input.numel() <= 0) {
    return;
  }

  if (LaunchAOTKernel()) {
    const T* input_data = input.data<T>();
    int input_rank = input.dims().size();
    const int64_t* input_dims = input.dims().Get();
    const int64_t* input_stride = input.strides().Get();

    T* output_data = out->data<T>();
    PADDLE_ENFORCE_NOT_NULL(output_data,
                            phi::errors::InvalidArgument(
                                "StridedCopyKernel's out tensor must complete "
                                "mutable data before call kernel."));
    int output_rank = meta.dims.size();
    const int64_t* output_dims = meta.dims.Get();
    const int64_t* output_stride = meta.strides.Get();

    auto numel = input.numel();

    for (int64_t i = 0; i < numel; i++) {
      int64_t input_offset = 0;
      int64_t index_tmp = i;
      for (int dim = input_rank - 1; dim >= 0; --dim) {
        input_offset += (index_tmp % input_dims[dim]) * input_stride[dim];
        index_tmp = index_tmp / input_dims[dim];
      }
      int64_t output_offset = 0;
      index_tmp = i;
      for (int dim = output_rank - 1; dim >= 0; --dim) {
        output_offset += (index_tmp % output_dims[dim]) * output_stride[dim];
        index_tmp = index_tmp / output_dims[dim];
      }

      C_Device_st device{input.place().GetDeviceId()};
      C_Stream stream = static_cast<C_Stream>(dev_ctx.stream());
      auto* dst_ptr = &output_data[output_offset];
      auto* src_ptr = &input_data[input_offset];
      AsyncMemCpyD2D(
          &device, stream, dst_ptr, src_ptr, phi::SizeOf(input.dtype()));
    }

  } else {  // kernel impl base on JIT
    THROW_JIT_UNIMPLEMENTED();
  }
}

template <typename T, typename Context>
void AsStridedKernel(const Context& dev_ctx,
                     const phi::DenseTensor& input,
                     const std::vector<int64_t>& dims,
                     const std::vector<int64_t>& stride,
                     int64_t offset,
                     phi::DenseTensor* out) {
  *out = input;
  auto meta = out->meta();
  meta.dims = common::make_ddim(dims);
  meta.strides = common::make_ddim(stride);
  meta.offset = offset;
  out->set_meta(meta);
}
}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(strided_copy,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::StridedCopyKernel,
                          bool,
                          uint8_t,
                          int8_t,
                          int16_t,
                          int32_t,
                          int64_t,
                          float,
                          double,
                          phi::dtype::float16,
                          phi::dtype::bfloat16,
                          phi::dtype::complex<float>,
                          phi::dtype::complex<double>) {}

PD_REGISTER_PLUGIN_KERNEL(as_strided,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::AsStridedKernel,
                          bool,
                          uint8_t,
                          int8_t,
                          int16_t,
                          int32_t,
                          int64_t,
                          float,
                          double,
                          phi::dtype::float16,
                          phi::dtype::bfloat16,
                          phi::dtype::complex<float>,
                          phi::dtype::complex<double>) {}
