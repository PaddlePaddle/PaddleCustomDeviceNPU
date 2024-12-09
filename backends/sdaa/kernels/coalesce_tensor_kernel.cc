// BSD 3- Clause License Copyright (c) 2023, Tecorigin Co., Ltd. All rights
// reserved.
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimer.
// Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution.
// Neither the name of the copyright holder nor the names of its contributors
// may be used to endorse or promote products derived from this software
// without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION)
// HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
// STRICT LIABILITY,OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)  ARISING IN ANY
// WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
// OF SUCH DAMAGE.

#include <iostream>
#include <sstream>

#include "kernels/funcs/sdaa_baseop.h"
#include "kernels/funcs/slice_utils.h"
#include "paddle/phi/extension.h"

namespace custom_kernel {

size_t Alignment(size_t size, const phi::Place& place, int align_size) {
  size_t alignment = 0;
  if (align_size > 0) {
    alignment = align_size;
  } else {
    alignment = 1 << 13;
  }
  size_t remaining = size % alignment;
  return remaining == 0 ? size : size + (alignment - remaining);
}

void GetMemSizeAndDtype(
    const std::vector<const phi::DenseTensor*>& dense_tensor,
    size_t* numel,
    const size_t& size_of_dtype,
    const phi::Place& place,
    const bool use_align = true,
    const int align_size = -1) {
  *numel = 0;
  std::stringstream ss;
  ss << "alloc_space_for_vars: ";
  for (size_t i = 0; i < dense_tensor.size(); i++) {
    auto size = dense_tensor[i]->numel();
    PADDLE_ENFORCE_GT(size,
                      0,
                      phi::errors::InvalidArgument(
                          "The number of %d-th tensor's elements is 0.", i));
    auto len = use_align ? custom_kernel::Alignment(
                               static_cast<size_t>(size) * size_of_dtype,
                               place,
                               align_size) /
                               size_of_dtype
                         : static_cast<size_t>(size);
    const void* ptr =
        dense_tensor[i]->initialized() ? dense_tensor[i]->data() : nullptr;
    ss << "input(" << i << "-th tensor) dim:(" << dense_tensor[i]->dims() << ")"
       << " address: " << ptr << " len: " << len << ", ";
    *numel += len;
  }
  VLOG(10) << ss.str();
}

template <typename Context>
struct FillConstantVisitor {
  FillConstantVisitor(const Context& dev_ctx,
                      phi::DenseTensor* tensor,
                      const float value,
                      phi::DataType dtype)
      : dev_ctx_(dev_ctx), tensor_(tensor), value_(value), dtype_(dtype) {}

  template <typename T>
  void apply(typename std::enable_if<std::is_same<T, int8_t>::value ||
                                     std::is_same<T, int16_t>::value>::type* =
                 nullptr) const {
    PADDLE_THROW(phi::errors::InvalidArgument(
        "Not support data type for set_constant attr."));
  }

  template <typename T>
  void apply(
      typename std::enable_if<!(std::is_same<T, int8_t>::value ||
                                std::is_same<T, int16_t>::value)>::type* =
          nullptr) const {
    sdaa_ops::doFillTensor<T>(
        dev_ctx_, static_cast<T>(value_), dtype_, tensor_);
  }

  const Context& dev_ctx_;
  phi::DenseTensor* tensor_;
  float value_;
  phi::DataType dtype_;
};

template <typename Visitor>
static void VisitDataType(phi::DataType type, Visitor visitor) {
  if (type == phi::DataType::FLOAT32) {
    visitor.template apply<float>();
  } else if (type == phi::DataType::FLOAT64) {
    visitor.template apply<double>();
  } else if (type == phi::DataType::INT32) {
    visitor.template apply<int>();
  } else if (type == phi::DataType::INT64) {
    visitor.template apply<int64_t>();
  } else {
    PADDLE_THROW(phi::errors::InvalidArgument(
        "The received values data type %s can not meet input requirements. "
        "Because the given values data type of searchsorted operators must be "
        "float32, float64, int32 or int64. Please input appropriate "
        "sorted_sequence again! ",
        type));
  }
}

template <typename T, typename Context>
void CoalesceTensorKernel(const Context& dev_ctx,
                          const std::vector<const phi::DenseTensor*>& input,
                          phi::DataType dtype,
                          bool copy_data,
                          bool set_constant,
                          bool persist_output,
                          float constant,
                          bool use_align,
                          int align_size,
                          int size_of_dtype,
                          const std::vector<int64_t>& concated_shapes,
                          const std::vector<int64_t>& concated_ranks,
                          std::vector<phi::DenseTensor*> output,
                          phi::DenseTensor* fused_output) {
  VLOG(4) << "CALL SDAA CoalesceTensorKernel";

  PADDLE_ENFORCE_GT(input.size(),
                    static_cast<size_t>(0),
                    phi::errors::InvalidArgument(
                        "The CoalesceTensor operator has no input."));
  PADDLE_ENFORCE_EQ(input.size(),
                    output.size(),
                    phi::errors::InvalidArgument(
                        "The number of CoalesceTensor operator's input and "
                        "output is not match, input number is %u, "
                        "output number is %u",
                        input.size(),
                        output.size()));

  // Input and Output check: only support DenseTensor
  bool has_not_in_vars = false;
  for (size_t i = 0; i < input.size(); i++) {
    PADDLE_ENFORCE_NOT_NULL(
        input[i],
        phi::errors::InvalidArgument(
            "The %d-th input tensor cannot be nullptr.", i));
    PADDLE_ENFORCE_NOT_NULL(
        output[i],
        phi::errors::InvalidArgument(
            "The %d-th output tensor cannot be nullptr.", i));
    if (!input[i]->initialized()) {
      has_not_in_vars = true;
    }
  }

  if (has_not_in_vars) {
    PADDLE_ENFORCE_EQ(concated_ranks.size(),
                      output.size(),
                      phi::errors::InvalidArgument(
                          "The attribute(concated_ranks) length must be "
                          "equal to the output tensor number."));
    int64_t accumulated_ranks = 0;
    for (size_t i = 0; i < input.size(); i++) {
      phi::DDim dims(concated_shapes.data() + accumulated_ranks,
                     concated_ranks[i]);
      if (!input[i]->initialized()) {
        PADDLE_ENFORCE_EQ(input[i],
                          output[i],
                          phi::errors::InvalidArgument(
                              "The %d-th output tensor and %d-th input tensor "
                              "when the %d-th input tensor is not initialized.",
                              i,
                              i,
                              i));
        output[i]->Resize(dims);
      } else {
        PADDLE_ENFORCE_EQ(
            input[i]->dims(),
            dims,
            phi::errors::InvalidArgument(
                "The %d-th input tensor shape does not match the "
                "attribute(concated_shapes) and attribute(concated_ranks).",
                i));
      }
      accumulated_ranks += concated_ranks[i];
      PADDLE_ENFORCE_LE(accumulated_ranks,
                        concated_shapes.size(),
                        phi::errors::InvalidArgument(
                            "The attribute(concated_shapes) and "
                            "attribute(concated_ranks) do not match."));
    }
    PADDLE_ENFORCE_EQ(accumulated_ranks,
                      concated_shapes.size(),
                      phi::errors::InvalidArgument(
                          "The attribute(concated_shapes) and "
                          "attribute(concated_ranks) do not match."));
  }

  // Init the ouput as input
  for (size_t i = 0; i < input.size(); i++) {
    output[i]->Resize(input[i]->dims());
  }

  // Get numel and dtype
  size_t numel = 0;
  if (size_of_dtype == -1) {
    size_of_dtype = phi::SizeOf(dtype);
  }
  PADDLE_ENFORCE_NE(
      size_of_dtype,
      0,
      phi::errors::InvalidArgument("The size of dtype cannot be 0."));

  GetMemSizeAndDtype(
      input, &numel, size_of_dtype, dev_ctx.GetPlace(), use_align, align_size);

  // Alloc the continuous space
  void* fused_output_ptr = dev_ctx.Alloc(
      &fused_output->Resize(phi::make_ddim({static_cast<int64_t>(numel)})),
      dtype);

  // If OpTest is not required, this step can be omitted.
  sdaa_ops::doMemsetTensor(dev_ctx, static_cast<int>(0), fused_output);

  size_t offset = 0;
  if (copy_data) {
    for (size_t i = 0; i < input.size(); i++) {
      size_t len = static_cast<size_t>(input[i]->numel());
      auto sub_tensor =
          custom_kernel::Slice(*fused_output,
                               static_cast<int64_t>(offset),
                               static_cast<int64_t>(offset + len));
      phi::Copy(dev_ctx, *input[i], dev_ctx.GetPlace(), false, &sub_tensor);

      offset += use_align
                    ? custom_kernel::Alignment(
                          len * size_of_dtype, dev_ctx.GetPlace(), align_size) /
                          size_of_dtype
                    : len;
    }
  } else if (set_constant) {
    custom_kernel::VisitDataType(
        dtype,
        FillConstantVisitor<Context>(dev_ctx, fused_output, constant, dtype));
  } else if (persist_output) {
    for (size_t i = 0; i < output.size(); i++) {
      size_t len = static_cast<size_t>(output[i]->numel());
      auto sub_tensor =
          custom_kernel::Slice(*fused_output,
                               static_cast<int64_t>(offset),
                               static_cast<int64_t>(offset + len));

      // some var may not persistable, or persistable var may not init
      if (output[i]->initialized()) {
        phi::Copy(dev_ctx, *output[i], dev_ctx.GetPlace(), false, &sub_tensor);
      }
      offset += use_align
                    ? custom_kernel::Alignment(
                          len * size_of_dtype, dev_ctx.GetPlace(), align_size) /
                          size_of_dtype
                    : len;
    }
  }

  // make the outputs point to the continuous space.
  offset = 0;
  std::stringstream ss;
  ss << "alloc_space_for_vars: ";

  for (size_t i = 0; i < output.size(); i++) {
    size_t len = static_cast<size_t>(output[i]->numel());
    auto dim = output[i]->dims();

    *output[i] = custom_kernel::Slice(*fused_output,
                                      static_cast<int64_t>(offset),
                                      static_cast<int64_t>(offset + len))
                     .Resize(dim);

    len = use_align ? custom_kernel::Alignment(
                          len * size_of_dtype, dev_ctx.GetPlace(), align_size) /
                          size_of_dtype
                    : len;

    ss << "output(" << i << "-th tensor) dim:(" << output[i]->dims() << ")"
       << " address: " << output[i]->data() << " len: " << len << ", ";
    offset += len;
  }
  PADDLE_ENFORCE_EQ(
      (int64_t)offset,
      fused_output->numel(),
      phi::errors::InvalidArgument("The alloc_space_for_var's offset: %s is "
                                   "unequal with fused_output's numel: %s.",
                                   offset,
                                   fused_output->numel()));
  VLOG(10) << ss.str();
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(coalesce_tensor,
                          sdaa,
                          ALL_LAYOUT,
                          custom_kernel::CoalesceTensorKernel,
                          float,
                          phi::dtype::float16) {
  kernel->InputAt(0).SetBackend(phi::Backend::ALL_BACKEND);
}
