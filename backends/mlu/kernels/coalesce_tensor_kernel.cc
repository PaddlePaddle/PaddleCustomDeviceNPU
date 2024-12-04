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

#include <sstream>
#include <vector>

#include "kernels/funcs/mlu_baseop.h"
#include "kernels/funcs/mlu_funcs.h"

namespace custom_kernel {
inline phi::DenseTensor Slice(const phi::DenseTensor &src,
                              int64_t begin_idx,
                              int64_t end_idx) {
  auto meta = src.meta();
  PADDLE_ENFORCE_GE(
      begin_idx,
      0,
      phi::errors::OutOfRange("The start row index must be greater than 0."
                              "But received the start index is d%.",
                              begin_idx));
  PADDLE_ENFORCE_LE(
      end_idx,
      meta.dims[0],
      phi::errors::OutOfRange("The end row index is out of bound."));

  PADDLE_ENFORCE_LT(
      begin_idx,
      end_idx,
      phi::errors::InvalidArgument(
          "The start row index must be less than the end row index."
          "But received the start index = %d, the end index = %d.",
          begin_idx,
          end_idx));

  if (meta.dims[0] == 1) {
    return src;
  } else {
    size_t base = src.numel() / meta.dims[0];
    phi::DenseTensor dst(src);
    phi::DDim dst_dims = meta.dims;
    dst_dims[0] = end_idx - begin_idx;
    size_t dst_offset =
        meta.offset + begin_idx * base * phi::SizeOf(meta.dtype);
    phi::DenseTensorMeta dst_meta = {
        meta.dtype, dst_dims, meta.layout, dst_offset};
    dst.set_meta(dst_meta);
    return dst;
  }
}

size_t Alignment(size_t size, const phi::Place &place, int align_size) {
  size_t alignment = 0;
  if (align_size > 0) {
    alignment = align_size;
  } else {
    alignment = 1 << 6;
  }
  size_t remaining = size % alignment;
  return remaining == 0 ? size : size + (alignment - remaining);
}

template <typename Context>
struct FillConstantVisitor {
  FillConstantVisitor(const Context &dev_ctx,
                      phi::DenseTensor *tensor,
                      const float value,
                      phi::DataType dtype)
      : dev_ctx_(dev_ctx), tensor_(tensor), value_(value), dtype_(dtype) {}

  template <typename T>
  void apply(typename std::enable_if<std::is_same<T, int8_t>::value ||
                                     std::is_same<T, int16_t>::value>::type * =
                 nullptr) const {
    PADDLE_THROW(phi::errors::InvalidArgument(
        "Not support data type for set_constant attr"));
  }

  template <typename T>
  void apply(typename std::enable_if<!(std::is_same<T, int8_t>::value ||
                                       std::is_same<T, int16_t>::value)>::type
                 * = nullptr) const {
    phi::DenseTensor tensor_tmp;
    phi::DenseTensorMeta meta = {dtype_, {1}};
    tensor_tmp.set_meta(meta);
    dev_ctx_.template Alloc<T>(&tensor_tmp);
    FillMLUTensorWithHostValue<T>(
        dev_ctx_, static_cast<T>(value_), &tensor_tmp);

    if (tensor_->numel() > 1) {
      tensor_->Resize(tensor_->dims());
      dev_ctx_.template Alloc<T>(tensor_);
      MLUCnnlTensorDesc tensor_desc(*tensor_);
      MLUCnnl::Fill(dev_ctx_,
                    CNNL_POINTER_MODE_DEVICE,
                    GetBasePtr(&tensor_tmp),
                    tensor_desc.get(),
                    GetBasePtr(tensor_));
    } else {
      // CANN op Fill/FillD would raise error when output's numel is 1.
      FillMLUTensorWithHostValue<T>(dev_ctx_, static_cast<T>(value_), tensor_);
    }
  }

  const Context &dev_ctx_;
  phi::DenseTensor *tensor_;
  float value_;
  phi::DataType dtype_;
};

void GetMemSizeAndDtype(
    const std::vector<const phi::DenseTensor *> &dense_tensors,
    size_t *numel,
    const size_t &size_of_dtype,
    const phi::Place &place,
    const bool use_align = true,
    const int align_size = -1) {
  *numel = 0;
  std::stringstream ss;
  ss << "alloc_space_for_vars: ";
  for (size_t i = 0; i < dense_tensors.size(); ++i) {
    auto size = dense_tensors[i]->numel();
    PADDLE_ENFORCE_GT(size,
                      0,
                      phi::errors::InvalidArgument(
                          "The number of `%d`-th tensor's elements is 0.", i));
    auto len = use_align ? custom_kernel::Alignment(
                               static_cast<size_t>(size) * size_of_dtype,
                               place,
                               align_size) /
                               size_of_dtype
                         : static_cast<size_t>(size);
    const void *ptr =
        dense_tensors[i]->initialized() ? dense_tensors[i]->data() : nullptr;
    VLOG(4) << size << " " << len;
    ss << "input(" << i << "-th tensor) dim:(" << dense_tensors[i]->dims()
       << ") "
       << " addres:" << ptr << " len: " << len << ", ";
    *numel += len;
  }
  VLOG(10) << ss.str();
}

template <typename T, typename Context>
void CoalesceTensorKernel(const Context &dev_ctx,
                          const std::vector<const phi::DenseTensor *> &input,
                          phi::DataType dtype,
                          bool copy_data,
                          bool set_constant,
                          bool persist_output,
                          float constant,
                          bool use_align,
                          int align_size,
                          int size_of_dtype,
                          const std::vector<int64_t> &concated_shapes,
                          const std::vector<int64_t> &concated_ranks,
                          std::vector<phi::DenseTensor *> output,
                          phi::DenseTensor *fused_output) {
  PADDLE_ENFORCE_GT(input.size(),
                    static_cast<size_t>(0),
                    phi::errors::InvalidArgument(
                        "The CoalesceTensor operator has no input."));
  PADDLE_ENFORCE_EQ(input.size(),
                    output.size(),
                    phi::errors::InvalidArgument(
                        "The number of CoalesceTensor operator's input and "
                        "output is not match, "
                        "input number is %u, output number is %u.",
                        input.size(),
                        output.size()));

  // Input & Output check: only support DenseTensor
  bool has_not_init_in_vars = false;
  for (size_t i = 0; i < input.size(); ++i) {
    PADDLE_ENFORCE_NOT_NULL(
        input[i],
        phi::errors::InvalidArgument(
            "The %d-th input tensor cannot be nullptr.", i));
    PADDLE_ENFORCE_NOT_NULL(
        output[i],
        phi::errors::InvalidArgument(
            "The %d-th output tensor cannot be nullptr.", i));
    if (!input[i]->initialized()) {
      has_not_init_in_vars = true;
    }
  }

  if (has_not_init_in_vars) {
    PADDLE_ENFORCE_EQ(concated_ranks.size(),
                      output.size(),
                      phi::errors::InvalidArgument(
                          "The attribute(concated_ranks) length must be "
                          "equal to the output tensor number."));
    int64_t accumulated_ranks = 0;
    for (size_t i = 0; i < input.size(); ++i) {
      phi::DDim dims(concated_shapes.data() + accumulated_ranks,
                     concated_ranks[i]);
      if (!input[i]->initialized()) {
        PADDLE_ENFORCE_EQ(
            input[i],
            output[i],
            phi::errors::InvalidArgument(
                "The %d-th output tensor and %d-th input tensor when the "
                "%d-th input tensor is not initialized.",
                i,
                i,
                i));
        output[i]->Resize(dims);
      } else {
        PADDLE_ENFORCE_EQ(input[i]->dims(),
                          dims,
                          phi::errors::InvalidArgument(
                              "The %d-th input tensor shape does not match the "
                              "attribute(concated_shapes) and "
                              "attribute(concated_ranks).",
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

  // Init the output as input
  for (size_t i = 0; i < input.size(); ++i) {
    output[i]->Resize(input[i]->dims());
  }

  // Get numel and dtype
  size_t numel = 0;

  if (size_of_dtype == -1) {
    size_of_dtype = phi::SizeOf(dtype);
  }
  GetMemSizeAndDtype(
      input, &numel, size_of_dtype, dev_ctx.GetPlace(), use_align, align_size);

  // Alloc the continuous space
  void *fused_tensor_ptr = dev_ctx.Alloc(
      &fused_output->Resize(phi::make_ddim({static_cast<int64_t>(numel)})),
      dtype);
  VLOG(10) << "Fused tensor addr " << fused_tensor_ptr;

  size_t offset = 0;
  if (copy_data) {
    phi::VisitDataType(
        dtype,
        FillConstantVisitor<Context>(
            dev_ctx, fused_output, static_cast<float>(0.0), dtype));
    for (size_t i = 0; i < input.size(); ++i) {
      size_t len = static_cast<size_t>(input[i]->numel());
      auto sub_tensor =
          custom_kernel::Slice(*fused_output,
                               static_cast<int64_t>(offset),
                               static_cast<int64_t>(offset + len));
      TensorCopy(dev_ctx, *input[i], true, &sub_tensor);

      offset += use_align
                    ? custom_kernel::Alignment(
                          len * size_of_dtype, dev_ctx.GetPlace(), align_size) /
                          size_of_dtype
                    : len;
    }
  } else if (set_constant) {
    phi::VisitDataType(
        dtype,
        FillConstantVisitor<Context>(dev_ctx, fused_output, constant, dtype));
  } else if (persist_output) {
    for (size_t i = 0; i < output.size(); ++i) {
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
  } else {
    phi::VisitDataType(
        dtype, FillConstantVisitor<Context>(dev_ctx, fused_output, 0, dtype));
  }
  // Make the outputs point to the continuous space.
  offset = 0;
  std::stringstream ss;
  ss << "alloc_space_for_vars: ";
  for (size_t i = 0; i < output.size(); ++i) {
    size_t len = static_cast<size_t>(output[i]->numel());
    auto dim = output[i]->dims();
    VLOG(4) << len << " " << dim << " " << offset;
    *output[i] = custom_kernel::Slice(*fused_output,
                                      static_cast<int64_t>(offset),
                                      static_cast<int64_t>(offset + len))
                     .Resize(dim);

    len = use_align ? custom_kernel::Alignment(
                          len * size_of_dtype, dev_ctx.GetPlace(), align_size) /
                          size_of_dtype
                    : len;
    ss << "output(" << i << "-th tensor) dim:(" << dim << ")"
       << " address: " << output[i]->data() << " len: " << len << ", ";
    offset += len;
  }
  PADDLE_ENFORCE_EQ((int64_t)offset,
                    fused_output->numel(),
                    phi::errors::InvalidArgument(
                        "The alloc_space_for_vars's offset: %s is unequal with "
                        "fused_output's numel: %s.",
                        offset,
                        fused_output->numel()));
  VLOG(10) << ss.str();
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(coalesce_tensor,
                          mlu,
                          ALL_LAYOUT,
                          custom_kernel::CoalesceTensorKernel,
                          int,
                          float,
                          phi::dtype::float16,
                          double) {
  kernel->InputAt(0).SetBackend(phi::Backend::ALL_BACKEND);
}
