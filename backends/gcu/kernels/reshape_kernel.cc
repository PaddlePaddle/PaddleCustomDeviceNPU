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
#include <cstring>

#include "common/gcu_op_runner.h"
#include "kernels/funcs/gcu_kernel_funcs.h"
// #include "paddle/phi/capi/all.h"

namespace custom_kernel {
namespace {
static phi::DDim ValidateShape(const std::vector<int64_t> shape,
                               const phi::DDim& in_dims) {
  const int64_t in_size = common::product(in_dims);
  auto in_dims_vec = common::vectorize(in_dims);
  std::vector<int64_t> output_shape(shape.size(), 0);
  int64_t capacity = 1;
  int unk_dim_idx = -1;

  for (size_t i = 0; i < shape.size(); ++i) {
    if (shape[i] == -1) {
      // only one dimension can be set to -1, whose size will be inferred.
      PADDLE_ENFORCE_EQ(
          unk_dim_idx,
          -1,
          phi::errors::InvalidArgument(
              "Only one dimension value of 'shape' in ReshapeOp can "
              "be -1. But received shape = [%s], shape[%d] is also -1.",
              common::make_ddim(shape),
              i));
      unk_dim_idx = static_cast<int>(i);
      output_shape[i] = shape[i];
    } else if (shape[i] == 0) {
      if (static_cast<int>(i) < in_dims.size()) {
        output_shape[i] = in_dims[static_cast<int>(i)];
      } else {
        PADDLE_ENFORCE_EQ(
            in_size,
            0,
            phi::errors::InvalidArgument("If The index of 0 in `shape` >= "
                                         "the input tensor X's dimensions, "
                                         "It can only be Zero-Sized Tensor"));
      }
      capacity *= output_shape[i];
    } else {
      PADDLE_ENFORCE_GT(
          shape[i],
          0,
          phi::errors::InvalidArgument(
              "Each dimension value of 'shape' in ReshapeOp must not "
              "be negative except one unknown dimension. "
              "But received  shape = [%s], shape[%d] = %d.",
              common::make_ddim(shape),
              i,
              shape[i]));
      output_shape[i] = shape[i];
      capacity *= output_shape[i];
    }
  }

  if (capacity == 0) {
    PADDLE_ENFORCE_EQ(in_size,
                      0,
                      phi::errors::InvalidArgument(
                          "Only Zero-Size Tensor'shape can contain 0"));
    PADDLE_ENFORCE_EQ(unk_dim_idx,
                      -1,
                      phi::errors::InvalidArgument(
                          "can not reshape %s to %s, because the unspecified "
                          "dimension %i can be any number and is ambiguous",
                          in_dims,
                          common::make_ddim(shape),
                          unk_dim_idx));
  }

  bool no_negative = std::all_of(in_dims_vec.cbegin(),
                                 in_dims_vec.cend(),
                                 [](int64_t i) { return i >= 0; });
  if (unk_dim_idx != -1) {
    // in compile time, no_negative may be False.
    if (no_negative) {
      output_shape[unk_dim_idx] = in_size / capacity;
      PADDLE_ENFORCE_EQ(
          output_shape[unk_dim_idx] * capacity,
          in_size,
          phi::errors::InvalidArgument(
              "The 'shape' attribute in ReshapeOp is invalid. "
              "The input tensor X'size must be divisible by known "
              "capacity of 'shape'. "
              "But received X's shape = [%s], X's size = %d, "
              "'shape' is [%s], known capacity of 'shape' is %d.",
              in_dims,
              in_size,
              common::make_ddim(shape),
              capacity));
    } else {
      // such as [-1, 8, 3]->[-1, 8], out_shape will remain [-1, 8]
      output_shape[unk_dim_idx] = -1;
    }
  } else {
    if (no_negative) {
      PADDLE_ENFORCE_EQ(
          capacity,
          in_size,
          phi::errors::InvalidArgument(
              "The 'shape' in ReshapeOp is invalid. "
              "The input tensor X'size must be equal to the capacity of "
              "'shape'. "
              "But received X's shape = [%s], X's size = %d, 'shape' is "
              "[%s], the capacity of 'shape' is %d.",
              in_dims,
              in_size,
              common::make_ddim(shape),
              capacity));
    }
  }

  return common::make_ddim(output_shape);
}

void InferMetaFromVecValue(const phi::DenseTensor& x,
                           const std::vector<int64_t>& shape,
                           phi::DenseTensor* out) {
  auto x_dims = x.dims();
  auto out_dims = ValidateShape(shape, x_dims);
  out->Resize(out_dims);
  auto meta = out->meta();
  meta.layout = common::DataLayout::kNCHW;
  meta.dtype = x.dtype();
  out->set_meta(meta);
  if (x_dims.size() > 0 && (x_dims[0] == out_dims[0])) {
    // Only pass LoD when the first dimension of output and Input(X)
    // are the same.
    out->ResetLoD(x.lod());
  }
}
}  // namespace

template <typename T, typename Context>
void ReshapeKernel(const Context& dev_ctx,
                   const phi::DenseTensor& x,
                   const phi::IntArray& shape,
                   phi::DenseTensor* out) {
  PADDLE_GCU_KERNEL_TRACE("reshape");
  PADDLE_ENFORCE_NE(
      x.layout(),
      common::DataLayout::kNDHWC,
      phi::errors::InvalidArgument("Reshape only support NCHW and NHWC "
                                   "format.But received NDHWC format."));

  InferMetaFromVecValue(x, shape.GetData(), out);
  // if numel() is 0, no need to do memcpy or nhwc to nchw
  if (out->numel() == 0) {
    return;
  }

  dev_ctx.Alloc(out, x.dtype());

  auto recovery = x;
  if (DataPdCustomNHWC(x)) {
    recovery = PdCustomNHWCTransToNCHW(dev_ctx, recovery);
  }
  if (out->data() == x.data()) {
    return;
  } else {
    auto meta = out->meta();
    TensorCopy(dev_ctx, recovery, false, out);
    out->set_meta(meta);
    out->ResetLoD(x.lod());
  }
}

template <typename T, typename Context>
void ReshapeWithXShapeKernel(const Context& dev_ctx,
                             const phi::DenseTensor& x,
                             const phi::IntArray& shape,
                             phi::DenseTensor* out,
                             phi::DenseTensor* xshape) {
  PADDLE_GCU_KERNEL_TRACE("reshape_with_xshape");
  ReshapeKernel<T>(dev_ctx, x, shape, out);
}

template <typename T, typename Context>
void ReshapeGradKernel(const Context& dev_ctx,
                       const phi::DenseTensor& out_grad,
                       phi::DenseTensor* x_grad) {
  PADDLE_GCU_KERNEL_TRACE("reshape_grad");
  dev_ctx.template Alloc<T>(x_grad);

  phi::DenseTensor* tmp_tensor = nullptr;
  if (LaunchAOTKernel()) {
    THROW_AOT_UNIMPLEMENTED();
  } else {  // kernel impl base on JIT
    auto out_shape = phi::vectorize<int64_t>(x_grad->dims());
    std::vector<int64_t> xshape = {0};
    for (auto dim : out_shape) {
      xshape.emplace_back(dim);
    }
    phi::DenseTensor x_shape;
    x_shape.Resize(phi::make_ddim(xshape));
    dev_ctx.template Alloc<T>(&x_shape);

    TensorNameMap input_names;
    input_names[GradVarName("Out")] = {"out_grad"};
    input_names["XShape"] = {"xshape"};

    TensorValueMap inputs;
    inputs[GradVarName("Out")] = {const_cast<DenseTensor*>(&out_grad)};
    inputs["XShape"] = {&x_shape};

    TensorNameMap output_names;
    output_names[GradVarName("X")] = {"x_grad"};

    TensorValueMap outputs;
    outputs[GradVarName("X")] = {(tmp_tensor == nullptr ? x_grad : tmp_tensor)};

    GcuAttributeMap attrs;

    GcuRunner(input_names,
              inputs,
              output_names,
              outputs,
              attrs,
              "reshape2_grad",
              dev_ctx);
  }
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(reshape,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::ReshapeKernel,
                          float,
                          phi::dtype::float16,
                          double,
                          int8_t,
                          int16_t,
                          int32_t,
                          int64_t,
                          uint8_t,
                          bool) {}

PD_REGISTER_PLUGIN_KERNEL(reshape_with_xshape,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::ReshapeWithXShapeKernel,
                          float,
                          phi::dtype::float16,
                          double,
                          int8_t,
                          int16_t,
                          int32_t,
                          int64_t,
                          uint8_t,
                          bool) {}
// PD_REGISTER_PLUGIN_KERNEL(reshape_grad,
//                           gcu,
//                           ALL_LAYOUT,
//                           custom_kernel::ReshapeGradKernel,
//                           float,
//                           phi::dtype::float16,
//                           double,
//                           int8_t,
//                           int16_t,
//                           int32_t,
//                           int64_t,
//                           uint8_t,
//                           bool) {}
