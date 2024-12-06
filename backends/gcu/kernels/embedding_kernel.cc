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
void FullKernel(const Context& dev_ctx,
                const phi::IntArray& shape,
                const phi::Scalar& val,
                phi::DataType dtype,
                phi::DenseTensor* out);
template <typename T, typename Context>
void EqualKernel(const Context& dev_ctx,
                 const phi::DenseTensor& x,
                 const phi::DenseTensor& y,
                 phi::DenseTensor* out);
template <typename T, typename Context>
void WhereKernel(const Context& dev_ctx,
                 const phi::DenseTensor& condition,
                 const phi::DenseTensor& x,
                 const phi::DenseTensor& y,
                 phi::DenseTensor* out);

template <typename T, typename Context>
void EmbeddingKernel(const Context& dev_ctx,
                     const phi::DenseTensor& inputx,
                     const phi::DenseTensor& weight,
                     int64_t padding_idx,
                     phi::DenseTensor* out) {
  PADDLE_GCU_KERNEL_TRACE("embedding");
  dev_ctx.template Alloc<T>(out);

  if (LaunchAOTKernel()) {
    phi::DenseTensor x = MaybeCreateOrTrans64To32bits(dev_ctx, inputx);
    LAUNCH_TOPSATENOP(
        topsatenEmbedding, dev_ctx, *out, weight, x, -1, false, false);
    if (padding_idx == -1) {
      return;
    }
    PADDLE_ENFORCE_EQ(
        x.dtype(),
        phi::DataType::INT32,
        phi::errors::Unimplemented(
            "The input tensor's dtype should be INT32 but get %s.",
            phi::DataTypeToString(x.dtype()).c_str()));
    // padding_idx is not -1
    // implement padding_idx by using where kernel
    // out = x_brd == padding_idx ? 0 : topsatenEmbedding(weight, x, -1)
    phi::DenseTensor pad_tensor;
    phi::DenseTensor zero_tensor;
    phi::DenseTensor mask_tensor;
    phi::IntArray shape(common::vectorize<int64_t>(out->dims()));
    phi::DenseTensorMeta meta_info = x.meta();
    meta_info.dims = out->dims();
    meta_info.strides = phi::DenseTensorMeta::calc_strides(meta_info.dims);
    pad_tensor.set_meta(meta_info);
    zero_tensor.set_meta(meta_info);
    custom_kernel::FullKernel<int32_t, phi::CustomContext>(
        dev_ctx, shape, phi::Scalar(padding_idx), x.dtype(), &pad_tensor);
    custom_kernel::FullKernel<T, phi::CustomContext>(
        dev_ctx, shape, phi::Scalar(0), x.dtype(), &zero_tensor);
    meta_info.dtype = phi::DataType::BOOL;
    mask_tensor.set_meta(meta_info);
    phi::DenseTensor x_brd;
    // x firstly expand to the same shape with pad_tensor for broadcast by
    // adding a new dimension on last
    phi::DenseTensor x_expand = x;
    auto x_expand_meta = x_expand.meta();
    auto x_expand_shape = common::vectorize<int64_t>(x_expand_meta.dims);
    x_expand_shape.push_back(1);
    x_expand_meta.dims = common::make_ddim(x_expand_shape);
    x_expand_meta.strides =
        phi::DenseTensorMeta::calc_strides(x_expand_meta.dims);
    x_expand.set_meta(x_expand_meta);
    x_brd.set_meta(pad_tensor.meta());
    dev_ctx.Alloc(&x_brd, x_brd.dtype());
    custom_kernel::Broadcast(dev_ctx, x_expand, &x_brd);
    custom_kernel::EqualKernel<bool, phi::CustomContext>(
        dev_ctx, pad_tensor, x_brd, &mask_tensor);
    pad_tensor.set_meta(out->meta());
    custom_kernel::WhereKernel<T, phi::CustomContext>(
        dev_ctx, mask_tensor, zero_tensor, *out, &pad_tensor);
    *out = pad_tensor;

  } else {  // kernel impl base on JIT
    TensorNameMap input_names;
    input_names["Ids"] = {"inputx"};
    input_names["W"] = {"weight"};

    TensorValueMap inputs;
    inputs["Ids"] = {const_cast<DenseTensor*>(&inputx)};
    inputs["W"] = {const_cast<DenseTensor*>(&weight)};

    TensorNameMap output_names;
    output_names["Out"] = {"out"};

    TensorValueMap outputs;
    outputs["Out"] = {out};

    GcuAttributeMap attrs;
    attrs["padding_idx"] = padding_idx;

    GcuRunner(input_names,
              inputs,
              output_names,
              outputs,
              attrs,
              "lookup_table_v2",
              dev_ctx);
  }
}

template <typename T, typename Context>
void EmbeddingGradKernel(const Context& dev_ctx,
                         const phi::DenseTensor& input,
                         const phi::DenseTensor& weight,
                         const phi::DenseTensor& out_grad,
                         int64_t padding_idx,
                         phi::DenseTensor* weight_grad) {
  PADDLE_GCU_KERNEL_TRACE("embedding_grad");
  dev_ctx.template Alloc<T>(weight_grad);

  if (LaunchAOTKernel()) {
    THROW_AOT_UNIMPLEMENTED();
  } else {  // kernel impl base on JIT
    TensorNameMap input_names;
    input_names["Ids"] = {"input"};
    input_names["W"] = {"weight"};
    input_names[GradVarName("Out")] = {"out_grad"};

    TensorValueMap inputs;
    inputs["Ids"] = {const_cast<DenseTensor*>(&input)};
    inputs["W"] = {const_cast<DenseTensor*>(&weight)};
    inputs[GradVarName("Out")] = {const_cast<DenseTensor*>(&out_grad)};

    TensorNameMap output_names;
    output_names[GradVarName("Weight")] = {"weight_grad"};

    TensorValueMap outputs;
    outputs[GradVarName("Weight")] = {weight_grad};

    GcuAttributeMap attrs;
    attrs["padding_idx"] = padding_idx;

    GcuRunner(input_names,
              inputs,
              output_names,
              outputs,
              attrs,
              "lookup_table_v2_grad",
              dev_ctx);
  }
}
}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(embedding,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::EmbeddingKernel,
                          float,
                          int,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(embedding_grad,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::EmbeddingGradKernel,
                          float,
                          int,
                          phi::dtype::float16) {}
