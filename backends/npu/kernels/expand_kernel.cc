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

namespace custom_kernel {

template <typename T, typename Context>
void CastKernel(const Context& dev_ctx,
                const phi::DenseTensor& x,
                phi::DataType dtype,
                phi::DenseTensor* out);

template <typename T, typename Context>
void SumKernel(const Context& dev_ctx,
               const phi::DenseTensor& x,
               const phi::IntArray& dims,
               phi::DataType out_dtype,
               bool keep_dim,
               phi::DenseTensor* out);

template <typename T, typename Context>
void AclopExpandKernel(const Context& dev_ctx,
                       const phi::DenseTensor& x,
                       const phi::IntArray& shape,
                       phi::DenseTensor* out) {
  auto in_dims = x.dims();
  auto expand_shape = shape.GetData();
  auto vec_in_dims = phi::vectorize<int>(in_dims);
  auto diff = expand_shape.size() - vec_in_dims.size();
  vec_in_dims.insert(vec_in_dims.begin(), diff, 1);
  std::vector<int> final_expand_shape(vec_in_dims.size());
  for (size_t i = 0; i < vec_in_dims.size(); ++i) {
    PADDLE_ENFORCE_NE(
        expand_shape[i],
        0,
        phi::errors::InvalidArgument("The expanded size cannot be zero."));
    if (i < diff) {  // expand_shape = [3,4,-1,-1], x = [10,2] -->
                     // final_expand_shape = [3,4,10,2]
      PADDLE_ENFORCE_GT(
          expand_shape[i],
          0,
          phi::errors::InvalidArgument(
              "The expanded size (%d) for non-existing dimensions must be "
              "positive for expand_v2 op.",
              expand_shape[i]));
      final_expand_shape[i] = expand_shape[i];
    } else if (expand_shape[i] > 0) {  // expand_shape = [3,4,10,4], x =
                                       // [10,1] --> final_expand_shape =
                                       // [3,4,10,4]
      if (vec_in_dims[i] != 1) {
        PADDLE_ENFORCE_EQ(
            vec_in_dims[i],
            expand_shape[i],
            phi::errors::InvalidArgument(
                "The value (%d) of the non-singleton dimension does not match"
                " the corresponding value (%d) in shape for expand_v2 op.",
                vec_in_dims[i],
                expand_shape[i]));
        final_expand_shape[i] = expand_shape[i];
      } else {
        final_expand_shape[i] = expand_shape[i];
      }
    } else {  // expand_shape = [3,4,-1,-1], x = [10,2] --> final_expand_shape
              // = [3,4,10,2]
      PADDLE_ENFORCE_EQ(
          expand_shape[i],
          -1,
          phi::errors::InvalidArgument(
              "When the value in shape is negative for expand_v2 op, "
              "only -1 is supported, but the value received is %d.",
              expand_shape[i]));
      final_expand_shape[i] = vec_in_dims[i];
    }
  }

  NPUAttributeMap attr_input = {{"shape", final_expand_shape}};

  auto rank = x.dims().size();

  PADDLE_ENFORCE_GE(
      rank,
      0,
      phi::errors::InvalidArgument(
          "The rank of the input 'x' for expand_v2_npu op must be positive, "
          "but the value received is %d.",
          rank));
  PADDLE_ENFORCE_LE(
      rank,
      MAX_RANK_SUPPORTED,
      phi::errors::InvalidArgument(
          "The rank of the input 'x' for expand_v2_npu op must be less than "
          "or equal to %d, but the value received is %d.",
          MAX_RANK_SUPPORTED,
          rank));
  auto shape_size = final_expand_shape.size();
  PADDLE_ENFORCE_GE(
      shape_size,
      rank,
      phi::errors::InvalidArgument(
          "The number (%d) of elements of 'shape' for expand_v2_npu op must "
          "be "
          "greater than or equal to the rank (%d) of the input 'x'.",
          shape_size,
          rank));
  PADDLE_ENFORCE_LE(
      shape_size,
      MAX_RANK_SUPPORTED,
      phi::errors::InvalidArgument("The number (%d) of elements of 'shape' for "
                                   "expand_v2_npu op must be "
                                   "less than or equal to %d.",
                                   shape_size,
                                   MAX_RANK_SUPPORTED));

  phi::DDim out_dims = phi::make_ddim(final_expand_shape);
  out->Resize(out_dims);
  dev_ctx.template Alloc<T>(out);

  auto op_func = [](const std::vector<phi::DenseTensor>& inputs,
                    const std::vector<phi::DenseTensor>& outputs,
                    const NPUAttributeMap& attrs,
                    const Context& dev_ctx) {
    const auto& runner = NpuOpRunner("Expand", inputs, outputs, attrs);
    runner.Run(dev_ctx.stream());
  };

  if (x.dtype() == phi::DataType::BOOL) {
    NpuOpRunner::TypeAdapter({x},
                             {*out},
                             attr_input,
                             dev_ctx,
                             op_func,
                             {phi::DataType::UINT8},
                             {phi::DataType::UINT8});
  } else if (x.dtype() == phi::DataType::INT64) {
    NpuOpRunner::TypeAdapter({x},
                             {*out},
                             attr_input,
                             dev_ctx,
                             op_func,
                             {phi::DataType::INT32},
                             {phi::DataType::INT32});
  } else if (x.dtype() == phi::DataType::FLOAT64) {
    NpuOpRunner::TypeAdapter({x},
                             {*out},
                             attr_input,
                             dev_ctx,
                             op_func,
                             {phi::DataType::FLOAT32},
                             {phi::DataType::FLOAT32});
  } else if (x.dtype() == phi::DataType::BFLOAT16) {
    NpuOpRunner::TypeAdapter({x},
                             {*out},
                             attr_input,
                             dev_ctx,
                             op_func,
                             {phi::DataType::FLOAT32},
                             {phi::DataType::FLOAT32});
  } else {
    const auto& runner = NpuOpRunner("Expand", {x}, {*out}, attr_input);
    runner.Run(dev_ctx.stream());
  }
}

template <typename T, typename Context>
void ExpandKernel(const Context& dev_ctx,
                  const phi::DenseTensor& x,
                  const phi::IntArray& shape,
                  phi::DenseTensor* out) {
  DO_COMPATIBILITY(
      aclnnExpand,
      (custom_kernel::AclopExpandKernel<T, Context>(dev_ctx, x, shape, out)));

  auto in_dims = x.dims();
  auto expand_shape = shape.GetData();
  auto vec_in_dims = phi::vectorize<int>(in_dims);
  auto diff = expand_shape.size() - vec_in_dims.size();
  vec_in_dims.insert(vec_in_dims.begin(), diff, 1);
  std::vector<int64_t> final_expand_shape(vec_in_dims.size());

  for (size_t i = 0; i < vec_in_dims.size(); ++i) {
    PADDLE_ENFORCE_NE(
        expand_shape[i],
        0,
        phi::errors::InvalidArgument("The expanded size cannot be zero."));
    if (i < diff) {  // expand_shape = [3,4,-1,-1], x = [10,2] -->
                     // final_expand_shape = [3,4,10,2]
      PADDLE_ENFORCE_GT(
          expand_shape[i],
          0,
          phi::errors::InvalidArgument(
              "The expanded size (%d) for non-existing dimensions must be "
              "positive for expand_v2 op.",
              expand_shape[i]));
      final_expand_shape[i] = expand_shape[i];
    } else if (expand_shape[i] > 0) {  // expand_shape = [3,4,10,4], x =
                                       // [10,1] --> final_expand_shape =
                                       // [3,4,10,4]
      if (vec_in_dims[i] != 1) {
        PADDLE_ENFORCE_EQ(
            vec_in_dims[i],
            expand_shape[i],
            phi::errors::InvalidArgument(
                "The value (%d) of the non-singleton dimension does not match"
                " the corresponding value (%d) in shape for expand_v2 op.",
                vec_in_dims[i],
                expand_shape[i]));
        final_expand_shape[i] = expand_shape[i];
      } else {
        final_expand_shape[i] = expand_shape[i];
      }
    } else {  // expand_shape = [3,4,-1,-1], x = [10,2] --> final_expand_shape
              // = [3,4,10,2]
      PADDLE_ENFORCE_EQ(
          expand_shape[i],
          -1,
          phi::errors::InvalidArgument(
              "When the value in shape is negative for expand_v2 op, "
              "only -1 is supported, but the value received is %d.",
              expand_shape[i]));
      final_expand_shape[i] = vec_in_dims[i];
    }
  }

  // NPUAttributeMap attr_input = {{"shape", final_expand_shape}};

  auto rank = x.dims().size();

  PADDLE_ENFORCE_GE(
      rank,
      0,
      phi::errors::InvalidArgument(
          "The rank of the input 'x' for expand_v2_npu op must be positive, "
          "but the value received is %d.",
          rank));
  PADDLE_ENFORCE_LE(
      rank,
      MAX_RANK_SUPPORTED,
      phi::errors::InvalidArgument(
          "The rank of the input 'x' for expand_v2_npu op must be less than "
          "or equal to %d, but the value received is %d.",
          MAX_RANK_SUPPORTED,
          rank));
  auto shape_size = final_expand_shape.size();
  PADDLE_ENFORCE_GE(
      shape_size,
      rank,
      phi::errors::InvalidArgument(
          "The number (%d) of elements of 'shape' for expand_v2_npu op must "
          "be "
          "greater than or equal to the rank (%d) of the input 'x'.",
          shape_size,
          rank));
  PADDLE_ENFORCE_LE(
      shape_size,
      MAX_RANK_SUPPORTED,
      phi::errors::InvalidArgument("The number (%d) of elements of 'shape' for "
                                   "expand_v2_npu op must be "
                                   "less than or equal to %d.",
                                   shape_size,
                                   MAX_RANK_SUPPORTED));

  phi::DDim out_dims = phi::make_ddim(final_expand_shape);
  out->Resize(out_dims);

  if (out_dims.size() == 0) {
    out->Resize(phi::make_ddim({1}));
    final_expand_shape = {1};
  }

  phi::DenseTensor x_trans(x);
  if (x.dims().size() == 0) {
    phi::DenseTensorMeta meta_1 = {x.dtype(), phi::make_ddim({1})};
    x_trans.set_meta(meta_1);
  }

  dev_ctx.template Alloc<T>(out);
  if (x_trans.dtype() == phi::DataType::FLOAT64) {
    phi::DenseTensor cast_x;
    phi::DenseTensorMeta cast_x_meta = {phi::DataType::FLOAT32, x_trans.dims()};
    cast_x.set_meta(cast_x_meta);
    dev_ctx.template Alloc<float>(&cast_x);

    phi::DenseTensor cast_out;
    phi::DenseTensorMeta cast_out_meta = {phi::DataType::FLOAT32, out->dims()};
    cast_out.set_meta(cast_out_meta);
    dev_ctx.template Alloc<float>(&cast_out);

    custom_kernel::CastKernel<T, Context>(
        dev_ctx, x_trans, phi::DataType::FLOAT32, &cast_x);
    EXEC_NPU_CMD(aclnnExpand, dev_ctx, cast_x, final_expand_shape, cast_out);
    custom_kernel::CastKernel<T, Context>(
        dev_ctx, cast_out, phi::DataType::FLOAT64, out);
  } else {
    EXEC_NPU_CMD(aclnnExpand, dev_ctx, x_trans, final_expand_shape, *out);
  }

  if (out_dims.size() == 0) {
    out->Resize(out_dims);
  }
}

template <typename T, typename Context>
void ExpandGradKernel(const Context& dev_ctx,
                      const phi::DenseTensor& x,
                      const phi::DenseTensor& out_grad,
                      const phi::IntArray& shape,
                      phi::DenseTensor* in_grad) {
  dev_ctx.template Alloc<T>(in_grad);
  auto stream = dev_ctx.stream();

  // case 1: reduce out_grad dims to in_grad dims
  // For example: [2, 120] --> [120]
  auto reduce_ndim = out_grad.dims().size() - in_grad->dims().size();
  std::vector<int> axes;
  for (auto i = 0; i < reduce_ndim; ++i) {
    axes.push_back(i);
  }

  phi::DenseTensor tmp_out_grad(out_grad);
  phi::DenseTensor reduced_out_grad;
  (in_grad->dtype());
  if (axes.size() != 0) {
    std::vector<int64_t> reduced_out_grad_dims;
    for (auto i = reduce_ndim; i < out_grad.dims().size(); ++i) {
      reduced_out_grad_dims.push_back(out_grad.dims()[i]);
    }
    tmp_out_grad.Resize(phi::make_ddim(reduced_out_grad_dims));
    phi::DenseTensorMeta meta = {in_grad->dtype(),
                                 phi::make_ddim(reduced_out_grad_dims)};
    reduced_out_grad.set_meta(meta);

    custom_kernel::SumKernel<T, Context>(dev_ctx,
                                         out_grad,
                                         phi::IntArray(axes),
                                         out_grad.dtype(),
                                         false,
                                         &reduced_out_grad);

    tmp_out_grad = reduced_out_grad;
  }

  // case 2: reduce axis of out_grad in which dim is 1
  // For example: [12, 140] --> [1, 140]

  // case 3: copy out_grad to in_grad when shape is totally same, and dim in
  // in_grad != 1 For example: [2, 10, 5] --> [2, 10, 5]
  axes.clear();
  for (auto i = 0; i < in_grad->dims().size(); ++i) {
    if (in_grad->dims()[i] == 1) {
      axes.push_back(i);
    }
  }
  if (axes.size() != 0) {
    custom_kernel::SumKernel<T, Context>(dev_ctx,
                                         tmp_out_grad,
                                         phi::IntArray(axes),
                                         tmp_out_grad.dtype(),
                                         true,
                                         in_grad);
  } else {
    TensorCopy(dev_ctx, tmp_out_grad, true, in_grad);
  }
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(expand,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::ExpandKernel,
                          bool,
                          int,
                          int64_t,
                          double,
                          float,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}

PD_REGISTER_PLUGIN_KERNEL(expand_grad,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::ExpandGradKernel,
                          int,
                          float,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}
