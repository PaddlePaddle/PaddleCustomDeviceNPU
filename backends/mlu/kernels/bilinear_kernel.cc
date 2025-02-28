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

#include "kernels/funcs/elementwise_utils.h"
#include "kernels/funcs/mlu_baseop.h"
#include "kernels/funcs/mlu_funcs.h"
#include "kernels/funcs/reduce_op.h"
#include "paddle/phi/kernels/funcs/slice_utils.h"

namespace custom_kernel {

template <typename T, typename Context>
void SetTensorValueKernel(const Context& dev_ctx,
                          const phi::DenseTensor& x,
                          const phi::DenseTensor& value,
                          const phi::IntArray& starts,
                          const phi::IntArray& ends,
                          const phi::IntArray& steps,
                          const std::vector<int64_t>& axes,
                          const std::vector<int64_t>& decrease_axes,
                          const std::vector<int64_t>& none_axes,
                          phi::DenseTensor* out);

template <typename T, typename Context>
void StridedSliceRawKernel(const Context& dev_ctx,
                           const phi::DenseTensor& x,
                           const std::vector<int>& axes,
                           const phi::IntArray& starts,
                           const phi::IntArray& ends,
                           const phi::IntArray& strides,
                           const std::vector<int>& infer_flags,
                           const std::vector<int>& decrease_axis,
                           phi::DenseTensor* out);

template <typename T, typename Context>
void BilinearKernel(const Context& dev_ctx,
                    const phi::DenseTensor& x,
                    const phi::DenseTensor& y,
                    const phi::DenseTensor& weight,
                    const paddle::optional<phi::DenseTensor>& bias,
                    phi::DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);

  auto batch_size = x.dims()[0];
  auto weight_dims = weight.dims();
  int out_dim = weight_dims[0];
  auto x_dim = weight_dims[1];
  auto y_dim = weight_dims[2];

  // Create the intermediate variable to calculate the result of
  // Input(X) multiplied by Input(Weight_i), the formula is:
  // left_mul = X Weight_i.
  Tensor left_mul;
  left_mul.Resize(phi::make_ddim({batch_size, y_dim}));
  dev_ctx.template Alloc<T>(&left_mul);

  MLUCnnlTensorDesc x_desc(x, CNNL_LAYOUT_ARRAY, ToCnnlDataType<T>());
  MLUCnnlTensorDesc y_desc(x, CNNL_LAYOUT_ARRAY, ToCnnlDataType<T>());
  MLUCnnlTensorDesc weight_desc(weight, CNNL_LAYOUT_ARRAY, ToCnnlDataType<T>());
  MLUCnnlTensorDesc left_mul_desc(
      left_mul, CNNL_LAYOUT_ARRAY, ToCnnlDataType<T>());

  phi::DenseTensor output_mat_slice;
  output_mat_slice.Resize(phi::make_ddim({batch_size}));

  phi::DenseTensor out_temp;
  out_temp.Resize(out->dims());
  dev_ctx.template Alloc<T>(&out_temp);
  FillMLUTensorWithHostValue(dev_ctx, static_cast<T>(0.0f), &out_temp);

  for (int64_t i = 0; i < out_dim; ++i) {
    phi::DenseTensor weight_slice;
    weight_slice.Resize(phi::make_ddim({x_dim, y_dim}));
    dev_ctx.template Alloc<T>(&weight_slice);
    MLUCnnlTensorDesc weight_slice_desc(weight_slice);

    phi::DenseTensor matmul_out;
    matmul_out.Resize(phi::make_ddim({batch_size, y_dim}));
    dev_ctx.template Alloc<T>(&matmul_out);
    MLUCnnlTensorDesc matmul_out_desc(matmul_out);
    int64_t next_i = i + 1;
    int64_t value = 1;
    const phi::IntArray& starts_indices = {i};
    const phi::IntArray& ends_indices = {next_i};
    const phi::IntArray& strides_indices = {value};
    std::vector<int> infer_flags(1);
    std::vector<int> decrease_axis;
    std::vector<int> axes = {0};
    custom_kernel::StridedSliceRawKernel<T, Context>(dev_ctx,
                                                     weight,
                                                     axes,
                                                     starts_indices,
                                                     ends_indices,
                                                     strides_indices,
                                                     infer_flags,
                                                     decrease_axis,
                                                     &weight_slice);

    MLUCnnl::Matmul(dev_ctx,
                    false,
                    false,
                    x_desc.get(),
                    GetBasePtr(&x),
                    weight_slice_desc.get(),
                    GetBasePtr(&weight_slice),
                    left_mul_desc.get(),
                    GetBasePtr(&left_mul));

    int axis = -1;
    MLUOpTensorKernel<T>(
        dev_ctx, left_mul, y, axis, CNNL_OP_TENSOR_MUL, &matmul_out);

    phi::DenseTensor sum_out;
    sum_out.Resize({batch_size});
    const std::vector<int64_t>& dims = {1};
    MLUReduceOp<T>(dev_ctx,
                   matmul_out,
                   dims,
                   false,
                   /*keep_dim*/ false,
                   /*reduce_all*/ "reduce_sum",
                   &sum_out);

    std::vector<int64_t> sum_axes = {1};
    std::vector<int64_t> decrease_axes;
    std::vector<int64_t> none_axes;
    custom_kernel::SetTensorValueKernel<T, Context>(dev_ctx,
                                                    *&out_temp,
                                                    sum_out,
                                                    starts_indices,
                                                    ends_indices,
                                                    strides_indices,
                                                    sum_axes,
                                                    decrease_axes,
                                                    none_axes,
                                                    &output_mat_slice);
  }

  if (bias.get_ptr()) {
    phi::DenseTensor new_bias;
    new_bias = bias.get();
    int axis = -1;
    MLUOpTensorKernel<T>(
        dev_ctx, out_temp, new_bias, axis, CNNL_OP_TENSOR_ADD, out);
  } else {
    TensorCopy(dev_ctx, out_temp, false, out);
  }
}

template <typename T, typename Context>
void BilinearGradKernel(const Context& dev_ctx,
                        const phi::DenseTensor& x,
                        const phi::DenseTensor& y,
                        const phi::DenseTensor& weight,
                        const phi::DenseTensor& dout,
                        phi::DenseTensor* dx,
                        phi::DenseTensor* dy,
                        phi::DenseTensor* dweight,
                        phi::DenseTensor* dbias) {
  auto batch_size = x.dims()[0];
  auto weight_dims = weight.dims();
  int out_dim = weight_dims[0];
  auto x_dim = weight_dims[1];
  auto y_dim = weight_dims[2];

  // Create the intermediate variable to calculate the Output(Y@Grad).
  phi::DenseTensor x_scale;
  x_scale.Resize(phi::make_ddim({batch_size, x_dim}));
  dev_ctx.template Alloc<T>(&x_scale);

  // Create the intermediate variable to calculate the Output(X@Grad).
  phi::DenseTensor y_scale;
  y_scale.Resize(phi::make_ddim({batch_size, y_dim}));
  dev_ctx.template Alloc<T>(&y_scale);

  if (dx) {
    dev_ctx.template Alloc<T>(dx);
    FillMLUTensorWithHostValue(dev_ctx, static_cast<T>(0.0f), dx);
  }
  if (dy) {
    dev_ctx.template Alloc<T>(dy);
    FillMLUTensorWithHostValue(dev_ctx, static_cast<T>(0.0f), dy);
  }
  if (dweight) {
    dev_ctx.template Alloc<T>(dweight);
    FillMLUTensorWithHostValue(dev_ctx, static_cast<T>(0.0f), dweight);
  }

  if (dx || dy || dweight) {
    phi::DenseTensor dx_temp;
    dx_temp.Resize(dx->dims());
    dev_ctx.template Alloc<T>(&dx_temp);
    MLUCnnlTensorDesc dx_temp_desc(dx_temp);

    phi::DenseTensor dy_temp;
    dy_temp.Resize(dy->dims());
    dev_ctx.template Alloc<T>(&dy_temp);
    MLUCnnlTensorDesc dy_temp_desc(dy_temp);

    phi::DenseTensor dweight_temp;
    dweight_temp.Resize(phi::make_ddim({x_dim, y_dim}));
    dev_ctx.template Alloc<T>(&dweight_temp);
    MLUCnnlTensorDesc dweight_temp_desc(dweight_temp);

    for (int64_t i = 0; i < out_dim; ++i) {
      phi::DenseTensor weight_slice;
      weight_slice.Resize(phi::make_ddim({x_dim, y_dim}));
      dev_ctx.template Alloc<T>(&weight_slice);
      int64_t next_i = i + 1;
      int64_t value = 1;
      const phi::IntArray& starts_indices = {i};
      const phi::IntArray& ends_indices = {next_i};
      const phi::IntArray& strides_indices = {value};
      std::vector<int> infer_flags(1);
      std::vector<int> decrease_axis;
      std::vector<int> axes = {0};
      custom_kernel::StridedSliceRawKernel<T, Context>(dev_ctx,
                                                       weight,
                                                       axes,
                                                       starts_indices,
                                                       ends_indices,
                                                       strides_indices,
                                                       infer_flags,
                                                       decrease_axis,
                                                       &weight_slice);
      weight_slice.Resize(phi::make_ddim({x_dim, y_dim}));
      MLUCnnlTensorDesc weight_slice_desc(weight_slice);
      MLUCnnlTensorDesc x_scale_desc(x_scale);
      MLUCnnlTensorDesc y_scale_desc(y_scale);
      MLUCnnlTensorDesc dx_desc(*dx);
      MLUCnnlTensorDesc dy_desc(*dy);
      MLUCnnlTensorDesc y_desc(y);

      // dout[:, i]
      std::vector<int> dout_axes = {1};
      std::vector<int> decrease_axes;
      phi::DenseTensor dout_mat_slice;
      dout_mat_slice.Resize(phi::make_ddim({batch_size}));
      custom_kernel::StridedSliceRawKernel<T, Context>(dev_ctx,
                                                       dout,
                                                       dout_axes,
                                                       starts_indices,
                                                       ends_indices,
                                                       strides_indices,
                                                       infer_flags,
                                                       decrease_axis,
                                                       &dout_mat_slice);
      if (dx) {
        int axis = -1;
        dout_mat_slice.Resize({batch_size, 1});
        MLUCnnlTensorDesc dout_mat_slice_desc(dout_mat_slice);
        MLUOpTensorKernel<T>(
            dev_ctx, dout_mat_slice, y, axis, CNNL_OP_TENSOR_MUL, &y_scale);
        MLUCnnl::Matmul(dev_ctx,
                        false,
                        true,
                        y_scale_desc.get(),
                        GetBasePtr(&y_scale),
                        weight_slice_desc.get(),
                        GetBasePtr(&weight_slice),
                        dx_temp_desc.get(),
                        GetBasePtr(&dx_temp));
        MLUOpTensorKernel<T>(
            dev_ctx, dx_temp, *dx, axis, CNNL_OP_TENSOR_ADD, dx);
      }
      if (dy || dweight) {
        int axis = -1;
        dout_mat_slice.Resize({batch_size, 1});
        MLUCnnlTensorDesc dout_mat_slice_desc(dout_mat_slice);
        MLUOpTensorKernel<T>(
            dev_ctx, dout_mat_slice, x, axis, CNNL_OP_TENSOR_MUL, &x_scale);
        if (dy) {
          MLUCnnl::Matmul(dev_ctx,
                          false,
                          false,
                          x_scale_desc.get(),
                          GetBasePtr(&x_scale),
                          weight_slice_desc.get(),
                          GetBasePtr(&weight_slice),
                          dy_temp_desc.get(),
                          GetBasePtr(&dy_temp));
          MLUOpTensorKernel<T>(
              dev_ctx, dy_temp, *dy, axis, CNNL_OP_TENSOR_ADD, dy);
        }
        if (dweight) {
          MLUCnnl::Matmul(dev_ctx,
                          true,
                          false,
                          x_scale_desc.get(),
                          GetBasePtr(&x_scale),
                          y_desc.get(),
                          GetBasePtr(&y),
                          dweight_temp_desc.get(),
                          GetBasePtr(&dweight_temp));

          std::vector<int64_t> dweight_axes = {0};
          std::vector<int64_t> decrease_axes;
          std::vector<int64_t> none_axes;
          phi::DenseTensor dweight_slice;
          dweight_slice.Resize(phi::make_ddim({x_dim, y_dim}));
          dev_ctx.template Alloc<T>(&dweight_slice);
          MLUCnnlTensorDesc dweight_slice_desc(dweight_slice);
          custom_kernel::SetTensorValueKernel<T, Context>(dev_ctx,
                                                          *dweight,
                                                          dweight_temp,
                                                          starts_indices,
                                                          ends_indices,
                                                          strides_indices,
                                                          dweight_axes,
                                                          decrease_axes,
                                                          none_axes,
                                                          &dweight_slice);
        }
      }
    }
    // calculate the gradient of Input(Bias).
    if (dbias) {
      dev_ctx.template Alloc<T>(dbias);
      const std::vector<int64_t>& dims = {0};
      MLUReduceOp<T>(dev_ctx,
                     dout,
                     dims,
                     false, /*keep_dim*/
                     false, /*reduce_all*/
                     "reduce_sum",
                     dbias);
    }
  }
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(
    bilinear, mlu, ALL_LAYOUT, custom_kernel::BilinearKernel, float, double) {}

PD_REGISTER_PLUGIN_KERNEL(bilinear_grad,
                          mlu,
                          ALL_LAYOUT,
                          custom_kernel::BilinearGradKernel,
                          float,
                          double) {}
