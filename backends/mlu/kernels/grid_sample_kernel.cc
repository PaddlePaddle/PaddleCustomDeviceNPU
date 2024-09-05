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
void GridSampleKernel(const Context& dev_ctx,
                      const phi::DenseTensor& x,
                      const phi::DenseTensor& grid,
                      const std::string& mode,
                      const std::string& padding_mode,
                      bool align_corners,
                      phi::DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);

  int n = x.dims()[0];
  int c = x.dims()[1];
  int out_h = grid.dims()[1];
  int out_w = grid.dims()[2];

  PADDLE_ENFORCE_EQ(
      mode == "bilinear",
      true,
      phi::errors::Unavailable(
          "Only support bilinear mode in mlu grid_sample kernel."));
  PADDLE_ENFORCE_EQ(
      padding_mode == "zeros",
      true,
      phi::errors::Unavailable(
          "Only support zeros padding_mode in mlu grid_sample kernel."));

  Tensor trans_input;
  // transpose x from NCHW to NHWC
  const std::vector<int> perm_to_nhwc = {0, 2, 3, 1};
  TransposeFromMLUTensor<T>(
      dev_ctx, perm_to_nhwc, &x, &trans_input, true /*need_reshape_or_alloc*/);

  Tensor tmp_output;
  tmp_output.Resize({n, out_h, out_w, c});
  dev_ctx.template Alloc<T>(&tmp_output);

  MLUCnnlGridSampleDesc grid_sample_desc(mode, padding_mode, align_corners);
  MLUCnnlTensorDesc input_desc(
      trans_input, CNNL_LAYOUT_NHWC, ToCnnlDataType<T>());
  MLUCnnlTensorDesc grid_desc(grid, CNNL_LAYOUT_NHWC, ToCnnlDataType<T>());
  MLUCnnlTensorDesc tmp_output_desc(
      tmp_output, CNNL_LAYOUT_NHWC, ToCnnlDataType<T>());

  MLUCnnl::GridSample(dev_ctx,
                      grid_sample_desc.get(),
                      input_desc.get(),
                      GetBasePtr(&trans_input),
                      grid_desc.get(),
                      GetBasePtr(&grid),
                      tmp_output_desc.get(),
                      GetBasePtr(&tmp_output));

  // transpose out from NHWC to NCHW
  const std::vector<int> perm_to_nchw = {
      0,
      3,
      1,
      2,
  };
  TransposeFromMLUTensor<T>(
      dev_ctx, perm_to_nchw, &tmp_output, out, false /*need_reshape_or_alloc*/);
}
template <typename T, typename Context>
void GridSampleGradKernel(const Context& dev_ctx,
                          const phi::DenseTensor& x,
                          const phi::DenseTensor& grid,
                          const phi::DenseTensor& out_grad,
                          const std::string& mode,
                          const std::string& padding_mode,
                          bool align_corners,
                          phi::DenseTensor* x_grad,
                          phi::DenseTensor* grid_grad) {
  const int n = grid.dims()[0];
  const int out_h = grid.dims()[1];
  const int out_w = grid.dims()[2];
  const int c = x.dims()[1];
  const int in_h = x.dims()[2];
  const int in_w = x.dims()[3];

  PADDLE_ENFORCE_EQ(
      mode == "bilinear",
      true,
      phi::errors::Unavailable(
          "Only support bilinear mode in mlu grid_sample kernel."));
  PADDLE_ENFORCE_EQ(
      padding_mode == "zeros",
      true,
      phi::errors::Unavailable(
          "Only support zeros padding_mode in mlu grid_sample kernel."));

  Tensor trans_input;
  // transpose x from NCHW to NHWC
  const std::vector<int> perm_to_nhwc = {0, 2, 3, 1};
  TransposeFromMLUTensor<T>(
      dev_ctx, perm_to_nhwc, &x, &trans_input, true /*need_reshape_or_alloc*/);

  x_grad->Resize({n, c, in_h, in_w});
  dev_ctx.template Alloc<T>(x_grad);

  grid_grad->Resize({n, out_h, out_w, 2});
  dev_ctx.template Alloc<T>(grid_grad);

  Tensor trans_out_grad;
  // transpose x from NCHW to NHWC
  TransposeFromMLUTensor<T>(dev_ctx,
                            perm_to_nhwc,
                            &out_grad,
                            &trans_out_grad,
                            true /*need_reshape_or_alloc*/);

  Tensor temp_x_grad;
  temp_x_grad.Resize({n, in_h, in_w, c});
  dev_ctx.template Alloc<T>(&temp_x_grad);

  MLUCnnlGridSampleDesc grid_sample_desc(mode, padding_mode, align_corners);
  MLUCnnlTensorDesc input_desc(
      trans_input, CNNL_LAYOUT_NHWC, ToCnnlDataType<T>());
  MLUCnnlTensorDesc grid_desc(grid, CNNL_LAYOUT_NHWC, ToCnnlDataType<T>());
  MLUCnnlTensorDesc out_grad_desc(
      trans_out_grad, CNNL_LAYOUT_NHWC, ToCnnlDataType<T>());
  MLUCnnlTensorDesc temp_x_grad_desc(
      temp_x_grad, CNNL_LAYOUT_NHWC, ToCnnlDataType<T>());
  MLUCnnlTensorDesc grid_grad_desc(
      *grid_grad, CNNL_LAYOUT_NHWC, ToCnnlDataType<T>());

  MLUCnnl::GridSampleBackward(dev_ctx,
                              grid_sample_desc.get(),
                              out_grad_desc.get(),
                              GetBasePtr(&trans_out_grad),
                              input_desc.get(),
                              GetBasePtr(&trans_input),
                              grid_desc.get(),
                              GetBasePtr(&grid),
                              temp_x_grad_desc.get(),
                              GetBasePtr(&temp_x_grad),
                              grid_grad_desc.get(),
                              GetBasePtr(grid_grad));

  const std::vector<int> perm_to_nchw = {0, 3, 1, 2};
  TransposeFromMLUTensor<T>(dev_ctx,
                            perm_to_nchw,
                            &temp_x_grad,
                            x_grad,
                            true /*need_reshape_or_alloc*/);
}
}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(grid_sample,
                          mlu,
                          ALL_LAYOUT,
                          custom_kernel::GridSampleKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(grid_sample_grad,
                          mlu,
                          ALL_LAYOUT,
                          custom_kernel::GridSampleGradKernel,
                          float,
                          phi::dtype::float16) {}
