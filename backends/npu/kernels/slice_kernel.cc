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
#include "kernels/funcs/slice_utils.h"

namespace custom_kernel {

template <typename T, typename Context>
void PadKernel(const Context& dev_ctx,
               const phi::DenseTensor& x,
               const std::vector<int>& paddings,
               const phi::Scalar& pad_value_scalar,
               phi::DenseTensor* out);

void UpdateAttr(const phi::DDim& in_dims,
                const std::vector<int> axes,
                const std::vector<int> starts,
                const std::vector<int> ends,
                std::vector<int>* offsets,
                std::vector<int>* size) {
  int cnt = 0;
  for (int i = 0; i < in_dims.size(); ++i) {
    int start = 0;
    int end = in_dims[i];
    // NOTE(zhiqiu): Becareful that cnt may > axes.size() and result in
    // overflow.
    int axis = cnt < static_cast<int>(axes.size()) ? axes[cnt] : -1;
    if (axis == i) {
      start = starts[cnt];
      if (start < 0) {
        start = (start + in_dims[i]);
      }
      start = std::max(start, static_cast<int>(0));
      end = ends[cnt];
      if (end < 0) {
        end = (end + in_dims[i]);
      }
      end = std::min(end, static_cast<int>(in_dims[i]));
      cnt++;
    }

    (*offsets)[i] = start;
    (*size)[i] = end - start;
  }
}

template <typename T, typename Context>
void AclopSliceRawKernel(const Context& dev_ctx,
                         const phi::DenseTensor& x,
                         const std::vector<int64_t>& axes_t,
                         const phi::IntArray& starts_array,
                         const phi::IntArray& ends_array,
                         const std::vector<int64_t>& infer_flags,
                         const std::vector<int64_t>& decrease_axis,
                         phi::DenseTensor* out) {
  std::vector<int> axes(axes_t.begin(), axes_t.end());
  auto starts_int = starts_array.GetData();
  auto ends_int = ends_array.GetData();
  std::vector<int> starts(starts_int.begin(), starts_int.end());
  std::vector<int> ends(ends_int.begin(), ends_int.end());

  PADDLE_ENFORCE_EQ(
      starts.size(),
      axes.size(),
      phi::errors::InvalidArgument(
          "The size of starts must be equal to the size of axes."));
  PADDLE_ENFORCE_EQ(ends.size(),
                    axes.size(),
                    phi::errors::InvalidArgument(
                        "The size of ends must be equal to the size of axes."));

  // Infer output dims
  const auto& in_dims = x.dims();
  auto out_dims = out->dims();
  auto slice_dims = out_dims;

  for (size_t i = 0; i < axes.size(); ++i) {
    // when start == -1 && end == start+1
    if (starts[i] == -1 && ends[i] == 0 && infer_flags[i] == -1) {
      auto ret = std::find(decrease_axis.begin(), decrease_axis.end(), axes[i]);
      if (ret != decrease_axis.end()) {
        ends[i] = in_dims[axes[i]];
      }
    }
  }

  custom_kernel::CheckAndUpdateSliceAttrs(in_dims, axes, &starts, &ends);
  slice_dims = custom_kernel::GetSliceDims<int>(
      in_dims, axes, starts, ends, nullptr, nullptr);
  out_dims = custom_kernel::GetDecreasedDims(slice_dims, decrease_axis);
  out->Resize(out_dims);

  dev_ctx.template Alloc<T>(out);

  std::vector<int> offsets(in_dims.size());
  std::vector<int> size(in_dims.size());

  custom_kernel::UpdateAttr(in_dims, axes, starts, ends, &offsets, &size);

  auto stream = static_cast<aclrtStream>(dev_ctx.stream());
  NpuOpRunner runner;
  runner.SetType("Slice")
      .AddInput(x)
      .AddInput(dev_ctx, std::move(offsets))
      .AddInput(dev_ctx, std::move(size))
      .AddOutput(*out)
      .Run(stream);
}

template <typename T, typename Context>
void SliceRawKernel(const Context& dev_ctx,
                    const phi::DenseTensor& x,
                    const std::vector<int64_t>& axes_t,
                    const phi::IntArray& starts_array,
                    const phi::IntArray& ends_array,
                    const std::vector<int64_t>& infer_flags,
                    const std::vector<int64_t>& decrease_axis,
                    phi::DenseTensor* out) {
  DO_COMPATIBILITY(
      aclnnSliceV2,
      (custom_kernel::AclopSliceRawKernel<T, Context>(dev_ctx,
                                                      x,
                                                      axes_t,
                                                      starts_array,
                                                      ends_array,
                                                      infer_flags,
                                                      decrease_axis,
                                                      out)));

  std::vector<int> axes(axes_t.begin(), axes_t.end());
  auto starts_int = starts_array.GetData();
  auto ends_int = ends_array.GetData();
  std::vector<int> starts(starts_int.begin(), starts_int.end());
  std::vector<int> ends(ends_int.begin(), ends_int.end());

  PADDLE_ENFORCE_EQ(
      starts.size(),
      axes.size(),
      phi::errors::InvalidArgument(
          "The size of starts must be equal to the size of axes."));
  PADDLE_ENFORCE_EQ(ends.size(),
                    axes.size(),
                    phi::errors::InvalidArgument(
                        "The size of ends must be equal to the size of axes."));

  // Infer output dims
  const auto& in_dims = x.dims();
  auto out_dims = out->dims();
  auto slice_dims = out_dims;

  for (size_t i = 0; i < axes.size(); ++i) {
    // when start == -1 && end == start+1
    if (starts[i] == -1 && ends[i] == 0 && infer_flags[i] == -1) {
      auto ret = std::find(decrease_axis.begin(), decrease_axis.end(), axes[i]);
      if (ret != decrease_axis.end()) {
        ends[i] = in_dims[axes[i]];
      }
    }
  }

  custom_kernel::CheckAndUpdateSliceAttrs(in_dims, axes, &starts, &ends);
  slice_dims = custom_kernel::GetSliceDims<int>(
      in_dims, axes, starts, ends, nullptr, nullptr);
  out_dims = custom_kernel::GetDecreasedDims(slice_dims, decrease_axis);
  out->Resize(out_dims);

  std::vector<int> offsets(in_dims.size());
  std::vector<int> size(in_dims.size());
  custom_kernel::UpdateAttr(in_dims, axes, starts, ends, &offsets, &size);
  out_dims = out->dims();
  out->Resize(phi::make_ddim(size));

  std::vector<int64_t> steps;
  for (int i = 0; i < axes_t.size(); i++) {
    steps.push_back(1.0);
  }

  dev_ctx.template Alloc<T>(out);
  EXEC_NPU_CMD(
      aclnnSliceV2, dev_ctx, x, starts_array, ends_array, axes_t, steps, *out);
  if (out->dims().size() != out_dims.size()) {
    out->Resize(out_dims);
  }
}

template <typename T, typename Context>
void SliceGradRawKernel(const Context& dev_ctx,
                        const phi::DenseTensor& x,
                        const phi::DenseTensor& out_grad,
                        const std::vector<int64_t>& axes_t,
                        const phi::IntArray& starts_array,
                        const phi::IntArray& ends_array,
                        const std::vector<int64_t>& infer_flags,
                        const std::vector<int64_t>& decrease_axis,
                        phi::DenseTensor* x_grad) {
  std::vector<int> axes(axes_t.begin(), axes_t.end());
  auto starts_int = starts_array.GetData();
  auto ends_int = ends_array.GetData();

  std::vector<int> starts(starts_int.begin(), starts_int.end());
  std::vector<int> ends(ends_int.begin(), ends_int.end());

  const auto& in_dims = x.dims();
  int rank = in_dims.size();

  std::vector<int> offsets(rank);
  std::vector<int> size(rank);
  UpdateAttr(in_dims, axes, starts, ends, &offsets, &size);

  std::vector<int> paddings(rank * 2);
  for (int i = 0; i < rank; i++) {
    paddings[2 * i] = offsets[i];
    paddings[2 * i + 1] = in_dims[i] - size[i] - offsets[i];
  }

  phi::DenseTensor tmp_dout(out_grad);
  auto out_dims = out_grad.dims();

  auto decrease_size = decrease_axis.size();
  if (decrease_size > 0) {
    if (decrease_size == static_cast<size_t>(in_dims.size())) {
      out_dims = phi::make_ddim(std::vector<int>(decrease_size, 1));
    } else {
      std::vector<int> origin_out_shape(out_dims.size() + decrease_size, -1);
      for (size_t i = 0; i < decrease_size; ++i) {
        origin_out_shape[decrease_axis[i]] = 1;
      }
      int index = 0;
      for (size_t i = 0; i < origin_out_shape.size(); ++i) {
        if (origin_out_shape[i] == -1) {
          origin_out_shape[i] = out_dims[index];
          ++index;
        }
      }
      out_dims = phi::make_ddim(origin_out_shape);
    }
    tmp_dout.Resize(out_dims);
  }

  dev_ctx.template Alloc<T>(x_grad);
  auto stream = static_cast<aclrtStream>(dev_ctx.stream());

  phi::Scalar pad_value_scalar = 0;
  custom_kernel::PadKernel<T, Context>(
      dev_ctx, tmp_dout, paddings, pad_value_scalar, x_grad);
}

template <typename T, typename Context>
void SliceArrayKernel(const Context& dev_ctx,
                      const phi::TensorArray& input,
                      const phi::IntArray& starts,
                      const phi::IntArray& ends,
                      phi::TensorArray* out) {
  int64_t in_size = input.size();
  int64_t start = starts[0] < 0 ? (starts[0] + in_size) : starts[0];
  int64_t end = ends[0] < 0 ? (ends[0] + in_size) : ends[0];

  start = std::max(start, static_cast<int64_t>(0));
  end = std::max(end, static_cast<int64_t>(0));
  end = std::min(end, in_size);

  if (starts[0] == -1 && end == 0) {
    end = start + 1;
  }

  PADDLE_ENFORCE_GT(end,
                    start,
                    phi::errors::InvalidArgument(
                        "Attr(ends) should be greater than attr(starts) in "
                        "slice op. But received end = %d, start = %d.",
                        ends[0],
                        starts[0]));
  int64_t out_size = end - start;

  out->resize(out_size);
  for (int i = 0; i < out_size; ++i) {
    auto* out_tensor = &out->at(i);
    const auto& in_tensor = input.at(i + start);
    out_tensor->Resize(in_tensor.dims());
    if (in_tensor.initialized() && in_tensor.numel() > 0) {
      phi::Copy<Context>(
          dev_ctx, in_tensor, dev_ctx.GetPlace(), false, out_tensor);
    } else {
      VLOG(10) << "WARNING: The input tensor 'x_tensor' holds no memory, so "
                  "nothing has been written to output array["
               << i << "].";
    }
  }
}

template <typename T, typename Context>
void SliceArrayDenseKernel(const Context& dev_ctx,
                           const phi::TensorArray& input,
                           const phi::IntArray& starts,
                           phi::DenseTensor* out) {
  int64_t in_size = input.size();
  int64_t start = starts[0] < 0 ? (starts[0] + in_size) : starts[0];
  start = std::max(start, static_cast<int64_t>(0));
  out->Resize(input[start].dims());
  phi::Copy<Context>(dev_ctx, input[start], dev_ctx.GetPlace(), false, out);
}

template <typename T, typename Context>
void SliceArrayGradKernel(const Context& dev_ctx,
                          const phi::TensorArray& input,
                          const phi::TensorArray& out_grad,
                          const phi::IntArray& starts,
                          const phi::IntArray& ends,
                          phi::TensorArray* input_grad) {
  int64_t d_in_size = input.size();
  input_grad->resize(d_in_size);
  // If the input is TensorArray, the rank of input is 1.
  // So only use the 0th element of starts.
  int64_t start = starts[0] < 0 ? (starts[0] + d_in_size) : starts[0];
  start = std::max(start, static_cast<int64_t>(0));
  // set zero
  for (int i = 0; i < d_in_size; ++i) {
    const auto& dim = input.at(i).dims();
    auto* in_grad_tensor = &input_grad->at(i);
    in_grad_tensor->Resize(dim);
    dev_ctx.template Alloc<T>(in_grad_tensor);
    FillNpuTensorWithConstant<T>(in_grad_tensor, dev_ctx, static_cast<T>(0));
    in_grad_tensor->Resize(dim);
  }

  int d_out_size = out_grad.size();
  for (int i = 0; i < d_out_size; ++i) {
    input_grad->at(start + i).Resize(out_grad[i].dims());
    phi::Copy<Context>(dev_ctx,
                       out_grad[i],
                       dev_ctx.GetPlace(),
                       false,
                       &input_grad->at(start + i));
  }
}

template <typename T, typename Context>
void SliceArrayDenseGradKernel(const Context& dev_ctx,
                               const phi::TensorArray& input,
                               const phi::DenseTensor& out_grad,
                               const phi::IntArray& starts,
                               phi::TensorArray* input_grad) {
  int64_t d_in_size = input.size();
  input_grad->resize(d_in_size);
  // If the input is TensorArray, the rank of input is 1.
  // So only use the 0th element of starts.
  int64_t start = starts[0] < 0 ? (starts[0] + d_in_size) : starts[0];
  start = std::max(start, static_cast<int64_t>(0));
  // set zero
  for (int i = 0; i < d_in_size; ++i) {
    const auto& dim = input.at(i).dims();
    auto* in_grad_tensor = &input_grad->at(i);
    in_grad_tensor->Resize(dim);
    dev_ctx.template Alloc<T>(in_grad_tensor);
    FillNpuTensorWithConstant<T>(in_grad_tensor, dev_ctx, static_cast<T>(0));
    in_grad_tensor->Resize(dim);
  }

  phi::Copy<Context>(
      dev_ctx, out_grad, dev_ctx.GetPlace(), false, &input_grad->at(start));
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(slice,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::SliceRawKernel,
                          phi::dtype::float16,
                          phi::dtype::bfloat16,
                          float,
                          double,
                          int16_t,
                          int32_t,
                          int64_t,
                          bool) {}

PD_REGISTER_PLUGIN_KERNEL(slice_array,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::SliceArrayKernel,
                          phi::dtype::float16,
                          phi::dtype::bfloat16,
                          float,
                          double,
                          int16_t,
                          int32_t,
                          int64_t,
                          bool) {}

PD_REGISTER_PLUGIN_KERNEL(slice_array_dense,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::SliceArrayDenseKernel,
                          phi::dtype::float16,
                          phi::dtype::bfloat16,
                          float,
                          double,
                          int16_t,
                          int32_t,
                          int64_t,
                          bool) {}

PD_REGISTER_PLUGIN_KERNEL(slice_grad,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::SliceGradRawKernel,
                          phi::dtype::float16,
                          phi::dtype::bfloat16,
                          float,
                          double,
                          int16_t,
                          int32_t,
                          int64_t,
                          bool) {}

PD_REGISTER_PLUGIN_KERNEL(slice_array_grad,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::SliceArrayGradKernel,
                          phi::dtype::float16,
                          phi::dtype::bfloat16,
                          float,
                          double,
                          int16_t,
                          int32_t,
                          int64_t,
                          bool) {}

PD_REGISTER_PLUGIN_KERNEL(slice_array_dense_grad,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::SliceArrayDenseGradKernel,
                          phi::dtype::float16,
                          phi::dtype::bfloat16,
                          float,
                          double,
                          int16_t,
                          int32_t,
                          int64_t,
                          bool) {}
