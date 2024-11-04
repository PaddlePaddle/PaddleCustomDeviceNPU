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

static void StridedSliceOutDims(const std::vector<int64_t>& starts,
                                const std::vector<int64_t>& ends,
                                const std::vector<int64_t>& strides,
                                const std::vector<int>& axes,
                                const std::vector<int>& infer_flags,
                                const phi::DDim in_dims,
                                const std::vector<int>& decrease_axis,
                                int64_t* out_dims_vector,
                                const size_t size,
                                bool infer_shape) {
  for (int i = 0; i < in_dims.size(); i++) {
    out_dims_vector[i] = in_dims[i];
  }
  int64_t stride_index, start_index, end_index;
  for (size_t i = 0; i < size; i++) {
    int axes_index = axes[i];
    start_index = starts[i];
    end_index = ends[i];
    stride_index = strides[i];
    bool decrease_axis_affect = false;
    if (start_index == -1 && end_index == 0 && infer_flags[i] == -1) {
      auto ret = std::find(decrease_axis.begin(), decrease_axis.end(), axes[i]);
      if (ret != decrease_axis.end()) {
        decrease_axis_affect = true;
      }
    }
    if (decrease_axis_affect) {
      out_dims_vector[axes_index] = 1;
      continue;
    }
    if (infer_shape && infer_flags[i] == -1) {
      out_dims_vector[axes_index] = -1;
      continue;
    }

    PADDLE_ENFORCE_NE(stride_index,
                      0,
                      phi::errors::InvalidArgument(
                          "stride index in StridedSlice operator is 0."));
    int64_t axis_size = in_dims[axes_index];

    if (axis_size < 0) {
      continue;
    }

    if (start_index < 0) {
      start_index = start_index + axis_size;
      start_index = std::max<int64_t>(start_index, 0);
    }
    if (end_index < 0) {
      if (!(end_index == -1 && stride_index < 0)) {  // skip None stop condition
        end_index = end_index + axis_size;
        if (end_index < 0) {
          end_index = 0;
        }
      }
    }

    if (stride_index < 0) {
      start_index = start_index + 1;
      end_index = end_index + 1;
    }

    bool neg_dim_condition = ((stride_index < 0 && (start_index < end_index)) ||
                              (stride_index > 0 && (start_index > end_index)));
    PADDLE_ENFORCE_EQ(neg_dim_condition,
                      false,
                      phi::errors::InvalidArgument(
                          "The start index and end index are invalid for their "
                          "corresponding stride."));

    int64_t left =
        std::max(static_cast<int64_t>(0), std::min(start_index, end_index));
    int64_t right = std::min(axis_size, std::max(start_index, end_index));
    int64_t step = std::abs(stride_index);

    auto out_dims_index = (std::abs(right - left) + step - 1) / step;

    out_dims_vector[axes_index] = out_dims_index;
  }
}

static void StridedSliceFunctor(int64_t* starts,
                                int64_t* ends,
                                int64_t* strides,
                                const int* axes,
                                int* reverse_axis,
                                const phi::DDim dims,
                                const std::vector<int>& infer_flags,
                                const std::vector<int>& decrease_axis,
                                const size_t size) {
  for (size_t axis = 0; axis < size; axis++) {
    int64_t axis_size = dims[axes[axis]];
    int axis_index = axis;
    if (axis_size < 0) {
      starts[axis_index] = 0;
      ends[axis_index] = 1;
      strides[axis_index] = 1;
    }
    bool decrease_axis_affect = false;
    if (starts[axis_index] == -1 && ends[axis_index] == 0 &&
        infer_flags[axis_index] == -1) {
      auto ret = std::find(
          decrease_axis.begin(), decrease_axis.end(), axes[axis_index]);
      if (ret != decrease_axis.end()) {
        decrease_axis_affect = true;
      }
    }
    // stride must not be zero
    if (starts[axis_index] < 0) {
      starts[axis_index] = starts[axis_index] + axis_size;
      starts[axis_index] = std::max<int64_t>(starts[axis_index], 0);
    }
    if (ends[axis_index] < 0) {
      if (!(ends[axis_index] == -1 &&
            strides[axis_index] < 0)) {  // skip None stop condition
        ends[axis_index] = ends[axis_index] + axis_size;
        if (ends[axis_index] < 0) {
          ends[axis_index] = 0;
        }
      }
    }
    if (decrease_axis_affect) {
      if (strides[axis_index] < 0) {
        ends[axis_index] = starts[axis_index] - 1;
      } else {
        ends[axis_index] = starts[axis_index] + 1;
      }
    }

    if (strides[axis_index] < 0) {
      reverse_axis[axis_index] = 1;
      strides[axis_index] = -strides[axis_index];
      if (starts[axis_index] > ends[axis_index]) {
        // swap the reverse
        auto end_dim = axis_size - 1 < starts[axis_index] ? axis_size - 1
                                                          : starts[axis_index];
        auto offset = (end_dim - ends[axis_index]) % strides[axis_index];
        offset = offset == 0 ? strides[axis_index] : offset;

        starts[axis_index] = starts[axis_index] + offset;
        ends[axis_index] = ends[axis_index] + offset;
      }
      std::swap(starts[axis_index], ends[axis_index]);
    } else {
      reverse_axis[axis_index] = 0;
      strides[axis_index] = strides[axis_index];
    }
  }
}

template <typename T, typename Context, size_t D>
void AclopStridedSliceCompute(const Context& dev_ctx,
                              const phi::DenseTensor& x,
                              const std::vector<int>& axes,
                              const phi::IntArray& starts_array,
                              const phi::IntArray& ends_array,
                              const phi::IntArray& strides_array,
                              const std::vector<int>& infer_flags,
                              const std::vector<int>& decrease_axis,
                              phi::DenseTensor* out) {
  auto stream = dev_ctx.stream();

  auto in_dims = x.dims();

  // list<int>
  auto starts = starts_array.GetData();
  auto ends = ends_array.GetData();
  auto strides = strides_array.GetData();

  // out dims calculation
  std::vector<int64_t> out_dims_vector(in_dims.size(), -1);
  custom_kernel::StridedSliceOutDims(starts,
                                     ends,
                                     strides,
                                     axes,
                                     infer_flags,
                                     in_dims,
                                     decrease_axis,
                                     out_dims_vector.data(),
                                     axes.size(),
                                     false);
  phi::DDim out_dims(phi::make_ddim(out_dims_vector));

  // check whether need to reverse (false: stride > 0; true: stride < 0)
  std::vector<int> reverse_vector(starts.size(), 0);
  custom_kernel::StridedSliceFunctor(starts.data(),
                                     ends.data(),
                                     strides.data(),
                                     axes.data(),
                                     reverse_vector.data(),
                                     in_dims,
                                     infer_flags,
                                     decrease_axis,
                                     starts.size());

  // construct the starts_indices, ends_indices and strides_indices tensor for
  // calling StridedSlice op
  std::vector<int64_t> starts_indices_vector(D, 0);
  std::vector<int64_t> ends_indices_vector(out_dims_vector.begin(),
                                           out_dims_vector.end());
  std::vector<int64_t> strides_indices_vector(D, 1);

  for (size_t axis = 0; axis < axes.size(); axis++) {
    int axis_index = axes[axis];
    starts_indices_vector[axis_index] = starts[axis];
    ends_indices_vector[axis_index] = ends[axis];
    strides_indices_vector[axis_index] = strides[axis];
  }

  auto out_dims_origin = out_dims;
  if (decrease_axis.size() > 0) {
    std::vector<int64_t> new_out_shape;
    for (size_t i = 0; i < decrease_axis.size(); ++i) {
      PADDLE_ENFORCE_EQ(
          out_dims[decrease_axis[i]],
          1,
          phi::errors::InvalidArgument(
              "the size of decrease dimension should be 1, but received %d.",
              out_dims[decrease_axis[i]]));
      out_dims_origin[decrease_axis[i]] = 0;
    }

    for (int i = 0; i < out_dims_origin.size(); ++i) {
      if (out_dims_origin[i] != 0) {
        new_out_shape.push_back(out_dims_origin[i]);
      }
    }
    if (new_out_shape.size() == 0) {
      new_out_shape.push_back(1);
    }
    out_dims_origin = phi::make_ddim(new_out_shape);
  }

  bool need_reverse = false;
  for (size_t axis = 0; axis < axes.size(); axis++) {
    if (reverse_vector[axis] == 1) {
      need_reverse = true;
      break;
    }
  }

  out->Resize(out_dims);
  dev_ctx.template Alloc<T>(out);

  if (x.dtype() == phi::DataType::BOOL) {
    const auto& runner = NpuOpRunner("StridedSliceD",
                                     {x},
                                     {*out},
                                     {{"begin", starts_indices_vector},
                                      {"end", ends_indices_vector},
                                      {"strides", strides_indices_vector},
                                      {"begin_mask", 0},
                                      {"end_mask", 0},
                                      {"ellipsis_mask", 0},
                                      {"new_axis_mask", 0},
                                      {"shrink_axis_mask", 0}});
    runner.Run(stream);
  } else {
    NpuOpRunner runner;
    runner.SetType("StridedSlice")
        .AddInput(x)
        .AddInput(dev_ctx, std::move(starts_indices_vector))
        .AddInput(dev_ctx, std::move(ends_indices_vector))
        .AddInput(dev_ctx, std::move(strides_indices_vector))
        .AddAttr("begin_mask", 0)
        .AddAttr("end_mask", 0)
        .AddAttr("ellipsis_mask", 0)
        .AddAttr("new_axis_mask", 0)
        .AddAttr("shrink_axis_mask", 0)
        .AddOutput(*out)
        .Run(stream);
  }

  if (need_reverse) {
    phi::DenseTensor out_tmp;
    out_tmp.Resize(out_dims);
    dev_ctx.template Alloc<T>(&out_tmp);
    TensorCopy(dev_ctx, *out, false, &out_tmp);

    phi::DenseTensor reverse_axis;
    std::vector<int> reverse_axis_vector;
    for (size_t axis = 0; axis < axes.size(); axis++) {
      if (reverse_vector[axis] == 1) {
        reverse_axis_vector.push_back(axes[axis]);
      }
    }
    reverse_axis.Resize({static_cast<int>(reverse_axis_vector.size())});
    dev_ctx.template Alloc<int>(&reverse_axis);
    TensorFromVector(dev_ctx, reverse_axis_vector, dev_ctx, &reverse_axis);

    const auto& runner_reverse =
        NpuOpRunner("ReverseV2", {out_tmp, reverse_axis}, {*out});
    runner_reverse.Run(stream);
  }

  if (decrease_axis.size() > 0) {
    out->Resize(out_dims_origin);
  }
}

template <typename T, typename Context, size_t D>
void StridedSliceCompute(const Context& dev_ctx,
                         const phi::DenseTensor& x,
                         const std::vector<int>& axes,
                         const phi::IntArray& starts_array,
                         const phi::IntArray& ends_array,
                         const phi::IntArray& strides_array,
                         const std::vector<int>& infer_flags,
                         const std::vector<int>& decrease_axis,
                         phi::DenseTensor* out) {
  DO_COMPATIBILITY(
      aclnnSliceV2,
      (custom_kernel::AclopStridedSliceCompute<T, Context, D>(dev_ctx,
                                                              x,
                                                              axes,
                                                              starts_array,
                                                              ends_array,
                                                              strides_array,
                                                              infer_flags,
                                                              decrease_axis,
                                                              out)));

  if (x.dtype() == phi::DataType::FLOAT64) {
    return custom_kernel::AclopStridedSliceCompute<T, Context, D>(dev_ctx,
                                                                  x,
                                                                  axes,
                                                                  starts_array,
                                                                  ends_array,
                                                                  strides_array,
                                                                  infer_flags,
                                                                  decrease_axis,
                                                                  out);
  }

  auto stream = dev_ctx.stream();
  auto in_dims = x.dims();

  // list<int>
  auto starts = starts_array.GetData();
  auto ends = ends_array.GetData();
  auto strides = strides_array.GetData();

  // out dims calculation
  std::vector<int64_t> out_dims_vector(in_dims.size(), -1);
  custom_kernel::StridedSliceOutDims(starts,
                                     ends,
                                     strides,
                                     axes,
                                     infer_flags,
                                     in_dims,
                                     decrease_axis,
                                     out_dims_vector.data(),
                                     axes.size(),
                                     false);
  phi::DDim out_dims(phi::make_ddim(out_dims_vector));

  // check whether need to reverse (false: stride > 0; true: stride < 0)
  std::vector<int> reverse_vector(starts.size(), 0);
  custom_kernel::StridedSliceFunctor(starts.data(),
                                     ends.data(),
                                     strides.data(),
                                     axes.data(),
                                     reverse_vector.data(),
                                     in_dims,
                                     infer_flags,
                                     decrease_axis,
                                     starts.size());

  // construct the starts_indices, ends_indices and strides_indices tensor for
  // calling StridedSlice op
  std::vector<int64_t> starts_indices_vector(D, 0);
  std::vector<int64_t> ends_indices_vector(out_dims_vector.begin(),
                                           out_dims_vector.end());
  std::vector<int64_t> strides_indices_vector(D, 1);

  for (size_t axis = 0; axis < axes.size(); axis++) {
    int axis_index = axes[axis];
    starts_indices_vector[axis_index] = starts[axis];
    ends_indices_vector[axis_index] = ends[axis];
    strides_indices_vector[axis_index] = strides[axis];
  }

  auto out_dims_origin = out_dims;
  if (decrease_axis.size() > 0) {
    std::vector<int64_t> new_out_shape;
    for (size_t i = 0; i < decrease_axis.size(); ++i) {
      PADDLE_ENFORCE_EQ(
          out_dims[decrease_axis[i]],
          1,
          phi::errors::InvalidArgument(
              "the size of decrease dimension should be 1, but received %d.",
              out_dims[decrease_axis[i]]));
      out_dims_origin[decrease_axis[i]] = 0;
    }

    for (int i = 0; i < out_dims_origin.size(); ++i) {
      if (out_dims_origin[i] != 0) {
        new_out_shape.push_back(out_dims_origin[i]);
      }
    }
    if (new_out_shape.size() == 0) {
      new_out_shape.push_back(1);
    }
    out_dims_origin = phi::make_ddim(new_out_shape);
  }

  bool need_reverse = false;
  for (size_t axis = 0; axis < axes.size(); axis++) {
    if (reverse_vector[axis] == 1) {
      need_reverse = true;
      break;
    }
  }

  out->Resize(out_dims);
  dev_ctx.template Alloc<T>(out);

  std::vector<int64_t> axes_vec;
  for (int i = 0; i < D; i++) {
    axes_vec.push_back(i);
  }

  EXEC_NPU_CMD(aclnnSliceV2,
               dev_ctx,
               x,
               starts_indices_vector,
               ends_indices_vector,
               axes_vec,
               strides_indices_vector,
               *out);

  if (need_reverse) {
    std::vector<int64_t> reverse_axis_vector;
    for (size_t axis = 0; axis < axes.size(); axis++) {
      if (reverse_vector[axis] == 1) {
        reverse_axis_vector.push_back(axes[axis]);
      }
    }
    EXEC_NPU_CMD(aclnnFlip, dev_ctx, *out, reverse_axis_vector, *out);
  }

  if (decrease_axis.size() > 0) {
    out->Resize(out_dims_origin);
  }
}

template <typename T, typename Context, size_t D>
void StridedSliceCompute(const Context& dev_ctx,
                         const phi::TensorArray& x,
                         const std::vector<int>& axes,
                         const phi::IntArray& starts,
                         const phi::IntArray& ends,
                         const phi::IntArray& strides,
                         const std::vector<int>& infer_flags,
                         const std::vector<int>& decrease_axis,
                         phi::TensorArray* out) {
  const int64_t size = x.size();
  auto in_dims = phi::make_ddim({size});

  auto starts_ = starts.GetData();
  auto ends_ = ends.GetData();
  auto strides_ = strides.GetData();
  std::vector<int64_t> out_dims_vector(in_dims.size(), -1);

  custom_kernel::StridedSliceOutDims(starts_,
                                     ends_,
                                     strides_,
                                     axes,
                                     infer_flags,
                                     in_dims,
                                     decrease_axis,
                                     out_dims_vector.data(),
                                     axes.size(),
                                     false);
  phi::DDim out_dims(phi::make_ddim(out_dims_vector));

  std::vector<int> reverse_vector(starts.size(), 0);
  custom_kernel::StridedSliceFunctor(starts_.data(),
                                     ends_.data(),
                                     strides_.data(),
                                     axes.data(),
                                     reverse_vector.data(),
                                     in_dims,
                                     infer_flags,
                                     decrease_axis,
                                     starts.size());

  std::vector<int64_t> starts_indices_vector(D, 0);
  std::vector<int64_t> ends_indices_vector(out_dims_vector.begin(),
                                           out_dims_vector.end());
  std::vector<int64_t> strides_indices_vector(D, 1);
  std::vector<bool> reverse_axis(D, 0);

  for (size_t axis = 0; axis < D; axis++) {
    starts_indices_vector[axis] = 0;
    ends_indices_vector[axis] = out_dims[axis];
    strides_indices_vector[axis] = 1;
    reverse_vector[axis] = false;
  }
  for (size_t axis = 0; axis < axes.size(); axis++) {
    int axis_index = axes[axis];
    starts_indices_vector[axis_index] = starts_[axis];
    ends_indices_vector[axis_index] = ends_[axis];
    strides_indices_vector[axis_index] = strides_[axis];
    reverse_axis[axis_index] = (reverse_vector[axis] == 1) ? true : false;
  }

  auto out_dims_origin = out_dims;
  if (decrease_axis.size() > 0) {
    std::vector<int64_t> new_out_shape;
    for (size_t i = 0; i < decrease_axis.size(); ++i) {
      PADDLE_ENFORCE_EQ(
          out_dims[decrease_axis[i]],
          1,
          phi::errors::InvalidArgument(
              "the size of decrease dimension should be 1, but received %d.",
              out_dims[decrease_axis[i]]));
      out_dims_origin[decrease_axis[i]] = 0;
    }

    for (int i = 0; i < out_dims_origin.size(); ++i) {
      if (out_dims_origin[i] != 0) {
        new_out_shape.push_back(out_dims_origin[i]);
      }
    }
    if (new_out_shape.size() == 0) {
      new_out_shape.push_back(1);
    }
    out_dims_origin = phi::make_ddim(new_out_shape);
  }

  bool need_reverse = false;
  for (size_t axis = 0; axis < axes.size(); axis++) {
    if (reverse_vector[axis] == 1) {
      need_reverse = true;
      break;
    }
  }

  PADDLE_ENFORCE_EQ(
      starts_indices_vector.size(),
      1,
      phi::errors::InvalidArgument(
          "When the input of 'strided_slice_op' is `TensorArray`, the "
          "dimension of start index  should be 1, but received %d.",
          starts_indices_vector.size()));

  PADDLE_ENFORCE_EQ(
      ends_indices_vector.size(),
      1,
      phi::errors::InvalidArgument(
          "When the input of 'strided_slice_op' is `TensorArray`, the "
          "dimension of end index should be 1, but received %d.",
          ends_indices_vector.size()));

  PADDLE_ENFORCE_EQ(
      strides_indices_vector.size(),
      1,
      phi::errors::InvalidArgument(
          "When the input of 'strided_slice_op' is `TensorArray`, the "
          "dimension of stride should be 1, but received %d.",
          strides_indices_vector.size()));

  PADDLE_ENFORCE_EQ(
      out_dims_origin.size(),
      1,
      phi::errors::InvalidArgument(
          "When the input of 'strided_slice_op' is `TensorArray`, the "
          "dimension of Output should be 1, but received %d",
          out_dims_origin.size()));

  out->resize(out_dims_origin[0]);
  size_t const in_array_size = x.size();
  for (size_t i = 0; i < out->size(); i++) {
    size_t in_offset = (starts_indices_vector[0] % in_array_size) +
                       i * strides_indices_vector[0];

    int64_t out_offset = i;
    if (need_reverse) {
      out_offset = out->size() - i - 1;
    }

    auto& in_tensor = x.at(in_offset);
    PADDLE_ENFORCE_GT(
        in_tensor.numel(),
        0,
        phi::errors::PreconditionNotMet(
            "The input LoDTensorArray Input[%d] holds no memory.", in_offset));
    auto& out_tensor = out->at(out_offset);
    out_tensor.Resize(in_tensor.dims());

    phi::Copy<Context>(
        dev_ctx, in_tensor, dev_ctx.GetPlace(), false, &out_tensor);
    out_tensor.Resize(in_tensor.dims());
  }
}
template <typename T, typename Context>
void StridedSliceRawKernel(const Context& dev_ctx,
                           const phi::DenseTensor& x,
                           const std::vector<int>& axes,
                           const phi::IntArray& starts,
                           const phi::IntArray& ends,
                           const phi::IntArray& strides,
                           const std::vector<int>& infer_flags,
                           const std::vector<int>& decrease_axis,
                           phi::DenseTensor* out) {
  int rank = x.dims().size();

  switch (rank) {
    case 1:
      custom_kernel::StridedSliceCompute<T, Context, 1>(dev_ctx,
                                                        x,
                                                        axes,
                                                        starts,
                                                        ends,
                                                        strides,
                                                        infer_flags,
                                                        decrease_axis,
                                                        out);
      break;
    case 2:
      custom_kernel::StridedSliceCompute<T, Context, 2>(dev_ctx,
                                                        x,
                                                        axes,
                                                        starts,
                                                        ends,
                                                        strides,
                                                        infer_flags,
                                                        decrease_axis,
                                                        out);
      break;
    case 3:
      custom_kernel::StridedSliceCompute<T, Context, 3>(dev_ctx,
                                                        x,
                                                        axes,
                                                        starts,
                                                        ends,
                                                        strides,
                                                        infer_flags,
                                                        decrease_axis,
                                                        out);
      break;
    case 4:
      custom_kernel::StridedSliceCompute<T, Context, 4>(dev_ctx,
                                                        x,
                                                        axes,
                                                        starts,
                                                        ends,
                                                        strides,
                                                        infer_flags,
                                                        decrease_axis,
                                                        out);
      break;
    case 5:
      custom_kernel::StridedSliceCompute<T, Context, 5>(dev_ctx,
                                                        x,
                                                        axes,
                                                        starts,
                                                        ends,
                                                        strides,
                                                        infer_flags,
                                                        decrease_axis,
                                                        out);
      break;
    case 6:
      custom_kernel::StridedSliceCompute<T, Context, 6>(dev_ctx,
                                                        x,
                                                        axes,
                                                        starts,
                                                        ends,
                                                        strides,
                                                        infer_flags,
                                                        decrease_axis,
                                                        out);
      break;
    default:
      PADDLE_THROW(phi::errors::InvalidArgument(
          "The rank of input is supported up to 6."));
      break;
  }
}

template <typename T, typename Context, size_t D>
void StridedSliceGradCompute(const Context& dev_ctx,
                             const phi::DenseTensor& x,
                             const phi::DenseTensor& out_grad,
                             const std::vector<int>& axes,
                             const phi::IntArray& starts_array,
                             const phi::IntArray& ends_array,
                             const phi::IntArray& strides_array,
                             const std::vector<int>& infer_flags,
                             const std::vector<int>& decrease_axis,
                             phi::DenseTensor* x_grad) {
  auto input_dims = x.dims();

  x_grad->Resize(input_dims);
  dev_ctx.template Alloc<T>(x_grad);

  auto starts = starts_array.GetData();
  auto ends = ends_array.GetData();
  auto strides = strides_array.GetData();

  std::vector<int64_t> out_dims_vector(input_dims.size(), -1);
  custom_kernel::StridedSliceOutDims(starts,
                                     ends,
                                     strides,
                                     axes,
                                     infer_flags,
                                     input_dims,
                                     decrease_axis,
                                     out_dims_vector.data(),
                                     axes.size(),
                                     false);

  std::vector<int> reverse_vector(starts.size(), 0);
  custom_kernel::StridedSliceFunctor(starts.data(),
                                     ends.data(),
                                     strides.data(),
                                     axes.data(),
                                     reverse_vector.data(),
                                     input_dims,
                                     infer_flags,
                                     decrease_axis,
                                     starts.size());

  std::vector<int64_t> starts_indices_vector(D, 0);
  std::vector<int64_t> ends_indices_vector(out_dims_vector.begin(),
                                           out_dims_vector.end());
  std::vector<int64_t> strides_indices_vector(D, 1);

  for (size_t axis = 0; axis < axes.size(); axis++) {
    int axis_index = axes[axis];
    starts_indices_vector[axis_index] = starts[axis];
    ends_indices_vector[axis_index] = ends[axis];
    strides_indices_vector[axis_index] = strides[axis];
  }

  std::vector<int64_t> input_dims_vector;
  for (int i = 0; i < input_dims.size(); i++) {
    input_dims_vector.push_back(input_dims[i]);
  }

  bool need_reverse = false;
  for (size_t axis = 0; axis < axes.size(); axis++) {
    if (reverse_vector[axis] == 1) {
      need_reverse = true;
      break;
    }
  }

  auto stream = dev_ctx.stream();
  NPUAttributeMap attr_input = {{"begin_mask", 0},
                                {"end_mask", 0},
                                {"ellipsis_mask", 0},
                                {"new_axis_mask", 0},
                                {"shrink_axis_mask", 0}};

  if (need_reverse) {
    phi::DenseTensor reverse_axis;
    std::vector<int> reverse_axis_vector;
    for (size_t axis = 0; axis < axes.size(); axis++) {
      if (reverse_vector[axis] == 1) {
        reverse_axis_vector.push_back(axes[axis]);
      }
    }
    TensorFromVector<int>(dev_ctx, reverse_axis_vector, dev_ctx, &reverse_axis);

    phi::DenseTensor out_grad_tmp;
    out_grad_tmp.Resize(out_grad.dims());
    dev_ctx.template Alloc<T>(&out_grad_tmp);
    const auto& runner_reverse =
        NpuOpRunner("ReverseV2", {out_grad, reverse_axis}, {out_grad_tmp});
    runner_reverse.Run(stream);

    NpuOpRunner runner;
    runner.SetType("StridedSliceGrad")
        .AddInput(dev_ctx, std::move(input_dims_vector))
        .AddInput(dev_ctx, std::move(starts_indices_vector))
        .AddInput(dev_ctx, std::move(ends_indices_vector))
        .AddInput(dev_ctx, std::move(strides_indices_vector))
        .AddInput(out_grad_tmp)
        .AddOutput(*x_grad)
        .AddAttrs(attr_input);
    runner.Run(stream);
  } else {
    NpuOpRunner runner;
    runner.SetType("StridedSliceGrad")
        .AddInput(dev_ctx, std::move(input_dims_vector))
        .AddInput(dev_ctx, std::move(starts_indices_vector))
        .AddInput(dev_ctx, std::move(ends_indices_vector))
        .AddInput(dev_ctx, std::move(strides_indices_vector))
        .AddInput(out_grad)
        .AddOutput(*x_grad)
        .AddAttrs(attr_input);
    runner.Run(stream);
  }
}

template <typename T, typename Context, size_t D>
void StridedSliceGradCompute(const Context& dev_ctx,
                             const phi::TensorArray& x,
                             const phi::TensorArray& out_grad,
                             const std::vector<int>& axes,
                             const phi::IntArray& starts,
                             const phi::IntArray& ends,
                             const phi::IntArray& strides,
                             const std::vector<int>& infer_flags,
                             const std::vector<int>& decrease_axis,
                             phi::TensorArray* x_grad) {
  // Note(weixin):Since the shape of `framework::GradVarName("Input")` of
  // StridedSliceGrad cannot be calculated by
  // `framework::GradVarName("Output")`, the dim of "Input" is used to
  // calculate the output shape. when set it to inplace OP, there may be
  // some problems.
  const int64_t size = x.size();
  phi::DDim out_dims = phi::make_ddim({size});
  auto out_dims_vector = phi::vectorize(out_dims);
  auto starts_ = starts.GetData();
  auto ends_ = ends.GetData();
  auto strides_ = strides.GetData();
  std::vector<int> reverse_vector(starts_.size(), 0);

  custom_kernel::StridedSliceFunctor(starts_.data(),
                                     ends_.data(),
                                     strides_.data(),
                                     axes.data(),
                                     reverse_vector.data(),
                                     out_dims,
                                     infer_flags,
                                     decrease_axis,
                                     starts_.size());

  std::vector<int64_t> starts_indices_vector(D, 0);
  std::vector<int64_t> ends_indices_vector(out_dims_vector.begin(),
                                           out_dims_vector.end());
  std::vector<int64_t> strides_indices_vector(D, 1);

  for (size_t axis = 0; axis < D; axis++) {
    starts_indices_vector[axis] = 0;
    ends_indices_vector[axis] = out_dims[axis];
    strides_indices_vector[axis] = 1;
  }
  for (size_t axis = 0; axis < axes.size(); axis++) {
    int axis_index = axes[axis];
    starts_indices_vector[axis_index] = starts_[axis];
    ends_indices_vector[axis_index] = ends_[axis];
    strides_indices_vector[axis_index] = strides_[axis];
    reverse_vector[axis_index] = (reverse_vector[axis] == 1) ? true : false;
  }

  bool need_reverse = false;
  for (size_t axis = 0; axis < axes.size(); axis++) {
    if (reverse_vector[axis] == 1) {
      need_reverse = true;
      break;
    }
  }
  PADDLE_ENFORCE_EQ(
      starts_indices_vector.size(),
      1,
      phi::errors::InvalidArgument(
          "When the input of 'strided_slice_grad_op' is `TensorArray`, the "
          "dimension of start index  should be 1, but received %d.",
          starts_indices_vector.size()));
  PADDLE_ENFORCE_EQ(
      ends_indices_vector.size(),
      1,
      phi::errors::InvalidArgument(
          "When the input of 'strided_slice_op' is `TensorArray`, the "
          "dimension of end index should be 1, but received %d.",
          ends_indices_vector.size()));
  PADDLE_ENFORCE_EQ(
      strides_indices_vector.size(),
      1,
      phi::errors::InvalidArgument(
          "When the input of 'strided_slice_grad_op' is `TensorArray`, the "
          "dimension of stride should be 1, but received %d.",
          strides_indices_vector.size()));

  PADDLE_ENFORCE_EQ(
      out_dims.size(),
      1,
      phi::errors::InvalidArgument(
          "When the output of `strided_slice_grad_op` is `TensorArray`, "
          "the dimension of output should be 1, but received %d.",
          out_dims.size()));

  auto const d_out_array_size = x_grad->size();

  for (size_t j = 0; j < d_out_array_size; j++) {
    auto& dim = x.at(j).dims();
    auto& d_out_tensor = x_grad->at(j);

    int64_t sub = j - starts_indices_vector[0];

    int64_t in_offset = sub / strides_indices_vector[0];

    if (need_reverse) {
      in_offset = out_grad.size() - in_offset - 1;
    }

    if ((sub % strides_indices_vector[0] == 0) && (0 <= in_offset) &&
        (static_cast<size_t>(in_offset) < out_grad.size())) {
      auto& in_tensor = out_grad.at(in_offset);
      PADDLE_ENFORCE_GT(
          in_tensor.numel(),
          0,
          phi::errors::PreconditionNotMet(
              "The input LoDTensorArray Input[%d] holds no memory.",
              in_offset));

      phi::Copy<Context>(
          dev_ctx, in_tensor, dev_ctx.GetPlace(), false, &d_out_tensor);
      d_out_tensor.Resize(in_tensor.dims());
    } else {
      d_out_tensor.Resize(dim);

      if (!d_out_tensor.initialized()) {
        dev_ctx.template Alloc<T>(&d_out_tensor);
      }

      custom_kernel::FillNpuTensorWithConstant<T>(
          &d_out_tensor, dev_ctx, static_cast<T>(0));
      d_out_tensor.Resize(dim);
    }
  }
}

template <typename T, typename Context>
void StridedSliceRawGradKernel(const Context& dev_ctx,
                               const phi::DenseTensor& x,
                               const phi::DenseTensor& out_grad,
                               const std::vector<int>& axes,
                               const phi::IntArray& starts,
                               const phi::IntArray& ends,
                               const phi::IntArray& strides,
                               const std::vector<int>& infer_flags,
                               const std::vector<int>& decrease_axis,
                               phi::DenseTensor* x_grad) {
  int rank = x.dims().size();

  switch (rank) {
    case 1:
      custom_kernel::StridedSliceGradCompute<T, Context, 1>(dev_ctx,
                                                            x,
                                                            out_grad,
                                                            axes,
                                                            starts,
                                                            ends,
                                                            strides,
                                                            infer_flags,
                                                            decrease_axis,
                                                            x_grad);
      break;
    case 2:
      custom_kernel::StridedSliceGradCompute<T, Context, 2>(dev_ctx,
                                                            x,
                                                            out_grad,
                                                            axes,
                                                            starts,
                                                            ends,
                                                            strides,
                                                            infer_flags,
                                                            decrease_axis,
                                                            x_grad);
      break;
    case 3:
      custom_kernel::StridedSliceGradCompute<T, Context, 3>(dev_ctx,
                                                            x,
                                                            out_grad,
                                                            axes,
                                                            starts,
                                                            ends,
                                                            strides,
                                                            infer_flags,
                                                            decrease_axis,
                                                            x_grad);
      break;
    case 4:
      custom_kernel::StridedSliceGradCompute<T, Context, 4>(dev_ctx,
                                                            x,
                                                            out_grad,
                                                            axes,
                                                            starts,
                                                            ends,
                                                            strides,
                                                            infer_flags,
                                                            decrease_axis,
                                                            x_grad);
      break;
    case 5:
      custom_kernel::StridedSliceGradCompute<T, Context, 5>(dev_ctx,
                                                            x,
                                                            out_grad,
                                                            axes,
                                                            starts,
                                                            ends,
                                                            strides,
                                                            infer_flags,
                                                            decrease_axis,
                                                            x_grad);
      break;
    case 6:
      custom_kernel::StridedSliceGradCompute<T, Context, 6>(dev_ctx,
                                                            x,
                                                            out_grad,
                                                            axes,
                                                            starts,
                                                            ends,
                                                            strides,
                                                            infer_flags,
                                                            decrease_axis,
                                                            x_grad);
      break;
    default:
      PADDLE_THROW(phi::errors::InvalidArgument(
          "The rank of input is supported up to 6."));
      break;
  }
}

template <typename T, typename Context>
void StridedSliceKernel(const Context& dev_ctx,
                        const phi::DenseTensor& x,
                        const std::vector<int>& axes,
                        const phi::IntArray& starts,
                        const phi::IntArray& ends,
                        const phi::IntArray& strides,
                        phi::DenseTensor* out) {
  std::vector<int> infer_flags(axes.size(), 1);
  std::vector<int> decrease_axis;
  custom_kernel::StridedSliceRawKernel<T, Context>(
      dev_ctx, x, axes, starts, ends, strides, infer_flags, decrease_axis, out);
}

template <typename T, typename Context>
void StridedSliceArrayKernel(const Context& dev_ctx,
                             const phi::TensorArray& x,
                             const std::vector<int>& axes,
                             const phi::IntArray& starts,
                             const phi::IntArray& ends,
                             const phi::IntArray& strides,
                             const std::vector<int>& infer_flags,
                             const std::vector<int>& decrease_axis,
                             phi::TensorArray* out) {
  custom_kernel::StridedSliceCompute<T, Context, 1>(
      dev_ctx, x, axes, starts, ends, strides, infer_flags, decrease_axis, out);
}

template <typename T, typename Context>
void StridedSliceGradKernel(const Context& dev_ctx,
                            const phi::DenseTensor& x,
                            const phi::DenseTensor& out_grad,
                            const std::vector<int>& axes,
                            const phi::IntArray& starts,
                            const phi::IntArray& ends,
                            const phi::IntArray& strides,
                            phi::DenseTensor* x_grad) {
  std::vector<int> infer_flags(axes.size(), 1);
  std::vector<int> decrease_axis;
  custom_kernel::StridedSliceRawGradKernel<T, Context>(dev_ctx,
                                                       x,
                                                       out_grad,
                                                       axes,
                                                       starts,
                                                       ends,
                                                       strides,
                                                       infer_flags,
                                                       decrease_axis,
                                                       x_grad);
}

template <typename T, typename Context>
void StridedSliceArrayGradKernel(const Context& dev_ctx,
                                 const phi::TensorArray& x,
                                 const phi::TensorArray& out_grad,
                                 const std::vector<int>& axes,
                                 const phi::IntArray& starts,
                                 const phi::IntArray& ends,
                                 const phi::IntArray& strides,
                                 const std::vector<int>& infer_flags,
                                 const std::vector<int>& decrease_axis,
                                 phi::TensorArray* x_grad) {
  custom_kernel::StridedSliceGradCompute<T, Context, 1>(dev_ctx,
                                                        x,
                                                        out_grad,
                                                        axes,
                                                        starts,
                                                        ends,
                                                        strides,
                                                        infer_flags,
                                                        decrease_axis,
                                                        x_grad);
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(strided_slice_raw,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::StridedSliceRawKernel,
                          bool,
                          int,
                          int64_t,
                          float,
                          double,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(strided_slice_raw_grad,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::StridedSliceRawGradKernel,
                          bool,
                          int,
                          int64_t,
                          float,
                          double,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(strided_slice,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::StridedSliceKernel,
                          bool,
                          int,
                          int64_t,
                          float,
                          double,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}

PD_REGISTER_PLUGIN_KERNEL(strided_slice_grad,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::StridedSliceGradKernel,
                          bool,
                          int,
                          int64_t,
                          float,
                          double,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}

PD_REGISTER_PLUGIN_KERNEL(strided_slice_array,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::StridedSliceArrayKernel,
                          bool,
                          int,
                          int64_t,
                          float,
                          double,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(strided_slice_array_grad,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::StridedSliceArrayGradKernel,
                          bool,
                          int,
                          int64_t,
                          float,
                          double,
                          phi::dtype::float16) {}
