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

#include <algorithm>

#include "common/gcu_op_runner.h"
#include "kernels/funcs/gcu_kernel_funcs.h"

namespace {

struct SliceInfo {
  explicit SliceInfo(const std::vector<int64_t>& input_dims)
      : input_dims_(input_dims), ends_(input_dims), output_dims_(input_dims) {
    size_t rank = input_dims.size();
    starts_.resize(rank, 0);
    steps_.resize(rank, 1);
  }

  std::vector<int64_t> input_dims_;
  std::vector<int64_t> starts_;
  std::vector<int64_t> ends_;
  std::vector<int64_t> steps_;
  std::vector<int64_t> output_dims_;
};

template <typename T>
T clamp(T value, T min, T max) {
  return std::max(min, std::min(value, max));
}

inline void PrepareSliceInfo(const std::vector<int64_t>& raw_starts,
                             const std::vector<int64_t>& raw_ends,
                             const std::vector<int64_t>& raw_steps,
                             const std::vector<int64_t>& raw_axes,
                             SliceInfo& slice_info) {  // NOLINT
  std::vector<int64_t> axes;
  if (raw_axes.empty()) {
    axes.resize(raw_starts.size());
    std::iota(axes.begin(), axes.end(), 0);
  } else {
    axes.assign(raw_axes.begin(), raw_axes.end());
  }

  const auto axes_count = axes.size();
  std::unordered_set<int64_t> unique_axes;
  unique_axes.reserve(axes_count);

  const int64_t input_rank = slice_info.input_dims_.size();
  for (size_t axis_index = 0; axis_index < axes_count; ++axis_index) {
    const auto axis =
        axes[axis_index] < 0 ? axes[axis_index] + input_rank : axes[axis_index];
    PADDLE_ENFORCE_EQ(axis >= 0 && axis < input_rank,
                      true,
                      phi::errors::InvalidArgument(
                          "'axes' has an axis outside of the input rank"));

    auto p = unique_axes.insert(axis);
    PADDLE_ENFORCE_EQ(
        p.second, true, phi::errors::InvalidArgument("'axes' has duplicates"));

    // 1. process step
    int64_t step = axis_index < raw_steps.size() ? raw_steps[axis_index] : 1L;
    PADDLE_ENFORCE_NE(
        step, 0, phi::errors::InvalidArgument("'axes' value cannot be 0"));

    const int64_t dim_value = slice_info.input_dims_[axis];
    if (dim_value == 0) {
      // shape with empty dim. only output_dims_ matters but set everything for
      // completeness.
      slice_info.steps_[axis] = step;
      slice_info.starts_[axis] = 0;
      slice_info.ends_[axis] = 0;
      slice_info.output_dims_[axis] = 0;
      continue;
    }

    // clamp step to avoid overflow if there's a stupidly large value (which
    // will be multiplied in Slice) as long as the clamped value is >= the size
    // of the dimension a single step will push us past the end
    step = clamp(step, -dim_value, dim_value);
    slice_info.steps_[axis] = step;

    // 2. process start
    auto start = raw_starts[axis_index];
    start = start < 0 ? start + dim_value : start;
    slice_info.starts_[axis] =
        clamp(start, 0L, step < 0 ? dim_value - 1 : dim_value);

    // 3. process end
    auto end = raw_ends[axis_index];
    // INT_MAX has a special meaning for end according to spec
    // equivalent to 'None' in numpy
    // it represent slicing to the end of the dimension
    if (end == std::numeric_limits<int32_t>::max() ||
        end == std::numeric_limits<int64_t>::max()) {
      end = step < 0 ? -1 : dim_value;
    } else {
      end = end < 0 ? end + dim_value : end;
      end = clamp(end, step < 0 ? -1L : 0L, dim_value);
    }
    slice_info.ends_[axis] = end;

    // find output dim value for this axis: tf use (e - b + s - 1) / s
    const auto temp = static_cast<int64_t>(
        ceil(1.0 * (slice_info.ends_[axis] - slice_info.starts_[axis]) / step));
    slice_info.output_dims_[axis] = temp < 0LL ? 0LL : temp;
  }

  return;
}
}  // namespace

namespace custom_kernel {
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

template <typename T, typename Context>
void StridedSliceKernel(const Context& dev_ctx,
                        const phi::DenseTensor& x,
                        const std::vector<int>& axes,
                        const phi::IntArray& starts,
                        const phi::IntArray& ends,
                        const phi::IntArray& strides,
                        phi::DenseTensor* out) {
  PADDLE_GCU_KERNEL_TRACE("strided_slice");
  dev_ctx.template Alloc<T>(out);

  if (LaunchAOTKernel()) {
    std::vector<int64_t> axes64(axes.begin(), axes.end());

    SliceInfo slice_info(common::vectorize(x.dims()));
    PrepareSliceInfo(starts.GetData(),
                     ends.GetData(),
                     strides.GetData(),
                     axes64,
                     slice_info);

    // output->Resize(Shape(slice_info.output_dims_));
    if (LIKELY(out->numel() != 0)) {
      std::vector<int64_t> dimensions_to_reverse;
      std::vector<int64_t> slice_begin, slice_end, slice_step;
      for (size_t i = 0; i < slice_info.steps_.size(); ++i) {
        if (slice_info.steps_[i] > 0) {
          slice_begin.push_back(slice_info.starts_[i]);
          slice_end.push_back(slice_info.ends_[i]);
          slice_step.push_back(slice_info.steps_[i]);
        } else {
          dimensions_to_reverse.push_back(i);

          int64_t b = x.dims().at(i) - slice_info.starts_[i] - 1;
          int64_t e = std::max(x.dims().at(i) - slice_info.ends_[i] - 1,
                               x.dims().at(i) - slice_info.starts_[i] - 1);
          int64_t s = -slice_info.steps_[i];
          slice_begin.push_back(b);
          slice_end.push_back(e);
          slice_step.push_back(s);
        }
      }

      auto reverse_input = x;
      if (!dimensions_to_reverse.empty()) {
        phi::DenseTensor reverse_input =
            custom_kernel::TensorEmpty(dev_ctx, x.meta());
        dev_ctx.template Alloc<T>(&reverse_input);

        LAUNCH_TOPSATENOP(
            topsatenFlip, dev_ctx, reverse_input, x, dimensions_to_reverse);
      }

      std::vector<int64_t> sizes(slice_info.output_dims_);
      std::vector<int64_t> strides = common::vectorize(x.meta().strides);
      int64_t offset = 0;
      for (size_t i = 0; i < slice_info.input_dims_.size(); ++i) {
        offset += starts[i] * strides[i];
      }

      for (size_t i = 0; i < slice_info.input_dims_.size(); ++i) {
        strides[i] *= slice_info.steps_[i];
      }

      phi::DenseTensor as_strides_out;
      auto x_tensor = CreateTopsatenTensor(reverse_input);
      auto out_tensor = CreateTopsatenTensor(*out);
      auto view_out_tensor = CreateTopsatenTensor(as_strides_out);
      auto aten_sizes = IntArrayToTopsatenSize(sizes);
      auto aten_strides = IntArrayToTopsatenSize(strides);

      std::string abstract_info =
          custom_kernel::GetAbstractInfo("StridedSlice_topsatenAsStrided",
                                         as_strides_out,
                                         reverse_input,
                                         sizes,
                                         strides,
                                         offset);
      LAUNCH_TOPSATENOP_WITH_RAW_ATEN_DEF(topsatenAsStrided,
                                          dev_ctx,
                                          abstract_info,
                                          view_out_tensor,
                                          x_tensor,
                                          aten_sizes,
                                          aten_strides,
                                          offset);

      abstract_info = custom_kernel::GetAbstractInfo(
          "StridedSlice_topsatenCopy", *out, as_strides_out, false);
      LAUNCH_TOPSATENOP_WITH_RAW_ATEN_DEF(topsatenCopy,
                                          dev_ctx,
                                          abstract_info,
                                          out_tensor,
                                          view_out_tensor,
                                          false);
    }
  } else {  // kernel impl base on JIT
    TensorNameMap input_names;
    input_names["Input"] = {"x"};

    TensorValueMap inputs;
    inputs["Input"] = {const_cast<DenseTensor*>(&x)};

    TensorNameMap output_names;
    output_names["Out"] = {"out"};

    TensorValueMap outputs;
    outputs["Out"] = {out};

    std::vector<int> infer_flags(axes.size(), 1);
    std::vector<int> decrease_axis;
    std::vector<int> starts_list = GetIntList(starts.GetData());
    std::vector<int> ends_list = GetIntList(ends.GetData());
    std::vector<int> strides_list = GetIntList(strides.GetData());

    GcuAttributeMap attrs;
    attrs["starts"] = starts_list;
    attrs["ends"] = ends_list;
    attrs["strides"] = strides_list;
    attrs["axes"] = axes;
    attrs["infer_flags"] = infer_flags;
    attrs["decrease_axis"] = decrease_axis;

    GcuRunner(input_names,
              inputs,
              output_names,
              outputs,
              attrs,
              "strided_slice",
              dev_ctx);
  }
}

template <typename T, typename Context>
void StridedSliceGradKernel(const Context& dev_ctx,
                            const DenseTensor& x,
                            const DenseTensor& out_grad,
                            const std::vector<int>& axes,
                            const phi::IntArray& starts,
                            const phi::IntArray& ends,
                            const phi::IntArray& strides,
                            DenseTensor* x_grad) {
  PADDLE_GCU_KERNEL_TRACE("strided_slice_grad");
  dev_ctx.template Alloc<T>(x_grad);

  if (LaunchAOTKernel()) {
    THROW_AOT_UNIMPLEMENTED();
  } else {  // kernel impl base on JIT
    TensorNameMap input_names;
    input_names["Input"] = {"x"};
    input_names[GradVarName("Out")] = {"out_grad"};

    TensorValueMap inputs;
    inputs["Input"] = {const_cast<DenseTensor*>(&x)};
    inputs[GradVarName("Out")] = {const_cast<DenseTensor*>(&out_grad)};

    TensorNameMap output_names;
    output_names[GradVarName("X")] = {"x_grad"};

    TensorValueMap outputs;
    outputs[GradVarName("X")] = {x_grad};

    std::vector<int> infer_flags(axes.size(), 1);
    std::vector<int> decrease_axis;
    std::vector<int> starts_list = GetIntList(starts.GetData());
    std::vector<int> ends_list = GetIntList(ends.GetData());
    std::vector<int> strides_list = GetIntList(strides.GetData());

    GcuAttributeMap attrs;
    attrs["starts"] = starts_list;
    attrs["ends"] = ends_list;
    attrs["strides"] = strides_list;
    attrs["axes"] = axes;
    attrs["infer_flags"] = infer_flags;
    attrs["decrease_axis"] = decrease_axis;

    GcuRunner(input_names,
              inputs,
              output_names,
              outputs,
              attrs,
              "strided_slice_grad",
              dev_ctx);
  }
}
}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(strided_slice,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::StridedSliceKernel,
                          bool,
                          int,
                          int64_t,
                          float,
                          double,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(strided_slice_grad,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::StridedSliceGradKernel,
                          bool,
                          int,
                          int64_t,
                          float,
                          double,
                          phi::dtype::float16) {}
