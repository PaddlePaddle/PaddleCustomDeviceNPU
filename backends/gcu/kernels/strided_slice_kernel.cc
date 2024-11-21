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
#include <set>

#include "common/gcu_op_runner.h"
#include "kernels/funcs/gcu_kernel_funcs.h"

namespace {

struct SliceInfo {
  explicit SliceInfo(const std::vector<int64_t>& input_dims)
      : input_dims_(input_dims), ends_(input_dims), output_dims_(input_dims) {
    auto rank = input_dims.size();
    starts_ = std::vector<int64_t>(rank, 0);
    steps_ = std::vector<int64_t>(rank, 1);
  }

  std::vector<int64_t> input_dims_;
  std::vector<int64_t> starts_;
  std::vector<int64_t> ends_;
  std::vector<int64_t> steps_;
  std::vector<int64_t> output_dims_;
};

void PrepareSliceInfo(const std::vector<int64_t>& raw_starts,
                      const std::vector<int64_t>& raw_ends,
                      const std::vector<int64_t>& raw_steps,
                      const std::vector<int>& raw_axes,
                      SliceInfo& slice_info) {  // NOLINT
  std::vector<int64_t> axes{raw_axes.begin(), raw_axes.end()};

  // positive
  std::for_each(axes.begin(), axes.end(), [&slice_info](int64_t& axis) {
    axis = axis < 0 ? (axis + slice_info.input_dims_.size()) : axis;
  });
  // unique
  std::set<int> unique_axes;
  int64_t axis = 0;
  for (int i = 0; i < axes.size(); i++) {
    if (unique_axes.count(axes[i]) != 0) {
      continue;
    }
    unique_axes.insert(axes[i]);
    // positive and limit to max shape
    axis = axes[i];
    int64_t dim_value = slice_info.input_dims_.at(axis);
    slice_info.starts_[axis] =
        raw_starts[i] >= 0
            ? (raw_starts[i] <= dim_value ? raw_starts[i] : dim_value)
            : raw_starts[i] + dim_value;
    slice_info.steps_[axis] = raw_steps[i];
    slice_info.ends_[axis] =
        raw_ends[i] >= 0 ? (raw_ends[i] <= dim_value ? raw_ends[i] : dim_value)
                         : raw_ends[i] + dim_value;
    // infer out shape
    auto out_dim = (slice_info.ends_[axis] - slice_info.starts_[axis]) /
                   slice_info.steps_[axis];
    int counter = 0;
    int64_t tmp = slice_info.starts_[axis];
    if (slice_info.steps_[axis] >= 0) {
      while (tmp < slice_info.ends_[axis]) {
        if (tmp < dim_value) {
          counter += 1;
        }
        tmp += slice_info.steps_[axis];
      }
    } else {
      while (tmp > slice_info.ends_[axis]) {
        if (tmp < dim_value) {
          counter += 1;
        }
        tmp += slice_info.steps_[axis];
      }
    }

    slice_info.output_dims_[axis] = counter;
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
    auto rank = x.dims().size();
    PADDLE_ENFORCE_EQ(
        AreEqual(starts.size(), ends.size(), strides.size(), axes.size()),
        true,
        phi::errors::InvalidArgument(
            "stridedslice axes[%lu] starts[%lu] ends[%lu] "
            "strides[%lu] rank should be same!",
            axes.size(),
            starts.size(),
            ends.size(),
            strides.size()));
    PADDLE_ENFORCE_LE(starts.size(),
                      rank,
                      phi::errors::InvalidArgument(
                          "starts rank should be less equal than x rank!"));
    VLOG(6) << "=== StridedSlice Debug Infos ===";
    VLOG(6) << " - origin axes: ";
    for (const auto& axis : axes) {
      VLOG(6) << "  - " << axis;
    }
    VLOG(6) << " - origin input dims:" << x.dims();
    VLOG(6) << " - origin starts:" << common::make_ddim(starts.GetData());
    VLOG(6) << " - origin ends:" << common::make_ddim(ends.GetData());
    VLOG(6) << " - origin strides:" << common::make_ddim(strides.GetData());
    VLOG(6) << " - origin output dims:" << out->dims();

    SliceInfo slice_info(common::vectorize(x.dims()));
    PrepareSliceInfo(
        starts.GetData(), ends.GetData(), strides.GetData(), axes, slice_info);
    // debug
    VLOG(6) << "======== After PrepareSliceInfo =================";
    VLOG(6) << " - computed start dims:"
            << common::make_ddim(slice_info.starts_);
    VLOG(6) << " - computed step dims:" << common::make_ddim(slice_info.steps_);
    VLOG(6) << " - computed end dims:" << common::make_ddim(slice_info.ends_);
    VLOG(6) << " - computed output dims:"
            << common::make_ddim(slice_info.output_dims_);

    // output->Resize(Shape(slice_info.output_dims_));
    if (LIKELY(out->numel() != 0)) {
      // output shape should be same as the result after preparing slice_info
      PADDLE_ENFORCE_EQ(
          out->dims(),
          common::make_ddim(slice_info.output_dims_),
          phi::errors::Fatal("output dims should be same as the result after "
                             "preparing slice_info"));
      // calc new_strides:tuple(x_stride[d] * strides[d] for d in
      // range(len(x.shape)))
      std::vector<int64_t> x_strides = common::vectorize(x.meta().strides);
      std::vector<int64_t> new_strides(x_strides.size(), 0);
      for (size_t i = 0; i < x_strides.size(); i++) {
        new_strides[i] = x_strides[i] * slice_info.steps_[i];
      }
      // calc offset:sum(b * s for b, s in zip(starts, x_stride))
      int64_t offset = 0;
      for (size_t i = 0; i < x_strides.size(); i++) {
        offset += slice_info.starts_[i] * x_strides[i];
      }

      phi::DenseTensor as_strides_out;
      auto x_tensor = CreateTopsatenTensor(x);
      auto out_tensor = CreateTopsatenTensor(*out);
      auto view_out_tensor = CreateTopsatenTensor(as_strides_out);
      topsatenSize_t aten_sizes{
          slice_info.output_dims_.data(),
          static_cast<int64_t>(slice_info.output_dims_.size())};
      topsatenSize_t aten_strides{new_strides.data(),
                                  static_cast<int64_t>(new_strides.size())};

      std::string abstract_info =
          custom_kernel::GetAbstractInfo("StridedSlice_topsatenAsStrided",
                                         as_strides_out,
                                         x,
                                         slice_info.output_dims_,
                                         new_strides,
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
