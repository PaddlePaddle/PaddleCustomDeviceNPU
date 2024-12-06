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
#include "kernels/funcs/mlu_funcs.h"

namespace custom_kernel {

/**
 * @brief Normalizes the slice interval [st, ed) with a given step and dimension
 * size.
 *
 * This function adjusts the interval [st, ed) to fit within the bounds defined
 * by the dimension size, taking into account the specified step. It handles
 * both positive and negative steps and accounts for negative indices by
 * converting them to equivalent positive indices within the dimension size.
 *
 * @tparam T The data type of the input parameters, which can be an integer or
 * floating-point type.
 * @param st The starting index of the interval.
 * @param ed The ending index of the interval (exclusive).
 * @param step The step size for iterating through the interval, which can be
 * positive or negative.
 * @param dim_size The size of the dimension, serving as the upper bound for
 * valid indices.
 * @param st_out Pointer to store the normalized starting index.
 * @param ed_out Pointer to store the normalized ending index.
 * @param zero_dim_out Pointer to a boolean flag that is set to true if the
 * resulting interval is empty.
 *
 * @details
 * - If `step > 0`, the function ensures that `st` and `ed` are adjusted to be
 * within the range [0, dim_size).
 * - If `step < 0`, the function adjusts `st` and `ed` to accommodate the
 * reverse traversal of the interval.
 * - Handles special cases where `st` and `ed` may be out of bounds or where
 * `dim_size` is zero.
 * - Uses pointer parameters for output to modify the values directly.
 * - The function also handles scenarios involving negative indices, converting
 * them appropriately.
 *
 * @example
 * T st_out, ed_out;
 * bool zero_dim;
 * normalize_interval(-3, -2, 1, 4, &st_out, &ed_out, &zero_dim);
 * // Results in: st_out = 1, ed_out = 2, zero_dim = false
 *
 * @note The function assumes that the pointers provided for output parameters
 * are valid and non-null.
 */
template <typename T>
void normalize_interval(
    T st, T ed, T step, T dim_size, T* st_out, T* ed_out, bool* zero_dim_out) {
  /* Normalize slice interval [st, ed) with given step and dim_size.
  e.g. if given st = -3, ed = -2, step = 1, dim_size = 4,
  then normalized st_out = 1(-3+4), st_ed = 2(-2+4).

  This function is general enough and applicable
  for both step > 0 and step < 0 scenarios.

  Indicices dipicted as below:

  ===============================================================
                |  0   1     2     3    ...  D-1 | D D+1 ...
  ... -D-2 -D-1 | -D  -D+1  -D+2  -D+3  ... -1   |
  ===============================================================
  */
  // 0 dim size, just return
  if (dim_size <= 0) {
    *st_out = *ed_out = 0;
    *zero_dim_out = true;
    return;
  }

  if (step > 0) {
    /* positive step */
    // 0 dim size case 1
    if (st >= dim_size) {
      *st_out = *ed_out = 0;
      *zero_dim_out = true;
      return;
    }

    // 0 dim size case 2
    if (ed <= -dim_size) {
      *st_out = *ed_out = 0;
      *zero_dim_out = true;
      return;
    }

    // make st belongs: (-inf, -D-1)∪[0, D)
    if (-dim_size <= st && st < 0) {
      st += dim_size;
    }
    // make st belongs: [0, D)
    st = std::max(st, static_cast<T>(0));

    // make ed belongs: [0, +inf)
    if (-dim_size <= ed && ed < 0) {
      ed += dim_size;
    }
    // make ed belongs: [0, D]
    ed = std::min(ed, dim_size);

    // 0 dim size case 3
    if (st >= ed) {
      *st_out = *ed_out = 0;
      *zero_dim_out = true;
      return;
    }
    *st_out = st;
    *ed_out = ed;
    return;

  } else {
    /* negative step */
    // 0 dim size case 1
    if (st <= -dim_size - 1) {
      *st_out = *ed_out = 0;
      *zero_dim_out = true;
      return;
    }

    // 0 dim size case 2
    if (ed >= dim_size - 1) {
      *st_out = *ed_out = 0;
      *zero_dim_out = true;
      return;
    }

    // make st belongs: [0, D)∪[0, +inf)
    if (-dim_size <= st && st < 0) {
      st += dim_size;
    }
    // make st belongs: [0, D)
    st = std::min(st, dim_size - 1);

    // make ed belongs: [-inf, -D)∪[0, D)
    if (-dim_size <= ed && ed < 0) {
      ed += dim_size;
    }
    // make ed belongs: [-D-1, -D)∪[0, D) ==> {-D-1}∪[0, D)
    ed = std::max(ed, -dim_size - 1);

    if (ed == -dim_size - 1) {
      // When ed=-D-1, it is symmetrical to when step is greater than 0 and
      // ed=D.
      *st_out = st;
      *ed_out = ed;
      return;
    }

    // now only remain the case that ed belongs to: [0, D)
    // 0 dim size case 3
    if (ed >= st) {
      *st_out = *ed_out = 0;
      *zero_dim_out = true;
      return;
    }

    *st_out = st;
    *ed_out = ed;
    return;
  }
}

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

template <typename T = int64_t>
inline void CheckAndUpdateSliceAttrs(const phi::DDim in_dims,
                                     const std::vector<T>& axes,
                                     std::vector<T>* starts,
                                     std::vector<T>* ends,
                                     std::vector<int64_t>* steps = nullptr,
                                     std::vector<T>* infer_flags = nullptr) {
  for (size_t i = 0; i < axes.size(); ++i) {
    T axis = axes[i];
    PADDLE_ENFORCE_LT(
        axis,
        in_dims.size(),
        phi::errors::InvalidArgument(
            "The axis value should be less than the rank of input, "
            "but received axes[%d] = %d, rank of input is %d.",
            i,
            axis,
            in_dims.size()));

    if (infer_flags != nullptr && (*infer_flags)[i] == -1) {
      continue;
    }

    T dim_value = in_dims[axis];

    if (dim_value > 0) {
      T step = steps == nullptr ? 1 : (*steps)[i];
      T start, end;
      bool dummy_zero_out_dim = false;
      normalize_interval((*starts)[i],
                         (*ends)[i],
                         step,
                         dim_value,
                         &start,
                         &end,
                         &dummy_zero_out_dim);
      if (end == -dim_value - 1) {
        end = -1;
      }

      (*starts)[i] = start;
      (*ends)[i] = end;
    } else if (dim_value == 0) {
      (*starts)[i] = 0;
      (*ends)[i] = 0;
    }
  }
}

template <typename T = int64_t>
inline phi::DDim GetSliceDims(const phi::DDim in_dims,
                              const std::vector<T>& axes,
                              const std::vector<T>& starts,
                              const std::vector<T>& ends,
                              std::vector<T>* steps = nullptr,
                              std::vector<T>* infer_flags = nullptr) {
  phi::DDim slice_dims(in_dims);

  for (size_t i = 0; i < axes.size(); ++i) {
    T axis = axes[i];
    if (infer_flags != nullptr && (*infer_flags)[i] == -1) {
      slice_dims[axis] = -1;
      continue;
    }

    T start = starts[i];
    T end = ends[i];
    T step = steps == nullptr ? 1 : (*steps)[i];

    if (step > 0) {
      slice_dims[axis] = (end - start + step - 1) / step;
    } else {
      slice_dims[axis] = (end - start + step + 1) / step;
    }
  }
  return slice_dims;
}

template <typename T = int64_t>
inline phi::DDim GetDecreasedDims(const phi::DDim slice_dims,
                                  const std::vector<T>& decrease_axes,
                                  std::vector<T>* infer_flags = nullptr) {
  phi::DDim decreased_dims(slice_dims);
  std::vector<uint8_t> decrease_flag(slice_dims.size(), 0);
  if (decrease_axes.size() > 0) {
    for (size_t i = 0; i < decrease_axes.size(); ++i) {
      T axis = decrease_axes[i];
      decrease_flag[axis] = 1;
      if (infer_flags && (*infer_flags)[i] != -1) {
        PADDLE_ENFORCE_EQ(decreased_dims[axis],
                          1,
                          phi::errors::InvalidArgument(
                              "Decrease dim should be 1, but now received %d",
                              decreased_dims[axis]));
      }
    }

    std::vector<T> new_shape;
    for (int i = 0; i < decreased_dims.size(); ++i) {
      if (decrease_flag[i] == 0) {
        new_shape.push_back(decreased_dims[i]);
      }
    }

    // NOTE(liym27): Paddle does not support that the rank of Tensor is 0, and
    // uses [1] instead.
    if (new_shape.size() == 0) {
      new_shape.push_back(1);
    }

    decreased_dims = phi::make_ddim(new_shape);
  }
  return decreased_dims;
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

  const auto& in_dims = x.dims();
  auto slice_dims = out->dims();
  bool reset_slice_dims = false;
  // Infer output dims
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
  reset_slice_dims = true;
  auto out_dims = custom_kernel::GetDecreasedDims(slice_dims, decrease_axis);

  out->Resize(out_dims);

  if (slice_dims.size() != in_dims.size() && !reset_slice_dims) {
    custom_kernel::CheckAndUpdateSliceAttrs(in_dims, axes, &starts, &ends);
    slice_dims = custom_kernel::GetSliceDims<int>(
        in_dims, axes, starts, ends, nullptr, nullptr);
  }

  int in_dim_size = x.dims().size();
  if (static_cast<int>(axes.size()) != in_dim_size) {
    std::vector<int> tmp_starts(in_dim_size, 0);
    const auto& in_dims_vec = phi::vectorize(x.dims());
    std::vector<int> tmp_ends(in_dims_vec.begin(), in_dims_vec.end());
    for (size_t i = 0; i < axes.size(); ++i) {
      tmp_starts[axes[i]] = starts[i];
      tmp_ends[axes[i]] = ends[i];
    }
    starts.swap(tmp_starts);
    ends.swap(tmp_ends);
  }
  std::vector<int> strides(in_dim_size, 1);

  dev_ctx.template Alloc<T>(out);

  MLUCnnlTensorDesc input_desc(x);
  MLUCnnlTensorDesc out_desc(slice_dims.size(),
                             phi::vectorize(slice_dims).data(),
                             ToCnnlDataType<T>());
  MLUCnnl::StridedSlice(dev_ctx,
                        starts.data(),
                        ends.data(),
                        strides.data(),
                        input_desc.get(),
                        GetBasePtr(&x),
                        out_desc.get(),
                        GetBasePtr(out));
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
  auto slice_dims = out_grad.dims();
  if (slice_dims.size() != in_dims.size()) {
    custom_kernel::CheckAndUpdateSliceAttrs(in_dims, axes, &starts, &ends);
    slice_dims = custom_kernel::GetSliceDims<int>(
        in_dims, axes, starts, ends, nullptr, nullptr);
  }

  int in_dim_size = x.dims().size();
  if (static_cast<int>(axes.size()) != in_dim_size) {
    std::vector<int> tmp_starts(in_dim_size, 0);
    const auto& in_dims_vec = phi::vectorize(x.dims());
    std::vector<int> tmp_ends(in_dims_vec.begin(), in_dims_vec.end());
    for (size_t i = 0; i < axes.size(); ++i) {
      tmp_starts[axes[i]] = starts[i];
      tmp_ends[axes[i]] = ends[i];
    }
    starts.swap(tmp_starts);
    ends.swap(tmp_ends);
  }
  std::vector<int> strides(in_dim_size, 1);

  dev_ctx.template Alloc<T>(x_grad);

  MLUCnnlTensorDesc dout_desc(slice_dims.size(),
                              phi::vectorize(slice_dims).data(),
                              ToCnnlDataType<T>());
  MLUCnnlTensorDesc x_grad_desc(*x_grad);
  MLUCnnl::StridedSliceGrad(dev_ctx,
                            starts.data(),
                            ends.data(),
                            strides.data(),
                            dout_desc.get(),
                            GetBasePtr(&out_grad),
                            x_grad_desc.get(),
                            GetBasePtr(x_grad));
}
}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(slice,
                          mlu,
                          ALL_LAYOUT,
                          custom_kernel::SliceRawKernel,
                          phi::dtype::float16,
                          float,
                          double,
                          int32_t,
                          int64_t,
                          bool) {}
PD_REGISTER_PLUGIN_KERNEL(slice_grad,
                          mlu,
                          ALL_LAYOUT,
                          custom_kernel::SliceGradRawKernel,
                          phi::dtype::float16,
                          float,
                          double,
                          int32_t,
                          int64_t,
                          bool) {}
