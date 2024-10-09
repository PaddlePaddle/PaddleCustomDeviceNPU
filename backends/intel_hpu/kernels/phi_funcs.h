// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may
// not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <algorithm>
#include <cmath>
#include <numeric>
#include <sstream>

#include "paddle/phi/capi/all.h"

namespace phi {

template <typename T>
T TolerableValue(const T& x) {
  const T kApproInf = 1e20;
  if (x == INFINITY) return kApproInf;
  if (x == -INFINITY) return -kApproInf;
  return x;
}

template <typename T>
static inline std::string to_string(const T& val) {
  std::stringstream ss;
  ss << val;
  return ss.str();
}

template <>
inline std::string to_string<phi::DataType>(const phi::DataType& val) {
  if (val == phi::DataType::FLOAT32) {
    return "float32";
  } else if (val == phi::DataType::FLOAT64) {
    return "float64";
  } else if (val == phi::DataType::INT32) {
    return "int32";
  } else if (val == phi::DataType::INT64) {
    return "int64";
  } else {
    return "undefined";
  }
}

template <>
inline std::string to_string<phi::DataLayout>(const phi::DataLayout& val) {
  if (val == phi::DataLayout::NCHW) {
    return "nchw";
  } else {
    return "undefined";
  }
}

template <typename T>
static inline std::string to_string(const std::vector<T>& vec) {
  std::stringstream ss;
  for (auto i = 0; i < vec.size(); ++i) {
    ss << to_string(vec[i]);
    if (i < vec.size() - 1) {
      ss << ", ";
    }
  }
  return ss.str();
}

static inline std::vector<int64_t> slice_ddim(const std::vector<int64_t>& dim,
                                              int begin,
                                              int end) {
  return std::vector<int64_t>(dim.cbegin() + begin, dim.cbegin() + end);
}

template <typename T>
static inline int64_t product(const std::vector<T>& ddim) {
  return std::accumulate(ddim.cbegin(), ddim.cend(), 1, std::multiplies<T>());
}

template <typename T>
T vec_product(const std::vector<T>& a, const std::vector<T>& b) {
  T ret = 0;
  for (auto i = 0; i < a.size(); ++i) {
    ret += a[i] * b[i];
  }
  return ret;
}

namespace funcs {

static inline int CanonicalAxis(const int axis, const int rank) {
  if (axis < 0) {
    return axis + rank;
  }
  return axis;
}

static inline int SizeToAxis(const int axis, std::vector<int64_t> dims) {
  int size = 1;
  for (int i = 0; i < axis; i++) {
    size *= dims[i];
  }
  return size;
}

static inline int SizeFromAxis(const int axis, std::vector<int64_t> dims) {
  int size = 1;
  for (int i = axis; i < dims.size(); i++) {
    size *= dims[i];
  }
  return size;
}

static inline int SizeOutAxis(const int axis, std::vector<int64_t> dims) {
  int size = 1;
  for (int i = axis + 1; i < dims.size(); i++) {
    size *= dims[i];
  }
  return size;
}

template <typename T = int64_t>
inline void CheckAndUpdateSliceAttrs(const std::vector<int64_t>& in_dims,
                                     const std::vector<T>& axes,
                                     std::vector<T>* starts,
                                     std::vector<T>* ends,
                                     std::vector<int64_t>* steps = nullptr,
                                     std::vector<T>* infer_flags = nullptr) {
  for (size_t i = 0; i < axes.size(); ++i) {
    T axis = axes[i];
    PD_CHECK(axis < in_dims.size(),
             "The axis value should be less than the rank of input, "
             "but received axes[%d] = %d, rank of input is %d.",
             i,
             axis,
             in_dims.size());

    if (infer_flags != nullptr && (*infer_flags)[i] == -1) {
      continue;
    }

    T dim_value = in_dims[axis];

    if (dim_value > 0) {
      T step = steps == nullptr ? 1 : (*steps)[i];
      PD_CHECK(
          step != 0, "Step should not be 0, but received step = %d.", step);

      T start = (*starts)[i] < 0 ? ((*starts)[i] + dim_value) : (*starts)[i];
      start = std::max(start, static_cast<T>(0));

      T end =
          0 < step && (*ends)[i] < 0 ? ((*ends)[i] + dim_value) : (*ends)[i];
      end = std::min(end, dim_value);

      if (step > 0) {
        start = std::min(start, dim_value);
        end = std::max(end, static_cast<T>(0));
        PD_CHECK(end >= start,
                 "When step > 0, end should be greater than start, but "
                 "received end = %d, start = %d.",
                 end,
                 start);
      } else {
        // NOTE(liym27): When step < 0, start should less and equal to
        // dim_value-1
        // "end is -1" means contain the 0-th element of this axis.
        start = std::min(start, dim_value - 1);
        end = std::max(end, static_cast<T>(-1));
        PD_CHECK(start >= end,
                 "When step < 0, start should be greater than end, but "
                 "received start = %d, end = %d.",
                 start,
                 end);
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
inline std::vector<int64_t> GetSliceDims(
    const std::vector<int64_t>& in_dims,
    const std::vector<T>& axes,
    const std::vector<T>& starts,
    const std::vector<T>& ends,
    std::vector<T>* steps = nullptr,
    std::vector<T>* infer_flags = nullptr) {
  std::vector<int64_t> slice_dims = in_dims;

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
inline std::vector<int64_t> GetDecreasedDims(
    const std::vector<int64_t>& slice_dims,
    const std::vector<T>& decrease_axes,
    std::vector<T>* infer_flags = nullptr) {
  std::vector<int64_t> decreased_dims = slice_dims;
  std::vector<uint8_t> decrease_flag(slice_dims.size(), 0);
  if (decrease_axes.size() > 0) {
    for (size_t i = 0; i < decrease_axes.size(); ++i) {
      T axis = decrease_axes[i];
      decrease_flag[axis] = 1;
      if (infer_flags && (*infer_flags)[i] != -1) {
        PD_CHECK(decreased_dims[axis] == 1,
                 "Decrease dim should be 1, but now received %d",
                 decreased_dims[axis]);
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
    decreased_dims = std::vector<int64_t>(new_shape.cbegin(), new_shape.cend());
  }
  return decreased_dims;
}

}  // namespace funcs

template <typename T>
static inline void BroadcastTo(const phi::Context& dev_ctx,
                               const phi::DenseTensor& in,
                               std::vector<int64_t> out_dims,
                               int axis,
                               phi::DenseTensor* out) {
  auto in_dims = in.dims();

  if (in_dims.size() == out_dims.size()) {
    bool broadcast = false;
    for (auto i = 0; i < in_dims.size(); ++i) {
      if (in_dims[i] != out_dims[i]) {
        broadcast = true;
        break;
      }
    }
    if (!broadcast) {
      out->ShareDataWith(in);
      return;
    }
  }

  out->Resize(out_dims);
  auto out_data = dev_ctx.template Alloc<T>(out);
  auto in_data = in.data<T>();

  axis = axis == -1 ? std::abs(static_cast<int>(in_dims.size()) -
                               static_cast<int>(out_dims.size()))
                    : axis;
  auto retain = static_cast<int>(out_dims.size()) - axis -
                static_cast<int>(in_dims.size());
  std::vector<size_t> tmp_dims;
  for (auto i = 0; i < axis; ++i) {
    tmp_dims.push_back(1);
  }
  tmp_dims.insert(tmp_dims.end(), in_dims.cbegin(), in_dims.cend());
  for (auto i = 0; i < retain; ++i) {
    tmp_dims.push_back(1);
  }

  auto numel = out->numel();
  std::vector<size_t> index(out_dims.size(), 0);
  std::vector<size_t> in_step(tmp_dims.size(), 1);
  std::vector<size_t> out_step(out_dims.size(), 1);
  for (auto i = tmp_dims.size() - 1; i > 0; --i) {
    in_step[i - 1] = in_step[i] * tmp_dims[i];
  }
  for (auto i = out_dims.size() - 1; i > 0; --i) {
    out_step[i - 1] = out_step[i] * out_dims[i];
  }

  for (auto i = 0; i < numel; ++i) {
    auto src_index = index;
    for (auto j = 0; j < tmp_dims.size(); ++j) {
      if (tmp_dims[j] == 1) {
        src_index[j] = 0;
      }
    }

    out_data[phi::vec_product(index, out_step)] =
        in_data[phi::vec_product(src_index, in_step)];

    index.back()++;
    for (auto j = index.size() - 1; j > 0; --j) {
      if (index[j] >= out_dims[j]) {
        index[j] = 0;
        index[j - 1]++;
      } else {
        break;
      }
    }
  }
}

static inline std::vector<int64_t> BroadcastDims(
    int axis,
    const std::vector<int64_t>& x_dims,
    const std::vector<int64_t>& y_dims) {
  axis = (axis == -1 ? std::abs(static_cast<int>(x_dims.size()) -
                                static_cast<int>(y_dims.size()))
                     : axis);
  std::vector<int64_t> dst_dims;
  if (x_dims.size() == y_dims.size()) {
    for (auto i = 0; i < x_dims.size(); ++i) {
      dst_dims.push_back(std::max(x_dims[i], y_dims[i]));
    }
  } else if (x_dims.size() >= y_dims.size()) {
    dst_dims = x_dims;
    for (auto i = 0; i < y_dims.size(); ++i) {
      dst_dims[axis + i] = std::max(dst_dims[axis + i], y_dims[i]);
    }
  } else {
    dst_dims = y_dims;
    for (auto i = 0; i < x_dims.size(); ++i) {
      dst_dims[axis + i] = std::max(dst_dims[axis + i], x_dims[i]);
    }
  }

  return dst_dims;
}

}  // namespace phi
