/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <memory>

#include "kernels/funcs/npu_enforce.h"
#include "kernels/funcs/npu_op_runner.h"
#include "runtime/runtime.h"

namespace custom_kernel {

template <typename T, typename Context>
void CastKernel(const Context& dev_ctx,
                const phi::DenseTensor& x,
                phi::DataType dtype,
                phi::DenseTensor* out);

/**
 * CPU -> NPU
 * NPU -> CPU
 * NPU -> NPU
 */
template <typename Context>
inline void TensorCopy(const Context& dev_ctx,
                       const phi::DenseTensor& src,
                       bool blocking,
                       phi::DenseTensor* dst,
                       const phi::Place& dst_place = phi::CustomPlace()) {
  auto* src_ptr = src.data();
  const auto& src_place = src.place();
  if (src_ptr == nullptr) {
    return;
  }
  auto dst_place_ = dst_place;
  if (dst_place_.GetType() != phi::AllocationType::CPU) {
    dst_place_ = dev_ctx.GetPlace();
  }

  if (&src == dst) {
    if (src_place == dst_place_) {
      VLOG(6) << "Skip copy the same data(" << src_ptr << ") from " << src_place
              << " to " << dst_place_;
    } else {
      VLOG(6) << "Src and dst are the same Tensor, in-place copy data("
              << src_ptr << ") from " << src_place << " to " << dst_place_;
      const phi::DenseTensor src_copy = src;
      TensorCopy(dev_ctx, src_copy, blocking, dst, dst_place_);
    }
    return;
  }

  VLOG(3) << "TensorCopy " << src.dims() << " from " << src_place << " to "
          << dst_place_;

  dst->Resize(src.dims());
  void* dst_ptr = nullptr;
  if (dst_place_.GetType() != phi::AllocationType::CPU) {
    dst_ptr = dev_ctx.Alloc(dst, src.dtype());
  } else {
    dst_ptr = dev_ctx.HostAlloc(dst, src.dtype());
  }

  PADDLE_ENFORCE_EQ(
      dst->place(),
      dst_place_,
      phi::errors::Unavailable(
          "The Dst Tensor's place and dst_place do not match, Tensor's place "
          "place is %s, dst_place is %s.",
          dst->place(),
          dst_place_));

  if (src_ptr == dst_ptr && src_place == dst_place_) {
    VLOG(3) << "Skip copy the same data async from " << src_ptr << " in "
            << src_place << " to " << dst_ptr << " in " << dst_place_;
    return;
  }
  VLOG(4) << "src:" << src_ptr << ", dst:" << dst_ptr;

  C_Stream stream = static_cast<C_Stream>(dev_ctx.stream());

  auto size =
      (src.dims().size() != 0 ? src.numel() : 1) * phi::SizeOf(src.dtype());
  if (UNLIKELY(size) == 0) {
    return;
  }

  if (src_place.GetType() == phi::AllocationType::CPU &&
      dst_place_.GetType() == phi::AllocationType::CUSTOM) {
    C_Device_st device{dst_place_.GetDeviceId()};
    AsyncMemCpyH2D(&device, stream, dst_ptr, src_ptr, size);
    if (blocking) {
      dev_ctx.Wait();
    }
  } else if (src_place.GetType() == phi::AllocationType::CUSTOM &&
             dst_place_.GetType() == phi::AllocationType::CPU) {
    C_Device_st device{src_place.GetDeviceId()};
    AsyncMemCpyD2H(&device, stream, dst_ptr, src_ptr, size);
    if (blocking) {
      dev_ctx.Wait();
    }
  } else if (src_place.GetType() == phi::AllocationType::CUSTOM &&
             dst_place_.GetType() == phi::AllocationType::CUSTOM) {
    if (src_place.GetDeviceType() == dst_place_.GetDeviceType()) {
      if (src_place.GetDeviceId() == dst_place_.GetDeviceId()) {
        C_Device_st device{src_place.GetDeviceId()};
        AsyncMemCpyD2D(&device, stream, dst_ptr, src_ptr, size);
        if (blocking) {
          dev_ctx.Wait();
        }
      } else {
        PADDLE_THROW(
            phi::errors::Unimplemented("TensorCopy is not supported."));
      }
    } else {
      PADDLE_THROW(phi::errors::Unimplemented("TensorCopy is not supported."));
    }
  } else if (src_place.GetType() == phi::AllocationType::CPU &&
             dst_place_.GetType() == phi::AllocationType::CPU) {
    std::memcpy(dst_ptr, src_ptr, size);
  }
}

/**
 * CPU -> NPU
 */
template <typename T>
inline void TensorFromVector(const phi::CustomContext& ctx,
                             const std::vector<T>& src,
                             const phi::CustomContext& dev_ctx,
                             phi::DenseTensor* dst) {
  auto dst_place = dev_ctx.GetPlace();
  auto src_ptr = static_cast<const void*>(src.data());
  dst->Resize({static_cast<int64_t>(src.size())});
  auto dst_ptr = static_cast<void*>(dev_ctx.template Alloc<T>(dst));
  auto size = src.size() * sizeof(T);
  if (UNLIKELY(size == 0)) return;

  if (dst_place.GetType() == phi::AllocationType::CUSTOM) {
    C_Device_st device{dst_place.GetDeviceId()};
    AsyncMemCpyH2D(&device,
                   static_cast<C_Stream>(dev_ctx.stream()),
                   dst_ptr,
                   src_ptr,
                   size);
  } else {
    PADDLE_THROW(phi::errors::Unimplemented(
        "TensorFromVector on %s is not supported.", dst_place));
  }
}

template <>
inline void TensorFromVector<bool>(const phi::CustomContext& ctx,
                                   const std::vector<bool>& src,
                                   const phi::CustomContext& dev_ctx,
                                   phi::DenseTensor* dst) {
  // vector<bool> has no data() member, use array instead.
  // See details:
  // https://stackoverflow.com/questions/46115669/why-does-stdvectorbool-have-no-data/46115714
  bool* array = new bool[src.size()];
  for (unsigned int i = 0; i < src.size(); i++) {
    array[i] = static_cast<bool>(src[i]);
  }

  auto dst_place = dev_ctx.GetPlace();
  auto src_ptr = static_cast<const void*>(array);
  dst->Resize({static_cast<int64_t>(src.size())});
  auto dst_ptr = static_cast<void*>(dev_ctx.template Alloc<bool>(dst));
  auto size = src.size() * sizeof(bool);
  if (UNLIKELY(size == 0)) return;

  if (dst_place.GetType() == phi::AllocationType::CUSTOM) {
    C_Device_st device{dst_place.GetDeviceId()};
    AsyncMemCpyH2D(&device,
                   static_cast<C_Stream>(dev_ctx.stream()),
                   dst_ptr,
                   src_ptr,
                   size);
  } else {
    PADDLE_THROW(phi::errors::Unimplemented(
        "TensorFromVector on %s is not supported.", dst_place));
  }
  delete[] array;
}

/**
 * CPU -> CPU
 * CPU -> NPU
 */
template <typename T>
inline void TensorFromVector(const phi::CustomContext& ctx,
                             const std::vector<T>& src,
                             const phi::CPUContext& dev_ctx,
                             phi::DenseTensor* dst) {
  auto dst_place = dev_ctx.GetPlace();
  auto src_ptr = static_cast<const void*>(src.data());
  dst->Resize({src.size()});
  auto dst_ptr = ctx.template HostAlloc<T>(dst);
  auto size = src.size() * sizeof(T);
  if (UNLIKELY(size == 0)) return;

  if (dst_place.GetType() == phi::AllocationType::CPU) {
    VLOG(4) << "src_ptr: " << src_ptr << ", dst_ptr: " << dst_ptr
            << ", size: " << size;
    std::memcpy(dst_ptr, src_ptr, size);
  } else if (dst_place.GetType() == phi::AllocationType::CUSTOM) {
    C_Device_st device{dst_place.GetDeviceId()};
    AsyncMemCpyH2D(
        &device, static_cast<C_Stream>(ctx.stream()), dst_ptr, src_ptr, size);
  } else {
    PADDLE_THROW(phi::errors::Unimplemented(
        "TensorFromVector on %s is not supported.", dst_place));
  }
}

template <>
inline void TensorFromVector<bool>(const phi::CustomContext& ctx,
                                   const std::vector<bool>& src,
                                   const phi::CPUContext& dev_ctx,
                                   phi::DenseTensor* dst) {
  auto dst_place = dev_ctx.GetPlace();
  PADDLE_THROW(phi::errors::Unimplemented(
      "TensorFromVector on %s is not supported.", dst_place));
}

template <typename T>
void TensorFromArray(const phi::CustomContext& ctx,
                     const T* src,
                     const size_t& array_size,
                     const phi::CustomContext& dev_ctx,
                     phi::DenseTensor* dst) {
  auto dst_place = dev_ctx.GetPlace();
  auto src_ptr = static_cast<const void*>(src);
  dst->Resize({static_cast<int64_t>(array_size)});
  auto dst_ptr = static_cast<void*>(dev_ctx.template Alloc<T>(dst));
  auto size = array_size * sizeof(T);

  if (dst_place.GetType() == phi::AllocationType::CUSTOM) {
    C_Device_st device{dst_place.GetDeviceId()};
    AsyncMemCpyH2D(&device,
                   static_cast<C_Stream>(dev_ctx.stream()),
                   dst_ptr,
                   src_ptr,
                   size);
  } else {  // NOLINT
    PADDLE_THROW(phi::errors::Unimplemented(
        "TensorFromArray on %s is not supported.", dst_place));
  }
}

/**
 * NPU -> CPU
 */
template <typename T>
inline void TensorToVector(const phi::CustomContext& ctx,
                           const phi::DenseTensor& src,
                           const phi::CustomContext& dev_ctx,
                           std::vector<T>* dst) {
  auto src_ptr = static_cast<const void*>(src.data<T>());
  auto size = src.numel() * sizeof(T);

  dst->resize(src.numel());
  auto dst_ptr = static_cast<void*>(dst->data());

  auto src_place = src.place();

  if (src_place.GetType() == phi::AllocationType::CUSTOM) {
    C_Device_st device{src_place.GetDeviceId()};
    AsyncMemCpyD2H(
        &device, static_cast<C_Stream>(ctx.stream()), dst_ptr, src_ptr, size);
    ctx.Wait();
  } else {
    PADDLE_THROW(phi::errors::Unimplemented(
        "TensorToVector on %s is not supported.", src_place));
  }
}

template <>
inline void TensorToVector<bool>(const phi::CustomContext& ctx,
                                 const phi::DenseTensor& src,
                                 const phi::CustomContext& dev_ctx,
                                 std::vector<bool>* dst) {
  auto src_ptr = static_cast<const void*>(src.data<bool>());
  C_Stream stream = static_cast<C_Stream>(ctx.stream());
  auto size = src.numel() * sizeof(bool);

  bool* array = new bool[src.numel()];

  phi::CPUPlace dst_place;
  dst->resize(src.numel());
  auto dst_ptr = static_cast<void*>(array);

  auto src_place = src.place();
  if (src_place.GetType() == phi::AllocationType::CUSTOM) {
    C_Device_st device{src_place.GetDeviceId()};
    AsyncMemCpyD2H(&device, stream, dst_ptr, src_ptr, size);
    ctx.Wait();
  } else {
    PADDLE_THROW(phi::errors::Unimplemented(
        "TensorToVector on %s is not supported.", src_place));
  }
  for (unsigned int i = 0; i < src.numel(); i++) {
    (*dst)[i] = static_cast<bool>(array[i]);
  }
  delete[] array;
}

/**
 * CPU -> NPU
 */
template <typename T>
inline void FillNpuTensorWithConstant(phi::DenseTensor* dst,
                                      const phi::CustomContext& dev_ctx,
                                      T val) {
  int numel = dst->numel();
  std::vector<T> src(numel, static_cast<T>(val));
  TensorFromVector(dev_ctx, src, dev_ctx, dst);
}

// src - broadcast -> transformed_src
template <typename T, typename Context>
inline void NpuBroadcast(const Context& dev_ctx,
                         const phi::DenseTensor* src,
                         int axis,
                         const phi::DDim& dst_dims,
                         phi::DenseTensor* transformed_src) {
  auto stream = dev_ctx.stream();

  // 1. expand the axis with dim 1
  auto src_dims = src->dims();
  phi::DenseTensor tmp_src(*src);
  tmp_src.Resize(src_dims);
  for (int i = 0; i < src_dims.size(); ++i) {
    if (src_dims[i] == 1 && dst_dims[i + axis] > 1) {
      phi::DenseTensor tmp_tensor;
      auto tmp_tensor_dims = tmp_src.dims();
      tmp_tensor_dims[i] = dst_dims[i + axis];
      tmp_tensor.Resize(tmp_tensor_dims);
      dev_ctx.template Alloc<T>(&tmp_tensor);
      const auto& runner =
          NpuOpRunner("TileWithAxis",
                      {tmp_src},
                      {tmp_tensor},
                      {{"axis", static_cast<int64_t>(i)},
                       {"tiles", static_cast<int64_t>(dst_dims[i + axis])}});
      runner.Run(stream);
      tmp_src = tmp_tensor;
      tmp_src.Resize(tmp_tensor_dims);
    }
  }

  // 2.expand the ahead axis
  auto prev = phi::product(phi::slice_ddim(dst_dims, 0, axis));
  if (prev > 1) {
    phi::DenseTensor tmp_tensor;
    auto tmp_tensor_dims = phi::slice_ddim(dst_dims, 0, axis + src_dims.size());
    tmp_tensor.Resize(tmp_tensor_dims);
    dev_ctx.template Alloc<T>(&tmp_tensor);

    // float64 fix
    std::vector<int64_t> final_expand_shape;
    for (int64_t i = 0; i < tmp_tensor_dims.size(); ++i) {
      final_expand_shape.push_back(tmp_tensor_dims[i]);
    }
    if (tmp_src.dtype() == phi::DataType::FLOAT64) {
      phi::DenseTensor cast_x;
      phi::DenseTensorMeta meta(tmp_src.meta());
      meta.dtype = phi::DataType::FLOAT32;
      cast_x.set_meta(meta);

      custom_kernel::CastKernel<T, Context>(
          dev_ctx, tmp_src, phi::DataType::FLOAT32, &cast_x);

      phi::DenseTensor cast_out;
      phi::DenseTensorMeta cast_out_meta = {phi::DataType::FLOAT32,
                                            tmp_tensor.dims()};
      cast_out.set_meta(cast_out_meta);
      dev_ctx.template Alloc<float>(&cast_out);

      EXEC_NPU_CMD(aclnnExpand, dev_ctx, cast_x, final_expand_shape, cast_out);
      custom_kernel::CastKernel<T, Context>(
          dev_ctx, cast_out, phi::DataType::FLOAT64, &tmp_tensor);
    } else {
      EXEC_NPU_CMD(
          aclnnExpand, dev_ctx, tmp_src, final_expand_shape, tmp_tensor);
    }

    tmp_src = tmp_tensor;
    tmp_src.Resize(tmp_tensor_dims);
  } else {
    tmp_src.Resize(phi::slice_ddim(dst_dims, 0, axis + src_dims.size()));
  }

  // 3.expand the tail axis
  auto post = phi::product(
      phi::slice_ddim(dst_dims, axis + src_dims.size(), dst_dims.size()));
  if (post > 1) {
    auto src_dims_vec = phi::vectorize<int>(tmp_src.dims());
    src_dims_vec.push_back(1);
    tmp_src.Resize(phi::make_ddim(src_dims_vec));

    phi::DenseTensor tmp_tensor;
    tmp_tensor.Resize(dst_dims);
    dev_ctx.template Alloc<T>(&tmp_tensor);
    const auto& runner =
        NpuOpRunner("TileWithAxis",
                    {tmp_src},
                    {tmp_tensor},
                    {{"axis", static_cast<int64_t>(axis + src_dims.size())},
                     {"tiles", static_cast<int64_t>(post)}});
    runner.Run(stream);
    tmp_src = tmp_tensor;
  }
  tmp_src.Resize(dst_dims);
  TensorCopy(dev_ctx, tmp_src, false, transformed_src);
}

inline void NpuElementWiseHelper(const phi::DenseTensor* x,
                                 const phi::DenseTensor* y,
                                 int axis,
                                 int* x_axis,
                                 int* y_axis,
                                 phi::DDim* dst_dims) {
  auto x_dims = x->dims();
  auto y_dims = y->dims();
  bool is_xsize_larger = true;
  int max_dim = x_dims.size();
  std::vector<int> dst_dims_vec = phi::vectorize<int>(x_dims);

  if (x_dims.size() < y_dims.size()) {
    is_xsize_larger = false;
    max_dim = y_dims.size();
    dst_dims_vec = phi::vectorize<int>(y_dims);
  }

  axis = (axis == -1 ? std::abs(x_dims.size() - y_dims.size()) : axis);
  *x_axis = is_xsize_larger ? 0 : axis;
  *y_axis = is_xsize_larger ? axis : 0;
  PADDLE_ENFORCE_GE(
      axis,
      0,
      phi::errors::InvalidArgument(
          "Axis should be great than or equal to 0, but received axis is %d.",
          axis));
  PADDLE_ENFORCE_LE(
      axis,
      max_dim,
      phi::errors::InvalidArgument(
          "Axis should be less than or equal to %d, but received axis is %d.",
          max_dim,
          axis));

  for (int i = 0; i < x_dims.size(); ++i) {
    dst_dims_vec[i + *x_axis] =
        std::max(dst_dims_vec[i + *x_axis], static_cast<int>(x_dims[i]));
  }
  for (int i = 0; i < y_dims.size(); ++i) {
    dst_dims_vec[i + *y_axis] =
        std::max(dst_dims_vec[i + *y_axis], static_cast<int>(y_dims[i]));
  }
  *dst_dims = phi::make_ddim(dst_dims_vec);
}

template <typename T, typename Context>
inline void NpuElementWiseOpBroadcast(const Context& dev_ctx,
                                      const phi::DenseTensor* x,
                                      const phi::DenseTensor* y,
                                      int axis,
                                      phi::DenseTensor* transformed_x,
                                      phi::DenseTensor* transformed_y) {
  int x_axis, y_axis;
  phi::DDim dst_dims;
  NpuElementWiseHelper(x, y, axis, &x_axis, &y_axis, &dst_dims);
  NpuBroadcast<T>(dev_ctx, x, x_axis, dst_dims, transformed_x);
  NpuBroadcast<T>(dev_ctx, y, y_axis, dst_dims, transformed_y);
}

static inline int CanonicalAxis(const int axis, const int rank) {
  if (axis < 0) {
    return axis + rank;
  }
  return axis;
}

inline phi::DataLayout StringToDataLayout(const std::string& str) {
  std::string s(str);
  for (size_t i = 0; i < s.size(); ++i) {
    s[i] = toupper(s[i]);
  }

  if (s == "NHWC") {
    return phi::DataLayout::kNHWC;
  } else if (s == "NCHW") {
    return phi::DataLayout::kNCHW;
  } else if (s == "ANYLAYOUT") {
    return phi::DataLayout::kAnyLayout;
  } else if (s == "MKLDNNLAYOUT") {
    return phi::DataLayout::kMKLDNN;
  } else if (s == "SPARSE_COO") {
    return phi::DataLayout::SPARSE_COO;
  } else if (s == "SPARSE_CSR") {
    return phi::DataLayout::SPARSE_CSR;
  } else {
  }
}

inline void ExtractNCDWH(const phi::DDim& dims,
                         const phi::DataLayout& data_layout,
                         int* N,
                         int* C,
                         int* D,
                         int* H,
                         int* W) {
  *N = dims[0];

  if (dims.size() == 3) {
    *C = data_layout == phi::DataLayout::kNCHW ? dims[1] : dims[2];
    *D = 1;
    *H = 1;
    *W = data_layout == phi::DataLayout::kNCHW ? dims[2] : dims[1];
  } else if (dims.size() == 4) {
    *C = data_layout == phi::DataLayout::kNCHW ? dims[1] : dims[3];
    *D = 1;
    *H = data_layout == phi::DataLayout::kNCHW ? dims[2] : dims[1];
    *W = data_layout == phi::DataLayout::kNCHW ? dims[3] : dims[2];
  } else {
    *C = data_layout == phi::DataLayout::kNCHW ? dims[1] : dims[4];
    *D = data_layout == phi::DataLayout::kNCHW ? dims[2] : dims[1];
    *H = data_layout == phi::DataLayout::kNCHW ? dims[3] : dims[2];
    *W = data_layout == phi::DataLayout::kNCHW ? dims[4] : dims[3];
  }
}

template <typename T>
inline std::vector<T> get_new_data_from_tensor(
    const phi::CustomContext& dev_ctx,
    const phi::DenseTensor* new_data_tensor) {
  std::vector<T> vec_new_data;
  auto* new_data = new_data_tensor->data<T>();
  phi::DenseTensor cpu_starts_tensor;
  if (new_data_tensor->place().GetType() == phi::AllocationType::CUSTOM) {
    TensorCopy(
        dev_ctx, *new_data_tensor, true, &cpu_starts_tensor, phi::CPUPlace());
    new_data = cpu_starts_tensor.data<T>();
  }
  vec_new_data = std::vector<T>(new_data, new_data + new_data_tensor->numel());
  return vec_new_data;
}

}  // namespace custom_kernel
