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

#include <cstdint>

#include "paddle/phi/backends/device_ext.h"
#include "paddle/phi/common/type_traits.h"
#include "paddle/phi/extension.h"
#include "runtime/runtime.h"

typedef std::vector<int64_t> DIMS;

namespace custom_kernel {

inline synDataType PDDataTypeToSynDataType(phi::DataType type) {
  if (type == phi::DataType::FLOAT32) {
    return syn_type_single;
  } else if (type == phi::DataType::FLOAT16) {
    return syn_type_fp16;
  } else if (type == phi::DataType::BFLOAT16) {
    return syn_type_bf16;
  } else if (type == phi::DataType::INT32) {
    return syn_type_int32;
  } else if (type == phi::DataType::INT8 || type == phi::DataType::BOOL) {
    return syn_type_int8;
  } else if (type == phi::DataType::UINT8) {
    return syn_type_uint8;
  } else if (type == phi::DataType::INT64) {
    return syn_type_int64;
  } else {
    PD_CHECK(false, "Unsupported cast dtype %d", type);
  }
}

inline std::string SynDataTypeToStr(synDataType s) {
  if (s == syn_type_int8 || s == syn_type_fixed) {
    return "i8";
  } else if (s == syn_type_single || s == syn_type_float) {
    return "f32";
  } else if (s == syn_type_fp8_143 || s == syn_type_fp8_152) {
    return "hf8";
  } else {
    switch (s) {
      case syn_type_bf16:
        return "bf16";
      case syn_type_int16:
        return "i16";
      case syn_type_uint8:
        return "u8";
      case syn_type_fp16:
        return "f16";
      case syn_type_int32:
        return "i32";
      case syn_type_int64:
        return "i64";
      case syn_type_uint64:
        return "u64";
      default:
        PD_CHECK(false, "Unsupported dtype %d", s);
    }
  }
}

/**
 * CPU       -> INTEL_HPU
 * INTEL_HPU -> CPU
 * INTEL_HPU -> INTEL_HPU
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
        PADDLE_THROW("TensorCopy is not supported.");
      }

    } else {
      PADDLE_THROW("TensorCopy is not supported.");
    }
  } else if (src_place.GetType() == phi::AllocationType::CPU &&
             dst_place_.GetType() == phi::AllocationType::CPU) {
    std::memcpy(dst_ptr, src_ptr, size);
  }
}

inline int CanonicalAxis(const int axis, const int rank) {
  if (axis < 0) {
    return axis + rank;
  }
  return axis;
}

struct TensorInfo {
  std::string name;
  std::vector<int64_t> dims;
  uint64_t device_addr;
  const void* host_addr;
  synDataType type;
  size_t num_elements;
};

class ConvertTensors {
 public:
  ConvertTensors() : count_(0) {}
  ~ConvertTensors() {
    y_tensors_.clear();
    x_tensors_.clear();
    x_host_tensor_.clear();
    y_host_tensor_.clear();
  }

  void Add(const phi::DenseTensor& x, bool is_input = true) {
    phi::DenseTensorMeta meta = x.meta();
    auto addr = x.data();

    phi::vectorize<int64_t>(x.dims());
    if (is_input) {
      auto it = x_tensors_.find(addr);
      if (it == x_tensors_.end()) {
        // new tensor
        TensorInfo info;
        info.name = "x_" + std::to_string(count_);
        count_++;
        info.dims = phi::vectorize<int64_t>(x.dims());
        if (x.dims().size() == 0) {
          info.dims.push_back({1});
        }
        info.host_addr = addr;
        info.device_addr = reinterpret_cast<uint64_t>(addr);
        info.type = PDDataTypeToSynDataType(x.dtype());
        info.num_elements = x.numel();
        x_tensors_.insert({addr, info});
        VLOG(6) << "add tensor " << info.name << ", " << addr;
      }
      x_host_tensor_.push_back(addr);
    } else {
      auto it = y_tensors_.find(addr);
      if (it == y_tensors_.end()) {
        // new tensor
        TensorInfo info;
        info.name = "y_" + std::to_string(count_);
        count_++;
        info.dims = phi::vectorize<int64_t>(x.dims());
        if (x.dims().size() == 0) {
          info.dims.push_back({1});
        }
        info.host_addr = addr;
        info.device_addr = reinterpret_cast<uint64_t>(addr);
        info.type = PDDataTypeToSynDataType(x.dtype());
        info.num_elements = x.numel();
        y_tensors_.insert({addr, info});
        VLOG(6) << "add tensor " << info.name << ", " << addr;
      }
      y_host_tensor_.push_back(addr);
    }
  }

  void Add(const phi::DenseTensor* x, bool is_input = true) {
    Add(*x, is_input);
  }

  std::vector<DIMS> GetDims(bool is_input = true) {
    std::vector<DIMS> out;
    if (is_input) {
      for (size_t i = 0; i < x_host_tensor_.size(); i++) {
        out.push_back(x_tensors_[x_host_tensor_[i]].dims);
      }
    } else {
      for (size_t i = 0; i < y_tensors_.size(); i++) {
        out.push_back(y_tensors_[y_host_tensor_[i]].dims);
      }
    }

    return out;
  }

  std::map<std::string, uint64_t> GetDeviceAddr() {
    std::map<std::string, uint64_t> out;
    for (auto it = x_tensors_.begin(); it != x_tensors_.end(); it++) {
      out.insert({it->second.name, it->second.device_addr});
    }

    for (auto it = y_tensors_.begin(); it != y_tensors_.end(); it++) {
      out.insert({it->second.name, it->second.device_addr});
    }
    return out;
  }

  std::vector<TensorInfo> GetTensors(bool is_input = true) {
    std::vector<TensorInfo> out;
    if (is_input) {
      for (size_t i = 0; i < x_host_tensor_.size(); i++) {
        out.push_back(x_tensors_[x_host_tensor_[i]]);
      }
    } else {
      for (size_t i = 0; i < y_tensors_.size(); i++) {
        out.push_back(y_tensors_[y_host_tensor_[i]]);
      }
    }
    return out;
  }

 protected:
  std::vector<const void*> x_host_tensor_;
  std::map<const void*, TensorInfo> x_tensors_;
  std::vector<const void*> y_host_tensor_;
  std::map<const void*, TensorInfo> y_tensors_;
  int32_t count_;
};

}  // namespace custom_kernel
