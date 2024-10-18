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

#include "kernels/funcs/op_utils.h"

#include "paddle/common/flags.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/core/dense_tensor.h"

PHI_DECLARE_bool(use_stride_kernel);

namespace custom_kernel {

void *GcuDataPtr(const phi::DenseTensor &tensor) {
  if (tensor.initialized()) {
    auto contiguous_strides = phi::DenseTensorMeta::calc_strides(tensor.dims());
    bool is_contiguous = (tensor.strides() == contiguous_strides);
    auto tensor_tmp = tensor;
    if (!is_contiguous && !FLAGS_use_stride_kernel) {
      auto meta = tensor_tmp.meta();
      meta.strides = contiguous_strides;
      tensor_tmp.set_meta(meta);
    }
    return const_cast<void *>(tensor_tmp.data());
  }
  return nullptr;
}

std::string TensorToString(const phi::DenseTensor &tensor) {
  std::stringstream ss;
  ss << "LoDTensor<";
  if (tensor.initialized()) {
    ss << phi::DataTypeToString(tensor.dtype()) << ", ";
    ss << tensor.place() << ", ";
    ss << "Shape(" << tensor.dims() << "), ";
    ss << "layout:" << tensor.layout();
  } else {
    ss << "NOT_INITED";
  }
  ss << ">";
  return ss.str();
}

std::string TensorVectorToString(const std::vector<phi::DenseTensor> &tensors) {
  std::stringstream ss;
  ss << "{";
  if (tensors.size() == 0) {
    ss << " There is no tensor!\n";
  } else {
    for (const auto &tensor : tensors) {
      ss << TensorToString(tensor);
      ss << ", ";
    }
  }
  ss << "}";
  return ss.str();
}

std::string TensorDetailsToString(const phi::DenseTensor &tensor) {
  std::stringstream ss;
  ss << "LoDTensor<";
  if (tensor.initialized()) {
    ss << phi::DataTypeToString(tensor.dtype()) << ", ";
    ss << tensor.place() << ", ";
    ss << "dims(" << tensor.dims() << "), ";
    ss << "strides(" << tensor.strides() << "), ";
    ss << "layout:" << tensor.layout() << ", ";
    ss << "data:" << const_cast<void *>(tensor.data());
  } else {
    ss << "NOT_INITED";
  }
  ss << ">";
  return ss.str();
}

std::string ScalarToString(const phi::Scalar &scalar_value) {
  std::stringstream ss;
  auto scalar_type = scalar_value.dtype();
  ss << "Scalar<" << phi::DataTypeToString(scalar_type) << ", ";
  switch (scalar_type) {
    case phi::DataType::BOOL:
      ss << scalar_value.to<bool>();
      break;
    case phi::DataType::UINT8:
      ss << scalar_value.to<uint8_t>();
      break;
    case phi::DataType::INT8:
      ss << scalar_value.to<int8_t>();
      break;
    case phi::DataType::INT16:
      ss << scalar_value.to<int16_t>();
      break;
    case phi::DataType::INT32:
      ss << scalar_value.to<int32_t>();
      break;
    case phi::DataType::INT64:
      ss << scalar_value.to<int64_t>();
      break;
    case phi::DataType::FLOAT16:
      ss << scalar_value.to<phi::float16>();
      break;
    case phi::DataType::BFLOAT16:
      ss << scalar_value.to<phi::bfloat16>();
      break;
    case phi::DataType::FLOAT32:
      ss << scalar_value.to<float>();
      break;
    case phi::DataType::FLOAT64:
      ss << scalar_value.to<double>();
      break;
    default: {
      PADDLE_THROW(phi::errors::Unimplemented(
          "ScalarToTopsatenScalar, unsupported data type %s",
          phi::DataTypeToString(scalar_type).c_str()));
      break;
    }
  }
  ss << ">";
  return ss.str();
}

std::string GetOpNameFromCallStatement(const std::string &call_state) {
  return call_state.substr(0, call_state.find_first_of("("));
}

std::vector<int64_t> InferSize(const std::vector<int64_t> &a,
                               const std::vector<int64_t> &b) {
  size_t dims_a = a.size();
  size_t dims_b = b.size();
  size_t ndim = dims_a > dims_b ? dims_a : dims_b;
  std::vector<int64_t> expanded_dims(ndim);

  // Use ptrdiff_t to ensure signed comparison.
  for (ptrdiff_t i = (ptrdiff_t)ndim - 1; i >= 0; --i) {
    ptrdiff_t offset = ndim - 1 - i;
    ptrdiff_t dim_a = dims_a - 1 - offset;
    ptrdiff_t dim_b = dims_b - 1 - offset;
    auto size_a = (dim_a >= 0) ? a[dim_a] : 1;
    auto size_b = (dim_b >= 0) ? b[dim_b] : 1;

    PADDLE_ENFORCE_EQ(size_a == size_b || size_a == 1 || size_b == 1,
                      true,
                      phi::errors::InvalidArgument(
                          "The size of tensor a (%ld) must match the size of "
                          "tensor b(%ld) at non-singleton dimension %d.",
                          size_a,
                          size_b,
                          i));

    // 1s map to the other size (even 0).
    expanded_dims[i] = size_a == 1 ? std::move(size_b) : std::move(size_a);
  }

  return expanded_dims;
}

std::vector<int64_t> ComputeBroadcastShape(
    const std::vector<phi::DenseTensor> operands) {
  auto operands_size = operands.size();
  PADDLE_ENFORCE_GT(
      operands_size,
      1,
      phi::errors::InvalidArgument(
          "ComputeBroadcastShape operands size must be 2 or more, but get %d.",
          operands_size));
  auto real_shape = phi::vectorize(operands[0].dims());
  for (int i = 1; i < operands_size; ++i) {
    auto shape = InferSize(real_shape, phi::vectorize(operands[i].dims()));
    real_shape = shape;
  }
  return real_shape;
}

void GcuOpMaybeStreamSync(const phi::DeviceContext &dev_ctx) {
  static const char *stream_async_env = std::getenv(env::kStreamAsync);
  static bool stream_async = ((stream_async_env == nullptr) ||
                              (stream_async_env != nullptr &&
                               std::string(stream_async_env) == "true"));
  static bool stream_sync = ((VLOG(0) << "AOT kernel stream mode:"
                                      << (stream_async ? "async" : "sync")),
                             (!stream_async));
  if (stream_sync) {
    dev_ctx.Wait();
  }
}
}  // namespace custom_kernel
