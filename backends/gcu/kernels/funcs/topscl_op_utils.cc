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

#include "kernels/funcs/topscl_op_utils.h"

#include "paddle/phi/common/data_type.h"
#include "paddle/phi/core/dense_tensor.h"

namespace custom_kernel {
topscl::Device GetCurrentDevice() {
  int device_id = 0;
  RT_CHECK(topsGetDevice(&device_id));
  return topscl::Device(topscl::DeviceType::GCU, device_id);
}

topscl::Tensor CreateTopsclTensor(const phi::DenseTensor &tensor, bool pinned) {
  if (UNLIKELY(!tensor.initialized())) {
    VLOG(6) << "Create default topsatenTensor.";
    return topscl::Tensor();
  }
  if (LIKELY(!pinned)) {
    PADDLE_ENFORCE_EQ(
        tensor.place().GetDeviceType(),
        "gcu",
        phi::errors::InvalidArgument("Not gcu tensor, current tensor place: %s",
                                     tensor.place().GetDeviceType().c_str()));
  }

  std::vector<int64_t> tensor_dims = phi::vectorize(tensor.dims());
  std::vector<int64_t> strides = phi::vectorize(tensor.strides());
  int64_t rank = tensor_dims.size();
  if (rank == 0) {
    rank = 1;
    tensor_dims = {1};
    strides = {1};
  }
  topscl::Shape xdims(tensor_dims);
  auto xdtype = DataTypeToTopsclDataType(tensor.dtype());
  auto xformat = DataLayoutToTopsclMemoryFormat(tensor.layout());
  auto device = GetCurrentDevice();

  VLOG(6) << "CreateTopsclTensor, input tensor:" << TensorToString(tensor);

  auto xt = topscl::Tensor(
      xdims, xdtype, strides, GcuDataPtr(tensor), xformat, device);
  VLOG(6) << "Create topscl Tensor: " << xt;
  return xt;
}

topscl::Tensor OptionalTensorToTopsclTensor(
    const paddle::optional<phi::DenseTensor> &opt_tensor) {
  if (opt_tensor) {
    return CreateTopsclTensor(opt_tensor.get());
  } else {
    return topscl::Tensor();
  }
}

topscl::DType DataTypeToTopsclDataType(const phi::DataType &dtype) {
  static std::unordered_map<phi::DataType, topscl::DType>
      kPhiDTypeToTopsclDtype = {
          {phi::DataType::BOOL, topscl::DType::kBool},
          {phi::DataType::UINT8, topscl::DType::kUInt8},
          {phi::DataType::INT8, topscl::DType::kInt8},
          {phi::DataType::INT16, topscl::DType::kInt16},
          {phi::DataType::INT32, topscl::DType::kInt32},
          {phi::DataType::INT64, topscl::DType::kInt64},
          {phi::DataType::FLOAT16, topscl::DType::kFloat16},
          {phi::DataType::BFLOAT16, topscl::DType::kBFloat16},
          {phi::DataType::FLOAT32, topscl::DType::kFloat32},
          {phi::DataType::FLOAT64, topscl::DType::kFloat64},
      };
  if (kPhiDTypeToTopsclDtype.count(dtype) == 0) {
    PADDLE_THROW(phi::errors::Unimplemented(
        "Unsupported data type %s", phi::DataTypeToString(dtype).c_str()));
  }
  return kPhiDTypeToTopsclDtype.at(dtype);
}

topscl::MemoryFormat DataLayoutToTopsclMemoryFormat(
    const common::DataLayout &layout) {
  static std::unordered_map<common::DataLayout, topscl::MemoryFormat>
      kDataLayoutToTopsclMemoryFormat = {
          {common::DataLayout::ANY, topscl::MemoryFormat::kAny},
          {common::DataLayout::NHWC, topscl::MemoryFormat::kNHWC},
          {common::DataLayout::NCHW, topscl::MemoryFormat::kNCHW},
          {common::DataLayout::NCDHW, topscl::MemoryFormat::kNCDHW},
          {common::DataLayout::NDHWC, topscl::MemoryFormat::kNDHWC},
      };
  if (kDataLayoutToTopsclMemoryFormat.count(layout) == 0) {
    PADDLE_THROW(
        phi::errors::Unimplemented("Unsupported data layout %s",
                                   common::DataLayoutToString(layout).c_str()));
  }
  return kDataLayoutToTopsclMemoryFormat.at(layout);
}

topscl::Scalar ScalarToTopsclScalar(const phi::Scalar &scalar_value) {
  auto scalar_type = scalar_value.dtype();
  switch (scalar_type) {
    case phi::DataType::BOOL: {
      bool b_value = scalar_value.to<bool>();
      return topscl::Scalar(b_value);
    }
    case phi::DataType::UINT8: {
      uint8_t u8_value = scalar_value.to<uint8_t>();
      return topscl::Scalar(u8_value);
    }
    case phi::DataType::INT8: {
      int8_t i8_value = scalar_value.to<int8_t>();
      return topscl::Scalar(i8_value);
    }
    case phi::DataType::INT16: {
      int16_t i16_value = scalar_value.to<int16_t>();
      return topscl::Scalar(i16_value);
    }
    case phi::DataType::INT32: {
      int32_t i32_value = scalar_value.to<int32_t>();
      return topscl::Scalar(i32_value);
    }
    case phi::DataType::INT64: {
      int64_t i64_value = scalar_value.to<int64_t>();
      return topscl::Scalar(i64_value);
    }
    case phi::DataType::FLOAT16: {
      float f16_value = static_cast<float>(scalar_value.to<phi::float16>());
      return topscl::Scalar(f16_value);
    }
    case phi::DataType::BFLOAT16: {
      float bf16_value = static_cast<float>(scalar_value.to<phi::bfloat16>());
      return topscl::Scalar(bf16_value);
    }
    case phi::DataType::FLOAT32: {
      float f32_value = scalar_value.to<float>();
      return topscl::Scalar(f32_value);
    }
    case phi::DataType::FLOAT64: {
      double f64_value = scalar_value.to<double>();
      return topscl::Scalar(f64_value);
    }
    default: {
      PADDLE_THROW(phi::errors::Unimplemented(
          "ScalarToTopsclScalar, unsupported data type %s",
          phi::DataTypeToString(scalar_type).c_str()));
      break;
    }
  }
  return topscl::Scalar();
}

topscl::Scalar OptionalScalarToTopsclScalar(
    const paddle::optional<phi::Scalar> &opt_scalar) {
  if (opt_scalar) {
    return ScalarToTopsclScalar(opt_scalar.get());
  } else {
    return topscl::Scalar();
  }
}

std::vector<int64_t> IntArrayToVector64(const phi::IntArray &array) {
  auto data = array.GetData();
  std::vector<int64_t> vec(data.begin(), data.end());
  return vec;
}

}  // namespace custom_kernel
