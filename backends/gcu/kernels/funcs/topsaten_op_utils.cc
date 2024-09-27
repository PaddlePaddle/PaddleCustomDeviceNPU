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

#include "kernels/funcs/topsaten_op_utils.h"

#include "paddle/phi/common/data_type.h"
#include "paddle/phi/core/dense_tensor.h"

namespace custom_kernel {

topsatenTensor CreateTopsatenTensor(const phi::DenseTensor &tensor,
                                    bool pinned) {
  if (UNLIKELY(!tensor.initialized())) {
    VLOG(6) << "Create default topsatenTensor.";
    return topsatenTensor();
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
  auto xdims = topsatenSize_t{tensor_dims.data(), rank};
  auto xstrides = topsatenSize_t{strides.data(), rank};
  auto xdtype = DataTypeToTopsatenDataType(tensor.dtype());
  VLOG(6) << "topsopCreateTensor, input tensor:" << TensorToString(tensor);

  auto xt = topsatenTensor(xdims, xstrides, xdtype, GcuDataPtr(tensor));
  VLOG(6) << "Create tensor handle successfully with\n"
          << "dims: " << xt.GetTensorShape()
          << "\nstrides: " << xt.GetTensorStrides()
          << "\ndtype: " << xt.GetTensorDataType();
  return xt;
}

topsatenTensor OptionalTensorToTopsatenTensor(
    const paddle::optional<phi::DenseTensor> &opt_tensor) {
  if (opt_tensor) {
    return CreateTopsatenTensor(opt_tensor.get());
  } else {
    return {};
  }
}

topsatenTensor CreateTopsatenTensorWithoutInitialized(
    const phi::DenseTensor &tensor) {
  PADDLE_ENFORCE_EQ(
      tensor.initialized(),
      false,
      phi::errors::InvalidArgument(
          "Tensor is initialized, please use CreateTopsatenTensor api."));

  std::vector<int64_t> tensor_dims = phi::vectorize(tensor.dims());
  std::vector<int64_t> strides = phi::vectorize(tensor.strides());
  int64_t rank = tensor_dims.size();
  if (rank == 0) {
    rank = 1;
    tensor_dims = {1};
    strides = {1};
  }
  auto xdims = topsatenSize_t{tensor_dims.data(), rank};
  auto xstrides = topsatenSize_t{strides.data(), rank};
  auto xdtype = DataTypeToTopsatenDataType(tensor.dtype());
  VLOG(6) << "CreateTopsatenTensorWithoutInitialized, input tensor:"
          << TensorToString(tensor);

  auto xt = topsatenTensor(xdims, xstrides, xdtype, nullptr);
  VLOG(6) << "Create tensor handle successfully with\n"
          << "dims: " << xt.GetTensorShape()
          << "\nstrides: " << xt.GetTensorStrides()
          << "\ndtype: " << xt.GetTensorDataType();
  return xt;
}

topsatenDataType_t DataTypeToTopsatenDataType(const phi::DataType &dtype) {
  switch (dtype) {
    case phi::DataType::BOOL:
      return TOPSATEN_DATA_PRED;
    case phi::DataType::UINT8:
      return TOPSATEN_DATA_U8;
    case phi::DataType::INT8:
      return TOPSATEN_DATA_I8;
    case phi::DataType::INT16:
      return TOPSATEN_DATA_I16;
    case phi::DataType::INT32:
      return TOPSATEN_DATA_I32;
    case phi::DataType::INT64:
      return TOPSATEN_DATA_I64;
    case phi::DataType::FLOAT16:
      return TOPSATEN_DATA_FP16;
    case phi::DataType::BFLOAT16:
      return TOPSATEN_DATA_BF16;
    case phi::DataType::FLOAT32:
      return TOPSATEN_DATA_FP32;
    case phi::DataType::FLOAT64:
      return TOPSATEN_DATA_F64;
    default: {
      PADDLE_THROW(phi::errors::Unimplemented(
          "Unsupported data type %s", phi::DataTypeToString(dtype).c_str()));
      return TOPSATEN_DATA_FP32;
    }
  }
}

topsatenScalar_t ScalarToTopsatenScalar(const phi::Scalar &scalar_value) {
  topsatenScalar_t xvalue;
  auto scalar_type = scalar_value.dtype();
  switch (scalar_type) {
    case phi::DataType::BOOL:
      xvalue.dtype = TOPSATEN_DATA_PRED;
      xvalue.ival = scalar_value.to<bool>();
      break;
    case phi::DataType::UINT8:
      xvalue.dtype = TOPSATEN_DATA_U8;
      xvalue.ival = scalar_value.to<uint8_t>();
      break;
    case phi::DataType::INT8:
      xvalue.dtype = TOPSATEN_DATA_I8;
      xvalue.ival = scalar_value.to<int8_t>();
      break;
    case phi::DataType::INT16:
      xvalue.dtype = TOPSATEN_DATA_I16;
      xvalue.ival = scalar_value.to<int16_t>();
      break;
    case phi::DataType::INT32:
      xvalue.dtype = TOPSATEN_DATA_I32;
      xvalue.ival = scalar_value.to<int>();
      break;
    case phi::DataType::INT64:
      xvalue.dtype = TOPSATEN_DATA_I64;
      xvalue.ival = scalar_value.to<int64_t>();
      break;
    case phi::DataType::FLOAT16:
      xvalue.dtype = TOPSATEN_DATA_FP16;
      xvalue.fval = scalar_value.to<phi::float16>();
      break;
    case phi::DataType::BFLOAT16:
      xvalue.dtype = TOPSATEN_DATA_BF16;
      xvalue.fval = scalar_value.to<phi::bfloat16>();
      break;
    case phi::DataType::FLOAT32:
      xvalue.dtype = TOPSATEN_DATA_FP32;
      xvalue.fval = scalar_value.to<float>();
      break;
    case phi::DataType::FLOAT64:
      xvalue.dtype = TOPSATEN_DATA_F64;
      xvalue.fval = scalar_value.to<double>();
      break;
    default: {
      PADDLE_THROW(phi::errors::Unimplemented(
          "ScalarToTopsatenScalar, unsupported data type %s",
          phi::DataTypeToString(scalar_type).c_str()));
      break;
    }
  }
  return xvalue;
}

topsatenScalar_t OptionalScalarToTopsatenScalar(
    const paddle::optional<phi::Scalar> &opt_scalar) {
  if (opt_scalar) {
    return ScalarToTopsatenScalar(opt_scalar.get());
  } else {
    return {TOPSATEN_DATA_NONE, {.ival = 0}};
  }
}

topsatenSize_t IntArrayToTopsatenSize(const phi::IntArray &int_array) {
  auto int_datas = int_array.GetData();
  return {int_datas.data(), static_cast<int64_t>(int_datas.size())};
}

topsatenSize_t IntArrayToTopsatenSize(const std::vector<int64_t> &int_array) {
  return {int_array.data(), static_cast<int64_t>(int_array.size())};
}

topsatenSize_t OptionalIntArrayToTopsatenSize(
    const paddle::optional<phi::IntArray> &opt_array) {
  if (opt_array) {
    return IntArrayToTopsatenSize(opt_array.get());
  } else {
    return {};
  }
}

}  // namespace custom_kernel
