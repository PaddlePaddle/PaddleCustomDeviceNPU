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

#pragma once

#include <algorithm>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "common/gcu_env_list.h"
#include "common/gcu_funcs.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/device_context.h"

namespace custom_kernel {

std::string GetTargetName();

int64_t GetCurrentTimestap();

double GetTimeCostInMs(int64_t start_time, int64_t end_time);

std::vector<int32_t> LayoutToVector(phi::DataLayout layout);

phi::DataLayout VectorToLayout(const std::vector<int32_t>& layout);

bool LaunchAOTKernel(
    const phi::DataType& dtype = phi::DataType::FLOAT32,
    const std::unordered_set<phi::DataType>& supported_types = {});

template <typename T = int64_t>
static std::string VectorToStr(std::vector<T> vec) {
  std::stringstream ss;
  auto len = vec.size();
  ss << "[";
  for (size_t i = 0; i < len; ++i) {
    ss << std::fixed << vec[i];
    if (i != len - 1) {
      ss << ", ";
    }
  }
  ss << "]";
  return ss.str();
}

template <typename T = float>
std::ostream& PrintTensor(std::ostream& os, const phi::DenseTensor& tensor) {
  phi::DenseTensor cpu_tensor;
  if (tensor.place().GetType() != phi::AllocationType::CPU) {
    auto dev_ctx = static_cast<const phi::CustomContext*>(
        phi::DeviceContextPool::Instance().Get(tensor.place()));
    TensorCopy(*dev_ctx, tensor, true, &cpu_tensor, phi::CPUPlace());
  } else {
    cpu_tensor = tensor;
  }
  os << "LoDTensor<";
  if (tensor.initialized()) {
    os << phi::DataTypeToString(tensor.dtype()) << ", ";
    os << tensor.place() << ", ";
    os << "Shape(" << tensor.dims() << "), ";
    os << "Strides(" << tensor.strides() << "), ";
    os << "layout:" << tensor.layout() << ", ";
    os << "data: [";

    auto ptr = cpu_tensor.data<T>();
    auto element_num = cpu_tensor.numel();
    // Note: int8_t && uint8_t is typedef of char, ostream unable to print
    // properly
    if (typeid(int8_t) == typeid(T) || typeid(uint8_t) == typeid(T)) {
      if (element_num > 0) {
        os << signed(ptr[0]);
        for (int j = 1; j < element_num; ++j) {
          os << " " << signed(ptr[j]);
        }
      }
    } else {
      if (element_num > 0) {
        os << ptr[0];
        for (int j = 1; j < element_num; ++j) {
          os << " " << ptr[j];
        }
      }
    }
    os << "]";
  } else {
    os << "NOT_INITED";
  }
  os << ">";
  return os;
}

std::ostream& operator<<(std::ostream& os, const phi::DenseTensor& t);

}  // namespace custom_kernel
