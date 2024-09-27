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

#include "common/utils.h"

#include <gcu/driver/device_manager.h>
#include <gcu/umd/device_ids.h>

#include <string>
#include <vector>

namespace custom_kernel {
namespace {
const int64_t kMicrosToMillis = 1000;
enum class ChipType {
  LEO,
  PAVO,
  PAVO_1C,
  DORADO,
  DORADO_2C,
  DORADO_3PG,
  LIBRA,
  SCORPIO,
  UNKNOW,
};

ChipType ParseChipType() {
  ChipType type = ChipType::UNKNOW;
  if (dtu::driver::DeviceManager::instance()->IsDorado()) {
    type = ChipType::DORADO;
    if (dtu::driver::DeviceManager::instance()->device_info().clusters_num ==
        2) {
      type = ChipType::DORADO_2C;
    } else {
      VLOG(1) << "[WARN] Paddle now only suport dorado_2c in dorado platform!";
    }
  } else if (dtu::driver::DeviceManager::instance()->IsScorpio()) {
    type = ChipType::SCORPIO;
  } else if (dtu::driver::DeviceManager::instance()->IsPavo()) {
    type = ChipType::PAVO;
  }
  PADDLE_ENFORCE_NE(
      type,
      ChipType::UNKNOW,
      phi::errors::Unavailable("unknown chip type is not support!"));
  return type;
}

static std::string GetChipTypeStr(ChipType type) {
  switch (type) {
    case ChipType::LEO:
      return "leo";
    case ChipType::PAVO:
      return "pavo";
    case ChipType::PAVO_1C:
      return "pavo_1c";
    case ChipType::DORADO:
      return "dorado";
    case ChipType::DORADO_2C:
      return "dorado_2c";
    case ChipType::DORADO_3PG:
      return "dorado_3pg";
    case ChipType::LIBRA:
      return "libra";
    case ChipType::SCORPIO:
      return "scorpio";
    default:
      return "unknown";
  }
}
}  // namespace

std::string GetTargetName() { return GetChipTypeStr(ParseChipType()); }

int64_t GetCurrentTimestap() {
  struct timeval tv;
  int ret = gettimeofday(&tv, nullptr);
  if (ret != 0) {
    VLOG(6) << "Func gettimeofday may failed, ret:" << ret;
    return 0;
  }
  int64_t totalUsec = tv.tv_usec + tv.tv_sec * 1000000;
  return totalUsec;
}

double GetTimeCostInMs(int64_t start_time, int64_t end_time) {
  return ((static_cast<double>(end_time) - static_cast<double>(start_time)) /
          static_cast<double>(kMicrosToMillis));
}

std::vector<int32_t> LayoutToVector(phi::DataLayout layout) {
  if (layout == phi::DataLayout::NHWC) {
    return {0, 2, 3, 1};
  } else if (layout == phi::DataLayout::NCHW) {
    return {0, 1, 2, 3};
  } else {
    phi::errors::Fatal("unsupport layout %s",
                       common::DataLayoutToString(layout).c_str());
    return {};
  }
}

phi::DataLayout VectorToLayout(const std::vector<int32_t>& layout) {
  static std::vector<int32_t> nhwc = {0, 2, 3, 1};
  static std::vector<int32_t> nchw = {0, 1, 2, 3};
  if (layout == nhwc) {
    return phi::DataLayout::NHWC;
  } else if (layout == nchw) {
    return phi::DataLayout::NCHW;
  } else {
    phi::errors::Fatal("unsupport layout %s",
                       custom_kernel::VectorToStr<int32_t>(layout).c_str());
    return phi::DataLayout::UNDEFINED;
  }
}

bool LaunchAOTKernel(const phi::DataType& dtype,
                     const std::unordered_set<phi::DataType>& supported_types) {
  static const char* use_jit_env = std::getenv(env::kUseJitKernels);
  static bool use_jit_only =
      (use_jit_env != nullptr && std::string(use_jit_env) == "true");
  // just for log
  static bool launch_jit_only = ((VLOG(0) << "Kernels launch in JIT ONLY mode:"
                                          << (use_jit_only ? "true" : "false")),
                                 (use_jit_only));
  if (launch_jit_only) {
    return false;
  }
  if (supported_types.empty()) {
    return true;
  }
  return (supported_types.count(dtype) > 0);
}

#define FOR_EACH_DATA_TYPE_TO_PRINT(_)      \
  _(bool, phi::DataType::BOOL)              \
  _(int8_t, phi::DataType::INT8)            \
  _(uint8_t, phi::DataType::UINT8)          \
  _(int16_t, phi::DataType::INT16)          \
  _(uint16_t, phi::DataType::UINT16)        \
  _(int32_t, phi::DataType::INT32)          \
  _(uint32_t, phi::DataType::UINT32)        \
  _(int64_t, phi::DataType::INT64)          \
  _(uint64_t, phi::DataType::UINT64)        \
  _(phi::bfloat16, phi::DataType::BFLOAT16) \
  _(phi::float16, phi::DataType::FLOAT16)   \
  _(float, phi::DataType::FLOAT32)          \
  _(double, phi::DataType::FLOAT64)

#define CALL_PRINT_TENSOR(cpp_type, data_type) \
  case data_type:                              \
    PrintTensor<cpp_type>(os, t);              \
    break;

std::ostream& operator<<(std::ostream& os, const phi::DenseTensor& t) {
  switch (t.dtype()) {
    FOR_EACH_DATA_TYPE_TO_PRINT(CALL_PRINT_TENSOR)
    default:
      VLOG(1) << "PrintTensor unrecognized data type:" << t.dtype();
  }
  return os;
}
#undef FOR_EACH_DATA_TYPE_TO_PRINT
#undef CALL_PRINT_TENSOR

}  // namespace custom_kernel
