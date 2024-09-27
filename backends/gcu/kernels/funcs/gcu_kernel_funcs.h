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

#include "common/gcu_funcs.h"
#include "common/utils.h"
#include "kernels/funcs/common_ops.h"
#include "kernels/funcs/gcu_layout_funcs.h"
#include "paddle/phi/common/data_type.h"

#define THROW_AOT_UNIMPLEMENTED()                                            \
  PADDLE_THROW(phi::errors::Unimplemented("AOT kernel is unimplemented: %s", \
                                          __FUNCTION__))

#define THROW_JIT_UNIMPLEMENTED()                                            \
  PADDLE_THROW(phi::errors::Unimplemented("JIT kernel is unimplemented: %s", \
                                          __FUNCTION__))

namespace custom_kernel {
using DenseTensor = phi::DenseTensor;
using TensorNameMap = std::map<std::string, std::vector<std::string>>;
using TensorValueMap = std::map<std::string, std::vector<DenseTensor*>>;

std::string GradVarName(const std::string& var_name);
std::vector<int> GetIntList(const phi::IntArray& int_array);

}  // namespace custom_kernel
