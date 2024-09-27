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

#pragma once
#include "kernels/funcs/op_utils.h"
#include "topscl/topscl.h"

namespace custom_kernel {

topscl::Device GetCurrentDevice();

topscl::Tensor CreateTopsclTensor(const phi::DenseTensor &tensor,
                                  bool pinned = false);

topscl::Tensor OptionalTensorToTopsclTensor(
    const paddle::optional<phi::DenseTensor> &opt_tensor);

topscl::DType DataTypeToTopsclDataType(const phi::DataType &dtype);

topscl::MemoryFormat DataLayoutToTopsclMemoryFormat(
    const common::DataLayout &layout);

topscl::Scalar ScalarToTopsclScalar(const phi::Scalar &scalar_value);

topscl::Scalar OptionalScalarToTopsclScalar(
    const paddle::optional<phi::Scalar> &opt_scalar);

std::vector<int64_t> IntArrayToVector64(const phi::IntArray &array);
}  // namespace custom_kernel
