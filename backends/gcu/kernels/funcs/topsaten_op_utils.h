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
#include "topsaten/topsaten_define.h"

namespace custom_kernel {

topsatenTensor CreateTopsatenTensor(const phi::DenseTensor &tensor,
                                    bool pinned = false);

topsatenTensor OptionalTensorToTopsatenTensor(
    const paddle::optional<phi::DenseTensor> &opt_tensor);

topsatenTensor CreateTopsatenTensorWithoutInitialized(
    const phi::DenseTensor &tensor);

topsatenDataType_t DataTypeToTopsatenDataType(const phi::DataType &dtype);

topsatenScalar_t ScalarToTopsatenScalar(const phi::Scalar &scalar_value);

topsatenScalar_t OptionalScalarToTopsatenScalar(
    const paddle::optional<phi::Scalar> &opt_scalar);

topsatenSize_t IntArrayToTopsatenSize(const phi::IntArray &int_array);

topsatenSize_t IntArrayToTopsatenSize(const std::vector<int64_t> &int_array);

topsatenSize_t OptionalIntArrayToTopsatenSize(
    const paddle::optional<phi::IntArray> &opt_array);

}  // namespace custom_kernel
