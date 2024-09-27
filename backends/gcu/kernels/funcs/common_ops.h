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

#include "common/gcu_funcs.h"
#include "kernels/funcs/topsaten_op_launch.h"
#include "kernels/funcs/topscl_op_launch.h"

namespace custom_kernel {

phi::DenseTensor MaybeCreateOrTrans(
    const phi::CustomContext& dev_ctx,
    const phi::DenseTensor& src,
    const std::unordered_map<phi::DataType, phi::DataType>& tans_map,
    bool need_cast = true);

phi::DenseTensor MaybeCreateOrTrans64To32bits(const phi::CustomContext& dev_ctx,
                                              const phi::DenseTensor& src,
                                              bool need_cast = true);

phi::DenseTensor MaybeCreateOrTransFp16ToFp32(const phi::CustomContext& dev_ctx,
                                              const phi::DenseTensor& src,
                                              bool need_cast = true);

void MaybeTransResult(const phi::CustomContext& dev_ctx,
                      const phi::DenseTensor& result,
                      phi::DenseTensor* dst);

void Broadcast(const phi::CustomContext& dev_ctx,
               const phi::DenseTensor& src,
               phi::DenseTensor* dst);

phi::DenseTensor Broadcast(const phi::CustomContext& dev_ctx,
                           const phi::DenseTensor& src,
                           const std::vector<int64_t>& output_shapes);

void Cast(const phi::CustomContext& dev_ctx,
          const phi::DenseTensor& x,
          const phi::DataType& dtype,
          phi::DenseTensor* out);

phi::DenseTensor Cast(const phi::CustomContext& dev_ctx,
                      const phi::DenseTensor& x,
                      const phi::DataType& dtype);

phi::DenseTensor CastOrCopyToPinnedMemory(const phi::CustomContext& dev_ctx,
                                          const phi::DenseTensor& x,
                                          const phi::DataType& dtype);

phi::DenseTensor ReshapeWithoutCopy(const phi::DenseTensor& src,
                                    const std::vector<int64_t>& out_shapes);

phi::DenseTensor TensorEmpty(const phi::CustomContext& dev_ctx,
                             const phi::DenseTensorMeta& meta);

phi::DenseTensor TensorOnes(const phi::CustomContext& dev_ctx,
                            const phi::DenseTensorMeta& meta);

phi::DenseTensor TensorZeros(const phi::CustomContext& dev_ctx,
                             const phi::DenseTensorMeta& meta);

// meta reuse ops
phi::DenseTensor Add(const phi::CustomContext& dev_ctx,
                     const phi::DenseTensor& x,
                     const phi::DenseTensor& y,
                     const phi::DenseTensorMeta& out_meta);

phi::DenseTensor Add(const phi::CustomContext& dev_ctx,
                     const phi::DenseTensor& x,
                     const phi::DenseTensor& y);

phi::DenseTensor Subtract(const phi::CustomContext& dev_ctx,
                          const phi::DenseTensor& x,
                          const phi::DenseTensor& y,
                          const phi::DenseTensorMeta& out_meta);

phi::DenseTensor Subtract(const phi::CustomContext& dev_ctx,
                          const phi::DenseTensor& x,
                          const phi::DenseTensor& y);

void SliceBase(const phi::CustomContext& dev_ctx,
               const phi::DenseTensor& x,
               const std::vector<int64_t>& axes,
               const std::vector<int64_t>& starts,
               phi::DenseTensor* out);

}  // namespace custom_kernel
