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

namespace custom_kernel {
namespace layout_trans {
const std::vector<int64_t> kNCHW_to_NHWC = {0, 2, 3, 1};
const std::vector<int64_t> kNHWC_to_NCHW = {0, 3, 1, 2};
}  // namespace layout_trans

bool EnableTransposeOptimize();

void SetLayout(phi::DenseTensor& tensor,  // NOLINT
               const common::DataLayout& layout);

void Transpose(const phi::CustomContext& dev_ctx,
               const phi::DenseTensor& x,
               const std::vector<int64_t>& axis,
               phi::DenseTensor* out);

phi::DenseTensor Transpose(const phi::CustomContext& dev_ctx,
                           const phi::DenseTensor& x,
                           const std::vector<int64_t>& axis);

bool DataPdCustomNHWC(const phi::DenseTensor& tensor);

bool DataPdCustomNHWC(const std::vector<phi::DenseTensor>& tensors);

// ////////////////  Permuted funcs ////////////////
void PermutedShapeWithcontiguousStrides(
    phi::DenseTensor& tensor,  // NOLINT
    const std::vector<int64_t>& permutation,
    const common::DataLayout& layout = common::DataLayout::kNCHW);

void RecoverPdCustomNHWCMeta(phi::DenseTensor& tensor);  // NOLINT

void PermutedStridesWithoutShape(
    phi::DenseTensor& tensor,  // NOLINT
    const std::vector<int64_t>& shape_perm,
    const std::vector<int64_t>& strides_perm,
    const common::DataLayout& layout = common::DataLayout::kNCHW);

void PermutedShapeAndStrides(
    phi::DenseTensor& tensor,  // NOLINT
    const std::vector<int64_t>& permutation,
    const common::DataLayout& layout = common::DataLayout::kNCHW);

// ////////////////  Transpose funcs ////////////////
phi::DenseTensor NCHWTransToPdOriginNHWC(const phi::CustomContext& dev_ctx,
                                         const phi::DenseTensor& x);

phi::DenseTensor NCHWTransToPdCustomNHWC(const phi::CustomContext& dev_ctx,
                                         const phi::DenseTensor& x);

phi::DenseTensor NCHWTransToAtenNHWC(const phi::CustomContext& dev_ctx,
                                     const phi::DenseTensor& x);

phi::DenseTensor PdCustomNHWCTransToNCHW(const phi::CustomContext& dev_ctx,
                                         const phi::DenseTensor& x);

phi::DenseTensor PdOriginNHWCTransToNCHW(const phi::CustomContext& dev_ctx,
                                         const phi::DenseTensor& x);

// ////////////////  Represent funcs ////////////////
phi::DenseTensor NoNeedTransNCHWRepresentAsOriginNHWC(
    const phi::DenseTensor& x);

void PdCustomNHWCRepresentAsAtenNHWC(phi::DenseTensor& x,  // NOLINT
                                     bool weight_or_output = false);

void AtenNHWCRepresentAsPdCustomNHWC(phi::DenseTensor& x,  // NOLINT
                                     bool raw_output = false);

void OriginNHWCRepresentAsAtenNHWC(phi::DenseTensor& x);  // NOLINT

void AtenNHWCRepresentAsOriginNHWC(phi::DenseTensor& x);  // NOLINT

void PdCustomNHWCRepresentAsOriginNHWC(phi::DenseTensor& x,  // NOLINT
                                       bool raw_output = false);

void OriginNHWCRepresentAsPdCustomNHWC(phi::DenseTensor& x);  // NOLINT

void RepresentPdCustomNHWC(phi::DenseTensor& x);  // NOLINT

}  // namespace custom_kernel
