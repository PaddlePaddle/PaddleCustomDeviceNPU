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

#include <vector>

#include "kernels/funcs/gcu_kernel_funcs.h"
#include "paddle/extension.h"

namespace custom_op_common {
paddle::Tensor CreateTensorFromDenseTensor(const phi::DenseTensor &x);

paddle::optional<paddle::Tensor> CreateOptionalTensorFromOptionalDense(
    const paddle::optional<phi::DenseTensor> &x);

phi::DenseTensor CreateDenseTensorFromTernsor(const paddle::Tensor &x);

std::vector<paddle::Tensor> FusedRotaryEmbedding(
    const paddle::Tensor &query,
    const paddle::Tensor &key,
    const paddle::Tensor &cos_sin_table,
    const paddle::Tensor &positions,
    bool is_neox);

std::vector<paddle::Tensor> FusedSdpFlashAttention(
    const paddle::Tensor &query,
    const paddle::Tensor &key,
    const paddle::Tensor &value,
    const paddle::optional<paddle::Tensor> &attn_mask,
    float dropout = 0.0,
    bool casual = false,
    bool is_test = true);

std::vector<paddle::Tensor> RmsNorm(const paddle::Tensor &x,
                                    const paddle::Tensor &weight,
                                    float epsilon);

std::vector<paddle::Tensor> FusedAddRmsNorm(const paddle::Tensor &x,
                                            const paddle::Tensor &residual,
                                            const paddle::Tensor &weight,
                                            float epsilon);

// *****************************************************************************
//
//                 transformer common
//
// *****************************************************************************
std::vector<paddle::Tensor> ComputeQKV(const paddle::Tensor &norm_weight,
                                       const paddle::Tensor &hidden_input,
                                       const paddle::Tensor &residual,
                                       const paddle::Tensor &qkv_weight,
                                       const paddle::Tensor &cache_kvs,
                                       float epsilon);

std::vector<paddle::Tensor> UpdateKvCache(paddle::Tensor &key,    // NOLINT
                                          paddle::Tensor &value,  // NOLINT
                                          const paddle::Tensor &cache_kvs,
                                          bool is_decoder);

paddle::Tensor SelfAttention(const paddle::Tensor &query,
                             paddle::Tensor &key,    // NOLINT
                             paddle::Tensor &value,  // NOLINT
                             const paddle::Tensor &position_ids,
                             const paddle::Tensor &cos_sin_table,
                             const paddle::Tensor &attention_mask,
                             const paddle::Tensor &attn_out_linear_weight,
                             const paddle::Tensor &cache_kvs,
                             bool is_decoder = true);

std::vector<paddle::Tensor> FeedForward(
    const paddle::Tensor &attn_out,
    const paddle::Tensor &attn_residual,
    const paddle::Tensor &ffn_norm_weight,
    const paddle::Tensor &mlp_gate_up_weight,
    const paddle::Tensor &mlp_down_weight,
    float epsilon);

std::vector<paddle::Tensor> FusedTransformerLayer(
    const paddle::Tensor &norm_weight,
    const paddle::Tensor &hidden_input,
    const paddle::Tensor &residual,
    const paddle::Tensor &position_ids,
    const paddle::Tensor &qkv_weight,
    const paddle::Tensor &cache_kvs,
    const paddle::Tensor &cos_sin_table,
    const paddle::Tensor &attention_mask,
    const paddle::Tensor &attn_out_linear_weight,
    const paddle::Tensor &ffn_norm_weight,
    const paddle::Tensor &mlp_gate_up_weight,
    const paddle::Tensor &mlp_down_weight,
    float epsilon,
    bool is_decoder = true);

std::vector<paddle::Tensor> TopPSampling(const paddle::Tensor &probs,
                                         const paddle::Tensor &top_p);

}  // namespace custom_op_common
