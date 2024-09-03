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
#ifdef PADDLE_WITH_ATB

#include "atb/atb_infer.h"

namespace atb_layers {

struct FusedBlhaLayerParam {
  float epsilon;
  bool rope_neox{false};
  bool use_alibi{false};
  bool trans_qkv_weight;
  bool trans_out_weight;
  bool trans_ffn1_weight;
  bool trans_ffn2_weight;
  bool has_qkv_bias{false};
  atb::infer::ActivationType ffn_act;
  float scale;  // for swish
  int64_t head_num;
  int64_t kv_head_num;
  int64_t head_dim;

  bool is_prefill;
  bool use_matmul_int8{false};
  float qkv_quant_scale{1.0f};
  float out_quant_scale{1.0f};
  float ffn1_quant_scale{1.0f};
  float ffn2_quant_scale{1.0f};
  bool use_smooth_quant{false};
  bool cache_kv_int8{false};

  int64_t rank;
  int64_t nranks;
  int64_t root;
  void* comm;
};

void CreateFusedBlhaLayer(const FusedBlhaLayerParam& param,
                          atb::Operation** operation);

}  // namespace atb_layers

namespace atb {
template <>
inline Status CreateOperation(const atb_layers::FusedBlhaLayerParam& opParam,
                              Operation** operation) {
  atb_layers::CreateFusedBlhaLayer(opParam, operation);
  return ErrorType::NO_ERROR;
}
}  // namespace atb

#endif
