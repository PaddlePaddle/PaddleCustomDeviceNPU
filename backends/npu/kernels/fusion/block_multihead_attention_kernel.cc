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

#include "kernels/funcs/npu_funcs.h"
#include "kernels/funcs/npu_op_runner.h"

namespace custom_kernel {

template <typename T, typename Context>
void BlockMultiheadAttentionKernel(
    const Context& dev_ctx,
    const phi::DenseTensor& qkv,
    const phi::DenseTensor& key_cache,
    const phi::DenseTensor& value_cache,
    const phi::DenseTensor& seq_lens_encoder,
    const phi::DenseTensor& seq_lens_decoder,
    const phi::DenseTensor& seq_lens_this_time,
    const phi::DenseTensor& padding_offsets,
    const phi::DenseTensor& cum_offsets,
    const phi::DenseTensor& cu_seqlens_q,
    const phi::DenseTensor& cu_seqlens_k,
    const phi::DenseTensor& block_tables,
    const paddle::optional<phi::DenseTensor>& pre_key_cache,
    const paddle::optional<phi::DenseTensor>& pre_value_cache,
    const paddle::optional<phi::DenseTensor>& cache_k_quant_scales,
    const paddle::optional<phi::DenseTensor>& cache_v_quant_scales,
    const paddle::optional<phi::DenseTensor>& cache_k_dequant_scales,
    const paddle::optional<phi::DenseTensor>& cache_v_dequant_scales,
    const paddle::optional<phi::DenseTensor>& qkv_out_scale,
    const paddle::optional<phi::DenseTensor>& qkv_bias,
    const paddle::optional<phi::DenseTensor>& out_shift,
    const paddle::optional<phi::DenseTensor>& out_smooth,
    const paddle::optional<phi::DenseTensor>& max_enc_len_this_time,
    const paddle::optional<phi::DenseTensor>& max_dec_len_this_time,
    const paddle::optional<phi::DenseTensor>& rope_emb,
    const paddle::optional<phi::DenseTensor>& mask,
    const paddle::optional<phi::DenseTensor>& tgt_mask,
    int max_seq_len,
    int block_size,
    bool use_neox_style,
    const bool dynamic_cachekv_quant,
    const int quant_round_type,
    const float quant_max_bound,
    const float quant_min_bound,
    const float out_scale,
    const std::string& compute_dtype,
    const float rope_theta,
    phi::DenseTensor* fmha_out,
    phi::DenseTensor* qkv_out,
    phi::DenseTensor* key_cache_out,
    phi::DenseTensor* value_cache_out) {
  PADDLE_THROW(phi::errors::Unimplemented("Only supports model export"));
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(block_multihead_attention,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::BlockMultiheadAttentionKernel,
                          phi::float16,
                          int32_t) {
  kernel->InputAt(24).SetBackend(phi::Backend::CPU);
  kernel->InputAt(25).SetBackend(phi::Backend::CPU);
}
