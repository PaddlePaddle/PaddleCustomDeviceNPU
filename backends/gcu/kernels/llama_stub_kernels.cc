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

#include "kernels/funcs/gcu_kernel_funcs.h"

namespace custom_kernel {

template <typename T, typename Context>
void RmsNormKernel(const Context& dev_ctx,
                   const phi::DenseTensor& x,
                   const paddle::optional<phi::DenseTensor>& bias,
                   const paddle::optional<phi::DenseTensor>& residual,
                   const phi::DenseTensor& norm_weight,
                   const paddle::optional<phi::DenseTensor>& norm_bias,
                   const float epsilon,
                   const int begin_norm_axis,
                   const float quant_scale,
                   const int quant_round_type,
                   const float quant_max_bound,
                   const float quant_min_bound,
                   phi::DenseTensor* out,
                   phi::DenseTensor* residual_out) {
  VLOG(0) << "====== GCU kernel stub: rms_norm =====";
  dev_ctx.template Alloc<T>(out);
  dev_ctx.template Alloc<T>(residual_out);
}

template <typename T, typename Context>
void MultiHeadAttentionVariableForwardKernel(
    const Context& dev_ctx,
    const phi::DenseTensor& query,
    const phi::DenseTensor& key,
    const phi::DenseTensor& value,
    const phi::DenseTensor& seq_lens,
    const phi::DenseTensor& kv_seq_lens,
    const paddle::optional<phi::DenseTensor>& mask,
    const float scale,
    const bool causal,
    const int pre_cache_length,
    phi::DenseTensor* out) {
  VLOG(0) << "====== GCU kernel stub: "
             "variable_length_memory_efficient_attention =====";
  dev_ctx.template Alloc<T>(out);
}

template <typename T, typename Context>
void FusedBiasActKernel(
    const Context& dev_ctx,
    const phi::DenseTensor& x,
    const paddle::optional<phi::DenseTensor>& bias,
    const paddle::optional<phi::DenseTensor>& dequant_scales,
    const paddle::optional<phi::DenseTensor>& shift,
    const paddle::optional<phi::DenseTensor>& smooth,
    const std::string& act_method,
    const std::string& compute_dtype,
    float quant_scale,
    int quant_round_type,
    float quant_max_bound,
    float quant_min_bound,
    phi::DenseTensor* out) {
  VLOG(0) << "====== GCU kernel stub: fused_bias_act =====";
  dev_ctx.template Alloc<T>(out);
}

template <typename T, typename Context>
void FusedLayerNormKernel(const Context& dev_ctx,
                          const phi::DenseTensor& x,
                          const paddle::optional<phi::DenseTensor>& bias,
                          const paddle::optional<phi::DenseTensor>& residual,
                          const paddle::optional<phi::DenseTensor>& norm_weight,
                          const paddle::optional<phi::DenseTensor>& norm_bias,
                          const float epsilon,
                          const float residual_alpha,
                          const int begin_norm_axis,
                          const float quant_scale,
                          const int quant_round_type,
                          const float quant_max_bound,
                          const float quant_min_bound,
                          phi::DenseTensor* out,
                          phi::DenseTensor* residual_out,
                          phi::DenseTensor* mean,
                          phi::DenseTensor* variance) {
  VLOG(0) << "====== GCU kernel stub: fused_bias_residual_layernorm =====";
  dev_ctx.template Alloc<T>(out);
  dev_ctx.template Alloc<T>(residual_out);
  dev_ctx.template Alloc<T>(mean);
  dev_ctx.template Alloc<T>(variance);
}

template <typename T, typename Context>
void MMHAKernel(const Context& dev_ctx,
                const phi::DenseTensor& x,
                const phi::DenseTensor& cache_kv,
                const paddle::optional<phi::DenseTensor>& bias,
                const paddle::optional<phi::DenseTensor>& src_mask,
                const paddle::optional<phi::DenseTensor>& cum_offsets,
                const paddle::optional<phi::DenseTensor>& sequence_lengths,
                const paddle::optional<phi::DenseTensor>& rotary_tensor,
                const paddle::optional<phi::DenseTensor>& beam_cache_offset,
                const paddle::optional<phi::DenseTensor>& qkv_out_scale,
                const paddle::optional<phi::DenseTensor>& out_shift,
                const paddle::optional<phi::DenseTensor>& out_smooth,
                int seq_len,
                int rotary_emb_dims,
                const bool use_neox_rotary_style,
                const std::string& compute_dtype,
                const float out_scale,
                const int quant_round_type,
                const float quant_max_bound,
                const float quant_min_bound,
                phi::DenseTensor* out,
                phi::DenseTensor* cache_kv_out,
                phi::DenseTensor* beam_cache_offset_out) {
  VLOG(0) << "====== GCU kernel stub: masked_multihead_attention =====";
  dev_ctx.template Alloc<T>(out);
  dev_ctx.template Alloc<T>(cache_kv_out);
  dev_ctx.template Alloc<T>(beam_cache_offset_out);
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(rms_norm,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::RmsNormKernel,
                          float,
                          double,
                          int64_t,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}

PD_REGISTER_PLUGIN_KERNEL(
    variable_length_memory_efficient_attention,
    gcu,
    ALL_LAYOUT,
    custom_kernel::MultiHeadAttentionVariableForwardKernel,
    float,
    double,
    int64_t,
    phi::dtype::float16,
    phi::dtype::bfloat16) {
  kernel->InputAt(3).SetDataType(phi::DataType::INT32);
}

PD_REGISTER_PLUGIN_KERNEL(fused_bias_act,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::FusedBiasActKernel,
                          float,
                          double,
                          int64_t,
                          int32_t,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}

PD_REGISTER_PLUGIN_KERNEL(fused_bias_residual_layernorm,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::FusedLayerNormKernel,
                          float,
                          double,
                          int64_t,
                          int32_t,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}

PD_REGISTER_PLUGIN_KERNEL(masked_multihead_attention,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::MMHAKernel,
                          float,
                          phi::dtype::float16,
                          phi::dtype::bfloat16,
                          int32_t) {}
