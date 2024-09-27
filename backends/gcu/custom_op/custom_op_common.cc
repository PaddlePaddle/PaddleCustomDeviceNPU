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

#include "custom_op/custom_op_common.h"

namespace custom_op_common {
paddle::Tensor CreateTensorFromDenseTensor(const phi::DenseTensor &x) {
  std::shared_ptr<phi::DenseTensor> impl =
      std::make_shared<phi::DenseTensor>(x);
  return paddle::Tensor(impl);
}

paddle::optional<paddle::Tensor> CreateOptionalTensorFromOptionalDense(
    const paddle::optional<phi::DenseTensor> &x) {
  if (x) {
    std::shared_ptr<phi::DenseTensor> impl =
        std::make_shared<phi::DenseTensor>(x.get());
    return paddle::make_optional<paddle::Tensor>(paddle::Tensor(impl));
  } else {
    return paddle::optional<paddle::Tensor>();
  }
}

phi::DenseTensor CreateDenseTensorFromTernsor(const paddle::Tensor &x) {
  auto impl_dense = static_cast<const phi::DenseTensor *>(x.impl().get());
  return *impl_dense;
}

std::vector<paddle::Tensor> FusedRotaryEmbedding(
    const paddle::Tensor &query,
    const paddle::Tensor &key,
    const paddle::Tensor &cos_sin_table,
    const paddle::Tensor &positions,
    bool is_neox) {
  //   PADDLE_GCU_KERNEL_TRACE("common_FusedRotaryEmbedding");
  auto dev_ctx = static_cast<const phi::CustomContext *>(
      paddle::experimental::DeviceContextPool::Instance().Get(query.place()));

  // [batch_size, seq_len, num_heads, head_dim]
  auto query_tensor = static_cast<const phi::DenseTensor *>(query.impl().get());

  // [batch_size, seq_len, num_kv_heads, head_dim]
  auto key_tensor = static_cast<const phi::DenseTensor *>(key.impl().get());

  // [max_position, rotary_dim]
  auto cos_sin_tensor =
      static_cast<const phi::DenseTensor *>(cos_sin_table.impl().get());

  // [batch_size, seq_len]
  auto positions_tensor =
      static_cast<const phi::DenseTensor *>(positions.impl().get());

  auto query_dims = query_tensor->dims();
  auto key_dims = key_tensor->dims();
  PADDLE_ENFORCE_EQ(
      query_dims.size(),
      4,
      phi::errors::InvalidArgument("The rank of query must be 4, but get %zu.",
                                   query_dims.size()));
  PADDLE_ENFORCE_EQ(
      key_dims.size(),
      4,
      phi::errors::InvalidArgument("The rank of key must be 4, but get %zu.",
                                   key_dims.size()));
  PADDLE_ENFORCE_EQ(cos_sin_tensor->dims().size(),
                    2,
                    phi::errors::InvalidArgument(
                        "The rank of cos_sin_tensor must be 2, but get %zu.",
                        cos_sin_tensor->dims().size()));
  PADDLE_ENFORCE_EQ(positions_tensor->dims().size(),
                    2,
                    phi::errors::InvalidArgument(
                        "The rank of positions must be 2, but get %zu.",
                        positions_tensor->dims().size()));

  auto head_dim = query_dims.at(3);
  auto num_tokens = query_dims.at(0) * query_dims.at(1);

  // [num_tokens, num_heads * head_dim]
  auto query_reshape = custom_kernel::ReshapeWithoutCopy(
      *query_tensor, {num_tokens, query_dims.at(2) * query_dims.at(3)});
  // [num_tokens, num_kv_heads * head_dim]
  auto key_reshape = custom_kernel::ReshapeWithoutCopy(
      *key_tensor, {num_tokens, key_dims.at(2) * key_dims.at(3)});
  // [num_tokens]
  auto positions_reshape = custom_kernel::ReshapeWithoutCopy(
      *positions_tensor, {positions_tensor->numel()});

  LAUNCH_TOPSATENOP(topsvllmRotaryEmbedding,
                    (*dev_ctx),
                    query_reshape,
                    key_reshape,
                    positions_reshape,
                    *cos_sin_tensor,
                    head_dim,
                    is_neox);

  return {query, key};
}

std::vector<paddle::Tensor> FusedSdpFlashAttention(
    const paddle::Tensor &query,
    const paddle::Tensor &key,
    const paddle::Tensor &value,
    const paddle::optional<paddle::Tensor> &attn_mask,
    float dropout,
    bool casual,
    bool is_test) {
  //   PADDLE_GCU_KERNEL_TRACE("common_FusedSdpFlashAttention");
  auto dev_ctx = static_cast<const phi::CustomContext *>(
      paddle::experimental::DeviceContextPool::Instance().Get(query.place()));

  PADDLE_ENFORCE_EQ(is_test,
                    true,
                    phi::errors::InvalidArgument("Only support inference now"));
  // TODO(wangzhengjun): Relies on the user to provide attn_mask temporarily.
  PADDLE_ENFORCE_EQ(
      !attn_mask,
      false,
      phi::errors::InvalidArgument("Should provide the attn_mask."));

  // [batch_size, seq_len, num_heads, head_dim]
  auto query_tensor = static_cast<const phi::DenseTensor *>(query.impl().get());
  // [batch_size, seq_len, num_kv_heads, head_dim]
  auto key_tensor = static_cast<const phi::DenseTensor *>(key.impl().get());
  // [batch_size, seq_len, num_kv_heads, head_dim]
  auto value_tensor = static_cast<const phi::DenseTensor *>(value.impl().get());
  // [batch_size, 1, target_seq_len, seq_len]
  auto attn_mask_tensor =
      static_cast<const phi::DenseTensor *>(attn_mask.get().impl().get());

  auto query_dims = query_tensor->dims();
  const double scale = 1.0f / std::sqrt(query_dims.at(3));

  // attention_out
  std::shared_ptr<phi::DenseTensor> attention_out =
      std::make_shared<phi::DenseTensor>();
  attention_out->Resize(query_dims);
  dev_ctx->Alloc(attention_out.get(), query_tensor->dtype());

  auto dropout_scalar = phi::Scalar(dropout);
  auto scale_scalar = phi::Scalar(scale);
  LAUNCH_TOPSATENOP(topsvllmMemoryEfficientAttention,
                    (*dev_ctx),
                    *attention_out,
                    *query_tensor,
                    *key_tensor,
                    *value_tensor,
                    *attn_mask_tensor,
                    dropout_scalar,
                    scale_scalar);

  return {paddle::Tensor(attention_out)};
}

std::vector<paddle::Tensor> RmsNorm(const paddle::Tensor &x,
                                    const paddle::Tensor &weight,
                                    float epsilon) {
  //   PADDLE_GCU_KERNEL_TRACE("common_RmsNorm");
  auto dev_ctx = static_cast<const phi::CustomContext *>(
      paddle::experimental::DeviceContextPool::Instance().Get(x.place()));

  auto x_tensor = static_cast<const phi::DenseTensor *>(x.impl().get());
  auto weight_tensor =
      static_cast<const phi::DenseTensor *>(weight.impl().get());

  auto x_input = *x_tensor;
  // TODO(wangzhengjun): topsvllmRmsNorm input.rank only suppport 2
  auto x_dims = x_tensor->dims();
  auto x_rank = x_dims.size();
  if (x_rank > 2) {
    int64_t pre_dims = 1;
    for (int i = x_rank - 2; i >= 0; --i) {
      pre_dims *= x_dims.at(i);
    }
    auto new_dims = common::make_ddim({pre_dims, x_dims.at(x_rank - 1)});
    x_input.Resize(new_dims);
    VLOG(3) << "Custom op RmsNorm x_rank > 2, origin x_dims:[" << x_dims
            << "], new_dims:[" << new_dims << "].";
  }

  std::shared_ptr<phi::DenseTensor> out_tensor =
      std::make_shared<phi::DenseTensor>();
  out_tensor->Resize(x_input.dims());
  dev_ctx->Alloc(out_tensor.get(), weight_tensor->dtype());

  auto epsilon_scalar = phi::Scalar(epsilon);
  LAUNCH_TOPSATENOP(topsvllmRmsNorm,
                    (*dev_ctx),
                    *out_tensor,
                    x_input,
                    *weight_tensor,
                    epsilon_scalar);

  if (x_rank > 2) {
    out_tensor->Resize(x_dims);
  }

  return {paddle::Tensor(out_tensor)};
}

std::vector<paddle::Tensor> FusedAddRmsNorm(const paddle::Tensor &x,
                                            const paddle::Tensor &residual,
                                            const paddle::Tensor &weight,
                                            float epsilon) {
  //   PADDLE_GCU_KERNEL_TRACE("common_FusedAddRmsNorm");
  auto dev_ctx = static_cast<const phi::CustomContext *>(
      paddle::experimental::DeviceContextPool::Instance().Get(x.place()));

  auto x_tensor = static_cast<const phi::DenseTensor *>(x.impl().get());
  auto residual_tensor =
      static_cast<const phi::DenseTensor *>(residual.impl().get());
  auto weight_tensor =
      static_cast<const phi::DenseTensor *>(weight.impl().get());

  std::shared_ptr<phi::DenseTensor> out_tensor =
      std::make_shared<phi::DenseTensor>();
  out_tensor->Resize(x.dims());
  dev_ctx->Alloc(out_tensor.get(), weight_tensor->dtype());

  std::shared_ptr<phi::DenseTensor> residual_update_tensor =
      std::make_shared<phi::DenseTensor>();
  residual_update_tensor->Resize(residual.dims());
  dev_ctx->Alloc(residual_update_tensor.get(), weight_tensor->dtype());

  LAUNCH_TOPSATENOP(topsvllmFusedAddRmsNorm,
                    (*dev_ctx),
                    *out_tensor,
                    *residual_update_tensor,
                    *x_tensor,
                    *residual_tensor,
                    *weight_tensor,
                    epsilon);

  return {paddle::Tensor(out_tensor), paddle::Tensor(residual_update_tensor)};
}

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
                                       float epsilon) {
  PADDLE_GCU_KERNEL_TRACE("common_ComputeQKV");
  // 1. norm before QKV
  std::vector<paddle::Tensor> norm_before_qkv;
  // first layer
  bool first_layer = (hidden_input.data() == residual.data());
  if (first_layer) {
    norm_before_qkv =
        custom_op_common::RmsNorm(hidden_input, norm_weight, epsilon);
    norm_before_qkv.emplace_back(hidden_input);
  } else {
    norm_before_qkv = custom_op_common::FusedAddRmsNorm(
        hidden_input, residual, norm_weight, epsilon);
  }

  // 2. Compute QKV
  auto qkv_hidden =
      paddle::experimental::matmul(norm_before_qkv[0], qkv_weight, false, true);

  // [batch_size, (q_num_head + k_num_head + v_num_head) * head_dim]
  const auto &qkv_hidden_dims = qkv_hidden.dims();
  // [2, batch_size, cache_seq_len, kv_num_head, head_dim]
  const auto &cache_kvs_dims = cache_kvs.dims();
  int bsz = qkv_hidden_dims[0];
  int cache_seq_len = cache_kvs_dims[2];
  int head_dim = cache_kvs_dims[4];

  int kv_num_head = cache_kvs_dims[3];
  int q_num_head = qkv_hidden_dims[qkv_hidden_dims.size() - 1] / head_dim -
                   kv_num_head - kv_num_head;
  int q_size = q_num_head * head_dim;
  int kv_size = kv_num_head * head_dim;

  paddle::experimental::IntArray sections({q_size, kv_size, kv_size});
  std::vector<paddle::Tensor> qkv = paddle::experimental::split(
      qkv_hidden, sections, paddle::experimental::Scalar(-1));

  // [batch_size, seq_len, q_num_head, head_dim]
  auto query =
      paddle::experimental::reshape_(qkv[0], {bsz, -1, q_num_head, head_dim});
  // [batch_size, seq_len, kv_num_head, head_dim]
  auto key =
      paddle::experimental::reshape_(qkv[1], {bsz, -1, kv_num_head, head_dim});
  // [batch_size, seq_len, kv_num_head, head_dim]
  auto value =
      paddle::experimental::reshape_(qkv[2], {bsz, -1, kv_num_head, head_dim});

  // q,k,v: [batch_size, q/kv_seq_len, q/kv_num_head, head_dim]
  // residual: [batch_size, q/kv_seq_len, hidden_size],
  return {query, key, value, norm_before_qkv[1]};
}

std::vector<paddle::Tensor> UpdateKvCache(paddle::Tensor &key,    // NOLINT
                                          paddle::Tensor &value,  // NOLINT
                                          const paddle::Tensor &cache_kvs,
                                          bool is_decoder) {
  PADDLE_GCU_KERNEL_TRACE("common_UpdateKvCache");
  // [1, batch_size, current_seq_len, kv_num_head, head_dim]
  key = paddle::experimental::unsqueeze_(key, {0});
  // [1, batch_size, current_seq_len, kv_num_head, head_dim]
  value = paddle::experimental::unsqueeze_(value, {0});

  // [2, batch_size, current_seq_len, kv_num_head, head_dim]
  auto current_kv = paddle::experimental::concat(
      {key, value}, paddle::experimental::Scalar(0));

  paddle::Tensor out_key;
  paddle::Tensor out_value;
  paddle::Tensor out_cache_kv;

  if (is_decoder) {
    // [2, batch_size, cached_seq_len + current_seq_len, kv_num_head, head_dim]
    out_cache_kv = paddle::experimental::concat(
        {cache_kvs, current_kv}, paddle::experimental::Scalar(2));
    auto new_kv =
        paddle::experimental::split(out_cache_kv,
                                    paddle::experimental::IntArray({1, 1}),
                                    paddle::experimental::Scalar(0));
    out_key = new_kv[0];
    out_value = new_kv[1];
  } else {
    out_key = key;
    out_value = value;
    out_cache_kv = current_kv;
  }

  out_key = paddle::experimental::squeeze_(out_key, {0});
  out_value = paddle::experimental::squeeze_(out_value, {0});

  VLOG(6) << "UpdateKvCache, key shape:" << out_key.dims().to_str()
          << ", value shape:" << out_value.dims().to_str()
          << ", input cache_kv shape:" << cache_kvs.dims().to_str()
          << ", out cache_kv shape:" << out_cache_kv.dims().to_str();

  auto p = const_cast<paddle::Tensor *>(&cache_kvs);
  p->set_impl(out_cache_kv.impl());

  //  [batch_size, total_seq_len, kv_num_head, head_dim]
  return {out_key, out_value};
}

paddle::Tensor SelfAttention(const paddle::Tensor &query,
                             paddle::Tensor &key,    // NOLINT
                             paddle::Tensor &value,  // NOLINT
                             const paddle::Tensor &position_ids,
                             const paddle::Tensor &cos_sin_table,
                             const paddle::Tensor &attention_mask,
                             const paddle::Tensor &attn_out_linear_weight,
                             const paddle::Tensor &cache_kvs,
                             bool is_decoder) {
  PADDLE_GCU_KERNEL_TRACE("common_SelfAttention");
  // 1. RoPE
  auto rope = custom_op_common::FusedRotaryEmbedding(
      query, key, cos_sin_table, position_ids, true);

  // 2. Update caches and KVs
  auto kvs =
      custom_op_common::UpdateKvCache(rope[1], value, cache_kvs, is_decoder);

  // 3. self-Attention
  // [batch_size, seq_len, q_num_head, head_dim]
  auto attn_out = custom_op_common::FusedSdpFlashAttention(
      rope[0], kvs[0], kvs[1], attention_mask);

  // 4. Attention out linear
  // [batch_size, seq_len, q_num_head, head_dim] ->
  // [batch_size, seq_len, hidden_size]
  auto dims = attn_out[0].dims();
  attn_out[0] = paddle::experimental::reshape_(
      attn_out[0], {dims[0], dims[1], dims[2] * dims[3]});
  auto attn_out_linear = paddle::experimental::matmul(
      attn_out[0], attn_out_linear_weight, false, false);

  // [batch_size, seq_len, hidden_size]
  return attn_out_linear;
}

std::vector<paddle::Tensor> FeedForward(
    const paddle::Tensor &attn_out,
    const paddle::Tensor &attn_residual,
    const paddle::Tensor &ffn_norm_weight,
    const paddle::Tensor &mlp_gate_up_weight,
    const paddle::Tensor &mlp_down_weight,
    float epsilon) {
  PADDLE_GCU_KERNEL_TRACE("common_FeedForward");
  // 1. FFN norm
  auto ffn_norm = custom_op_common::FusedAddRmsNorm(
      attn_out, attn_residual, ffn_norm_weight, epsilon);

  // 2. Compute FFN1, MLP gate up
  auto mlp_gate_up = paddle::experimental::matmul(
      ffn_norm[0], mlp_gate_up_weight, false, false);

  // 3. Compute activation
  auto mlp_act = paddle::experimental::swiglu(
      mlp_gate_up, paddle::optional<paddle::Tensor>());

  // 4. Compute FFN2, MLP down
  auto mlp_out =
      paddle::experimental::matmul(mlp_act, mlp_down_weight, false, false);

  // [batch_size, seq_len, hidden_size], [batch_size, seq_len, hidden_size]
  return {mlp_out, ffn_norm[1]};
}

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
    bool is_decoder) {
  VLOG(6) << "FusedTransformerLayer_" << (is_decoder ? "Decoder" : "Encoder")
          << ", hidden_input shape:" << hidden_input.dims().to_str()
          << ", norm_weight shape:" << norm_weight.dims().to_str()
          << ", residual shape:" << residual.dims().to_str()
          << ", position_ids shape:" << position_ids.dims().to_str()
          << ", qkv_weight shape:" << qkv_weight.dims().to_str()
          << ", cache_kvs shape:" << cache_kvs.dims().to_str()
          << ", cos_sin_table shape:" << cos_sin_table.dims().to_str()
          << ", attention_mask shape:" << attention_mask.dims().to_str()
          << ", attn_out_linear_weight shape:"
          << attn_out_linear_weight.dims().to_str()
          << ", ffn_norm_weight shape:" << ffn_norm_weight.dims().to_str()
          << ", mlp_gate_up_weight shape:" << mlp_gate_up_weight.dims().to_str()
          << ", mlp_down_weight shape:" << mlp_down_weight.dims().to_str()
          << ", epsilon:" << epsilon;

  // query, key, value, mlp_residual
  auto qkv_residual = custom_op_common::ComputeQKV(
      norm_weight, hidden_input, residual, qkv_weight, cache_kvs, epsilon);

  // attn_out_linear
  auto attn_out = custom_op_common::SelfAttention(qkv_residual[0],
                                                  qkv_residual[1],
                                                  qkv_residual[2],
                                                  position_ids,
                                                  cos_sin_table,
                                                  attention_mask,
                                                  attn_out_linear_weight,
                                                  cache_kvs,
                                                  is_decoder);

  // mlp_out, residual_out
  auto ffn_out = custom_op_common::FeedForward(attn_out,
                                               qkv_residual[3],
                                               ffn_norm_weight,
                                               mlp_gate_up_weight,
                                               mlp_down_weight,
                                               epsilon);

  // ret: mlp_out, residual_out
  return {ffn_out[0], ffn_out[1]};
}

std::vector<paddle::Tensor> TopPSampling(const paddle::Tensor &probs,
                                         const paddle::Tensor &top_p) {
  //   PADDLE_GCU_KERNEL_TRACE("common_TopPSampling");
  auto sort_out = paddle::experimental::argsort(probs, -1, true);
  paddle::Tensor sorted_probs = std::get<0>(sort_out);
  paddle::Tensor sorted_indices = std::get<1>(sort_out);

  auto cumulative_probs = paddle::experimental::cumsum(
      sorted_probs, paddle::experimental::Scalar(-1));
  auto sorted_indices_to_remove =
      paddle::experimental::greater_than(cumulative_probs, top_p);
  sorted_indices_to_remove = paddle::experimental::cast(
      sorted_indices_to_remove, paddle::DataType::INT32);

  auto probs_dims = probs.dims();
  int64_t bsz = probs_dims[0];
  int64_t vocab_size = probs_dims[1];
  auto first_token_remove =
      paddle::full({bsz, 1}, 0, paddle::DataType::INT32, probs.place());
  auto remain_token_remove = paddle::experimental::slice(
      sorted_indices_to_remove, {1}, {1}, {vocab_size}, {}, {});
  sorted_indices_to_remove =
      paddle::experimental::concat({first_token_remove, remain_token_remove},
                                   paddle::experimental::Scalar(1));

  auto index_offset_cpu =
      paddle::full({bsz, 1}, 0, paddle::DataType::INT32, paddle::CPUPlace());
  int32_t *index_offset_cpu_data = index_offset_cpu.data<int32_t>();
  for (int i = 0; i < bsz; ++i) {
    index_offset_cpu_data[i] = i * vocab_size;
  }
  auto index_offset = index_offset_cpu.copy_to(probs.place(), true);
  sorted_indices = paddle::experimental::add(sorted_indices, index_offset);

  int64_t probs_numel = bsz * vocab_size;
  auto flatten_sorted_indices_to_remove =
      paddle::experimental::reshape_(sorted_indices_to_remove, {probs_numel});
  auto flatten_sorted_indices =
      paddle::experimental::reshape_(sorted_indices, {probs_numel});
  //   auto condition = paddle::experimental::scatter(
  //       flatten_sorted_indices_to_remove, flatten_sorted_indices,
  //       flatten_sorted_indices_to_remove);
  auto condition =
      paddle::experimental::index_put(flatten_sorted_indices_to_remove,
                                      {flatten_sorted_indices},
                                      flatten_sorted_indices_to_remove,
                                      false);
  condition = paddle::experimental::reshape_(condition, {bsz, vocab_size});

  auto zero_probs =
      paddle::experimental::full_like(probs, 0.0, probs.dtype(), probs.place());
  auto new_probs = paddle::experimental::where(condition, zero_probs, probs);

  auto next_tokens = paddle::experimental::multinomial(new_probs);
  auto next_scores = paddle::experimental::index_sample(new_probs, next_tokens);

  return {next_scores, next_tokens};
}

}  // namespace custom_op_common
