/* Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "custom_op/custom_op_common.h"

std::vector<std::vector<int64_t>> FusedSelfAttnInferShape(
    const std::vector<int64_t>& x_shape, int head_dim) {
  auto new_dims = x_shape;
  auto shape_size = new_dims.size();
  new_dims[shape_size - 1] /= 3;  // QKV fused
  return {new_dims};
}

std::vector<paddle::DataType> FusedSelfAttnInferDtype(
    const paddle::DataType& x_dtype, int head_dim) {
  return {x_dtype};
}

// The linear implemented here must be passed in bias
std::vector<paddle::Tensor> FusedSelfAttn(const paddle::Tensor& x,
                                          int head_dim) {
  PADDLE_GCU_KERNEL_TRACE("fused_self_attn");
  VLOG(6) << "[CUSTOM_KERNEL] Custom Operator: fused_self_attn";
  auto dev_ctx = static_cast<const phi::CustomContext*>(
      paddle::experimental::DeviceContextPool::Instance().Get(x.place()));

  // [batch_size, seq_len, (q_num_head + k_num_head + v_num_head) * head_dim]
  const auto& qkv_hidden_dims = x.dims();
  int bsz = qkv_hidden_dims[0];

  // q_num_head == k_num_head == v_num_head
  int num_head = qkv_hidden_dims[qkv_hidden_dims.size() - 1] / head_dim / 3;

  int q_size = num_head * head_dim;
  int kv_size = q_size;

  paddle::experimental::IntArray sections({q_size, kv_size, kv_size});
  std::vector<paddle::Tensor> qkv = paddle::experimental::split(
      x, sections, paddle::experimental::Scalar(-1));

  // [batch_size, seq_len, q_num_head, head_dim]
  auto query =
      paddle::experimental::reshape_(qkv[0], {bsz, -1, num_head, head_dim});
  // [batch_size, seq_len, kv_num_head, head_dim]
  auto key =
      paddle::experimental::reshape_(qkv[1], {bsz, -1, num_head, head_dim});
  // [batch_size, seq_len, kv_num_head, head_dim]
  auto value =
      paddle::experimental::reshape_(qkv[2], {bsz, -1, num_head, head_dim});

  int seq_len = query.dims().at(1);
  auto attention_mask = paddle::experimental::zeros(
      {bsz, 1, seq_len, seq_len}, query.dtype(), x.place());

  auto attn_out = custom_op_common::FusedSdpFlashAttention(
      query, key, value, attention_mask);

  auto output = paddle::experimental::reshape_(
      attn_out[0], {bsz, seq_len, num_head * head_dim});

  return {output};
}

PD_BUILD_OP(fused_self_attn)
    .Inputs({"X"})
    .Outputs({"Out"})
    .Attrs({"head_dim: int"})
    .SetKernelFn(PD_KERNEL(FusedSelfAttn))
    .SetInferShapeFn(PD_INFER_SHAPE(FusedSelfAttnInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(FusedSelfAttnInferDtype));
