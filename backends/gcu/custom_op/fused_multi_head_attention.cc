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

std::vector<std::vector<int64_t>> FusedMultiHeadAttnInferShape(
    const std::vector<int64_t>& q_shape,
    const std::vector<int64_t>& k_shape,
    const std::vector<int64_t>& v_shape,
    int head_dim) {
  return {q_shape};
}

std::vector<paddle::DataType> FusedMultiHeadAttnInferDtype(
    const paddle::DataType& q_dtype,
    const paddle::DataType& k_dtype,
    const paddle::DataType& v_dtype,
    int head_dim) {
  return {q_dtype};
}

std::vector<paddle::Tensor> FusedMultiHeadAttn(const paddle::Tensor& q,
                                               const paddle::Tensor& k,
                                               const paddle::Tensor& v,
                                               int head_dim) {
  PADDLE_GCU_KERNEL_TRACE("fused_multi_head_attention");
  VLOG(6) << "[CUSTOM_KERNEL] Custom Operator: fused_multi_head_attention";
  auto dev_ctx = static_cast<const phi::CustomContext*>(
      paddle::experimental::DeviceContextPool::Instance().Get(q.place()));

  // [batch_size, seq_len, num_head, head_dim]
  const auto& q_dims = q.dims();
  int bsz = q_dims[0];
  int seq_len = q_dims[1];

  auto attention_mask = paddle::experimental::zeros(
      {bsz, 1, seq_len, seq_len}, q.dtype(), q.place());

  auto attn_out =
      custom_op_common::FusedSdpFlashAttention(q, k, v, attention_mask);

  return {attn_out};
}

PD_BUILD_OP(fused_multi_head_attention)
    .Inputs({"Q", "K", "V"})
    .Outputs({"Out"})
    .Attrs({"head_dim: int"})
    .SetKernelFn(PD_KERNEL(FusedMultiHeadAttn))
    .SetInferShapeFn(PD_INFER_SHAPE(FusedMultiHeadAttnInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(FusedMultiHeadAttnInferDtype));
