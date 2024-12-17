# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle
import paddlenlp_ops

paddle.device.set_device("intel_hpu")

paddle.seed(105)


def ref_result(
    query_states,
    key_states,
    value_states,
    attention_mask,
    linear_weights,
    scaling_factor,
):
    bsz, q_len, num_heads, head_dim = query_states.shape
    attn_output = paddle.incubate.nn.functional.fused_dot_product_attention(
        query_states,
        key_states,
        value_states,
        attention_mask,
        0.0,
        attention_mask is None,
        scaling_factor,
        False,
    )
    attn_output = attn_output.reshape([bsz, q_len, head_dim * num_heads])

    out_linear_out = paddle.matmul(attn_output, linear_weights)

    return out_linear_out


head_dim = 128
num_head = 32
kv_num_heads = num_head
hidden_size = num_head * head_dim

batch_size = 5
seq_len = 1
# seq_len = 25
kv_seq_len = 25
max_seq_length = 2048

query_states = paddle.rand(
    [batch_size, num_head, seq_len, head_dim], dtype=paddle.float32
).to(paddle.bfloat16)
key_states = paddle.rand(
    [batch_size, kv_num_heads, kv_seq_len, head_dim], dtype=paddle.float32
).to(paddle.bfloat16)
value_states = paddle.rand(
    [batch_size, kv_num_heads, kv_seq_len, head_dim], dtype=paddle.float32
).to(paddle.bfloat16)

attn_mask = paddle.ones([1, 1, max_seq_length, max_seq_length], dtype=paddle.bfloat16)
attn_mask = paddle.tril(attn_mask)
attn_mask = (1.0 - attn_mask) * -10000.0

linear_weights = paddle.rand([hidden_size, hidden_size], dtype=paddle.float32).to(
    paddle.bfloat16
)


def main():
    attention_mask = attn_mask[..., :seq_len, :kv_seq_len]
    attention_mask = attention_mask.astype(query_states.dtype)

    out_linear_out_op = paddlenlp_ops.fused_sdpa_proj(
        query_states,
        key_states,
        value_states,
        attention_mask,
        linear_weights,
        scaling_factor=head_dim**-0.5,
    )

    out_linear_out_ref = ref_result(
        query_states.transpose([0, 2, 1, 3]),
        key_states.transpose([0, 2, 1, 3]),
        value_states.transpose([0, 2, 1, 3]),
        attention_mask,
        linear_weights,
        scaling_factor=head_dim**-0.5,
    )

    print(out_linear_out_ref)
    print(out_linear_out_op)
    print((out_linear_out_op == out_linear_out_ref).all())


if __name__ == "__main__":
    main()
