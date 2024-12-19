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
from paddlenlp.transformers.llama.modeling import LlamaRotaryEmbedding
import paddlenlp_ops

paddle.device.set_device("intel_hpu")

paddle.seed(102)


def ref_result(
    src, ln_scales, qkv_weights, cos, sin, position_ids, epsilon, head_dim, num_head
):
    hidden_states = paddle.incubate.nn.functional.fused_rms_norm(
        src, ln_scales, None, epsilon, 2
    )[0]

    qkv_out = paddle.matmul(hidden_states, qkv_weights, False, True)

    fused_hidden_size = qkv_out.shape[2]
    kv_num_heads = (fused_hidden_size - num_head * head_dim) // head_dim // 2
    num_groups = num_head // kv_num_heads
    target_shape = [0, 0, (num_groups + 2) * kv_num_heads, head_dim]

    qkv_out = paddle.reshape_(qkv_out, target_shape)

    qkv_out = paddle.transpose(qkv_out, [0, 2, 1, 3])

    query_states, key_states, value_states = paddle.split(
        qkv_out,
        num_or_sections=[num_head, kv_num_heads, kv_num_heads],
        axis=1,
    )

    cos = cos.squeeze().unsqueeze(0).unsqueeze(0)
    sin = sin.squeeze().unsqueeze(0).unsqueeze(0)
    query_states, _, _ = paddle.incubate.nn.functional.fused_rotary_position_embedding(
        query_states, None, None, sin=sin, cos=cos, position_ids=position_ids
    )
    key_states, _, _ = paddle.incubate.nn.functional.fused_rotary_position_embedding(
        key_states, None, None, sin=sin, cos=cos, position_ids=position_ids
    )

    return query_states, key_states, value_states


head_dim = 128
num_head = 32
kv_num_heads = 32
hidden_size = num_head * head_dim

batch_size = 8
seq_len = 34
epsilon = 1e-06

src = paddle.rand([batch_size, seq_len, hidden_size], dtype=paddle.bfloat16)
ln_scales = paddle.rand([hidden_size], dtype=paddle.bfloat16)
qkv_weights = paddle.rand([hidden_size * 3, hidden_size], dtype=paddle.float32).to(
    paddle.bfloat16
)


def main():
    position_id = paddle.arange(seq_len, dtype=paddle.int64).to(paddle.int64)
    # position_id = paddle.to_tensor([80])
    new_rope = paddle.expand(position_id, shape=[batch_size, seq_len])

    rotary_emb = LlamaRotaryEmbedding(head_dim)
    cos, sin = rotary_emb(src, seq_len=new_rope[0][-1] + 1)

    query_states_ref, key_states_ref, value_states_ref = ref_result(
        src, ln_scales, qkv_weights, cos, sin, new_rope, epsilon, head_dim, num_head
    )

    query_states_op, key_states_op, value_states_op = paddlenlp_ops.fused_rms_qkv_rope(
        src, ln_scales, qkv_weights, cos, sin, new_rope, epsilon, head_dim, num_head
    )

    # print(query_states_ref)
    # print(query_states_op)

    # print(key_states_ref)
    # print(key_states_op)

    # print(value_states_ref)
    # print(value_states_op)

    print((query_states_op == query_states_ref).all())
    print((key_states_op == key_states_ref).all())
    print((value_states_op == value_states_ref).all())


if __name__ == "__main__":
    main()
