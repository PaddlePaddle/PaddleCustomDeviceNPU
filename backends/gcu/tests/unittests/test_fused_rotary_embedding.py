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
import os
import numpy as np
import unittest
from ddt import ddt, data, unpack
from api_base import TestAPIBase

for lib in os.listdir(os.getenv("CUSTOM_DEVICE_ROOT")):
    if lib.endswith(".so"):
        paddle.utils.cpp_extension.extension_utils.load_op_meta_info_and_register_op(
            lib
        )

# The table retains its original format for better comparison of parameter settings.
# fmt: off
FUSED_ROPE_CASE = [
    # for llama2
    {"batch_size": 4, "seq_len": 512, "num_heads": 40, "num_kv_heads": 40, "dtype": np.float32, "position_dtype": np.int64},
    {"batch_size": 4, "seq_len": 512, "num_heads": 40, "num_kv_heads": 40, "dtype": np.float16, "position_dtype": np.int64},

    {"batch_size": 5, "seq_len": 512, "num_heads": 40, "num_kv_heads": 40, "dtype": np.float32, "position_dtype": np.int32},
    {"batch_size": 5, "seq_len": 512, "num_heads": 40, "num_kv_heads": 40, "dtype": np.float16, "position_dtype": np.int32},

]
# fmt: on

NATIVE_IMPL_DEV = "gcu"


class LlamaRotaryEmbedding(paddle.nn.Layer):
    def __init__(self, dim, max_position_embeddings=2048, base=10000):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        # [dim / 2]
        self.inv_freq = 1.0 / (
            self.base
            ** (paddle.cast(paddle.arange(0, self.dim, 2), dtype="float32") / self.dim)
        )
        self._set_cos_sin_cache(seq_len=max_position_embeddings)

    def _set_cos_sin_cache(self, seq_len):
        self.max_seq_len_cached = seq_len
        # [seq_len]
        t = paddle.arange(seq_len, dtype="float32")
        # [seq_len, dim/2]
        freqs = paddle.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        # [seq_len, dim]
        emb = paddle.concat([freqs, freqs], axis=-1)
        # [1, seqlen, 1, dim]
        self.cos_cached = emb.cos()[None, :, None, :]
        self.sin_cached = emb.sin()[None, :, None, :]
        self.cos_sin_table = paddle.concat([freqs.cos(), freqs.sin()], axis=-1)

    def forward(self, dtype, seq_len=None):
        cos = self.cos_cached[:, :seq_len, :, :]
        sin = self.sin_cached[:, :seq_len, :, :]
        return (cos, sin, self.cos_sin_table)


def get_rope(dim, max_position_embeddings):
    paddle.set_device(NATIVE_IMPL_DEV)
    rope = LlamaRotaryEmbedding(dim, max_position_embeddings)
    return rope


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return paddle.concat([-x2, x1], axis=-1)  # shape is the same as x


def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    if position_ids is None:
        cos = cos[:, : q.shape[1], :, :]  # [bs, seq_len, 1, dim]
        sin = sin[:, : q.shape[1], :, :]  # [bs, seq_len, 1, dim]
    else:
        cos = cos.squeeze(axis=[0, 2])  # [seq_len, dim]
        sin = sin.squeeze(axis=[0, 2])  # [seq_len, dim]
        cos = cos[position_ids].unsqueeze(2)  # [bs, seq_len, 1, dim]
        sin = sin[position_ids].unsqueeze(2)  # [bs, seq_len, 1, dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


@ddt
class TestFusedRotaryEmbedding(TestAPIBase):
    def setUp(self):
        self.init_attrs()

    def init_attrs(self):
        self.head_dim = 128
        self.max_position = 512
        self.is_neox = True
        self.rotary_dim = self.head_dim
        self.rope = get_rope(self.head_dim, self.max_position)

        self.batch_size = 4
        self.seq_len = 512
        self.num_heads = 40
        self.num_kv_heads = 40

        self.dtype = np.float32
        self.position_dtype = np.int64

        self.init_shapes()

    def init_shapes(self):
        self.q_shape = [self.batch_size, self.seq_len, self.num_heads, self.head_dim]
        self.k_shape = [self.batch_size, self.seq_len, self.num_kv_heads, self.head_dim]
        self.positions_shape = [self.batch_size, self.seq_len]
        self.cos_sin_table_shape = [self.max_position, self.rotary_dim]

    def prepare_data(self):
        self.data_q = self.generate_data(self.q_shape, self.dtype)
        self.data_k = self.generate_data(self.k_shape, self.dtype)
        self.data_position_ids = np.tile(
            np.arange(self.seq_len, dtype=self.position_dtype), [self.batch_size, 1]
        )

        self.cos, self.sin, self.cos_sin = self.rope(self.dtype, self.seq_len)
        self.cos_sin.to("gcu")

    def forward(self):
        q = paddle.to_tensor(self.data_q, dtype=self.dtype)
        k = paddle.to_tensor(self.data_k, dtype=self.dtype)
        position_ids = paddle.to_tensor(
            self.data_position_ids, dtype=self.position_dtype
        )
        cos_sin = self.cos_sin.astype(self.dtype)
        return paddle.base.core.eager._run_custom_op(
            "fused_rotary_embedding_gcu", q, k, cos_sin, position_ids, self.is_neox
        )

    def fused_rotary_embedding_impl(self, dtype):
        paddle.set_device(NATIVE_IMPL_DEV)
        q = paddle.to_tensor(self.data_q, dtype=dtype)
        k = paddle.to_tensor(self.data_k, dtype=dtype)
        position_ids = paddle.to_tensor(
            self.data_position_ids, dtype=self.position_dtype
        )
        cos = self.cos.astype(dtype)
        sin = self.sin.astype(dtype)
        q_embed, k_embed = apply_rotary_pos_emb(q, k, cos, sin, position_ids)
        if dtype != self.dtype:
            return [q_embed.astype(self.dtype), k_embed.astype(self.dtype)]
        else:
            return [q_embed, k_embed]

    def expect_output(self):
        if NATIVE_IMPL_DEV == "cpu" and self.dtype == np.float16:
            outs = self.fused_rotary_embedding_impl(np.float32)
        else:
            outs = self.fused_rotary_embedding_impl(self.dtype)
        return outs

    @data(*FUSED_ROPE_CASE)
    @unpack
    def test_check_output(
        self, batch_size, seq_len, num_heads, num_kv_heads, dtype, position_dtype
    ):
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads

        self.dtype = dtype
        self.position_dtype = position_dtype
        self.init_shapes()

        rtol = 1e-5
        atol = 1e-5
        if dtype == np.float16:
            rtol = 1e-3
            atol = 1e-3
        self.check_output_gcu_with_customized(
            self.forward, self.expect_output, rtol=rtol, atol=atol
        )


if __name__ == "__main__":
    unittest.main()
