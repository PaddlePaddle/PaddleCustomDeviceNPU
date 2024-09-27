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
import math
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
ATTENTION_CASE = [
    # for llama2
    # topsvllmMemoryEfficientAttention only support float16
    {"batch_size": 4, "q_seq_len": 512, "kv_seq_len": 512, "num_heads": 40, "num_kv_heads": 40, "head_dim": 128, "dtype": np.float16, "casual": False},
    {"batch_size": 4, "q_seq_len": 512, "kv_seq_len": 512, "num_heads": 40, "num_kv_heads": 40, "head_dim": 128, "dtype": np.float16, "casual": True},

    {"batch_size": 4, "q_seq_len": 512, "kv_seq_len": 512, "num_heads": 40, "num_kv_heads": 40, "head_dim": 512, "dtype": np.float16, "casual": False},
    {"batch_size": 4, "q_seq_len": 512, "kv_seq_len": 512, "num_heads": 40, "num_kv_heads": 40, "head_dim": 512, "dtype": np.float16, "casual": True},

    {"batch_size": 4, "q_seq_len": 512, "kv_seq_len": 256, "num_heads": 40, "num_kv_heads": 40, "head_dim": 128, "dtype": np.float16, "casual": False},
    {"batch_size": 4, "q_seq_len": 512, "kv_seq_len": 256, "num_heads": 40, "num_kv_heads": 40, "head_dim": 128, "dtype": np.float16, "casual": True},

]
# fmt: on

NATIVE_IMPL_DEV = "gcu"


def get_triangle_upper_mask(shape, dtype):
    paddle.set_device(NATIVE_IMPL_DEV)
    #  [batch_size, 1, q_seq_len, kv_seq_len]
    shape[1] = 1
    paddle_dtype = paddle.base.data_feeder.convert_dtype(dtype)
    mask = paddle.full(shape, paddle.finfo(paddle_dtype).min, dtype=paddle_dtype)
    mask = paddle.triu(mask, diagonal=1)
    return mask


@ddt
class TestFusedSdpFlashAttention(TestAPIBase):
    def setUp(self):
        self.init_attrs()

    def init_attrs(self):
        self.batch_size = 4
        self.q_seq_len = 512
        self.kv_seq_len = 512
        self.num_heads = 40
        self.num_kv_heads = 40
        self.head_dim = 128
        self.casual = False
        self.dtype = np.float16

        self.dropout = 0.0
        self.is_test = True

    def prepare_data(self):
        self.data_q = self.generate_data(self.q_shape, self.dtype)
        self.data_k = self.generate_data(self.k_shape, self.dtype)
        self.data_v = self.generate_data(self.v_shape, self.dtype)
        if self.casual:
            self.data_attn_mask = get_triangle_upper_mask(
                self.attn_mask_shape, self.dtype
            )
        else:
            self.data_attn_mask = paddle.to_tensor(
                self.generate_data(self.attn_mask_shape, self.dtype), dtype=self.dtype
            )

    def forward(self):
        q = paddle.to_tensor(self.data_q, dtype=self.dtype)
        k = paddle.to_tensor(self.data_k, dtype=self.dtype)
        v = paddle.to_tensor(self.data_v, dtype=self.dtype)
        attn_mask = self.data_attn_mask.to("gcu")
        return paddle.base.core.eager._run_custom_op(
            "fused_sdp_flash_attention_gcu",
            q,
            k,
            v,
            attn_mask,
            self.dropout,
            self.casual,
            self.is_test,
        )[0]

    def sdp_flash_attention_impl(self, dtype):
        paddle.set_device(NATIVE_IMPL_DEV)
        query_states = paddle.to_tensor(self.data_q, dtype=dtype)
        key_states = paddle.to_tensor(self.data_k, dtype=dtype)
        value_states = paddle.to_tensor(self.data_v, dtype=dtype)
        attention_mask = self.data_attn_mask.to(NATIVE_IMPL_DEV)

        _, _, _, head_dim = query_states.shape

        #  [batch_size, seq_len, num_heads, head_dim] -> [batch_size, num_heads, seq_len, head_dim]
        query_states = paddle.transpose(query_states, [0, 2, 1, 3])
        key_states = paddle.transpose(key_states, [0, 2, 1, 3])
        value_states = paddle.transpose(value_states, [0, 2, 1, 3])

        # matmul and devide by sqrt(head_dim)
        attn_weights = paddle.matmul(
            query_states / math.sqrt(head_dim), key_states.transpose([0, 1, 3, 2])
        )

        attn_weights = attn_weights + attention_mask
        attn_weights = paddle.nn.functional.softmax(
            attn_weights, axis=-1, dtype="float32"
        ).astype(query_states.dtype)

        attn_output = paddle.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose([0, 2, 1, 3])
        if dtype != self.dtype:
            return attn_output.astype(self.dtype)
        return attn_output

    def expect_output(self):
        if NATIVE_IMPL_DEV == "cpu" and self.dtype == np.float16:
            out = self.sdp_flash_attention_impl(np.float32)
        else:
            out = self.sdp_flash_attention_impl(self.dtype)
        return out

    @data(*ATTENTION_CASE)
    @unpack
    def test_check_output(
        self,
        batch_size,
        q_seq_len,
        kv_seq_len,
        num_heads,
        num_kv_heads,
        head_dim,
        dtype,
        casual,
    ):
        self.batch_size = batch_size
        self.q_seq_len = q_seq_len
        self.kv_seq_len = kv_seq_len
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.casual = casual

        self.q_shape = [self.batch_size, self.q_seq_len, self.num_heads, self.head_dim]
        self.k_shape = [
            self.batch_size,
            self.kv_seq_len,
            self.num_kv_heads,
            self.head_dim,
        ]
        self.v_shape = [
            self.batch_size,
            self.kv_seq_len,
            self.num_kv_heads,
            self.head_dim,
        ]
        self.attn_mask_shape = [self.batch_size, 1, self.q_seq_len, self.kv_seq_len]

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
