# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License.
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

paddle.set_device("intel_hpu")
# paddle.set_device("cpu")


class KVCache(paddle.nn.Layer):
    def __init__(self):
        super(KVCache, self).__init__()
        self.cache = None
        self.inp_seq_len = -1

    def allocate(self, inp_seq_len, dtype, shape):
        if self.cache is None or self.cache.shape != shape:
            self.inp_seq_len = inp_seq_len
            self.cache = paddle.zeros(shape, dtype=dtype)
        else:
            assert (
                self.inp_seq_len == inp_seq_len
            ), f"inp_seq_len must be the same. self.inp_seq_len:{self.inp_seq_len} inp_seq_len:{inp_seq_len}"
            self.cache.fill_(0)

    @staticmethod
    def update(prev, cur, dim, idx, inp_seq_len):
        orig_cur = cur
        if prev.shape == cur.shape:
            # prev.copy_(cur)
            paddle.assign(cur, output=prev)
            return orig_cur
        if idx is not None and cur.shape[2] > 1 and cur.shape[2] <= prev.shape[2]:
            # Initialize
            # prev[:, :, :inp_seq_len, :].copy_(cur)
            prev[:, :, :inp_seq_len, :] = cur
            # paddle.assign(cur, output=prev[:, :, :inp_seq_len, :])
            return orig_cur
        if idx is not None:
            # prev.index_copy_(dim, idx - 1, cur)
            if dim == 0:
                prev.scatter_(idx, cur)
            else:
                times, temp_shape, temp_index = (
                    paddle.prod(paddle.to_tensor(prev.shape[:dim])),
                    prev.shape,
                    idx,
                )
                prev, new_t = prev.reshape([-1] + temp_shape[dim + 1 :]), cur.reshape(
                    [-1] + temp_shape[dim + 1 :]
                )
                for i in range(1, times):
                    temp_index = paddle.concat([temp_index, idx + temp_shape[dim] * i])
                prev.scatter_(temp_index, new_t).reshape_(temp_shape)
            return prev
        else:
            return paddle.concat((prev, cur), dim=dim)

    def get_shape(self):
        if self.cache is None:
            return None
        return self.cache.shape

    def forward(self, cur, dim, idx):
        return self.update(self.cache, cur, dim, idx, self.inp_seq_len)


batch_size = 1
num_key_value_heads = 32
max_seq_len = 1024
head_dim = 128

# paddle case
cache_shape = (batch_size, num_key_value_heads, max_seq_len, head_dim)
dtype = "float32"

inp_seq_len = 128

k_cache = KVCache()
k_cache.allocate(inp_seq_len, dtype, cache_shape)

key_states = paddle.rand(
    (batch_size, num_key_value_heads, inp_seq_len, head_dim), dtype=dtype
)

token_idx = paddle.to_tensor([0], dtype="int64")
prefill = k_cache(key_states, 2, token_idx)

print((prefill == k_cache.cache[:, :, :inp_seq_len, :]).all())

inp_seq_len = 1
token_idx = paddle.to_tensor([128], dtype="int64")
key_state = paddle.ones(
    (batch_size, num_key_value_heads, inp_seq_len, head_dim), dtype=dtype
)
decode = k_cache(key_state, 2, token_idx)
print((key_state == decode[:, :, token_idx, :]).all())

if 0:
    # torch case
    import torch

    class KVCache_torch(torch.nn.Module):
        def __init__(self):
            super(KVCache_torch, self).__init__()
            self.cache = None
            self.inp_seq_len = -1

        def allocate(self, inp_seq_len, dtype, shape):
            if self.cache is None or self.cache.shape != shape:
                self.inp_seq_len = inp_seq_len
                self.cache = torch.zeros(shape, dtype=dtype)
            else:
                assert (
                    self.inp_seq_len == inp_seq_len
                ), f"inp_seq_len must be the same. self.inp_seq_len:{self.inp_seq_len} inp_seq_len:{inp_seq_len}"
                self.cache.fill_(0)

        @staticmethod
        def update(prev, cur, dim, idx, inp_seq_len):
            orig_cur = cur
            if prev.shape == cur.shape:
                prev.copy_(cur)
                return orig_cur
            if idx is not None and cur.shape[2] > 1 and cur.shape[2] <= prev.shape[2]:
                # Initialize
                prev[:, :, :inp_seq_len, :].copy_(cur)
                return orig_cur
            if idx is not None:
                prev.index_copy_(dim, idx - 1, cur)
                return prev
            else:
                return torch.cat((prev, cur), dim=dim)

        def get_shape(self):
            if self.cache is None:
                return None
            return self.cache.shape

        def forward(self, cur, dim, idx):
            return self.update(self.cache, cur, dim, idx, self.inp_seq_len)

    dtype = torch.float32

    inp_seq_len = 128

    v_cache = KVCache_torch()
    v_cache.allocate(inp_seq_len, dtype, cache_shape)

    value_states = torch.rand(
        batch_size, num_key_value_heads, inp_seq_len, head_dim, dtype=dtype
    )

    token_idx = 1
    token_idx_t = torch.tensor(token_idx, dtype=torch.int64)
    prefill = v_cache(value_states, 2, token_idx_t)
    print((prefill == v_cache.cache[:, :, :inp_seq_len, :]).all())

    inp_seq_len = 1
    token_idx = 129
    token_idx_t = torch.tensor(token_idx, dtype=torch.int64)
    value_state = torch.ones(
        batch_size, num_key_value_heads, inp_seq_len, head_dim, dtype=dtype
    )
    decode = v_cache(value_state, 2, token_idx_t)
    print((value_state == decode[:, :, token_idx - 1, :]).all())
