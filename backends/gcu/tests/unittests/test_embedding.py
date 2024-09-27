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
import numpy as np
import unittest
from ddt import ddt, data, unpack
from api_base import TestAPIBase


# The table retains its original format for better comparison of parameter settings.
# fmt: off
EMBEDDING_CASE = [
    {"bsz": 6, "seqlen": 8, "vocab": 10, "emb_size": 20, "ids_dtype": np.int32, "w_dtype": np.float32, "padding_idx": None},
    {"bsz": 6, "seqlen": 8, "vocab": 10, "emb_size": 20, "ids_dtype": np.int64, "w_dtype": np.float32, "padding_idx": None},
    {"bsz": 6, "seqlen": 8, "vocab": 10, "emb_size": 20, "ids_dtype": np.int32, "w_dtype": np.float16, "padding_idx": None},
    {"bsz": 6, "seqlen": 8, "vocab": 10, "emb_size": 20, "ids_dtype": np.int64, "w_dtype": np.float16, "padding_idx": None},
    # Topsaten only support padding_idx-none now
    # {"bsz": 6, "seqlen": 8, "vocab": 10, "emb_size": 20, "ids_dtype": np.int32, "w_dtype": np.float32, "padding_idx": -1},
    # {"bsz": 6, "seqlen": 8, "vocab": 10, "emb_size": 20, "ids_dtype": np.int64, "w_dtype": np.float32, "padding_idx": -1},
    # {"bsz": 6, "seqlen": 8, "vocab": 10, "emb_size": 20, "ids_dtype": np.int32, "w_dtype": np.float16, "padding_idx": -1},
    # {"bsz": 6, "seqlen": 8, "vocab": 10, "emb_size": 20, "ids_dtype": np.int64, "w_dtype": np.float16, "padding_idx": -1},
    # {"bsz": 6, "seqlen": 8, "vocab": 10, "emb_size": 20, "ids_dtype": np.int64, "w_dtype": np.float16, "padding_idx": 0},
    # {"bsz": 6, "seqlen": 8, "vocab": 10, "emb_size": 20, "ids_dtype": np.int64, "w_dtype": np.float16, "padding_idx": 1},
]
# fmt: on


@ddt
class TestEmbedding(TestAPIBase):
    def setUp(self):
        self.init_attrs()

    def init_attrs(self):
        self.bsz = 6
        self.seqlen = 8
        self.vocab = 10
        self.emb_size = 20
        self.ids_dtype = np.int32
        self.w_dtype = np.float32
        self.padding_idx = None

    def prepare_data(self):
        self.w = np.random.random([self.vocab, self.emb_size]).astype(self.w_dtype)
        self.x = np.random.randint(0, self.vocab, size=(self.bsz, self.seqlen)).astype(
            self.ids_dtype
        )

    def forward(self):
        w = paddle.to_tensor(self.w, dtype=self.w_dtype)
        x = paddle.to_tensor(self.x, dtype=self.ids_dtype)
        return paddle.nn.functional.embedding(x, w, padding_idx=self.padding_idx)

    def expect_output(self):
        if self.w_dtype != np.float16:
            out = self.calc_result(self.forward, "cpu")
        else:
            out = self.w[self.x]
            if self.padding_idx != -1:
                out[np.squeeze(self.x == self.padding_idx)] = np.zeros(self.emb_size)
        return out

    @data(*EMBEDDING_CASE)
    @unpack
    def test_check_output(
        self, bsz, seqlen, vocab, emb_size, ids_dtype, w_dtype, padding_idx
    ):
        self.bsz = bsz
        self.seqlen = seqlen
        self.vocab = vocab
        self.emb_size = emb_size
        self.ids_dtype = ids_dtype
        self.w_dtype = w_dtype
        self.padding_idx = padding_idx
        self.check_output_gcu_with_customized(self.forward, self.expect_output)


if __name__ == "__main__":
    unittest.main()
