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
TRANSPOSE_CASE = [
    {"x_shape": [2, 3, 32, 64], "perm": [0, 2, 3, 1], "x_dtype": np.float32},
    {"x_shape": [2, 3, 32, 64], "perm": [0, 2, 3, 1], "x_dtype": np.float64},
    {"x_shape": [2, 3, 32, 48], "perm": [0, 2, 1, 3], "x_dtype": np.int32},
    {"x_shape": [2, 3, 32, 48], "perm": [0, 2, 1, 3], "x_dtype": np.int64},
    {"x_shape": [2, 4, 16], "perm": [0, 2, 1], "x_dtype": np.float32},
    {"x_shape": [2, 4, 16], "perm": [0, 2, 1], "x_dtype": np.float64},
    {"x_shape": [2, 4, 32], "perm": [1, 2, 0], "x_dtype": np.int32},
    {"x_shape": [2, 4, 32], "perm": [1, 2, 0], "x_dtype": np.int64},
    {"x_shape": [4, 48], "perm": [1, 0], "x_dtype": np.float32},
    {"x_shape": [4, 48], "perm": [1, 0], "x_dtype": np.float64},
    {"x_shape": [4, 64], "perm": [1, 0], "x_dtype": np.int32},
    {"x_shape": [4, 64], "perm": [1, 0], "x_dtype": np.int64},
    {"x_shape": [4, 48, 16, 32, 64], "perm": [1, 0, 3, 4, 2], "x_dtype": np.float32},
    {"x_shape": [4, 48, 16, 32, 64], "perm": [1, 0, 3, 4, 2], "x_dtype": np.float64},
    {"x_shape": [4, 48, 16, 32, 64], "perm": [1, 0, 3, 4, 2], "x_dtype": np.int32},
    {"x_shape": [4, 48, 16, 32, 64], "perm": [1, 0, 3, 4, 2], "x_dtype": np.int64},

    {"x_shape": [2, 3, 32, 72], "perm": [1, 2, 3, 0], "x_dtype": np.float16},
    {"x_shape": [2, 4, 16], "perm": [0, 2, 1], "x_dtype": np.float16},
    {"x_shape": [4, 48], "perm": [1, 0], "x_dtype": np.float16},
    {"x_shape": [4, 48, 16, 32, 64], "perm": [1, 0, 3, 4, 2], "x_dtype": np.float16},

    {"x_shape": [2, 3, 32, 72], "perm": [1, 2, 3, 0], "x_dtype": np.uint8},
    {"x_shape": [2, 3, 32, 48], "perm": [0, 2, 1, 3], "x_dtype": np.int8},
    {"x_shape": [2, 4, 16], "perm": [0, 2, 1], "x_dtype": np.uint8},
    {"x_shape": [2, 4, 32], "perm": [1, 2, 0], "x_dtype": np.int8},
    {"x_shape": [4, 48], "perm": [1, 0], "x_dtype": np.uint8},
    {"x_shape": [4, 64], "perm": [1, 0], "x_dtype": np.int8},
    {"x_shape": [4, 48, 16, 32, 64], "perm": [1, 0, 3, 4, 2], "x_dtype": np.uint8},
    {"x_shape": [4, 48, 16, 32, 64], "perm": [1, 0, 3, 4, 2], "x_dtype": np.int8},
]
# fmt: on


@ddt
class TestTranspose(TestAPIBase):
    def setUp(self):
        self.init_attrs()

    def init_attrs(self):
        self.x_shape = [2, 3, 32, 64]
        self.perm = [0, 2, 3, 1]
        self.x_dtype = np.float32

    def prepare_data(self):
        self.data_x = self.generate_data(self.x_shape, self.x_dtype)

    def forward(self):
        x = paddle.to_tensor(self.data_x, dtype=self.x_dtype)
        return paddle.transpose(x, perm=self.perm)

    def expect_output(self):
        if self.x_dtype != np.float16:
            out = self.calc_result(self.forward, "cpu")
        else:
            out = paddle.to_tensor(self.data_x.transpose(self.perm))
        return out

    @data(*TRANSPOSE_CASE)
    @unpack
    def test_check_output(self, x_shape, perm, x_dtype):
        self.x_shape = x_shape
        self.perm = perm
        self.x_dtype = x_dtype
        self.check_output_gcu_with_customized(self.forward, self.expect_output)


if __name__ == "__main__":
    unittest.main()
