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
ADD_N_CASE = [
    {"x_shape": [2, 3], "x_dtype": np.float32, "x_num": 1},
    {"x_shape": [2, 3], "x_dtype": np.float32, "x_num": 2},
    {"x_shape": [2, 3], "x_dtype": np.float32, "x_num": 3},
    {"x_shape": [2, 3], "x_dtype": np.float32, "x_num": 6},

    {"x_shape": [3, 2, 3], "x_dtype": np.float32, "x_num": 1},
    {"x_shape": [3, 2, 3], "x_dtype": np.float32, "x_num": 2},
    {"x_shape": [3, 2, 3], "x_dtype": np.float32, "x_num": 3},
    {"x_shape": [3, 2, 3], "x_dtype": np.float32, "x_num": 6},

    {"x_shape": [2, 3], "x_dtype": np.float16, "x_num": 1},
    {"x_shape": [2, 3], "x_dtype": np.float16, "x_num": 2},
    {"x_shape": [2, 3], "x_dtype": np.float16, "x_num": 3},
    {"x_shape": [2, 3], "x_dtype": np.float16, "x_num": 6},

    {"x_shape": [3, 2, 3], "x_dtype": np.float16, "x_num": 1},
    {"x_shape": [3, 2, 3], "x_dtype": np.float16, "x_num": 2},
    {"x_shape": [3, 2, 3], "x_dtype": np.float16, "x_num": 3},
    {"x_shape": [3, 2, 3], "x_dtype": np.float16, "x_num": 6},
]
# fmt: on


@ddt
class TestAddN(TestAPIBase):
    def setUp(self):
        self.init_attrs()

    def init_attrs(self):
        self.x_shape = [2, 3]
        self.x_dtype = np.float32
        self.x_num = 1

    def prepare_data(self):
        self.data_x = [
            self.generate_data(self.x_shape, self.x_dtype) for _ in range(self.x_num)
        ]

    def forward(self):
        x = [
            paddle.to_tensor(self.data_x[i], dtype=self.x_dtype)
            for i in range(self.x_num)
        ]
        return paddle.add_n(x)

    def get_numpy_output(self):
        out = np.zeros(self.x_shape, self.x_dtype)
        for i in range(self.x_num):
            out = out + self.data_x[i]
        return out

    def expect_output(self):
        if self.x_dtype != np.float16:
            out = self.calc_result(self.forward, "cpu")
        else:
            out = self.get_numpy_output()
        return out

    @data(*ADD_N_CASE)
    @unpack
    def test_check_output(self, x_shape, x_dtype, x_num):
        self.x_shape = x_shape
        self.x_dtype = x_dtype
        self.x_num = x_num
        self.check_output_gcu_with_customized(self.forward, self.expect_output)


if __name__ == "__main__":
    unittest.main()
