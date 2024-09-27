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
import unittest
from ddt import ddt, data, unpack
from api_base import TestAPIBase


# The table retains its original format for better comparison of parameter settings.
# fmt: off
UNIFORM_RANDOM_CASE = [
    {"shape": [3, 2], "dtype": "float32", "min": -1.0, "max": 1.0, "seed": 0},
    {"shape": [3, 2], "dtype": "float32", "min": -1.0, "max": 6.0, "seed": 0},
    {"shape": [3, 2], "dtype": "float32", "min": -6.0, "max": 1.0, "seed": 0},
    {"shape": [3, 2], "dtype": "float32", "min": 3.0, "max": 6.0, "seed": 0},
    {"shape": [3, 2], "dtype": "float32", "min": -1.0, "max": 1.0, "seed": 100},
    {"shape": [], "dtype": "float32", "min": -1.0, "max": 1.0, "seed": 0},
    {"shape": [2], "dtype": "float32", "min": -1.0, "max": 1.0, "seed": 0},
    {"shape": [2, 3, 2], "dtype": "float32", "min": -1.0, "max": 1.0, "seed": 0},
    {"shape": [2, 2, 3, 2], "dtype": "float32", "min": -1.0, "max": 1.0, "seed": 0},
    {"shape": [3, 2], "dtype": "float64", "min": -1.0, "max": 1.0, "seed": 0},
    {"shape": [3, 2], "dtype": "float64", "min": -1.0, "max": 6.0, "seed": 0},
    {"shape": [3, 2], "dtype": "float64", "min": -6.0, "max": 1.0, "seed": 0},
    {"shape": [3, 2], "dtype": "float64", "min": 3.0, "max": 6.0, "seed": 0},
    {"shape": [3, 2], "dtype": "float64", "min": -1.0, "max": 1.0, "seed": 100},
    {"shape": [], "dtype": "float64", "min": -1.0, "max": 1.0, "seed": 0},
    {"shape": [2], "dtype": "float64", "min": -1.0, "max": 1.0, "seed": 0},
    {"shape": [2, 3, 2], "dtype": "float64", "min": -1.0, "max": 1.0, "seed": 0},
    {"shape": [2, 2, 3, 2], "dtype": "float64", "min": -1.0, "max": 1.0, "seed": 0},
    {"shape": [3, 2], "dtype": "float16", "min": -1.0, "max": 1.0, "seed": 0},
    {"shape": [3, 2], "dtype": "float16", "min": -1.0, "max": 6.0, "seed": 0},
    {"shape": [3, 2], "dtype": "float16", "min": -6.0, "max": 1.0, "seed": 0},
    {"shape": [3, 2], "dtype": "float16", "min": 3.0, "max": 6.0, "seed": 0},
    {"shape": [3, 2], "dtype": "float16", "min": -1.0, "max": 1.0, "seed": 100},
    {"shape": [], "dtype": "float16", "min": -1.0, "max": 1.0, "seed": 0},
    {"shape": [2], "dtype": "float16", "min": -1.0, "max": 1.0, "seed": 0},
    {"shape": [2, 3, 2], "dtype": "float16", "min": -1.0, "max": 1.0, "seed": 0},
    {"shape": [2, 2, 3, 2], "dtype": "float16", "min": -1.0, "max": 1.0, "seed": 0},
]
# fmt: on


@ddt
class TestUniform(TestAPIBase):
    def setUp(self):
        self.init_attrs()

    def init_attrs(self):
        self.shape = [3, 2]
        self.dtype = "float32"
        self.min = -1.0
        self.max = 1.0
        self.seed = 0

    def prepare_data(self):
        pass

    def forward(self):
        paddle.seed(2035)
        return paddle.uniform(self.shape, self.dtype, self.min, self.max, self.seed)

    def uniform_cast(self):
        paddle.seed(2035)
        out = paddle.uniform(self.shape, "float32", self.min, self.max, self.seed)
        out = out.astype("float16")
        return out

    def expect_output(self):
        if self.dtype != "float16":
            out = self.calc_result(self.forward, "cpu")
        else:
            out = self.uniform_cast()
        return out

    @data(*UNIFORM_RANDOM_CASE)
    @unpack
    def test_check_output(self, shape, dtype, min, max, seed):
        self.shape = shape
        self.dtype = dtype
        self.min = min
        self.max = max
        self.seed = seed
        # self.check_output_gcu_with_customized(self.forward, self.expect_output)


if __name__ == "__main__":
    unittest.main()
