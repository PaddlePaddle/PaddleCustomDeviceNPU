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
GAUSSIAN_RANDOM_CASE = [
    {"shape": [3, 2], "dtype": "float32", "mean": 0.0, "std": 1.0, "seed": 0},
    {"shape": [3, 2], "dtype": "float32", "mean": 1.0, "std": 1.0, "seed": 0},
    {"shape": [3, 2], "dtype": "float32", "mean": 0.0, "std": 2.0, "seed": 0},
    {"shape": [3, 2], "dtype": "float32", "mean": 0.0, "std": 1.0, "seed": 100},
    {"shape": [3, 2], "dtype": "float32", "mean": 1.0, "std": 2.0, "seed": 0},
    {"shape": [3, 2], "dtype": "float32", "mean": 1.0, "std": 2.0, "seed": 100},
    {"shape": [], "dtype": "float32", "mean": 0.0, "std": 1.0, "seed": 0},
    {"shape": [2], "dtype": "float32", "mean": 0.0, "std": 1.0, "seed": 0},
    {"shape": [2, 3, 2], "dtype": "float32", "mean": 0.0, "std": 1.0, "seed": 0},
    {"shape": [2, 2, 3, 2], "dtype": "float32", "mean": 0.0, "std": 1.0, "seed": 0},
    {"shape": [3, 2], "dtype": "float64", "mean": 0.0, "std": 1.0, "seed": 0},
    {"shape": [3, 2], "dtype": "float64", "mean": 1.0, "std": 1.0, "seed": 0},
    {"shape": [3, 2], "dtype": "float64", "mean": 0.0, "std": 2.0, "seed": 0},
    {"shape": [3, 2], "dtype": "float64", "mean": 0.0, "std": 1.0, "seed": 100},
    {"shape": [3, 2], "dtype": "float64", "mean": 1.0, "std": 2.0, "seed": 0},
    {"shape": [3, 2], "dtype": "float64", "mean": 1.0, "std": 2.0, "seed": 100},
    {"shape": [2, 3, 2], "dtype": "float64", "mean": 0.0, "std": 1.0, "seed": 0},
    {"shape": [2, 2, 3, 2], "dtype": "float64", "mean": 0.0, "std": 1.0, "seed": 0},
    {"shape": [3, 2], "dtype": "float16", "mean": 0.0, "std": 1.0, "seed": 0},
    {"shape": [3, 2], "dtype": "float16", "mean": 1.0, "std": 1.0, "seed": 0},
    {"shape": [3, 2], "dtype": "float16", "mean": 0.0, "std": 2.0, "seed": 0},
    {"shape": [3, 2], "dtype": "float16", "mean": 0.0, "std": 1.0, "seed": 100},
    {"shape": [3, 2], "dtype": "float16", "mean": 1.0, "std": 2.0, "seed": 0},
    {"shape": [3, 2], "dtype": "float16", "mean": 1.0, "std": 2.0, "seed": 100},
    {"shape": [2, 3, 2], "dtype": "float16", "mean": 0.0, "std": 1.0, "seed": 0},
    {"shape": [2, 2, 3, 2], "dtype": "float16", "mean": 0.0, "std": 1.0, "seed": 0},
]
# fmt: on


@ddt
class TestGaussian(TestAPIBase):
    def setUp(self):
        self.init_attrs()

    def init_attrs(self):
        self.shape = [3, 2]
        self.dtype = "float32"
        self.mean = 0.0
        self.std = 1.0
        self.seed = 0

    def prepare_data(self):
        pass

    def forward(self):
        paddle.seed(2035)
        return paddle.tensor.random.gaussian(
            self.shape, self.mean, self.std, self.seed, dtype=self.dtype
        )

    def gaussian_cast(self):
        paddle.seed(2035)
        out = paddle.tensor.random.gaussian(
            self.shape, self.mean, self.std, self.seed, dtype="float32"
        )
        out = out.astype("float16")
        return out

    def expect_output(self):
        if self.dtype != "float16":
            out = self.calc_result(self.forward, "cpu")
        else:
            out = self.gaussian_cast()
        return out

    @data(*GAUSSIAN_RANDOM_CASE)
    @unpack
    def test_check_output(self, shape, dtype, mean, std, seed):
        self.shape = shape
        self.dtype = dtype
        self.mean = mean
        self.std = std
        self.seed = seed
        # self.check_output_gcu_with_customized(self.forward, self.expect_output)


if __name__ == "__main__":
    unittest.main()
