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
MULTINOMIAL_CASE = [
    {"x_shape": [3, 16], "x_dtype": np.float32, "num_samples": 1, "replacement": False},
    {"x_shape": [3, 16], "x_dtype": np.float32, "num_samples": 1, "replacement": True},
    {"x_shape": [3, 16], "x_dtype": np.float32, "num_samples": 5, "replacement": False},
    {"x_shape": [3, 16], "x_dtype": np.float32, "num_samples": 5, "replacement": True},
    {"x_shape": [3, 16], "x_dtype": np.float32, "num_samples": 3, "replacement": False},
    {"x_shape": [3, 16], "x_dtype": np.float32, "num_samples": 3, "replacement": True},
    # float64 not support yet
    # {"x_shape": [3, 16], "x_dtype": np.float64, "num_samples": 1, "replacement": False},
    # {"x_shape": [3, 16], "x_dtype": np.float64, "num_samples": 1, "replacement": True},
    # {"x_shape": [3, 16], "x_dtype": np.float64, "num_samples": 5, "replacement": False},
    # {"x_shape": [3, 16], "x_dtype": np.float64, "num_samples": 5, "replacement": True},
    # {"x_shape": [3, 16], "x_dtype": np.float64, "num_samples": 3, "replacement": False},
    # {"x_shape": [3, 16], "x_dtype": np.float64, "num_samples": 3, "replacement": True},
]
# fmt: on


@ddt
class TestMultinomial(TestAPIBase):
    def setUp(self):
        self.init_attrs()

    def init_attrs(self):
        self.x_shape = [2, 6]
        self.x_dtype = np.float32
        self.num_samples = 1
        self.replacement = False

    def prepare_data(self):
        self.data_x = self.generate_data(self.x_shape, self.x_dtype)

    def forward(self):
        paddle.seed(200 + self.num_samples)
        x = paddle.to_tensor(self.data_x, dtype=self.x_dtype)
        # return paddle.multinomial(x, self.num_samples, self.replacement)
        return paddle.multinomial(x, self.num_samples, self.replacement).shape

    @data(*MULTINOMIAL_CASE)
    @unpack
    def test_check_output(self, x_shape, x_dtype, num_samples, replacement):
        self.x_shape = x_shape
        self.x_dtype = x_dtype
        self.num_samples = num_samples
        self.replacement = replacement
        self.check_output_gcu_with_cpu(self.forward)


if __name__ == "__main__":
    unittest.main()
