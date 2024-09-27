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
CLIP_CASE = [
    {"x_shape": [2, 3], "x_dtype": np.int32, "min": 0, "max": 1},
    {"x_shape": [2, 3], "x_dtype": np.int32, "min": 2, "max": 5},
    {"x_shape": [2, 3], "x_dtype": np.float32, "min": 0.1, "max": 1.1},
    {"x_shape": [2, 3], "x_dtype": np.float32, "min": 2.5, "max": 6.5},
    {"x_shape": [2, 3], "x_dtype": np.float16, "min": 0.1, "max": 1.1},
    {"x_shape": [2, 3], "x_dtype": np.float16, "min": 2.5, "max": 6.5},

    {"x_shape": [2, 3, 28, 28], "x_dtype": np.int32, "min": 0, "max": 1},
    {"x_shape": [2, 3, 28, 28], "x_dtype": np.int32, "min": 2, "max": 5},
    {"x_shape": [2, 3, 28, 28], "x_dtype": np.float32, "min": 0.1, "max": 1.1},
    {"x_shape": [2, 3, 28, 28], "x_dtype": np.float32, "min": 2.5, "max": 6.5},
    {"x_shape": [2, 3, 28, 28], "x_dtype": np.float16, "min": 0.1, "max": 1.1},
    {"x_shape": [2, 3, 28, 28], "x_dtype": np.float16, "min": 2.5, "max": 6.5},
]
# fmt: on


@ddt
class TestClip(TestAPIBase):
    def setUp(self):
        self.init_attrs()

    def init_attrs(self):
        self.x_shape = [2, 3]
        self.x_dtype = np.float32
        self.min = 0
        self.max = 1

    def prepare_data(self):
        self.data_x = self.generate_data(self.x_shape, self.x_dtype)

    def forward(self):
        x = paddle.to_tensor(self.data_x, dtype=self.x_dtype)
        return paddle.clip(x, self.min, self.max)

    def get_numpy_output(self):
        return np.clip(self.data_x, self.min, self.max)

    def expect_output(self):
        if self.x_dtype != np.float16:
            out = self.calc_result(self.forward, "cpu")
        else:
            out = self.get_numpy_output()
        return out

    @data(*CLIP_CASE)
    @unpack
    def test_check_output(self, x_shape, x_dtype, min, max):
        self.x_shape = x_shape
        self.x_dtype = x_dtype
        self.min = min
        self.max = max
        self.check_output_gcu_with_customized(self.forward, self.expect_output)


if __name__ == "__main__":
    unittest.main()
