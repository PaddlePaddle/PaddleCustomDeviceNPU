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
SCALE_CASE = [
    {"x_shape": [2, 6], "x_dtype": np.float32, "scale": 2.0, "bias": 1.0, "bias_after_scale": False},
    {"x_shape": [2, 6], "x_dtype": np.float32, "scale": 2.0, "bias": 1.0, "bias_after_scale": True},
    {"x_shape": [2, 2, 6], "x_dtype": np.float32, "scale": 2.0, "bias": 1.0, "bias_after_scale": False},
    {"x_shape": [2, 6], "x_dtype": np.float64, "scale": 2.0, "bias": 1.0, "bias_after_scale": False},
    {"x_shape": [2, 6], "x_dtype": np.float64, "scale": 2.0, "bias": 1.0, "bias_after_scale": True},
    {"x_shape": [2, 2, 6], "x_dtype": np.float64, "scale": 2.0, "bias": 1.0, "bias_after_scale": False},
    {"x_shape": [2, 6], "x_dtype": np.int32, "scale": 2.0, "bias": 1.0, "bias_after_scale": False},
    {"x_shape": [2, 6], "x_dtype": np.int32, "scale": 2.0, "bias": 1.0, "bias_after_scale": True},
    {"x_shape": [2, 2, 6], "x_dtype": np.int32, "scale": 2.0, "bias": 1.0, "bias_after_scale": False},
    {"x_shape": [2, 6], "x_dtype": np.int64, "scale": 2.0, "bias": 1.0, "bias_after_scale": False},
    {"x_shape": [2, 6], "x_dtype": np.int64, "scale": 2.0, "bias": 1.0, "bias_after_scale": True},
    {"x_shape": [2, 2, 6], "x_dtype": np.int64, "scale": 2.0, "bias": 1.0, "bias_after_scale": False},
    {"x_shape": [2, 6], "x_dtype": np.float16, "scale": 2.0, "bias": 1.0, "bias_after_scale": False},
    {"x_shape": [2, 6], "x_dtype": np.float16, "scale": 2.0, "bias": 1.0, "bias_after_scale": True},
    {"x_shape": [2, 2, 6], "x_dtype": np.float16, "scale": 2.0, "bias": 1.0, "bias_after_scale": False},

]
# fmt: on


@ddt
class TestScale(TestAPIBase):
    def setUp(self):
        self.init_attrs()

    def init_attrs(self):
        self.x_shape = [2, 6]
        self.x_dtype = np.float32
        self.scale = 1.0
        self.bias = 0.0
        self.bias_after_scale = True

    def prepare_data(self):
        self.data_x = np.random.uniform(-10, 10, self.x_shape).astype(self.x_dtype)

    def forward(self):
        x = paddle.to_tensor(self.data_x, dtype=self.x_dtype)
        return paddle.scale(x, self.scale, self.bias, self.bias_after_scale)

    def expect_output(self):
        if self.x_dtype != np.float16:
            out = self.calc_result(self.forward, "cpu")
        else:
            if self.bias_after_scale:
                out = self.data_x * self.x_dtype(self.scale) + self.x_dtype(self.bias)
            else:
                out = (self.data_x + self.x_dtype(self.bias)) * self.x_dtype(self.scale)
        return out

    @data(*SCALE_CASE)
    @unpack
    def test_check_output(self, x_shape, x_dtype, scale, bias, bias_after_scale):
        self.x_shape = x_shape
        self.x_dtype = x_dtype
        self.scale = scale
        self.bias = bias
        self.bias_after_scale = bias_after_scale
        self.check_output_gcu_with_customized(self.forward, self.expect_output)


if __name__ == "__main__":
    unittest.main()
