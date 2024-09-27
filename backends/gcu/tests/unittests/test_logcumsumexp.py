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
LOG_CUMSUM_EXP_CASE = [

    {"x_shape": [2, 6], "x_dtype": np.float32, "axis": None, "dtype": np.float32},
    {"x_shape": [2, 6], "x_dtype": np.float32, "axis": 0, "dtype": np.float32},
    {"x_shape": [2, 6], "x_dtype": np.float32, "axis": -1, "dtype": np.float32},
    {"x_shape": [2, 6], "x_dtype": np.float32, "axis": None, "dtype": np.float16},
    {"x_shape": [2, 6], "x_dtype": np.float32, "axis": 0, "dtype": np.float16},
    {"x_shape": [2, 6], "x_dtype": np.float32, "axis": -1, "dtype": np.float16},
    {"x_shape": [2, 3, 4], "x_dtype": np.float32, "axis": None, "dtype": np.float32},
    {"x_shape": [2, 3, 4], "x_dtype": np.float32, "axis": 0, "dtype": np.float32},
    {"x_shape": [2, 3, 4], "x_dtype": np.float32, "axis": 1, "dtype": np.float32},
    {"x_shape": [2, 3, 4], "x_dtype": np.float32, "axis": 2, "dtype": np.float32},
    {"x_shape": [2, 3, 4], "x_dtype": np.float32, "axis": -2, "dtype": np.float32},
    {"x_shape": [2, 3, 4], "x_dtype": np.float32, "axis": None, "dtype": np.float16},
    {"x_shape": [2, 3, 4], "x_dtype": np.float32, "axis": 0, "dtype": np.float16},
    {"x_shape": [2, 3, 4], "x_dtype": np.float32, "axis": 1, "dtype": np.float16},
    {"x_shape": [2, 3, 4], "x_dtype": np.float32, "axis": 2, "dtype": np.float16},
    {"x_shape": [2, 3, 4], "x_dtype": np.float32, "axis": -2, "dtype": np.float16},


    {"x_shape": [2, 6], "x_dtype": np.float16, "axis": None, "dtype": np.float16},
    {"x_shape": [2, 6], "x_dtype": np.float16, "axis": 0, "dtype": np.float16},
    {"x_shape": [2, 6], "x_dtype": np.float16, "axis": -1, "dtype": np.float16},
    {"x_shape": [2, 6], "x_dtype": np.float16, "axis": None, "dtype": np.float32},
    {"x_shape": [2, 6], "x_dtype": np.float16, "axis": 0, "dtype": np.float32},
    {"x_shape": [2, 6], "x_dtype": np.float16, "axis": -1, "dtype": np.float32},
    {"x_shape": [2, 3, 4], "x_dtype": np.float16, "axis": None, "dtype": np.float16},
    {"x_shape": [2, 3, 4], "x_dtype": np.float16, "axis": 0, "dtype": np.float16},
    {"x_shape": [2, 3, 4], "x_dtype": np.float16, "axis": 1, "dtype": np.float16},
    {"x_shape": [2, 3, 4], "x_dtype": np.float16, "axis": 2, "dtype": np.float16},
    {"x_shape": [2, 3, 4], "x_dtype": np.float16, "axis": -2, "dtype": np.float16},
    {"x_shape": [2, 3, 4], "x_dtype": np.float16, "axis": None, "dtype": np.float32},
    {"x_shape": [2, 3, 4], "x_dtype": np.float16, "axis": 0, "dtype": np.float32},
    {"x_shape": [2, 3, 4], "x_dtype": np.float16, "axis": 1, "dtype": np.float32},
    {"x_shape": [2, 3, 4], "x_dtype": np.float16, "axis": 2, "dtype": np.float32},
    {"x_shape": [2, 3, 4], "x_dtype": np.float16, "axis": -2, "dtype": np.float32},

]
# fmt: on


@ddt
class TestLogCumsumumExp(TestAPIBase):
    def setUp(self):
        self.init_attrs()

    def init_attrs(self):
        self.x_shape = [2, 6]
        self.x_dtype = np.float32
        self.axis = None
        self.dtype = np.float32

    def prepare_data(self):
        self.data_x = self.generate_data(self.x_shape, self.x_dtype)

    def forward(self):
        x = paddle.to_tensor(self.data_x, dtype=self.x_dtype)
        return paddle.logcumsumexp(x, axis=self.axis, dtype=self.dtype)

    def get_numpy_output(self):
        return np.log(np.cumsum(np.exp(self.data_x), axis=self.axis))

    def expect_output(self):
        if self.x_dtype != np.float16 and self.dtype != np.float16:
            out = self.calc_result(self.forward, "cpu")
        else:
            out = self.get_numpy_output()
        return out

    @data(*LOG_CUMSUM_EXP_CASE)
    @unpack
    def test_check_output(self, x_shape, x_dtype, axis, dtype):
        self.x_shape = x_shape
        self.x_dtype = x_dtype
        self.axis = axis
        self.dtype = dtype
        rtol = 1e-5
        atol = 1e-5
        if x_dtype == np.float16 or dtype == np.float16:
            rtol = 1e-3
            atol = 1e-3
        self.check_output_gcu_with_customized(
            self.forward, self.expect_output, rtol=rtol, atol=atol
        )


if __name__ == "__main__":
    unittest.main()
