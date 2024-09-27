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
UNBIND_CASE = [
    {"x_shape": [5, 6, 7], "x_dtype": np.float32, "axis": 0, "num": None},
    {"x_shape": [5, 6, 7], "x_dtype": np.float32, "axis": 1, "num": None},
    {"x_shape": [5, 6, 7], "x_dtype": np.float32, "axis": 2, "num": None},
    {"x_shape": [5, 6, 7], "x_dtype": np.float32, "axis": -1, "num": None},
    {"x_shape": [5, 6, 7], "x_dtype": np.float32, "axis": -2, "num": None},

    {"x_shape": [5, 6, 7], "x_dtype": np.int32, "axis": 0, "num": None},
    {"x_shape": [5, 6, 7], "x_dtype": np.int32, "axis": 1, "num": None},
    {"x_shape": [5, 6, 7], "x_dtype": np.int32, "axis": 2, "num": None},
    {"x_shape": [5, 6, 7], "x_dtype": np.int32, "axis": -1, "num": None},
    {"x_shape": [5, 6, 7], "x_dtype": np.int32, "axis": -2, "num": None},
]
# fmt: on


@ddt
class TestUnbind(TestAPIBase):
    def setUp(self):
        self.init_attrs()

    def init_attrs(self):
        self.x_shape = [3, 6]
        self.x_dtype = np.float32
        self.num = None
        self.axis = 0

    def prepare_data(self):
        if self.x_dtype == np.int64:
            self.data = self.generate_integer_data(
                self.x_shape, self.x_dtype, -32768, 32767
            )
        elif self.x_dtype == np.float64:
            self.data = self.generate_float_data(self.x_shape, np.float32).astype(
                self.x_dtype
            )
        else:
            self.data = self.generate_data(self.x_shape, self.x_dtype)

    def forward(self):
        x = paddle.to_tensor(self.data, dtype=self.x_dtype)
        return paddle.unbind(x, self.axis)

    def expect_output(self):
        if self.x_dtype != np.float16:
            out = self.calc_result(self.forward, "cpu")
        else:
            out = np.unbind(self.data, axis=self.axis)
        return out

    @data(*UNBIND_CASE)
    @unpack
    def test_check_output(self, x_shape, x_dtype, axis, num):
        self.x_shape = x_shape
        self.x_dtype = x_dtype
        self.axis = axis
        self.num = num
        self.check_output_gcu_with_cpu(self.forward)


if __name__ == "__main__":
    unittest.main()
