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
STACK_CASE = [
    {"x_shape": [3, 6], "x_dtype": np.float32, "num_inputs": 2, "axis": 0},
    {"x_shape": [5, 6, 7], "x_dtype": np.float32, "num_inputs": 4, "axis": 0},
    {"x_shape": [5, 6, 7], "x_dtype": np.float32, "num_inputs": 6, "axis": 0},
    {"x_shape": [5, 6, 7], "x_dtype": np.float32, "num_inputs": 6, "axis": -1},
    {"x_shape": [5, 6, 7], "x_dtype": np.float32, "num_inputs": 6, "axis": 1},
    {"x_shape": [5, 6, 7], "x_dtype": np.float32, "num_inputs": 6, "axis": 2},

    {"x_shape": [3, 6], "x_dtype": np.int32, "num_inputs": 2, "axis": 0},
    {"x_shape": [5, 6, 7], "x_dtype": np.int32, "num_inputs": 4, "axis": 0},
    {"x_shape": [5, 6, 7], "x_dtype": np.int32, "num_inputs": 6, "axis": 0},
    {"x_shape": [5, 6, 7], "x_dtype": np.int32, "num_inputs": 6, "axis": -1},
    {"x_shape": [5, 6, 7], "x_dtype": np.int32, "num_inputs": 6, "axis": 1},
    {"x_shape": [5, 6, 7], "x_dtype": np.int32, "num_inputs": 6, "axis": 2},

    {"x_shape": [3, 6], "x_dtype": np.int64, "num_inputs": 2, "axis": 0},
    {"x_shape": [5, 6, 7], "x_dtype": np.int64, "num_inputs": 4, "axis": 0},
    {"x_shape": [5, 6, 7], "x_dtype": np.int64, "num_inputs": 6, "axis": 0},
    {"x_shape": [5, 6, 7], "x_dtype": np.int64, "num_inputs": 6, "axis": -1},
    {"x_shape": [5, 6, 7], "x_dtype": np.int64, "num_inputs": 6, "axis": 1},
    {"x_shape": [5, 6, 7], "x_dtype": np.int64, "num_inputs": 6, "axis": 2},

    {"x_shape": [3, 6], "x_dtype": np.float64, "num_inputs": 2, "axis": 0},
    {"x_shape": [5, 6, 7], "x_dtype": np.float64, "num_inputs": 4, "axis": 0},
    {"x_shape": [5, 6, 7], "x_dtype": np.float64, "num_inputs": 6, "axis": 0},
    {"x_shape": [5, 6, 7], "x_dtype": np.float64, "num_inputs": 6, "axis": -1},
    {"x_shape": [5, 6, 7], "x_dtype": np.float64, "num_inputs": 6, "axis": 1},
    {"x_shape": [5, 6, 7], "x_dtype": np.float64, "num_inputs": 6, "axis": 2},

    {"x_shape": [3, 6], "x_dtype": np.float16, "num_inputs": 2, "axis": 0},
    {"x_shape": [5, 6, 7], "x_dtype": np.float16, "num_inputs": 4, "axis": 0},
    {"x_shape": [5, 6, 7], "x_dtype": np.float16, "num_inputs": 6, "axis": 0},
    {"x_shape": [5, 6, 7], "x_dtype": np.float16, "num_inputs": 6, "axis": -1},
    {"x_shape": [5, 6, 7], "x_dtype": np.float16, "num_inputs": 6, "axis": 1},
    {"x_shape": [5, 6, 7], "x_dtype": np.float16, "num_inputs": 6, "axis": 2},

]
# fmt: on


@ddt
class TestStack(TestAPIBase):
    def setUp(self):
        self.init_attrs()

    def init_attrs(self):
        self.x_shape = [3, 6]
        self.x_dtype = np.float32
        self.num_inputs = 1
        self.axis = 0

    def prepare_data(self):
        self.data_x = []
        for i in range(self.num_inputs):
            if self.x_dtype == np.int64:
                data = self.generate_integer_data(
                    self.x_shape, self.x_dtype, -32768, 32767
                )
            elif self.x_dtype == np.float64:
                data = self.generate_float_data(self.x_shape, np.float32).astype(
                    self.x_dtype
                )
            else:
                data = self.generate_data(self.x_shape, self.x_dtype)
            self.data_x.append(data)

    def forward(self):
        x = []
        for i in range(self.num_inputs):
            x.append(paddle.to_tensor(self.data_x[i], dtype=self.x_dtype))
        return paddle.stack(x, self.axis)

    def expect_output(self):
        if self.x_dtype != np.float16:
            out = self.calc_result(self.forward, "cpu")
        else:
            out = np.stack(self.data_x, axis=self.axis)
        return out

    @data(*STACK_CASE)
    @unpack
    def test_check_output(self, x_shape, x_dtype, num_inputs, axis):
        self.x_shape = x_shape
        self.x_dtype = x_dtype
        self.num_inputs = num_inputs
        self.axis = axis
        self.check_output_gcu_with_customized(self.forward, self.expect_output)


if __name__ == "__main__":
    unittest.main()
