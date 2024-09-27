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
ROLL_CASE = [
    {"shape": [3, 3], "dtype": np.int32, "shifts": 1, "axis": 1},
    {"shape": [3, 3], "dtype": np.int32, "shifts": -1, "axis": 0},
    {"shape": [3, 3], "dtype": np.int32, "shifts": 1, "axis": None},

    {"shape": [16, 3, 28, 28], "dtype": np.float32, "shifts": [1, 2, 3, 4], "axis": [0, 1, 2, 3]},
    {"shape": [16, 3, 28, 28], "dtype": np.float32, "shifts": [2, 3, 4], "axis": [1, 2, 3]},
    {"shape": [16, 3, 28, 28], "dtype": np.float32, "shifts": [10, 15], "axis": [2, 3]},
    {"shape": [16, 3, 28, 28], "dtype": np.float32, "shifts": [10, 15], "axis": [2, -1]},
    {"shape": [16, 3, 28, 28], "dtype": np.float32, "shifts": [10, 15], "axis": [-2, -1]},
    {"shape": [16, 3, 28, 28], "dtype": np.float32, "shifts": [30], "axis": None},

    {"shape": [16, 3, 28, 28], "dtype": np.float16, "shifts": [1, 2, 3, 4], "axis": [0, 1, 2, 3]},
    {"shape": [16, 3, 28, 28], "dtype": np.float16, "shifts": [2, 3, 4], "axis": [1, 2, 3]},
    {"shape": [16, 3, 28, 28], "dtype": np.float16, "shifts": [10, 15], "axis": [2, 3]},
    {"shape": [16, 3, 28, 28], "dtype": np.float16, "shifts": [10, 15], "axis": [2, -1]},
    {"shape": [16, 3, 28, 28], "dtype": np.float16, "shifts": [10, 15], "axis": [-2, -1]},
    {"shape": [16, 3, 28, 28], "dtype": np.float16, "shifts": [30], "axis": None},

]
# fmt: on


@ddt
class TestRoll(TestAPIBase):
    def setUp(self):
        self.init_attrs()

    def init_attrs(self):
        self.shape = [2, 3]
        self.dtype = np.float32
        self.shifts = 1
        self.axis = 1

    def prepare_data(self):
        self.data_x = self.generate_data(self.shape, self.dtype)

    def forward(self):
        x = paddle.to_tensor(self.data_x, dtype=self.dtype)
        return paddle.roll(x, self.shifts, self.axis)

    def get_numpy_output(self):
        return np.roll(self.data_x, self.shifts, self.axis)

    def expect_output(self):
        if self.dtype != np.float16:
            out = self.calc_result(self.forward, "cpu")
        else:
            out = self.get_numpy_output()
        return out

    @data(*ROLL_CASE)
    @unpack
    def test_check_output(self, shape, dtype, shifts, axis):
        self.shape = shape
        self.dtype = dtype
        self.shifts = shifts
        self.axis = axis
        self.check_output_gcu_with_customized(self.forward, self.expect_output)


if __name__ == "__main__":
    unittest.main()
