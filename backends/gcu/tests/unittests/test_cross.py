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
CROSS_CASE = [
    {"shape": [3, 3], "dtype": np.float32, "axis": None},
    {"shape": [3, 3], "dtype": np.float32, "axis": 0},
    {"shape": [3, 3], "dtype": np.float32, "axis": 1},

    {"shape": [3, 3], "dtype": np.int32, "axis": None},
    {"shape": [3, 3], "dtype": np.int32, "axis": 0},
    {"shape": [3, 3], "dtype": np.int32, "axis": 1},

    {"shape": [3, 3], "dtype": np.float16, "axis": None},
    {"shape": [3, 3], "dtype": np.float16, "axis": 0},
    {"shape": [3, 3], "dtype": np.float16, "axis": 1},

    {"shape": [16, 3, 3], "dtype": np.float32, "axis": None},
    {"shape": [16, 3, 3], "dtype": np.float32, "axis": 1},
    {"shape": [16, 3, 3], "dtype": np.float32, "axis": 2},
    {"shape": [2, 16, 3, 3], "dtype": np.float32, "axis": None},
    {"shape": [2, 16, 3, 3], "dtype": np.float32, "axis": 2},
    {"shape": [2, 16, 3, 3], "dtype": np.float32, "axis": 3},

    {"shape": [16, 3, 3], "dtype": np.float16, "axis": None},
    {"shape": [16, 3, 3], "dtype": np.float16, "axis": 1},
    {"shape": [16, 3, 3], "dtype": np.float16, "axis": 2},
    {"shape": [2, 16, 3, 3], "dtype": np.float16, "axis": None},
    {"shape": [2, 16, 3, 3], "dtype": np.float16, "axis": 2},
    {"shape": [2, 16, 3, 3], "dtype": np.float16, "axis": 3},

]
# fmt: on


@ddt
class TestCross(TestAPIBase):
    def setUp(self):
        self.init_attrs()

    def init_attrs(self):
        self.shape = [3, 3]
        self.dtype = np.float32
        self.axis = None

    def prepare_data(self):
        self.data_x = self.generate_data(self.shape, self.dtype)
        self.data_y = self.generate_data(self.shape, self.dtype)

    def forward_with_dtype(self, dtype):
        x = paddle.to_tensor(self.data_x, dtype=dtype)
        y = paddle.to_tensor(self.data_y, dtype=dtype)
        return paddle.cross(x, y, self.axis)

    def forward(self):
        return self.forward_with_dtype(self.dtype)

    def cross_cast(self):
        return self.forward_with_dtype(np.float32).astype("float16")

    def expect_output(self):
        if self.dtype != np.float16:
            out = self.forward()
        else:
            out = self.cross_cast()
        return out

    @data(*CROSS_CASE)
    @unpack
    def test_check_output(self, shape, dtype, axis):
        self.shape = shape
        self.dtype = dtype
        self.axis = axis
        rtol = 1e-5
        atol = 1e-5
        if dtype == np.float16:
            rtol = 1e-3
            atol = 1e-3
        self.check_output_gcu_with_customized(
            self.forward, self.expect_output, rtol=rtol, atol=atol
        )


if __name__ == "__main__":
    unittest.main()
