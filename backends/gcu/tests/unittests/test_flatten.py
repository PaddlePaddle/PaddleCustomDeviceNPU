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
FLATTEN_CASE = [
    # float32
    {"x_shape": [6], "x_dtype": np.float32, "start_axis": 0, "stop_axis": -1},
    {"x_shape": [3, 6], "x_dtype": np.float32, "start_axis": 0, "stop_axis": -1},
    {"x_shape": [2, 3, 6], "x_dtype": np.float32, "start_axis": 1, "stop_axis": 2},
    {"x_shape": [3, 2, 3, 6], "x_dtype": np.float32, "start_axis": 0, "stop_axis": 2},

    # int32
    {"x_shape": [6], "x_dtype": np.int32, "start_axis": 0, "stop_axis": -1},
    {"x_shape": [3, 6], "x_dtype": np.int32, "start_axis": 0, "stop_axis": -1},
    {"x_shape": [2, 3, 6], "x_dtype": np.int32, "start_axis": 1, "stop_axis": 2},
    {"x_shape": [3, 2, 3, 6], "x_dtype": np.int32, "start_axis": 0, "stop_axis": 2},

    # float16
    {"x_shape": [6], "x_dtype": np.float16, "start_axis": 0, "stop_axis": -1},
    {"x_shape": [3, 6], "x_dtype": np.float16, "start_axis": 0, "stop_axis": -1},
    {"x_shape": [2, 3, 6], "x_dtype": np.float16, "start_axis": 1, "stop_axis": 2},
    {"x_shape": [3, 2, 3, 6], "x_dtype": np.float16, "start_axis": 0, "stop_axis": 2},

]
# fmt: on


@ddt
class TestFlatten(TestAPIBase):
    def setUp(self):
        self.init_attrs()

    def init_attrs(self):
        self.x_shape = [2, 4]
        self.x_dtype = np.float32
        self.start_axis = 0
        self.stop_axis = -1

    def prepare_data(self):
        self.data_x = self.generate_data(self.x_shape, self.x_dtype)

    def forward_with_dtype(self, dtype):
        x = paddle.to_tensor(self.data_x, dtype=dtype)
        return paddle.flatten(x, self.start_axis, self.stop_axis)

    def forward(self):
        return self.forward_with_dtype(self.x_dtype)

    def flatten_cast(self):
        return self.forward_with_dtype(np.float32).astype("float16")

    def expect_output(self):
        if self.x_dtype != np.float16:
            out = self.forward()
        else:
            out = self.flatten_cast()
        return out

    @data(*FLATTEN_CASE)
    @unpack
    def test_check_output(self, x_shape, x_dtype, start_axis, stop_axis):
        self.x_shape = x_shape
        self.x_dtype = x_dtype
        self.start_axis = start_axis
        self.stop_axis = stop_axis
        self.check_output_gcu_with_customized(self.forward, self.expect_output)


if __name__ == "__main__":
    unittest.main()
