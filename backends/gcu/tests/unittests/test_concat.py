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
CONCAT_CASE = [
    {"x_shape": [2, 3, 224, 224], "y_shape": [2, 4, 224, 224], "dtype": np.float32, "axis": 1},
    {"x_shape": [2, 3, 224, 224], "y_shape": [2, 4, 224, 224], "dtype": np.float16, "axis": 1},
    {"x_shape": [2, 3, 224, 224], "y_shape": [2, 4, 224, 224], "dtype": np.int32, "axis": 1},
    {"x_shape": [2, 3, 224, 224], "y_shape": [2, 4, 224, 224], "dtype": np.int64, "axis": 1},
    {"x_shape": [2, 3, 224, 224], "y_shape": [2, 4, 224, 224], "dtype": np.uint8, "axis": 1},
    {"x_shape": [2, 3, 224, 224], "y_shape": [2, 4, 224, 224], "dtype": bool, "axis": 1},
    {"x_shape": [3, 224, 224], "y_shape": [4, 224, 224], "dtype": np.float32, "axis": 0},
    {"x_shape": [224, 224, 3], "y_shape": [224, 224, 4], "dtype": np.float32, "axis": -1},
    {"x_shape": [3, 224], "y_shape": [4, 224], "dtype": np.float32, "axis": 0},
    {"x_shape": [224, 224, 3], "y_shape": [224, 224, 4], "dtype": np.int64, "axis": -1},
]
# fmt: on


@ddt
class TestConcat(TestAPIBase):
    def setUp(self):
        self.init_attrs()

    def init_attrs(self):
        self.x_shape = [2, 3, 224, 224]
        self.y_shape = [2, 4, 224, 224]
        self.axis = 1
        self.dtype = np.float32

    def prepare_data(self):
        self.data_x = self.generate_data(self.x_shape, self.dtype)
        self.data_y = self.generate_data(self.y_shape, self.dtype)

    def forward(self):
        x = paddle.to_tensor(self.data_x, dtype=self.dtype)
        y = paddle.to_tensor(self.data_x, dtype=self.dtype)
        return paddle.concat([x, y], self.axis)

    @data(*CONCAT_CASE)
    @unpack
    def test_check_output(self, x_shape, y_shape, dtype, axis):
        self.x_shape = x_shape
        self.y_shape = y_shape
        self.dtype = dtype
        self.axis = axis
        self.check_output_gcu_with_cpu(self.forward)


if __name__ == "__main__":
    unittest.main()
