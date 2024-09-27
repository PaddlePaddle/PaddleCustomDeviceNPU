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
TAKE_ALONG_AXIS_CASE = [
    {"x_shape": [2, 2], "x_dtype": np.float32, "indices": np.array([[0, 0], [1, 0]]).astype('int32'), "axis": 1},
    {"x_shape": [2, 3, 3], "x_dtype": np.float32, "indices": np.array([[[1]]]).astype('int32'), "axis": 0},
    {"x_shape": [2, 2], "x_dtype": np.float32, "indices": np.array([[0, 1],
                                                                    [1, 0],
                                                                    [0, 0],
                                                                    [1, 1]]).astype('int32'), "axis": 0},
]
# fmt: on


@ddt
class TestTakeAlongAxis(TestAPIBase):
    def setUp(self):
        self.init_attrs()

    def init_attrs(self):
        self.x_shape = [2, 3, 2]
        self.x_dtype = np.int32
        self.indices = np.array([[[1]]]).astype("int32")
        self.axis = 0

    def prepare_data(self):
        self.data_x = self.generate_data(self.x_shape, self.x_dtype)

    def forward(self):
        x = paddle.to_tensor(self.data_x, dtype=self.x_dtype)
        indices = paddle.to_tensor(self.indices)
        return paddle.take_along_axis(x, indices, self.axis)

    def cast(self):
        x = paddle.to_tensor(self.data_x, dtype="float32")
        indices = paddle.to_tensor(self.indices, dtype="int32")
        return paddle.take_along_axis(x, indices, self.axis)

    def expect_output(self):
        if self.x_dtype != np.float16:
            out = self.calc_result(self.forward, "cpu")
        else:
            out = self.cast()
        return out

    @data(*TAKE_ALONG_AXIS_CASE)
    @unpack
    def test_check_output(self, x_shape, x_dtype, indices, axis):
        self.x_shape = x_shape
        self.x_dtype = x_dtype
        self.indices = indices
        self.axis = axis
        self.check_output_gcu_with_customized(self.forward, self.expect_output)


if __name__ == "__main__":
    unittest.main()
