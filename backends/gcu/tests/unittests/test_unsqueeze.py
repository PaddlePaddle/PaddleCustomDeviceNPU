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
UNSQUEEZE_CASE = [
    {"x_shape": [3, 6], "x_dtype": np.float32, "axis": 0, "new_shape": None},
    {"x_shape": [3, 6], "x_dtype": np.float32, "axis": [0, 2], "new_shape": None},
    {"x_shape": [3, 6], "x_dtype": np.float32, "axis": [0, 1, 2], "new_shape": None},
    {"x_shape": [3, 6], "x_dtype": np.float32, "axis": [-1], "new_shape": None},
    {"x_shape": [3, 6], "x_dtype": np.float32, "axis": [-2], "new_shape": None},

    {"x_shape": [3, 6], "x_dtype": np.float16, "axis": 0, "new_shape": [1, 3, 6]},
    {"x_shape": [3, 6], "x_dtype": np.float16, "axis": [0, 2], "new_shape": [1, 3, 1, 6]},
    {"x_shape": [3, 6], "x_dtype": np.float16, "axis": [0, 1, 2], "new_shape": [1, 1, 1, 3, 6]},
    {"x_shape": [3, 6], "x_dtype": np.float16, "axis": [-1], "new_shape": [3, 6, 1]},
    {"x_shape": [3, 6], "x_dtype": np.float16, "axis": [-2], "new_shape": [3, 1, 6]},

]
# fmt: on


@ddt
class TestUnsqueeze(TestAPIBase):
    def setUp(self):
        self.init_attrs()

    def init_attrs(self):
        self.x_shape = [3, 6]
        self.x_dtype = np.float32
        self.new_shape = [1, 3, 6]
        self.axis = 0

    def prepare_data(self):
        self.data_x = self.generate_data(self.x_shape, self.x_dtype)

    def forward(self):
        x = paddle.to_tensor(self.data_x, dtype=self.x_dtype)
        return paddle.unsqueeze(x, self.axis)

    def expect_output(self):
        if self.x_dtype != np.float16:
            out = self.calc_result(self.forward, "cpu")
        else:
            out = paddle.to_tensor(
                self.data_x.reshape(self.new_shape), dtype=self.x_dtype
            )
        return out

    @data(*UNSQUEEZE_CASE)
    @unpack
    def test_check_output(self, x_shape, x_dtype, axis, new_shape):
        self.x_shape = x_shape
        self.x_dtype = x_dtype
        self.axis = axis
        self.new_shape = new_shape
        self.check_output_gcu_with_customized(self.forward, self.expect_output)


if __name__ == "__main__":
    unittest.main()
