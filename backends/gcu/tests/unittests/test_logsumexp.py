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
LOG_SUM_EXP_CASE = [

    {"x_shape": [2, 6], "x_dtype": np.float32, "axis": None, "keepdim": False},
    {"x_shape": [2, 6], "x_dtype": np.float32, "axis": 0, "keepdim": False},
    {"x_shape": [2, 6], "x_dtype": np.float32, "axis": -1, "keepdim": False},
    {"x_shape": [2, 6], "x_dtype": np.float32, "axis": None, "keepdim": True},
    {"x_shape": [2, 6], "x_dtype": np.float32, "axis": 0, "keepdim": True},
    {"x_shape": [2, 6], "x_dtype": np.float32, "axis": -1, "keepdim": True},
    {"x_shape": [2, 3, 4], "x_dtype": np.float32, "axis": None, "keepdim": False},
    {"x_shape": [2, 3, 4], "x_dtype": np.float32, "axis": [0, 1], "keepdim": False},
    {"x_shape": [2, 3, 4], "x_dtype": np.float32, "axis": [0, 2], "keepdim": False},
    {"x_shape": [2, 3, 4], "x_dtype": np.float32, "axis": [-2, -1], "keepdim": False},
    {"x_shape": [2, 3, 4], "x_dtype": np.float32, "axis": [1, 2], "keepdim": False},
    {"x_shape": [2, 3, 4], "x_dtype": np.float32, "axis": None, "keepdim": True},
    {"x_shape": [2, 3, 4], "x_dtype": np.float32, "axis": [0, 1], "keepdim": True},
    {"x_shape": [2, 3, 4], "x_dtype": np.float32, "axis": [0, 2], "keepdim": True},
    {"x_shape": [2, 3, 4], "x_dtype": np.float32, "axis": [-2, -1], "keepdim": True},
    {"x_shape": [2, 3, 4], "x_dtype": np.float32, "axis": [1, 2], "keepdim": True},


    {"x_shape": [2, 6], "x_dtype": np.float16, "axis": None, "keepdim": False},
    {"x_shape": [2, 6], "x_dtype": np.float16, "axis": 0, "keepdim": False},
    {"x_shape": [2, 6], "x_dtype": np.float16, "axis": -1, "keepdim": False},
    {"x_shape": [2, 6], "x_dtype": np.float16, "axis": None, "keepdim": True},
    {"x_shape": [2, 6], "x_dtype": np.float16, "axis": 0, "keepdim": True},
    {"x_shape": [2, 6], "x_dtype": np.float16, "axis": -1, "keepdim": True},
    {"x_shape": [2, 3, 4], "x_dtype": np.float16, "axis": None, "keepdim": False},
    {"x_shape": [2, 3, 4], "x_dtype": np.float16, "axis": [0, 1], "keepdim": False},
    {"x_shape": [2, 3, 4], "x_dtype": np.float16, "axis": [0, 2], "keepdim": False},
    {"x_shape": [2, 3, 4], "x_dtype": np.float16, "axis": [-2, -1], "keepdim": False},
    {"x_shape": [2, 3, 4], "x_dtype": np.float16, "axis": [1, 2], "keepdim": False},
    {"x_shape": [2, 3, 4], "x_dtype": np.float16, "axis": None, "keepdim": True},
    {"x_shape": [2, 3, 4], "x_dtype": np.float16, "axis": [0, 1], "keepdim": True},
    {"x_shape": [2, 3, 4], "x_dtype": np.float16, "axis": [0, 2], "keepdim": True},
    {"x_shape": [2, 3, 4], "x_dtype": np.float16, "axis": [-2, -1], "keepdim": True},
    {"x_shape": [2, 3, 4], "x_dtype": np.float16, "axis": [1, 2], "keepdim": True},

]
# fmt: on


@ddt
class TestLogSumExp(TestAPIBase):
    def setUp(self):
        self.init_attrs()

    def init_attrs(self):
        self.x_shape = [2, 6]
        self.x_dtype = np.float32
        self.axis = None
        self.keepdim = False

    def prepare_data(self):
        self.data_x = self.generate_data(self.x_shape, self.x_dtype)

    def forward(self):
        x = paddle.to_tensor(self.data_x, dtype=self.x_dtype)
        return paddle.logsumexp(x, axis=self.axis, keepdim=self.keepdim)

    def get_numpy_output(self):
        axis = tuple(self.axis) if isinstance(self.axis, list) else self.axis
        return np.log(np.exp(self.data_x).sum(axis=axis, keepdims=self.keepdim))

    def expect_output(self):
        if self.x_dtype != np.float16:
            out = self.calc_result(self.forward, "cpu")
        else:
            out = self.get_numpy_output()
        return out

    @data(*LOG_SUM_EXP_CASE)
    @unpack
    def test_check_output(self, x_shape, x_dtype, axis, keepdim):
        self.x_shape = x_shape
        self.x_dtype = x_dtype
        self.axis = axis
        self.keepdim = keepdim
        rtol = 1e-5
        atol = 1e-5
        if x_dtype == np.float16:
            rtol = 1e-3
            atol = 1e-3
        self.check_output_gcu_with_customized(
            self.forward, self.expect_output, rtol=rtol, atol=atol
        )


if __name__ == "__main__":
    unittest.main()
