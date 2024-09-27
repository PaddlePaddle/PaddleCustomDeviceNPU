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
MATMUL_CASE = [
    {"x_shape": [1], "y_shape": [1], "dtype": np.float32, "trans_x": False, "trans_y": False},
    {"x_shape": [6], "y_shape": [6], "dtype": np.float32, "trans_x": False, "trans_y": False},
    {"x_shape": [6], "y_shape": [6], "dtype": np.float32, "trans_x": True, "trans_y": False},
    {"x_shape": [6], "y_shape": [6], "dtype": np.float32, "trans_x": False, "trans_y": True},
    {"x_shape": [3, 2], "y_shape": [2, 3], "dtype": np.float32, "trans_x": False, "trans_y": False},
    {"x_shape": [3, 2], "y_shape": [3, 2], "dtype": np.float32, "trans_x": False, "trans_y": True},
    {"x_shape": [3, 2], "y_shape": [2], "dtype": np.float32, "trans_x": False, "trans_y": True},
    {"x_shape": [3, 512, 256], "y_shape": [3, 256, 256], "dtype": np.float32, "trans_x": False, "trans_y": False},
    {"x_shape": [3, 256, 512], "y_shape": [3, 256, 512], "dtype": np.float32, "trans_x": True, "trans_y": False},
    {"x_shape": [3, 512, 256], "y_shape": [3, 512, 256], "dtype": np.float32, "trans_x": False, "trans_y": True},
    {"x_shape": [2, 3, 512, 256], "y_shape": [2, 3, 256, 256], "dtype": np.float32, "trans_x": False, "trans_y": False},
    {"x_shape": [2, 3, 256, 512], "y_shape": [2, 3, 256, 512], "dtype": np.float32, "trans_x": True, "trans_y": False},
    {"x_shape": [2, 3, 512, 256], "y_shape": [2, 3, 512, 256], "dtype": np.float32, "trans_x": False, "trans_y": True},
    {"x_shape": [2, 3, 256, 512], "y_shape": [2, 3, 512, 256], "dtype": np.float32, "trans_x": True, "trans_y": True},
    {"x_shape": [4, 5, 6], "y_shape": [6, 6], "dtype": np.float32, "trans_x": False, "trans_y": False},
    {"x_shape": [4, 5, 6, 7], "y_shape": [4, 5, 7, 6], "dtype": np.float32, "trans_x": False, "trans_y": False},

    # float16
    {"x_shape": [1], "y_shape": [1], "dtype": np.float16, "trans_x": False, "trans_y": False},
    {"x_shape": [3, 2], "y_shape": [2, 3], "dtype": np.float16, "trans_x": False, "trans_y": False},
    {"x_shape": [3, 2], "y_shape": [3, 2], "dtype": np.float16, "trans_x": False, "trans_y": True},
    {"x_shape": [3, 2], "y_shape": [2], "dtype": np.float16, "trans_x": False, "trans_y": True},
    {"x_shape": [3, 512, 256], "y_shape": [3, 256, 256], "dtype": np.float16, "trans_x": False, "trans_y": False},
    {"x_shape": [3, 256, 512], "y_shape": [3, 256, 512], "dtype": np.float16, "trans_x": True, "trans_y": False},
    {"x_shape": [3, 512, 256], "y_shape": [3, 512, 256], "dtype": np.float16, "trans_x": False, "trans_y": True},
    {"x_shape": [2, 3, 512, 256], "y_shape": [2, 3, 256, 256], "dtype": np.float16, "trans_x": False, "trans_y": False},
    {"x_shape": [2, 3, 256, 512], "y_shape": [2, 3, 256, 512], "dtype": np.float16, "trans_x": True, "trans_y": False},
    {"x_shape": [2, 3, 512, 256], "y_shape": [2, 3, 512, 256], "dtype": np.float16, "trans_x": False, "trans_y": True},
    {"x_shape": [2, 3, 256, 512], "y_shape": [2, 3, 512, 256], "dtype": np.float16, "trans_x": True, "trans_y": True},
]
# fmt: on


@ddt
class TestMatmul(TestAPIBase):
    def setUp(self):
        self.init_attrs()

    def init_attrs(self):
        self.x_shape = [3, 2]
        self.y_shape = [2, 3]
        self.dtype = np.float32
        self.trans_x = False
        self.trans_y = False

    def prepare_data(self):
        self.data_x = self.generate_data(self.x_shape, self.dtype)
        self.data_y = self.generate_data(self.y_shape, self.dtype)

    def forward(self):
        x = paddle.to_tensor(self.data_x, dtype=self.dtype)
        y = paddle.to_tensor(self.data_y, dtype=self.dtype)
        return paddle.matmul(x, y, self.trans_x, self.trans_y)

    def numpy_matmul(self, x, y, transpose_x=False, transpose_y=False):
        if transpose_x:
            if x.ndim == 1:
                x = x.reshape((x.size,))
            elif x.ndim == 2:
                x = x.T
            else:
                dim = list(range(len(x.shape)))
                dim[-1], dim[len(x.shape) - 2] = dim[len(x.shape) - 2], dim[-1]
                x = np.transpose(x, tuple(dim))
        if transpose_y:
            if y.ndim == 1:
                y = y.reshape((y.size,))
            else:
                dim = list(range(len(y.shape)))
                dim[-1], dim[len(y.shape) - 2] = dim[len(y.shape) - 2], dim[-1]
                y = np.transpose(y, tuple(dim))

        return np.matmul(x, y)

    def expect_output(self):
        if self.dtype != np.float16:
            out = self.calc_result(self.forward, "cpu")
        else:
            out = self.numpy_matmul(
                self.data_x, self.data_y, self.trans_x, self.trans_y
            )
        return out

    @data(*MATMUL_CASE)
    @unpack
    def test_check_output(self, x_shape, y_shape, dtype, trans_x, trans_y):
        self.x_shape = x_shape
        self.y_shape = y_shape
        self.dtype = dtype
        self.trans_x = trans_x
        self.trans_y = trans_y
        rtol = 1e-5
        atol = 1e-5
        if self.dtype == np.float16:
            rtol = 1e-3
            atol = 1e-3
        self.check_output_gcu_with_customized(
            self.forward, self.expect_output, rtol=rtol, atol=atol
        )


if __name__ == "__main__":
    unittest.main()
