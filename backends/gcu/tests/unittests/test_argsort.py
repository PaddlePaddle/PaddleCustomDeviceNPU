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
ARG_SORT_CASE = [
    {"x_shape": [2, 3], "x_dtype": np.float32, "axis": -1, "descending": False, "stable": False},
    {"x_shape": [2, 3], "x_dtype": np.float32, "axis": 0, "descending": False, "stable": False},
    {"x_shape": [2, 3], "x_dtype": np.float32, "axis": 1, "descending": False, "stable": False},
    {"x_shape": [2, 3], "x_dtype": np.float32, "axis": -1, "descending": True, "stable": False},
    {"x_shape": [2, 3], "x_dtype": np.float32, "axis": 0, "descending": True, "stable": False},
    {"x_shape": [2, 3], "x_dtype": np.float32, "axis": 1, "descending": True, "stable": False},
    {"x_shape": [1, 3], "x_dtype": np.float32, "axis": -1, "descending": False, "stable": False},
    {"x_shape": [1, 3], "x_dtype": np.float32, "axis": 0, "descending": False, "stable": False},
    {"x_shape": [1, 3], "x_dtype": np.float32, "axis": 1, "descending": False, "stable": False},
    {"x_shape": [1, 3], "x_dtype": np.float32, "axis": -1, "descending": True, "stable": False},
    {"x_shape": [1, 3], "x_dtype": np.float32, "axis": 0, "descending": True, "stable": False},
    {"x_shape": [1, 3], "x_dtype": np.float32, "axis": 1, "descending": True, "stable": False},
    {"x_shape": [2, 3], "x_dtype": np.int32, "axis": -1, "descending": False, "stable": False},
    {"x_shape": [2, 3], "x_dtype": np.int32, "axis": 0, "descending": False, "stable": False},
    {"x_shape": [2, 3], "x_dtype": np.int32, "axis": 1, "descending": False, "stable": False},
    {"x_shape": [2, 3], "x_dtype": np.int32, "axis": -1, "descending": True, "stable": False},
    {"x_shape": [2, 3], "x_dtype": np.int32, "axis": 0, "descending": True, "stable": False},
    {"x_shape": [2, 3], "x_dtype": np.int32, "axis": 1, "descending": True, "stable": False},
    {"x_shape": [2, 3], "x_dtype": np.float16, "axis": -1, "descending": False, "stable": False},
    {"x_shape": [2, 3], "x_dtype": np.float16, "axis": 0, "descending": False, "stable": False},
    {"x_shape": [2, 3], "x_dtype": np.float16, "axis": 1, "descending": False, "stable": False},
    {"x_shape": [2, 3], "x_dtype": np.float16, "axis": -1, "descending": True, "stable": False},
    {"x_shape": [2, 3], "x_dtype": np.float16, "axis": 0, "descending": True, "stable": False},
    {"x_shape": [2, 3], "x_dtype": np.float16, "axis": 1, "descending": True, "stable": False},
    {"x_shape": [2, 3, 4], "x_dtype": np.float16, "axis": -1, "descending": False, "stable": False},
    {"x_shape": [2, 3, 4, 5], "x_dtype": np.float16, "axis": -1, "descending": False, "stable": False},

    {"x_shape": [2, 3], "x_dtype": np.float32, "axis": -1, "descending": False, "stable": True},
    {"x_shape": [2, 3, 4], "x_dtype": np.float16, "axis": 0, "descending": False, "stable": True},
    {"x_shape": [2, 3, 4, 5], "x_dtype": np.float16, "axis": -1, "descending": False, "stable": True},
    {"x_shape": [2, 3], "x_dtype": np.float32, "axis": -1, "descending": True, "stable": True},
    {"x_shape": [2, 3, 4], "x_dtype": np.float16, "axis": 0, "descending": True, "stable": True},
    {"x_shape": [2, 3, 4, 5], "x_dtype": np.float16, "axis": -1, "descending": True, "stable": True},
]
# fmt: on


@ddt
class TestArgSort(TestAPIBase):
    def setUp(self):
        self.init_attrs()

    def init_attrs(self):
        self.x_shape = [2, 3]
        self.x_dtype = np.float32
        self.axis = -1
        self.descending = False
        self.stable = False

    def prepare_data(self):
        self.data_x = self.generate_data(self.x_shape, self.x_dtype)

    def forward_with_dtype(self, dtype):
        x = paddle.to_tensor(self.data_x, dtype=dtype)
        return [
            paddle.sort(
                x, axis=self.axis, descending=self.descending, stable=self.stable
            ),
            paddle.argsort(
                x, axis=self.axis, descending=self.descending, stable=self.stable
            ),
        ]

    def forward(self):
        return self.forward_with_dtype(self.x_dtype)

    def argsort_cast(self):
        out_fp32 = self.forward_with_dtype(np.float32)
        return [out_fp32[0].astype("float16"), out_fp32[1]]

    def expect_output(self):
        if self.x_dtype != np.float16:
            out = self.forward()
        else:
            out = self.argsort_cast()
        return out

    @data(*ARG_SORT_CASE)
    @unpack
    def test_check_output(self, x_shape, x_dtype, axis, descending, stable):
        self.x_shape = x_shape
        self.x_dtype = x_dtype
        self.axis = axis
        self.descending = descending
        self.stable = stable
        self.check_output_gcu_with_customized(self.forward, self.expect_output)


if __name__ == "__main__":
    unittest.main()
