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
TOPK_CASE = [
    {"x_shape": [6], "x_dtype": np.float32, "k": 1, "axis": None, "largest": True, "sorted": True},
    {"x_shape": [2, 3], "x_dtype": np.float32, "k": 3, "axis": None, "largest": True, "sorted": True},
    {"x_shape": [5, 6, 7, 8], "x_dtype": np.float32, "k": 6, "axis": None, "largest": True, "sorted": True},
    {"x_shape": [5, 6, 7, 8], "x_dtype": np.float32, "k": 3, "axis": 1, "largest": True, "sorted": True},
    {"x_shape": [5, 6, 7, 8], "x_dtype": np.float32, "k": 3, "axis": -2, "largest": True, "sorted": True},
    # {"x_shape": [5, 6, 7, 8], "x_dtype": np.float32, "k": 3, "axis": 1, "largest": False, "sorted": True},
    # {"x_shape": [5, 6, 7, 8], "x_dtype": np.float32, "k": 3, "axis": 1, "largest": True, "sorted": False},

    {"x_shape": [6], "x_dtype": np.float64, "k": 1, "axis": None, "largest": True, "sorted": True},
    {"x_shape": [2, 3], "x_dtype": np.float64, "k": 3, "axis": None, "largest": True, "sorted": True},
    {"x_shape": [5, 6, 7, 8], "x_dtype": np.float64, "k": 8, "axis": None, "largest": True, "sorted": True},
    {"x_shape": [5, 6, 7, 8], "x_dtype": np.float64, "k": 3, "axis": 1, "largest": True, "sorted": True},
    {"x_shape": [5, 6, 7, 8], "x_dtype": np.float64, "k": 3, "axis": -2, "largest": True, "sorted": True},
    # {"x_shape": [5, 6, 7, 8], "x_dtype": np.float64, "k": 3, "axis": 1, "largest": False, "sorted": True},
    # {"x_shape": [5, 6, 7, 8], "x_dtype": np.float64, "k": 3, "axis": 1, "largest": True, "sorted": False},

    {"x_shape": [6], "x_dtype": np.int32, "k": 1, "axis": None, "largest": True, "sorted": True},
    {"x_shape": [2, 3], "x_dtype": np.int32, "k": 3, "axis": None, "largest": True, "sorted": True},
    {"x_shape": [5, 6, 7, 8], "x_dtype": np.int32, "k": 8, "axis": None, "largest": True, "sorted": True},
    {"x_shape": [5, 6, 7, 8], "x_dtype": np.int32, "k": 3, "axis": 1, "largest": True, "sorted": True},
    {"x_shape": [5, 6, 7, 8], "x_dtype": np.int32, "k": 3, "axis": -2, "largest": True, "sorted": True},
    # {"x_shape": [5, 6, 7, 8], "x_dtype": np.int32, "k": 3, "axis": 1, "largest": False, "sorted": True},
    # {"x_shape": [5, 6, 7, 8], "x_dtype": np.int32, "k": 3, "axis": 1, "largest": True, "sorted": False},

    {"x_shape": [6], "x_dtype": np.int64, "k": 1, "axis": None, "largest": True, "sorted": True},
    {"x_shape": [2, 3], "x_dtype": np.int64, "k": 3, "axis": None, "largest": True, "sorted": True},
    {"x_shape": [5, 6, 7, 8], "x_dtype": np.int64, "k": 8, "axis": None, "largest": True, "sorted": True},
    {"x_shape": [5, 6, 7, 8], "x_dtype": np.int64, "k": 3, "axis": 1, "largest": True, "sorted": True},
    {"x_shape": [5, 6, 7, 8], "x_dtype": np.int64, "k": 3, "axis": -2, "largest": True, "sorted": True},
    # {"x_shape": [5, 6, 7, 8], "x_dtype": np.int64, "k": 3, "axis": 1, "largest": False, "sorted": True},
    # {"x_shape": [5, 6, 7, 8], "x_dtype": np.int64, "k": 3, "axis": 1, "largest": True, "sorted": False},
]
# fmt: on


@ddt
class TestTopK(TestAPIBase):
    def setUp(self):
        self.init_attrs()

    def init_attrs(self):
        self.x_shape = [2, 4]
        self.x_dtype = np.float32
        self.k = 1
        self.axis = None
        self.largest = True
        self.sorted = True

    def prepare_data(self):
        self.data_x = self.generate_data(self.x_shape, self.x_dtype)

    def forward(self):
        x = paddle.to_tensor(self.data_x, dtype=self.x_dtype)
        return paddle.topk(x, self.k, self.axis, self.largest, self.sorted)

    def topk_cast(self):
        x = paddle.to_tensor(self.data_x, dtype="float32")
        return paddle.topk(x, self.k, self.axis, self.largest, self.sorted)

    def expect_output(self):
        if self.x_dtype != np.float16:
            out = self.calc_result(self.forward, "cpu")
        else:
            out = self.topk_cast()
        return out

    @data(*TOPK_CASE)
    @unpack
    def test_check_output(self, x_shape, x_dtype, k, axis, largest, sorted):
        self.x_shape = x_shape
        self.x_dtype = x_dtype
        self.k = k
        self.axis = axis
        self.largest = largest
        self.sorted = sorted
        self.check_output_gcu_with_customized(self.forward, self.expect_output)


if __name__ == "__main__":
    unittest.main()
