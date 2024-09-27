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
COMPARE_CASE = [
    {"compare_api": paddle.greater_than, "shape": [2, 3, 32, 32], "dtype": np.float32},
    {"compare_api": paddle.greater_than, "shape": [2, 3, 32, 32], "dtype": np.float64},
    {"compare_api": paddle.greater_than, "shape": [2, 3, 32, 32], "dtype": np.int32},
    {"compare_api": paddle.greater_than, "shape": [2, 3, 32, 32], "dtype": np.int64},
    {"compare_api": paddle.greater_than, "shape": [2, 4, 4], "dtype": np.float32},
    {"compare_api": paddle.greater_than, "shape": [2, 4, 4], "dtype": np.float64},
    {"compare_api": paddle.greater_than, "shape": [2, 3, 32, 32], "dtype": bool},
    {"compare_api": paddle.greater_than, "shape": [2, 4, 4], "dtype": bool},

    {"compare_api": paddle.greater_equal, "shape": [2, 3, 32, 32], "dtype": np.float32},
    {"compare_api": paddle.greater_equal, "shape": [2, 3, 32, 32], "dtype": np.float64},
    {"compare_api": paddle.greater_equal, "shape": [2, 3, 32, 32], "dtype": np.int32},
    {"compare_api": paddle.greater_equal, "shape": [2, 3, 32, 32], "dtype": np.int64},
    {"compare_api": paddle.greater_equal, "shape": [2, 4, 4], "dtype": np.float32},
    {"compare_api": paddle.greater_equal, "shape": [2, 4, 4], "dtype": np.float64},
    {"compare_api": paddle.greater_equal, "shape": [2, 3, 32, 32], "dtype": bool},
    {"compare_api": paddle.greater_equal, "shape": [2, 4, 4], "dtype": bool},

    {"compare_api": paddle.equal, "shape": [2, 3, 32, 32], "dtype": np.float32},
    {"compare_api": paddle.equal, "shape": [2, 3, 32, 32], "dtype": np.float64},
    {"compare_api": paddle.equal, "shape": [2, 3, 32, 32], "dtype": np.int32},
    {"compare_api": paddle.equal, "shape": [2, 3, 32, 32], "dtype": np.int64},
    {"compare_api": paddle.equal, "shape": [2, 4, 4], "dtype": np.float32},
    {"compare_api": paddle.equal, "shape": [2, 4, 4], "dtype": np.float64},
    {"compare_api": paddle.equal, "shape": [2, 3, 32, 32], "dtype": bool},
    {"compare_api": paddle.equal, "shape": [2, 4, 4], "dtype": bool},

    {"compare_api": paddle.not_equal, "shape": [2, 3, 32, 32], "dtype": np.float32},
    {"compare_api": paddle.not_equal, "shape": [2, 3, 32, 32], "dtype": np.float64},
    {"compare_api": paddle.not_equal, "shape": [2, 3, 32, 32], "dtype": np.int32},
    {"compare_api": paddle.not_equal, "shape": [2, 3, 32, 32], "dtype": np.int64},
    {"compare_api": paddle.not_equal, "shape": [2, 4, 4], "dtype": np.float32},
    {"compare_api": paddle.not_equal, "shape": [2, 4, 4], "dtype": np.float64},
    {"compare_api": paddle.not_equal, "shape": [2, 3, 32, 32], "dtype": bool},
    {"compare_api": paddle.not_equal, "shape": [2, 4, 4], "dtype": bool},

    {"compare_api": paddle.less_than, "shape": [2, 3, 32, 32], "dtype": np.float32},
    {"compare_api": paddle.less_than, "shape": [2, 3, 32, 32], "dtype": np.float64},
    {"compare_api": paddle.less_than, "shape": [2, 3, 32, 32], "dtype": np.int32},
    {"compare_api": paddle.less_than, "shape": [2, 3, 32, 32], "dtype": np.int64},
    {"compare_api": paddle.less_than, "shape": [2, 4, 4], "dtype": np.float32},
    {"compare_api": paddle.less_than, "shape": [2, 4, 4], "dtype": np.float64},
    {"compare_api": paddle.less_than, "shape": [2, 3, 32, 32], "dtype": bool},
    {"compare_api": paddle.less_than, "shape": [2, 4, 4], "dtype": bool},

    {"compare_api": paddle.less_equal, "shape": [2, 3, 32, 32], "dtype": np.float32},
    {"compare_api": paddle.less_equal, "shape": [2, 3, 32, 32], "dtype": np.float64},
    {"compare_api": paddle.less_equal, "shape": [2, 3, 32, 32], "dtype": np.int32},
    {"compare_api": paddle.less_equal, "shape": [2, 3, 32, 32], "dtype": np.int64},
    {"compare_api": paddle.less_equal, "shape": [2, 4, 4], "dtype": np.float32},
    {"compare_api": paddle.less_equal, "shape": [2, 4, 4], "dtype": np.float64},
    {"compare_api": paddle.less_equal, "shape": [2, 3, 32, 32], "dtype": bool},
    {"compare_api": paddle.less_equal, "shape": [2, 4, 4], "dtype": bool},
]

COMPARE_F16_CASE = [
    {"compare_api": paddle.greater_than, "numpy_api": np.greater, "shape": [2, 3, 32, 32], "dtype": np.float16},
    {"compare_api": paddle.greater_equal, "numpy_api": np.greater_equal, "shape": [2, 3, 32, 32], "dtype": np.float16},
    {"compare_api": paddle.equal, "numpy_api": np.equal, "shape": [2, 3, 32, 32], "dtype": np.float16},
    {"compare_api": paddle.not_equal, "numpy_api": np.not_equal, "shape": [2, 3, 32, 32], "dtype": np.float16},
    {"compare_api": paddle.less_than, "numpy_api": np.less, "shape": [2, 3, 32, 32], "dtype": np.float16},
    {"compare_api": paddle.less_equal, "numpy_api": np.less_equal, "shape": [2, 3, 32, 32], "dtype": np.float16},
]
# fmt: on


class TestCompareOps(TestAPIBase):
    def setUp(self):
        self.init_attrs()

    def init_attrs(self):
        self.compare_api = paddle.add
        self.shape = [4]
        self.dtype = np.float32

    def prepare_data(self):
        self.data_x = self.generate_data(self.shape, self.dtype)
        self.data_y = self.generate_data(self.shape, self.dtype)

        if self.dtype == np.int64 or self.dtype == np.int32:
            self.data_x = self.generate_integer_data(self.shape, self.dtype, -200, 200)
            self.data_y = self.generate_integer_data(self.shape, self.dtype, -200, 200)

    def compare_forward(self):
        x = paddle.to_tensor(self.data_x, dtype=self.dtype)
        y = paddle.to_tensor(self.data_y, dtype=self.dtype)
        return self.compare_api(x, y)


@ddt
class TestCompareOpsComputation(TestCompareOps):
    @data(*COMPARE_CASE)
    @unpack
    def test_check_output(self, compare_api, shape, dtype):
        self.compare_api = compare_api
        self.shape = shape
        self.dtype = dtype
        self.check_output_gcu_with_cpu(self.compare_forward)


@ddt
class TestCompareOpsComputationF16(TestCompareOps):
    def numpy_binary(self):
        return self.numpy_api(self.data_x, self.data_y)

    @data(*COMPARE_F16_CASE)
    @unpack
    def test_check_output(self, compare_api, numpy_api, shape, dtype):
        self.compare_api = compare_api
        self.numpy_api = numpy_api
        self.shape = shape
        self.dtype = dtype
        self.check_output_gcu_with_customized(self.compare_forward, self.numpy_binary)


if __name__ == "__main__":
    unittest.main()
