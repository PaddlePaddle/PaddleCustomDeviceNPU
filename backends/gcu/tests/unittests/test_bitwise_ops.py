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
BITWISE_CASE = [
    {"bitwise_api": paddle.bitwise_and, "shape": [2, 3, 32, 32], "dtype": np.int8},
    {"bitwise_api": paddle.bitwise_and, "shape": [2, 3, 32, 32], "dtype": np.int16},
    {"bitwise_api": paddle.bitwise_and, "shape": [2, 3, 32, 32], "dtype": np.int32},
    # {"bitwise_api": paddle.bitwise_and, "shape": [2, 3, 32, 32], "dtype": np.int64},
    {"bitwise_api": paddle.bitwise_and, "shape": [2, 4, 4], "dtype": np.int8},
    {"bitwise_api": paddle.bitwise_and, "shape": [2, 4, 4], "dtype": np.int16},
    {"bitwise_api": paddle.bitwise_and, "shape": [2, 3, 32, 32], "dtype": bool},
    {"bitwise_api": paddle.bitwise_and, "shape": [2, 4, 4], "dtype": bool},

    {"bitwise_api": paddle.bitwise_not, "shape": [2, 3, 32, 32], "dtype": np.int8},
    {"bitwise_api": paddle.bitwise_not, "shape": [2, 3, 32, 32], "dtype": np.int16},
    {"bitwise_api": paddle.bitwise_not, "shape": [2, 3, 32, 32], "dtype": np.int32},
    # {"bitwise_api": paddle.bitwise_not, "shape": [2, 3, 32, 32], "dtype": np.int64},
    {"bitwise_api": paddle.bitwise_not, "shape": [2, 4, 4], "dtype": np.int8},
    {"bitwise_api": paddle.bitwise_not, "shape": [2, 4, 4], "dtype": np.int16},
    {"bitwise_api": paddle.bitwise_not, "shape": [2, 3, 32, 32], "dtype": bool},
    {"bitwise_api": paddle.bitwise_not, "shape": [2, 4, 4], "dtype": bool},

    {"bitwise_api": paddle.bitwise_or, "shape": [2, 3, 32, 32], "dtype": np.int8},
    {"bitwise_api": paddle.bitwise_or, "shape": [2, 3, 32, 32], "dtype": np.int16},
    {"bitwise_api": paddle.bitwise_or, "shape": [2, 3, 32, 32], "dtype": np.int32},
    # {"bitwise_api": paddle.bitwise_or, "shape": [2, 3, 32, 32], "dtype": np.int64},
    {"bitwise_api": paddle.bitwise_or, "shape": [2, 4, 4], "dtype": np.int8},
    {"bitwise_api": paddle.bitwise_or, "shape": [2, 4, 4], "dtype": np.int16},
    {"bitwise_api": paddle.bitwise_or, "shape": [2, 3, 32, 32], "dtype": bool},
    {"bitwise_api": paddle.bitwise_or, "shape": [2, 4, 4], "dtype": bool},

    {"bitwise_api": paddle.bitwise_xor, "shape": [2, 3, 32, 32], "dtype": np.int8},
    {"bitwise_api": paddle.bitwise_xor, "shape": [2, 3, 32, 32], "dtype": np.int16},
    {"bitwise_api": paddle.bitwise_xor, "shape": [2, 3, 32, 32], "dtype": np.int32},
    # {"bitwise_api": paddle.bitwise_xor, "shape": [2, 3, 32, 32], "dtype": np.int64},
    {"bitwise_api": paddle.bitwise_xor, "shape": [2, 4, 4], "dtype": np.int8},
    {"bitwise_api": paddle.bitwise_xor, "shape": [2, 4, 4], "dtype": np.int16},
    {"bitwise_api": paddle.bitwise_xor, "shape": [2, 3, 32, 32], "dtype": bool},
    {"bitwise_api": paddle.bitwise_xor, "shape": [2, 4, 4], "dtype": bool},
]

BITWISE_SHIFT_CASE = [
    {"bitwise_api": paddle.bitwise_left_shift, "shape": [2, 3, 32, 32], "dtype": np.int8, "is_arithmetic": True},
    {"bitwise_api": paddle.bitwise_left_shift, "shape": [2, 3, 32, 32], "dtype": np.int16, "is_arithmetic": True},
    {"bitwise_api": paddle.bitwise_left_shift, "shape": [2, 3, 32, 32], "dtype": np.int32, "is_arithmetic": True},
    # {"bitwise_api": paddle.bitwise_left_shift, "shape": [2, 3, 32, 32], "dtype": np.int64, "is_arithmetic": True},
    {"bitwise_api": paddle.bitwise_left_shift, "shape": [2, 4, 4], "dtype": np.int8, "is_arithmetic": True},
    {"bitwise_api": paddle.bitwise_left_shift, "shape": [2, 4, 4], "dtype": np.int16, "is_arithmetic": True},

    {"bitwise_api": paddle.bitwise_left_shift, "shape": [2, 3, 32, 32], "dtype": np.int8, "is_arithmetic": False},
    {"bitwise_api": paddle.bitwise_left_shift, "shape": [2, 3, 32, 32], "dtype": np.int16, "is_arithmetic": False},
    {"bitwise_api": paddle.bitwise_left_shift, "shape": [2, 3, 32, 32], "dtype": np.int32, "is_arithmetic": False},
    # {"bitwise_api": paddle.bitwise_left_shift, "shape": [2, 3, 32, 32], "dtype": np.int64, "is_arithmetic": False},
    {"bitwise_api": paddle.bitwise_left_shift, "shape": [2, 4, 4], "dtype": np.int8, "is_arithmetic": False},
    {"bitwise_api": paddle.bitwise_left_shift, "shape": [2, 4, 4], "dtype": np.int16, "is_arithmetic": False},

    # {"bitwise_api": paddle.bitwise_right_shift, "shape": [2, 3, 32, 32], "dtype": np.int8, "is_arithmetic": True},
    # {"bitwise_api": paddle.bitwise_right_shift, "shape": [2, 3, 32, 32], "dtype": np.int16, "is_arithmetic": True},
    # {"bitwise_api": paddle.bitwise_right_shift, "shape": [2, 3, 32, 32], "dtype": np.int32, "is_arithmetic": True},
    # # {"bitwise_api": paddle.bitwise_right_shift, "shape": [2, 3, 32, 32], "dtype": np.int64, "is_arithmetic": True},
    # {"bitwise_api": paddle.bitwise_right_shift, "shape": [2, 4, 4], "dtype": np.int8, "is_arithmetic": True},
    # {"bitwise_api": paddle.bitwise_right_shift, "shape": [2, 4, 4], "dtype": np.int16, "is_arithmetic": True},

    # # Logical RightShift is unimplemented.
    # {"bitwise_api": paddle.bitwise_right_shift, "shape": [2, 3, 32, 32], "dtype": np.int8, "is_arithmetic": False},
    # {"bitwise_api": paddle.bitwise_right_shift, "shape": [2, 3, 32, 32], "dtype": np.int16, "is_arithmetic": False},
    # {"bitwise_api": paddle.bitwise_right_shift, "shape": [2, 3, 32, 32], "dtype": np.int32, "is_arithmetic": False},
    # # {"bitwise_api": paddle.bitwise_right_shift, "shape": [2, 3, 32, 32], "dtype": np.int64, "is_arithmetic": False},
    # {"bitwise_api": paddle.bitwise_right_shift, "shape": [2, 4, 4], "dtype": np.int8, "is_arithmetic": False},
    # {"bitwise_api": paddle.bitwise_right_shift, "shape": [2, 4, 4], "dtype": np.int16, "is_arithmetic": False},
]
# fmt: on


class TestBitwiseOps(TestAPIBase):
    def setUp(self):
        self.init_attrs()

    def init_attrs(self):
        self.bitwise_api = paddle.add
        self.shape = [4]
        self.dtype = np.int8

    def prepare_data(self):
        self.data_x = self.generate_data(self.shape, self.dtype)
        self.data_y = self.generate_data(self.shape, self.dtype)

        if self.dtype == np.int64 or self.dtype == np.int32:
            self.data_x = self.generate_integer_data(self.shape, self.dtype, -200, 200)
            self.data_y = self.generate_integer_data(self.shape, self.dtype, -200, 200)

    def bitwise_forward(self):
        x = paddle.to_tensor(self.data_x, dtype=self.dtype)
        y = paddle.to_tensor(self.data_y, dtype=self.dtype)
        if self.bitwise_api in [paddle.bitwise_not]:
            return self.bitwise_api(x)
        return self.bitwise_api(x, y)


@ddt
class TestBitwiseOpsComputation(TestBitwiseOps):
    @data(*BITWISE_CASE)
    @unpack
    def test_check_output(self, bitwise_api, shape, dtype):
        self.bitwise_api = bitwise_api
        self.shape = shape
        self.dtype = dtype
        self.check_output_gcu_with_cpu(self.bitwise_forward)


@ddt
class TestBitwiseShiftOpsComputation(TestBitwiseOps):
    def prepare_data(self):
        self.data_x = self.generate_integer_data(self.shape, self.dtype, -10, 10)
        self.data_y = self.generate_integer_data(self.shape, self.dtype, 0, 3)

    def bitwise_forward(self):
        x = paddle.to_tensor(self.data_x, dtype=self.dtype)
        y = paddle.to_tensor(self.data_y, dtype=self.dtype)
        return self.bitwise_api(x, y, self.is_arithmetic)

    @data(*BITWISE_SHIFT_CASE)
    @unpack
    def test_check_output(self, bitwise_api, shape, dtype, is_arithmetic):
        self.bitwise_api = bitwise_api
        self.shape = shape
        self.dtype = dtype
        self.is_arithmetic = is_arithmetic
        self.check_output_gcu_with_cpu(self.bitwise_forward)


if __name__ == "__main__":
    unittest.main()
