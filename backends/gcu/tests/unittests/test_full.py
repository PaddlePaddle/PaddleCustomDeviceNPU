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
from paddle import _C_ops


# The table retains its original format for better comparison of parameter settings.
# fmt: off
FULL_CASE = [
    {"shape": [1], "dtype": np.float32, "fill_value": 6.6},
    {"shape": [9], "dtype": np.float32, "fill_value": 6},
    {"shape": [3, 6], "dtype": np.float32, "fill_value": 6.8},
    {"shape": [2, 3, 6], "dtype": np.float32, "fill_value": 6},
    {"shape": [2, 2, 3, 6], "dtype": np.float32, "fill_value": 6},
    {"shape": [1], "dtype": np.float16, "fill_value": 6.8},
    {"shape": [9], "dtype": np.float16, "fill_value": 6},
    {"shape": [3, 6], "dtype": np.float16, "fill_value": 6.6},
    {"shape": [2, 3, 6], "dtype": np.float16, "fill_value": 6},
    {"shape": [2, 2, 3, 6], "dtype": np.float16, "fill_value": 6},
    {"shape": [2, 2, 3, 6], "dtype": bool, "fill_value": True},
    {"shape": [2, 2, 3, 6], "dtype": bool, "fill_value": False},
    {"shape": [2, 2, 3, 6], "dtype": np.int32, "fill_value": 6},
    {"shape": [2, 2, 3, 6], "dtype": np.int64, "fill_value": 6},
    {"shape": [1], "dtype": np.float64, "fill_value": 6.6},
    {"shape": [9], "dtype": np.float64, "fill_value": 6},
    {"shape": [3, 6], "dtype": np.float64, "fill_value": 6.8},
    {"shape": [2, 3, 6], "dtype": np.float64, "fill_value": 6},
    {"shape": [2, 2, 3, 6], "dtype": np.float64, "fill_value": 6},
]

FULL_LIKE_CASE = [
    {"x_shape": [3, 6], "x_dtype": np.float32, "fill_value": 6, "dtype": np.float32},
    {"x_shape": [2, 3, 6], "x_dtype": np.float32, "fill_value": 6, "dtype": np.float32},
    {"x_shape": [1, 5, 3, 6], "x_dtype": np.float32, "fill_value": 6, "dtype": np.float32},
    {"x_shape": [3, 6], "x_dtype": np.float16, "fill_value": 6, "dtype": np.float16},
    {"x_shape": [2, 3, 6], "x_dtype": np.float16, "fill_value": 6, "dtype": np.float16},
    {"x_shape": [1, 5, 3, 6], "x_dtype": np.float16, "fill_value": 6, "dtype": np.float16},
    {"x_shape": [3, 6], "x_dtype": np.float32, "fill_value": 6, "dtype": np.float16},
    {"x_shape": [2, 3, 6], "x_dtype": np.float32, "fill_value": 6, "dtype": np.float16},
    {"x_shape": [1, 5, 3, 6], "x_dtype": np.float32, "fill_value": 6, "dtype": np.float16},
    {"x_shape": [3, 6], "x_dtype": np.float16, "fill_value": 6, "dtype": None},
    {"x_shape": [2, 3, 6], "x_dtype": np.float16, "fill_value": 6, "dtype": None},
    {"x_shape": [1, 5, 3, 6], "x_dtype": np.float16, "fill_value": 6, "dtype": None},
    {"x_shape": [1, 5, 3, 6], "x_dtype": bool, "fill_value": True, "dtype": None},
    {"x_shape": [1, 5, 3, 6], "x_dtype": bool, "fill_value": False, "dtype": None},
    {"x_shape": [1, 5, 3, 6], "x_dtype": np.int32, "fill_value": 6, "dtype": None},
    {"x_shape": [1, 5, 3, 6], "x_dtype": np.int64, "fill_value": 6, "dtype": None},
    {"x_shape": [3, 6], "x_dtype": np.float64, "fill_value": 6, "dtype": np.float32},
    {"x_shape": [2, 3, 6], "x_dtype": np.float64, "fill_value": 6, "dtype": np.float32},
    {"x_shape": [1, 5, 3, 6], "x_dtype": np.float64, "fill_value": 6, "dtype": np.float32},

]
# fmt: on


@ddt
class TestFull(TestAPIBase):
    def setUp(self):
        self.init_attrs()

    def init_attrs(self):
        self.shape = [3, 6]
        self.fill_value = 1
        self.dtype = np.float32

    def prepare_data(self):
        pass

    def forward(self):
        return paddle.full(self.shape, self.fill_value, self.dtype)

    def expect_output(self):
        if self.dtype != np.float16:
            out = self.calc_result(self.forward, "cpu")
        else:
            out = self.fill_value * np.ones(self.shape, self.dtype)
        return out

    @data(*FULL_CASE)
    @unpack
    def test_check_output(self, shape, fill_value, dtype):
        self.shape = shape
        self.fill_value = fill_value
        self.dtype = dtype
        self.check_output_gcu_with_customized(self.forward, self.expect_output)


@ddt
class TestFullLike(TestAPIBase):
    def setUp(self):
        self.init_attrs()

    def init_attrs(self):
        self.x_shape = [3, 6]
        self.x_dtype = np.float32
        self.fill_value = 1
        self.dtype = np.float32

    def prepare_data(self):
        self.data_x = self.generate_data(self.x_shape, self.x_dtype)

    def forward(self):
        x = paddle.to_tensor(self.data_x, dtype=self.x_dtype)
        return paddle.full_like(x, self.fill_value, dtype=self.dtype)

    def expect_output(self):
        if self.x_dtype != np.float16 and self.dtype != np.float16:
            out = self.calc_result(self.forward, "cpu")
        else:
            out = self.fill_value * np.ones_like(self.data_x, self.dtype)
        return out

    @data(*FULL_LIKE_CASE)
    @unpack
    def test_check_output(self, x_shape, x_dtype, fill_value, dtype):
        self.x_shape = x_shape
        self.x_dtype = x_dtype
        self.fill_value = fill_value
        self.dtype = dtype
        self.check_output_gcu_with_customized(self.forward, self.expect_output)


# The table retains its original format for better comparison of parameter settings.
# fmt: off
FULL_BATCH_SIZE_LIKE_CASE = [
    {"input_shape": [3, 6], "out_shape": [2, 4], "value": 6, "input_dim_idx": 0, "output_dim_idx": 0},
    {"input_shape": [1, 6], "out_shape": [2, 3], "value": 6, "input_dim_idx": 0, "output_dim_idx": 0},
    {"input_shape": [6, 6, 6, 6], "out_shape": [2, 3], "value": 6, "input_dim_idx": 0, "output_dim_idx": 1},
    {"input_shape": [9, 6], "out_shape": [2, 3, 3], "value": 6, "input_dim_idx": 0, "output_dim_idx": 1},
    {"input_shape": [16, 6], "out_shape": [2, 3, 3], "value": 6, "input_dim_idx": 0, "output_dim_idx": 2},
]
# fmt: on


@ddt
class TestFullBatchSizeLike(TestAPIBase):
    def setUp(self):
        self.init_attrs()

    def init_attrs(self):
        self.input_shape = [2, 4]
        self.out_shape = [1, 3]
        self.value = 0
        self.input_dim_idx = 0
        self.output_dim_idx = 0

    def prepare_data(self):
        self.value = 0

    def forward(self):
        place = paddle.CustomPlace("gcu", 0)
        self.like = _C_ops.uniform(
            self.input_shape, paddle.base.core.DataType.FLOAT32, 2.0, 2.0, 0, place
        )
        data = _C_ops.full_batch_size_like(
            self.like,
            self.out_shape,
            paddle.base.core.DataType.FLOAT32,
            self.value,
            self.input_dim_idx,
            self.output_dim_idx,
            place,
        )
        return data

    def expect_output(self):
        place = paddle.CPUPlace()
        self.like = _C_ops.uniform(
            self.input_shape, paddle.base.core.DataType.FLOAT32, 2.0, 2.0, 0, place
        )
        data = _C_ops.full_batch_size_like(
            self.like,
            self.out_shape,
            paddle.base.core.DataType.FLOAT32,
            self.value,
            self.input_dim_idx,
            self.output_dim_idx,
            place,
        )
        return data

    @data(*FULL_BATCH_SIZE_LIKE_CASE)
    @unpack
    def test_check_output(
        self, input_shape, out_shape, value, input_dim_idx, output_dim_idx
    ):
        self.input_shape = input_shape
        self.out_shape = out_shape
        self.value = value
        self.input_dim_idx = input_dim_idx
        self.output_dim_idx = output_dim_idx
        self.check_output_gcu_with_customized(self.forward, self.expect_output)


if __name__ == "__main__":
    unittest.main()
