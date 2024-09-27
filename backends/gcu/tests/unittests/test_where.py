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
WHERE_CASE = [
    {"x_shape": [3, 2], "y_shape": [3, 2], "dtype": np.int32},
    {"x_shape": [3, 2], "y_shape": [3, 2], "dtype": np.float32},
    {"x_shape": [3, 1, 3], "y_shape": [3, 1, 3], "dtype": np.float32},
    {"x_shape": [1, 3, 128, 1], "y_shape": [1, 3, 128, 1], "dtype": np.float32},
    {"x_shape": [32, 3, 1, 128, 128], "y_shape": [32, 3, 1, 128, 128], "dtype": np.float32},
    {"x_shape": [1, 3, 128, 1], "y_shape": [1, 3, 128, 1], "dtype": np.float64},
    {"x_shape": [1, 3, 128, 1], "y_shape": [1, 3, 128, 1], "dtype": np.int64},
]

WHERE_F16_CASE = [
    {"x_shape": [3, 2], "y_shape": [3, 2], "dtype": np.float16},
    {"x_shape": [32, 3, 1, 128, 128], "y_shape": [32, 3, 1, 128, 128], "dtype": np.float16},
]
# fmt: on


class TestWhere(TestAPIBase):
    def setUp(self):
        self.init_attrs()

    def init_attrs(self):
        self.x_shape = [3, 2]
        self.y_shape = [3, 2]
        self.dtype = np.float32

    def prepare_data(self):
        self.data_x = self.generate_data(self.x_shape, self.dtype)
        self.data_y = self.generate_data(self.y_shape, self.dtype)

    def forward(self):
        x = paddle.to_tensor(self.data_x, dtype=self.dtype)
        y = paddle.to_tensor(self.data_y, dtype=self.dtype)
        return paddle.where(x > y, x, y)

    def where_cast(self):
        x = paddle.to_tensor(self.data_x, dtype="float32")
        y = paddle.to_tensor(self.data_y, dtype="float32")
        return paddle.where(x > y, x, y)


@ddt
class TestWhereCommon(TestWhere):
    @data(*WHERE_CASE)
    @unpack
    def test_check_output(self, x_shape, y_shape, dtype):
        self.x_shape = x_shape
        self.y_shape = y_shape
        self.dtype = dtype
        self.check_output_gcu_with_cpu(self.forward)


@ddt
class TestWhereF16(TestWhere):
    @data(*WHERE_F16_CASE)
    @unpack
    def test_check_output(self, x_shape, y_shape, dtype):
        self.x_shape = x_shape
        self.y_shape = y_shape
        self.dtype = dtype
        self.check_output_gcu_with_customized(self.forward, self.where_cast)


@ddt
class TestWhereNoneCommon(TestWhere):
    def where_none_forward(self):
        x = paddle.to_tensor(self.data_x, dtype=self.dtype)
        y = paddle.to_tensor(self.data_y, dtype=self.dtype)
        return paddle.where(x > y)

    @data(*WHERE_CASE)
    @unpack
    def test_check_output(self, x_shape, y_shape, dtype):
        self.x_shape = x_shape
        self.y_shape = y_shape
        self.dtype = dtype
        self.check_output_gcu_with_cpu(self.where_none_forward)


@ddt
class TestWhereOtherCommon(TestWhere):
    def prepare_data(self):
        self.data_x = self.generate_data(self.x_shape, self.dtype)
        self.data_y = self.generate_data(self.y_shape, self.dtype)
        self.data_z = self.generate_data(self.x_shape, np.float32)

    def where_other_forward(self):
        x = paddle.to_tensor(self.data_x, dtype=self.dtype)
        y = paddle.to_tensor(self.data_y, dtype=self.dtype)
        z = paddle.to_tensor(self.data_z, dtype="float32")
        return paddle.where(z > 0.5, x, y)

    @data(*WHERE_CASE)
    @unpack
    def test_check_output(self, x_shape, y_shape, dtype):
        self.x_shape = x_shape
        self.y_shape = y_shape
        self.dtype = dtype
        self.check_output_gcu_with_cpu(self.where_other_forward)


if __name__ == "__main__":
    unittest.main()
