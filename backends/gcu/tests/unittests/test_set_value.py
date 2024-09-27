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
SET_VALUE_CASE = [
    {"x_shape": [8, 9, 10], "x_dtype": np.float32},
    {"x_shape": [6, 8, 9, 10], "x_dtype": np.float32},
    {"x_shape": [8, 9, 10], "x_dtype": np.int32},
    {"x_shape": [6, 8, 9, 10], "x_dtype": np.int32},
    {"x_shape": [8, 9, 10], "x_dtype": np.int64},
    {"x_shape": [6, 8, 9, 10], "x_dtype": np.int64},
]
# fmt: on


@ddt
class TestSetValueBase(TestAPIBase):
    def setUp(self):
        self.init_attrs()

    def init_attrs(self):
        self.x_shape = [2, 3, 4]
        self.x_dtype = np.float32

    def prepare_data(self):
        self.data_x = self.generate_data(self.x_shape, self.x_dtype)

    def _call_setitem(self, x):
        x[0, 0] = 6

    def forward(self):
        x = paddle.to_tensor(self.data_x, dtype=self.x_dtype)
        self._call_setitem(x)
        return x

    def set_value_cast(self):
        x = paddle.to_tensor(self.data_x, dtype="float32")
        self._call_setitem(x)
        return x

    def expect_output(self):
        if self.x_dtype != np.float16:
            out = self.calc_result(self.forward, "cpu")
        else:
            out = self.set_value_cast()
        return out

    @data(*SET_VALUE_CASE)
    @unpack
    def test_check_output(self, x_shape, x_dtype):
        self.x_shape = x_shape
        self.x_dtype = x_dtype
        self.check_output_gcu_with_customized(self.forward, self.expect_output)


class TestSetValueItemInt(TestSetValueBase):
    def _call_setitem(self, x):
        x[0] = 6


class TestSetValueItemSlice(TestSetValueBase):
    def _call_setitem(self, x):
        x[0:2] = 6


class TestSetValueItemSlice1(TestSetValueBase):
    def _call_setitem(self, x):
        x[0:-1] = 6


class TestSetValueItemSlice2(TestSetValueBase):
    def _call_setitem(self, x):
        x[0:-1, 0:2] = 6


class TestSetValueItemSliceStep(TestSetValueBase):
    def _call_setitem(self, x):
        x[0:2:2] = 6


class TestSetValueItemSliceStep1(TestSetValueBase):
    def _call_setitem(self, x):
        x[0:-1, 0:2, ::2] = 6


class TestSetValueItemSliceStep2(TestSetValueBase):
    def _call_setitem(self, x):
        x[0:, 1:2:2, :] = 6


class TestSetValueItemSliceNegetiveStep(TestSetValueBase):
    def _call_setitem(self, x):
        x[5:2:-1] = 6


class TestSetValueItemEllipsis(TestSetValueBase):
    def _call_setitem(self, x):
        x[0:, ..., 1:] = 6


class TestSetValueItemEllipsis1(TestSetValueBase):
    def _call_setitem(self, x):
        x[0:, ...] = 6


class TestSetValueItemTensor(TestSetValueBase):
    def _call_setitem(self, x):
        zero = paddle.full([], 0, dtype="int32")
        two = paddle.full([], 2, dtype="int64")
        x[zero:two] = 6


class TestSetValueItemTensor1(TestSetValueBase):
    def _call_setitem(self, x):
        value = paddle.full(shape=[], fill_value=3, dtype=self.x_dtype)
        x[0, 1] = value


class TestSetValueItemTensor2(TestSetValueBase):
    def _call_setitem(self, x):
        x[..., 1:] = x[..., :-1].clone()


class TestSetValueItemTensorSliceStep(TestSetValueBase):
    def _call_setitem(self, x):
        value = paddle.full(shape=[], fill_value=6, dtype=self.x_dtype)
        x[0:-1, 0:2, ::2] = value


class TestSetValueItemNone(TestSetValueBase):
    def _call_setitem(self, x):
        x[None] = 6


class TestSetValueItemNone1(TestSetValueBase):
    def _call_setitem(self, x):
        x[0, None, 1] = 6


if __name__ == "__main__":
    unittest.main()
