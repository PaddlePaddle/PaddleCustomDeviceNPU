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
IS_CLOSE_CASE = [
    {"shape": [1], "dtype": np.float32, "rtol": 1e-5, "atol": 1e-8, "equal_nan": False},
    {"shape": [9], "dtype": np.float32, "rtol": 1e-5, "atol": 1e-8, "equal_nan": False},
    {"shape": [3, 60], "dtype": np.float32, "rtol": 1e-5, "atol": 1e-8, "equal_nan": False},
    {"shape": [2, 30, 60], "dtype": np.float32, "rtol": 1e-5, "atol": 1e-8, "equal_nan": False},
    {"shape": [2, 2, 30, 60], "dtype": np.float32, "rtol": 1e-5, "atol": 1e-8, "equal_nan": False},

    {"shape": [1], "dtype": np.float32, "rtol": 1e-5, "atol": 1e-8, "equal_nan": True},
    {"shape": [9], "dtype": np.float32, "rtol": 1e-5, "atol": 1e-8, "equal_nan": True},
    {"shape": [3, 60], "dtype": np.float32, "rtol": 1e-5, "atol": 1e-8, "equal_nan": True},
    {"shape": [2, 30, 60], "dtype": np.float32, "rtol": 1e-5, "atol": 1e-8, "equal_nan": True},
    {"shape": [2, 2, 30, 60], "dtype": np.float32, "rtol": 1e-5, "atol": 1e-8, "equal_nan": True},

    {"shape": [1], "dtype": np.float16, "rtol": 1e-5, "atol": 1e-8, "equal_nan": True},
    {"shape": [9], "dtype": np.float16, "rtol": 1e-5, "atol": 1e-8, "equal_nan": True},
    {"shape": [3, 60], "dtype": np.float16, "rtol": 1e-5, "atol": 1e-8, "equal_nan": True},
    {"shape": [2, 30, 60], "dtype": np.float16, "rtol": 1e-5, "atol": 1e-8, "equal_nan": True},
    {"shape": [2, 2, 30, 60], "dtype": np.float16, "rtol": 1e-5, "atol": 1e-8, "equal_nan": True},

    {"shape": [1], "dtype": np.float16, "rtol": 1e-5, "atol": 1e-8, "equal_nan": True},
    {"shape": [9], "dtype": np.float16, "rtol": 1e-5, "atol": 1e-8, "equal_nan": True},
    {"shape": [3, 60], "dtype": np.float16, "rtol": 1e-5, "atol": 1e-8, "equal_nan": True},
    {"shape": [2, 30, 60], "dtype": np.float16, "rtol": 1e-5, "atol": 1e-8, "equal_nan": True},
    {"shape": [2, 2, 30, 60], "dtype": np.float16, "rtol": 1e-5, "atol": 1e-8, "equal_nan": True},

]
# fmt: on


@ddt
class TestIsclose(TestAPIBase):
    def setUp(self):
        self.init_attrs()

    def init_attrs(self):
        self.shape = [3, 6]
        self.dtype = np.float32
        self.rtol = 1e-05
        self.atol = 1e-08
        self.equal_nan = False
        self.data_set = [-6, 0, 6, np.nan, -np.nan]
        self.data_set_size = len(self.data_set)

    def prepare_data(self):
        index = np.random.randint(
            low=0, high=self.data_set_size, size=self.shape, dtype=np.int32
        ).flatten()
        value_x = [self.data_set[i] for i in index]
        value_y = []
        for i in range(len(value_x)):
            val = value_x[i]
            if val not in [np.nan, -np.nan]:
                val = val + self.rtol * np.random.uniform(low=-2, high=2)
            value_y.append(val)
        self.data_x = np.array(value_x, self.dtype).reshape(self.shape)
        self.data_y = np.array(value_y, self.dtype).reshape(self.shape)

    def forward(self):
        x = paddle.to_tensor(self.data_x, dtype=self.dtype)
        y = paddle.to_tensor(self.data_y, dtype=self.dtype)
        return paddle.isclose(x, y, self.rtol, self.atol, self.equal_nan)

    def expect_output(self):
        return np.isclose(
            self.data_x, self.data_y, self.rtol, self.atol, self.equal_nan
        )

    @data(*IS_CLOSE_CASE)
    @unpack
    def test_check_output(self, shape, dtype, rtol, atol, equal_nan):
        self.shape = shape
        self.dtype = dtype
        self.rtol = rtol
        self.atol = atol
        self.equal_nan = equal_nan
        self.check_output_gcu_with_customized(self.forward, self.expect_output)


if __name__ == "__main__":
    unittest.main()
