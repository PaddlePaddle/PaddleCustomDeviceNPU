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
HARD_SIGMOID_CASE = [
    {"x_shape": [2, 3, 1, 1], "x_dtype": np.float32, "slope": 0.125, "offset": 0.4},
    {"x_shape": [2, 3, 32, 32], "x_dtype": np.int32, "slope": 0.125, "offset": 0.4},
    {"x_shape": [2, 4, 4], "x_dtype": np.float32, "slope": 0.16666667, "offset": 0.5},
    {"x_shape": [2, 4, 4], "x_dtype": np.int32, "slope": 0.16666667, "offset": 0.5},
    {"x_shape": [2, 3, 32, 32], "x_dtype": np.float16, "slope": 0.16666667, "offset": 0.5},
    {"x_shape": [2, 4, 4], "x_dtype": np.float16, "slope": 0.16666667, "offset": 0.5},
    {"x_shape": [64, 40, 1, 1], "x_dtype": np.float32, "slope": 0.2, "offset": 0.5},
    {"x_shape": [64, 40, 1, 1], "x_dtype": np.float32, "slope": 0.2, "offset": 0.5},
    {"x_shape": [64, 168, 1, 1], "x_dtype": np.float32, "slope": 0.2, "offset": 0.5},
    {"x_shape": [64, 232, 1, 1], "x_dtype": np.float32, "slope": 0.2, "offset": 0.5},
    {"x_shape": [64, 336, 1, 1], "x_dtype": np.float32, "slope": 0.2, "offset": 0.5},
]
# fmt: on


@ddt
class TestHardSigmoid(TestAPIBase):
    def setUp(self):
        self.rtol = 1e-5
        self.atol = 1e-5
        self.init_attrs()

    def prepare_data(self):
        self.init_api_and_data()

    def init_attrs(self):
        self.x_shape = [2, 3, 32, 32]
        self.x_dtype = np.float32
        self.slope = 1 / 6
        self.offset = 0.5
        self.support_dtype = [np.float32, np.float16, np.int32]

    def init_attrs(self):
        self.support_dtype = [np.float32, np.float16]

    def forward(self):
        x = paddle.to_tensor(self.data_x, dtype=self.x_dtype)
        return paddle.nn.functional.hardsigmoid(x, self.slope, self.offset)

    def init_api_and_data(self):
        self.data_x = self.generate_data(self.x_shape, self.x_dtype) * 20 - 10
        # print("init_api_and_data, data_x:{}".format(self.data_x), flush=True)

    def expect_output(self):
        if self.x_dtype != np.float16:
            out = self.calc_result(self.forward, "cpu")
        else:
            out = self.get_numpy_out()
        return out

    def get_numpy_out(self):
        return np.clip(self.data_x * self.slope + self.offset, 0, 1)

    @data(*HARD_SIGMOID_CASE)
    @unpack
    def test_check_output(self, x_shape, x_dtype, slope, offset):
        self.x_shape = x_shape
        self.x_dtype = x_dtype
        self.slope = slope
        self.offset = offset
        if x_dtype not in self.support_dtype:
            return
        rtol = self.rtol
        atol = self.atol
        if x_dtype == np.float16:
            rtol = 1e-3
            atol = 1e-3
        self.check_output_gcu_with_customized(
            self.forward, self.expect_output, rtol=rtol, atol=atol
        )


if __name__ == "__main__":
    unittest.main()
