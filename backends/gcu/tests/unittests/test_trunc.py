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
TRUNC_CASE = [
    {"shape": [2, 3], "dtype": np.int32},
    {"shape": [2, 3], "dtype": np.float32},
    {"shape": [2, 3], "dtype": np.float16},

    {"shape": [2, 3, 28, 28], "dtype": np.int32},
    {"shape": [2, 3, 28, 28], "dtype": np.float32},
    {"shape": [2, 3, 28, 28], "dtype": np.float16},
]
# fmt: on


@ddt
class TestTrunc(TestAPIBase):
    def setUp(self):
        self.init_attrs()

    def init_attrs(self):
        self.shape = [2, 3]
        self.dtype = np.float32

    def prepare_data(self):
        self.data_x = self.generate_data(self.shape, self.dtype)

    def forward(self):
        x = paddle.to_tensor(self.data_x, dtype=self.dtype)
        return paddle.trunc(x)

    def get_numpy_output(self):
        return np.trunc(self.data_x)

    def expect_output(self):
        if self.dtype != np.float16:
            out = self.calc_result(self.forward, "cpu")
        else:
            out = self.get_numpy_output()
        return out

    @data(*TRUNC_CASE)
    @unpack
    def test_check_output(self, shape, dtype):
        self.shape = shape
        self.dtype = dtype
        self.check_output_gcu_with_customized(self.forward, self.expect_output)


if __name__ == "__main__":
    unittest.main()
