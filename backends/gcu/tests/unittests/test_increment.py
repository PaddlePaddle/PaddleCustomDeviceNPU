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
INCREMENT_CASE = [
    {"shape": [1], "dtype": np.float32, "value": 1},
    {"shape": [1], "dtype": np.int32, "value": 1},
    {"shape": [], "dtype": np.float32, "value": 1},
    {"shape": [1], "dtype": np.float32, "value": 2},
    {"shape": [], "dtype": np.float32, "value": 2},
]
# fmt: on


class TestIncrement(TestAPIBase):
    def setUp(self):
        self.init_attrs()

    def init_attrs(self):
        self.shape = [3]
        self.dtype = np.float32
        self.value = 0

    def prepare_data(self):
        self.data_x = self.generate_data(self.shape, self.dtype)

    def forward(self):
        return paddle.increment(x=paddle.to_tensor(self.data_x), value=self.value)


@ddt
class TestDiagCommon(TestIncrement):
    @data(*INCREMENT_CASE)
    @unpack
    def test_check_output(self, shape, dtype, value):
        self.shape = shape
        self.dtype = dtype
        self.value = value
        self.check_output_gcu_with_cpu(self.forward)


if __name__ == "__main__":
    unittest.main()
