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
DIAGONAL_CASE = [
    {"shape": [2, 2, 3], "dtype": np.float32, "offset": 0, "axis1": 0, "axis2": 1},
    {"shape": [2, 2, 3], "dtype": np.float32, "offset": 0, "axis1": 2, "axis2": 1},
    {"shape": [2, 2, 3], "dtype": np.float32, "offset": 1, "axis1": 0, "axis2": 1},
    {"shape": [2, 2, 3], "dtype": np.float32, "offset": 0, "axis1": 1, "axis2": 2}
]
# fmt: on


class TestDiagonal(TestAPIBase):
    def setUp(self):
        self.init_attrs()

    def init_attrs(self):
        self.shape = [3]
        self.dtype = np.float32
        self.offset = 0
        self.axis1 = 0
        self.axis2 = 1

    def prepare_data(self):
        self.data_x = self.generate_data(self.shape, self.dtype)

    def forward(self):
        return paddle.diagonal(
            x=paddle.to_tensor(self.data_x),
            offset=self.offset,
            axis1=self.axis1,
            axis2=self.axis2,
        )


@ddt
class TestDiagonalCommon(TestDiagonal):
    @data(*DIAGONAL_CASE)
    @unpack
    def test_check_output(self, shape, dtype, offset, axis1, axis2):
        self.shape = shape
        self.dtype = dtype
        self.offset = offset
        self.axis1 = axis1
        self.axis2 = axis2
        self.check_output_gcu_with_cpu(self.forward)


if __name__ == "__main__":
    unittest.main()
