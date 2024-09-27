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
ONE_HOT_CASE = [
    {"x_shape": [4], "dtype": np.int32, "num_classes": 5},
    {"x_shape": [3, 4], "dtype": np.int32, "num_classes": 10},
    {"x_shape": [2, 3, 16], "dtype": np.int32, "num_classes": 20},

]
# fmt: on


@ddt
class TestOneHot(TestAPIBase):
    def setUp(self):
        self.init_attrs()

    def init_attrs(self):
        self.x_shape = [4]
        self.dtype = np.int32
        self.num_classes = 5

    def prepare_data(self):
        self.data_x = self.generate_data(self.x_shape, self.dtype) % self.num_classes

    def forward(self):
        x = paddle.to_tensor(self.data_x, dtype=self.dtype)
        return paddle.nn.functional.one_hot(x, self.num_classes)

    def expect_output(self):
        return self.calc_result(self.forward, "cpu")

    @data(*ONE_HOT_CASE)
    @unpack
    def test_check_output(self, x_shape, dtype, num_classes):
        self.x_shape = x_shape
        self.dtype = dtype
        self.num_classes = num_classes
        self.check_output_gcu_with_customized(self.forward, self.expect_output)


if __name__ == "__main__":
    unittest.main()
