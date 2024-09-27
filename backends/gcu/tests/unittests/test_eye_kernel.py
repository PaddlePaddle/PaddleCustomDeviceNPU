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
EYE_CASE = [
    {"num_rows": 3, "num_cols": 3, "dtype": np.float32},
    {"num_rows": 3, "num_cols": 2, "dtype": np.float32},
    {"num_rows": 3, "num_cols": 5, "dtype": np.float32},
]
# fmt: on


class TestEye(TestAPIBase):
    def setUp(self):
        self.init_attrs()

    def init_attrs(self):
        self.num_rows = 3
        self.num_cols = 3
        self.dtype = np.float32

    def forward(self):
        return paddle.eye(
            num_rows=self.num_rows, num_columns=self.num_cols, dtype=self.dtype
        )

    def check_output_gcu_with_cpu(self, forward, rtol=1e-5, atol=1e-5):
        gcu_out = self.calc_result(forward, "gcu")
        cpu_out = self.calc_result(forward, "cpu")
        self.check_value(gcu_out, cpu_out, rtol, atol)


@ddt
class TestEyeCommon(TestEye):
    @data(*EYE_CASE)
    @unpack
    def test_check_output(self, num_rows, num_cols, dtype):
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.dtype = dtype
        self.check_output_gcu_with_cpu(self.forward)


if __name__ == "__main__":
    unittest.main()
