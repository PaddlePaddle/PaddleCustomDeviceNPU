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
CHOLESKY_CASE = [

    {"shape": [28, 28], "dtype": np.float32, "upper": True},
    {"shape": [28, 28], "dtype": np.float32, "upper": False},
    {"shape": [3, 28, 28], "dtype": np.float32, "upper": True},
    {"shape": [3, 28, 28], "dtype": np.float32, "upper": False},

    # TODO: Aten's support for topsatenCholesky is not yet complete.
    # {"shape": [28, 28], "dtype": np.float16, "upper": True},
    # {"shape": [28, 28], "dtype": np.float16, "upper": False},
    # {"shape": [3, 28, 28], "dtype": np.float16, "upper": True},
    # {"shape": [3, 28, 28], "dtype": np.float16, "upper": False},
]
# fmt: on


@ddt
class TestCholesky(TestAPIBase):
    def setUp(self):
        self.init_attrs()

    def init_attrs(self):
        self.shape = [2, 3]
        self.dtype = np.float32
        self.upper = False

    def prepare_data(self):
        trans_dims = list(range(len(self.shape) - 2)) + [
            len(self.shape) - 1,
            len(self.shape) - 2,
        ]
        root_data = np.random.random(self.shape).astype(self.dtype)
        # construct symmetric positive-definite matrice
        self.data_x = np.matmul(root_data, root_data.transpose(trans_dims)) + 1e-03

    def forward(self):
        x = paddle.to_tensor(self.data_x, dtype=self.dtype)
        return paddle.cholesky(x, self.upper)

    def get_numpy_output(self):
        return np.linalg.cholesky(self.data_x, self.upper).astype(self.dtype)

    def expect_output(self):
        if self.dtype != np.float16:
            out = self.calc_result(self.forward, "cpu")
        else:
            out = self.get_numpy_output()
        return out

    @data(*CHOLESKY_CASE)
    @unpack
    def test_check_output(self, shape, dtype, upper):
        self.shape = shape
        self.dtype = dtype
        self.upper = upper
        # The accuracy of fp32 is only 1e-3
        rtol = 1e-3
        atol = 1e-3
        if dtype == np.float16:
            rtol = 1e-3
            atol = 1e-3
        self.check_output_gcu_with_customized(
            self.forward, self.expect_output, rtol=rtol, atol=atol
        )


if __name__ == "__main__":
    unittest.main()
