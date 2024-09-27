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
LOG_LOSS_CASE = [
    {"x_shape": [16, 1], "dtype": np.float32, "epsilon": 1e-4},
    {"x_shape": [32, 1], "dtype": np.float32, "epsilon": 1e-5},

    {"x_shape": [16, 1], "dtype": np.float16, "epsilon": 1e-4},
    {"x_shape": [32, 1], "dtype": np.float16, "epsilon": 1e-5},

]
# fmt: on


@ddt
class TestLogLoss(TestAPIBase):
    def setUp(self):
        self.init_attrs()

    def init_attrs(self):
        self.x_shape = [16, 1]
        self.label_shape = self.x_shape
        self.dtype = np.float32
        self.epsilon = 1e-4

    def prepare_data(self):
        self.data_x = self.generate_data(self.x_shape, self.dtype)
        self.data_label = self.generate_data(self.label_shape, self.dtype)

    def forward(self):
        x = paddle.to_tensor(self.data_x, dtype=self.dtype)
        label = paddle.to_tensor(self.data_label, dtype=self.dtype)
        return paddle.nn.functional.log_loss(x, label, self.epsilon)

    def log_loss_cast(self):
        x = paddle.to_tensor(self.data_x, dtype="float32")
        label = paddle.to_tensor(self.data_label, dtype="float32")
        out = paddle.nn.functional.log_loss(x, label, self.epsilon)
        return out.astype("float16")

    def expect_output(self):
        if self.dtype != np.float16:
            out = self.calc_result(self.forward, "cpu")
        else:
            out = self.log_loss_cast()
        return out

    @data(*LOG_LOSS_CASE)
    @unpack
    def test_check_output(self, x_shape, dtype, epsilon):
        self.x_shape = x_shape
        self.label_shape = self.x_shape
        self.dtype = dtype
        self.epsilon = epsilon
        rtol = 1e-5
        atol = 1e-5
        if dtype == np.float16:
            rtol = 1e-3
            atol = 1e-3
        self.check_output_gcu_with_customized(
            self.forward, self.expect_output, rtol=rtol, atol=atol
        )


if __name__ == "__main__":
    unittest.main()
