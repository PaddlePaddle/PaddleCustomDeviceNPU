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
HUBER_LOSS_CASE = [
    {"shape": [2, 3], "dtype": np.float32, "reduction": "none", "delta": 1.0},
    {"shape": [2, 3], "dtype": np.float32, "reduction": "mean", "delta": 1.0},
    {"shape": [2, 3], "dtype": np.float32, "reduction": "sum", "delta": 1.0},
    {"shape": [32, 3], "dtype": np.float32, "reduction": "none", "delta": 0.6},
    {"shape": [32, 3], "dtype": np.float32, "reduction": "none", "delta": 1.6},
    {"shape": [32, 3], "dtype": np.float32, "reduction": "none", "delta": 100},
    {"shape": [32, 3], "dtype": np.float32, "reduction": "none", "delta": 0.001},

    {"shape": [32, 3], "dtype": np.float16, "reduction": "none", "delta": 0.6},
    {"shape": [32, 3], "dtype": np.float16, "reduction": "mean", "delta": 1.6},
    {"shape": [32, 3], "dtype": np.float16, "reduction": "none", "delta": 100},
    {"shape": [32, 3], "dtype": np.float16, "reduction": "sum", "delta": 0.001},
]
# fmt: on


def huber_loss_forward(val, delta):
    abs_val = abs(val)
    if abs_val <= delta:
        return 0.5 * val * val
    else:
        return delta * (abs_val - 0.5 * delta)


@ddt
class TestHuberLoss(TestAPIBase):
    def setUp(self):
        self.init_attrs()

    def init_attrs(self):
        self.shape = [2, 3]
        self.dtype = np.float32
        self.reduction = "mean"
        self.delta = 1.0

    def prepare_data(self):
        self.data_x = self.generate_data(self.shape, self.dtype)
        self.data_y = self.generate_data(self.shape, self.dtype)

    def forward(self):
        x = paddle.to_tensor(self.data_x, dtype=self.dtype)
        y = paddle.to_tensor(self.data_y, dtype=self.dtype)
        return paddle.nn.functional.smooth_l1_loss(
            x, y, reduction=self.reduction, delta=self.delta
        )

    def get_numpy_output(self):
        residual = self.data_y - self.data_x
        loss = np.vectorize(huber_loss_forward)(residual, self.delta)
        if self.reduction == "none":
            return loss
        elif self.reduction == "mean":
            return np.mean(loss)
        elif self.reduction == "sum":
            return np.sum(loss)
        return loss

    def expect_output(self):
        if self.dtype != np.float16:
            out = self.calc_result(self.forward, "cpu")
        else:
            out = self.get_numpy_output()
        return out

    @data(*HUBER_LOSS_CASE)
    @unpack
    def test_check_output(self, shape, dtype, reduction, delta):
        self.shape = shape
        self.dtype = dtype
        self.reduction = reduction
        self.delta = delta
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
