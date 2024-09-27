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
LAYER_NORM_CASE = [
    {"x_shape": [2, 3, 32, 32], "dtype": np.float32, "normalized_shape": [3, 32, 32], "epsilon": 1e-05, "with_weight": True, "with_bias": True},
    {"x_shape": [2, 3, 32, 32], "dtype": np.float32, "normalized_shape": [3, 32, 32], "epsilon": 1e-05, "with_weight": True, "with_bias": False},
    {"x_shape": [2, 3, 32, 32], "dtype": np.float32, "normalized_shape": [3, 32, 32], "epsilon": 1e-05, "with_weight": False, "with_bias": True},

    {"x_shape": [2, 3, 32, 32], "dtype": np.float32, "normalized_shape": 32, "epsilon": 1e-05, "with_weight": True, "with_bias": True},
    {"x_shape": [2, 3, 32, 32], "dtype": np.float32, "normalized_shape": [32], "epsilon": 1e-05, "with_weight": True, "with_bias": True},
    {"x_shape": [2, 3, 32, 32], "dtype": np.float32, "normalized_shape": [32, 32], "epsilon": 1e-05, "with_weight": True, "with_bias": True},

    {"x_shape": [2, 3, 32, 32], "dtype": np.float16, "normalized_shape": [3, 32, 32], "epsilon": 1e-05, "with_weight": True, "with_bias": True},
    {"x_shape": [2, 3, 32, 32], "dtype": np.float16, "normalized_shape": [3, 32, 32], "epsilon": 1e-05, "with_weight": True, "with_bias": False},
    {"x_shape": [2, 3, 32, 32], "dtype": np.float16, "normalized_shape": [3, 32, 32], "epsilon": 1e-05, "with_weight": False, "with_bias": True},


]
# fmt: on


@ddt
class TestLayerNorm(TestAPIBase):
    def setUp(self):
        self.init_attrs()

    def init_attrs(self):
        self.x_shape = [2, 3, 32, 32]
        self.dtype = np.float32
        self.normalized_shape = [3, 32, 32]
        self.epsilon = 1e-05
        self.with_weight = True
        self.with_bias = True
        self.weight_bias_shape = [np.prod(self.normalized_shape)]

    def prepare_data(self):
        self.data_x = self.generate_data(self.x_shape, self.dtype)
        if self.with_weight:
            self.data_w = self.generate_data(self.weight_bias_shape, self.dtype)
        if self.with_bias:
            self.data_b = self.generate_data(self.weight_bias_shape, self.dtype)

    def forward_with_dtype(self, dtype):
        x = paddle.to_tensor(self.data_x, dtype=dtype)
        if self.with_weight:
            w = paddle.to_tensor(self.data_w, dtype=dtype)
        if self.with_bias:
            b = paddle.to_tensor(self.data_b, dtype=dtype)
        return paddle.nn.functional.layer_norm(
            x=x,
            normalized_shape=self.normalized_shape,
            weight=w if self.with_weight else None,
            bias=b if self.with_bias else None,
            epsilon=self.epsilon,
        )

    def forward(self):
        return self.forward_with_dtype(self.dtype)

    def norm_cast(self):
        return self.forward_with_dtype(np.float32).astype("float16")

    def expect_output(self):
        if self.dtype != np.float16:
            out = self.forward()
        else:
            out = self.norm_cast()
        return out

    @data(*LAYER_NORM_CASE)
    @unpack
    def test_check_output(
        self, x_shape, dtype, normalized_shape, epsilon, with_weight, with_bias
    ):
        self.x_shape = x_shape
        self.dtype = dtype
        self.normalized_shape = normalized_shape
        self.epsilon = epsilon
        self.with_weight = with_weight
        self.with_bias = with_bias
        self.weight_bias_shape = [np.prod(self.normalized_shape)]
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
