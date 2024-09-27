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
INSTANCE_NORM_CASE = [
    # Obsolete Parameters of dynamic_or_pir_mode: running_mean/running_var/use_input_stats/momentum/data_format
    {"x_shape": [2, 3, 32, 32], "dtype": np.float32, "use_input_stats": True, "momentum": 0.9, "eps": 1e-05, "data_format": "NCHW", "with_weight": True, "with_bias": True},
    {"x_shape": [2, 3, 32, 32], "dtype": np.float32, "use_input_stats": True, "momentum": 0.9, "eps": 1e-05, "data_format": "NCHW", "with_weight": True, "with_bias": False},
    {"x_shape": [2, 3, 32, 32], "dtype": np.float32, "use_input_stats": True, "momentum": 0.9, "eps": 1e-05, "data_format": "NCHW", "with_weight": False, "with_bias": True},
    {"x_shape": [2, 32, 32, 32], "dtype": np.float32, "use_input_stats": True, "momentum": 0.9, "eps": 1e-06, "data_format": "NCHW", "with_weight": True, "with_bias": True},

    {"x_shape": [2, 3, 32, 32], "dtype": np.float16, "use_input_stats": True, "momentum": 0.9, "eps": 1e-05, "data_format": "NCHW", "with_weight": True, "with_bias": True},
    {"x_shape": [2, 3, 32, 32], "dtype": np.float16, "use_input_stats": True, "momentum": 0.9, "eps": 1e-05, "data_format": "NCHW", "with_weight": True, "with_bias": False},
    {"x_shape": [2, 3, 32, 32], "dtype": np.float16, "use_input_stats": True, "momentum": 0.9, "eps": 1e-05, "data_format": "NCHW", "with_weight": False, "with_bias": True},
    {"x_shape": [2, 32, 32, 32], "dtype": np.float16, "use_input_stats": True, "momentum": 0.9, "eps": 1e-06, "data_format": "NCHW", "with_weight": True, "with_bias": True},

]
# fmt: on


@ddt
class TestInstanceNorm(TestAPIBase):
    def setUp(self):
        self.init_attrs()

    def init_attrs(self):
        self.x_shape = [2, 3, 32, 32]
        self.dtype = np.float32
        self.use_input_stats = True
        self.momentum = 0.9
        self.eps = 1e-05
        self.data_format = "NCHW"
        self.with_weight = True
        self.with_bias = True
        self.weight_bias_shape = [3]

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
        return paddle.nn.functional.instance_norm(
            x=x,
            running_mean=None,
            running_var=None,
            weight=w if self.with_weight else None,
            bias=b if self.with_bias else None,
            use_input_stats=self.use_input_stats,
            momentum=self.momentum,
            eps=self.eps,
            data_format=self.data_format,
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

    @data(*INSTANCE_NORM_CASE)
    @unpack
    def test_check_output(
        self,
        x_shape,
        dtype,
        use_input_stats,
        momentum,
        eps,
        data_format,
        with_weight,
        with_bias,
    ):
        self.x_shape = x_shape
        self.dtype = dtype
        self.use_input_stats = use_input_stats
        self.momentum = momentum
        self.eps = eps
        self.data_format = data_format
        self.with_weight = with_weight
        self.with_bias = with_bias
        c_dim = -1
        if data_format == "NCHW":
            c_dim = 1
        elif data_format == "NHWC":
            c_dim = 3
        self.weight_bias_shape = [x_shape[c_dim]]
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
