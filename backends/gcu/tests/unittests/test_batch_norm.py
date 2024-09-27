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
BATCH_NORM_CASE = [
    # Only support inference now.
    {"x_shape": [2, 64, 480, 336], "x_dtype": np.float32, "has_weight": True, "has_bias": True, "training": False, "momentum": 0.9, "epsilon": 1e-05, "data_format": 'NCHW', "use_global_stats": True},
    {"x_shape": [2, 32, 228, 228], "x_dtype": np.float32, "has_weight": False, "has_bias": True, "training": False, "momentum": 0.9, "epsilon": 1e-05, "data_format": 'NCHW', "use_global_stats": True},
    {"x_shape": [2, 16, 228, 228], "x_dtype": np.float32, "has_weight": True, "has_bias": False, "training": False, "momentum": 0.9, "epsilon": 1e-05, "data_format": 'NCHW', "use_global_stats": True},
    {"x_shape": [2, 8, 228, 228], "x_dtype": np.float32, "has_weight": False, "has_bias": False, "training": False, "momentum": 0.9, "epsilon": 1e-05, "data_format": 'NCHW', "use_global_stats": True},

    {"x_shape": [2, 64, 480, 336], "x_dtype": np.float16, "has_weight": True, "has_bias": True, "training": False, "momentum": 0.9, "epsilon": 1e-05, "data_format": 'NCHW', "use_global_stats": True},
    {"x_shape": [2, 32, 228, 228], "x_dtype": np.float16, "has_weight": False, "has_bias": True, "training": False, "momentum": 0.9, "epsilon": 1e-05, "data_format": 'NCHW', "use_global_stats": True},
    {"x_shape": [2, 16, 228, 228], "x_dtype": np.float16, "has_weight": True, "has_bias": False, "training": False, "momentum": 0.9, "epsilon": 1e-05, "data_format": 'NCHW', "use_global_stats": True},
    {"x_shape": [2, 8, 228, 228], "x_dtype": np.float16, "has_weight": False, "has_bias": False, "training": False, "momentum": 0.9, "epsilon": 1e-05, "data_format": 'NCHW', "use_global_stats": True},
]
# fmt: on


@ddt
class TestBatchNorm(TestAPIBase):
    def setUp(self):
        self.init_attrs()

    def init_attrs(self):
        self.x_shape = [2, 64, 480, 336]
        self.x_dtype = np.float32
        self.has_weight = True
        self.has_bias = True
        self.training = False
        self.momentum = 0.9
        self.epsilon = 1e-05
        self.data_format = "NCHW"
        self.use_global_stats = False

    def prepare_data(self):
        self.data_x = self.generate_data(self.x_shape, self.x_dtype)
        if self.data_format == "NCHW":
            c_dim = self.x_shape[1]
        else:
            c_dim = self.x_shape[3]
        shape = [c_dim]
        self.running_mean = self.generate_data(shape, self.x_dtype)
        self.running_var = self.generate_data(shape, self.x_dtype)
        if self.has_weight:
            self.weight = self.generate_data(shape, self.x_dtype)
        else:
            self.weight = None
        if self.has_bias:
            self.bias = self.generate_data(shape, self.x_dtype)
        else:
            self.bias = None

    def forward_with_dtype(self, dtype):
        x = paddle.to_tensor(self.data_x, dtype=dtype)
        running_mean = paddle.to_tensor(self.running_mean, dtype=dtype)
        running_var = paddle.to_tensor(self.running_var, dtype=dtype)
        if self.has_weight:
            weight = paddle.to_tensor(self.weight, dtype=dtype)
        else:
            weight = None
        if self.has_bias:
            bias = paddle.to_tensor(self.bias, dtype=dtype)
        else:
            bias = None

        return paddle.nn.functional.batch_norm(
            x=x,
            running_mean=running_mean,
            running_var=running_var,
            weight=weight,
            bias=bias,
            training=self.training,
            momentum=self.momentum,
            epsilon=self.epsilon,
            data_format=self.data_format,
            use_global_stats=self.use_global_stats,
        )

    def forward(self):
        return self.forward_with_dtype(self.x_dtype)

    def batch_norm_cast(self):
        return self.forward_with_dtype(np.float32).astype("float16")

    def expect_output(self):
        if self.x_dtype != np.float16:
            out = self.forward()
        else:
            out = self.batch_norm_cast()
        return out

    @data(*BATCH_NORM_CASE)
    @unpack
    def test_check_output(
        self,
        x_shape,
        x_dtype,
        has_weight,
        has_bias,
        training,
        momentum,
        epsilon,
        data_format,
        use_global_stats,
    ):
        self.x_shape = x_shape
        self.x_dtype = x_dtype
        self.has_weight = has_weight
        self.has_bias = has_bias
        self.training = training
        self.momentum = momentum
        self.epsilon = epsilon
        self.data_format = data_format
        self.use_global_stats = use_global_stats
        rtol = 1e-5
        atol = 1e-5
        if x_dtype == np.float16:
            rtol = 1e-3
            atol = 1e-3
        self.check_output_gcu_with_customized(
            self.forward, self.expect_output, rtol=rtol, atol=atol
        )


if __name__ == "__main__":
    unittest.main()
