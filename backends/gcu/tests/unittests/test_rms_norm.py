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
import os
import numpy as np
import unittest
from ddt import ddt, data, unpack
from api_base import TestAPIBase

for lib in os.listdir(os.getenv("CUSTOM_DEVICE_ROOT")):
    if lib.endswith(".so"):
        paddle.utils.cpp_extension.extension_utils.load_op_meta_info_and_register_op(
            lib
        )

# The table retains its original format for better comparison of parameter settings.
# fmt: off
RMSNORM_CASE = [
    {"x_shape": [6, 60], "weight_shape": [60], "dtype": np.float32, "epsilon": 1e-6},
    {"x_shape": [6, 60], "weight_shape": [60], "dtype": np.float32, "epsilon": 1e-5},
    {"x_shape": [5, 6, 60], "weight_shape": [60], "dtype": np.float32, "epsilon": 1e-6},
    {"x_shape": [5, 6, 60], "weight_shape": [60], "dtype": np.float32, "epsilon": 1e-5},
    {"x_shape": [6, 60], "weight_shape": [60], "dtype": np.float16, "epsilon": 1e-6},
    {"x_shape": [6, 60], "weight_shape": [60], "dtype": np.float16, "epsilon": 1e-5},
    {"x_shape": [5, 6, 60], "weight_shape": [60], "dtype": np.float16, "epsilon": 1e-6},
    {"x_shape": [5, 6, 60], "weight_shape": [60], "dtype": np.float16, "epsilon": 1e-5},
    # for llama2
    {"x_shape": [4, 512, 5120], "weight_shape": [5120], "dtype": np.float16, "epsilon": 1e-5},

]
# fmt: on


@ddt
class TestRMSNorm(TestAPIBase):
    def setUp(self):
        self.init_attrs()

    def init_attrs(self):
        self.x_shape = [6, 30]
        self.weight_shape = [30]
        self.dtype = np.float32
        self.epsilon = 1e-6

    def prepare_data(self):
        self.data_x = self.generate_data(self.x_shape, self.dtype)
        self.data_weight = self.generate_data(self.weight_shape, self.dtype)

    def forward(self):
        x = paddle.to_tensor(self.data_x, dtype=self.dtype)
        weight = paddle.to_tensor(self.data_weight, dtype=self.dtype)
        return paddle.base.core.eager._run_custom_op(
            "rms_norm_gcu", x, weight, self.epsilon
        )[0]

    def rms_norm_impl(self, dtype):
        x_fp32 = paddle.to_tensor(self.data_x, dtype=np.float32)
        weight = paddle.to_tensor(self.data_weight, dtype=dtype)
        var = paddle.mean(paddle.pow(x_fp32, 2), axis=-1, keepdim=True)
        rstd = paddle.rsqrt(var + self.epsilon)
        y = x_fp32 * rstd
        y = y.cast(dtype)
        return y * weight

    def expect_output(self):
        if self.dtype != np.float16:
            out = self.rms_norm_impl(self.dtype)
        else:
            out = self.rms_norm_impl(np.float32).astype("float16")
        return out

    @data(*RMSNORM_CASE)
    @unpack
    def test_check_output(self, x_shape, weight_shape, dtype, epsilon):
        self.x_shape = x_shape
        self.weight_shape = weight_shape
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
