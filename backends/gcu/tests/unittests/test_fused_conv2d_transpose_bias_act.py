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
CONV2D_TRANSPOSE_BIAS_ACT_CASE = [
    {"x_shape": [1, 2, 8, 8], "weight_shape": [2, 4, 3, 3], "dtype": np.float32, "stride": [1, 1], "padding": [0, 0], "output_padding": [0, 0], "dilation": [1, 1], "groups": 1, "activation": "identity", "data_format": "NCHW"},
    {"x_shape": [1, 2, 8, 8], "weight_shape": [2, 4, 3, 3], "dtype": np.float32, "stride": [1, 1], "padding": [0, 0], "output_padding": [0, 0], "dilation": [1, 1], "groups": 1, "activation": "relu", "data_format": "NCHW"},
    {"x_shape": [1, 2, 8, 8], "weight_shape": [2, 4, 3, 3], "dtype": np.float32, "stride": [1, 1], "padding": [0, 0], "output_padding": [0, 0], "dilation": [1, 1], "groups": 1, "activation": "sigmoid", "data_format": "NCHW"},
    {"x_shape": [1, 2, 8, 8], "weight_shape": [2, 4, 3, 3], "dtype": np.float32, "stride": [1, 1], "padding": [0, 0], "output_padding": [0, 0], "dilation": [1, 1], "groups": 1, "activation": "swish", "data_format": "NCHW"},

    {"x_shape": [2, 3, 32, 32], "weight_shape": [3, 6, 3, 3], "dtype": np.float32, "stride": [2, 2], "padding": [1, 1], "output_padding": [1, 1], "dilation": [1, 1], "groups": 1, "activation": "identity", "data_format": "NCHW"},
    {"x_shape": [2, 3, 32, 32], "weight_shape": [3, 6, 3, 3], "dtype": np.float32, "stride": [2, 2], "padding": [1, 1], "output_padding": [1, 1], "dilation": [1, 1], "groups": 1, "activation": "relu", "data_format": "NCHW"},
    {"x_shape": [2, 3, 32, 32], "weight_shape": [3, 6, 3, 3], "dtype": np.float32, "stride": [2, 2], "padding": [1, 1], "output_padding": [1, 1], "dilation": [1, 1], "groups": 1, "activation": "sigmoid", "data_format": "NCHW"},
    {"x_shape": [2, 3, 32, 32], "weight_shape": [3, 6, 3, 3], "dtype": np.float32, "stride": [2, 2], "padding": [1, 1], "output_padding": [1, 1], "dilation": [1, 1], "groups": 1, "activation": "swish", "data_format": "NCHW"},

    {"x_shape": [2, 3, 32, 32], "weight_shape": [3, 6, 3, 3], "dtype": np.float16, "stride": [2, 2], "padding": [1, 1], "output_padding": [1, 1], "dilation": [1, 1], "groups": 1, "activation": "identity", "data_format": "NCHW"},
    {"x_shape": [2, 3, 32, 32], "weight_shape": [3, 6, 3, 3], "dtype": np.float16, "stride": [2, 2], "padding": [1, 1], "output_padding": [1, 1], "dilation": [1, 1], "groups": 1, "activation": "relu", "data_format": "NCHW"},
    {"x_shape": [2, 3, 32, 32], "weight_shape": [3, 6, 3, 3], "dtype": np.float16, "stride": [2, 2], "padding": [1, 1], "output_padding": [1, 1], "dilation": [1, 1], "groups": 1, "activation": "sigmoid", "data_format": "NCHW"},
    # {"x_shape": [2, 3, 32, 32], "weight_shape": [3, 6, 3, 3], "dtype": np.float16, "stride": [2, 2], "padding": [1, 1], "output_padding": [1, 1], "dilation": [1, 1], "groups": 1, "activation": "swish", "data_format": "NCHW"},


]
# fmt: on

NATIVE_IMPL_DEV = "gcu"


@ddt
class TestFusedConvTransposeBiasAct(TestAPIBase):
    def setUp(self):
        self.init_attrs()

    def init_attrs(self):
        self.x_shape = [2, 3, 32, 32]
        self.w_shape = [6, 3, 3, 3]
        self.dtype = np.float32
        self.stride = [1, 1]
        self.padding = [0, 0]
        self.output_padding = [0, 0]
        self.dilation = [1, 1]
        self.groups = 1
        self.data_format = "NCHW"

        self.has_residual = False
        self.b_shape = [6]

        self.activation = "identity"
        self.padding_algorithm = "EXPLICIT"
        self.output_size = None

    def prepare_data(self):
        self.data_x = self.generate_data(self.x_shape, self.dtype)
        self.data_w = self.generate_data(self.w_shape, self.dtype)
        self.data_b = self.generate_data(self.b_shape, self.dtype)

    def forward(self):
        x = paddle.to_tensor(self.data_x, dtype=self.dtype)
        w = paddle.to_tensor(self.data_w, dtype=self.dtype)
        b = paddle.to_tensor(self.data_b, dtype=self.dtype)
        return paddle.base.core.eager._run_custom_op(
            "fused_conv2d_transpose_bias_act",
            x,
            w,
            b,
            self.stride,
            self.padding,
            self.output_padding,
            self.output_size,
            self.padding_algorithm,
            self.groups,
            self.dilation,
            self.data_format,
            self.activation,
        )[0]

    def conv2d_transpose_bias_act_impl(self, dtype):
        paddle.set_device(NATIVE_IMPL_DEV)
        x = paddle.to_tensor(self.data_x, dtype=dtype)
        w = paddle.to_tensor(self.data_w, dtype=dtype)
        b = paddle.to_tensor(self.data_b, dtype=dtype)
        out = paddle.nn.functional.conv2d_transpose(
            x=x,
            weight=w,
            bias=b,
            stride=self.stride,
            padding=self.padding,
            output_padding=self.output_padding,
            groups=self.groups,
            dilation=self.dilation,
            data_format=self.data_format,
            output_size=self.output_size,
        )
        assert self.activation in ["identity", "relu", "sigmoid", "swish"]
        if self.activation == "relu":
            out = paddle.nn.functional.relu(out)
        elif self.activation == "sigmoid":
            out = paddle.nn.functional.sigmoid(out)
        elif self.activation == "swish":
            out = paddle.nn.functional.swish(out)
        return out

    def expect_output(self):
        if NATIVE_IMPL_DEV == "cpu" and self.dtype == np.float16:
            out = self.conv2d_transpose_bias_act_impl(np.float32)
        else:
            out = self.conv2d_transpose_bias_act_impl(self.dtype)
        return out

    @data(*CONV2D_TRANSPOSE_BIAS_ACT_CASE)
    @unpack
    def test_check_output(
        self,
        x_shape,
        weight_shape,
        dtype,
        stride,
        padding,
        output_padding,
        dilation,
        groups,
        activation,
        data_format,
    ):
        self.x_shape = x_shape
        self.w_shape = weight_shape
        self.dtype = dtype
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.dilation = dilation
        self.groups = groups
        self.activation = activation
        self.data_format = data_format
        oc = self.w_shape[1]
        self.b_shape = [oc]
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
