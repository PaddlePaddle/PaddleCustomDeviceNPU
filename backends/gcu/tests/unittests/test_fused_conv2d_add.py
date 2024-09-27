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
CONV2D_ADD_CASE = [
    {"x_shape": [1, 2, 8, 8], "weight_shape": [4, 2, 3, 3], "dtype": np.float32, "stride": [1, 1], "padding": [0, 0], "dilation": [1, 1], "groups": 1, "data_format": "NCHW", "residual_shape": None},
    {"x_shape": [1, 2, 8, 8], "weight_shape": [4, 2, 3, 3], "dtype": np.float32, "stride": [1, 1], "padding": [0, 0], "dilation": [1, 1], "groups": 1, "data_format": "NCHW", "residual_shape": [1, 4, 6, 6]},
    {"x_shape": [2, 3, 32, 32], "weight_shape": [6, 3, 3, 3], "dtype": np.float32, "stride": [2, 2], "padding": [1, 1], "dilation": [1, 1], "groups": 1, "data_format": "NCHW", "residual_shape": None},
    {"x_shape": [2, 3, 32, 32], "weight_shape": [6, 3, 3, 3], "dtype": np.float32, "stride": [2, 2], "padding": [1, 1], "dilation": [1, 1], "groups": 1, "data_format": "NCHW", "residual_shape": [2, 6, 16, 16]},

    {"x_shape": [1, 2, 8, 8], "weight_shape": [4, 2, 3, 3], "dtype": np.float16, "stride": [1, 1], "padding": [0, 0], "dilation": [1, 1], "groups": 1, "data_format": "NCHW", "residual_shape": None},
    {"x_shape": [1, 2, 8, 8], "weight_shape": [4, 2, 3, 3], "dtype": np.float16, "stride": [1, 1], "padding": [0, 0], "dilation": [1, 1], "groups": 1, "data_format": "NCHW", "residual_shape": [1, 4, 6, 6]},
    {"x_shape": [2, 3, 32, 32], "weight_shape": [6, 3, 3, 3], "dtype": np.float16, "stride": [2, 2], "padding": [1, 1], "dilation": [1, 1], "groups": 1, "data_format": "NCHW", "residual_shape": None},
    {"x_shape": [2, 3, 32, 32], "weight_shape": [6, 3, 3, 3], "dtype": np.float16, "stride": [2, 2], "padding": [1, 1], "dilation": [1, 1], "groups": 1, "data_format": "NCHW", "residual_shape": [2, 6, 16, 16]},

]
# fmt: on

NATIVE_IMPL_DEV = "gcu"


@ddt
class TestFusedConvAdd(TestAPIBase):
    def setUp(self):
        self.init_attrs()

    def init_attrs(self):
        self.x_shape = [2, 3, 32, 32]
        self.w_shape = [6, 3, 3, 3]
        self.dtype = np.float32
        self.stride = [1, 1]
        self.padding = [0, 0]
        self.dilation = [1, 1]
        self.groups = 1
        self.data_format = "NCHW"

        self.has_residual = False
        self.b_shape = [6]
        self.res_shape = [2, 3, 32, 32]

        self.activation = "identity"
        self.padding_algorithm = "EXPLICIT"
        self.split_channels = []
        self.exhaustive_search = False
        self.workspace_size_MB = 512
        self.fuse_alpha = 0.0

    def prepare_data(self):
        self.data_x = self.generate_data(self.x_shape, self.dtype)
        self.data_w = self.generate_data(self.w_shape, self.dtype)
        if self.has_residual:
            self.data_b_or_res = self.generate_data(self.res_shape, self.dtype)
        else:
            self.data_b_or_res = self.generate_data(self.b_shape, self.dtype)

    def forward(self):
        x = paddle.to_tensor(self.data_x, dtype=self.dtype)
        w = paddle.to_tensor(self.data_w, dtype=self.dtype)
        b_or_res = paddle.to_tensor(self.data_b_or_res, dtype=self.dtype)
        return paddle.base.core.eager._run_custom_op(
            "fused_conv2d_add",
            x,
            w,
            b_or_res,
            b_or_res,
            self.stride,
            self.padding,
            self.padding_algorithm,
            self.dilation,
            self.groups,
            self.data_format,
            self.activation,
            self.split_channels,
            self.exhaustive_search,
            self.workspace_size_MB,
            self.fuse_alpha,
        )[0]

    def conv2d_add_impl(self, dtype):
        paddle.set_device(NATIVE_IMPL_DEV)
        x = paddle.to_tensor(self.data_x, dtype=dtype)
        w = paddle.to_tensor(self.data_w, dtype=dtype)
        b_or_res = paddle.to_tensor(self.data_b_or_res, dtype=dtype)
        b = None if self.has_residual else b_or_res
        out = paddle.nn.functional.conv2d(
            x=x,
            weight=w,
            bias=b,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
            data_format=self.data_format,
        )
        if self.has_residual:
            out = out + b_or_res
        return out

    def expect_output(self):
        if NATIVE_IMPL_DEV == "cpu" and self.dtype == np.float16:
            out = self.conv2d_add_impl(np.float32)
        else:
            out = self.conv2d_add_impl(self.dtype)
        return out

    @data(*CONV2D_ADD_CASE)
    @unpack
    def test_check_output(
        self,
        x_shape,
        weight_shape,
        dtype,
        stride,
        padding,
        dilation,
        groups,
        data_format,
        residual_shape,
    ):
        self.x_shape = x_shape
        self.w_shape = weight_shape
        self.dtype = dtype
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.data_format = data_format
        self.res_shape = residual_shape
        self.has_residual = residual_shape is not None
        oc = self.w_shape[0]
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
