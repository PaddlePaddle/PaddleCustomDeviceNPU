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
from paddle import base

# The table retains its original format for better comparison of parameter settings.
# fmt: off
CONV_ADD_ACT_CASE = [

    {"x_shape": [1, 2, 8, 8], "weight_shape": [4, 2, 3, 3], "dtype": np.float32, "stride": [1, 1], "padding": [0, 0], "dilation": [1, 1], "groups": 1, "activation": "relu", "data_format": "NCHW", "residual_shape": None},
    {"x_shape": [1, 2, 8, 8], "weight_shape": [4, 2, 3, 3], "dtype": np.float32, "stride": [1, 1], "padding": [0, 0], "dilation": [1, 1], "groups": 1, "activation": "relu", "data_format": "NCHW", "residual_shape": [1, 4, 6, 6]},
    {"x_shape": [2, 3, 32, 32], "weight_shape": [6, 3, 3, 3], "dtype": np.float32, "stride": [2, 2], "padding": [1, 1], "dilation": [1, 1], "groups": 1, "activation": "relu", "data_format": "NCHW", "residual_shape": None},
    {"x_shape": [2, 3, 32, 32], "weight_shape": [6, 3, 3, 3], "dtype": np.float32, "stride": [2, 2], "padding": [1, 1], "dilation": [1, 1], "groups": 1, "activation": "relu", "data_format": "NCHW", "residual_shape": [2, 6, 16, 16]},
    {"x_shape": [2, 3, 32, 32], "weight_shape": [6, 3, 3, 3], "dtype": np.float32, "stride": [2, 2], "padding": [1, 1], "dilation": [1, 1], "groups": 1, "activation": "identity", "data_format": "NCHW", "residual_shape": [2, 6, 16, 16]},
    {"x_shape": [2, 3, 32, 32], "weight_shape": [6, 3, 3, 3], "dtype": np.float32, "stride": [2, 2], "padding": [1, 1], "dilation": [1, 1], "groups": 1, "activation": "sigmoid", "data_format": "NCHW", "residual_shape": [2, 6, 16, 16]},

    {"x_shape": [2, 3, 32, 32], "weight_shape": [6, 3, 3, 3], "dtype": np.float16, "stride": [2, 2], "padding": [1, 1], "dilation": [1, 1], "groups": 1, "activation": "relu", "data_format": "NCHW", "residual_shape": [2, 6, 16, 16]},
    {"x_shape": [2, 3, 32, 32], "weight_shape": [6, 3, 3, 3], "dtype": np.float16, "stride": [2, 2], "padding": [1, 1], "dilation": [1, 1], "groups": 1, "activation": "sigmoid", "data_format": "NCHW", "residual_shape": [2, 6, 16, 16]},

    # depthwise_conv2d
    {"x_shape": [1, 2, 8, 8], "weight_shape": [2, 1, 3, 3], "dtype": np.float32, "stride": [1, 1], "padding": [0, 0], "dilation": [1, 1], "groups": 2, "activation": "identity", "data_format": "NCHW", "residual_shape": None},
    {"x_shape": [2, 9, 32, 32], "weight_shape": [9, 1, 3, 3], "dtype": np.float32, "stride": [2, 2], "padding": [1, 1], "dilation": [1, 1], "groups": 9, "activation": "identity", "data_format": "NCHW", "residual_shape": None},
    {"x_shape": [2, 9, 32, 32], "weight_shape": [9, 1, 3, 3], "dtype": np.float16, "stride": [2, 2], "padding": [1, 1], "dilation": [1, 1], "groups": 9, "activation": "identity", "data_format": "NCHW", "residual_shape": None},

]
# fmt: on

NATIVE_IMPL_DEV = "gcu"


@ddt
class TestConvAddAct(TestAPIBase):
    def setUp(self):
        self.init_attrs()

    def init_attrs(self):
        self.x_shape = [2, 3, 32, 32]
        self.w_shape = [6, 3, 3, 3]
        self.dtype = np.float32
        self.stride = [1, 1]
        self.padding = [0, 0]
        self.padding_algorithm = "EXPLICIT"
        self.dilation = [1, 1]
        self.groups = 1
        self.activation = "relu"
        self.data_format = "NCHW"
        self.has_residual = False
        self.b_shape = [6]
        self.res_shape = [2, 3, 32, 32]

        self.split_channels = []
        self.exhaustive_search = False
        self.workspace_size_MB = 512
        self.fuse_alpha = 0

    def prepare_data(self):
        self.data_x = self.generate_data(self.x_shape, self.dtype)
        self.data_w = self.generate_data(self.w_shape, self.dtype)
        self.data_b = self.generate_data(self.b_shape, self.dtype)
        self.data_res = (
            self.generate_data(self.res_shape, self.dtype)
            if self.has_residual
            else None
        )

    def forward(self):
        return self.run_program(base.CustomPlace("gcu", 0))[0]

    def conv2d_add_act_impl(self, dtype):
        paddle.set_device(NATIVE_IMPL_DEV)
        paddle.disable_static()
        x = paddle.to_tensor(self.data_x, dtype=dtype)
        w = paddle.to_tensor(self.data_w, dtype=dtype)
        b = paddle.to_tensor(self.data_b, dtype=dtype)
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
            res = paddle.to_tensor(self.data_res, dtype=dtype)
            out = out + res

        assert self.activation in ["identity", "relu", "sigmoid"]
        if self.activation == "relu":
            out = paddle.nn.functional.relu(out)
        elif self.activation == "sigmoid":
            out = paddle.nn.functional.sigmoid(out)
        return out

    def expect_output(self):
        if NATIVE_IMPL_DEV == "cpu" and self.dtype == np.float16:
            out = self.conv2d_add_act_impl(np.float32)
        else:
            out = self.conv2d_add_act_impl(self.dtype)
        return out

    def create_program(self, dtype):
        paddle.seed(2036)
        np.random.seed(2036)
        paddle.enable_static()
        startup_program = paddle.static.Program()
        main_program = paddle.static.Program()

        with paddle.static.program_guard(main_program, startup_program):
            x = paddle.static.data(name="Input", shape=self.x_shape, dtype=dtype)
            w = paddle.static.data(name="Filter", shape=self.w_shape, dtype=dtype)
            b = paddle.static.data(name="Bias", shape=self.b_shape, dtype=dtype)
            res = (
                paddle.static.data(
                    name="ResidualData", shape=self.res_shape, dtype=dtype
                )
                if self.has_residual
                else None
            )
            out = paddle._C_ops.fused_conv2d_add_act(
                x,
                w,
                b,
                res,
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
            )
        # print("DEBUG startup_program:{}".format(startup_program))
        # print("DEBUG main_program:{}".format(main_program))
        cpu_exe = base.Executor(place=base.CPUPlace())
        cpu_exe.run(startup_program)
        return main_program, out[0]

    def run_program(self, place):
        paddle.enable_static()
        prog, out = self.create_program(self.dtype)
        exe = base.Executor(place=place)
        feed = (
            {
                "Input": self.data_x,
                "Filter": self.data_w,
                "Bias": self.data_b,
                "ResidualData": self.data_res,
            }
            if self.has_residual
            else {"Input": self.data_x, "Filter": self.data_w, "Bias": self.data_b}
        )
        out = exe.run(prog, feed=feed, fetch_list=[out])
        return out

    @data(*CONV_ADD_ACT_CASE)
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
        activation,
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
        self.activation = activation
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
