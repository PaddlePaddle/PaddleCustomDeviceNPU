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
from paddle.base import Program, program_guard
from paddle.base.layer_helper import LayerHelper

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
        self.dilation = [1, 1]
        self.groups = 1
        self.activation = "relu"
        self.data_format = "NCHW"
        self.has_residual = False
        self.b_shape = [6]
        self.res_shape = [2, 3, 32, 32]

    def prepare_data(self):
        self.data_x = self.generate_data(self.x_shape, self.dtype)
        self.data_w = self.generate_data(self.w_shape, self.dtype)
        self.data_b = self.generate_data(self.b_shape, self.dtype)
        self.data_res = (
            self.generate_data(self.res_shape, self.dtype)
            if self.has_residual
            else None
        )
        dtype = self.dtype if self.dtype != np.float16 else np.float32
        self.prog, self.out_names = self.create_program(dtype)

    def forward(self):
        return self.run_program(self.prog, base.CustomPlace("gcu", 0), self.out_names)[
            0
        ]

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
        startup_program = Program()
        main_program = Program()

        with program_guard(main_program, startup_program):
            attrs = {
                "strides": self.stride,
                "paddings": self.padding,
                "padding_algorithm": "EXPLICIT",
                "dilations": self.dilation,
                "groups": self.groups,
                "data_format": self.data_format,
                "activation": self.activation,
                "split_channels": [],
                "exhaustive_search": False,
                "workspace_size_MB": 512,
                "fuse_alpha": 0,
            }
            helper = LayerHelper("fused_conv2d_add_act")
            x = helper.create_variable(name="Input", shape=self.x_shape, dtype=dtype)
            w = helper.create_variable(name="Filter", shape=self.w_shape, dtype=dtype)
            b = helper.create_variable(name="Bias", shape=self.b_shape, dtype=dtype)
            res = helper.create_variable(
                name="ResidualData", shape=self.b_shape, dtype=dtype
            )
            out = helper.create_variable_for_type_inference(dtype=dtype)

            inputs = (
                {"Input": x, "Filter": w, "Bias": b, "ResidualData": res}
                if self.has_residual
                else {"Input": x, "Filter": w, "Bias": b}
            )
            outputs = {"Output": out}
            helper.append_op(
                type="fused_conv2d_add_act",
                inputs=inputs,
                outputs=outputs,
                attrs=attrs,
            )
        # print("DEBUG startup_program:{}".format(startup_program))
        # print("DEBUG main_program:{}".format(main_program))
        cpu_exe = base.Executor(place=base.CPUPlace())
        cpu_exe.run(startup_program)
        return main_program, out.name

    def run_program(self, main_program, place, out_name):
        paddle.enable_static()
        exe = base.Executor(place=place)
        x = self.data_x
        w = self.data_w
        b = self.data_b
        res = self.data_res
        feed = (
            {"Input": x, "Filter": w, "Bias": b, "ResidualData": res}
            if self.has_residual
            else {"Input": x, "Filter": w, "Bias": b}
        )
        out = exe.run(main_program, feed=feed, fetch_list=[out_name])
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
