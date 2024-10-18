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
FC_CASE = [
    {"x_shape": [2, 3, 28, 28], "dtype": np.float32, "num_flatten_dims": 2, "has_bias": True},
    {"x_shape": [2, 3, 28, 28], "dtype": np.float32, "num_flatten_dims": 2, "has_bias": False},
    {"x_shape": [2, 3, 28, 28], "dtype": np.float32, "num_flatten_dims": 1, "has_bias": True},
    {"x_shape": [2, 3, 28, 28], "dtype": np.float32, "num_flatten_dims": 3, "has_bias": True},

    {"x_shape": [2, 3, 28, 28], "dtype": np.float16, "num_flatten_dims": 2, "has_bias": True},
    {"x_shape": [2, 3, 28, 28], "dtype": np.float16, "num_flatten_dims": 2, "has_bias": False},
    {"x_shape": [2, 3, 28, 28], "dtype": np.float16, "num_flatten_dims": 1, "has_bias": True},
    {"x_shape": [2, 3, 28, 28], "dtype": np.float16, "num_flatten_dims": 3, "has_bias": True},

]
# fmt: on


@ddt
class TestFc(TestAPIBase):
    def setUp(self):
        self.init_attrs()

    def init_attrs(self):
        self.x_shape = [3, 6]
        self.w_shape = [6, 6]
        self.b_shape = [6]
        self.dtype = np.float32
        self.num_flatten_dims = 1
        self.has_bias = True

    def prepare_data(self):
        self.data_x = self.generate_data(self.x_shape, self.dtype)
        self.data_w = self.generate_data(self.w_shape, self.dtype)
        self.data_b = (
            self.generate_data(self.b_shape, self.dtype) if self.has_bias else None
        )

    def forward(self):
        return self.run_program(base.CustomPlace("gcu", 0))[0]

    def expect_output(self):
        cast_inputs = True if self.dtype == np.float16 else False
        out = self.run_program(base.CPUPlace(), cast_inputs)
        if self.dtype == np.float16:
            out[0] = out[0].astype(self.dtype)
        return out[0]

    def create_program(self, num_flatten_dims, dtype):
        paddle.seed(2036)
        np.random.seed(2036)
        paddle.enable_static()
        startup_program = paddle.static.Program()
        main_program = paddle.static.Program()

        with paddle.static.program_guard(main_program, startup_program):
            x = paddle.static.data(name="Input", shape=self.x_shape, dtype=dtype)
            w = paddle.static.data(name="W", shape=self.w_shape, dtype=dtype)
            b = (
                paddle.static.data(name="Bias", shape=self.b_shape, dtype=dtype)
                if self.has_bias
                else None
            )
            out = paddle._C_ops.fc(x, w, b, num_flatten_dims, "", False)
        # print("DEBUG startup_program:{}".format(startup_program))
        # print("DEBUG main_program:{}".format(main_program))
        cpu_exe = base.Executor(place=base.CPUPlace())
        cpu_exe.run(startup_program)
        return main_program, out

    def run_program(self, place, cast_inputs=False):
        paddle.enable_static()
        dtype = (
            np.float32
            if (place._type() == base.CPUPlace()._type() and self.dtype == np.float16)
            else self.dtype
        )
        prog, outs = self.create_program(self.num_flatten_dims, dtype)
        exe = base.Executor(place=place)
        if cast_inputs:
            x = self.data_x.astype(np.float32)
            w = self.data_w.astype(np.float32)
            if self.has_bias:
                b = self.data_b.astype(np.float32)
        else:
            x = self.data_x
            w = self.data_w
            b = self.data_b
        feed = (
            {"Input": x, "W": w, "Bias": b} if self.has_bias else {"Input": x, "W": w}
        )
        out = exe.run(prog, feed=feed, fetch_list=[outs])
        return out

    @data(*FC_CASE)
    @unpack
    def test_check_output(self, x_shape, dtype, num_flatten_dims, has_bias):
        self.x_shape = x_shape
        self.dtype = dtype
        self.num_flatten_dims = num_flatten_dims
        self.has_bias = has_bias
        dim = 1
        for i in range(num_flatten_dims, len(x_shape)):
            dim = dim * x_shape[i]
        self.w_shape = [dim, dim]
        self.b_shape = [dim]
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
