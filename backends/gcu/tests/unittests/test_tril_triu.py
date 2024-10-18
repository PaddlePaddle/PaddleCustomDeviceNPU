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
from paddle.base.layer_helper import LayerHelper

# The table retains its original format for better comparison of parameter settings.
# fmt: off
TRIL_TRIU_CASE = [
    #  # Triu
    {"x_shape": [2, 3, 28, 28], "dtype": np.float32, "diagonal": 0, "lower": False},
    {"x_shape": [2, 3, 28, 28], "dtype": np.float32, "diagonal": 2, "lower": False},
    {"x_shape": [2, 3, 28, 28], "dtype": np.float32, "diagonal": -2, "lower": False},
    {"x_shape": [2, 3, 28, 28], "dtype": np.float16, "diagonal": 0, "lower": False},
    {"x_shape": [2, 3, 28, 28], "dtype": np.float16, "diagonal": 2, "lower": False},
    {"x_shape": [2, 3, 28, 28], "dtype": np.float16, "diagonal": -2, "lower": False},

    # Tril
    {"x_shape": [2, 3, 28, 28], "dtype": np.float32, "diagonal": 0, "lower": True},
    {"x_shape": [2, 3, 28, 28], "dtype": np.float32, "diagonal": 2, "lower": True},
    {"x_shape": [2, 3, 28, 28], "dtype": np.float32, "diagonal": -2, "lower": True},
    {"x_shape": [2, 3, 28, 28], "dtype": np.float16, "diagonal": 0, "lower": True},
    {"x_shape": [2, 3, 28, 28], "dtype": np.float16, "diagonal": 2, "lower": True},
    {"x_shape": [2, 3, 28, 28], "dtype": np.float16, "diagonal": -2, "lower": True},

]
# fmt: on


@ddt
class TestTrilTriu(TestAPIBase):
    def setUp(self):
        self.init_attrs()

    def init_attrs(self):
        self.x_shape = [3, 6]
        self.dtype = np.float32
        self.diagonal = 0
        self.lower = False

    def prepare_data(self):
        self.data_x = self.generate_data(self.x_shape, self.dtype)
        dtype = self.dtype if self.dtype != np.float16 else np.float32
        self.prog, self.out_names = self.create_program(
            dtype, self.diagonal, self.lower
        )

    def forward(self):
        return self.run_program(self.prog, base.CustomPlace("gcu", 0), self.out_names)[
            0
        ]

    def expect_output(self):
        cast_inputs = True if self.dtype == np.float16 else False
        out = self.run_program(self.prog, base.CPUPlace(), self.out_names, cast_inputs)
        if self.dtype == np.float16:
            out[0] = out[0].astype(self.dtype)
        return out[0]

    def create_program(self, dtype, diagonal, lower):
        paddle.seed(2036)
        np.random.seed(2036)
        paddle.enable_static()
        with paddle.pir_utils.OldIrGuard():
            startup_program = paddle.static.Program()
            main_program = paddle.static.Program()

            with paddle.static.program_guard(main_program, startup_program):
                attrs = {
                    "diagonal": diagonal,
                    "lower": lower,
                }
                helper = LayerHelper("tril_triu")
                x = helper.create_variable(name="X", shape=self.x_shape, dtype=dtype)
                out = helper.create_variable_for_type_inference(dtype=dtype)

                inputs = {"X": x}
                outputs = {"Out": out}
                helper.append_op(
                    type="tril_triu",
                    inputs=inputs,
                    outputs=outputs,
                    attrs=attrs,
                )
            # print("DEBUG startup_program:{}".format(startup_program))
            # print("DEBUG main_program:{}".format(main_program))
            cpu_exe = base.Executor(place=base.CPUPlace())
            cpu_exe.run(startup_program)
        return main_program, out.name

    def run_program(self, main_program, place, out_name, cast_inputs=False):
        paddle.enable_static()
        with paddle.pir_utils.OldIrGuard():
            exe = base.Executor(place=place)
            if cast_inputs:
                x = self.data_x.astype(np.float32)
            else:
                x = self.data_x
            feed = {"X": x}
            out = exe.run(main_program, feed=feed, fetch_list=[out_name])
        return out

    @data(*TRIL_TRIU_CASE)
    @unpack
    def test_check_output(self, x_shape, dtype, diagonal, lower):
        self.x_shape = x_shape
        self.dtype = dtype
        self.diagonal = diagonal
        self.lower = lower
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
