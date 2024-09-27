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
    {"input_size": 8, "hidden_size": 4, "num_layers": 2, "batch": 4, "time_steps": 1, "direction": "bidirect"},
    {"input_size": 16, "hidden_size": 8, "num_layers": 2, "batch": 4, "time_steps": 1, "direction": "bidirect"},
    {"input_size": 16, "hidden_size": 8, "num_layers": 2, "batch": 4, "time_steps": 1, "direction": "forward"},
    {"input_size": 16, "hidden_size": 8, "num_layers": 2, "batch": 4, "time_steps": 1, "direction": "bidirect"},
]
# fmt: on


@ddt
class TestRNN(TestAPIBase):
    def setUp(self):
        self.init_attrs()

    def init_attrs(self):
        self.batch = 4
        self.time_steps = 1
        self.input_size = 8
        self.hidden_size = 4
        self.num_layers = 2

        self.direction = "bidirect"
        self.time_major = True
        self.weight_ih = 1.0
        self.weight_hh = 1.0

        self.dtype = "float32"

    def prepare_data(self):
        self.direction_num = 1
        if self.direction == "bidirect":
            self.direction_num = 2
        self.x = paddle.uniform(
            (self.batch, self.time_steps, self.input_size),
            dtype=self.dtype,
            min=2.0,
            max=2.0,
        )
        self.prev_h = paddle.uniform(
            (self.num_layers * self.direction_num, self.batch, self.hidden_size),
            dtype=self.dtype,
            min=1.0,
            max=1.0,
        )
        self.prev_c = paddle.uniform(
            (self.num_layers * self.direction_num, self.batch, self.hidden_size),
            dtype=self.dtype,
            min=2.0,
            max=2.0,
        )

    def forward(self):
        weight_ih_attr = paddle.nn.initializer.Constant(value=self.weight_ih)
        weight_hh_attr = paddle.nn.initializer.Constant(value=self.weight_hh)
        rnn = paddle.nn.LSTM(
            self.input_size,
            self.hidden_size,
            self.num_layers,
            self.direction,
            time_major=False,
            weight_ih_attr=weight_ih_attr,
            weight_hh_attr=weight_hh_attr,
        )
        return rnn(self.x, (self.prev_h, self.prev_c))

    def expect_output(self):
        out = self.calc_result(self.forward, "cpu")
        return out

    @data(*RMSNORM_CASE)
    @unpack
    def test_check_output(
        self, input_size, hidden_size, num_layers, batch, time_steps, direction
    ):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch = batch
        self.time_steps = time_steps
        self.direction = direction
        rtol = 1e-5
        atol = 1e-5
        self.check_output_gcu_with_customized(
            self.forward, self.expect_output, rtol=rtol, atol=atol
        )


if __name__ == "__main__":
    unittest.main()
