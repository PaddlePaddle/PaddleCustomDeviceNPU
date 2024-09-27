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
MESHGRID_CASE = [
    {"x_shape": [100], "x_dtype": np.float32, "input_num": 2},
    {"x_shape": [10], "x_dtype": np.float32, "input_num": 3},
    {"x_shape": [2], "x_dtype": np.float32, "input_num": 6},

    {"x_shape": [100], "x_dtype": np.float16, "input_num": 2},
    {"x_shape": [10], "x_dtype": np.float16, "input_num": 3},
    {"x_shape": [2], "x_dtype": np.float16, "input_num": 6},

]
# fmt: on


@ddt
class TestMeshgrid(TestAPIBase):
    def setUp(self):
        self.init_attrs()

    def init_attrs(self):
        self.x_shape = [100]
        self.x_dtype = np.float32
        self.input_num = 2

    def prepare_data(self):
        self.data_x = []
        for i in range(self.input_num):
            self.data_x.append(self.generate_data(self.x_shape, self.x_dtype))

    def forward(self):
        x = []
        for i in range(len(self.data_x)):
            x.append(paddle.to_tensor(self.data_x[i], dtype=self.x_dtype))
        return paddle.meshgrid(x)

    def meshgrid_cast(self):
        x = []
        for i in range(len(self.data_x)):
            x.append(paddle.to_tensor(self.data_x[i], dtype="float32"))
        outs = paddle.meshgrid(x)
        y = []
        for i in range(len(outs)):
            y.append(outs[i].astype("float16"))
        return y

    def expect_output(self):
        if self.x_dtype != np.float16:
            out = self.calc_result(self.forward, "cpu")
        else:
            out = self.meshgrid_cast()
        return out

    @data(*MESHGRID_CASE)
    @unpack
    def test_check_output(self, x_shape, x_dtype, input_num):
        self.x_shape = x_shape
        self.x_dtype = x_dtype
        self.input_num = input_num
        self.check_output_gcu_with_customized(self.forward, self.expect_output)


if __name__ == "__main__":
    unittest.main()
