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
LERP_CASE = [
    {"x_shape": [2, 3, 28, 28], "x_dtype": np.float32, "weight": 0.5},
    {"x_shape": [2, 3, 28, 28], "x_dtype": np.float32, "weight": None},

    {"x_shape": [2, 3, 28, 28], "x_dtype": np.float16, "weight": 0.5},
    {"x_shape": [2, 3, 28, 28], "x_dtype": np.float16, "weight": None},

]
# fmt: on


@ddt
class TestLerp(TestAPIBase):
    def setUp(self):
        self.init_attrs()

    def init_attrs(self):
        self.x_shape = [2, 3, 28, 28]
        self.x_dtype = np.float32
        self.weight = 0.5

    def prepare_data(self):
        self.data_x = self.generate_data(self.x_shape, self.x_dtype)
        self.data_y = self.data_x * 6
        self.data_w = (
            self.generate_data(self.x_shape, self.x_dtype)
            if self.weight is None
            else self.weight
        )

    def forward(self):
        x = paddle.to_tensor(self.data_x, dtype=self.x_dtype)
        y = paddle.to_tensor(self.data_y, dtype=self.x_dtype)
        w = (
            paddle.to_tensor(self.data_w, dtype=self.x_dtype)
            if self.weight is None
            else self.weight
        )
        return paddle.lerp(x, y, w)

    def lerp_cast(self):
        x = paddle.to_tensor(self.data_x, dtype="float32")
        y = paddle.to_tensor(self.data_y, dtype="float32")
        w = (
            paddle.to_tensor(self.data_w, dtype="float32")
            if self.weight is None
            else self.weight
        )
        out = paddle.lerp(x, y, w)
        return out.astype("float16")

    def expect_output(self):
        if self.x_dtype != np.float16:
            out = self.calc_result(self.forward, "cpu")
        else:
            out = self.lerp_cast()
        return out

    @data(*LERP_CASE)
    @unpack
    def test_check_output(self, x_shape, x_dtype, weight):
        self.x_shape = x_shape
        self.x_dtype = x_dtype
        self.weight = weight
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
