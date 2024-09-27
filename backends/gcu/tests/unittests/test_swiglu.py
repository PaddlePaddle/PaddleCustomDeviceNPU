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
SWIGLU_CASE = [
    {"x_shape": [6, 30], "x_dtype": np.float32, "has_y": True},
    {"x_shape": [5, 6, 30], "x_dtype": np.float32, "has_y": True},
    {"x_shape": [6, 30], "x_dtype": np.float32, "has_y": False},
    {"x_shape": [5, 6, 30], "x_dtype": np.float32, "has_y": False},

    {"x_shape": [6, 30], "x_dtype": np.float16, "has_y": True},
    {"x_shape": [5, 6, 30], "x_dtype": np.float16, "has_y": True},
    {"x_shape": [6, 30], "x_dtype": np.float16, "has_y": False},
    {"x_shape": [5, 6, 30], "x_dtype": np.float16, "has_y": False},

    # for llama2
    {"x_shape": [4, 512, 13824], "x_dtype": np.float16, "has_y": True},
    {"x_shape": [4, 512, 27648], "x_dtype": np.float16, "has_y": False},

]
# fmt: on


@ddt
class TestSwiglu(TestAPIBase):
    def setUp(self):
        self.init_attrs()

    def init_attrs(self):
        self.x_shape = [6, 30]
        self.x_dtype = np.float32
        self.has_y = True

    def prepare_data(self):
        self.data_x = self.generate_data(self.x_shape, self.x_dtype)
        if self.has_y:
            self.data_y = self.generate_data(self.x_shape, self.x_dtype)

    def forward_with_dtype(self, dtype):
        x = paddle.to_tensor(self.data_x, dtype=dtype)
        if self.has_y:
            y = paddle.to_tensor(self.data_y, dtype=dtype)
            return paddle.incubate.nn.functional.swiglu(x, y)
        else:
            return paddle.incubate.nn.functional.swiglu(x)

    def forward(self):
        return self.forward_with_dtype(self.x_dtype)

    # def swiglu_impl(self):
    #     data = paddle.to_tensor(self.data_x, dtype=self.x_dtype)
    #     if not self.has_y:
    #         x, y = paddle.chunk(data, chunks=2, axis=-1)
    #     else:
    #         x = data
    #         y = paddle.to_tensor(self.data_y, dtype=self.x_dtype)
    #     return paddle.incubate.nn.functional.silu(x) * y

    def swiglu_cast(self):
        return self.forward_with_dtype(np.float32).astype("float16")

    def expect_output(self):
        if self.x_dtype != np.float16:
            out = self.forward()
        else:
            out = self.swiglu_cast()
        return out

    @data(*SWIGLU_CASE)
    @unpack
    def test_check_output(self, x_shape, x_dtype, has_y):
        self.x_shape = x_shape
        self.x_dtype = x_dtype
        self.has_y = has_y
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
