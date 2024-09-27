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
GRID_SAMPLE_CASE = [

    # TODO: Aten's support for topsatenGridSampler is not yet complete.

    # {"x_shape": [1, 1, 3, 3], "grid_shape": [1, 3, 4, 2], "dtype": np.float32, "mode": "bilinear", "padding_mode": "zeros", "align_corners": True},

]
# fmt: on


@ddt
class TestGridSample(TestAPIBase):
    def setUp(self):
        self.init_attrs()

    def init_attrs(self):
        self.x_shape = [1, 1, 3, 3]
        self.grid_shape = [1, 1, 3, 3]
        self.dtype = np.float32
        self.mode = "bilinear"
        self.padding_mode = "zeros"
        self.align_corners = True

    def prepare_data(self):
        self.data_x = self.generate_data(self.x_shape, self.dtype)
        self.data_grid = self.generate_data(self.grid_shape, self.dtype)

    def forward(self):
        x = paddle.to_tensor(self.data_x, dtype=self.dtype)
        grid = paddle.to_tensor(self.data_grid, dtype=self.dtype)
        return paddle.nn.functional.grid_sample(
            x, grid, self.mode, self.padding_mode, self.align_corners
        )

    def grid_sample_cast(self):
        x = paddle.to_tensor(self.data_x, dtype="float32")
        grid = paddle.to_tensor(self.data_grid, dtype="float32")
        out = paddle.nn.functional.grid_sample(
            x, grid, self.mode, self.padding_mode, self.align_corners
        )
        return out.astype("float16")

    def expect_output(self):
        if self.dtype != np.float16:
            out = self.calc_result(self.forward, "cpu")
        else:
            out = self.grid_sample_cast()
        return out

    @data(*GRID_SAMPLE_CASE)
    @unpack
    def test_check_output(
        self, x_shape, grid_shape, dtype, mode, padding_mode, align_corners
    ):
        self.x_shape = x_shape
        self.grid_shape = grid_shape
        self.dtype = dtype
        self.mode = mode
        self.padding_mode = padding_mode
        self.align_corners = align_corners
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
