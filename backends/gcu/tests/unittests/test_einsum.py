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
EINSUM_CASE = [
    {"equation": 'i->', "x_shapes": [(4), ], "dtype": np.float32},
    {"equation": 'i,i->', "x_shapes": [(4), (4)], "dtype": np.float32},
    {"equation": 'i,j->ij', "x_shapes": [(4), (5)], "dtype": np.float32},
    {"equation": 'ijk->kji', "x_shapes": [(2, 3, 2)], "dtype": np.float32},
    {"equation": 'ijk, ikl->ijl', "x_shapes": [(2, 3, 2), (2, 2, 3)], "dtype": np.float32},
    {"equation": '...jk->...kj', "x_shapes": [(2, 3, 2), ], "dtype": np.float32},
    {"equation": '...jk, ...kl->...jl', "x_shapes": [(2, 3, 2), (2, 2, 3)], "dtype": np.float32},
    {"equation": 'mij,jk->ki', "x_shapes": [(10, 10, 20), (20, 6)], "dtype": np.float32},
    {"equation": 'i->', "x_shapes": [(4), ], "dtype": np.float16},
    {"equation": 'i,i->', "x_shapes": [(4), (4)], "dtype": np.float16},
    {"equation": 'i,j->ij', "x_shapes": [(4), (5)], "dtype": np.float16},
    {"equation": 'ijk->kji', "x_shapes": [(2, 3, 2)], "dtype": np.float16},
    {"equation": 'ijk, ikl->ijl', "x_shapes": [(2, 3, 2), (2, 2, 3)], "dtype": np.float16},
    {"equation": '...jk->...kj', "x_shapes": [(2, 3, 2), ], "dtype": np.float16},
    {"equation": '...jk, ...kl->...jl', "x_shapes": [(2, 3, 2), (2, 2, 3)], "dtype": np.float16},
    {"equation": 'mij,jk->ki', "x_shapes": [(10, 10, 20), (20, 6)], "dtype": np.float16},

]
# fmt: on


@ddt
class TestCumsum(TestAPIBase):
    def setUp(self):
        self.init_attrs()

    def init_attrs(self):
        self.dtype = np.float32
        self.shapes = [(10, 10, 20), (20, 6)]
        self.equation = "mij,jk->ki"

    def prepare_data(self):
        self.inputs = []
        for shape in self.shapes:
            self.inputs.append(self.generate_data(shape, self.dtype))

    def forward(self):
        tensor_operands = [
            paddle.to_tensor(inp, dtype=self.dtype) for inp in self.inputs
        ]
        return paddle.einsum(self.equation, *tensor_operands)

    def expect_output(self):
        if self.dtype != np.float16:
            out = self.calc_result(self.forward, "cpu")
        else:
            out = np.einsum(self.equation, *self.inputs)
        return out

    @data(*EINSUM_CASE)
    @unpack
    def test_check_output(self, equation, x_shapes, dtype):
        self.equation = equation
        self.shapes = x_shapes
        self.dtype = dtype
        rtol = 1e-5
        atol = 1e-5
        if dtype == np.float16:
            rtol = 2e-2
            atol = 1e-3
        self.check_output_gcu_with_customized(
            self.forward, self.expect_output, rtol=rtol, atol=atol
        )


if __name__ == "__main__":
    unittest.main()
