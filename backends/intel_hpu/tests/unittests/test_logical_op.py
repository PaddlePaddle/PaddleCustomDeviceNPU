# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function, division

import unittest

import numpy as np
import paddle

from ddt import ddt, data, unpack
from api_base import TestAPIBase

paddle.disable_static()

import os
from util import enable_paddle_static_mode

intel_hpus_static_mode = os.environ.get(
    "FLAGS_static_mode_intel_hpus", 0
)  # default is dynamic mode test FLAGS_static_mode_intel_hpus=0

LOGICAL_CASE = [
    {"logical_api": paddle.logical_and, "shape": [2, 3, 32, 32], "dtype": np.float32},
    {"logical_api": paddle.logical_and, "shape": [2, 3, 32, 32], "dtype": np.int32},
    {"logical_api": paddle.logical_and, "shape": [2, 3, 32, 32], "dtype": np.int64},
    {"logical_api": paddle.logical_and, "shape": [2, 4, 4], "dtype": np.float32},
    {"logical_api": paddle.logical_and, "shape": [2, 3, 32, 32], "dtype": bool},
    {"logical_api": paddle.logical_and, "shape": [2, 4, 4], "dtype": bool},
    {"logical_api": paddle.logical_not, "shape": [2, 3, 32, 32], "dtype": np.float32},
    {"logical_api": paddle.logical_not, "shape": [2, 3, 32, 32], "dtype": np.int32},
    {"logical_api": paddle.logical_not, "shape": [2, 3, 32, 32], "dtype": np.int64},
    {"logical_api": paddle.logical_not, "shape": [2, 4, 4], "dtype": np.float32},
    {"logical_api": paddle.logical_not, "shape": [2, 3, 32, 32], "dtype": bool},
    {"logical_api": paddle.logical_not, "shape": [2, 4, 4], "dtype": bool},
    {"logical_api": paddle.logical_or, "shape": [2, 3, 32, 32], "dtype": np.float32},
    {"logical_api": paddle.logical_or, "shape": [2, 3, 32, 32], "dtype": np.int32},
    {"logical_api": paddle.logical_or, "shape": [2, 3, 32, 32], "dtype": np.int64},
    {"logical_api": paddle.logical_or, "shape": [2, 4, 4], "dtype": np.float32},
    {"logical_api": paddle.logical_or, "shape": [2, 3, 32, 32], "dtype": bool},
    {"logical_api": paddle.logical_or, "shape": [2, 4, 4], "dtype": bool},
    {"logical_api": paddle.logical_xor, "shape": [2, 3, 32, 32], "dtype": np.float32},
    {"logical_api": paddle.logical_xor, "shape": [2, 3, 32, 32], "dtype": np.int32},
    {"logical_api": paddle.logical_xor, "shape": [2, 3, 32, 32], "dtype": np.int64},
    {"logical_api": paddle.logical_xor, "shape": [2, 4, 4], "dtype": np.float32},
    {"logical_api": paddle.logical_xor, "shape": [2, 3, 32, 32], "dtype": bool},
    {"logical_api": paddle.logical_xor, "shape": [2, 4, 4], "dtype": bool},
]

LOGICAL_F16_CASE = [
    {
        "logical_api": paddle.logical_and,
        "numpy_api": np.logical_and,
        "shape": [2, 3, 32, 32],
        "dtype": np.float16,
    },
    {
        "logical_api": paddle.logical_not,
        "numpy_api": np.logical_not,
        "shape": [2, 3, 32, 32],
        "dtype": np.float16,
    },
    {
        "logical_api": paddle.logical_or,
        "numpy_api": np.logical_or,
        "shape": [2, 3, 32, 32],
        "dtype": np.float16,
    },
    {
        "logical_api": paddle.logical_xor,
        "numpy_api": np.logical_xor,
        "shape": [2, 3, 32, 32],
        "dtype": np.float16,
    },
]


class TestLogicalOp(TestAPIBase):
    def setUp(self):
        self.init_attrs()
        self.set_hpu()

    def set_hpu(self):
        enable_paddle_static_mode(int(intel_hpus_static_mode))

    def init_attrs(self):
        self.shape = [4]
        self.dtype = np.float32

    def prepare_data(self):
        self.data_x = self.generate_data(self.shape, self.dtype)
        self.data_y = self.generate_data(self.shape, self.dtype)

        if self.dtype == np.int64 or self.dtype == np.int32:
            self.data_x = self.generate_integer_data(self.shape, self.dtype, -200, 200)
            self.data_y = self.generate_integer_data(self.shape, self.dtype, -200, 200)

    def logical_forward(self):
        x = paddle.to_tensor(self.data_x, dtype=self.dtype)
        y = paddle.to_tensor(self.data_y, dtype=self.dtype)
        if self.logical_api in [paddle.logical_not]:
            return self.logical_api(x)
        return self.logical_api(x, y)

    def init_dtype(self):
        self.dtype = np.float32


@ddt
class TestLogicalOpComputation(TestLogicalOp):
    @data(*LOGICAL_CASE)
    @unpack
    def test_check_output(self, logical_api, shape, dtype):
        self.logical_api = logical_api
        self.shape = shape
        self.dtype = dtype
        self.check_output_hpu_with_cpu(self.logical_forward)


@ddt
class TestLogicalOpComputationF16(TestLogicalOp):
    def numpy_binary(self):
        if self.numpy_api in [np.logical_not]:
            return self.numpy_api(self.data_x)
        return self.numpy_api(self.data_x, self.data_y)

    @data(*LOGICAL_F16_CASE)
    @unpack
    def test_check_output(self, logical_api, numpy_api, shape, dtype):
        self.logical_api = logical_api
        self.numpy_api = numpy_api
        self.shape = shape
        self.dtype = dtype
        self.check_output_hpu_with_customized(self.logical_forward, self.numpy_binary)


if __name__ == "__main__":
    unittest.main()
