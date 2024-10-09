# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function

import unittest

import numpy as np
import paddle
from tests.op_test import OpTest

SEED = 2021


class TestSliceOp(OpTest):
    def setUp(self):
        self.set_hpu()
        self.place = paddle.CustomPlace("intel_hpu", 0)
        self.op_type = "slice"
        self.init_dtype()
        self.config()
        self.inputs = {"Input": self.input}
        self.outputs = {"Out": self.out}
        self.attrs = {
            "axes": self.axes,
            "starts": self.starts,
            "ends": self.ends,
            "infer_flags": self.infer_flags,
        }

    def config(self):
        self.input = np.random.random([3, 4, 5, 6]).astype(self.dtype)
        self.starts = [1, 0, 2]
        self.ends = [3, 3, 4]
        self.axes = [0, 1, 2]
        self.infer_flags = [1, 1, 1]
        self.out = self.input[1:3, 0:3, 2:4, :]
        # np.set_printoptions(precision=10, linewidth=300, floatmode='fixed')
        # print(self.input)
        # print(self.out)

    def set_hpu(self):
        self.__class__.use_custom_device = True

    def init_dtype(self):
        self.dtype = np.float32

    def test_check_output(self):
        self.check_output_with_place(self.place)


class TestSliceMultiDim(TestSliceOp):
    def config(self):
        self.input = np.random.random([3, 2, 5, 4, 8]).astype(self.dtype)
        self.starts = [0]
        self.ends = [1]
        self.axes = [4]
        self.infer_flags = [1]
        self.out = self.input[:, :, :, :, 0:1]


if __name__ == "__main__":
    unittest.main()
