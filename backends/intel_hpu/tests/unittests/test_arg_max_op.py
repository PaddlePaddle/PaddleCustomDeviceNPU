# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import print_function

import unittest
import numpy as np
from tests.op_test import OpTest
import paddle
import paddle.base.core as core

paddle.enable_static()


class TestHpuArgMaxOp(OpTest):
    def setUp(self):
        self.op_type = "arg_max"
        self.set_hpu()
        self.init_dtype()
        self.initTestCase()

        self.inputs = {"X": self.x}
        self.outputs = {"Out": self.out}

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def set_hpu(self):
        self.__class__.use_custom_device = True
        self.place = paddle.CustomPlace("intel_hpu", 0)

    def init_dtype(self):
        self.dtype = np.int32

    def initTestCase(self):
        self.dims = (3, 4, 6)
        self.axis = 2
        self.x = (1000 * np.random.random(self.dims)).astype(self.dtype)
        self.attrs = {"axis": self.axis, "dtype": int(core.VarDesc.VarType.INT64)}
        self.out = np.argmax(self.x, axis=self.axis)


class TestArgMaxFloat16Case1(TestHpuArgMaxOp):
    def initTestCase(self):
        self.dims = (3, 4, 6)
        self.axis = 1
        self.keep_dims = True
        self.x = (1000 * np.random.random(self.dims)).astype(self.dtype)
        self.attrs = {
            "axis": self.axis,
            "dtype": int(core.VarDesc.VarType.INT32),
            "keepdims": self.keep_dims,
        }
        self.out = np.argmax(self.x, axis=self.axis).reshape(3, 1, 6)


class TestArgMaxFloat16Case2(TestHpuArgMaxOp):
    def init_dtype(self):
        self.dtype = "float16"

    def initTestCase(self):
        self.dims = (3, 4, 6)
        self.axis = -1
        self.keep_dims = True
        self.x = (1000 * np.random.random(self.dims)).astype(self.dtype)
        self.attrs = {
            "axis": self.axis,
            "dtype": int(core.VarDesc.VarType.INT64),
            "keepdims": self.keep_dims,
        }
        self.out = np.argmax(self.x, axis=self.axis).reshape(3, 4, 1)


class TestArgMaxFloat16Case3(TestHpuArgMaxOp):
    def initTestCase(self):
        self.dims = (3, 4, 6)
        self.axis = 2
        self.x = (1000 * np.random.random(self.dims)).astype(self.dtype)
        self.attrs = {"axis": self.axis, "dtype": int(core.VarDesc.VarType.INT32)}
        self.out = np.argmax(self.x, axis=self.axis)


if __name__ == "__main__":
    unittest.main()
