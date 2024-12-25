# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
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

import numpy as np
import unittest

from tests.op_test import OpTest
import paddle

paddle.enable_static()
SEED = 2021


def reference_diag(x, offset=0, padding_value=0.0):
    """Reference forward implementation using np.diag."""
    v = np.array(x)
    out = np.diag(v, offset)
    return out


class TestDiagOp(OpTest):
    """
    case 0
    """

    def set_npu(self):
        self.__class__.use_custom_device = True
        self.place = paddle.CustomPlace("npu", 0)

    def config(self):
        self.x_shape = 100
        self.offset = 0
        self.padding_value = 0.0

    def init_kernel_type(self):
        self.dtype = np.float32

    def setUp(self):
        self.set_npu()
        self.init_kernel_type()
        self.config()
        self.op_type = "diag_v2"
        x = np.random.random(self.x_shape).astype(self.dtype)
        # -0.1 ~ 0.1
        x = -0.1 + 0.2 * x
        result = reference_diag(x, self.offset)
        result = result.astype(self.dtype)
        self.inputs = {"X": x}
        self.attrs = {"offset": self.offset, "padding_value": self.padding_value}
        self.outputs = {"Out": result}

    def test_check_output(self):
        self.check_output_with_place(self.place, atol=1e-7)


class TestDiagOp1(TestDiagOp):
    """
    case 1
    """

    def config(self):
        self.x_shape = 100
        self.offset = 1
        self.padding_value = 0.0

    def init_kernel_type(self):
        self.dtype = np.float32

    def test_check_output(self):
        self.check_output_with_place(self.place, atol=1e-3)


class TestDiagOp2(TestDiagOp):
    """
    case 2
    """

    def config(self):
        self.x_shape = 100
        self.offset = 0
        self.padding_value = 0.0

    def init_kernel_type(self):
        self.dtype = np.float64

    def test_check_output(self):
        self.check_output_with_place(self.place, atol=1e-3)


class TestDiagOp3(TestDiagOp):
    """
    case 3
    """

    def config(self):
        self.x_shape = 100
        self.offset = 0
        self.padding_value = 0.0

    def setUp(self):
        self.set_npu()
        self.init_kernel_type()
        self.config()
        self.op_type = "diag_v2"
        x = np.random.randint(0, high=5, size=self.x_shape).astype(self.dtype)
        result = reference_diag(x, self.offset)
        result = result.astype(self.dtype)
        self.inputs = {"X": x}
        self.attrs = {"offset": self.offset, "padding_value": self.padding_value}
        self.outputs = {"Out": result}

    def init_kernel_type(self):
        self.dtype = np.int32

    def test_check_output(self):
        self.check_output_with_place(self.place, atol=1e-3)


class TestDiagOp4(TestDiagOp3):
    def config(self):
        self.x_shape = (100, 2)
        self.offset = 0
        self.padding_value = 0.0


if __name__ == "__main__":
    unittest.main()
