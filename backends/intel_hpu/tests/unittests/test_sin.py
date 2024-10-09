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


paddle.enable_static()
SEED = 2021


class TestActivation(OpTest):
    def set_npu(self):
        self.__class__.use_custom_device = True
        self.place = paddle.CustomPlace("intel_hpu", 0)

    def setUp(self):
        self.set_npu()
        self.op_type = "exp"
        self.init_dtype()
        self.init_shape()
        self.init_kernel_type()
        self.check_dygraph = True
        self.python_api = paddle.exp

        np.random.seed(2049)
        x = np.random.uniform(0.1, 1, self.shape).astype(self.dtype)
        out = np.exp(x)

        self.inputs = {"X": OpTest.np_dtype_to_base_dtype(x)}
        self.outputs = {"Out": out}

    def test_check_output(self):
        check_dygraph = False
        if hasattr(self, "check_dygraph"):
            check_dygraph = self.check_dygraph
        self.check_output_with_place(self.place, check_dygraph=check_dygraph)

    def test_check_grad(self):
        pass

    def init_dtype(self):
        self.dtype = np.float32

    def init_shape(self):
        self.shape = [11, 17]

    def init_kernel_type(self):
        pass


class TestSin(TestActivation):
    def setUp(self):
        self.set_npu()
        self.op_type = "sin"
        self.init_dtype()
        self.init_shape()

        np.random.seed(1024)
        x = np.random.uniform(-1, 1, self.shape).astype(self.dtype)
        out = np.sin(x)

        self.inputs = {"X": OpTest.np_dtype_to_base_dtype(x)}
        self.outputs = {"Out": out}

    def init_shape(self):
        self.shape = [10, 12]

    def test_check_grad(self):
        pass


if __name__ == "__main__":
    unittest.main()
