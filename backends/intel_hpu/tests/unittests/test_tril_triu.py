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

from __future__ import print_function, division

import unittest

import numpy as np
import paddle
from tests.op_test import OpTest

paddle.enable_static()


class TestTrilTriu(OpTest):
    def setUp(self):
        self.op_type = "tril_triu"
        self.set_hpu()
        self.init_dtype()
        self.initTestCase()
        self.real_np_op = getattr(np, self.real_op_type)

        self.attrs = {
            "diagonal": self.diagonal,
            "lower": True if self.real_op_type == "tril" else False,
        }
        self.inputs = {"X": OpTest.np_dtype_to_base_dtype(self.x)}
        self.outputs = {
            "Out": self.real_np_op(self.x, self.diagonal)
            if self.diagonal
            else self.real_np_op(self.x)
        }

    def set_hpu(self):
        self.__class__.use_custom_device = True
        self.__class__.no_need_check_grad = True
        self.place = paddle.CustomPlace("intel_hpu", 0)

    def initTestCase(self):
        np.random.seed(1024)
        self.x = np.random.uniform(1, 2, [8, 8]).astype(self.dtype)
        # self.x = np.arange(1, 101, dtype=self.dtype).reshape([10, -1])
        # self.real_op_type = np.random.choice(["triu", "tril"])
        self.real_op_type = "triu"
        # self.diagonal = None
        self.diagonal = -3

    def init_dtype(self):
        self.dtype = np.float32

    def test_check_output(self):
        self.check_output_with_place(self.place)


class TestTrilNone(TestTrilTriu):
    def initTestCase(self):
        np.random.seed(1024)
        self.x = np.random.uniform(1, 2, [8, 8]).astype(self.dtype)
        self.real_op_type = "tril"
        self.diagonal = None


class TestTril0(TestTrilTriu):
    def initTestCase(self):
        np.random.seed(1024)
        self.x = np.random.uniform(1, 2, [8, 8]).astype(self.dtype)
        self.real_op_type = "tril"
        self.diagonal = 0


class TestTrilp(TestTrilTriu):
    def initTestCase(self):
        np.random.seed(1024)
        self.x = np.random.uniform(1, 2, [8, 8]).astype(self.dtype)
        self.real_op_type = "tril"
        self.diagonal = 4


class TestTriln(TestTrilTriu):
    def initTestCase(self):
        np.random.seed(1024)
        self.x = np.random.uniform(1, 2, [8, 8]).astype(self.dtype)
        self.real_op_type = "tril"
        self.diagonal = -4


class TestTriuNone(TestTrilTriu):
    def initTestCase(self):
        np.random.seed(1024)
        self.x = np.random.uniform(1, 2, [8, 8]).astype(self.dtype)
        self.real_op_type = "triu"
        self.diagonal = None


class TestTriu0(TestTrilTriu):
    def initTestCase(self):
        np.random.seed(1024)
        self.x = np.random.uniform(1, 2, [8, 8]).astype(self.dtype)
        self.real_op_type = "triu"
        self.diagonal = 0


class TestTriup(TestTrilTriu):
    def initTestCase(self):
        np.random.seed(1024)
        self.x = np.random.uniform(1, 2, [8, 8]).astype(self.dtype)
        self.real_op_type = "triu"
        self.diagonal = 4


class TestTriun(TestTrilTriu):
    def initTestCase(self):
        np.random.seed(1024)
        self.x = np.random.uniform(1, 2, [8, 8]).astype(self.dtype)
        self.real_op_type = "triu"
        self.diagonal = -4


if __name__ == "__main__":
    unittest.main()
