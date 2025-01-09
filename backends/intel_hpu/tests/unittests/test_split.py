# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import print_function, division

import numpy as np
import unittest
from tests.op_test import OpTest
import paddle

import os

intel_hpus_module_id = os.environ.get("FLAGS_selected_intel_hpus", 0)


class HPUOpTest(OpTest):
    def set_plugin(self):
        self.__class__.use_custom_device = True
        self.__class__.no_need_check_grad = True
        self.place = paddle.CustomPlace("intel_hpu", int(intel_hpus_module_id))


class TestHPUSplitOp(HPUOpTest):
    def setUp(self):
        self.set_plugin()
        self._set_op_type()
        self.dtype = self.get_dtype()
        axis = 1
        x = np.random.random((4, 5, 6)).astype(self.dtype)
        out = np.split(x, [2, 3], axis)
        self.inputs = {"X": x}
        self.outputs = {"Out": [("out%d" % i, out[i]) for i in range(len(out))]}
        self.attrs = {"axis": axis, "sections": [2, 1, 2]}

    def get_dtype(self):
        return "float32"

    def _set_op_type(self):
        self.op_type = "split"

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def test_check_grad(self):
        pass


class TestHPUSplitOp_2(HPUOpTest):
    def setUp(self):
        self.set_plugin()
        self._set_op_type()
        self.dtype = self.get_dtype()
        self.init_data()
        self.inputs = {"X": self.x}
        self.attrs = {"axis": self.axis, "sections": self.sections, "num": self.num}

        out = np.split(self.x, self.indices_or_sections, self.axis)
        self.outputs = {"Out": [("out%d" % i, out[i]) for i in range(len(out))]}

    def init_data(self):
        self.x = np.random.random((4, 96, 128)).astype(self.dtype)
        self.axis = 1
        self.sections = []
        self.num = 3
        self.indices_or_sections = 3

    def get_dtype(self):
        return "float32"

    def _set_op_type(self):
        self.op_type = "split"

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def test_check_grad(self):
        pass


class TestHPUSplitOpFp16(HPUOpTest):
    def setUp(self):
        self.set_plugin()
        self._set_op_type()
        self.dtype = self.get_dtype()
        self.init_data()
        self.inputs = {"X": self.x}
        self.attrs = {"axis": self.axis, "sections": self.sections, "num": self.num}

        out = np.split(self.x, self.indices_or_sections, self.axis)
        self.outputs = {"Out": [("out%d" % i, out[i]) for i in range(len(out))]}

    def init_data(self):
        self.x = np.random.random((4, 5, 6)).astype(self.dtype)
        self.axis = 2
        self.sections = []
        self.num = 3
        self.indices_or_sections = 3

    def get_dtype(self):
        return "float16"

    def _set_op_type(self):
        self.op_type = "split"

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def test_check_grad(self):
        pass


if __name__ == "__main__":
    unittest.main()
