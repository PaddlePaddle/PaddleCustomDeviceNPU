#  Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np
import unittest
from tests.op_test import OpTest, convert_float_to_uint16, convert_uint16_to_float
import paddle


class TestTransposeOp(OpTest):
    def setUp(self):
        self.set_npu()
        self.op_type = "transpose2"
        self.place = paddle.CustomPlace("intel_hpu", 0)
        self.init_dtype()
        self.init_shape_axis()

        self.inputs = {
            "X": convert_float_to_uint16(
                np.random.random(self.shape).astype(np.float32)
            )
        }
        self.outputs = {
            "Out": convert_uint16_to_float(self.inputs["X"].transpose(self.axis))
        }
        self.attrs = {"axis": self.axis, "data_format": "AnyLayout"}

    def set_npu(self):
        self.__class__.use_custom_device = True

    def init_dtype(self):
        self.dtype = np.float32

    def init_shape_axis(self):
        self.shape = (3, 40)
        self.axis = (1, 0)

    def test_check_output(self):
        self.check_output_with_place(self.place)


class TestTransposeOpRank1(TestTransposeOp):
    def init_shape_axis(self):
        self.shape = (3584, 8192)
        self.axis = (1, 0)


class TestTransposeOpRank2(TestTransposeOp):
    def init_shape_axis(self):
        self.shape = (1024, 8192)
        self.axis = (1, 0)


if __name__ == "__main__":
    unittest.main()
