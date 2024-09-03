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
from paddle.base.framework import convert_np_dtype_to_dtype_

paddle.enable_static()

class TestHPU(OpTest):
    def setUp(self):
        self.op_type = "fill_constant"
        self.set_npu()
        self.init_dtype()

        self.shape = [2, 3, 4, 5]
        self.fill_value = 10.0

        self.inputs = {}
        self.attrs = {
            "shape": self.shape,
            "value": self.fill_value,
            "dtype": convert_np_dtype_to_dtype_(self.dtype),
        }
        self.outputs = {"Out": np.full(self.shape, self.fill_value, self.dtype)}
        

    def set_npu(self):
        self.__class__.use_custom_device = True
        self.__class__.no_need_check_grad = True
        self.place = paddle.CustomPlace("intel_hpu", 0)

    def init_dtype(self):
        self.dtype = np.float32

    def test_check_output(self):
        self.check_output_with_place(self.place)


class TestHPU_BOOL(OpTest):
    def setUp(self):
        self.op_type = "fill_constant"
        self.set_npu()
        self.init_dtype()

        self.shape = [2, 3, 4, 5]
        self.fill_value = True

        self.inputs = {}
        self.attrs = {
            "shape": self.shape,
            "value": self.fill_value,
            "dtype": convert_np_dtype_to_dtype_(self.dtype),
        }
        self.outputs = {"Out": np.full(self.shape, self.fill_value, self.dtype)}
        

    def set_npu(self):
        self.__class__.use_custom_device = True
        self.__class__.no_need_check_grad = True
        self.place = paddle.CustomPlace("intel_hpu", 0)

    def init_dtype(self):
        self.dtype = np.bool

    def test_check_output(self):
        self.check_output_with_place(self.place)
        
if __name__ == "__main__":
    unittest.main()
