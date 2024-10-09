# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
import paddle.base.core as core

from tests.op_test import (
    OpTest,
    convert_float_to_uint16,
    convert_uint16_to_float,
    skip_check_grad_ci,
)

SEED = 2021


@skip_check_grad_ci(reason="[skip INTEL HPU cast grad check] not implemented yet.")
class TestCastBF16(OpTest):
    def setUp(self):
        self.set_npu()
        self.init_dtype()
        self.init_shape()
        self.op_type = "cast"
        self.place = paddle.CustomPlace("intel_hpu", 0)

        ipt = np.random.random(size=self.shape) + 1
        x = convert_float_to_uint16(ipt.astype(self.input_dtype))
        self.inputs = {"X": x}
        self.outputs = {"Out": convert_uint16_to_float(x).astype(self.output_dtype)}

        self.attrs = {
            "in_dtype": self.in_dtype,
            "out_dtype": self.out_dtype,
        }

    def set_npu(self):
        self.__class__.use_custom_device = True

    def init_shape(self):
        self.shape = [10, 10]

    def init_dtype(self):
        self.input_dtype = "float16"
        self.output_dtype = "float32"
        self.in_dtype = int(core.VarDesc.VarType.FP16)
        self.out_dtype = int(core.VarDesc.VarType.FP32)

    def test_check_output(self):
        self.check_output_with_place(self.place)


class TestCastBF16_1(TestCastBF16):
    def init_shape(self):
        self.shape = [2, 1, 4096, 4096]

    def init_dtype(self):
        self.input_dtype = "float32"
        self.output_dtype = "bool"
        self.in_dtype = int(core.VarDesc.VarType.BF16)
        self.out_dtype = int(core.VarDesc.VarType.BOOL)


class TestCastBF16_2(TestCastBF16_1):
    def init_dtype(self):
        self.input_dtype = "float32"
        self.output_dtype = "float32"
        self.in_dtype = int(core.VarDesc.VarType.BF16)
        self.out_dtype = int(core.VarDesc.VarType.FP32)


if __name__ == "__main__":
    unittest.main()
