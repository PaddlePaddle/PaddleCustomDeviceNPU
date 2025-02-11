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

paddle.enable_static()

import os

intel_hpus_module_id = os.environ.get("FLAGS_selected_intel_hpus", 0)


class TestHpuScaleOp(OpTest):
    def setUp(self):
        self.op_type = "scale"
        self.set_hpu()
        self.init_dtype()
        self.init_input()
        self.attrs = {
            "scale": self.scale,
            "bias": self.bias,
            "bias_after_scale": self.bias_after_scale,
        }
        self.inputs = {"X": self.x}
        self.outputs = {"Out": self.out}

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def set_hpu(self):
        self.__class__.use_custom_device = True
        self.place = paddle.CustomPlace("intel_hpu", int(intel_hpus_module_id))

    def init_dtype(self):
        self.dtype = np.float16

    def init_input(self):
        self.scale = 2.8
        self.bias = 3.2
        self.bias_after_scale = True
        np.random.seed(1024)
        self.x = np.random.random((2, 3, 4)).astype(self.dtype)
        self.out = (
            self.x * self.scale
            + (self.bias if self.bias_after_scale else self.scale * self.bias)
        ).astype(self.dtype)


class TestFP16Scale(TestHpuScaleOp):
    def init_dtype(self):
        self.dtype = np.float32


class TestScaleInt32(TestHpuScaleOp):
    def init_dtype(self):
        self.dtype = np.int32


class TestScaleInt64(TestHpuScaleOp):
    def init_dtype(self):
        self.dtype = np.int64


class TestScaleDouble(TestHpuScaleOp):
    def init_dtype(self):
        self.dtype = np.double


class TestBiasAfterScale(OpTest):
    def setUp(self):
        self.set_npu()
        self.op_type = "scale"
        self.place = paddle.CustomPlace("intel_hpu", int(intel_hpus_module_id))
        self.init_dtype()

        self.inputs = {
            "X": OpTest.np_dtype_to_base_dtype(
                np.random.random((10, 10)).astype(self.dtype)
            )
        }
        self.attrs = {"scale": -2.3, "bias": 0, "bias_after_scale": False}
        self.outputs = {"Out": self.inputs["X"] * self.dtype(self.attrs["scale"])}

    def set_npu(self):
        self.__class__.use_custom_device = True

    def init_dtype(self):
        self.dtype = np.float32

    def test_check_output(self):
        self.check_output_with_place(self.place)


if __name__ == "__main__":
    unittest.main()
