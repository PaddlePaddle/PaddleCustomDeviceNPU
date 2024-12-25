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

from tests.op_test import OpTest, skip_check_grad_ci
import paddle
import paddle.base.core as core
from npu_utils import check_run_big_shape_test

paddle.enable_static()
SEED = 2021


@skip_check_grad_ci(reason="[skip NPU cast grad check] not implemented yet.")
class TestCast1(OpTest):
    def setUp(self):
        self.set_npu()
        self.init_dtype()
        self.init_shape()
        self.op_type = "cast"
        self.place = paddle.CustomPlace("npu", 0)

        ipt = (np.random.random(size=self.shape) + 1).astype(self.input_dtype)
        self.inputs = {"X": ipt}
        self.outputs = {"Out": ipt.astype(self.output_dtype)}

        self.attrs = {
            "in_dtype": self.in_dtype,
            "out_dtype": self.out_dtype,
        }

    def set_npu(self):
        self.__class__.use_custom_device = True

    def init_shape(self):
        self.shape = [10, 10]

    def init_dtype(self):
        self.input_dtype = "float32"
        self.output_dtype = "float16"
        self.in_dtype = int(core.VarDesc.VarType.FP32)
        self.out_dtype = int(core.VarDesc.VarType.FP16)

    def test_check_output(self):
        self.check_output_with_place(self.place)


@skip_check_grad_ci(reason="[skip NPU cast grad check] not implemented yet.")
class TestCast2(TestCast1):
    def init_dtype(self):
        self.input_dtype = "float16"
        self.output_dtype = "float32"
        self.in_dtype = int(core.VarDesc.VarType.FP16)
        self.out_dtype = int(core.VarDesc.VarType.FP32)

    def test_check_output(self):
        self.check_output_with_place(self.place, atol=1e-3)


@skip_check_grad_ci(reason="[skip NPU cast grad check] not implemented yet.")
class TestCast3(TestCast1):
    def init_dtype(self):
        self.input_dtype = "int32"
        self.output_dtype = "int32"
        self.in_dtype = int(core.VarDesc.VarType.INT32)
        self.out_dtype = int(core.VarDesc.VarType.INT32)

    def test_check_output(self):
        self.check_output_with_place(self.place, atol=1e-3)


class TestCast4(TestCast1):
    def init_shape(self):
        self.shape = [1]


@check_run_big_shape_test()
class TestCast5(TestCast1):
    def init_shape(self):
        self.shape = [1024, 8192]


class TestCast6(TestCast1):
    def init_shape(self):
        self.shape = [2, 4096, 1]

    def init_dtype(self):
        self.input_dtype = "float"
        self.output_dtype = "bool"
        self.in_dtype = int(core.VarDesc.VarType.FP32)
        self.out_dtype = int(core.VarDesc.VarType.BOOL)


@check_run_big_shape_test()
class TestCast7(TestCast1):
    def init_shape(self):
        self.shape = [4096, 4096]

    def init_dtype(self):
        self.input_dtype = "float"
        self.output_dtype = "int32"
        self.in_dtype = int(core.VarDesc.VarType.FP32)
        self.out_dtype = int(core.VarDesc.VarType.INT32)


class TestCast8(TestCast1):
    def init_shape(self):
        self.shape = [8192]

    def init_dtype(self):
        self.input_dtype = "int64"
        self.output_dtype = "bool"
        self.in_dtype = int(core.VarDesc.VarType.INT64)
        self.out_dtype = int(core.VarDesc.VarType.BOOL)


class TestCast9(TestCast1):
    def init_shape(self):
        self.shape = [2, 1, 1, 4096]

    def init_dtype(self):
        self.input_dtype = "bool"
        self.output_dtype = "uint16"
        self.in_dtype = int(core.VarDesc.VarType.BOOL)
        self.out_dtype = int(core.VarDesc.VarType.BF16)


@check_run_big_shape_test()
class TestCast10(TestCast1):
    def init_shape(self):
        self.shape = [2, 4096, 4000]

    def init_dtype(self):
        self.input_dtype = "float"
        self.output_dtype = "uint16"
        self.in_dtype = int(core.VarDesc.VarType.FP32)
        self.out_dtype = int(core.VarDesc.VarType.BF16)


@check_run_big_shape_test()
class TestCast11(TestCast10):
    def init_shape(self):
        self.shape = [3584, 8192]


@check_run_big_shape_test()
class TestCast12(TestCast10):
    def init_shape(self):
        self.shape = [4000, 8192]


@check_run_big_shape_test()
class TestCast13(TestCast10):
    def init_shape(self):
        self.shape = [8192, 1280]


@check_run_big_shape_test()
class TestCast14(TestCast10):
    def init_shape(self):
        self.shape = [8192, 7168]


class TestCast15(TestCast1):
    def init_dtype(self):
        self.input_dtype = "int64"
        self.output_dtype = "float32"
        self.in_dtype = int(core.VarDesc.VarType.INT64)
        self.out_dtype = int(core.VarDesc.VarType.FP32)

    def init_shape(self):
        self.shape = [8192, 1]


class TestCast16(TestCast1):
    def init_dtype(self):
        self.input_dtype = "bool"
        self.output_dtype = "float32"
        self.in_dtype = int(core.VarDesc.VarType.BOOL)
        self.out_dtype = int(core.VarDesc.VarType.FP32)

    def init_shape(self):
        self.shape = [8192, 4000]


class TestCast17(TestCast1):
    def init_dtype(self):
        self.input_dtype = "float16"
        self.output_dtype = "int16"
        self.in_dtype = int(core.VarDesc.VarType.FP16)
        self.out_dtype = int(core.VarDesc.VarType.INT32)

    def test_check_output(self):
        self.check_output_with_place(self.place, atol=1e-3)


class TestCast18(TestCast1):
    def init_dtype(self):
        self.input_dtype = "float16"
        self.output_dtype = "uint8"
        self.in_dtype = int(core.VarDesc.VarType.FP16)
        self.out_dtype = int(core.VarDesc.VarType.UINT8)

    def test_check_output(self):
        self.check_output_with_place(self.place, atol=1e-3)


class TestCast19(TestCast1):
    def init_dtype(self):
        self.input_dtype = "float16"
        self.output_dtype = "int8"
        self.in_dtype = int(core.VarDesc.VarType.FP16)
        self.out_dtype = int(core.VarDesc.VarType.INT8)

    def test_check_output(self):
        self.check_output_with_place(self.place, atol=1e-3)


class TestCast20(TestCast1):
    def init_dtype(self):
        self.input_dtype = "float16"
        self.output_dtype = "complex64"
        self.in_dtype = int(core.VarDesc.VarType.FP16)
        self.out_dtype = int(core.VarDesc.VarType.COMPLEX64)

    def test_check_output(self):
        self.check_output_with_place(self.place, atol=1e-3)


class TestCast21(TestCast1):
    def init_dtype(self):
        self.input_dtype = "float16"
        self.output_dtype = "complex128"
        self.in_dtype = int(core.VarDesc.VarType.FP16)
        self.out_dtype = int(core.VarDesc.VarType.COMPLEX128)

    def test_check_output(self):
        self.check_output_with_place(self.place, atol=1e-3)


if __name__ == "__main__":
    unittest.main()
