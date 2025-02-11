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

import os

intel_hpus_module_id = os.environ.get("FLAGS_selected_intel_hpus", 0)


@skip_check_grad_ci(reason="[skip INTEL HPU cast grad check] not implemented yet.")
class TestCastBF16(OpTest):
    def setUp(self):
        self.set_npu()
        self.init_dtype()
        self.init_shape()
        self.op_type = "cast"
        self.place = paddle.CustomPlace("intel_hpu", int(intel_hpus_module_id))

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


class TestCastBF16_3(TestCastBF16):
    def init_shape(self):
        self.shape = [2, 1, 1, 4096]

    def init_dtype(self):
        self.input_dtype = "bool"
        self.output_dtype = "uint16"
        self.in_dtype = int(core.VarDesc.VarType.BOOL)
        self.out_dtype = int(core.VarDesc.VarType.BF16)


class TestCastBF16_4(TestCastBF16):
    def init_shape(self):
        self.shape = [1]


class TestCastBF16_5(TestCastBF16):
    def init_shape(self):
        self.shape = [1024, 8192]


@skip_check_grad_ci(reason="[skip NPU cast grad check] not implemented yet.")
class TestCastBF16_5(TestCastBF16):
    def init_dtype(self):
        self.input_dtype = "float16"
        self.output_dtype = "float32"
        self.in_dtype = int(core.VarDesc.VarType.FP16)
        self.out_dtype = int(core.VarDesc.VarType.FP32)

    def test_check_output(self):
        self.check_output_with_place(self.place, atol=1e-3)


@skip_check_grad_ci(reason="[skip NPU cast grad check] not implemented yet.")
class TestCastBF16_6(TestCastBF16):
    def init_dtype(self):
        self.input_dtype = "int32"
        self.output_dtype = "int32"
        self.in_dtype = int(core.VarDesc.VarType.INT32)
        self.out_dtype = int(core.VarDesc.VarType.INT32)

    def test_check_output(self):
        self.check_output_with_place(self.place, atol=1e-3)


class TestCastBF16_7(TestCastBF16):
    def init_shape(self):
        self.shape = [2, 4096, 1]

    def init_dtype(self):
        self.input_dtype = "float"
        self.output_dtype = "bool"
        self.in_dtype = int(core.VarDesc.VarType.FP32)
        self.out_dtype = int(core.VarDesc.VarType.BOOL)


class TestCastBF16_8(TestCastBF16):
    def init_shape(self):
        self.shape = [4096, 4096]

    def init_dtype(self):
        self.input_dtype = "float"
        self.output_dtype = "int32"
        self.in_dtype = int(core.VarDesc.VarType.FP32)
        self.out_dtype = int(core.VarDesc.VarType.INT32)


class TestCastBF16_9(TestCastBF16):
    def init_shape(self):
        self.shape = [8192]

    def init_dtype(self):
        self.input_dtype = "int64"
        self.output_dtype = "bool"
        self.in_dtype = int(core.VarDesc.VarType.INT64)
        self.out_dtype = int(core.VarDesc.VarType.BOOL)


class TestCastBF16_10(TestCastBF16):
    def init_shape(self):
        self.shape = [2, 1, 1, 4096]

    def init_dtype(self):
        self.input_dtype = "bool"
        self.output_dtype = "uint16"
        self.in_dtype = int(core.VarDesc.VarType.BOOL)
        self.out_dtype = int(core.VarDesc.VarType.BF16)


class TestCast10(TestCastBF16):
    def init_shape(self):
        self.shape = [2, 4096, 4000]

    def init_dtype(self):
        self.input_dtype = "float"
        self.output_dtype = "uint16"
        self.in_dtype = int(core.VarDesc.VarType.FP32)
        self.out_dtype = int(core.VarDesc.VarType.BF16)


class TestCast11(TestCast10):
    def init_shape(self):
        self.shape = [3584, 8192]


class TestCast12(TestCast10):
    def init_shape(self):
        self.shape = [4000, 8192]


class TestCast13(TestCast10):
    def init_shape(self):
        self.shape = [8192, 1280]


class TestCast14(TestCast10):
    def init_shape(self):
        self.shape = [8192, 7168]


class TestCast15(TestCastBF16):
    def init_dtype(self):
        self.input_dtype = "int64"
        self.output_dtype = "float32"
        self.in_dtype = int(core.VarDesc.VarType.INT64)
        self.out_dtype = int(core.VarDesc.VarType.FP32)

    def init_shape(self):
        self.shape = [8192, 1]


class TestCast16(TestCastBF16):
    def init_dtype(self):
        self.input_dtype = "bool"
        self.output_dtype = "float32"
        self.in_dtype = int(core.VarDesc.VarType.BOOL)
        self.out_dtype = int(core.VarDesc.VarType.FP32)

    def init_shape(self):
        self.shape = [8192, 4000]


class TestCastOpFp32ToFp64(OpTest):
    def setUp(self):
        self.set_npu()
        self.op_type = "cast"
        self.place = paddle.CustomPlace("intel_hpu", int(intel_hpus_module_id))

        ipt = np.random.random(size=[10, 10])
        self.inputs = {"X": ipt.astype("float32")}
        self.outputs = {"Out": ipt.astype("float64")}
        self.attrs = {
            "in_dtype": int(core.VarDesc.VarType.FP32),
            "out_dtype": int(core.VarDesc.VarType.FP64),
        }

    def set_npu(self):
        self.__class__.use_custom_device = True

    def test_check_output(self):
        self.check_output()

    def test_grad(self):
        self.check_grad(["X"], ["Out"])


class TestCastOpFp16ToFp32(OpTest):
    def setUp(self):
        self.set_npu()
        ipt = np.random.random(size=[10, 10])
        self.inputs = {"X": ipt.astype("float16")}
        self.outputs = {"Out": ipt.astype("float32")}
        self.attrs = {
            "in_dtype": int(core.VarDesc.VarType.FP16),
            "out_dtype": int(core.VarDesc.VarType.FP32),
        }
        self.op_type = "cast"
        self.__class__.no_need_check_grad = True

    def set_npu(self):
        self.__class__.use_custom_device = True

    def test_check_output(self):
        self.check_output(atol=1e-3)


class TestCastOpBf16ToFp32(OpTest):
    def setUp(self):
        self.set_npu()
        ipt = np.array(np.random.randint(10, size=[10, 10])).astype("uint16")
        self.inputs = {"X": ipt}
        self.outputs = {"Out": convert_uint16_to_float(ipt)}
        self.attrs = {
            "in_dtype": int(core.VarDesc.VarType.BF16),
            "out_dtype": int(core.VarDesc.VarType.FP32),
        }
        self.op_type = "cast"
        self.__class__.no_need_check_grad = True

    def set_npu(self):
        self.__class__.use_custom_device = True

    def test_check_output(self):
        self.check_output()


if __name__ == "__main__":
    unittest.main()
