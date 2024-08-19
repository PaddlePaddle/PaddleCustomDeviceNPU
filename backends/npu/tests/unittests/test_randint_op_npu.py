#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import print_function

import unittest
import numpy as np

from tests.op_test import OpTest
import paddle
import paddle.base.core as core
from paddle.static import program_guard, Program

paddle.enable_static()


def check_randint_out(data, low, high):
    assert isinstance(data, np.ndarray), "The input data should be np.ndarray."
    mask = (data < low) | (data >= high)
    return not mask.any()


def convert_dtype(dtype_str):
    dtype_str_list = ["int32", "int64", "float32", "float64"]
    dtype_num_list = [
        core.VarDesc.VarType.INT32,
        core.VarDesc.VarType.INT64,
        core.VarDesc.VarType.FP32,
        core.VarDesc.VarType.FP64,
    ]
    assert dtype_str in dtype_str_list, dtype_str + " should in " + str(dtype_str_list)
    return dtype_num_list[dtype_str_list.index(dtype_str)]


class TestRandintOp(OpTest):
    """Test randint op."""

    def setUp(self):
        self.set_npu()
        self.op_type = "randint"
        self.low = 0
        self.high = 10
        self.shape = [3, 3]
        self.dtype = "int64"

        self.inputs = {}
        self.outputs = {"Out": np.zeros(self.shape).astype(self.dtype)}
        self.init_attrs()
        self.attrs = {
            "low": self.low,
            "high": self.high,
            "shape": self.shape,
            "dtype": convert_dtype(self.dtype),
        }

    def set_npu(self):
        self.__class__.use_custom_device = True

    def _get_places(self):
        return [paddle.CustomPlace("npu", 0)]

    def init_attrs(self):
        pass

    def test_check_output(self):
        self.check_output_customized(self.verify_output)

    def verify_output(self, outs):
        out_np = np.array(outs[0])
        self.assertTrue(check_randint_out(out_np, self.low, self.high))


class TestRandintOpLow(TestRandintOp):
    def init_attrs(self):
        self.low = -10


class TestRandintOpHigh(TestRandintOp):
    def init_attrs(self):
        self.high = 5


class TestRandintOpInt32(TestRandintOp):
    def init_attrs(self):
        self.dtype = "int32"


class TestRandintAPI(unittest.TestCase):
    def test_out(self):
        low = -5
        high = 5
        shape = [2, 3]
        place = paddle.CustomPlace("npu", 0)
        with program_guard(Program(), Program()):
            x1 = paddle.randint(low, high, shape=shape)
            x2 = paddle.randint(low, high, shape=shape, dtype="int32")

            exe = paddle.static.Executor(place)
            res = exe.run(fetch_list=[x1, x2])

            self.assertEqual(res[0].dtype, np.int64)
            self.assertEqual(res[1].dtype, np.int32)
            self.assertTrue(check_randint_out(res[0], low, high))
            self.assertTrue(check_randint_out(res[1], low, high))


class TestRandintImperative(unittest.TestCase):
    def test_out(self):
        paddle.disable_static(paddle.CustomPlace("npu", 0))
        low = -10
        high = 10
        shape = [5, 3]
        for dtype in ["int32", np.int64]:
            data_p = paddle.randint(low, high, shape=shape, dtype=dtype)
            data_np = data_p.numpy()
            self.assertTrue(check_randint_out(data_np, low, high))
        paddle.enable_static()


if __name__ == "__main__":
    unittest.main()
