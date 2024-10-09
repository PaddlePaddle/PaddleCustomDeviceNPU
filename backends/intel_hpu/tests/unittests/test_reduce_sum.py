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
from tests.op_test import OpTest, skip_check_grad_ci

paddle.enable_static()
SEED = 2021


class TestReduceSum(OpTest):
    def setUp(self):
        np.random.seed(SEED)
        self.set_npu()
        self.init_dtype()
        self.place = paddle.CustomPlace("intel_hpu", 0)
        self.init_op_type()
        self.initTestCase()

        self.use_mkldnn = False
        self.attrs = {
            "dim": self.axis,
            "keep_dim": self.keep_dim,
            "reduce_all": self.reduce_all,
        }
        self.inputs = {"X": np.random.random(self.shape).astype(self.dtype)}
        if self.attrs["reduce_all"]:
            self.outputs = {"Out": self.inputs["X"].sum()}
        else:
            self.outputs = {
                "Out": self.inputs["X"].sum(
                    axis=self.axis, keepdims=self.attrs["keep_dim"]
                )
            }

    def set_npu(self):
        self.__class__.use_custom_device = True

    def init_dtype(self):
        self.dtype = np.float32

    def init_op_type(self):
        self.op_type = "reduce_sum"
        self.use_mkldnn = False
        self.keep_dim = True
        self.reduce_all = False

    def initTestCase(self):
        self.shape = (5, 6, 10)
        self.axis = (0,)

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def test_check_grad(self):
        pass


@skip_check_grad_ci(
    reason="reduce_max is discontinuous non-derivable function,"
    " its gradient check is not supported by unittest framework."
)
class TestReduceSumFP16(TestReduceSum):
    def init_dtype(self):
        self.dtype = np.float16


# class TestReduceSumOp5D(TestReduceSum):
#     def initTestCase(self):
#         self.shape = (1, 2, 5, 6, 10)
#         self.axis = (-1, -2)


class TestReduceSumOpRank1(TestReduceSum):
    def initTestCase(self):
        self.shape = (2, 4096, 1)
        self.axis = (0,)


class TestKeepDimReduce(TestReduceSum):
    def init_op_type(self):
        self.op_type = "reduce_sum"
        self.use_mkldnn = False
        self.keep_dim = True
        self.reduce_all = False


# class TestReduceAll(TestReduceSum):
#     def init_op_type(self):
#         self.op_type = "reduce_sum"
#         self.use_mkldnn = False
#         self.keep_dim = False
#         self.reduce_all = True


if __name__ == "__main__":
    unittest.main()
