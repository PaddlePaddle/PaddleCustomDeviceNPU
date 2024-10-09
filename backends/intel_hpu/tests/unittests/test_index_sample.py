#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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


class TestIndexSampleOp(OpTest):
    def set_npu(self):
        self.__class__.use_custom_device = True

    def setUp(self):
        self.set_npu()
        self.op_type = "index_sample"
        self.config()
        xnp = np.random.random(self.x_shape).astype(self.dtype)
        indexnp = np.random.randint(
            low=0, high=self.x_shape[1], size=self.index_shape
        ).astype(self.index_type)
        self.inputs = {"X": xnp, "Index": indexnp}
        index_array = []
        for i in range(self.index_shape[0]):
            for j in indexnp[i]:
                index_array.append(xnp[i, j])
        index_array = np.array(index_array).astype(self.dtype)
        out = np.reshape(index_array, self.index_shape)
        self.outputs = {"Out": out}

    def test_check_output(self):
        self.check_output_with_place(paddle.CustomPlace("intel_hpu", 0))

    def test_check_grad(self):
        self.check_grad_with_place(paddle.CustomPlace("intel_hpu", 0), ["X"], "Out")

    def config(self):
        """
        For multi-dimension input
        """
        self.x_shape = (10, 20)
        self.dtype = "float32"
        self.index_shape = (10, 10)
        self.index_type = "int32"


class TestCase1(TestIndexSampleOp):
    def config(self):
        """
        For one dimension input
        """
        self.x_shape = (100, 1)
        self.dtype = "float32"
        self.index_shape = (100, 1)
        self.index_type = "int32"


# class TestCase2(TestIndexSampleOp):
#     def config(self):
#         """
#         For int64_t index type
#         """
#         self.x_shape = (10, 100)
#         self.dtype = "float32"
#         self.index_shape = (10, 10)
#         self.index_type = "int64"


class TestCase3(TestIndexSampleOp):
    def config(self):
        """
        For int index type
        """
        self.x_shape = (10, 100)
        self.dtype = "float32"
        self.index_shape = (10, 10)
        self.index_type = "int32"


# class TestCase4(TestIndexSampleOp):
#     def config(self):
#         """
#         For int64 index type
#         """
#         self.x_shape = (10, 128)
#         self.dtype = "float32"
#         self.index_shape = (10, 64)
#         self.index_type = "int64"


# class TestCase5(TestIndexSampleOp):
#     def config(self):
#         """
#         For float16 x type
#         """
#         self.__class__.no_need_check_grad = True
#         self.x_shape = (10, 128)
#         self.dtype = "float16"
#         self.index_shape = (10, 64)
#         self.index_type = "int64"

#     def test_check_grad(self):
#         pass

if __name__ == "__main__":
    unittest.main()
