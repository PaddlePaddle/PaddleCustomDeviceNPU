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
import paddle
import paddle.base as base
from tests.op_test import OpTest

paddle.enable_static()
SEED = 2021


def gather_numpy(x, index, axis):
    x_transpose = np.swapaxes(x, 0, axis)
    tmp_gather = x_transpose[index, ...]
    gather = np.swapaxes(tmp_gather, 0, axis)
    return gather


class TestGatherOp(OpTest):
    def setUp(self):
        self.set_intel_hpu()
        self.place = paddle.CustomPlace("intel_hpu", 0)
        self.op_type = "gather"
        self.config()
        xnp = np.random.random(self.x_shape).astype(self.x_type)
        self.iintel_hputs = {
            "X": xnp,
            "Index": np.array(self.index).astype(self.index_type),
        }
        self.outputs = {"Out": self.iintel_hputs["X"][self.iintel_hputs["Index"]]}

    def set_intel_hpu(self):
        self.__class__.use_custom_device = True

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def test_check_grad(self):
        self.check_grad_with_place(
            self.place,
            ["X"],
            "Out",
            max_relative_error=0.006,
        )

    def config(self):
        """
        For multi-dimension iintel_hput
        """
        self.x_shape = (10, 20)
        self.x_type = "float32"
        self.index = [1, 3, 5]
        self.index_type = "int32"


class TestCase1(TestGatherOp):
    def config(self):
        """
        For one dimension iintel_hput
        """
        self.x_shape = 100
        self.x_type = "float32"
        self.index = [1, 3, 5]
        self.index_type = "int32"


class TestCase2(TestGatherOp):
    def config(self):
        """
        For one dimension iintel_hput
        """
        self.x_shape = 100
        self.x_type = "int64"
        self.index = [1, 3, 5]
        self.index_type = "int32"

    def test_check_grad(self):
        pass


class TestCase3(TestGatherOp):
    def config(self):
        """
        For one dimension iintel_hput
        """
        self.x_shape = 100
        self.x_type = "int32"
        self.index = [1, 3, 5]
        self.index_type = "int32"

    def test_check_grad(self):
        pass


# class TestCase4(TestGatherOp):
#     def config(self):
#         """
#         For one dimension iintel_hput
#         """
#         self.x_shape = 100
#         self.x_type = "bool"
#         self.index = [1, 3, 5]
#         self.index_type = "int32"

#     def test_check_grad(self):
#         pass


class TestCase5(TestGatherOp):
    def config(self):
        """
        For one dimension iintel_hput
        """
        self.x_shape = [4000, 8192]
        self.x_type = "float32"
        self.index = [1, 3, 5]
        self.index_type = "int32"


class API_TestGather(unittest.TestCase):
    def test_out1(self):
        with base.program_guard(base.Program(), base.Program()):
            data1 = paddle.static.data("data1", shape=[-1, 2], dtype="float32")
            index = paddle.static.data("index", shape=[-1, 1], dtype="int32")
            out = paddle.gather(data1, index)
            place = paddle.CustomPlace("intel_hpu", 0)
            exe = base.Executor(place)
            iintel_hput = np.array([[1, 2], [3, 4], [5, 6]]).astype("float32")
            index_1 = np.array([1, 2]).astype("int32")
            (result,) = exe.run(
                feed={"data1": iintel_hput, "index": index_1}, fetch_list=[out]
            )
            expected_output = np.array([[3, 4], [5, 6]])
        self.assertTrue(np.allclose(result, expected_output))

    def test_out2(self):
        with paddle.static.program_guard(
            paddle.static.Program(), paddle.static.Program()
        ):
            x = paddle.static.data("x", shape=[-1, 2], dtype="float32")
            index = paddle.static.data("index", shape=[-1, 1], dtype="int32")
            out = paddle.gather(x, index)
            place = paddle.CustomPlace("intel_hpu", 0)
            exe = paddle.static.Executor(place)
            x_np = np.array([[1, 2], [3, 4], [5, 6]]).astype("float32")
            index_np = np.array([1, 1]).astype("int32")
            (result,) = exe.run(feed={"x": x_np, "index": index_np}, fetch_list=[out])
            expected_output = gather_numpy(x_np, index_np, axis=0)
        self.assertTrue(np.allclose(result, expected_output))


if __name__ == "__main__":
    unittest.main()
