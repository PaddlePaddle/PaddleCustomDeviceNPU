# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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


import unittest

from tests.op_test import OpTest
import numpy as np

import paddle
from paddle import base
from paddle.base.backward import append_backward

paddle.enable_static()


class TestAssignOp(OpTest):
    def setUp(self):
        self.init_dtype()
        self.place = paddle.CustomPlace("gcu", 0)
        self.python_api = paddle.assign
        self.public_python_api = paddle.assign
        self.op_type = "assign"
        x = np.random.random(size=(100, 10)).astype(self.dtype)
        self.inputs = {"X": x}
        self.outputs = {"Out": x}

    def init_dtype(self):
        self.dtype = np.float32

    def test_forward(self):
        self.check_output_with_place(self.place)

    def test_backward(self):
        self.check_grad_with_place(self.place, ["X"], "Out")


class TestAssignFloat64Op(TestAssignOp):
    def init_dtype(self):
        self.dtype = np.float64


class TestAssignOApi(unittest.TestCase):
    def test_assign_DenseTensorArray(self):
        paddle.enable_static()
        x = paddle.static.data(name="x", shape=[100, 10], dtype="float32")
        x.stop_gradient = False
        y = paddle.tensor.fill_constant(shape=[100, 10], dtype="float32", value=1)
        z = paddle.add(x=x, y=y)
        array = paddle.assign(z)
        mean = paddle.mean(array)
        append_backward(mean)

        feed_x = np.random.random(size=(100, 10)).astype("float32")
        ones = np.ones((100, 10)).astype("float32")
        feed_add = feed_x + ones
        self.place = paddle.CustomPlace("gcu", 0)
        exe = paddle.static.Executor(self.place)
        res = exe.run(feed={"x": feed_x}, fetch_list=[array, x.grad_name])
        np.testing.assert_allclose(res[0], feed_add, rtol=1e-05)
        np.testing.assert_allclose(res[1], ones / 1000.0, rtol=1e-05)
        paddle.disable_static()

    def test_assign_NumpyArray(self):
        self.place = paddle.CustomPlace("gcu", 0)
        paddle.disable_static(self.place)
        for dtype in [np.bool_, np.float32, np.int32, np.int64]:
            with base.dygraph.guard():
                array = np.random.random(size=(100, 10)).astype(dtype)
                result1 = paddle.zeros(shape=[3, 3], dtype="float32")
                paddle.assign(array, result1)
            np.testing.assert_allclose(result1.numpy(), array, rtol=1e-05)
        paddle.enable_static()

    def test_assign_List(self):
        self.place = paddle.CustomPlace("gcu", 0)
        paddle.disable_static(self.place)
        l = [1, 2, 3]
        result = paddle.assign(l)
        np.testing.assert_allclose(result.numpy(), np.array(l), rtol=1e-05)
        paddle.enable_static()

    def test_assign_BasicTypes(self):
        self.place = paddle.CustomPlace("gcu", 0)
        paddle.disable_static(self.place)
        result1 = paddle.assign(2)
        result2 = paddle.assign(3.0)
        result3 = paddle.assign(True)
        np.testing.assert_allclose(result1.numpy(), np.array([2]), rtol=1e-05)
        np.testing.assert_allclose(result2.numpy(), np.array([3.0]), rtol=1e-05)
        np.testing.assert_allclose(result3.numpy(), np.array([1]), rtol=1e-05)
        paddle.enable_static()

    def test_clone(self):
        self.place = paddle.CustomPlace("gcu", 0)
        paddle.disable_static(self.place)
        self.python_api = paddle.clone

        x = paddle.ones([2])
        x.stop_gradient = False
        x.retain_grads()
        clone_x = paddle.clone(x)
        clone_x.retain_grads()

        y = clone_x**3
        y.backward()

        np.testing.assert_array_equal(x, [1, 1])
        np.testing.assert_array_equal(clone_x.grad.numpy(), [3, 3])
        np.testing.assert_array_equal(x.grad.numpy(), [3, 3])
        paddle.enable_static()


class TestAssignOpErrorApi(unittest.TestCase):
    def test_errors(self):
        paddle.enable_static()
        # The type of input must be Variable or numpy.ndarray.
        x1 = base.create_lod_tensor(
            np.array([[-1]]), [[1]], paddle.CustomPlace("gcu", 0)
        )
        self.assertRaises(TypeError, paddle.assign, x1)
        # When the type of input is numpy.ndarray, the dtype of input must be float32, int32.
        x2 = np.array([[2.5, 2.5]], dtype="uint8")
        self.assertRaises(TypeError, paddle.assign, x2)
        paddle.disable_static()

    def test_type_error(self):
        paddle.enable_static()
        x = [paddle.randn([3, 3]), paddle.randn([3, 3])]
        # not support to assign list(var)
        self.assertRaises(TypeError, paddle.assign, x)
        paddle.disable_static()


if __name__ == "__main__":
    unittest.main()
