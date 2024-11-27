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

import numpy as np
from tests.op_test import OpTest
import paddle
from paddle import base
from paddle.base import core
from paddle.base.framework import Program

np.random.seed(1024)
paddle.enable_static()


class TestArgsortOp(OpTest):
    def setUp(self):
        self.set_gcu()
        self.op_type = "argsort"
        self.init_dtype()
        self.init_inputshape()
        self.init_axis()
        self.init_direction()
        np.random.seed(1024)
        self.x = np.random.random(self.input_shape).astype(self.dtype)
        self.inputs = {"X": self.x}
        self.attrs = {"axis": self.axis, "descending": self.descending}
        self.get_output()
        self.outputs = {"Out": self.sorted_x, "Indices": self.indices}

    def get_output(self):
        if self.descending:
            self.indices = np.flip(
                np.argsort(self.x, kind="quicksort", axis=self.axis), self.axis
            )
            self.sorted_x = np.flip(
                np.sort(self.x, kind="quicksort", axis=self.axis), self.axis
            )
        else:
            self.indices = np.argsort(self.x, kind="quicksort", axis=self.axis)
            self.sorted_x = np.sort(self.x, kind="quicksort", axis=self.axis)

    def set_gcu(self):
        self.__class__.use_custom_device = True
        self.__class__.no_need_check_grad = True
        self.place = paddle.CustomPlace("gcu", 0)

    def init_inputshape(self):
        self.input_shape = (2, 2, 3, 3)
        self.element_size = 1
        for i in self.input_shape:
            self.element_size *= i

    def init_dtype(self):
        self.dtype = np.float16

    def init_axis(self):
        self.axis = -1

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def init_direction(self):
        self.descending = False


class TestArgsortOpAxis0GCU(TestArgsortOp):
    def init_axis(self):
        self.axis = 0


class TestArgsortOpAxis1GCU(TestArgsortOp):
    def init_axis(self):
        self.axis = 1


class TestArgsortOpAxis2GCU(TestArgsortOp):
    def init_axis(self):
        self.axis = 2


class TestArgsortOpAxisNeg1GCU(TestArgsortOp):
    def init_axis(self):
        self.axis = -1


class TestArgsortOpAxisNeg2GCU(TestArgsortOp):
    def init_axis(self):
        self.axis = -2


class TestArgsortOpDescendingAxisGCU(TestArgsortOp):
    def init_direction(self):
        self.descending = True


class TestArgsortOpDescendingAxis0GCU(TestArgsortOpAxis0GCU):
    def init_direction(self):
        self.descending = True


class TestArgsortOpDescendingAxis1GCU(TestArgsortOpAxis1GCU):
    def init_direction(self):
        self.descending = True


class TestArgsortOpDescendingAxis2GCU(TestArgsortOpAxis2GCU):
    def init_direction(self):
        self.descending = True


class TestArgsortOpDescendingAxisNeg1GCU(TestArgsortOpAxisNeg1GCU):
    def init_direction(self):
        self.descending = True


class TestArgsortOpDescendingAxisNeg2GCU(TestArgsortOpAxisNeg2GCU):
    def init_direction(self):
        self.descending = True


class TestArgsortOpAxis0GCUFP32(TestArgsortOp):
    def init_axis(self):
        self.axis = -1

    def init_dtype(self):
        self.dtype = np.float32

    def test_check_output(self):
        self.check_output_with_place(self.place, atol=1e-5)

    def set_gcu(self):
        self.__class__.use_custom_device = True
        self.place = paddle.CustomPlace("gcu", 0)

    def test_check_grad(self):
        self.check_grad_with_place(self.place, ["X"], "Out", max_relative_error=0.03)


class PyArgsort:
    def __init__(self, input_shape, axis, descending, dtype):
        self.x = np.random.random(input_shape).astype(dtype)
        self.label = np.random.random(input_shape).astype(dtype)
        if axis < 0:
            self.axis = axis + len(self.x.shape)
        else:
            self.axis = axis
        self.descending = descending

    def forward(self):
        if self.descending:
            self.indices = np.flip(
                np.argsort(self.x, kind="quicksort", axis=self.axis), self.axis
            )
            self.sorted_x = np.flip(
                np.sort(self.x, kind="quicksort", axis=self.axis), self.axis
            )
        else:
            self.indices = np.argsort(self.x, kind="quicksort", axis=self.axis)
            self.sorted_x = np.sort(self.x, kind="quicksort", axis=self.axis)
        self.loss = self.sorted_x * self.label
        self.loss = np.sum(self.loss)
        out = (
            np.array(self.indices, dtype=self.indices.dtype),
            np.array(self.sorted_x, dtype=self.sorted_x.dtype),
            np.array(self.loss, dtype=self.loss.dtype),
        )
        return out


def create_tensor(np_data, place):
    tensor = core.DenseTensor()
    tensor.set(np_data, place)
    return tensor


class TestArgsortOpGCU(unittest.TestCase):
    def setup_program(self):
        self.main_program = Program()
        self.startup_program = Program()
        self.init_place()

    def setUp(self):
        self.init_axis()
        self.init_datatype()
        self.init_direction()
        self.init_inputshape()

        self.setup_program()
        self.feed_data_field = {"x", "label"}
        self.grad_data_field = {"x"}

        self.py_argsort = PyArgsort(
            self.input_shape, self.axis, self.descending, self.dtype
        )

        with base.program_guard(self.main_program, self.startup_program):
            x = paddle.static.data(
                name="x", shape=[-1] + list(self.input_shape), dtype=self.dtype
            )
            x.stop_gradient = False
            x.desc.set_need_check_feed(False)
            label = paddle.static.data(
                name="label",
                shape=[-1] + list(self.input_shape),
                dtype=self.dtype,
            )
            label.desc.set_need_check_feed(False)
            self.index = paddle.argsort(x=x, axis=self.axis, descending=self.descending)
            self.sorted_x = paddle.sort(x=x, axis=self.axis, descending=self.descending)
            self.sorted_x.stop_gradient = False
            loss = paddle.multiply(self.sorted_x, label)
            self.loss = paddle.sum(loss)

    def forward(self):
        self.feed_map = {
            x: create_tensor(getattr(self.py_argsort, x), self.place)
            for x in self.feed_data_field
        }
        exe = paddle.static.Executor(self.place)
        out = exe.run(
            self.main_program,
            feed=self.feed_map,
            fetch_list=[self.index, self.sorted_x, self.loss],
        )
        return out

    def check_forward(self):
        pd_outputs = self.forward()
        py_outputs = self.py_argsort.forward()
        for pd_output, py_output in zip(pd_outputs, py_outputs):
            self.assertEqual(pd_output.shape, py_output.shape)
            np.testing.assert_allclose(
                pd_output, py_output, rtol=1e-05, atol=0, equal_nan=False
            )

    def get_numerical_gradient(self, delta=1e-7):
        if self.dtype == "float16":
            delta = np.array(delta).astype(np.float16)
        feed_list = [getattr(self.py_argsort, x) for x in self.grad_data_field]
        grad_list = [np.zeros_like(x) for x in feed_list]
        for feed, grad in zip(feed_list, grad_list):
            for f, g in np.nditer([feed, grad], op_flags=["readwrite"]):
                o = float(f)
                f[...] = o + delta
                y_pos = self.forward()[2]

                f[...] = o - delta
                y_neg = self.forward()[2]

                f[...] = o
                dout_dfeed = (y_pos - y_neg) / (delta * 2)
                g[...] = dout_dfeed

        return grad_list

    def assert_is_close(
        self,
        numeric_grads,
        analytic_grads,
        names,
        max_relative_error,
        msg_prefix,
    ):
        for a, b, name in zip(numeric_grads, analytic_grads, names):
            abs_a = np.abs(a)
            abs_a[abs_a < 1e-3] = 1

            diff_mat = np.abs(a - b) / abs_a
            max_diff = np.max(diff_mat)

            def err_msg():
                offset = np.argmax(diff_mat > max_relative_error)
                return (
                    "%s error, %s variable %s max gradient diff %f over limit %f, "
                    "the first error element is %d, expected %f, but got %f."
                ) % (
                    "argsort",
                    msg_prefix,
                    name,
                    max_diff,
                    max_relative_error,
                    offset,
                    a.flatten()[offset],
                    b.flatten()[offset],
                )

            self.assertLessEqual(max_diff, max_relative_error, err_msg())

    def init_axis(self):
        self.axis = -1

    def init_datatype(self):
        self.dtype = "float32"

    def init_direction(self):
        self.descending = False

    def init_inputshape(self):
        self.input_shape = (2, 2, 2, 2, 3)

    def init_place(self):
        self.place = paddle.CustomPlace("gcu", 0)


class TestArgsortOpAxis0GCU(TestArgsortOpGCU):
    def init_axis(self):
        self.axis = 0


class TestArgsortOpAxis1GCU(TestArgsortOpGCU):
    def init_axis(self):
        self.axis = 1


class TestArgsortOpAxis2GCU(TestArgsortOpGCU):
    def init_axis(self):
        self.axis = 2


class TestArgsortOpAxisNeg1GCU(TestArgsortOpGCU):
    def init_axis(self):
        self.axis = -1


class TestArgsortOpAxisNeg2GCU(TestArgsortOpGCU):
    def init_axis(self):
        self.axis = -2


class TestArgsortOpDescendingAxisGCU(TestArgsortOpGCU):
    def init_direction(self):
        self.descending = True


class TestArgsortOpDescendingAxis0GCU(TestArgsortOpAxis0GCU):
    def init_direction(self):
        self.descending = True


class TestArgsortOpDescendingAxis1GCU(TestArgsortOpAxis1GCU):
    def init_direction(self):
        self.descending = True


class TestArgsortOpDescendingAxis2GCU(TestArgsortOpAxis2GCU):
    def init_direction(self):
        self.descending = True


class TestArgsortOpDescendingAxisNeg1GCU(TestArgsortOpAxisNeg1GCU):
    def init_direction(self):
        self.descending = True


class TestArgsortOpDescendingAxisNeg2GCU(TestArgsortOpAxisNeg2GCU):
    def init_direction(self):
        self.descending = True


class TestArgsortErrorOnGCU(unittest.TestCase):
    def setUp(self):
        self.place = paddle.CustomPlace("gcu", 0)

    def test_error(self):
        def test_fluid_var_type():
            with base.program_guard(base.Program()):
                x = [1]
                output = paddle.argsort(x=x)
            self.assertRaises(TypeError, test_fluid_var_type)

        def test_paddle_var_type():
            with base.program_guard(base.Program()):
                x = [1]
                output = paddle.argsort(x=x)
            self.assertRaises(TypeError, test_paddle_var_type)


class TestArgsort(unittest.TestCase):
    def init(self):
        self.input_shape = [
            1000,
        ]
        self.axis = 0

    def setUp(self):
        self.init()
        self.place = paddle.CustomPlace("gcu", 0)
        self.data = np.random.rand(*self.input_shape).astype("float32")

    def test_api(self):
        with base.program_guard(base.Program()):
            input = paddle.static.data(
                name="input", shape=self.input_shape, dtype="float32"
            )

            output = paddle.argsort(input, axis=self.axis)
            output2 = paddle.argsort(input, axis=self.axis, descending=True)

            exe = paddle.static.Executor(self.place)
            result, result2 = exe.run(
                feed={"input": self.data}, fetch_list=[output, output2]
            )

            np_result = np.argsort(self.data, axis=self.axis)
            self.assertEqual((result == np_result).all(), True)

            np_result2 = np.argsort(-self.data, axis=self.axis)
            self.assertEqual((result2 == np_result2).all(), True)


class TestArgsort2(TestArgsort):
    def init(self):
        self.input_shape = [1000, 1]
        self.axis = 0


class TestArgsort3(TestArgsort):
    def init(self):
        self.input_shape = [1, 1000]
        self.axis = 1


class TestArgsort4(TestArgsort):
    def init(self):
        self.input_shape = [2, 3, 4]
        self.axis = 1


class TestArgsortImperative(unittest.TestCase):
    def init(self):
        self.input_shape = [
            1000,
        ]
        self.axis = 0

    def setUp(self):
        self.init()
        self.input_data = np.random.uniform(-100, 100, self.input_shape).astype(
            "float32"
        )
        self.place = paddle.CustomPlace("gcu", 0)

    def test_api(self):
        paddle.disable_static(self.place)
        var_x = paddle.to_tensor(self.input_data)
        out = paddle.argsort(var_x, axis=self.axis)
        expect = np.argsort(self.input_data, axis=self.axis)
        np.testing.assert_allclose(out.numpy(), expect, rtol=1e-05)

        out = paddle.argsort(var_x, axis=self.axis, descending=True)
        expect2 = np.argsort(-self.input_data, axis=self.axis)
        np.testing.assert_allclose(out.numpy(), expect2, rtol=1e-05)

        paddle.enable_static()


class TestArgsortImperative2(TestArgsortImperative):
    def init(self):
        self.input_shape = [1000, 1]
        self.axis = 0


class TestArgsortImperative3(TestArgsortImperative):
    def init(self):
        self.input_shape = [1, 1000]
        self.axis = 1


class TestArgsortImperative4(TestArgsortImperative):
    def init(self):
        self.input_shape = [2, 3, 4]
        self.axis = 1


class TestArgsortWithInputNaN(unittest.TestCase):
    def init(self):
        self.axis = 0

    def setUp(self):
        self.init()
        self.input_data = np.array([1.0, 0.0, 3.0, 2.0]).astype("float32")
        self.place = paddle.CustomPlace("gcu", 0)

    def test_api(self):
        paddle.disable_static(self.place)
        var_x = paddle.to_tensor(self.input_data)
        out = paddle.argsort(var_x, axis=self.axis)
        res_np = np.argsort(self.input_data, axis=self.axis)
        self.assertEqual((out.numpy() == res_np).all(), True)

        res_np_flip = np.flip(res_np)
        out = paddle.argsort(var_x, axis=self.axis, descending=True)
        self.assertEqual((out.numpy() == res_np_flip).all(), True)
        paddle.enable_static()


class TestArgsortOpFp16(unittest.TestCase):
    def test_fp16(self):
        x_np = np.random.random((2, 8)).astype("float16")
        x = paddle.static.data(shape=[2, 8], name="x", dtype="float16")
        out = paddle.argsort(x)

        self.place = paddle.CustomPlace("gcu", 0)
        exe = paddle.static.Executor(self.place)
        res = exe.run(feed={"x": x_np}, fetch_list=[out])

        res_np = np.argsort(x_np)
        for r in res:
            self.assertEqual((res_np == r).all(), True)


if __name__ == "__main__":
    unittest.main()
