# BSD 3- Clause License Copyright (c) 2023, Tecorigin Co., Ltd. All rights
# reserved.
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
# Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
# Neither the name of the copyright holder nor the names of its contributors
# may be used to endorse or promote products derived from this software
# without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION)
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY,OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)  ARISING IN ANY
# WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
# OF SUCH DAMAGE.

from __future__ import print_function

import unittest

import numpy as np
import paddle
import paddle.base as base
from op_test import OpTest

paddle.enable_static()
from white_list import no_grad_set_white_list

no_grad_set_white_list.NEED_TO_FIX_OP_LIST.append("stack")


class TestStackOpBase(OpTest):
    def initDefaultParameters(self):
        self.num_inputs = 4
        self.input_dim = (5, 6, 7)
        self.axis = 0

    def initParameters(self):
        pass

    def get_x_names(self):
        x_names = []
        for i in range(self.num_inputs):
            x_names.append("x{}".format(i))
        return x_names

    def setUp(self):
        self.initDefaultParameters()
        self.initParameters()
        self.op_type = "stack"
        self.python_api = paddle.stack
        self.set_sdaa()
        self.init_dtype()
        self.x = []
        for i in range(self.num_inputs):
            self.x.append(np.random.random(size=self.input_dim).astype(self.dtype))

        tmp = []
        x_names = self.get_x_names()
        for i in range(self.num_inputs):
            tmp.append((x_names[i], self.x[i]))

        self.inputs = {"X": tmp}
        self.outputs = {"Y": np.stack(self.x, axis=self.axis)}
        self.attrs = {"axis": self.axis}

    def set_sdaa(self):
        self.__class__.use_custom_device = True
        self.__class__.no_need_check_grad = True
        self.place = paddle.CustomPlace("sdaa", 0)

    def init_dtype(self):
        self.dtype = np.float32

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def test_check_grad(self):
        if self.dtype in [np.int32, np.int64, np.bool_, np.uint8]:
            return
        self.check_grad_with_place(
            self.place,
            self.get_x_names(),
            "Y",
            max_relative_error=0.03,
            numeric_place=paddle.CPUPlace(),
        )
        self.check_grad_with_place(
            self.place,
            self.get_x_names(),
            "Y",
            max_relative_error=0.03,
            no_grad_set=set(self.get_x_names()[0]),
            numeric_place=paddle.CPUPlace(),
        )
        self.check_grad_with_place(
            self.place,
            self.get_x_names(),
            "Y",
            max_relative_error=0.03,
            no_grad_set=set(self.get_x_names()[0:2]),
            numeric_place=paddle.CPUPlace(),
        )
        self.check_grad_with_place(
            self.place,
            self.get_x_names(),
            "Y",
            max_relative_error=0.03,
            no_grad_set=set(self.get_x_names()[1:3]),
            numeric_place=paddle.CPUPlace(),
        )


class TestStackOp1(TestStackOpBase):
    def initParameters(self):
        self.num_inputs = 16


class TestStackOp2(TestStackOpBase):
    def initParameters(self):
        self.num_inputs = 20


class TestStackOp3(TestStackOpBase):
    def initParameters(self):
        self.axis = -1


class TestStackOp4(TestStackOpBase):
    def initParameters(self):
        self.axis = -4


class TestStackOp5(TestStackOpBase):
    def initParameters(self):
        self.axis = 1


class TestStackOp6(TestStackOpBase):
    def initParameters(self):
        self.axis = 3


class TestStackOpINT32(TestStackOpBase):
    def init_dtype(self):
        self.dtype = np.int32


class TestStackOpINT64(TestStackOpBase):
    def init_dtype(self):
        self.dtype = np.int64


class TestStackOpHalf(TestStackOpBase):
    def init_dtype(self):
        self.dtype = np.float16


class TestStackOpBool(TestStackOpBase):
    def init_dtype(self):
        self.dtype = np.bool_


class TestStackOpUint8(TestStackOpBase):
    def init_dtype(self):
        self.dtype = np.uint8


# ----------------Stack Fp16----------------
def create_test_fp16(parent):
    class TestStackFp16(parent):
        def init_dtype(self):
            self.dtype = np.float16

    cls_name = "{0}_{1}".format(parent.__name__, "Fp16")
    TestStackFp16.__name__ = cls_name
    globals()[cls_name] = TestStackFp16


create_test_fp16(TestStackOpBase)
create_test_fp16(TestStackOp1)
create_test_fp16(TestStackOp2)
create_test_fp16(TestStackOp3)
create_test_fp16(TestStackOp4)
create_test_fp16(TestStackOp5)
create_test_fp16(TestStackOp6)


class TestStackAPIWithDenseTensorArray(unittest.TestCase):
    """
    Test stack api when the input(x) is a DenseTensorArray.
    """

    def setUp(self):
        self.axis = 1
        self.iter_num = 3
        self.input_shape = [2, 3]
        self.x = np.random.random(self.input_shape).astype("float32")
        self.place = paddle.CustomPlace("sdaa", 0)
        self.set_program()

    def set_program(self):
        self.program = base.Program()
        with base.program_guard(self.program):
            input = paddle.assign(self.x)
            tensor_array = paddle.tensor.create_array(dtype="float32")
            zero = paddle.tensor.fill_constant(shape=[1], value=0, dtype="int64")

            for i in range(self.iter_num):
                paddle.tensor.array_write(input, zero + i, tensor_array)

            self.out_var = paddle.stack(tensor_array, axis=self.axis)

    def test_case(self):
        self.assertTrue(self.out_var.shape[self.axis] == -1)
        exe = base.Executor(self.place)
        res = exe.run(self.program, fetch_list=self.out_var)
        self.assertTrue(
            np.array_equal(res[0], np.stack([self.x] * self.iter_num, axis=self.axis))
        )


class TestTensorStackAPIWithDenseTensorArray(unittest.TestCase):
    """
    Test stack api when the input(x) is a DenseTensorArray.
    """

    def setUp(self):
        self.axis = 1
        self.iter_num = 3
        self.input_shape = [2, 3]
        self.x = np.random.random(self.input_shape).astype("float32")
        self.place = paddle.CustomPlace("sdaa", 0)
        self.set_program()

    def set_program(self):
        self.program = base.Program()
        with base.program_guard(self.program):
            input = paddle.assign(self.x)
            tensor_array = paddle.tensor.create_array(dtype="float32")
            zero = paddle.tensor.fill_constant(shape=[1], value=0, dtype="int64")

            for i in range(self.iter_num):
                paddle.tensor.array_write(input, zero + i, tensor_array)

            self.out_var = paddle.stack(tensor_array, axis=self.axis)

    def test_case(self):
        self.assertTrue(self.out_var.shape[self.axis] == -1)
        exe = base.Executor(self.place)
        res = exe.run(self.program, fetch_list=self.out_var)
        self.assertTrue(
            np.array_equal(res[0], np.stack([self.x] * self.iter_num, axis=self.axis))
        )


class API_test(unittest.TestCase):
    def test_out(self):
        paddle.enable_static()
        with base.program_guard(base.Program(), base.Program()):
            data1 = paddle.static.data("data1", shape=[1, 2], dtype="float32")
            data2 = paddle.static.data("data2", shape=[1, 2], dtype="float32")
            data3 = paddle.static.data("data3", shape=[1, 2], dtype="float32")
            result_stack = paddle.stack([data1, data2, data3], axis=0)
            place = paddle.CustomPlace("sdaa", 0)
            exe = base.Executor(place)
            input1 = np.random.random([1, 2]).astype("float32")
            input2 = np.random.random([1, 2]).astype("float32")
            input3 = np.random.random([1, 2]).astype("float32")
            (result,) = exe.run(
                feed={"data1": input1, "data2": input2, "data3": input3},
                fetch_list=[result_stack],
            )
            expected_result = np.stack([input1, input2, input3], axis=0)
            self.assertTrue(np.allclose(expected_result, result))

    def test_single_tensor_error(self):
        paddle.enable_static()
        with base.program_guard(base.Program(), base.Program()):
            x = paddle.rand([2, 3])
            self.assertRaises(TypeError, paddle.stack, x)


class API_DygraphTest(unittest.TestCase):
    def test_out(self):
        paddle.disable_static()
        data1 = np.array([[1.0, 2.0]], dtype="float32")
        data2 = np.array([[3.0, 4.0]], dtype="float32")
        data3 = np.array([[5.0, 6.0]], dtype="float32")
        with base.dygraph.guard(place=paddle.CustomPlace("sdaa", 0)):
            x1 = base.dygraph.to_variable(data1)
            x2 = base.dygraph.to_variable(data2)
            x3 = base.dygraph.to_variable(data3)
            result = paddle.stack([x1, x2, x3])
            result_np = result.numpy()
        expected_result = np.stack([data1, data2, data3])
        self.assertTrue(np.allclose(expected_result, result_np))

        with base.dygraph.guard(place=paddle.CustomPlace("sdaa", 0)):
            y1 = base.dygraph.to_variable(data1)
            result = paddle.stack([y1], axis=0)
            result_np_2 = result.numpy()
        expected_result_2 = np.stack([data1], axis=0)
        self.assertTrue(np.allclose(expected_result_2, result_np_2))
        paddle.enable_static()

    def test_single_tensor_error(self):
        paddle.disable_static()
        with base.dygraph.guard(place=paddle.CustomPlace("sdaa", 0)):
            x = paddle.to_tensor([1, 2, 3])
            self.assertRaises(Exception, paddle.stack, x)
        paddle.enable_static()


if __name__ == "__main__":
    unittest.main()
