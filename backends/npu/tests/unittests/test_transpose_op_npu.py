#  Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
from tests.op_test import OpTest
import paddle

paddle.enable_static()


class TestTransposeOp(OpTest):
    def setUp(self):
        self.set_npu()
        self.op_type = "transpose2"
        self.place = paddle.CustomPlace("npu", 0)
        self.init_dtype()
        self.init_shape_axis()

        self.inputs = {"X": np.random.random(self.shape).astype(self.dtype)}
        self.attrs = {"axis": self.axis, "data_format": "AnyLayout"}
        self.outputs = {"Out": self.inputs["X"].transpose(self.axis)}

    def set_npu(self):
        self.__class__.use_custom_device = True

    def init_dtype(self):
        self.dtype = np.float32

    def init_shape_axis(self):
        self.shape = (3, 40)
        self.axis = (1, 0)

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def test_check_grad(self):
        self.check_grad_with_place(self.place, ["X"], "Out")


class TestCase0(TestTransposeOp):
    def init_shape_axis(self):
        self.shape = (100,)
        self.axis = (0,)


class TestCase1(TestTransposeOp):
    def init_shape_axis(self):
        self.shape = (3, 4, 10)
        self.axis = (0, 2, 1)


class TestCase2(TestTransposeOp):
    def init_shape_axis(self):
        self.shape = (2, 3, 4, 5)
        self.axis = (0, 2, 3, 1)


class TestCase3(TestTransposeOp):
    def init_shape_axis(self):
        self.shape = (2, 3, 4, 5, 6)
        self.axis = (4, 2, 3, 1, 0)


class TestCase4(TestTransposeOp):
    def init_shape_axis(self):
        self.shape = (2, 3, 4, 5, 6, 1)
        self.axis = (4, 2, 3, 1, 0, 5)


class TestCase5(TestTransposeOp):
    def init_shape_axis(self):
        self.shape = (2, 16, 96)
        self.axis = (0, 2, 1)


class TestCase6(TestTransposeOp):
    def init_shape_axis(self):
        self.shape = (2, 10, 12, 16)
        self.axis = (3, 1, 2, 0)


class TestCase7(TestTransposeOp):
    def init_shape_axis(self):
        self.shape = (2, 10, 2, 16)
        self.axis = (0, 1, 3, 2)


class TestCase8(TestTransposeOp):
    def init_shape_axis(self):
        self.shape = (2, 3, 2, 3, 2, 4, 3, 3)
        self.axis = (0, 1, 3, 2, 4, 5, 6, 7)


class TestCase9(TestTransposeOp):
    def init_shape_axis(self):
        self.shape = (2, 3, 2, 3, 2, 4, 3, 3)
        self.axis = (6, 1, 3, 5, 0, 2, 4, 7)


class TestCase10(TestTransposeOp):
    def init_shape_axis(self):
        self.shape = []
        self.axis = []


class TestTransposeOpFP16(TestTransposeOp):
    def init_dtype(self):
        self.dtype = np.float16

    def test_check_grad(self):
        pass


class TestTransposeOpInt64(TestTransposeOp):
    def init_dtype(self):
        self.dtype = np.int64

    def test_check_grad(self):
        pass


class TestTransposeAPIWithNPUStroageFormat(unittest.TestCase):
    def setUp(self):
        self.shape_x = [4, 6, 24, 24]
        self.x = np.random.random(self.shape_x).astype(np.float32)
        self.axis = [0, 3, 1, 2]
        self.format = 3  # ACL_FORMAT_NC1HWC0 = 3
        self.place = paddle.CustomPlace("npu", 0)

    def test_api_static(self):
        paddle.enable_static()

        main_program = paddle.static.default_main_program()
        startup_program = paddle.static.default_startup_program()
        with paddle.static.program_guard(main_program, startup_program):
            x_data = paddle.static.data(
                shape=self.shape_x, name="data_x", dtype="float32"
            )

            out_expect = paddle.transpose(x=x_data, perm=self.axis)

            x_format = paddle.incubate._npu_identity(x=x_data, format=self.format)
            out_format = paddle.transpose(x=x_format, perm=self.axis)
            out_actual = paddle.incubate._npu_identity(x=out_format, format=-1)
        exe = paddle.static.Executor()
        exe.run(startup_program)
        result = exe.run(
            main_program,
            feed={x_data.name: self.x},
            fetch_list=[out_expect, out_actual],
        )

        np.testing.assert_allclose(result[0], result[1], rtol=1e-08)

    def test_api_dygraph(self):
        paddle.disable_static(self.place)

        # fwd and bwd with normal format
        x = paddle.to_tensor(self.x)
        axis = self.axis
        x.stop_gradient = False
        out_expect = paddle.transpose(x, axis)
        loss = out_expect.sum()
        loss.backward()
        x_grad_expect = x.grad

        # fwd and bwd with storage format
        paddle.base.set_flags({"FLAGS_use_stride_kernel": False})
        x_format = paddle.incubate._npu_identity(x, self.format)
        x_format.stop_gradient = False
        out_format = paddle.transpose(x_format, axis)
        loss_format = out_format.sum()
        loss_format.backward()
        out_actual = paddle.incubate._npu_identity(out_format, -1)
        x_grad_actual = paddle.incubate._npu_identity(x_format.grad, -1)

        # compare results
        np.testing.assert_allclose(out_expect.numpy(), out_actual.numpy(), rtol=1e-08)
        np.testing.assert_allclose(
            x_grad_expect.numpy(), x_grad_actual.numpy(), rtol=1e-08
        )
        paddle.enable_static()


if __name__ == "__main__":
    unittest.main()
