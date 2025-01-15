#   Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
import paddle
import paddle.nn.functional as F
import paddle.incubate.nn.functional.swiglu as swigluimpl
from tests.op_test import convert_float_to_uint16, convert_uint16_to_float
from npu_utils import check_soc_version


def swiglu_naive(x, y=None):
    if y is None:
        x, y = paddle.chunk(x, chunks=2, axis=-1)
    return F.silu(x) * y


#  只有X，Y为空
class TestNPUSwigluFP16OnlyX(unittest.TestCase):
    def setUp(self):
        self.npu_place = paddle.CustomPlace("npu", 0)
        self.shape = (20, 512)
        self.init_dtype()

    def init_dtype(self):
        self.dtype = "float16"

    def check_result(self, golden_res, fused_res):
        if self.dtype == "float16":
            rtol = 1e-3
            atol = 1e-3
        elif self.dtype == "bfloat16":
            rtol = 8e-3
            atol = 8e-3
        elif self.dtype == "float32":
            rtol = 1e-5
            atol = 1e-5
        else:
            self.assertTrue(
                False,
                msg="Swiglu input dtype only supports bfloat16, \
                     float16,float32, but got "
                + self.dtype,
            )
        golden_y, golden_dx = golden_res
        fused_y, fused_dx = fused_res
        np.testing.assert_allclose(golden_y, fused_y, rtol=rtol, atol=atol)
        np.testing.assert_allclose(golden_dx, fused_dx, rtol=rtol, atol=atol)

    def golden_swiglu(self, x, y=None):
        res = swiglu_naive(x)
        res.backward()
        if self.dtype == "bfloat16":
            res = convert_uint16_to_float(res.numpy())
            dx = convert_uint16_to_float(x.grad.numpy())
            return res, dx
        return res.numpy(), x.grad.numpy()

    def fused_swiglu(self, x, y=None):
        res = swigluimpl(x)
        res.backward()
        dx = x.grad
        if self.dtype == "bfloat16":
            res = convert_uint16_to_float(res.numpy())
            dx = convert_uint16_to_float(dx.numpy())
            return res, dx
        return res.numpy(), dx.numpy()

    def gen_input(self):
        x = np.random.randn(self.shape[0], self.shape[1])
        return x

    @check_soc_version
    def test_swiglu(self):
        np_x = self.gen_input()
        if self.dtype == "bfloat16":
            np_x = convert_float_to_uint16(np_x)
        golden_x = paddle.to_tensor(
            np_x, place=self.npu_place, dtype=self.dtype, stop_gradient=False
        )
        fused_x = paddle.to_tensor(
            np_x, place=self.npu_place, dtype=self.dtype, stop_gradient=False
        )

        golden_res = self.golden_swiglu(golden_x)
        fused_res = self.fused_swiglu(fused_x)
        self.check_result(golden_res, fused_res)


#  X，Y都为空
class TestNPUSwigluFP16BothXY(TestNPUSwigluFP16OnlyX):
    def check_result(self, golden_res, fused_res):
        if self.dtype == "float16":
            rtol = 1e-3
            atol = 1e-3
        elif self.dtype == "bfloat16":
            rtol = 8e-3
            atol = 8e-3
        elif self.dtype == "float32":
            rtol = 1e-5
            atol = 1e-5
        else:
            self.assertTrue(
                False,
                msg="Swiglu input dtype only supports bfloat16, \
                     float16,float32, but got "
                + self.dtype,
            )
        golden_y, golden_dx, golden_dy = golden_res
        fused_y, fused_dx, fused_dy = fused_res
        if golden_dx is None and fused_dx is None:
            np.testing.assert_allclose(golden_y, fused_y, rtol=rtol, atol=atol)
            np.testing.assert_allclose(golden_dy, fused_dy, rtol=rtol, atol=atol)
        elif golden_dy is None and fused_dy is None:
            np.testing.assert_allclose(golden_y, fused_y, rtol=rtol, atol=atol)
            np.testing.assert_allclose(golden_dx, fused_dx, rtol=rtol, atol=atol)
        else:
            np.testing.assert_allclose(golden_y, fused_y, rtol=rtol, atol=atol)
            np.testing.assert_allclose(golden_dx, fused_dx, rtol=rtol, atol=atol)
            np.testing.assert_allclose(golden_dy, fused_dy, rtol=rtol, atol=atol)

    def gen_input(self):
        x = np.random.randn(self.shape[0], self.shape[1])
        y = np.random.randn(self.shape[0], self.shape[1])
        return x, y

    def golden_swiglu(self, x, y=None):
        res = swiglu_naive(x, y)
        res.backward()
        if self.dtype == "bfloat16":
            res = convert_uint16_to_float(res.numpy())
            dx = x.grad if x.grad is None else convert_uint16_to_float(x.grad.numpy())
            dy = y.grad if y.grad is None else convert_uint16_to_float(y.grad.numpy())
            return res, dx, dy
        if x.grad is None:
            return res.numpy(), x.grad, y.grad.numpy()
        if y.grad is None:
            return res.numpy(), x.grad.numpy(), y.grad
        return res.numpy(), x.grad.numpy(), y.grad.numpy()

    def fused_swiglu(self, x, y):
        res = swigluimpl(x, y)
        res.backward()
        if self.dtype == "bfloat16":
            res = convert_uint16_to_float(res.numpy())
            dx = x.grad if x.grad is None else convert_uint16_to_float(x.grad.numpy())
            dy = y.grad if y.grad is None else convert_uint16_to_float(y.grad.numpy())
            return res, dx, dy
        if x.grad is None:
            return res.numpy(), x.grad, y.grad.numpy()
        if y.grad is None:
            return res.numpy(), x.grad.numpy(), y.grad
        return res.numpy(), x.grad.numpy(), y.grad.numpy()

    @check_soc_version
    def test_swiglu(self):
        np_x, np_y = self.gen_input()
        if self.dtype == "bfloat16":
            np_x = convert_float_to_uint16(np_x)
            np_y = convert_float_to_uint16(np_y)
        golden_x = paddle.to_tensor(
            np_x, place=self.npu_place, dtype=self.dtype, stop_gradient=False
        )
        golden_y = paddle.to_tensor(
            np_y, place=self.npu_place, dtype=self.dtype, stop_gradient=False
        )
        fused_x = paddle.to_tensor(
            np_x, place=self.npu_place, dtype=self.dtype, stop_gradient=False
        )
        fused_y = paddle.to_tensor(
            np_y, place=self.npu_place, dtype=self.dtype, stop_gradient=False
        )

        golden_res = self.golden_swiglu(golden_x, golden_y)
        fused_res = self.fused_swiglu(fused_x, fused_y)
        self.check_result(golden_res, fused_res)


class TestNPUSwigluOnlyX(TestNPUSwigluFP16BothXY):
    @check_soc_version
    def test_swiglu(self):
        np_x, np_y = self.gen_input()
        if self.dtype == "bfloat16":
            np_x = convert_float_to_uint16(np_x)
            np_y = convert_float_to_uint16(np_y)
        golden_x = paddle.to_tensor(
            np_x, place=self.npu_place, dtype=self.dtype, stop_gradient=True
        )
        golden_y = paddle.to_tensor(
            np_y, place=self.npu_place, dtype=self.dtype, stop_gradient=False
        )
        fused_x = paddle.to_tensor(
            np_x, place=self.npu_place, dtype=self.dtype, stop_gradient=True
        )
        fused_y = paddle.to_tensor(
            np_y, place=self.npu_place, dtype=self.dtype, stop_gradient=False
        )

        golden_res = self.golden_swiglu(golden_x, golden_y)
        fused_res = self.fused_swiglu(fused_x, fused_y)
        self.check_result(golden_res, fused_res)


class TestNPUSwigluOnlyY(TestNPUSwigluFP16BothXY):
    @check_soc_version
    def test_swiglu(self):
        np_x, np_y = self.gen_input()
        if self.dtype == "bfloat16":
            np_x = convert_float_to_uint16(np_x)
            np_y = convert_float_to_uint16(np_y)
        golden_x = paddle.to_tensor(
            np_x, place=self.npu_place, dtype=self.dtype, stop_gradient=False
        )
        golden_y = paddle.to_tensor(
            np_y, place=self.npu_place, dtype=self.dtype, stop_gradient=True
        )
        fused_x = paddle.to_tensor(
            np_x, place=self.npu_place, dtype=self.dtype, stop_gradient=False
        )
        fused_y = paddle.to_tensor(
            np_y, place=self.npu_place, dtype=self.dtype, stop_gradient=True
        )

        golden_res = self.golden_swiglu(golden_x, golden_y)
        fused_res = self.fused_swiglu(fused_x, fused_y)
        self.check_result(golden_res, fused_res)


class TestNPUSwigluBF16OnlyX(TestNPUSwigluFP16OnlyX):
    def init_dtype(self):
        self.dtype = "bfloat16"


class TestNPUSwigluFP32OnlyX(TestNPUSwigluFP16OnlyX):
    def init_dtype(self):
        self.dtype = "float32"


class TestNPUSwigluBF16BothXY(TestNPUSwigluFP16BothXY):
    def init_dtype(self):
        self.dtype = "bfloat16"


class TestNPUSwigluFP32BothXY(TestNPUSwigluFP16BothXY):
    def init_dtype(self):
        self.dtype = "float32"


class TestNPUSwigluFP16OnlyX3D(TestNPUSwigluFP16OnlyX):
    def setUp(self):
        self.npu_place = paddle.CustomPlace("npu", 0)
        self.shape = (2, 20, 512)
        self.init_dtype()

    def gen_input(self):
        x = np.random.randn(self.shape[0], self.shape[1], self.shape[1])
        return x


class TestNPUSwigluFP16BothXY3D(TestNPUSwigluFP16BothXY):
    def setUp(self):
        self.npu_place = paddle.CustomPlace("npu", 0)
        self.shape = (2, 20, 512)
        self.init_dtype()

    def gen_input(self):
        x = np.random.randn(self.shape[0], self.shape[1], self.shape[2])
        y = np.random.randn(self.shape[0], self.shape[1], self.shape[2])
        return x, y


class TestNPUSwigluBF16OnlyX3D(TestNPUSwigluFP16OnlyX3D):
    def init_dtype(self):
        self.dtype = "bfloat16"


class TestNPUSwigluFP32OnlyX3D(TestNPUSwigluFP16OnlyX3D):
    def init_dtype(self):
        self.dtype = "float32"


class TestNPUSwigluBF16BothXY3D(TestNPUSwigluFP16BothXY3D):
    def init_dtype(self):
        self.dtype = "bfloat16"


class TestNPUSwigluFP32BothXY3D(TestNPUSwigluFP16BothXY3D):
    def init_dtype(self):
        self.dtype = "float32"


if __name__ == "__main__":
    np.random.seed(2024)
    unittest.main()
